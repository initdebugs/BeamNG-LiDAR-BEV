"""
Driving into a selected parking bay: the path, the tracker, the state machine.

This is deliberately NOT the arc planner. That planner scores a fan of
hold-then-bend candidates against free distance and re-chooses every tick,
which is right for open-road driving and wrong for a manoeuvre: CLAUDE.md
records an S-curve family that was built, measured to fix the narrow-road
pass, and NOT landed because under per-tick re-planning the car drove the
outbound half and re-picked it for ever -- "landing it safely needs the
planner to COMMIT to a manoeuvre it has begun". Parking is that problem in
its purest form, so it gets its own controller with a fixed goal.

The worker commits an immutable WORLD-space bay and the planner commits a
bay-relative TRAJECTORY. The controller re-projects both into the moving BEV
frame, advances monotonically along the path, and replans only for explicit
events: blockage, excessive tracking error, no progress, or a cusp. Hybrid A*
may produce forward and reverse legs, with a stopped gear handshake between
them. Success is a separate stopped verification dwell followed by the
parking brake; merely crossing the path endpoint is never success.

Qt-free and BeamNGpy-free, like `planner`, `aeb` and `parking`: config +
models + numpy. It produces a `ControlCommand`, so `worker._actuate` sends it
exactly as it sends the road controller's -- including the gear handling and
the AEB override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import (
    BRAKE_GAIN_MPS2,
    MIN_TURN_RADIUS_M,
    PARKING_ARRIVE_TOLERANCE_M,
    PARKING_BLOCKED_CLEAR_DWELL_S,
    PARKING_BODY_CLEARANCE_M,
    PARKING_DRIVE_APPROACH_M,
    PARKING_DRIVE_CREEP_HOLD_M,
    PARKING_DRIVE_CREEP_MPS,
    PARKING_DRIVE_DECEL_MPS2,
    PARKING_DRIVE_LOOKAHEAD_M,
    PARKING_DRIVE_MAX_CROSS_TRACK_M,
    PARKING_DRIVE_SPEED_MPS,
    PARKING_HEAD_CLEARANCE_M,
    PARKING_LEG_CLOSE_M,
    PARKING_LEG_SQUARE_DEG,
    PARKING_MAX_REPLANS,
    PARKING_OVERSHOOT_M,
    PARKING_PATH_SAMPLES,
    PARKING_PATH_SLACK_M,
    PARKING_PROGRESS_TIMEOUT_S,
    PARKING_SHIFT_DWELL_S,
    PARKING_SHIFT_SPEED_MPS,
    PARKING_STOP_BRAKE,
    PARKING_SUCCESS_BOUNDARY_TOLERANCE_M,
    PARKING_SUCCESS_DWELL_S,
    PARKING_SUCCESS_HEADING_DEG,
    PARKING_SUCCESS_POSITION_M,
    PARKING_SUCCESS_SPEED_MPS,
    PARKING_TURN_SLOW_DEG,
    STEERING_SIGN,
    THROTTLE_GAIN_MPS2,
)
from .controller import REVERSE_GEAR, is_forward_gear, is_reverse_gear
from .hybrid_astar import Occupancy, PlannedPath, Pose, plan
from .models import ControlCommand, ParkingSlot, VehicleGeometry

_MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M

# Display/controller phases. ParkingJob maps these onto durable job states.
PARK_APPROACH = "APPROACH"
PARK_ARRIVED = "ARRIVED"
PARK_BACKING = "BACKING"
PARK_SHIFTING = "SHIFTING"
PARK_BLOCKED = "BLOCKED"
PARK_SECURING = "SECURING"
PARK_UNREACHABLE = "UNREACHABLE"


@dataclass(frozen=True)
class ParkingPath:
    """A sampled route from the car's current pose into the bay, in BEV."""

    points: np.ndarray
    """(N, 2) right/forward samples, starting at the reference node."""
    cumulative_m: np.ndarray
    """(N,) arc length along `points`, so `cumulative_m[-1]` is the total."""
    max_curvature: float
    entry_index: int
    """Where the swept approach ends and the straight run into the bay begins."""

    @property
    def length_m(self) -> float:
        return float(self.cumulative_m[-1])


@dataclass(frozen=True)
class ParkingDriveState:
    """What the parking manoeuvre is doing, for the overlay and the log."""

    phase: str
    remaining_m: float
    cross_track_m: float
    curvature: float
    target_speed_mps: float
    reason: str
    path: ParkingPath | None = None

    @property
    def finished(self) -> bool:
        return self.phase in (PARK_ARRIVED, PARK_UNREACHABLE)


def bay_axis(slot: ParkingSlot) -> np.ndarray:
    """The unit vector pointing INTO the bay, in BEV (right, forward)."""
    return np.asarray(
        (math.sin(slot.heading_rad), math.cos(slot.heading_rad))
    )


def stop_pose(
    slot: ParkingSlot, geometry: VehicleGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the REFERENCE NODE must finish, and pointing which way.

    The node is not the middle of the car -- `front_m` is the distance from it
    to the nose -- so parking "the bay centre" would leave the car sitting
    however far off-centre the node happens to be. The nose is placed a fixed
    clearance short of the bay head and the node follows from that.
    """
    axis = bay_axis(slot)
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    head = centre + axis * (slot.depth_m * 0.5)
    return head - axis * (PARKING_HEAD_CLEARANCE_M + geometry.front_m), axis


def _straight_arc_straight(
    entry: np.ndarray, turn: float, radius: float, samples: int
) -> np.ndarray | None:
    """
    Run straight, turn through one arc of exactly `radius`, run straight to
    the entry pose. None when no such path exists going forwards.

    A cubic Bezier was tried first and is the wrong family for this: it cannot
    be told to respect a minimum radius, only measured afterwards, and on the
    commonest case in a lot -- a bay square-on to the aisle -- the flattest
    Bezier that reaches the pose still bent to 0.20 1/m against the car's 0.167
    limit, so every such bay came back unreachable. This construction hits the
    limit exactly, which is what a driver does: straighten, one steady turn,
    straighten.

    Headings are measured from +forward toward +right (`ParkingSlot`'s own
    convention), so a direction is `(sin t, cos t)` and moving at a signed
    curvature `c` integrates to the closed form below.
    """
    if abs(math.sin(turn)) < 1e-6:
        # Square on (nothing to turn) or dead astern (no forward path).
        if abs(turn) > 1e-6:
            return None
        if abs(float(entry[0])) > 0.05 or float(entry[1]) <= 0.0:
            return None
        return np.stack((np.zeros(2), entry))

    curvature = math.copysign(1.0 / radius, turn)
    # End of the arc, entering it at the origin heading +forward.
    arc_x = (1.0 - math.cos(turn)) / curvature
    arc_y = math.sin(turn) / curvature
    run_out = (float(entry[0]) - arc_x) / math.sin(turn)
    run_in = float(entry[1]) - arc_y - run_out * math.cos(turn)
    if run_in < -PARKING_PATH_SLACK_M or run_out < -PARKING_PATH_SLACK_M:
        # The pose is behind the turn, or well inside it: no forward
        # straight-arc-straight reaches it. A wider radius or a different
        # start does.
        return None
    # A SMALL negative is tracking lag, not an impossible pose, and refusing
    # it is what made a turning manoeuvre give up a few centimetres from
    # arriving: the car follows the path with a lookahead, so it reaches the
    # turn-in point marginally past it and `run_in` goes to -0.016 m. Clamping
    # starts the arc now and lets the tracker absorb the rest, which is what a
    # feedback law is for. The caller closes any residual by running straight
    # to the target from wherever the arc actually finishes.
    run_in = max(0.0, run_in)
    run_out = max(0.0, run_out)

    arc_length = abs(turn) * radius
    total = run_in + arc_length + run_out
    span = np.linspace(0.0, total, max(samples, 12))
    points = np.empty((len(span), 2))
    for index, distance in enumerate(span):
        if distance <= run_in:
            points[index] = (0.0, distance)
        elif distance <= run_in + arc_length:
            travelled = distance - run_in
            heading = curvature * travelled
            points[index] = (
                (1.0 - math.cos(heading)) / curvature,
                run_in + math.sin(heading) / curvature,
            )
        else:
            tail = distance - run_in - arc_length
            points[index] = (
                arc_x + tail * math.sin(turn),
                run_in + arc_y + tail * math.cos(turn),
            )
    return points


def _curvature(points: np.ndarray) -> np.ndarray:
    """Discrete curvature along a polyline, by the circumscribed-circle rule."""
    if len(points) < 3:
        return np.zeros(len(points))
    previous, current, following = points[:-2], points[1:-1], points[2:]
    first = current - previous
    second = following - current
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    lengths = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(following - previous, axis=1)
    )
    curvature = np.zeros(len(points))
    safe = lengths > 1e-9
    curvature[1:-1][safe] = 2.0 * cross[safe] / lengths[safe]
    return curvature


def plan_parking_path(
    slot: ParkingSlot, geometry: VehicleGeometry
) -> ParkingPath | None:
    """
    A path from the car's current pose into the bay, or None if it will not fit.

    A swept approach onto the bay's own axis, then a straight run in.

    **There is a hard geometric envelope and it is not a tuning failure.** One
    arc of radius R changes heading by 90 degrees and displaces the car
    EXACTLY R sideways and R forwards, so a square-on bay nearer than about
    `MIN_TURN_RADIUS_M` ahead, or nearer than that to the side, cannot be
    entered nose-first at all. A smaller displacement would need a smaller
    radius, which the car does not have, and an S-turn does not help -- a
    single arc is already the minimum lateral displacement for a given heading
    change, so any S displaces MORE.

    A straight staging run up the aisle was tried as the remedy and removed:
    it moves the bay CLOSER, which is the wrong direction for exactly the
    bays that fail. Measured, the reachable envelope was identical with and
    without it. What does reach a near bay is reversing in or repositioning
    first, and both are the multi-phase manoeuvre this deliberately does not
    attempt -- so `reachability` names the reason and the caller says it.
    """
    target, axis = stop_pose(slot, geometry)
    direct = _reach(target, axis, PARKING_PATH_SAMPLES)
    return None if direct is None else _as_path(direct)


def reachability(slot: ParkingSlot, geometry: VehicleGeometry) -> str:
    """
    Why a bay cannot be driven into, in words the driver can act on.

    "No single forward move fits" is true and useless: it does not say whether
    to roll forward, pick another bay, or that the car is simply on the wrong
    side of it. The envelope is known in closed form, so the reason can be.
    """
    target, axis = stop_pose(slot, geometry)
    turn = math.atan2(float(axis[0]), float(axis[1]))
    ahead = float(target[1])
    lateral = abs(float(target[0]))
    needed = MIN_TURN_RADIUS_M * abs(math.sin(turn))
    if ahead <= 0.0:
        return (
            "the bay is level with or behind the car -- pull forward past it "
            "and try again, or pick one further ahead"
        )
    if ahead < needed * 0.95:
        return (
            f"the bay is only {ahead:.1f} m ahead and turning into it needs "
            f"about {needed:.1f} m -- back up, or pick one further ahead"
        )
    if lateral < needed * 0.95:
        return (
            f"the bay is only {lateral:.1f} m to the side and turning into it "
            f"needs about {needed:.1f} m -- pick one further over"
        )
    return "no forward path fits between the car and the bay"


def _reach(
    target: np.ndarray, axis: np.ndarray, samples: int
) -> np.ndarray | None:
    """The direct swept approach plus straight run-in, from the origin."""
    turn = math.atan2(float(axis[0]), float(axis[1]))

    # Longest approach and TIGHTEST radius first, taking the first that
    # solves. Both orderings serve the same goal -- the car should be square
    # before it crosses the lines -- and for the radius that is the opposite
    # of the obvious choice: on a square-on bay the arc has to deliver a fixed
    # lateral offset, so a tighter arc finishes sooner and leaves a LONGER
    # straight run-in. Measured, taking the widest feasible radius instead
    # left 0.6 m of run-in and the car still 7.2 degrees off as it crossed the
    # stop plane; the tightest leaves 1.8 m and it arrives square. A tight
    # turn costs nothing at parking speed -- 0.33 m/s^2 at the 6 m minimum.
    #
    # The approach distance must be searched rather than fixed, and a
    # square-on bay is why. The entry point sits that far back ALONG the bay
    # axis, which for a 90-degree bay means back across the aisle toward the
    # car -- so a long approach leaves only a couple of metres of lateral
    # offset, while turning 90 degrees displaces the car by a whole radius.
    # At the 5 m default every square-on bay came back unreachable; they solve
    # at about a metre, which is also how a driver does it: turn in AT the
    # mouth, not five metres before it.
    #
    # The 0.0 at the end is the ENDGAME and it is not optional. In the last
    # metre of the run-in every non-zero approach puts the entry BEHIND the
    # car, so the manoeuvre reported unreachable a few centimetres short of
    # arriving. At zero the path is simply what is left of the run-in.
    approaches = tuple(
        PARKING_DRIVE_APPROACH_M * scale
        for scale in (1.6, 1.0, 0.7, 0.5, 0.35, 0.2, 0.1)
    ) + (0.0,)
    for approach in approaches:
        entry = target - axis * approach
        for radius in (
            MIN_TURN_RADIUS_M,
            MIN_TURN_RADIUS_M * 1.2,
            MIN_TURN_RADIUS_M * 1.6,
            MIN_TURN_RADIUS_M * 2.2,
        ):
            swept = _straight_arc_straight(entry, turn, radius, samples)
            if swept is None:
                continue
            # Always run straight from wherever the sweep actually finished
            # to the target, rather than from the nominal entry point: the
            # slack clamp may have moved the end by a few centimetres, and
            # the path must terminate ON the stop pose or the tracker aims at
            # the wrong place for the last metre.
            tail = float(np.linalg.norm(target - swept[-1]))
            run = np.linspace(0.0, 1.0, max(4, int(tail * 4)))[:, None]
            straight = swept[-1] + (target - swept[-1]) * run
            return np.concatenate((swept, straight[1:]))

    # The ENDGAME fallback: close to the stop pose and nearly square to it, no
    # straight-arc-straight fits any more -- there is not enough room left for
    # a run-in, an arc and a run-out, and every radius comes back about a
    # metre short. Measured, a turning manoeuvre gave up here at 2.2 m out and
    # 6 degrees off, a few centimetres of correction from being parked. What
    # is left at that range is a nudge, so the path becomes the single arc
    # through the target -- exactly the circle pure pursuit would follow.
    if (
        float(target[1]) > 0.0
        and abs(turn) <= math.radians(25.0)
        and abs(float(target[0])) <= 1.5
    ):
        return _single_arc(target, samples)
    return None


def _as_path(points: np.ndarray) -> ParkingPath:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return ParkingPath(
        points=points,
        cumulative_m=np.concatenate(([0.0], np.cumsum(steps))),
        max_curvature=float(np.abs(_curvature(points)).max()),
        entry_index=len(points) - 1,
    )


def _single_arc(target: np.ndarray, samples: int) -> np.ndarray | None:
    """The one circular arc from the origin, heading +forward, to `target`."""
    reach = float(np.linalg.norm(target))
    if reach < 1e-3:
        return None
    curvature = 2.0 * float(target[0]) / reach**2
    if abs(curvature) > _MAX_CURVATURE:
        return None
    if abs(curvature) < 1e-6:
        return np.stack((np.zeros(2), target))
    turn = 2.0 * math.atan2(float(target[0]), float(target[1]))
    span = np.linspace(0.0, turn / curvature, max(samples // 2, 8))
    return np.column_stack(
        (
            (1.0 - np.cos(curvature * span)) / curvature,
            np.sin(curvature * span) / curvature,
        )
    )


@dataclass(frozen=True)
class ParkingLeg:
    """
    One committed leg of a manoeuvre, expressed in the BAY's own frame.

    Held in bay coordinates rather than as a path, and that is the whole
    design. The bay is world-anchored and re-projected every tick, so a leg
    stays put while the path to it is re-derived from wherever the car
    actually is -- the same feedback argument the single forward move uses,
    extended to a sequence. Storing a PATH instead would freeze a plan the
    car then drifts off, and re-deriving the whole SEQUENCE every tick would
    reintroduce exactly the re-choice problem this controller exists to
    avoid: it would flip between manoeuvres and never finish one.
    """

    along_m: float
    """Along the bay axis from its centre; positive is deeper into the bay."""
    across_m: float
    """Across the bay axis; positive is to the axis's right."""
    heading_rad: float
    """Car heading RELATIVE to the bay axis. 0 faces in, pi faces out."""
    reverse: bool
    """Whether this leg is driven backwards."""
    path_bay: np.ndarray | None = None
    """
    (N, 2) along/across samples of this leg, in the BAY's frame, or None.

    A SEARCHED leg carries its own path and a canned one does not, and the
    difference is not cosmetic. The canned legs are each a single
    straight-arc-straight, so the path to one can be re-derived from the
    endpoint alone; a searched leg is whatever shape the search found, and
    re-deriving it from its endpoint with the single-arc solver produced a
    148 m path for a leg 6 m away. Held in bay coordinates for the same
    reason the endpoint is: it stays put while the car moves.
    """


def leg_pose(
    leg: ParkingLeg, slot: ParkingSlot
) -> tuple[np.ndarray, np.ndarray]:
    """A leg's target position and heading axis in the current BEV frame."""
    axis = bay_axis(slot)
    across = np.asarray((axis[1], -axis[0]))
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    position = centre + axis * leg.along_m + across * leg.across_m
    heading = slot.heading_rad + leg.heading_rad
    return position, np.asarray((math.sin(heading), math.cos(heading)))


def _reverse_reach(
    target: np.ndarray, axis: np.ndarray, samples: int
) -> np.ndarray | None:
    """
    The same solver, driving BACKWARDS.

    Reversing traces the same curves as driving forward with the heading
    flipped, so the reverse problem is the forward one in a frame rotated
    180 degrees: solve to the negated target and negate the answer. That is
    the trick `aeb.mirror_points` and the steered reverse already use -- a
    rotation preserves handedness, so every helper applies unchanged.
    """
    forward = _reach(-np.asarray(target, dtype=np.float64), axis, samples)
    return None if forward is None else -forward


def _leg_path(
    leg: ParkingLeg,
    slot: ParkingSlot,
    origin: np.ndarray | None = None,
    origin_axis: np.ndarray | None = None,
    samples: int = PARKING_PATH_SAMPLES,
) -> np.ndarray | None:
    """
    The path to a leg's pose, from the car or from another pose.

    `origin`/`origin_axis` let a leg be solved from where a PREVIOUS leg will
    finish, which is what makes a sequence checkable before any of it is
    driven -- a manoeuvre whose second half cannot be done is not a manoeuvre
    to start.
    """
    target, axis = leg_pose(leg, slot)
    if origin is not None and origin_axis is not None:
        # Into the origin pose's own frame: rotate so it faces +forward.
        forward = np.asarray(origin_axis, dtype=np.float64)
        right = np.asarray((forward[1], -forward[0]))
        delta = target - np.asarray(origin, dtype=np.float64)
        target = np.asarray((float(delta @ right), float(delta @ forward)))
        axis = np.asarray((float(axis @ right), float(axis @ forward)))
    if leg.path_bay is not None:
        # A searched leg drives the path it was planned with. Projected from
        # bay coordinates into the current BEV frame every tick, so it stays
        # anchored to the ground rather than to where the car used to be.
        bay_axis_v = bay_axis(slot)
        across_v = np.asarray((bay_axis_v[1], -bay_axis_v[0]))
        centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
        points = (
            centre
            + np.outer(leg.path_bay[:, 0], bay_axis_v)
            + np.outer(leg.path_bay[:, 1], across_v)
        )
        if origin is not None and origin_axis is not None:
            forward = np.asarray(origin_axis, dtype=np.float64)
            right = np.asarray((forward[1], -forward[0]))
            delta = points - np.asarray(origin, dtype=np.float64)
            points = np.column_stack((delta @ right, delta @ forward))
        return points
    solver = _reverse_reach if leg.reverse else _reach
    return solver(target, axis, samples)


def legs_from_path(
    path: PlannedPath, slot: ParkingSlot
) -> list[ParkingLeg]:
    """
    A searched path as committed legs, in the BAY's frame.

    Split at every direction change, because that is where the car has to
    stop and shift, and each piece's END is what the executor drives to. Bay
    coordinates rather than BEV so the legs stay world-anchored while the car
    moves -- the same contract the hand-written legs use, so the executor
    consumes either without knowing which planner produced it.
    """
    axis = bay_axis(slot)
    across_axis = np.asarray((axis[1], -axis[0]))
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    legs: list[ParkingLeg] = []
    for piece in path.legs():
        end = piece[-1]
        offset = np.asarray((end[0], end[1])) - centre
        relative = piece[:, :2] - centre
        legs.append(
            ParkingLeg(
                along_m=float(offset @ axis),
                across_m=float(offset @ across_axis),
                heading_rad=float(end[2] - slot.heading_rad),
                reverse=bool(end[3] < 0),
                path_bay=np.column_stack(
                    (relative @ axis, relative @ across_axis)
                ),
            )
        )
    return legs


def _committed_leg(
    leg: ParkingLeg, path: np.ndarray, slot: ParkingSlot
) -> ParkingLeg:
    """Attach a local path to its immutable bay-relative ground coordinates."""
    axis = bay_axis(slot)
    across_axis = np.asarray((axis[1], -axis[0]))
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    relative = np.asarray(path, dtype=np.float64) - centre
    return replace(
        leg,
        path_bay=np.column_stack((relative @ axis, relative @ across_axis)),
    )


def _clear(
    path: np.ndarray | None,
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None,
    leg: ParkingLeg,
) -> bool:
    """
    Whether a canned move exists AND is actually free.

    The canned families are pure geometry and know nothing about obstacles,
    so without this a wall straight across the lot did not stop them -- they
    are tried before the search precisely because they are cheap, and cheap
    must not mean blind.
    """
    if path is None:
        return False
    if occupancy is None:
        return True
    half_width = geometry.width_m * 0.5 + PARKING_BODY_CLEARANCE_M
    sampled = path[:: max(1, len(path) // 40)]
    if len(sampled) < 2:
        return False
    tangents = np.diff(sampled, axis=0)
    headings = np.arctan2(tangents[:, 0], tangents[:, 1])
    if leg.reverse:
        headings += math.pi
    poses = [
        Pose(
            float(point[0]),
            float(point[1]),
            float(headings[min(i, len(headings) - 1)]),
        )
        for i, point in enumerate(sampled)
    ]
    for start, end in zip(poses, poses[1:]):
        if occupancy.motion_cost(
            start, end, half_width, geometry.front_m, geometry.rear_m
        ) is None:
            return False
    return True


def _search_manoeuvre(
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None,
) -> list[ParkingLeg] | None:
    """Hybrid A* from the car to the bay's stop pose, as committed legs."""
    target, axis = stop_pose(slot, geometry)
    # Reverse entry ends the car facing OUT, so the tail clears the head and
    # the stop pose measures from `rear_m`. Both entries are offered to the
    # search and the cheaper one wins, which is how a driver chooses too.
    back_target = (
        np.asarray((slot.centre_right_m, slot.centre_forward_m))
        + axis * (slot.depth_m * 0.5 - PARKING_HEAD_CLEARANCE_M - geometry.rear_m)
    )
    grid = occupancy or Occupancy(None, None)
    best: PlannedPath | None = None
    for goal in (
        Pose(float(target[0]), float(target[1]), slot.heading_rad),
        Pose(
            float(back_target[0]),
            float(back_target[1]),
            slot.heading_rad + math.pi,
        ),
    ):
        found = plan(
            Pose(0.0, 0.0, 0.0),
            goal,
            grid,
            geometry.width_m * 0.5 + PARKING_BODY_CLEARANCE_M,
            geometry.front_m,
            geometry.rear_m,
        )
        if found is not None and (best is None or found.cost < best.cost):
            best = found
    return None if best is None else legs_from_path(best, slot)


def plan_manoeuvre(
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None = None,
) -> list[ParkingLeg] | None:
    """
    The legs that put the car in the bay, or None if nothing does.

    Tried in the order a driver would think of them: straight in if it fits,
    straight back in if the car is already lined up beyond the bay, and
    otherwise position first and then reverse in -- which is what a real
    parking manoeuvre is, and what makes bays the nose-in envelope cannot
    reach (nearer than a turning radius ahead or to the side) reachable at
    all.

    Every leg is checked from where the PREVIOUS leg finishes, so a sequence
    is only offered when all of it solves. Starting a manoeuvre whose second
    half is impossible would leave the car parked across the aisle.
    """
    nose_in = ParkingLeg(
        along_m=slot.depth_m * 0.5
        - PARKING_HEAD_CLEARANCE_M
        - geometry.front_m,
        across_m=0.0,
        heading_rad=0.0,
        reverse=False,
    )
    nose_path = _leg_path(nose_in, slot)
    if _clear(nose_path, slot, geometry, occupancy, nose_in):
        assert nose_path is not None
        return [_committed_leg(nose_in, nose_path, slot)]

    # Reverse entry: the car ends facing OUT, so it is the TAIL that clears
    # the head of the bay and the stop pose is measured from `rear_m`.
    back_in = ParkingLeg(
        along_m=slot.depth_m * 0.5
        - PARKING_HEAD_CLEARANCE_M
        - geometry.rear_m,
        across_m=0.0,
        heading_rad=math.pi,
        reverse=True,
    )
    back_path = _leg_path(back_in, slot)
    if _clear(back_path, slot, geometry, occupancy, back_in):
        assert back_path is not None
        return [_committed_leg(back_in, back_path, slot)]

    # Neither canned move fits, so SEARCH. Hybrid A* plans over the car's
    # own state and can shuffle as much as it needs, which is what covers the
    # bays the two families above refuse -- anything nearer than a turning
    # radius to the side, level with the car, or behind it. It is also the
    # only part of this that knows about obstacles.
    searched = _search_manoeuvre(slot, geometry, occupancy)
    if searched is not None:
        return searched

    # No canned fallback beyond this point, deliberately. A hand-built
    # "position then reverse" family used to sit here and it is strictly
    # worse than the search: it covers a subset of the same poses, and the
    # sequences it produced could leave the car at a pose its own single-arc
    # solver could not then drive out of -- stuck mid-manoeuvre with nowhere
    # to go. If the search finds nothing, nothing fits.
    return None


def blocking_distance(
    path: ParkingPath,
    obstacles: np.ndarray,
    geometry: VehicleGeometry,
    reverse: bool = False,
    start_index: int = 0,
) -> float:
    """
    Arc length at which something first intrudes on the swept body, or inf.

    Parking drives deliberately close to things, so it cannot lean on AEB --
    which is in STANDBY at parking speed by design (see
    `PARKING_DRIVE_SPEED_MPS`) and would be the wrong instrument anyway. This
    is the manoeuvre's own check, over the corridor the body actually sweeps.
    """
    if obstacles is None or not len(obstacles):
        return math.inf
    points = np.asarray(obstacles, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return math.inf
    half_width = geometry.width_m * 0.5 + PARKING_BODY_CLEARANCE_M
    start_index = max(0, min(int(start_index), len(path.points) - 1))
    if start_index + 1 < len(path.points):
        initial_tangent = path.points[start_index + 1] - path.points[start_index]
    elif start_index:
        initial_tangent = path.points[start_index] - path.points[start_index - 1]
    else:
        initial_tangent = np.asarray((0.0, 1.0))
    initial_heading = math.atan2(
        float(initial_tangent[0]), float(initial_tangent[1])
    ) + (math.pi if reverse else 0.0)
    initial_forward = np.asarray(
        (math.sin(initial_heading), math.cos(initial_heading))
    )
    initial_right = np.asarray((initial_forward[1], -initial_forward[0]))
    initial_relative = points - path.points[start_index]
    initially_overlapping = (
        (np.abs(initial_relative @ initial_right) <= half_width)
        & (initial_relative @ initial_forward <= geometry.front_m)
        & (initial_relative @ initial_forward >= -geometry.rear_m)
    )
    # Returns already inside the current body envelope are self/adjacent
    # overlap, not a new obstacle ahead. They cannot become a future collision
    # and used to make every close parking start report BLOCKED at zero metres.
    points = points[~initially_overlapping]
    if not len(points):
        return math.inf
    for index in range(start_index, len(path.points)):
        if index + 1 < len(path.points):
            tangent = path.points[index + 1] - path.points[index]
        elif index:
            tangent = path.points[index] - path.points[index - 1]
        else:
            tangent = np.asarray((0.0, 1.0))
        if float(np.linalg.norm(tangent)) < 1e-9:
            continue
        travel_heading = math.atan2(float(tangent[0]), float(tangent[1]))
        heading = travel_heading + (math.pi if reverse else 0.0)
        forward = np.asarray((math.sin(heading), math.cos(heading)))
        right = np.asarray((forward[1], -forward[0]))
        relative = points - path.points[index]
        lateral = relative @ right
        longitudinal = relative @ forward
        inside = (
            (np.abs(lateral) <= half_width)
            & (longitudinal <= geometry.front_m)
            & (longitudinal >= -geometry.rear_m)
        )
        if inside.any():
            return float(
                path.cumulative_m[index] - path.cumulative_m[start_index]
            )
    return math.inf


class ParkingDriver:
    """
    Walks the planned legs: one goal per leg, re-derived path, own pedals.

    The manoeuvre is planned ONCE, at engage, and then committed to. Only the
    path to the CURRENT leg is re-derived each tick -- the leg itself is held
    in the bay's frame, so it stays put while the car moves. Re-planning the
    sequence every tick would reintroduce the re-choice problem this whole
    controller exists to avoid: the car would flip between manoeuvres and
    finish none, which is precisely why the S-curve family was never landed
    in the arc planner.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._curvature = 0.0
        self._throttle = 0.0
        self._brake = 0.0
        self._phase = PARK_APPROACH
        self._legs: list[ParkingLeg] | None = None
        self._leg_index = 0
        self._shifting = False
        self._held = False
        self._securing = False
        self._secure_dwell = 0.0
        self._blocked = False
        self._blocked_clear_dwell = 0.0
        self._replans = 0
        self._shift_dwell = 0.0
        self._gear_probe = 0
        self._progress_index = 0
        self._progress_elapsed = 0.0
        self._last_remaining = math.inf
        # The last gear actually commanded. A finished park must keep asking
        # for the gear it ARRIVED in: the car is still rolling to a stop, and
        # a park that ends by reversing in would otherwise be sent back to
        # drive at a metre per second -- caught in test at 1.01 m/s.
        self._gear = 2

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def legs(self) -> list[ParkingLeg] | None:
        return self._legs

    def step(
        self,
        slot: ParkingSlot | None,
        geometry: VehicleGeometry,
        speed_mps: float,
        dt: float,
        obstacles: np.ndarray | None = None,
        occupancy: Occupancy | None = None,
        reported_gear: object = None,
        forward_gear: int = 2,
        rear_aeb_braking: bool = False,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """One tick of the manoeuvre. `slot` is this tick's projected bay."""
        if slot is None:
            return self._halt(
                PARK_UNREACHABLE,
                "The selected bay is no longer in view",
                forward_gear,
                speed_mps,
                dt,
            )
        if self._held:
            return self._halt(
                PARK_ARRIVED,
                "Parked",
                self._gear,
                speed_mps,
                dt,
                parking_brake=1.0,
            )
        if self._legs is None:
            self._legs = plan_manoeuvre(slot, geometry, occupancy)
            self._leg_index = 0
            self._progress_index = 0
            if self._legs is None:
                return self._halt(
                    PARK_UNREACHABLE,
                    reachability(slot, geometry),
                    forward_gear,
                    speed_mps,
                    dt,
                )
        leg = self._legs[self._leg_index]
        gear = REVERSE_GEAR if leg.reverse else forward_gear

        if self._securing:
            return self._secure(slot, geometry, leg, gear, speed_mps, dt)
        # NOT assigned to `self._gear` here. That field is what the shift
        # holds while the car is still rolling, so setting it to the gear we
        # are about to ask for makes the hold a no-op and sends the shift at
        # speed -- caught at 1.13 m/s. It is committed in `_shift`, once the
        # car has actually stopped.

        # The rear brake ARMS at parking speed (0.5 m/s against the forward
        # system's 2.0), so unlike a forward park it really can fire while
        # backing in -- which is exactly when it should. It is left armed and
        # allowed to win; if it fires the manoeuvre hands back rather than
        # fighting it, because a brake that has decided the car is about to
        # hit something outranks a plan that says otherwise.
        if rear_aeb_braking and leg.reverse:
            self._blocked = True
            self._blocked_clear_dwell = 0.0
        if self._blocked:
            clear_and_stopped = (
                not rear_aeb_braking
                and abs(speed_mps) <= PARKING_SHIFT_SPEED_MPS
            )
            self._blocked_clear_dwell = (
                self._blocked_clear_dwell + dt if clear_and_stopped else 0.0
            )
            if self._blocked_clear_dwell < PARKING_BLOCKED_CLEAR_DWELL_S:
                return self._halt(
                    PARK_BLOCKED,
                    "Waiting for the obstruction to clear",
                    gear,
                    speed_mps,
                    dt,
                )
            self._blocked = False
            self._blocked_clear_dwell = 0.0

        if self._gear_probe:
            requested = -1 if leg.reverse else 1
            if abs(speed_mps) > PARKING_SHIFT_SPEED_MPS:
                if speed_mps * requested < 0.0:
                    return self._halt(
                        PARK_UNREACHABLE,
                        "Transmission moved opposite the requested direction",
                        gear,
                        speed_mps,
                        dt,
                    )
                self._gear_probe = 0
            elif reported_gear is not None:
                engaged = (
                    is_reverse_gear(reported_gear)
                    if leg.reverse
                    else is_forward_gear(reported_gear)
                )
                if engaged:
                    self._gear_probe = 0

        if rear_aeb_braking and leg.reverse:
            return self._halt(
                PARK_BLOCKED,
                "Reverse emergency braking stopped the manoeuvre",
                gear,
                speed_mps,
                dt,
            )

        target, axis = leg_pose(leg, slot)
        final = self._leg_index + 1 >= len(self._legs)
        if self._reached(target, axis, leg.reverse, final):
            return self._advance(slot, geometry, gear, speed_mps, dt)

        # A direction change is triggered by the gear this leg NEEDS differing
        # from the one last committed -- not by what the box reports. Keying
        # it on the report re-entered the shift on every tick whenever the
        # gearbox could not be read, so it could never finish: the dwell
        # cleared it and the very next tick set it again. What the box reports
        # still decides when the shift is DONE; it just no longer decides when
        # one starts.
        if gear != self._gear:
            self._shifting = True
        if self._shifting:
            return self._shift(
                leg, forward_gear, reported_gear, speed_mps, dt
            )

        path = _leg_path(leg, slot)
        if path is None:
            # Close to the leg's pose, "no path reaches it" means the car is
            # essentially ON it -- the same endgame the single forward move
            # hit, now once per leg: the tracker rolls a little past and no
            # forward construction reaches back. Treat it as made.
            if float(np.linalg.norm(target)) <= PARKING_LEG_CLOSE_M:
                return self._advance(slot, geometry, gear, speed_mps, dt)
            # Otherwise the car has drifted off the pose the sequence was
            # planned from, and the REST of that sequence no longer solves.
            # Re-plan once from where it actually is. This is not per-tick
            # re-choosing -- it happens only when the committed plan has
            # become undriveable, and the goal (the bay) never changes.
            # Bounded: a re-plan that keeps producing a sequence the car
            # cannot drive would otherwise cycle for ever -- plan, reach the
            # setup, fail the next leg, re-plan. Giving up and handing back is
            # the honest end.
            return self._replan_or_fail(
                slot,
                geometry,
                occupancy,
                forward_gear,
                reported_gear,
                gear,
                speed_mps,
                dt,
                reachability(slot, geometry),
            )
        sampled = _as_path(path)
        # Where the car IS on this path. A re-derived path starts at the car
        # so index 0 is right, but a SEARCHED leg is a fixed path over the
        # ground and the car is somewhere along it -- assuming index 0 sent
        # the tracker chasing the far end and drove the car 71 m away.
        nearest = int(np.argmin(np.linalg.norm(sampled.points, axis=1)))
        here = max(self._progress_index, nearest)
        self._progress_index = here
        cross_track = float(np.linalg.norm(sampled.points[here]))
        blocked_at = blocking_distance(
            sampled,
            obstacles,
            geometry,
            reverse=leg.reverse,
            start_index=here,
        )
        remaining = sampled.length_m - float(sampled.cumulative_m[here])
        if cross_track > PARKING_DRIVE_MAX_CROSS_TRACK_M:
            return self._replan_or_fail(
                slot,
                geometry,
                occupancy,
                forward_gear,
                reported_gear,
                gear,
                speed_mps,
                dt,
                f"Tracking error grew to {cross_track:.1f} m",
            )
        if remaining < self._last_remaining - 0.03:
            self._progress_elapsed = 0.0
        elif abs(speed_mps) > PARKING_SHIFT_SPEED_MPS:
            self._progress_elapsed += dt
        self._last_remaining = remaining
        if self._progress_elapsed >= PARKING_PROGRESS_TIMEOUT_S:
            return self._replan_or_fail(
                slot,
                geometry,
                occupancy,
                forward_gear,
                reported_gear,
                gear,
                speed_mps,
                dt,
                "The vehicle stopped making progress",
            )
        if remaining <= PARKING_ARRIVE_TOLERANCE_M:
            # Driven this leg's path to its end. For a SEARCHED leg that is
            # the arrival test that matters: the nearest point can reach the
            # last sample while the car is still half a metre from the pose
            # itself, so a pose-distance test alone left it stuck in BACKING
            # with nowhere further to drive.
            return self._advance(slot, geometry, gear, speed_mps, dt)
        if blocked_at < remaining:
            self._blocked = True
            self._blocked_clear_dwell = 0.0
            return self._halt(
                PARK_BLOCKED,
                f"Something is in the way at {blocked_at:.1f} m",
                gear,
                speed_mps,
                dt,
                path=sampled,
            )

        self._pursue(sampled, speed_mps, dt, leg.reverse, here)
        heading_left = abs(math.atan2(float(axis[0]), float(axis[1])))
        target_speed = self._target_speed(remaining, heading_left)
        throttle, brake = self._pedals(target_speed, abs(speed_mps), dt)
        if self._gear_probe:
            throttle = min(throttle, 0.12)
        self._phase = PARK_BACKING if leg.reverse else PARK_APPROACH
        reason = (
            f"{'Reversing in' if leg.reverse else 'Positioning'} "
            f"(leg {self._leg_index + 1} of {len(self._legs)}), "
            f"{remaining:.1f} m to go"
        )
        return (
            ControlCommand(
                steering=self._steering(),
                throttle=throttle,
                brake=brake,
                gear=gear,
                mode="PARKING",
                target_speed_mps=target_speed,
                reason=reason,
            ),
            ParkingDriveState(
                phase=self._phase,
                remaining_m=remaining,
                cross_track_m=cross_track,
                curvature=self._curvature,
                target_speed_mps=target_speed,
                reason=reason,
                path=sampled,
            ),
        )

    # --- the pieces ----------------------------------------------------------

    def _advance(
        self,
        slot: ParkingSlot,
        geometry: VehicleGeometry,
        gear: int,
        speed: float,
        dt: float,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """Finish this leg: park if it was the last, otherwise shift."""
        assert self._legs is not None
        if self._leg_index + 1 >= len(self._legs):
            self._securing = True
            self._secure_dwell = 0.0
            return self._secure(
                slot, geometry, self._legs[self._leg_index], gear, speed, dt
            )
        # RE-PLAN at the cusp, but BOUNDED. Unbounded it can cycle -- plan,
        # drive to the cusp, re-plan the same thing -- and every cycle costs a
        # whole search, so it presents as the car shuffling for ever AND as
        # the worker thread hitching. Measured on a bay 2 m to the side and
        # level with the car, which never converged. Past the budget the rest
        # of the stored plan is driven as-is, which is the honest fallback:
        # slightly less accurate than a fresh plan, and finite.
        self._replans += 1
        self._progress_index = 0
        if self._replans > PARKING_MAX_REPLANS:
            self._leg_index += 1
            self._shifting = True
            return self._halt(
                PARK_SHIFTING, "Selecting gear", gear, speed, dt
            )
        # Within the budget, plan afresh from where the car ACTUALLY is. Leg
        # two was planned from where leg one was meant to finish, and over a
        # few metres of reversing there is no distance in which to soak that
        # up -- measured, it parked 1.20 m off the centreline. Planning afresh
        # resets the error, and it is affordable because a cusp happens once
        # or twice a manoeuvre, not once a tick.
        self._legs = None
        self._leg_index = 0
        self._shifting = False
        return self._halt(PARK_SHIFTING, "Selecting gear", gear, speed, dt)

    def _secure(
        self,
        slot: ParkingSlot,
        geometry: VehicleGeometry,
        leg: ParkingLeg,
        gear: int,
        speed: float,
        dt: float,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """Verify the final pose at rest, dwell, then secure the vehicle."""
        target, axis = leg_pose(leg, slot)
        heading_error = abs(math.atan2(float(axis[0]), float(axis[1])))
        bay_forward = bay_axis(slot)
        bay_right = np.asarray((bay_forward[1], -bay_forward[0]))
        centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
        corners = np.asarray(
            (
                (-geometry.left_m, geometry.front_m),
                (geometry.right_m, geometry.front_m),
                (-geometry.left_m, -geometry.rear_m),
                (geometry.right_m, -geometry.rear_m),
            )
        )
        relative = corners - centre
        inside = bool(
            np.all(
                np.abs(relative @ bay_right)
                <= slot.width_m * 0.5 + PARKING_SUCCESS_BOUNDARY_TOLERANCE_M
            )
            and np.all(
                np.abs(relative @ bay_forward)
                <= slot.depth_m * 0.5 + PARKING_SUCCESS_BOUNDARY_TOLERANCE_M
            )
        )
        valid_pose = (
            float(np.linalg.norm(target)) <= PARKING_SUCCESS_POSITION_M
            and heading_error <= math.radians(PARKING_SUCCESS_HEADING_DEG)
            and inside
        )
        stopped = abs(speed) <= PARKING_SUCCESS_SPEED_MPS
        self._secure_dwell = (
            self._secure_dwell + dt if valid_pose and stopped else 0.0
        )
        if self._secure_dwell >= PARKING_SUCCESS_DWELL_S:
            self._held = True
            self._securing = False
            return self._halt(
                PARK_ARRIVED,
                "Parked and secured",
                gear,
                speed,
                dt,
                parking_brake=1.0,
            )
        if stopped and not valid_pose:
            self._replans += 1
            if self._replans > PARKING_MAX_REPLANS:
                return self._halt(
                    PARK_UNREACHABLE,
                    "Could not settle fully inside the selected bay",
                    gear,
                    speed,
                    dt,
                )
            # The goal stays latched; only the path is reconsidered from the
            # measured stopped pose. This is the event-driven correction for
            # endpoint overshoot, not the old every-frame plan re-choice.
            self._securing = False
            self._legs = None
            self._leg_index = 0
            self._progress_index = 0
            self._progress_elapsed = 0.0
            self._last_remaining = math.inf
            return self._halt(
                PARK_SHIFTING,
                "Correcting the final parking pose",
                gear,
                speed,
                dt,
            )
        reason = "Stopping and verifying the final pose"
        return self._halt(PARK_SECURING, reason, gear, speed, dt)

    def _replan_or_fail(
        self,
        slot: ParkingSlot,
        geometry: VehicleGeometry,
        occupancy: Occupancy | None,
        forward_gear: int,
        reported_gear: object,
        gear: int,
        speed: float,
        dt: float,
        failure_reason: str,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """Perform one bounded event-driven replan without changing the bay."""
        self._replans += 1
        replanned = (
            plan_manoeuvre(slot, geometry, occupancy)
            if self._replans <= PARKING_MAX_REPLANS
            else None
        )
        if replanned is None:
            return self._halt(
                PARK_UNREACHABLE, failure_reason, gear, speed, dt
            )
        self._legs = replanned
        self._leg_index = 0
        self._progress_index = 0
        self._progress_elapsed = 0.0
        self._last_remaining = math.inf
        self._shifting = False
        return self._shift(
            replanned[0], forward_gear, reported_gear, speed, dt
        )

    @staticmethod
    def _reached(
        target: np.ndarray, axis: np.ndarray, reverse: bool, final: bool
    ) -> bool:
        """
        Whether this leg's pose has been made -- POSITION and, for a setup
        leg, HEADING.

        The heading test is not a refinement. A setup pose exists to leave the
        car square to the aisle so the reverse can solve from it, and position
        alone declared it made with the car still turning: the next leg then
        would not solve from where the car actually was, the manoeuvre
        re-planned, arrived at the same setup, and cycled -- measured, stuck
        in SHIFTING for the whole run. The FINAL leg is exempt because there
        is nothing after it to solve, and its heading is what the tracker
        spends the last metre correcting.

        The distance test is primary and the "driven past it" test only
        applies CLOSE IN: a setup pose can start out beside or behind the car,
        and a bare projection test would call the leg finished before it
        began.
        """
        distance = float(np.linalg.norm(target))
        square = final or abs(
            math.atan2(float(axis[0]), float(axis[1]))
        ) <= math.radians(PARKING_LEG_SQUARE_DEG)
        if distance <= PARKING_ARRIVE_TOLERANCE_M and square:
            return True
        # The "driven past it" test is for OVERSHOOT, which is centimetres,
        # so it only applies very close in. At a metre it fired while the car
        # was still a metre off to the side and latched the park there --
        # measured, 0.96 m off the centreline of a 3.18 m bay.
        progress = -float(target[1]) if reverse else float(target[1])
        return distance < PARKING_OVERSHOOT_M and progress <= 0.0 and square

    def _shift(
        self,
        leg: ParkingLeg,
        forward_gear: int,
        reported_gear: object,
        speed: float,
        dt: float,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """
        Brake to rest, THEN select the new gear, then wait for confirmation.

        The order is the whole point and the obvious version is wrong: a leg
        ends while the car is still rolling, so commanding the new direction
        as soon as the leg ends sends a reverse shift at over a metre per
        second -- caught in test at 1.22 m/s. `shiftToGearIndex` has side
        effects and the box only engages at rest, so the old gear is held
        until the car has actually stopped, and only then is the new one
        asked for. The gear REPORTED is the only evidence it took; commanding
        it and driving on would have the car pulling against a box still in
        the old direction.
        """
        stopped = abs(speed) <= PARKING_SHIFT_SPEED_MPS
        self._shift_dwell = self._shift_dwell + dt if stopped else 0.0
        wanted = REVERSE_GEAR if leg.reverse else forward_gear
        # Hold whatever the box is ACTUALLY in until the car has stopped,
        # rather than deriving it from the previous leg -- a fresh plan at a
        # cusp has no previous leg to derive it from, and sending the new
        # direction while rolling shifted a moving box at 1.13 m/s.
        gear = wanted if stopped else self._gear
        self._gear = gear
        engaged = (
            is_reverse_gear(reported_gear)
            if leg.reverse
            else is_forward_gear(reported_gear)
        )
        # A readable report is confirmation. If electrics are unavailable,
        # the dwell enters a tightly limited direction probe; signed motion
        # must then agree with the request before normal propulsion continues.
        if stopped and engaged:
            self._shifting = False
            self._shift_dwell = 0.0
            self._gear_probe = 0
        elif (
            stopped
            and reported_gear is None
            and self._shift_dwell >= PARKING_SHIFT_DWELL_S
        ):
            self._shifting = False
            self._shift_dwell = 0.0
            self._gear_probe = -1 if leg.reverse else 1
        self._throttle = 0.0
        self._curvature = 0.0
        self._brake = _slew(self._brake, PARKING_STOP_BRAKE, 6.0, 4.0, dt)
        reason = "Selecting reverse" if leg.reverse else "Selecting drive"
        return (
            ControlCommand(
                steering=0.0,
                throttle=0.0,
                brake=self._brake,
                gear=gear,
                mode="PARKING",
                target_speed_mps=0.0,
                reason=reason,
            ),
            ParkingDriveState(
                phase=PARK_SHIFTING,
                remaining_m=0.0,
                cross_track_m=0.0,
                curvature=0.0,
                target_speed_mps=0.0,
                reason=reason,
            ),
        )

    def _pursue(
        self,
        path: ParkingPath,
        speed: float,
        dt: float,
        reverse: bool,
        here: int = 0,
    ) -> float:
        """
        Pure pursuit on the path, mirrored when the car is going backwards.

        A reverse leg's samples lie BEHIND the car, so the tracker runs on the
        negated path -- the frame the car is actually travelling in -- and the
        curvature comes back negated. That is the same relation the steered
        reverse uses: travel-frame k_m equals minus the front-frame k_f,
        because reversing yaw is `v_signed * k_f` with the speed negative.
        Negating here, before STEERING_SIGN, keeps the one-place-reconciles
        rule intact.
        """
        points = -path.points if reverse else path.points
        lookahead = max(
            1.2, min(PARKING_DRIVE_LOOKAHEAD_M, 1.2 + abs(speed) * 1.1)
        )
        # Look ahead from where the car actually IS on the path, not from the
        # path's beginning.
        ahead = np.searchsorted(
            path.cumulative_m, path.cumulative_m[here] + lookahead
        )
        goal = points[min(ahead, len(points) - 1)]
        distance = float(np.linalg.norm(goal))
        pursuit = (
            0.0 if distance < 1e-3 else -2.0 * float(goal[0]) / distance**2
        )
        if reverse:
            pursuit = -pursuit
        tangent_index = min(max(here, 0), len(path.points) - 2)
        tangent = path.points[tangent_index + 1] - path.points[tangent_index]
        travel_heading = math.atan2(float(tangent[0]), float(tangent[1]))
        desired_heading = travel_heading + (math.pi if reverse else 0.0)
        heading_error = (desired_heading + math.pi) % (2.0 * math.pi) - math.pi
        path_curvature = _curvature(path.points)
        feed_forward = float(path_curvature[tangent_index])
        if reverse:
            feed_forward = -feed_forward
        # Pure pursuit brings the car back to the geometric path. Curvature
        # feed-forward stops it cutting a committed bend, while explicit
        # heading feedback is what makes the final straight actually finish
        # square instead of declaring arrival with several degrees still left.
        target = (
            0.65 * pursuit
            + 0.55 * feed_forward
            - 0.8 * heading_error / max(lookahead, 1.0)
        )
        target = max(-_MAX_CURVATURE, min(_MAX_CURVATURE, target))
        rate = 0.9
        self._curvature = max(
            self._curvature - rate * dt,
            min(self._curvature + rate * dt, target),
        )
        return self._curvature

    @staticmethod
    def _target_speed(remaining: float, turn_remaining: float) -> float:
        """
        Distance-to-go, not the road law -- and capped while still turning.

        The road speed law is a proportional term on a target chosen from free
        distance, behind a low-pass, and it carries a standing error of a few
        m/s, which is the whole speed budget here. Braking to a stop over a
        known distance is what parking needs, and it is exact.

        The TURN cap is the second half and it is not comfort. Distance alone
        let the car cross the stop plane at cruising speed with the wheel
        still wound on, arriving 7.2 degrees off square -- centred in the bay
        but visibly crooked. Slowing while there is heading left to lose gives
        the tracker the time to straighten, which is what a driver does.
        """
        stopping = math.sqrt(
            2.0 * PARKING_DRIVE_DECEL_MPS2 * max(remaining, 0.0)
        )
        if turn_remaining > math.radians(PARKING_TURN_SLOW_DEG):
            stopping = min(stopping, PARKING_DRIVE_CREEP_MPS * 1.6)
        rolling = min(PARKING_DRIVE_SPEED_MPS, stopping)
        if remaining <= PARKING_DRIVE_CREEP_HOLD_M:
            # Inside the last metre the creep floor is what would carry the
            # car through the stop point, so the profile is allowed all the
            # way to zero. Above it the floor keeps the car moving against a
            # gentle grade rather than stalling short.
            return rolling
        return max(rolling, PARKING_DRIVE_CREEP_MPS)

    def _pedals(
        self, target: float, speed: float, dt: float
    ) -> tuple[float, float]:
        """One foot works both pedals, as in the road controller."""
        error = target - speed
        demand = max(-PARKING_DRIVE_DECEL_MPS2 * 2.0, min(1.5, error * 1.8))
        if demand >= 0.0:
            throttle = min(1.0, demand / THROTTLE_GAIN_MPS2)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -demand / BRAKE_GAIN_MPS2)
        self._throttle = _slew(self._throttle, throttle, 3.0, 6.0, dt)
        self._brake = _slew(self._brake, brake, 4.0, 8.0, dt)
        if self._brake > 0.02:
            self._throttle = 0.0
        return self._throttle, self._brake

    def _steering(self) -> float:
        return max(
            -1.0, min(1.0, STEERING_SIGN * self._curvature / _MAX_CURVATURE)
        )

    def _halt(
        self,
        phase: str,
        reason: str,
        gear: int,
        speed: float,
        dt: float,
        path: ParkingPath | None = None,
        parking_brake: float = 0.0,
    ) -> tuple[ControlCommand, ParkingDriveState]:
        """
        Stop and hold. ARRIVED holds the car; every other end hands it back.

        A finished park must stay put -- releasing at the stop line lets the
        car roll on out of the bay -- so the brake is HELD there. A blocked or
        unreachable manoeuvre stops just as hard but is a handover, and the
        worker disengages on it, which releases through the usual funnel.
        """
        self._phase = phase
        self._throttle = 0.0
        self._curvature = 0.0
        self._brake = _slew(
            self._brake,
            PARKING_STOP_BRAKE
            if abs(speed) > 0.05
            else PARKING_STOP_BRAKE * 0.6,
            6.0,
            2.0,
            dt,
        )
        return (
            ControlCommand(
                steering=self._steering(),
                throttle=0.0,
                brake=self._brake,
                gear=gear,
                mode="PARKING",
                target_speed_mps=0.0,
                reason=reason,
                parking_brake=parking_brake,
            ),
            ParkingDriveState(
                phase=phase,
                remaining_m=0.0 if path is None else path.length_m,
                cross_track_m=0.0,
                curvature=0.0,
                target_speed_mps=0.0,
                reason=reason,
                path=path,
            ),
        )


def _slew(previous: float, target: float, up: float, down: float, dt: float) -> float:
    limit = up * dt if target > previous else down * dt
    return previous + max(-limit, min(limit, target - previous))
