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
bay-relative TRAJECTORY -- planned once, partitioned at its cusps, smoothed
per leg (`parking_smooth`), and then DRIVEN, not re-chosen. The controller
re-projects both into the moving BEV frame, advances monotonically along the
path, and re-solves geometry only on explicit triggers: a persistent
blockage (after braking to a stop SHORT of it, never slamming for something
metres down the path), gross tracking error, no progress, or a final pose
that missed by more than the analytic nudge law can fix. A cusp is NOT a
trigger: the next committed leg is driven as planned, locally repaired from
the actual pose only if the car stopped meaningfully off it -- re-searching
the whole manoeuvre at every cusp re-chose the topology every time, which
was most of the live shuffling. Hybrid A* may produce forward and reverse
legs, with a stopped gear handshake between them, during which the wheels
are pre-aimed at the next leg's entry curvature. Success is a separate
stopped verification dwell followed by the parking brake; merely crossing
the path endpoint is never success.

Tracking is an error-state law, not pure pursuit: curvature feed-forward
read a little AHEAD of the matched point (compensating the steering
actuator's own winding time) plus lateral- and heading-error feedback, in
the travel frame, mirrored once for reverse -- and a clamped adaptive gain
absorbs the difference between the assumed steering map and the one the car
measurably drives, exactly as `controller._adapt_gain` does on the road.

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
    PARKING_BAY_KEEPOUT_M,
    PARKING_BLOCK_STANDOFF_M,
    PARKING_BLOCKED_CLEAR_DWELL_S,
    PARKING_BLOCKED_REPLAN_S,
    PARKING_BODY_CLEARANCE_M,
    PARKING_CONTROL_LAG_S,
    PARKING_DRIVE_APPROACH_M,
    PARKING_DRIVE_CREEP_HOLD_M,
    PARKING_DRIVE_CREEP_MPS,
    PARKING_DRIVE_DECEL_MPS2,
    PARKING_DRIVE_MAX_CROSS_TRACK_M,
    PARKING_DRIVE_SPEED_MPS,
    PARKING_HEAD_CLEARANCE_M,
    PARKING_LEG_CLOSE_M,
    PARKING_LEG_REPAIR_DEG,
    PARKING_LEG_REPAIR_M,
    PARKING_LEG_SQUARE_DEG,
    PARKING_MAX_CUSP_REPLANS,
    PARKING_MAX_REPLANS,
    PARKING_NUDGE_HEADING_DEG,
    PARKING_NUDGE_MAX_M,
    PARKING_NUDGE_OUT_MAX_M,
    PARKING_NUDGE_OUT_MIN_M,
    PARKING_OVERSHOOT_M,
    PARKING_PATH_SAMPLES,
    PARKING_PATH_SLACK_M,
    PARKING_PATH_STEP_M,
    PARKING_PLAN_RADIUS_MARGIN,
    PARKING_PROGRESS_TIMEOUT_S,
    PARKING_REAR_AXLE_OFFSET_M,
    PARKING_REVERSE_FIRST_DEG,
    PARKING_SEARCH_RUN_IN_M,
    PARKING_SHIFT_DWELL_S,
    PARKING_SHIFT_SPEED_MPS,
    PARKING_SMOOTH_MAX_DEVIATION_M,
    PARKING_STEER_GAIN_ADAPT_RATE,
    PARKING_STEER_GAIN_MAX,
    PARKING_STEER_GAIN_MIN,
    PARKING_STEER_GAIN_MIN_CURVATURE,
    PARKING_STEER_GAIN_MIN_SPEED_MPS,
    PARKING_STEER_RATE_PER_S,
    PARKING_STOP_BRAKE,
    PARKING_SUCCESS_BOUNDARY_TOLERANCE_M,
    PARKING_SUCCESS_DWELL_S,
    PARKING_SUCCESS_HEADING_DEG,
    PARKING_SUCCESS_POSITION_M,
    PARKING_SUCCESS_SPEED_MPS,
    PARKING_TARGET_RAMP_DOWN_MPS2,
    PARKING_TARGET_RAMP_UP_MPS2,
    PARKING_TRACK_APPROACH_GAIN,
    PARKING_TRACK_APPROACH_MAX_DEG,
    PARKING_TRACK_FEEDFORWARD_GAIN,
    PARKING_TRACK_HEADING_GAIN,
    PARKING_TRACK_PREVIEW_S,
    PARKING_TURN_SLOW_DEG,
    PARKING_TURN_SLOW_RANGE_M,
    PARKING_TURN_SPEED_CURVATURE,
    PARKING_TURN_SPEED_MPS,
    STEERING_SIGN,
    THROTTLE_GAIN_MPS2,
)
from .controller import REVERSE_GEAR, is_forward_gear, is_reverse_gear
from .hybrid_astar import Occupancy, PlannedPath, Pose, plan
from .models import ControlCommand, ParkingSlot, VehicleGeometry
from .parking_smooth import discrete_curvature, smooth_path

_MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M

# Where the most recent plan came from, for the worker's Park plan: line.
# A diagnostic, not state: worker-thread confined like everything here, and
# it exists because a 4.0x initial plan and a clean 2-legger looked identical
# in the log -- whether a shape came from the clean keepout round, the bare
# round, the line-clipping fallback, a canned family or the nudge is the
# first question a bad plan raises.
LAST_PLAN_NOTE = ""

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
    # At least as fine as the grid every path is finally re-sampled to.
    # `resample` interpolates LINEARLY between the samples it is given, so
    # upsampling a curve puts the new points on its CHORDS -- and the discrete
    # curvature then reads zero along a chord and spikes at every original
    # vertex. Measured on a 6 m arc emitted at 0.41 m and re-sampled to 0.25:
    # a clean 0.167 came back as 0.245 on every third sample, a steering limit
    # violation that was an artefact of the sampling rather than of the path.
    span = np.linspace(
        0.0,
        total,
        max(samples, 12, int(math.ceil(total / PARKING_PATH_STEP_M)) + 1),
    )
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


# Shared with the smoothing stage so the curvature the smoother bounds and the
# curvature the tracker feeds forward are one computation, not two to drift.
_curvature = discrete_curvature


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
            # Close the residual to the target with an ARC that continues the
            # heading the sweep finished on, not with a straight line to it.
            #
            # The slack clamp above may have moved the end by a few
            # centimetres, so the sweep does not in general finish pointing at
            # the target -- and a straight drawn from there to it is a CORNER.
            # It was invisible while the path was sampled per-segment, because
            # a kink's discrete curvature scales with the sample spacing and
            # the join sat next to the coarsest gap in the path; measured on a
            # uniform grid it reads 0.245 against a steering limit of 0.167,
            # and the tracker saturates there. An arc is continuous with the
            # sweep by construction and stays inside the limit or is refused,
            # which is what the (approach, radius) search is for.
            tail = _tail_arc(swept, target, turn, samples)
            if tail is None:
                continue
            return np.concatenate((swept, tail))

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


def resample(
    points: np.ndarray, step_m: float = PARKING_PATH_STEP_M
) -> np.ndarray:
    """
    A path re-sampled at a CONSTANT arc-length step, ending where it ended.

    Every construction here samples its own SEGMENTS rather than the finished
    path, and the segments have wildly different lengths: a square-on bay
    comes back as two points for the run-in and then the tail at a quarter of
    a metre, so the path can open with one 6.3 m gap. The tracker locates the
    car by the NEAREST SAMPLE, and across a gap like that sample 0 stays
    nearest for the first three metres -- which quietly redefines three
    different quantities (see `PARKING_PATH_STEP_M`, which records what each
    one became and what it cost).

    The endpoints are preserved exactly. The tracker aims at the last sample
    for the final metre, so a path that no longer terminates ON the stop pose
    parks the car somewhere else.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= step_m:
        return pts
    # Coincident samples make `interp`'s x non-increasing, and it does not
    # sort -- it silently returns nonsense. Drop them rather than assuming
    # the constructions never emit one.
    keep = np.concatenate(([True], steps > 1e-9))
    if int(keep.sum()) < 2:
        return pts
    wanted = np.linspace(0.0, total, max(2, int(math.ceil(total / step_m)) + 1))
    return np.column_stack(
        [
            np.interp(wanted, cumulative[keep], pts[keep, axis])
            for axis in (0, 1)
        ]
    )


def _tail_arc(
    swept: np.ndarray, target: np.ndarray, turn: float, samples: int
) -> np.ndarray | None:
    """
    The curvature-bounded arc from the sweep's end onto the target, or None.

    Solved in the sweep's own final frame -- it finishes heading `turn` from
    +forward -- so `_single_arc`'s "from the origin, heading +forward" applies
    unchanged, exactly as `_reverse_reach` reuses the forward solver in a
    rotated frame. None means this radius cannot close the gap without
    exceeding the steering limit, which is a reason to try the next candidate
    rather than to draw a corner.
    """
    forward = np.asarray((math.sin(turn), math.cos(turn)))
    right = np.asarray((forward[1], -forward[0]))
    delta = np.asarray(target, dtype=np.float64) - swept[-1]
    local = np.asarray((float(delta @ right), float(delta @ forward)))
    if float(np.linalg.norm(local)) < 1e-6:
        return np.empty((0, 2))
    arc = _single_arc(local, samples)
    if arc is None:
        return None
    return swept[-1] + np.outer(arc[1:, 0], right) + np.outer(arc[1:, 1], forward)


def _as_path(points: np.ndarray) -> ParkingPath:
    points = resample(points)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return ParkingPath(
        points=points,
        cumulative_m=np.concatenate(([0.0], np.cumsum(steps))),
        max_curvature=float(np.abs(_curvature(points)).max()),
        entry_index=len(points) - 1,
    )


def cross_track(points: np.ndarray, index: int) -> float:
    """
    Perpendicular distance from the car to the path, not to a SAMPLE of it.

    The car sits at the BEV origin, so the distance to `points[index]` is only
    a tracking error when that sample happens to be abeam. Measuring to the
    sample instead reported the distance TRAVELLED whenever the nearest sample
    was stale, and `PARKING_DRIVE_MAX_CROSS_TRACK_M` then fired on a car
    driving perfectly straight down a perfectly straight path. Measuring to
    the two adjoining SEGMENTS is what makes the number mean what its name
    says.
    """
    if len(points) < 2:
        return 0.0 if not len(points) else float(np.linalg.norm(points[0]))
    index = max(0, min(int(index), len(points) - 1))
    best = float(np.linalg.norm(points[index]))
    for start in (index - 1, index):
        if start < 0 or start + 1 >= len(points):
            continue
        segment = points[start + 1] - points[start]
        length_sq = float(segment @ segment)
        if length_sq < 1e-12:
            continue
        # Projection of the origin onto the segment, clamped to it.
        along = float((-points[start]) @ segment) / length_sq
        along = max(0.0, min(1.0, along))
        foot = points[start] + segment * along
        best = min(best, float(np.linalg.norm(foot)))
    return best


def _segment_distance(points: np.ndarray, start: int) -> float:
    """Distance from the car (the origin) to one clamped path segment."""
    segment = points[start + 1] - points[start]
    length_sq = float(segment @ segment)
    if length_sq < 1e-12:
        return float(np.linalg.norm(points[start]))
    along = float((-points[start]) @ segment) / length_sq
    along = max(0.0, min(1.0, along))
    return float(np.linalg.norm(points[start] + segment * along))


def match_index(points: np.ndarray, start: int) -> int:
    """
    The LOCAL perpendicular foot: walk forward from `start` while segments
    keep getting closer, stop at the first minimum.

    Never an argmin over a range. On any bend a car displaced toward the
    inside is closer to samples further AROUND the curve than to its own
    abeam point, so a nearest-sample search creeps ahead along the chord,
    the matched tangent demands ever more turn, the command saturates, and
    the car genuinely cuts inside -- a positive feedback loop measured at
    full lock for five seconds with cross-track growing the whole time. The
    walk is monotone from the last known progress, so it can neither jump
    ahead around a bend nor slide backwards.
    """
    index = max(0, min(int(start), len(points) - 2))
    distance = _segment_distance(points, index)
    while index + 1 <= len(points) - 2:
        ahead = _segment_distance(points, index + 1)
        if ahead < distance - 1e-9:
            index += 1
            distance = ahead
        else:
            break
    return index


def _signed_offset(points: np.ndarray, index: int) -> tuple[float, int]:
    """
    Signed lateral offset of the PATH from the car, and the matched segment.

    Positive when the path lies to the car's LEFT in the frame `points` are
    expressed in, which is the sign that makes `+ k_y * offset` steer toward
    it under the positive-curvature-is-left convention. The magnitude agrees
    with `cross_track` -- same two adjoining segments, same clamped
    projection -- so the tracker corrects exactly the error the watchdogs
    measure.
    """
    index = max(0, min(int(index), len(points) - 2))
    best: tuple[float, float, int] | None = None
    for start in (index - 1, index):
        if start < 0 or start + 1 >= len(points):
            continue
        segment = points[start + 1] - points[start]
        length_sq = float(segment @ segment)
        if length_sq < 1e-12:
            continue
        along = float((-points[start]) @ segment) / length_sq
        along = max(0.0, min(1.0, along))
        foot = points[start] + segment * along
        distance = float(np.linalg.norm(foot))
        if best is None or distance < best[0]:
            best = (distance, float(-foot[0]), start)
    if best is None:
        anchor = points[max(0, min(index, len(points) - 1))]
        return float(-anchor[0]), index
    return best[1], best[2]


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
    arc_length = turn / curvature
    span = np.linspace(
        0.0,
        arc_length,
        max(
            samples // 2,
            8,
            int(math.ceil(abs(arc_length) / PARKING_PATH_STEP_M)) + 1,
        ),
    )
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


def leg_length(leg: ParkingLeg) -> float:
    """How far this leg actually drives, in metres."""
    if leg.path_bay is None or len(leg.path_bay) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(leg.path_bay, axis=0), axis=1).sum()
    )


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
        # The search mixes two spacings -- HYBRID_STEP_M for the expanded
        # primitives and a quarter of a metre for the analytic tail -- so a
        # searched leg needs the same uniform grid the canned ones get, for
        # the same reason: the executor finds the car by nearest sample.
        relative = resample(np.asarray(piece[:, :2], dtype=np.float64)) - centre
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
    # Resampled BEFORE it is frozen into bay coordinates: the projection is a
    # rigid transform, so a uniformly spaced path stays uniformly spaced for
    # the life of the leg and the executor never has to re-derive one.
    relative = resample(np.asarray(path, dtype=np.float64)) - centre
    return replace(
        leg,
        path_bay=np.column_stack((relative @ axis, relative @ across_axis)),
    )


def body_corners(
    points: np.ndarray, reverse: bool, geometry: VehicleGeometry
) -> np.ndarray:
    """
    The four body corners at every sample of a path, in the path's own frame.

    `(N, 4, 2)`. The heading at a sample is the direction of travel there,
    turned through pi when the leg is reversed -- the car faces backwards
    along a reverse leg, and a body drawn along the travel direction would be
    the right rectangle in the wrong place by the whole overhang difference.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.empty((0, 4, 2))
    tangents = np.diff(pts, axis=0)
    tangents = np.vstack((tangents, tangents[-1:]))
    headings = np.arctan2(tangents[:, 0], tangents[:, 1])
    if reverse:
        headings = headings + math.pi
    forward = np.column_stack((np.sin(headings), np.cos(headings)))
    right = np.column_stack((forward[:, 1], -forward[:, 0]))
    corners = (
        (-geometry.left_m, geometry.front_m),
        (geometry.right_m, geometry.front_m),
        (geometry.right_m, -geometry.rear_m),
        (-geometry.left_m, -geometry.rear_m),
    )
    return np.stack(
        [pts + right * cr + forward * cf for cr, cf in corners], axis=1
    )


def bay_intrusion(
    points: np.ndarray,
    reverse: bool,
    slot: ParkingSlot,
    geometry: VehicleGeometry,
) -> float:
    """
    How far the body strays past the bay's SIDE lines while inside its depth.

    **Nothing else anywhere asks this.** Paint is not an obstacle -- there is
    no return to collide with -- so no corridor check, no occupancy cell and
    no cost term prefers staying between the lines, and `_secure` inspects
    only the pose the car finishes in, which says nothing about the path that
    reached it. Measured before this existed: entering a 3.18 m bay from an
    aisle 5 m off its centre, a corner reached 0.74 m past the side line with
    0.58 m of margin available -- roughly three quarters of a metre into the
    neighbouring bay, every time, reported as ARRIVED.

    Bounded on BOTH axes, and the second bound is not tidiness. The depth band
    is an infinite strip across the whole lot, so measured on depth alone a car
    manoeuvring thirty metres away at the same depth as the bay reads as a
    thirty-metre intrusion -- measured, 13.64 m on a perfectly sensible
    positioning leg, which rejected the very manoeuvre this test exists to
    prefer. Past a bay's width either side the car is somewhere else in the
    lot, not across the neighbour's lines.

    Only counted once a corner is INSIDE the bay's depth. A car driving up a
    narrow aisle is legitimately alongside the bay and level with its mouth;
    that is passing it, not parking across it.
    """
    corners = body_corners(points, reverse, geometry)
    if not len(corners):
        return 0.0
    axis = bay_axis(slot)
    across_axis = np.asarray((axis[1], -axis[0]))
    relative = corners.reshape(-1, 2) - np.asarray(
        (slot.centre_right_m, slot.centre_forward_m)
    )
    across = np.abs(relative @ across_axis)
    nearby = (np.abs(relative @ axis) <= slot.depth_m * 0.5) & (
        across <= slot.width_m * 1.5
    )
    if not nearby.any():
        return 0.0
    return float(max(0.0, (across[nearby] - slot.width_m * 0.5).max()))


def _respects_bay(
    points: np.ndarray | None,
    reverse: bool,
    slot: ParkingSlot,
    geometry: VehicleGeometry,
) -> bool:
    if points is None or len(points) < 2:
        return False
    return bay_intrusion(points, reverse, slot, geometry) <= (
        PARKING_BAY_KEEPOUT_M
    )


def prefers_reverse(slot: ParkingSlot) -> bool:
    """
    Whether this bay should be REVERSED into rather than driven into.

    Backing in, the rear axle is the pivot, so the body comes square to the
    bay while it is still outside the mouth. Nose first the front swings wide,
    and one arc through 90 degrees displaces the car by a whole turning
    radius -- which in an ordinary aisle cannot be delivered without crossing
    the neighbouring bay. That is arithmetic rather than taste, and it is why
    a real parking assist reverses into a perpendicular bay.

    "Square-on" is measured against the CAR's heading rather than against a
    modelled aisle, because there is no aisle model here and in the normal
    workflow the car is driving along the aisle when the bay is picked. A bay
    in LINE with the aisle is a different manoeuvre and still goes in nose
    first.
    """
    turn = abs(math.degrees(
        math.atan2(math.sin(slot.heading_rad), math.cos(slot.heading_rad))
    ))
    return abs(turn - 90.0) <= PARKING_REVERSE_FIRST_DEG


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

    Blind to the PAINT as well, which is a separate omission with the same
    shape: there is no return to collide with on a bay line, so the occupancy
    check below can never see one. `_respects_bay` is what stops a canned move
    being taken when it would cut the corner through the neighbouring bay --
    and falling through to the search, which can shuffle, is the right
    consequence rather than driving the cheap path anyway.
    """
    if path is None:
        return False
    if not _respects_bay(path, leg.reverse, slot, geometry):
        return False
    return _occupancy_clear(path, leg.reverse, geometry, occupancy)


def _occupancy_clear(
    path: np.ndarray,
    reverse: bool,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None,
) -> bool:
    """Whether the swept body stays out of KNOWN-blocked cells along `path`."""
    if occupancy is None:
        return True
    half_width = geometry.width_m * 0.5 + PARKING_BODY_CLEARANCE_M
    sampled = path[:: max(1, len(path) // 40)]
    if len(sampled) < 2:
        return False
    tangents = np.diff(sampled, axis=0)
    headings = np.arctan2(tangents[:, 0], tangents[:, 1])
    if reverse:
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


def _smooth_legs(
    legs: list[ParkingLeg],
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None,
) -> list[ParkingLeg]:
    """
    Every committed leg, smoothed and RE-VALIDATED, raw where validation fails.

    Smoothing happens in the bay frame -- a rigid transform of the frame the
    checks run in, so nothing is lost -- and a smoothed leg is kept only when
    it is no worse than the raw one on BOTH counts: bay intrusion (the search
    fallback may already clip a line, so "no worse" rather than "clean") and
    known-blocked occupancy. Peak curvature is allowed a hair over the raw
    figure for discretisation, no more: the deviation clamp can locally
    sharpen what it clips, and a leg the car cannot steer is worse than one
    it tracks late.
    """
    smoothed: list[ParkingLeg] = []
    for leg in legs:
        if leg.path_bay is None or len(leg.path_bay) < 5:
            smoothed.append(leg)
            continue
        raw_path = _leg_path(leg, slot)
        if raw_path is None:
            smoothed.append(leg)
            continue
        raw_peak = float(np.abs(discrete_curvature(raw_path)).max())
        raw_intrusion = bay_intrusion(raw_path, leg.reverse, slot, geometry)
        chosen = leg
        # Full deviation first, a tighter budget as the fallback: a raw
        # searched leg carries a concentrated tangent kink at every steering
        # change (the primitives are sampled at 0.7 m endpoints, so the arc
        # between them is a chord and the join is a corner -- a one-sample
        # 0.4 1/m spike after resampling, far beyond anything the car can
        # steer). Rejecting the smooth outright over a couple of
        # centimetres of intrusion kept exactly those kinks, and the
        # tracker diverged on them at speed.
        for deviation in (
            PARKING_SMOOTH_MAX_DEVIATION_M,
            PARKING_SMOOTH_MAX_DEVIATION_M * 0.4,
        ):
            # NOT re-resampled afterwards: linear interpolation puts the new
            # samples on the smoothed polyline's CHORDS, which re-concentrates
            # the curvature at the original vertices -- measured, it turned a
            # 0.405 kink into 0.519. Smoothing preserves near-uniform spacing
            # (drift is bounded well under a sample step), which is all the
            # nearest-segment matching needs.
            candidate_bay = smooth_path(
                resample(leg.path_bay), max_deviation_m=deviation
            )
            candidate = replace(leg, path_bay=candidate_bay)
            new_path = _leg_path(candidate, slot)
            if new_path is None:
                continue
            new_peak = float(np.abs(discrete_curvature(new_path)).max())
            new_intrusion = bay_intrusion(
                new_path, leg.reverse, slot, geometry
            )
            acceptable = (
                new_peak <= max(raw_peak, _MAX_CURVATURE) + 2e-3
                and new_intrusion
                <= max(raw_intrusion, PARKING_BAY_KEEPOUT_M) + 0.06
                and _occupancy_clear(
                    new_path, leg.reverse, geometry, occupancy
                )
            )
            if acceptable:
                chosen = candidate
                break
        smoothed.append(chosen)
    return smoothed


def bay_side_strips(slot: ParkingSlot) -> np.ndarray:
    """
    The bay's two flanks as synthetic blocked points, in the current BEV.

    Paint returns no LiDAR points, so nothing in the real occupancy can ever
    forbid the search a path across the neighbouring bays -- and given the
    chance it takes one, because crossing the line is always shorter. The
    strips cover the bay's depth either side of it, wide enough that the
    footprint probes cannot step over them, sampled finer than the occupancy
    cell so every cell in the band is filled.
    """
    axis = bay_axis(slot)
    across_axis = np.asarray((axis[1], -axis[0]))
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    along = np.arange(
        -slot.depth_m * 0.5 - 0.3, slot.depth_m * 0.5 + 0.3, 0.25
    )
    across = np.arange(slot.width_m * 0.5 + 0.10, slot.width_m * 0.5 + 2.6, 0.25)
    along_grid, across_grid = np.meshgrid(along, across)
    flanks = np.concatenate((across_grid.ravel(), -across_grid.ravel()))
    alongs = np.concatenate((along_grid.ravel(), along_grid.ravel()))
    return (
        centre
        + np.outer(alongs, axis)
        + np.outer(flanks, across_axis)
    )


def _search_manoeuvre(
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None,
    start_gear: int = 1,
) -> list[ParkingLeg] | None:
    """
    Hybrid A* from the car to the bay's stop pose, as committed legs.

    Both entries are offered. For a square-on bay the REVERSE goal is tried
    first and taken if it solves at all, rather than being priced against the
    nose-in one: `HYBRID_REVERSE_PENALTY` and `HYBRID_GEAR_PENALTY` are set
    for road driving, where an unnecessary reverse is a fault, and in a lot
    they bias against precisely the manoeuvre that fits. Cost is the wrong
    arbiter here -- backing in is longer and slower and still correct, because
    the shorter answer is the one that drives across the neighbouring bay.

    The search is asked at a RADIUS MARGIN rather than at the car's absolute
    minimum, so the tracker keeps authority to tighten. A path the car can
    only just drive is not a path it can follow.
    """
    target, axis = stop_pose(slot, geometry)
    # Reverse entry ends the car facing OUT, so the tail clears the head and
    # the stop pose measures from `rear_m`.
    back_target = (
        np.asarray((slot.centre_right_m, slot.centre_forward_m))
        + axis * (slot.depth_m * 0.5 - PARKING_HEAD_CLEARANCE_M - geometry.rear_m)
    )
    # Searched TO a pose set back along the bay axis, with the last stretch
    # appended as a straight. See `PARKING_SEARCH_RUN_IN_M`: a Reeds-Shepp
    # shot is the shortest path to a pose and may arrive still turning, and a
    # car that crosses the stop plane mid-curve parks crooked however well it
    # tracked. The set-back is along the axis for both entries, because
    # backing in the car still TRAVELS deeper into the bay -- only its heading
    # is reversed.
    run_in = axis * PARKING_SEARCH_RUN_IN_M
    nose_goal = Pose(
        float(target[0] - run_in[0]),
        float(target[1] - run_in[1]),
        slot.heading_rad,
    )
    back_goal = Pose(
        float(back_target[0] - run_in[0]),
        float(back_target[1] - run_in[1]),
        slot.heading_rad + math.pi,
    )
    # Each goal carries the direction its ENTRY leg must be driven in, and
    # that pairing is load-bearing rather than descriptive. Nothing in the
    # occupancy knows the bay has a HEAD -- a kerb annotates as an obstacle
    # live, but an unseen lot is all UNKNOWN and traversable -- so asked only
    # for a pose facing out of the bay, the search cheerfully drove FORWARD
    # in through the head and stopped facing the mouth. Arithmetically a
    # solution, and not a park. Reversing in means the last leg reverses.
    goals = (
        ((back_goal, True), (nose_goal, False))
        if prefers_reverse(slot)
        else ((nose_goal, False), (back_goal, True))
    )
    grid = occupancy or Occupancy(None, None)
    # The search runs first against the occupancy WITH the bay's side lines
    # as virtual obstacles -- paint returns no LiDAR points, so this is the
    # only way the search can be told the lines exist, and given the chance
    # it plans across the neighbouring bay because that is always shorter.
    # The bare grid is the second round: better a manoeuvre that clips a
    # line than no manoeuvre, but only after the clean answers were sought.
    global LAST_PLAN_NOTE
    # The strips must never cover the CAR: engaging from abeam the bay -- an
    # ordinary aisle position -- puts the body inside the flank zone, and a
    # virtual obstacle on top of the car kills the clean round before its
    # first expansion. The car being there is proof the spot is drivable;
    # the strip resumes beyond the body.
    strips = bay_side_strips(slot)
    margin = 0.25
    outside_body = ~(
        (strips[:, 0] >= -geometry.left_m - margin)
        & (strips[:, 0] <= geometry.right_m + margin)
        & (strips[:, 1] >= -geometry.rear_m - margin)
        & (strips[:, 1] <= geometry.front_m + margin)
    )
    keepout = grid.with_blocked(strips[outside_body])
    fallback: PlannedPath | None = None
    for active in (keepout, grid):
        for goal, entry_reverses in goals:
            found = plan(
                Pose(0.0, 0.0, 0.0),
                goal,
                active,
                geometry.width_m * 0.5 + PARKING_BODY_CLEARANCE_M,
                geometry.front_m,
                geometry.rear_m,
                radius=MIN_TURN_RADIUS_M * PARKING_PLAN_RADIUS_MARGIN,
                start_gear=start_gear,
            )
            if found is None:
                continue
            legs = _with_run_in(
                legs_from_path(found, slot), slot, geometry, entry_reverses
            )
            # The clean round is held to a STRICTER planned margin than the
            # keepout itself: the driven body adds a tracking transient on
            # top of whatever the plan uses, so a plan that hugs the full
            # keepout leaves nothing for the car -- measured, 0.15 m of plan
            # plus 0.17 m of transient put a corner 0.32 m over the line.
            tolerance = 0.05 if active is keepout else PARKING_BAY_KEEPOUT_M
            if (
                legs
                and legs[-1].reverse == entry_reverses
                and _legs_respect_bay(legs, slot, geometry, tolerance)
            ):
                LAST_PLAN_NOTE = (
                    f"search, {'keepout' if active is keepout else 'bare'} "
                    f"round, {'reverse' if entry_reverses else 'nose'} entry"
                )
                return _smooth_legs(legs, slot, geometry, occupancy)
            if fallback is None:
                fallback = (found, entry_reverses)
    # Nothing kept off the lines. The search found SOMETHING, and stopping in
    # the aisle is worse than a manoeuvre that clips a line, so the best
    # available is driven -- but only after the clean answers were sought.
    if fallback is None:
        return None
    found, entry_reverses = fallback
    LAST_PLAN_NOTE = "search FALLBACK -- may clip a line"
    return _smooth_legs(
        _with_run_in(
            legs_from_path(found, slot), slot, geometry, entry_reverses
        ),
        slot,
        geometry,
        occupancy,
    )


def _with_run_in(
    legs: list[ParkingLeg],
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    entry_reverses: bool,
) -> list[ParkingLeg]:
    """
    Extend the entry leg from the set-back search goal to the real stop pose.

    Done in the BAY's frame, where the run-in is a straight in `along` at a
    fixed `across`, so it needs no knowledge of where the car currently is --
    the same property that lets a leg be committed at all. The leg's own
    endpoint is moved with it, or the executor would drive the extended path
    and then judge arrival against the old, shorter goal.

    **When the search arrives travelling the wrong way the run-in becomes its
    own LEG, and that is what actually delivers a reverse park.** Asked only
    for a pose deep in the bay facing out, the search almost never chose to
    reverse into it -- a Reeds-Shepp shot reaching that pose forwards is
    shorter and, with an empty occupancy, unobstructed -- so the reverse entry
    was proposed and rejected on every geometry tried. Splitting it is also
    what a driver does: get to the mouth however suits, stop, then back
    straight in. The cusp is the handbrake moment, not an inefficiency.
    """
    if not legs:
        return legs
    leg = legs[-1]
    if leg.path_bay is None or not len(leg.path_bay):
        return legs
    stop_along = slot.depth_m * 0.5 - PARKING_HEAD_CLEARANCE_M - (
        geometry.rear_m if entry_reverses else geometry.front_m
    )
    start = leg.path_bay[-1]
    end = np.asarray((stop_along, 0.0))
    span = float(np.linalg.norm(end - start))
    if span < 1e-3:
        return legs
    steps = np.linspace(
        0.0, 1.0, max(2, int(math.ceil(span / PARKING_PATH_STEP_M)) + 1)
    )[1:, None]
    run = start + (end - start) * steps
    if leg.reverse == entry_reverses:
        return legs[:-1] + [
            replace(
                leg,
                along_m=float(end[0]),
                across_m=float(end[1]),
                path_bay=np.concatenate((leg.path_bay, run)),
            )
        ]
    return legs + [
        ParkingLeg(
            along_m=float(end[0]),
            across_m=float(end[1]),
            heading_rad=math.pi if entry_reverses else 0.0,
            reverse=entry_reverses,
            path_bay=np.concatenate((start[None, :], run)),
        )
    ]


def _legs_respect_bay(
    legs: list[ParkingLeg],
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    tolerance: float = PARKING_BAY_KEEPOUT_M,
) -> bool:
    """
    Whether EVERY leg stays off the bay's side lines.

    Asking only the entry leg was tried and is not enough: once the run-in
    became a leg of its own, that final leg was a clean straight every time
    while the leg BEFORE it swung through the bay to line up -- measured, 1.36
    m over the line on every square-on geometry, with the guard reporting the
    manoeuvre clean. A positioning leg is still allowed to pass the mouth and
    sit alongside; that is what the lateral bound in `bay_intrusion` is for,
    and it is what makes asking every leg affordable rather than absurd.
    """
    if not legs:
        return False
    for leg in legs:
        path = _leg_path(leg, slot)
        if path is None or len(path) < 2:
            return False
        if bay_intrusion(path, leg.reverse, slot, geometry) > tolerance:
            return False
    return True


def plan_manoeuvre(
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    occupancy: Occupancy | None = None,
    start_gear: int = 1,
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
    # A SQUARE-ON bay is reversed into, and that decision comes before the
    # canned families rather than after them. The nose-in construction very
    # often "fits" such a bay -- it is pure geometry and, until `_clear`
    # learned to ask, nothing told it the arc crosses the neighbour -- so
    # trying it first meant the cheap wrong answer always won. See
    # `prefers_reverse`.
    if prefers_reverse(slot):
        reversed_in = _search_manoeuvre(
            slot, geometry, occupancy, start_gear=start_gear
        )
        if reversed_in is not None:
            return reversed_in

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
        global LAST_PLAN_NOTE
        LAST_PLAN_NOTE = "canned nose-in"
        return _smooth_legs(
            [_committed_leg(nose_in, nose_path, slot)],
            slot,
            geometry,
            occupancy,
        )

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
        LAST_PLAN_NOTE = "canned reverse-in"
        return _smooth_legs(
            [_committed_leg(back_in, back_path, slot)],
            slot,
            geometry,
            occupancy,
        )

    # Neither canned move fits, so SEARCH. Hybrid A* plans over the car's
    # own state and can shuffle as much as it needs, which is what covers the
    # bays the two families above refuse -- anything nearer than a turning
    # radius to the side, level with the car, or behind it. It is also the
    # only part of this that knows about obstacles.
    searched = _search_manoeuvre(
        slot, geometry, occupancy, start_gear=start_gear
    )
    if searched is not None:
        return searched

    # No canned fallback beyond this point, deliberately. A hand-built
    # "position then reverse" family used to sit here and it is strictly
    # worse than the search: it covers a subset of the same poses, and the
    # sequences it produced could leave the car at a pose its own single-arc
    # solver could not then drive out of -- stuck mid-manoeuvre with nowhere
    # to go. If the search finds nothing, nothing fits.
    return None


def _arc_move(
    curvature: float, length: float, reverse: bool
) -> tuple[np.ndarray, float]:
    """
    Body positions along one fixed-length constant-curvature move, from the
    origin heading +forward, plus the BODY heading where it ends.

    Front-frame curvature in, exactly what the steering commands: reversing
    with the wheels left swings the tail left and the nose right, which is the
    sign flip the closed form carries through `sign`.
    """
    steps = max(2, int(math.ceil(length / PARKING_PATH_STEP_M)) + 1)
    span = np.linspace(0.0, length, steps)
    sign = -1.0 if reverse else 1.0
    if abs(curvature) < 1e-9:
        points = np.column_stack((np.zeros(steps), sign * span))
        return points, 0.0
    # dh/ds_signed = -curvature (a direction is (sin h, cos h), positive
    # curvature is LEFT and decreases h), so h(s) = omega * s with
    # omega = -curvature * sign, and position integrates to the chord forms.
    omega = -curvature * sign
    points = np.column_stack(
        (
            sign * (1.0 - np.cos(omega * span)) / omega,
            sign * np.sin(omega * span) / omega,
        )
    )
    return points, float(omega * length)


def plan_nudge(
    slot: ParkingSlot,
    geometry: VehicleGeometry,
    final_leg: ParkingLeg,
    occupancy: Occupancy | None,
) -> list[ParkingLeg] | None:
    """
    The last-metre correction: pull out a couple of metres, re-enter exactly.

    A failed final-pose check used to hand the error to the SEARCH, whose
    0.7 m primitives and 0.5 m cells cannot express a half-metre correction
    -- measured live, 9.2 m of two-leg shuffle for a 0.5 m error. Inside
    `PARKING_NUDGE_MAX_M` / `PARKING_NUDGE_HEADING_DEG` the correction is
    closed-form instead: back (or pull) out along one arc, chosen so the
    canned entry solver reaches the stop pose from where it ends, and drive
    back in. The out leg is allowed to sweep the TARGET bay -- the car is
    standing in it -- but must not worsen the standing intrusion, and the
    re-entry must be clean; anything less falls back to the full replan.
    """
    target, axis = leg_pose(final_leg, slot)
    error = float(np.linalg.norm(target))
    heading_error = abs(math.atan2(float(axis[0]), float(axis[1])))
    if error > PARKING_NUDGE_MAX_M or heading_error > math.radians(
        PARKING_NUDGE_HEADING_DEG
    ):
        return None
    out_reverse = not final_leg.reverse
    out_length = min(
        PARKING_NUDGE_OUT_MAX_M, PARKING_NUDGE_OUT_MIN_M + 1.5 * error
    )
    bay_forward = bay_axis(slot)
    across_axis = np.asarray((bay_forward[1], -bay_forward[0]))
    centre = np.asarray((slot.centre_right_m, slot.centre_forward_m))
    standing = bay_intrusion(
        np.asarray(((0.0, 0.0), (0.0, 0.05))), False, slot, geometry
    )
    out_tolerance = max(PARKING_BAY_KEEPOUT_M, standing + 0.05)
    peak = _MAX_CURVATURE / PARKING_PLAN_RADIUS_MARGIN
    probe = replace(final_leg, path_bay=None)
    for out_curvature in (
        0.0,
        peak / 3.0,
        -peak / 3.0,
        2.0 * peak / 3.0,
        -2.0 * peak / 3.0,
        peak,
        -peak,
    ):
        out_points, out_heading = _arc_move(
            out_curvature, out_length, out_reverse
        )
        if bay_intrusion(
            out_points, out_reverse, slot, geometry
        ) > out_tolerance:
            continue
        if not _occupancy_clear(out_points, out_reverse, geometry, occupancy):
            continue
        end = out_points[-1]
        forward_end = np.asarray(
            (math.sin(out_heading), math.cos(out_heading))
        )
        right_end = np.asarray((forward_end[1], -forward_end[0]))
        local = _leg_path(probe, slot, origin=end, origin_axis=forward_end)
        if local is None or len(local) < 2:
            continue
        re_entry = (
            end
            + np.outer(local[:, 0], right_end)
            + np.outer(local[:, 1], forward_end)
        )
        # The re-entry must END ALIGNED, not merely end at the pose: the
        # canned solver's single-arc endgame reaches a nearby pose still
        # turned, which would re-create the exact skew the nudge exists to
        # remove and cycle the secure check for ever.
        tail = re_entry[-1] - re_entry[-2]
        tail_angle = math.atan2(float(tail[0]), float(tail[1]))
        if final_leg.reverse:
            tail_angle += math.pi
        axis_angle = math.atan2(float(axis[0]), float(axis[1]))
        skew = abs(
            (tail_angle - axis_angle + math.pi) % (2.0 * math.pi) - math.pi
        )
        if skew > math.radians(PARKING_LEG_SQUARE_DEG):
            continue
        if not _respects_bay(re_entry, final_leg.reverse, slot, geometry):
            continue
        if not _occupancy_clear(
            re_entry, final_leg.reverse, geometry, occupancy
        ):
            continue
        offset = end - centre
        out_relative = out_points - centre
        global LAST_PLAN_NOTE
        LAST_PLAN_NOTE = "nudge"
        out_leg = ParkingLeg(
            along_m=float(offset @ bay_forward),
            across_m=float(offset @ across_axis),
            heading_rad=out_heading - slot.heading_rad,
            reverse=out_reverse,
            path_bay=np.column_stack(
                (out_relative @ bay_forward, out_relative @ across_axis)
            ),
        )
        return [out_leg, _committed_leg(final_leg, re_entry, slot)]
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
        # How long the car has stood at the standoff with the blockage still
        # there, and whether the one permitted route-around replan for this
        # blockage episode has been spent. The old semantics only ever WAITED,
        # which is right for a crossing pedestrian and an infinite loop for a
        # parked car.
        self._blocked_elapsed = 0.0
        self._blocked_replan_tried = False
        self._replans = 0
        # Cusps have their OWN budget, now spent on LOCAL repairs -- a leg
        # re-derived from the actual pose when the car stopped meaningfully
        # off it -- never on re-searching the manoeuvre. Re-searching at every
        # cusp re-chose the topology every time (measured live: seven
        # different manoeuvres for one bay), which is the exact re-choice
        # problem this controller exists to avoid.
        self._cusp_replans = 0
        self._shift_dwell = 0.0
        self._gear_probe = 0
        self._progress_index = 0
        self._progress_elapsed = 0.0
        self._last_remaining = math.inf
        # One local repair ATTEMPT per leg entry, successful or not: the
        # canned solver legitimately has no aligned answer from some poses,
        # and re-asking it every tick is pure waste.
        self._repair_tried_leg = -1
        # How hard the path bends at the tracker's preview point, feeding the
        # curvature-scheduled speed cap.
        self._path_curvature = 0.0
        # The slewed speed target -- the ramp that makes cap handoffs smooth.
        self._speed_target = 0.0
        # The clamped steering-map trim, mirroring `controller._adapt_gain`:
        # MIN_TURN_RADIUS_M is a guess and the steering map is assumed linear,
        # and this is what absorbs both on the real car. `_cmd_filtered` is
        # the command lagged to match the measured yaw, so the trim compares
        # like with like.
        self._gain = 1.0
        self._cmd_filtered = 0.0
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

    @property
    def steering_gain(self) -> float:
        """The adapted steering-map trim, for the Steer check: line."""
        return self._gain

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
        measured_yaw_rate: float | None = None,
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
            self._legs = plan_manoeuvre(
                slot, geometry, occupancy, start_gear=self._start_gear()
            )
            self._leg_index = 0
            self._progress_index = 0
            if self._legs is None:
                # With an occupancy present the SEARCH failed, and blaming the
                # canned nose-in envelope would misdiagnose it -- the live log
                # said "the bay is level with or behind the car" after a
                # session of shuffling ended beside it. Only without one is
                # the envelope the honest explanation.
                return self._halt(
                    PARK_UNREACHABLE,
                    reachability(slot, geometry)
                    if occupancy is None
                    else (
                        "no clear route to the bay was found -- clear the "
                        "approach, or pick another bay"
                    ),
                    forward_gear,
                    speed_mps,
                    dt,
                )
        leg = self._legs[self._leg_index]
        gear = REVERSE_GEAR if leg.reverse else forward_gear

        if self._securing:
            return self._secure(
                slot, geometry, leg, occupancy, gear, speed_mps, dt
            )
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
            self._blocked_elapsed = 0.0
            return self._halt(
                PARK_BLOCKED,
                "Reverse emergency braking stopped the manoeuvre",
                gear,
                speed_mps,
                dt,
            )

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

        target, axis = leg_pose(leg, slot)
        final = self._leg_index + 1 >= len(self._legs)
        if self._reached(target, axis, leg.reverse, final):
            return self._advance(
                slot, geometry, occupancy, gear, speed_mps, dt
            )

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
                leg, forward_gear, reported_gear, speed_mps, dt, slot
            )

        path = _leg_path(leg, slot)
        if path is None:
            # Close to the leg's pose, "no path reaches it" means the car is
            # essentially ON it -- the same endgame the single forward move
            # hit, now once per leg: the tracker rolls a little past and no
            # forward construction reaches back. Treat it as made.
            if float(np.linalg.norm(target)) <= PARKING_LEG_CLOSE_M:
                return self._advance(
                    slot, geometry, occupancy, gear, speed_mps, dt
                )
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
        # Where the car IS on this path: the local perpendicular foot,
        # walked monotonically forward from the last known progress -- see
        # `match_index` for why it must never be a nearest-sample argmin.
        # Every committed leg starts at the pose it was planned from, so the
        # foot only ever advances by what the car drove this tick.
        here = match_index(sampled.points, self._progress_index)
        self._progress_index = here
        tracking_error = cross_track(sampled.points, here)
        blocked_at = blocking_distance(
            sampled,
            obstacles,
            geometry,
            reverse=leg.reverse,
            start_index=here,
        )
        remaining = sampled.length_m - float(sampled.cumulative_m[here])

        # Blockage comes BEFORE every tracking judgement: while the car is
        # stopped at (or braking for) an obstruction, cross-track and
        # progress are meaningless noise, and letting them trigger a replan
        # from inside the wait re-chose the manoeuvre for no reason. An
        # obstruction is a SPEED LIMIT first and a stop second: the car
        # brakes to a stop PARKING_BLOCK_STANDOFF_M short of it on the
        # normal profile, and only a blockage the car is already standing at
        # latches BLOCKED. Halting the moment anything appeared anywhere on
        # the remaining path -- measured live, a full stop for an
        # obstruction 12.9 m down the leg, then resume, then again at 10.7
        # and 5.0 -- was the reported brake-accelerate-brake cycling.
        stop_short = (
            math.inf
            if math.isinf(blocked_at)
            else blocked_at - PARKING_BLOCK_STANDOFF_M
        )
        if self._blocked:
            stopped = abs(speed_mps) <= PARKING_SHIFT_SPEED_MPS
            if stopped:
                self._blocked_elapsed += dt
            obstructed = rear_aeb_braking or stop_short <= max(
                0.3, PARKING_BLOCK_STANDOFF_M * 0.6
            )
            self._blocked_clear_dwell = (
                self._blocked_clear_dwell + dt
                if stopped and not obstructed
                else 0.0
            )
            if self._blocked_clear_dwell >= PARKING_BLOCKED_CLEAR_DWELL_S:
                self._blocked = False
                self._blocked_clear_dwell = 0.0
                self._blocked_elapsed = 0.0
                self._blocked_replan_tried = False
            else:
                # A blockage that persists with the car stopped is a parked
                # car, not a crossing pedestrian: try ONCE to route around it
                # -- the accumulated occupancy holds it by now -- and only
                # then settle in to wait. The old semantics only ever waited,
                # for ever.
                if (
                    occupancy is not None
                    and stopped
                    and obstructed
                    and not self._blocked_replan_tried
                    and self._blocked_elapsed >= PARKING_BLOCKED_REPLAN_S
                    and self._replans < PARKING_MAX_REPLANS
                ):
                    self._blocked_replan_tried = True
                    fresh = plan_manoeuvre(
                        slot,
                        geometry,
                        occupancy,
                        start_gear=self._start_gear(),
                    )
                    if fresh is not None:
                        self._replans += 1
                        self._adopt(fresh)
                        self._blocked = False
                        self._blocked_clear_dwell = 0.0
                        self._blocked_elapsed = 0.0
                        return self._halt(
                            PARK_SHIFTING,
                            "Replanning around an obstruction",
                            gear,
                            speed_mps,
                            dt,
                        )
                return self._halt(
                    PARK_BLOCKED,
                    "Waiting for the obstruction to clear",
                    gear,
                    speed_mps,
                    dt,
                    path=sampled,
                )
        elif stop_short <= 0.05:
            self._blocked = True
            self._blocked_clear_dwell = 0.0
            self._blocked_elapsed = 0.0
            return self._halt(
                PARK_BLOCKED,
                f"Something is in the way {blocked_at:.1f} m along the path",
                gear,
                speed_mps,
                dt,
                path=sampled,
            )

        if tracking_error > PARKING_DRIVE_MAX_CROSS_TRACK_M:
            return self._replan_or_fail(
                slot,
                geometry,
                occupancy,
                forward_gear,
                reported_gear,
                gear,
                speed_mps,
                dt,
                f"Tracking error grew to {tracking_error:.1f} m",
            )
        # A cusp is reached by TRACKING, so a new leg can start with the
        # heading error the previous leg ended with -- measured, 19 degrees,
        # which the tracker then fought at full lock. Near the leg start
        # that is a repair case exactly like a positional overshoot.
        travel_points = -sampled.points if leg.reverse else sampled.points
        seg = min(max(here, 0), len(travel_points) - 2)
        tangent = travel_points[seg + 1] - travel_points[seg]
        misaligned = here <= 6 and abs(
            math.atan2(float(tangent[0]), float(tangent[1]))
        ) > math.radians(PARKING_LEG_REPAIR_DEG)
        if (
            (tracking_error > PARKING_LEG_REPAIR_M or misaligned)
            and abs(speed_mps) <= PARKING_DRIVE_CREEP_MPS
            and self._cusp_replans < PARKING_MAX_CUSP_REPLANS
            and self._repair_tried_leg != self._leg_index
        ):
            self._repair_tried_leg = self._leg_index
            # A LOCAL repair: the car stopped meaningfully off this leg's
            # start (cusps overshoot by design -- the profile ends at the
            # actuation lag's worth of roll), so re-derive THIS leg's path
            # from the actual pose and keep the committed sequence. This is
            # what replaced re-searching the whole manoeuvre at every cusp.
            repaired = self._repair_leg(leg, slot, geometry, occupancy)
            if repaired is not None:
                self._cusp_replans += 1
                self._legs[self._leg_index] = repaired
                self._progress_index = 0
                self._last_remaining = math.inf
                return self._halt(
                    PARK_BACKING if leg.reverse else PARK_APPROACH,
                    "Re-joining the planned path",
                    gear,
                    speed_mps,
                    dt,
                    cross_track_m=tracking_error,
                )
        # Progress is judged RELATIVE TO SPEED, not against a fixed step. The
        # fixed 0.03 m-per-tick test was 0.75 m/s in disguise, and the speed
        # profile deliberately spends whole approach-tails below that (creep
        # 0.5, turn cap 0.8) -- so three seconds of intentional slow driving
        # read as "stopped making progress" and fired a full replan mid-park:
        # the live 18:10 session's silent 4.6x detour came from exactly this.
        # What the watchdog exists to catch -- moving without advancing along
        # the path (circling, diverging) -- still trips it: a car doing that
        # covers far less remaining-path than a quarter of its own travel.
        if remaining < self._last_remaining - max(
            0.005, 0.25 * abs(speed_mps) * dt
        ):
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
            return self._advance(
                slot, geometry, occupancy, gear, speed_mps, dt
            )

        drive_remaining = (
            remaining
            if math.isinf(stop_short)
            else min(remaining, max(stop_short, 0.0))
        )

        self._track(sampled, speed_mps, dt, leg.reverse, here)
        self._adapt(measured_yaw_rate, speed_mps, dt)
        heading_left = abs(math.atan2(float(axis[0]), float(axis[1])))
        target_speed = self._target_speed(
            drive_remaining, heading_left, self._path_curvature
        )
        # The caps hand off between themselves as the leg unwinds; slewing
        # their argmin is what turns the handoffs into one smooth profile
        # instead of a surge -- see PARKING_TARGET_RAMP_UP_MPS2.
        self._speed_target = _slew(
            self._speed_target,
            target_speed,
            PARKING_TARGET_RAMP_UP_MPS2,
            PARKING_TARGET_RAMP_DOWN_MPS2,
            dt,
        )
        target_speed = self._speed_target
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
                cross_track_m=tracking_error,
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
        occupancy: Occupancy | None,
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
                slot,
                geometry,
                self._legs[self._leg_index],
                occupancy,
                gear,
                speed,
                dt,
            )
        # The committed sequence SURVIVES the cusp. Re-searching here "to
        # reset the error" re-chose the manoeuvre topology from every
        # slightly different pose -- measured live, seven different plans for
        # one bay in two minutes, which was most of the shuffling. The next
        # leg is world-anchored and the tracker's job is exactly to absorb
        # the few centimetres a cusp overshoots by; a stop that landed
        # genuinely off the leg gets a LOCAL repair from the driving loop
        # (`_repair_leg`), never a fresh search.
        self._leg_index += 1
        self._progress_index = 0
        self._progress_elapsed = 0.0
        self._last_remaining = math.inf
        self._shifting = True
        return self._halt(PARK_SHIFTING, "Selecting gear", gear, speed, dt)

    def _secure(
        self,
        slot: ParkingSlot,
        geometry: VehicleGeometry,
        leg: ParkingLeg,
        occupancy: Occupancy | None,
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
            # measured stopped pose. A small miss gets the analytic NUDGE --
            # the search's 0.7 m primitives cannot express a half-metre
            # correction, and handing it one produced 9.2 m of shuffle for a
            # 0.5 m error, live, twice. Only an error beyond the nudge's
            # envelope re-runs the full planner.
            correction = plan_nudge(slot, geometry, leg, occupancy)
            if correction is None:
                correction = plan_manoeuvre(
                    slot, geometry, occupancy, start_gear=self._start_gear()
                )
            if correction is None:
                return self._halt(
                    PARK_UNREACHABLE,
                    "Could not settle fully inside the selected bay",
                    gear,
                    speed,
                    dt,
                )
            self._securing = False
            self._adopt(correction)
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
            plan_manoeuvre(
                slot, geometry, occupancy, start_gear=self._start_gear()
            )
            if self._replans <= PARKING_MAX_REPLANS
            else None
        )
        if replanned is None:
            return self._halt(
                PARK_UNREACHABLE, failure_reason, gear, speed, dt
            )
        self._adopt(replanned)
        # One tick that NAMES the trigger before the shift takes over. The
        # shift's own reason ("Selecting drive") used to overwrite it, so a
        # successful replan was invisible in the log -- the live 18:10
        # session's mid-park detour could only be attributed by elimination.
        self._shifting = True
        return self._halt(
            PARK_SHIFTING,
            f"Replanned -- {failure_reason}",
            gear,
            speed,
            dt,
        )

    def _adopt(self, legs: list[ParkingLeg]) -> None:
        """Commit a fresh sequence and zero every per-leg tracker."""
        self._legs = legs
        self._leg_index = 0
        self._progress_index = 0
        self._progress_elapsed = 0.0
        self._last_remaining = math.inf
        self._repair_tried_leg = -1
        self._shifting = False

    def _start_gear(self) -> int:
        """The direction the box is committed to, as the search's start gear.

        Without this every replan was costed as if the car were in DRIVE, so
        a plan begun at a reverse cusp was charged a phantom gear change for
        continuing the direction it was already in -- a bias toward forward
        continuations exactly when the manoeuvre is mid-reverse.
        """
        return 1 if self._gear >= 0 else -1

    def _repair_leg(
        self,
        leg: ParkingLeg,
        slot: ParkingSlot,
        geometry: VehicleGeometry,
        occupancy: Occupancy | None,
    ) -> ParkingLeg | None:
        """This leg re-derived from the ACTUAL pose, or None to leave it be.

        The canned solver reaches the leg's committed end pose from wherever
        the car really stopped; the sequence, the goal and every later leg
        are untouched. None simply means the car keeps tracking the stored
        path -- and if the error keeps growing, the gross threshold hands the
        case to the bounded full replan as before.
        """
        probe = replace(leg, path_bay=None)
        path = _leg_path(probe, slot)
        if path is None or len(path) < 2:
            return None
        # The canned solver may close its residual with a long shallow tail
        # arc that reaches the POSE without reaching its HEADING -- driven
        # from mid-manoeuvre it once ended a reverse entry 16 degrees skewed
        # with the tracker following it perfectly. A repair that does not end
        # aligned is worse than no repair.
        _, axis = leg_pose(leg, slot)
        tail = path[-1] - path[-2]
        tail_angle = math.atan2(float(tail[0]), float(tail[1]))
        axis_angle = math.atan2(float(axis[0]), float(axis[1]))
        if leg.reverse:
            tail_angle += math.pi
        skew = abs(
            (tail_angle - axis_angle + math.pi) % (2.0 * math.pi) - math.pi
        )
        if skew > math.radians(PARKING_LEG_SQUARE_DEG):
            return None
        candidate = _smooth_legs(
            [_committed_leg(leg, path, slot)], slot, geometry, occupancy
        )[0]
        repaired = _leg_path(candidate, slot)
        if repaired is None:
            return None
        if not _respects_bay(repaired, leg.reverse, slot, geometry):
            return None
        if not _occupancy_clear(repaired, leg.reverse, geometry, occupancy):
            return None
        return candidate

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
        slot: ParkingSlot | None = None,
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
        self._speed_target = 0.0
        # PRE-AIM the wheels while the box is being confirmed: the dwell is
        # dead time and the steering can wind during it (the wheels turn at a
        # standstill), so the new leg starts with its entry curvature already
        # on instead of spending its first metre winding -- which is where
        # cusp cross-track came from. While still rolling to the stop the
        # wheel returns to centre, so the last of the roll stays straight.
        stopped_for_aim = abs(speed) <= PARKING_SHIFT_SPEED_MPS
        aim = (
            self._entry_curvature(leg, slot)
            if stopped_for_aim and slot is not None
            else 0.0
        )
        rate = PARKING_STEER_RATE_PER_S * dt
        self._curvature = max(
            self._curvature - rate, min(self._curvature + rate, aim)
        )
        self._brake = _slew(self._brake, PARKING_STOP_BRAKE, 6.0, 4.0, dt)
        reason = "Selecting reverse" if leg.reverse else "Selecting drive"
        return (
            ControlCommand(
                steering=self._steering(),
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

    def _track(
        self,
        path: ParkingPath,
        speed: float,
        dt: float,
        reverse: bool,
        here: int = 0,
    ) -> float:
        """
        Error-state tracking, mirrored once when the car is going backwards.

        Everything is computed in the TRAVEL frame -- a reverse leg's samples
        lie behind the car, so the tracker runs on the negated path, the
        frame the car is actually travelling in -- and the command comes back
        negated. That is the same relation the steered reverse uses:
        travel-frame k_m equals minus the front-frame k_f, because reversing
        yaw is `v_signed * k_f` with the speed negative. Negating here,
        before STEERING_SIGN, keeps the one-place-reconciles rule intact.

        There is deliberately NO lookahead point. Pure pursuit's lookahead
        chord-cuts every committed bend and runs off the path end exactly
        where parking needs its precision; the matched-point errors do
        neither. The feed-forward is read a little AHEAD instead
        (`PARKING_TRACK_PREVIEW_S`) -- that is actuator-lag compensation, a
        different thing from a lookahead: the wheel takes real time to wind,
        so the command must lead the geometry by roughly the winding time.
        """
        points = -path.points if reverse else path.points
        if len(points) < 2:
            return self._curvature
        # Errors are measured at the REAR AXLE, not at the node the path was
        # planned for. The axle is the bicycle's no-slip pivot, minimum-phase
        # in both directions; the node ahead of it initially swings the WRONG
        # way in reverse, and controlling it there put the tracker in a
        # saturated full-lock-to-full-lock limit cycle. The axle sits at
        # body (0, -offset), which the reverse negation carries to the travel
        # frame like every other point.
        axle = np.asarray(
            (
                0.0,
                PARKING_REAR_AXLE_OFFSET_M
                if reverse
                else -PARKING_REAR_AXLE_OFFSET_M,
            )
        )
        shifted = points - axle
        lateral_error, segment = _signed_offset(shifted, here)
        tangent = points[segment + 1] - points[segment]
        heading_error = math.atan2(float(tangent[0]), float(tangent[1]))
        preview = max(abs(speed), 0.4) * PARKING_TRACK_PREVIEW_S
        ahead = int(
            np.searchsorted(
                path.cumulative_m,
                float(path.cumulative_m[here]) + preview,
            )
        )
        # Signed curvature is rotation-invariant, so the curvature of the
        # negated points IS the travel-frame curvature -- no extra sign.
        # Clamped to len-2: the discrete curvature's endpoints are zero by
        # construction, so an unclamped preview fades the feed-forward out
        # over the last half-metre of every leg -- which is exactly where a
        # leg that ends AT a cusp is still turning, and how each cusp
        # inherited a few degrees of heading error.
        feed_forward = float(
            _curvature(points)[max(0, min(ahead, len(points) - 2))]
        )
        # Lateral error becomes a bounded APPROACH ANGLE and the heading loop
        # tracks tangent-plus-approach -- see PARKING_TRACK_APPROACH_GAIN for
        # why the structure (damping surviving saturation) is the point.
        # Path to the travel-left (positive error) means approach LEFT of the
        # tangent, which in the toward-right heading convention is negative.
        approach_cap = math.radians(PARKING_TRACK_APPROACH_MAX_DEG)
        approach = -math.atan(PARKING_TRACK_APPROACH_GAIN * lateral_error)
        approach = max(-approach_cap, min(approach_cap, approach))
        feedback = -PARKING_TRACK_HEADING_GAIN * (heading_error + approach)
        travel = PARKING_TRACK_FEEDFORWARD_GAIN * feed_forward + feedback
        # For the curvature-scheduled speed cap: how hard the path bends here.
        self._path_curvature = abs(feed_forward)
        target = -travel if reverse else travel
        target = max(-_MAX_CURVATURE, min(_MAX_CURVATURE, target))
        rate = PARKING_STEER_RATE_PER_S * dt
        self._curvature = max(
            self._curvature - rate,
            min(self._curvature + rate, target),
        )
        return self._curvature

    def _adapt(
        self,
        measured_yaw_rate: float | None,
        forward_speed: float,
        dt: float,
    ) -> None:
        """
        Trim the steering-map gain from what the car measurably drives.

        `controller._adapt_gain` for the parking regime: MIN_TURN_RADIUS_M is
        a guess and the steering map is assumed linear, and neither error is
        visible offline because the test plant IS the assumption. Only above
        walking pace at meaningful curvature, and only when the car turns the
        way it was asked -- a sign mismatch is a slide or a kerb, not data.
        Yaw over SIGNED speed is the front-frame curvature in both directions
        of travel, so reversing needs no special case.
        """
        # The command is FILTERED with the same time constant as the measured
        # yaw before any comparison, and adaptation runs only while the two
        # could possibly describe the same arc. Comparing the instant command
        # against yaw that lags half a second of actuator and filter behind
        # it made every transient a fake measurement -- the live 18:10
        # session logged ratios from -3.68 to +1.92 and the gain wandered
        # 0.66-1.26, injecting its own steering noise -- while the sustained
        # arcs in the same log read 1.02-1.11. Those are the samples this
        # gate keeps.
        self._cmd_filtered = (
            0.35 * self._curvature + 0.65 * self._cmd_filtered
        )
        if measured_yaw_rate is None:
            return
        if abs(forward_speed) < PARKING_STEER_GAIN_MIN_SPEED_MPS:
            return
        if abs(self._cmd_filtered) < PARKING_STEER_GAIN_MIN_CURVATURE:
            return
        if abs(self._curvature - self._cmd_filtered) > 0.02:
            return
        measured = measured_yaw_rate / forward_speed
        if measured * self._cmd_filtered <= 0.0:
            return
        ratio = min(3.0, measured / self._cmd_filtered)
        self._gain = max(
            PARKING_STEER_GAIN_MIN,
            min(
                PARKING_STEER_GAIN_MAX,
                self._gain
                + PARKING_STEER_GAIN_ADAPT_RATE * (1.0 - ratio) * dt,
            ),
        )

    def _entry_curvature(
        self, leg: ParkingLeg, slot: ParkingSlot
    ) -> float:
        """The front-frame curvature this leg begins with, for pre-aiming."""
        path = _leg_path(leg, slot)
        if path is None or len(path) < 3:
            return 0.0
        points = -path if leg.reverse else path
        index = min(match_index(points, 0) + 2, len(points) - 1)
        travel = float(_curvature(points)[index])
        front = -travel if leg.reverse else travel
        return max(-_MAX_CURVATURE, min(_MAX_CURVATURE, front))

    @staticmethod
    def _target_speed(
        remaining: float,
        turn_remaining: float,
        path_curvature: float = 0.0,
    ) -> float:
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

        It applies only NEAR THE END, which is the half that was missing. The
        heading it reads is the error against the leg's final pose, so entering
        a square-on bay there is about 90 degrees to lose from the moment the
        manoeuvre begins -- the cap was satisfied continuously and the car
        crept from the aisle to the bay at 0.8 m/s. Before the last few metres
        the heading is SUPPOSED to be changing.
        """
        # The lag-aware stopping profile: the speed that solves
        # v^2 = 2a(remaining - v*lag), i.e. the bare sqrt law with the
        # distance the car rolls during PARKING_CONTROL_LAG_S already spent.
        # The bare law only began braking inside the last metre and demanded
        # the nominal deceleration with zero latency, so every leg ended
        # still rolling -- measured live, SHIFTING entered at 1.19 m/s -- and
        # each overshot cusp handed the replanner a new pose.
        decel = PARKING_DRIVE_DECEL_MPS2
        lag = PARKING_CONTROL_LAG_S
        stopping = (
            decel
            * lag
            * (
                math.sqrt(
                    1.0 + 2.0 * max(remaining, 0.0) / (decel * lag * lag)
                )
                - 1.0
            )
        )
        if (
            remaining <= PARKING_TURN_SLOW_RANGE_M
            and turn_remaining > math.radians(PARKING_TURN_SLOW_DEG)
        ):
            stopping = min(stopping, PARKING_DRIVE_CREEP_MPS * 1.6)
        # Slow through the tight part of a bend, anywhere along the leg: the
        # tracker's lag-induced deviation scales with speed, and the tight
        # half of a reverse swing at full manoeuvring speed is where the
        # body strayed past a bay side line.
        if path_curvature > PARKING_TURN_SPEED_CURVATURE:
            stopping = min(stopping, PARKING_TURN_SPEED_MPS)
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
            -1.0,
            min(
                1.0,
                STEERING_SIGN
                * self._gain
                * self._curvature
                / _MAX_CURVATURE,
            ),
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
        cross_track_m: float = 0.0,
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
        self._speed_target = 0.0
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
                cross_track_m=cross_track_m,
                curvature=0.0,
                target_speed_mps=0.0,
                reason=reason,
                path=path,
            ),
        )


def _slew(previous: float, target: float, up: float, down: float, dt: float) -> float:
    limit = up * dt if target > previous else down * dt
    return previous + max(-limit, min(limit, target - previous))
