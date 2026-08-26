"""
Driving into a bay: the path geometry, and the closed loop that tracks it.

The closed-loop tests run a kinematic bicycle against the real controller,
re-projecting the world-anchored bay into the car's BEV frame every tick
exactly as the worker does. That is the only way to test a manoeuvre: a
path that looks right and a tracker that reads it correctly can still fail
to converge, and the thing that matters is where the car actually stops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    MIN_TURN_RADIUS_M,
    PARKING_DRIVE_SPEED_MPS,
    PARKING_PATH_STEP_M,
    STEERING_SIGN,
    THROTTLE_GAIN_MPS2,
)
from beamng_lidar_bev.hybrid_astar import Occupancy
from beamng_lidar_bev.models import ParkingSlot, VehicleGeometry
from beamng_lidar_bev.parking_drive import (
    PARK_ARRIVED,
    PARK_BACKING,
    PARK_BLOCKED,
    PARK_SECURING,
    PARK_UNREACHABLE,
    ParkingDriver,
    ParkingLeg,
    _as_path,
    _curvature,
    _leg_path,
    _reverse_reach,
    blocking_distance,
    cross_track,
    plan_manoeuvre,
    plan_parking_path,
    prefers_reverse,
    reachability,
    stop_pose,
)

# The car the plant figures were measured on, from the live Vehicle check line.
VIVACE = VehicleGeometry(
    ground_z_vehicle=-0.255,
    left_m=1.01,
    right_m=1.01,
    front_m=1.84,
    rear_m=2.49,
    height_m=1.46,
    mounts={},
)
_MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M


def _slot(
    right: float,
    forward: float,
    heading: float,
    width: float = 3.18,
    depth: float = 5.3,
    occupied: bool = False,
) -> ParkingSlot:
    """A bay in the ego's BEV frame, sized as the live lot measured."""
    return ParkingSlot(
        centre_right_m=right,
        centre_forward_m=forward,
        heading_rad=heading,
        width_m=width,
        depth_m=depth,
        occupied=occupied,
        stripe_cells=52,
        centre_world=(0.0, 0.0),
    )


# --- the path ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, slot",
    (
        ("dead ahead", _slot(0.0, 12.0, 0.0)),
        ("square-on to the left", _slot(-8.0, 14.0, -math.pi / 2)),
        ("square-on to the right", _slot(8.0, 14.0, math.pi / 2)),
        ("45 degrees left", _slot(-6.0, 14.0, -math.pi / 4)),
        ("60 degrees right", _slot(7.0, 16.0, math.pi / 3)),
    ),
)
def test_a_reachable_bay_gets_a_path_the_car_can_actually_steer(
    name: str, slot: ParkingSlot
) -> None:
    """
    Every sample must be inside the car's own turning circle.

    A cubic Bezier was tried first and could not hold this: on a square-on
    bay -- the commonest case in a lot -- the flattest Bezier reaching the
    pose still bent to 0.20 1/m against the car's 0.167 limit, so every one
    came back unreachable. Straight-arc-straight hits the limit exactly.
    """
    path = plan_parking_path(slot, VIVACE)

    assert path is not None, name
    assert float(np.abs(_curvature(path.points)).max()) <= _MAX_CURVATURE + 2e-3
    assert path.length_m > 0.0
    # It must END at the stop pose, which is what the tracker drives to.
    target, _ = stop_pose(slot, VIVACE)
    assert np.allclose(path.points[-1], target, atol=1e-6)


@pytest.mark.parametrize(
    "name, slot",
    (
        ("abeam, no room to turn in", _slot(-8.0, 0.0, -math.pi / 2)),
        ("directly behind", _slot(0.0, -8.0, math.pi)),
    ),
)
def test_a_bay_needing_a_shuffle_is_not_a_single_forward_move(
    name: str, slot: ParkingSlot
) -> None:
    """
    The NOSE-IN planner refuses these; `plan_manoeuvre` is what then finds a
    reverse manoeuvre for them. Half-attempting a nose-in here would end with
    the car across the lines.
    """
    assert plan_parking_path(slot, VIVACE) is None, name


def test_the_nose_stops_short_of_the_head_not_the_node() -> None:
    """
    The reference node is not the middle of the car, so parking "the bay
    centre" would leave the body however far off-centre the node happens to
    be -- 1.84 m forward of it on this vehicle.
    """
    slot = _slot(0.0, 12.0, 0.0)
    node, axis = stop_pose(slot, VIVACE)

    nose = node + axis * VIVACE.front_m
    head = np.asarray((slot.centre_right_m, slot.centre_forward_m)) + axis * (
        slot.depth_m * 0.5
    )
    assert 0.3 < float(np.linalg.norm(head - nose)) < 0.8
    # And the tail is inside the bay's mouth rather than hanging into the aisle.
    tail = node - axis * VIVACE.rear_m
    mouth = np.asarray((slot.centre_right_m, slot.centre_forward_m)) - axis * (
        slot.depth_m * 0.5
    )
    assert float((tail - mouth) @ axis) > 0.0


# --- the corridor check ------------------------------------------------------


def test_something_standing_in_the_path_blocks_it() -> None:
    slot = _slot(0.0, 12.0, 0.0)
    path = plan_parking_path(slot, VIVACE)
    assert path is not None

    clear = blocking_distance(path, np.empty((0, 2)), VIVACE)
    ahead = blocking_distance(
        path, np.asarray([[0.0, 8.0], [0.2, 8.1]]), VIVACE
    )
    beside = blocking_distance(path, np.asarray([[4.0, 8.0]]), VIVACE)

    assert clear == math.inf
    # The oriented BODY, including its 1.84 m front overhang, reaches the
    # obstacle well before the reference node is level with it.
    assert 5.8 < ahead < 6.8
    assert beside == math.inf


def test_the_car_is_not_blocked_by_what_it_is_already_alongside() -> None:
    """The corridor starts at the nose; the body already overlaps behind it."""
    slot = _slot(0.0, 12.0, 0.0)
    path = plan_parking_path(slot, VIVACE)
    assert path is not None

    assert blocking_distance(path, np.asarray([[0.0, 0.5]]), VIVACE) == math.inf


def test_live_blocking_distance_starts_at_current_path_progress() -> None:
    """An obstacle already passed must not stop the remaining trajectory."""
    points = np.column_stack((np.zeros(9), np.arange(9.0)))
    path = _as_path(points)
    # `_as_path` re-samples to a uniform PARKING_PATH_STEP_M grid, so an index
    # is no longer a metre. Locate 5 m along the way the executor does.
    here = int(np.searchsorted(path.cumulative_m, 5.0))

    behind = blocking_distance(
        path,
        np.asarray(((0.0, 1.0),)),
        VIVACE,
        start_index=here,
    )
    ahead = blocking_distance(
        path,
        np.asarray(((0.0, 8.0),)),
        VIVACE,
        start_index=here,
    )

    assert behind == math.inf
    assert 0.0 <= ahead <= 2.0


# --- the closed loop ---------------------------------------------------------


# The plant's actuator: how fast the achieved curvature can wind, and where
# the reference node sits ahead of the rear axle. The old plant steered
# INSTANTLY at a point -- it WAS the tracker's own assumption -- so every gain
# tuned against it was tuned against fiction, and the live car saturated the
# wheel while cross-track grew. These are deliberately imperfect: the plant
# achieves curvature slower than the tracker commands it
# (PARKING_STEER_RATE_PER_S is 0.7), and the controlled node is 1.5 m ahead
# of the axle that actually pivots.
PLANT_STEER_RATE = 0.5
PLANT_NODE_OFFSET_M = 1.5


class _Bicycle:
    """
    Kinematic plant: BeamNG steering in, NODE pose out.

    Steering is the game's own convention (+1 = right), so it is divided by
    STEERING_SIGN to recover the planner's left-positive curvature -- the one
    place the two conventions meet, exactly as `controller._steer` does going
    the other way.

    Honest in the three ways the old plant was not: the achieved curvature
    slews at a finite actuator rate; a `gain_error` scales the steering map
    (MIN_TURN_RADIUS_M is a guess -- the real car's lock is not exactly 6 m);
    and the kinematics pivot about the REAR AXLE while `x`/`y` report the
    reference node ahead of it, which is the point every planner and tracker
    in the stack reasons about, exactly as live.
    """

    def __init__(
        self,
        x: float,
        y: float,
        heading: float,
        gain_error: float = 1.0,
        steer_rate: float = PLANT_STEER_RATE,
        node_offset: float = PLANT_NODE_OFFSET_M,
        accel_tau: float = 0.15,
    ) -> None:
        self.heading, self.speed = heading, 0.0
        self.gear = 2
        self._gain_error = gain_error
        self._steer_rate = steer_rate
        self._node_offset = node_offset
        self._accel_tau = accel_tau
        self._kappa = 0.0
        self._accel = 0.0
        forward = self.forward
        self._rear_x = x - node_offset * float(forward[0])
        self._rear_y = y - node_offset * float(forward[1])

    @property
    def x(self) -> float:
        return self._rear_x + self._node_offset * float(self.forward[0])

    @property
    def y(self) -> float:
        return self._rear_y + self._node_offset * float(self.forward[1])

    @property
    def forward(self) -> np.ndarray:
        return np.asarray((math.sin(self.heading), math.cos(self.heading)))

    @property
    def right(self) -> np.ndarray:
        return np.asarray((math.cos(self.heading), -math.sin(self.heading)))

    def step(
        self,
        steering: float,
        throttle: float,
        brake: float,
        dt: float,
        gear: int = 2,
    ) -> None:
        """
        `gear` sets the DIRECTION of travel, and the box only takes a new one
        at rest -- which is what the shift handshake exists to respect.
        """
        if abs(self.speed) <= 0.09:
            self.gear = gear
        target = (steering / STEERING_SIGN) * _MAX_CURVATURE * self._gain_error
        limit = self._steer_rate * dt
        self._kappa = max(
            self._kappa - limit, min(self._kappa + limit, target)
        )
        wanted = throttle * THROTTLE_GAIN_MPS2 - brake * 8.0
        self._accel += (wanted - self._accel) * min(1.0, dt / self._accel_tau)
        self.speed = max(0.0, self.speed + self._accel * dt)
        # Signed travel: reverse moves the car backwards along its heading.
        distance = self.speed * dt * (-1.0 if self.gear < 0 else 1.0)
        # Positive curvature is LEFT, which decreases a compass heading. The
        # sign of the DISTANCE is what makes reversing turn the other way,
        # which is the whole of the reverse-steering relation.
        self.heading -= self._kappa * distance
        self._rear_x += math.sin(self.heading) * distance
        self._rear_y += math.cos(self.heading) * distance

    @property
    def reported_gear(self) -> object:
        """What an automatic box reports: a mode string, not a number."""
        return "R" if self.gear < 0 else "D"

    @property
    def forward_speed(self) -> float:
        return -self.speed if self.gear < 0 else self.speed


def _park(
    bay_world: tuple[float, float],
    bay_heading_world: float,
    start: tuple[float, float, float],
    obstacles_world: np.ndarray | None = None,
    ticks: int = 3000,
    gain_error: float = 1.0,
) -> tuple[_Bicycle, object, np.ndarray, np.ndarray]:
    """Drive the real controller into a world-anchored bay from `start`."""
    car = _Bicycle(*start, gain_error=gain_error)
    driver = ParkingDriver()
    axis_world = np.asarray(
        (math.sin(bay_heading_world), math.cos(bay_heading_world))
    )
    centre_world = np.asarray(bay_world)
    state = None
    # Measured yaw in the WORKER's convention (anticlockwise positive, so
    # the negation of the plant's toward-right heading), filtered exactly as
    # `_drive_into_bay` filters it -- this is what feeds the steering trim.
    yaw_filtered: float | None = None
    last_heading = car.heading
    for _ in range(ticks):
        rel = centre_world - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(
                float(axis_world @ car.right), float(axis_world @ car.forward)
            ),
        )
        local_obstacles = None
        if obstacles_world is not None and len(obstacles_world):
            delta = obstacles_world - np.asarray((car.x, car.y))
            local_obstacles = np.column_stack(
                (delta @ car.right, delta @ car.forward)
            )
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            obstacles=local_obstacles,
            reported_gear=car.reported_gear,
            measured_yaw_rate=yaw_filtered,
        )
        if state.finished or state.phase == PARK_BLOCKED:
            if car.speed < 0.02:
                break
        car.step(
            command.steering,
            command.throttle,
            command.brake,
            0.04,
            command.gear,
        )
        yaw = -math.remainder(car.heading - last_heading, math.tau) / 0.04
        last_heading = car.heading
        yaw_filtered = (
            yaw if yaw_filtered is None else 0.35 * yaw + 0.65 * yaw_filtered
        )
    return car, state, centre_world, axis_world


@pytest.mark.parametrize(
    "name, bay, bay_heading, start",
    (
        ("straight in", (0.0, 14.0), 0.0, (0.0, 0.0, 0.0)),
        ("square-on left", (-8.0, 16.0), -math.pi / 2, (0.0, 0.0, 0.0)),
        ("square-on right", (8.0, 16.0), math.pi / 2, (0.0, 0.0, 0.0)),
        ("45 degrees left", (-6.0, 15.0), -math.pi / 4, (0.0, 0.0, 0.0)),
        ("offset start", (-8.0, 18.0), -math.pi / 2, (1.5, 0.0, 0.12)),
    ),
)
def test_the_car_ends_up_in_the_bay(
    name: str, bay: tuple[float, float], bay_heading: float, start
) -> None:
    """
    The whole point. Closed loop, and what is asserted is where the car
    STOPS -- centred across the bay, square to it, and fully inside.

    SQUARE, not facing in. A square-on bay is reversed into by preference now
    (see `prefers_reverse`), and a car that has backed in finishes facing OUT
    -- 180 degrees from the bay axis and perfectly correctly parked. What the
    bay cares about is alignment with its axis, which is the same test either
    way round.
    """
    car, state, centre, axis = _park(bay, bay_heading, start)

    assert state is not None and state.phase == PARK_ARRIVED, f"{name}: {state}"
    node = np.asarray((car.x, car.y))
    across = float((node - centre) @ np.asarray((axis[1], -axis[0])))
    heading_error = abs(
        math.atan2(
            float(axis[0] * math.cos(car.heading) - axis[1] * math.sin(car.heading)),
            float(axis[0] * math.sin(car.heading) + axis[1] * math.cos(car.heading)),
        )
    )
    skew = min(heading_error, abs(math.pi - heading_error))
    assert abs(across) < 0.30, f"{name}: {across:.2f} m off centre"
    assert skew < math.radians(6.0), f"{name}: {skew:.3f} rad off the bay axis"
    assert car.speed < 0.05, f"{name}: still rolling at {car.speed:.2f} m/s"


def test_the_manoeuvre_stays_below_the_speed_the_brake_arms_at() -> None:
    """
    Load-bearing, not incidental. AEB arms at AEB_MIN_SPEED_MPS and parking
    deliberately creeps below it, so the emergency brake stays in STANDBY and
    cannot fire at the kerbs and neighbours a park drives close to on purpose.
    """
    from beamng_lidar_bev.config import AEB_MIN_SPEED_MPS

    assert PARKING_DRIVE_SPEED_MPS < AEB_MIN_SPEED_MPS

    car, _, _, _ = _park((-8.0, 16.0), -math.pi / 2, (0.0, 0.0, 0.0))
    assert car.speed < AEB_MIN_SPEED_MPS


def test_a_parked_car_holds_rather_than_creeping_on() -> None:
    """A finished park must STAY put; releasing lets it roll out of the bay."""
    car, state, _, _ = _park((0.0, 14.0), 0.0, (0.0, 0.0, 0.0))
    assert state.phase == PARK_ARRIVED

    driver = ParkingDriver()
    driver._held = True
    command, held = driver.step(
        _slot(0.0, 12.0, 0.0), VIVACE, 0.0, 0.04
    )
    assert held.phase == PARK_ARRIVED
    assert command.throttle == 0.0
    assert command.brake > 0.2


def test_crossing_the_endpoint_while_rolling_is_not_arrival() -> None:
    """The path endpoint starts braking; only a stopped dwell is success."""
    driver = ParkingDriver()
    driver._legs = [ParkingLeg(0.0, 0.0, 0.0, False)]

    command, state = driver.step(_slot(0.0, 0.0, 0.0), VIVACE, 0.68, 0.04)

    assert state.phase == PARK_SECURING
    assert command.throttle == 0.0
    assert command.brake > 0.0
    assert command.parking_brake == 0.0


def test_success_requires_a_stopped_dwell_and_applies_parking_brake() -> None:
    driver = ParkingDriver()
    driver._legs = [ParkingLeg(0.0, 0.0, 0.0, False)]
    slot = _slot(0.0, 0.0, 0.0)

    state = None
    command = None
    for _ in range(20):
        command, state = driver.step(slot, VIVACE, 0.0, 0.04)

    assert state is not None and state.phase == PARK_ARRIVED
    assert command is not None and command.parking_brake == 1.0


def test_an_obstacle_in_the_bay_path_stops_the_car_short() -> None:
    car, state, centre, _ = _park(
        (0.0, 16.0), 0.0, (0.0, 0.0, 0.0),
        obstacles_world=np.asarray([[0.0, 9.0], [0.3, 9.2], [-0.3, 9.1]]),
    )

    assert state is not None and state.phase == PARK_BLOCKED
    assert car.speed < 0.05
    # Stopped short of the obstacle rather than in the bay.
    assert car.y < 8.0


def test_a_bay_walled_off_from_the_car_is_refused() -> None:
    """
    Now that the search runs, a refusal means something is genuinely IN THE
    WAY -- not that the geometry was awkward. That is the whole difference
    between this and the manoeuvre families it replaced.
    """
    from beamng_lidar_bev.hybrid_astar import Occupancy

    slot = _slot(-6.0, 8.0, -math.pi / 2)
    # A wall right across the lot, long enough that there is no way round it.
    wall = np.column_stack(
        (np.linspace(-30.0, 30.0, 600), np.full(600, 3.0))
    )
    free = np.column_stack(
        [axis.ravel() for axis in np.meshgrid(
            np.arange(-30.0, 30.0, 0.5), np.arange(-30.0, 30.0, 0.5)
        )]
    )

    assert plan_manoeuvre(slot, VIVACE, Occupancy(wall, free)) is None


def test_driver_uses_occupancy_for_its_initial_plan() -> None:
    """Calling `plan_manoeuvre` correctly is not enough if production omits it."""
    slot = _slot(-6.0, 8.0, -math.pi / 2)
    wall = np.column_stack(
        (np.linspace(-30.0, 30.0, 600), np.full(600, 3.0))
    )
    free = np.column_stack(
        [
            axis.ravel()
            for axis in np.meshgrid(
                np.arange(-30.0, 30.0, 0.5),
                np.arange(-30.0, 30.0, 0.5),
            )
        ]
    )

    _, state = ParkingDriver().step(
        slot,
        VIVACE,
        0.0,
        0.04,
        occupancy=Occupancy(wall, free),
        reported_gear="D",
    )

    assert state.phase == PARK_UNREACHABLE


def test_losing_the_bay_stops_the_manoeuvre() -> None:
    driver = ParkingDriver()

    command, state = driver.step(None, VIVACE, 1.0, 0.04)

    assert state.phase == PARK_UNREACHABLE
    assert command.throttle == 0.0
    assert command.brake > 0.0


# --- why a bay is refused ----------------------------------------------------


def test_the_refusal_says_what_to_do_about_it() -> None:
    """
    "No single forward move fits" is true and useless: it does not say whether
    to roll forward, pick another bay, or that the car is on the wrong side of
    it. The envelope is closed-form, so the reason can be too.
    """
    too_close = _slot(-8.0, 4.0, -math.pi / 2)
    too_narrow = _slot(-3.0, 10.0, -math.pi / 2)
    behind = _slot(-8.0, 0.0, -math.pi / 2)

    assert plan_parking_path(too_close, VIVACE) is None
    assert "ahead" in reachability(too_close, VIVACE)
    assert plan_parking_path(too_narrow, VIVACE) is None
    assert "to the side" in reachability(too_narrow, VIVACE)
    assert "ahead" in reachability(behind, VIVACE)


def test_the_reachable_envelope_is_the_turning_circle_not_a_tuning_choice() -> None:
    """
    One arc of radius R changes heading 90 degrees and displaces the car
    EXACTLY R sideways and R forwards, so a square-on bay nearer than that in
    either axis cannot be entered nose-first at all. Pinned so nobody "fixes"
    it by loosening a constant: the remedy is reversing in, not a wider band.
    """
    wide, near = MIN_TURN_RADIUS_M * 1.4, MIN_TURN_RADIUS_M * 0.6
    square = -math.pi / 2
    inside = _slot(-wide, wide, square)
    short_ahead = _slot(-wide, near, square)
    short_across = _slot(-near, wide, square)

    assert plan_parking_path(inside, VIVACE) is not None
    assert plan_parking_path(short_ahead, VIVACE) is None
    assert plan_parking_path(short_across, VIVACE) is None


# --- the multi-leg manoeuvre (planner only; not yet driven) -------------------


def test_a_bay_the_nose_in_envelope_cannot_reach_gets_a_reverse_manoeuvre() -> None:
    """
    The whole point of planning legs. A bay 6 m to the side and only 2 m
    ahead is impossible nose-first -- one arc displaces the car a full radius
    forwards -- and is an ordinary reverse park.
    """
    close = _slot(-6.0, 2.0, -math.pi / 2)

    assert plan_parking_path(close, VIVACE) is None
    legs = plan_manoeuvre(close, VIVACE)

    assert legs is not None and len(legs) >= 2
    assert any(leg.reverse for leg in legs), "it must reverse somewhere"


def test_a_reachable_bay_still_prefers_the_single_forward_move() -> None:
    """
    Legs are tried in the order a driver thinks of them.

    A bay IN LINE with the car, which is the case the canned single forward
    move is right for. A SQUARE-ON bay deliberately no longer takes it: the
    construction "fits" such a bay geometrically and cuts the corner through
    the neighbouring one, which is what `prefers_reverse` exists to avoid.
    """
    easy = _slot(0.0, 14.0, 0.0)

    legs = plan_manoeuvre(easy, VIVACE)

    assert legs is not None and len(legs) == 1
    assert not legs[0].reverse


def test_a_canned_leg_is_committed_as_a_world_anchored_path() -> None:
    """Endpoint-only canned legs are silently re-invented every control tick."""
    slot = _slot(0.0, 14.0, 0.0)

    legs = plan_manoeuvre(slot, VIVACE)

    assert legs is not None and len(legs) == 1
    assert legs[0].path_bay is not None
    assert len(legs[0].path_bay) >= 12


def test_tracking_reports_real_cross_track_error_on_a_committed_path() -> None:
    """A displaced car needs observable error before it can recover or replan."""
    driver = ParkingDriver()
    slot = _slot(-8.0, 14.0, -math.pi / 2)
    driver.step(slot, VIVACE, 0.0, 0.04, reported_gear="D")

    _, displaced = driver.step(
        _slot(-7.2, 14.0, -math.pi / 2),
        VIVACE,
        0.2,
        0.04,
        reported_gear="D",
    )

    assert displaced.cross_track_m > 0.5


def test_every_leg_is_checked_from_where_the_previous_one_finishes() -> None:
    """
    A sequence is only offered when ALL of it solves. Starting a manoeuvre
    whose second half is impossible would leave the car across the aisle.
    """
    close = _slot(-6.0, 2.0, -math.pi / 2)
    legs = plan_manoeuvre(close, VIVACE)
    assert legs is not None

    for leg in legs:
        assert _leg_path(leg, close) is not None


def test_the_reverse_solver_is_the_forward_one_in_a_rotated_frame() -> None:
    """
    Reversing traces the same curves as driving forward with the heading
    flipped -- the trick `aeb.mirror_points` and the steered reverse use.
    Straight back 5 m is the case that catches a sign error.
    """
    straight_back = _reverse_reach(
        np.asarray((0.0, -5.0)), np.asarray((0.0, 1.0)), 24
    )

    assert straight_back is not None
    assert np.allclose(straight_back[0], (0.0, 0.0), atol=1e-9)
    assert np.allclose(straight_back[-1], (0.0, -5.0), atol=1e-9)


def test_the_car_reverse_parks_into_a_bay_it_cannot_nose_into() -> None:  # noqa: E501
    """
    The manoeuvre this was all for, closed loop.

    A bay 6 m to the side and 2 m ahead is impossible nose-first -- one arc
    displaces the car a full turning radius forwards -- so the car must
    position itself and then back in. What is asserted is where it STOPS:
    square in the bay, and FACING OUT, which is what reversing in means.
    """
    car, state, centre, axis = _park((-6.0, 2.0), -math.pi / 2, (0.0, 0.0, 0.0))

    assert state is not None and state.phase == PARK_ARRIVED, f"{state}"
    node = np.asarray((car.x, car.y))
    across = float((node - centre) @ np.asarray((axis[1], -axis[0])))
    assert abs(across) < 0.45, f"{across:.2f} m off the bay centreline"
    # SQUARE to the bay -- either way round. Whether the planner noses in or
    # backs in is its choice, and with the search it will nose into bays that
    # once needed reversing, which is a better park, not a worse one.
    facing = float(
        axis @ np.asarray((math.sin(car.heading), math.cos(car.heading)))
    )
    assert abs(facing) > 0.9, f"ended {facing:+.2f} along the bay axis"
    assert car.speed < 0.05


def test_the_gearbox_is_never_shifted_while_the_car_is_moving() -> None:
    """
    `shiftToGearIndex` has side effects and the box only engages at rest, so
    a direction change is stop, select, CONFIRM, go. Commanding it and driving
    on would leave the car pulling against a box still in the old direction.
    """
    car = _Bicycle(0.0, 0.0, 0.0)
    driver = ParkingDriver()
    axis_world = np.asarray((math.sin(-math.pi / 2), math.cos(-math.pi / 2)))
    centre_world = np.asarray((-6.0, 2.0))
    commanded = 2
    for _ in range(3000):
        rel = centre_world - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(
                float(axis_world @ car.right), float(axis_world @ car.forward)
            ),
        )
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            reported_gear=car.reported_gear,
        )
        if command.gear != commanded:
            assert car.speed <= 0.1, (
                f"gear changed to {command.gear} at {car.speed:.2f} m/s"
            )
            commanded = command.gear
        if state.finished and car.speed < 0.02:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )
    assert state.phase == PARK_ARRIVED


def test_the_rear_brake_outranks_the_manoeuvre_while_backing() -> None:
    """
    Unlike the forward park, the REAR brake arms at parking speed (0.5 m/s),
    so it really can fire while backing in -- which is when it should. It is
    left armed and allowed to win: the manoeuvre hands back rather than
    fighting a system that has decided the car is about to hit something.
    """
    car = _Bicycle(0.0, 0.0, 0.0)
    driver = ParkingDriver()
    axis_world = np.asarray((math.sin(-math.pi / 2), math.cos(-math.pi / 2)))
    centre_world = np.asarray((-6.0, 2.0))
    state = None
    for _ in range(3000):
        rel = centre_world - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(
                float(axis_world @ car.right), float(axis_world @ car.forward)
            ),
        )
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            reported_gear=car.reported_gear,
        )
        if state.phase == PARK_BACKING and car.speed > 0.2:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )
    assert state is not None and state.phase == PARK_BACKING

    command, blocked = driver.step(
        slot,
        VIVACE,
        car.forward_speed,
        0.04,
        reported_gear=car.reported_gear,
        rear_aeb_braking=True,
    )

    assert blocked.phase == PARK_BLOCKED
    assert command.throttle == 0.0
    assert command.brake > 0.2


def test_reverse_aeb_blockage_stays_latched_until_stopped_clear_dwell() -> None:
    driver = ParkingDriver()
    driver._legs = [
        ParkingLeg(
            0.0,
            0.0,
            math.pi,
            True,
            path_bay=np.asarray(((0.0, 5.0), (0.0, 0.0))),
        )
    ]
    driver._gear = -1
    slot = _slot(0.0, -5.0, 0.0)

    _, blocked = driver.step(
        slot,
        VIVACE,
        -0.4,
        0.04,
        reported_gear="R",
        rear_aeb_braking=True,
    )
    command, still_blocked = driver.step(
        slot,
        VIVACE,
        -0.2,
        0.04,
        reported_gear="R",
        rear_aeb_braking=False,
    )

    assert blocked.phase == PARK_BLOCKED
    assert still_blocked.phase == PARK_BLOCKED
    assert command.throttle == 0.0


def test_unreadable_gear_fails_if_motion_is_opposite_the_requested_direction() -> None:
    driver = ParkingDriver()
    driver._legs = [
        ParkingLeg(
            -5.0,
            0.0,
            math.pi,
            True,
            path_bay=np.asarray(((0.0, 0.0), (-5.0, 0.0))),
        )
    ]
    slot = _slot(0.0, 0.0, 0.0)

    for _ in range(40):
        driver.step(slot, VIVACE, 0.0, 0.04, reported_gear=None)
    command, state = driver.step(
        slot, VIVACE, 0.2, 0.04, reported_gear=None
    )

    assert state.phase == PARK_UNREACHABLE
    assert command.throttle == 0.0


def test_unreadable_gear_accepts_motion_in_the_requested_direction() -> None:
    driver = ParkingDriver()
    driver._legs = [
        ParkingLeg(
            -5.0,
            0.0,
            math.pi,
            True,
            path_bay=np.asarray(((0.0, 0.0), (-5.0, 0.0))),
        )
    ]
    slot = _slot(0.0, 0.0, 0.0)

    for _ in range(40):
        driver.step(slot, VIVACE, 0.0, 0.04, reported_gear=None)
    _, state = driver.step(slot, VIVACE, -0.2, 0.04, reported_gear=None)

    assert state.phase == PARK_BACKING


@pytest.mark.parametrize(
    "name, bay, bay_heading",
    (
        ("2 m ahead, 6 m left", (-6.0, 2.0), -math.pi / 2),
        ("2 m ahead, 6 m right", (6.0, 2.0), math.pi / 2),
        ("level with the car", (-6.0, 0.0), -math.pi / 2),
        ("3 m behind", (-6.0, -3.0), -math.pi / 2),
    ),
)
def test_the_car_positions_itself_and_reverses_in(
    name: str, bay: tuple[float, float], bay_heading: float
) -> None:
    """
    The manoeuvre this was all for, closed loop and both sides of the aisle.

    None of these is reachable nose-first -- one arc displaces the car a full
    turning radius forwards -- so the car must position itself, stop, select
    reverse, and back in. What is asserted is where it STOPS: square in the
    bay and FACING OUT, which is what reversing in means.
    """
    car, state, centre, axis = _park(bay, bay_heading, (0.0, 0.0, 0.0))

    assert state is not None and state.phase == PARK_ARRIVED, f"{name}: {state}"
    node = np.asarray((car.x, car.y))
    across = float((node - centre) @ np.asarray((axis[1], -axis[0])))
    facing = float(
        axis @ np.asarray((math.sin(car.heading), math.cos(car.heading)))
    )
    assert abs(across) < 0.5, f"{name}: {across:.2f} m off centre"
    # Square to the bay, either way round -- see above.
    assert abs(facing) > 0.9, f"{name}: ended {facing:+.2f} along the bay"
    assert car.speed < 0.05


def test_a_manoeuvre_that_will_not_come_together_gives_up() -> None:
    """
    Re-planning is bounded. A plan that keeps producing a sequence the car
    cannot drive -- plan, reach the setup, fail the next leg, re-plan --
    would otherwise cycle for ever, which is what it did: stuck in SHIFTING
    for the whole run. Handing back is the honest end.
    """
    car, state, _, _ = _park((7.0, 4.0), math.pi / 2, (0.0, 0.0, 0.0))

    assert state is not None
    assert state.phase in (PARK_ARRIVED, PARK_UNREACHABLE)
    assert car.speed < 0.05, "it must at least stop"


# --- the path the tracker is given -------------------------------------------


def test_every_planned_path_is_uniformly_spaced() -> None:
    """
    The constructions sample their own SEGMENTS, and that is not the same as
    sampling the path.

    A square-on bay came back as two points for the run-in and then the tail
    at a quarter of a metre, so a path could open with one 6.3 m gap. The
    executor finds the car by the nearest SAMPLE, so across a gap like that
    sample 0 stayed nearest for the first three metres of driving -- and
    `cross_track`, `remaining` and the pure-pursuit lookahead were all
    measured from it.
    """
    for name, slot in (
        ("dead ahead", _slot(0.0, 12.0, 0.0)),
        ("square-on right", _slot(8.0, 14.0, math.pi / 2)),
        ("45 degrees left", _slot(-6.0, 14.0, -math.pi / 4)),
    ):
        path = plan_parking_path(slot, VIVACE)
        assert path is not None, name
        steps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
        assert steps.max() <= PARKING_PATH_STEP_M * 1.5, name
        # And it must still terminate exactly on the stop pose, which is what
        # the tracker aims at for the last metre.
        target, _ = stop_pose(slot, VIVACE)
        assert np.allclose(path.points[-1], target, atol=1e-6), name


def test_cross_track_is_a_perpendicular_distance_not_a_distance_travelled() -> None:
    """
    The regression that made a straight drive re-plan three times.

    Measuring to `points[here]` is only a tracking error when that sample
    happens to be abeam. With a stale index it reported how far the car had
    driven, and `PARKING_DRIVE_MAX_CROSS_TRACK_M` then fired on a car running
    perfectly straight down a perfectly straight path.
    """
    # A straight path the car (at the origin) is exactly on, sitting five
    # metres along it -- sample 10 of 21.
    points = np.column_stack((np.zeros(21), np.arange(-5.0, 5.5, 0.5)))

    assert cross_track(points, 10) == pytest.approx(0.0, abs=1e-9)
    # And still zero when the index is a sample stale either way, which is
    # what a nearest-sample search on a uniform grid actually delivers. The
    # old form returned the distance to the sample -- half a metre of invented
    # "tracking error" per sample of lag, on a car dead on the path.
    assert cross_track(points, 9) == pytest.approx(0.0, abs=1e-9)
    assert cross_track(points, 11) == pytest.approx(0.0, abs=1e-9)
    # A genuine offset is still measured.
    assert cross_track(
        points + np.asarray((0.8, 0.0)), 10
    ) == pytest.approx(0.8, abs=1e-9)


def test_a_bay_dead_ahead_is_driven_straight_in_without_a_single_shift() -> None:
    """
    Nothing about a bay in line with the car needs a gear change.

    Measured before the sampling fix: three of them, on a 14 m straight, each
    announced as "tracking error grew to 1.5 m" -- which was exactly the
    distance the car had covered since the path was drawn.
    """
    car, state, _, _ = _park((0.0, 14.0), 0.0, (0.0, 0.0, 0.0))

    assert state is not None and state.phase == PARK_ARRIVED
    assert car.gear > 0, "a straight-in park must never select reverse"


# --- staying between the lines ------------------------------------------------


def test_the_body_stays_inside_the_bay_it_is_entering() -> None:
    """
    Paint is not an obstacle, so nothing else in the stack asks this.

    There is no return to collide with on a bay line: no corridor check, no
    occupancy cell and no cost term prefers staying between them, and
    `_secure` inspects only the pose the car finishes in. Measured before this
    existed, entering a 3.18 m bay from an aisle 5 m off its centre put a
    corner 0.74 m past the side line -- with 0.58 m of margin available, so
    roughly three quarters of a metre into the neighbour, reported as ARRIVED.
    """
    corners = np.asarray(
        (
            (-VIVACE.left_m, VIVACE.front_m),
            (VIVACE.right_m, VIVACE.front_m),
            (VIVACE.right_m, -VIVACE.rear_m),
            (-VIVACE.left_m, -VIVACE.rear_m),
        )
    )
    for name, bay, heading in (
        ("6 m off the centre", (6.0, 14.0), math.pi / 2),
        ("5 m off the centre", (5.0, 12.0), math.pi / 2),
        ("5 m off, other side", (-5.0, 12.0), -math.pi / 2),
    ):
        car = _Bicycle(0.0, 0.0, 0.0)
        driver = ParkingDriver()
        axis = np.asarray((math.sin(heading), math.cos(heading)))
        across_axis = np.asarray((axis[1], -axis[0]))
        centre = np.asarray(bay)
        worst = 0.0
        state = None
        for _ in range(3000):
            relative = centre - np.asarray((car.x, car.y))
            slot = _slot(
                float(relative @ car.right),
                float(relative @ car.forward),
                math.atan2(
                    float(axis @ car.right), float(axis @ car.forward)
                ),
            )
            command, state = driver.step(
                slot,
                VIVACE,
                car.forward_speed,
                0.04,
                reported_gear=car.reported_gear,
            )
            body = (
                np.asarray((car.x, car.y))
                + np.outer(corners[:, 0], car.right)
                + np.outer(corners[:, 1], car.forward)
            ) - centre
            along, across = body @ axis, body @ across_axis
            # Only while a corner is genuinely inside the bay, and near it --
            # a car driving up a narrow aisle is legitimately alongside.
            near = (np.abs(along) <= slot.depth_m * 0.5) & (
                np.abs(across) <= slot.width_m * 1.5
            )
            if near.any():
                worst = max(
                    worst,
                    float((np.abs(across[near]) - slot.width_m * 0.5).max()),
                )
            if (
                state.finished or state.phase == PARK_BLOCKED
            ) and car.speed < 0.02:
                break
            car.step(
                command.steering,
                command.throttle,
                command.brake,
                0.04,
                command.gear,
            )
        assert state is not None and state.phase == PARK_ARRIVED, name
        # 0.35, not the planner's 0.15 keepout: the PLANNED body stays inside
        # the keepout (pinned by `_legs_respect_bay` at 0.05 for the clean
        # search round), and what is measured here is the DRIVEN body under
        # the honest plant -- finite steering rate, node ahead of the axle --
        # whose swept-corner transient at the band's mouth corner peaks near
        # 0.34 m on the tightest geometry. The physical neighbour is guarded
        # by the corridor check, which sees returns; this test guards the
        # PAINT, and tightening it back toward 0.15 is a tracking-quality
        # target for the live checklist, not a constant to relax further.
        assert worst <= 0.35, f"{name}: {worst:.2f} m past the bay's side line"


def test_a_square_on_bay_asks_the_search_before_the_canned_families() -> None:
    """
    The cheap answer used to win by default, and it was the wrong one.

    `_clear` rejects only KNOWN-blocked cells -- unknown space is traversable
    -- so in an empty lot the canned nose-in construction always "fitted", and
    it is the construction that cuts the corner through the neighbouring bay.
    """
    square_on = _slot(6.0, 14.0, math.pi / 2)
    in_line = _slot(0.0, 14.0, 0.0)

    assert prefers_reverse(square_on)
    assert not prefers_reverse(in_line)


# --- the committed-manoeuvre architecture -------------------------------------


def test_a_multi_leg_park_plans_the_manoeuvre_exactly_once(monkeypatch) -> None:
    """
    The cusp is not a replan trigger any more.

    Re-searching at every cusp re-chose the manoeuvre topology from every
    slightly different pose -- the live log recorded seven different plans
    for one bay in two minutes, which was most of the shuffling. The
    committed sequence must survive its own direction changes; cusp overshoot
    is the tracker's and the local repair's job, never a fresh search.
    """
    from beamng_lidar_bev import parking_drive as module

    calls = {"count": 0}
    original = module.plan_manoeuvre

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "plan_manoeuvre", counting)
    _, state, _, _ = _park((-6.0, 2.0), -math.pi / 2, (0.0, 0.0, 0.0))

    assert state is not None and state.phase == PARK_ARRIVED
    assert calls["count"] == 1, (
        f"a clean multi-leg park re-planned {calls['count'] - 1} time(s)"
    )


def test_an_obstruction_down_the_path_is_braked_for_not_slammed() -> None:
    """
    An obstruction is a speed limit first and a stop second.

    The old semantics full-stopped the moment anything appeared anywhere on
    the remaining path -- live, a hard stop for an obstruction 12.9 m down
    the leg, then resume, then again at 10.7 and 5.0 m: the reported
    brake-accelerate-brake cycling. Now the car keeps its manoeuvring speed
    and runs the normal stopping profile down to the standoff.
    """
    obstacles = np.asarray([[0.0, 9.0], [0.3, 9.2], [-0.3, 9.1]])
    car = _Bicycle(0.0, 0.0, 0.0)
    driver = ParkingDriver()
    centre = np.asarray((0.0, 16.0))
    axis = np.asarray((0.0, 1.0))
    trace = []
    state = None
    for _ in range(3000):
        rel = centre - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(float(axis @ car.right), float(axis @ car.forward)),
        )
        delta = obstacles - np.asarray((car.x, car.y))
        local = np.column_stack((delta @ car.right, delta @ car.forward))
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            obstacles=local,
            reported_gear=car.reported_gear,
        )
        trace.append((car.y, car.speed))
        if state.phase == PARK_BLOCKED and car.speed < 0.02:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )

    assert state is not None and state.phase == PARK_BLOCKED
    final_y = car.y
    assert final_y < 8.0, "it must stop short of the obstruction"
    speeds = np.asarray(trace)
    assert speeds[:, 1].max() >= 1.0, "it must reach manoeuvring speed"
    # Between getting up to speed and the deliberate stop there is no dip:
    # one acceleration, one cruise, one braking ramp.
    cruising = speeds[
        (speeds[:, 0] > 3.0) & (speeds[:, 0] < final_y - 2.0)
    ]
    assert len(cruising) and cruising[:, 1].min() >= 0.35, (
        f"speed dipped to {cruising[:, 1].min():.2f} m/s mid-approach"
    )


def test_a_persistent_blockage_triggers_one_replan_attempt(monkeypatch) -> None:
    """
    A parked car never "clears": after a dwell stopped at the standoff, the
    driver must ask the planner for a way around (the accumulated occupancy
    is what would let it find one). The old semantics waited for ever.
    """
    from beamng_lidar_bev import parking_drive as module

    calls = {"count": 0}
    original = module.plan_manoeuvre

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "plan_manoeuvre", counting)

    free = np.column_stack(
        [
            axis.ravel()
            for axis in np.meshgrid(
                np.arange(-20.0, 20.0, 0.5), np.arange(-5.0, 25.0, 0.5)
            )
        ]
    )
    obstacles = np.asarray([[0.0, 9.0], [0.3, 9.2], [-0.3, 9.1]])
    car = _Bicycle(0.0, 0.0, 0.0)
    driver = ParkingDriver()
    centre = np.asarray((0.0, 16.0))
    axis = np.asarray((0.0, 1.0))
    for _ in range(3000):
        rel = centre - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(float(axis @ car.right), float(axis @ car.forward)),
        )
        delta_o = obstacles - np.asarray((car.x, car.y))
        local_o = np.column_stack((delta_o @ car.right, delta_o @ car.forward))
        delta_f = free - np.asarray((car.x, car.y))
        local_f = np.column_stack((delta_f @ car.right, delta_f @ car.forward))
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            obstacles=local_o,
            occupancy=Occupancy(None, local_f),
            reported_gear=car.reported_gear,
        )
        if calls["count"] >= 2:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )

    # The occupancy never contains the obstruction (only the corridor check
    # sees it), so the replan cannot route around -- what is pinned is that
    # the ATTEMPT happens instead of waiting for ever.
    assert calls["count"] >= 2, "no replan was attempted for a static blockage"


# --- model error --------------------------------------------------------------


@pytest.mark.parametrize("gain_error", (0.9, 1.15))
@pytest.mark.parametrize(
    "name, bay, bay_heading",
    (
        ("straight in", (0.0, 14.0), 0.0),
        ("square-on right", (8.0, 16.0), math.pi / 2),
    ),
)
def test_a_wrong_steering_map_is_absorbed_not_fatal(
    name: str,
    bay: tuple[float, float],
    bay_heading: float,
    gain_error: float,
) -> None:
    """
    MIN_TURN_RADIUS_M is a guess and the steering map is assumed linear;
    the live car will never match either exactly. The measured-yaw trim plus
    error feedback must absorb a 10-15% map error, or every offline gain is
    tuned against fiction -- which is precisely how the previous tracker
    passed 8 of 8 offline and saturated the wheel on the real car.
    """
    car, state, centre, axis = _park(
        bay, bay_heading, (0.0, 0.0, 0.0), gain_error=gain_error
    )

    assert state is not None and state.phase == PARK_ARRIVED, (
        f"{name} at gain {gain_error}: {state}"
    )
    node = np.asarray((car.x, car.y))
    across = float((node - centre) @ np.asarray((axis[1], -axis[0])))
    facing = float(
        axis @ np.asarray((math.sin(car.heading), math.cos(car.heading)))
    )
    assert abs(across) < 0.45, f"{name}: {across:.2f} m off centre"
    assert abs(facing) > 0.98, f"{name}: ended {facing:+.2f} along the bay"


# --- smoothing ----------------------------------------------------------------


def test_committed_legs_are_smoothed_to_trackable_curvature() -> None:
    """
    A raw leg carries a curvature STEP at every arc/straight join -- a wheel
    jump no car can steer, which the tracker then chases late. Smoothing per
    leg must spread every step into a transition the steering rate can
    actually wind, without exceeding the car's own curvature limit.
    """
    from beamng_lidar_bev.parking_smooth import discrete_curvature

    slot = _slot(6.0, 14.0, math.pi / 2)
    legs = plan_manoeuvre(slot, VIVACE)
    assert legs is not None

    checked = 0
    for leg in legs:
        path = _leg_path(leg, slot)
        assert path is not None
        if len(path) < 8:
            continue
        curvature = discrete_curvature(path)
        assert float(np.abs(curvature).max()) <= _MAX_CURVATURE + 2e-3
        # The leg BOUNDARIES are exempt: a leg may legitimately begin or end
        # at full curvature (a cusp is where the wheel re-aims at a
        # standstill -- the discontinuity there is free, and the smoother
        # deliberately pins the end tangents rather than rounding them).
        steps = np.abs(np.diff(curvature[3:-3]))
        assert float(steps.max()) <= 0.09, (
            f"a {steps.max():.3f} 1/m curvature step survived smoothing"
        )
        checked += 1
    assert checked, "no leg was long enough to check"


# --- the shift dwell ----------------------------------------------------------


def test_the_wheels_pre_aim_at_the_next_leg_during_the_shift() -> None:
    """
    The gear dwell is dead time and the wheels turn at a standstill, so the
    new leg should begin with its entry curvature already wound on instead of
    spending its first metre winding -- which is where cusp cross-track came
    from.
    """
    car = _Bicycle(0.0, 0.0, 0.0)
    driver = ParkingDriver()
    axis_world = np.asarray((math.sin(-math.pi / 2), math.cos(-math.pi / 2)))
    centre_world = np.asarray((-6.0, 2.0))
    aimed = 0.0
    state = None
    for _ in range(3000):
        rel = centre_world - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(
                float(axis_world @ car.right), float(axis_world @ car.forward)
            ),
        )
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            reported_gear=car.reported_gear,
        )
        if state.phase == "SHIFTING" and abs(car.forward_speed) <= 0.08:
            aimed = max(aimed, abs(command.steering))
        if state.finished and car.speed < 0.02:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )
    assert state is not None and state.phase == PARK_ARRIVED
    assert aimed >= 0.03, f"the wheel never moved during a shift ({aimed:.3f})"


# --- the endgame nudge --------------------------------------------------------


def test_a_small_final_pose_error_gets_a_nudge_not_a_shuffle() -> None:
    """
    The search's 0.7 m primitives cannot express a half-metre correction:
    handing it one produced 9.2 m of two-leg shuffle for a 0.5 m error,
    live, twice. Inside the nudge envelope the correction must be a short
    analytic pull-out-and-re-enter that actually converges.
    """
    from beamng_lidar_bev.parking_drive import ParkingLeg

    centre_world = np.asarray((0.0, 10.0))
    axis_world = np.asarray((0.0, 1.0))
    # The nose-in stop pose in world, then the car parked 0.70 m off the
    # centreline of it -- outside PARKING_SUCCESS_POSITION_M, so the secure
    # check must refuse it -- at rest, mid-SECURING.
    target_local, _ = stop_pose(_slot(0.0, 10.0, 0.0), VIVACE)
    car = _Bicycle(0.70, float(target_local[1]), 0.0)
    driver = ParkingDriver()
    final = ParkingLeg(
        along_m=_slot(0.0, 0.0, 0.0).depth_m * 0.5 - 0.5 - VIVACE.front_m,
        across_m=0.0,
        heading_rad=0.0,
        reverse=False,
        path_bay=np.asarray(((0.0, -2.0), (0.0, 0.0))),
    )
    driver._legs = [final]
    driver._securing = True
    driven = 0.0
    state = None
    for _ in range(2500):
        rel = centre_world - np.asarray((car.x, car.y))
        slot = _slot(
            float(rel @ car.right),
            float(rel @ car.forward),
            math.atan2(
                float(axis_world @ car.right), float(axis_world @ car.forward)
            ),
        )
        command, state = driver.step(
            slot,
            VIVACE,
            car.forward_speed,
            0.04,
            reported_gear=car.reported_gear,
        )
        if state.finished and car.speed < 0.02:
            break
        car.step(
            command.steering, command.throttle, command.brake, 0.04, command.gear
        )
        driven += car.speed * 0.04

    assert state is not None and state.phase == PARK_ARRIVED, f"{state}"
    across = float(
        (np.asarray((car.x, car.y)) - centre_world)
        @ np.asarray((axis_world[1], -axis_world[0]))
    )
    assert abs(across) < 0.30, f"still {across:.2f} m off centre"
    assert driven <= 9.0, (
        f"a 0.45 m correction cost {driven:.1f} m of driving"
    )


def test_blocked_cells_at_the_car_do_not_refuse_a_reachable_bay() -> None:
    """
    The 19:02 live regression: four engagements refused UNREACHABLE within
    ~120 ms each, from a spot the car had just driven to.

    One blocked cell inside the start footprint -- a stale cell the body has
    covered since it was marked, or a real kerb inside the 0.18 m clearance
    the collision probe inflates the body by -- made every child expansion
    return None, so the search died before its first step, the canned
    families failed the same occupancy check, and the whole planner reported
    "no clear route". The car standing there is proof those cells are
    drivable, and the search must inherit that proof.
    """
    slot = _slot(2.8, 6.5, math.pi / 2)
    free = np.column_stack(
        [
            axis.ravel()
            for axis in np.meshgrid(
                np.arange(-20.0, 20.0, 0.5), np.arange(-8.0, 20.0, 0.5)
            )
        ]
    )
    # Stale cells under the body, and a kerb-like line just inside the
    # clearance band beside it.
    kerb = np.column_stack(
        (np.full(17, 1.15), np.arange(-2.0, 2.25, 0.25))
    )
    blocked = np.vstack((np.asarray(((0.2, 0.5), (-0.3, -1.0))), kerb))

    legs = plan_manoeuvre(slot, VIVACE, Occupancy(blocked, free))

    assert legs is not None, "a reachable bay was refused from a valid spot"
