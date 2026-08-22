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
    plan_manoeuvre,
    plan_parking_path,
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

    behind = blocking_distance(
        path,
        np.asarray(((0.0, 1.0),)),
        VIVACE,
        start_index=5,
    )
    ahead = blocking_distance(
        path,
        np.asarray(((0.0, 8.0),)),
        VIVACE,
        start_index=5,
    )

    assert behind == math.inf
    assert 0.0 <= ahead <= 2.0


# --- the closed loop ---------------------------------------------------------


class _Bicycle:
    """
    Kinematic plant: BeamNG steering in, pose out.

    Steering is the game's own convention (+1 = right), so it is divided by
    STEERING_SIGN to recover the planner's left-positive curvature -- the one
    place the two conventions meet, exactly as `controller._steer` does going
    the other way.
    """

    def __init__(self, x: float, y: float, heading: float) -> None:
        self.x, self.y, self.heading, self.speed = x, y, heading, 0.0
        self.gear = 2

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
        accel = throttle * THROTTLE_GAIN_MPS2 - brake * 8.0
        self.speed = max(0.0, self.speed + accel * dt)
        # Signed travel: reverse moves the car backwards along its heading.
        distance = self.speed * dt * (-1.0 if self.gear < 0 else 1.0)
        curvature = (steering / STEERING_SIGN) * _MAX_CURVATURE
        # Positive curvature is LEFT, which decreases a compass heading. The
        # sign of the DISTANCE is what makes reversing turn the other way,
        # which is the whole of the reverse-steering relation.
        self.heading -= curvature * distance
        self.x += math.sin(self.heading) * distance
        self.y += math.cos(self.heading) * distance

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
) -> tuple[_Bicycle, object, np.ndarray, np.ndarray]:
    """Drive the real controller into a world-anchored bay from `start`."""
    car = _Bicycle(*start)
    driver = ParkingDriver()
    axis_world = np.asarray(
        (math.sin(bay_heading_world), math.cos(bay_heading_world))
    )
    centre_world = np.asarray(bay_world)
    state = None
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
    assert abs(across) < 0.30, f"{name}: {across:.2f} m off centre"
    assert heading_error < math.radians(6.0), f"{name}: {heading_error:.3f} rad"
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
    """Legs are tried in the order a driver thinks of them."""
    easy = _slot(-8.0, 14.0, -math.pi / 2)

    legs = plan_manoeuvre(easy, VIVACE)

    assert legs is not None and len(legs) == 1
    assert not legs[0].reverse


def test_a_canned_leg_is_committed_as_a_world_anchored_path() -> None:
    """Endpoint-only canned legs are silently re-invented every control tick."""
    slot = _slot(-8.0, 14.0, -math.pi / 2)

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
