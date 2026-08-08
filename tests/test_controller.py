from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    BLOCKED_HOLD_S,
    BRAKE_HOLD_FRACTION,
    COAST_DECEL_MPS2,
    COMFORT_DECEL_MPS2,
    CORNERING_ACCEL_MPS2,
    K_RATE_CEIL_PER_S,
    LAT_JERK_MAX_MPS3,
    MAX_LATERAL_ACCEL_MPS2,
    MAX_RECOVERY_ATTEMPTS,
    MAX_SPEED_MPS,
    MIN_TURN_RADIUS_M,
    PLANNER_HORIZON_M,
    REVERSE_DISTANCE_M,
    SPEED_KV,
    STEER_GAIN_MIN,
    STEERING_SIGN,
    STOP_MARGIN_M,
    STUCK_TIMEOUT_S,
    THROTTLE_SLEW_UP_PER_S,
)
from beamng_lidar_bev.controller import (
    AUTOMATIC_FORWARD_GEAR,
    MANUAL_FORWARD_GEAR,
    REVERSE_GEAR,
    DrivingController,
    forward_gear_index,
    is_forward_gear,
    is_reverse_gear,
)
from beamng_lidar_bev.models import ArcPlan

DT = 0.04
OPEN_REAR = 20.0
_EMPTY = np.zeros(1, dtype=np.float32)


def _plan(
    free_distance_m: float = PLANNER_HORIZON_M, curvature: float = 0.0
) -> ArcPlan:
    return ArcPlan(
        curvature=curvature,
        free_distance_m=free_distance_m,
        clearance_m=3.0,
        keep_right_target_m=None,
        nav_heading_rad=None,
        candidate_curvatures=_EMPTY,
        candidate_costs=_EMPTY,
        candidate_free_distances=_EMPTY,
    )


def _run(
    controller: DrivingController,
    plan: ArcPlan,
    speed: float,
    ticks: int,
    rear_free_distance_m: float = OPEN_REAR,
):
    command = None
    for _ in range(ticks):
        command = controller.step(
            plan, speed, DT, rear_free_distance_m=rear_free_distance_m
        )
    assert command is not None
    return command


# --- gear selection ----------------------------------------------------------
#
# Regression cover for the bug where forward drive commanded gear 1, which is
# PARK on every automatic-family gearbox (hShifterModeLookup = {[-1]="R",
# [0]="N", "P", "D", ...}, so 1 -> "P" and 2 -> "D"). The car applied throttle
# and never moved. Manual boxes have no such lookup: 1 really is first gear.


@pytest.mark.parametrize("reported", ["P", "N", "D", "R", "S2", "M3"])
def test_a_gearbox_reporting_a_mode_string_is_driven_in_second_index(
    reported: str,
) -> None:
    assert forward_gear_index(reported) == AUTOMATIC_FORWARD_GEAR == 2


@pytest.mark.parametrize("reported", [0, 3, -1, 1.0])
def test_a_gearbox_reporting_a_number_is_driven_in_first_index(
    reported: object,
) -> None:
    assert forward_gear_index(reported) == MANUAL_FORWARD_GEAR == 1


def test_an_unknown_gearbox_defaults_to_the_automatic_index() -> None:
    """
    The two failure modes are not symmetric. Index 2 on a manual is second gear
    and the car still crawls away; index 1 on an automatic is PARK and the car
    is dead. So an unreadable gearbox defaults to the survivable guess.
    """
    assert forward_gear_index(None) == AUTOMATIC_FORWARD_GEAR


@pytest.mark.parametrize("reported", ["D", "S3", "M1", 2, 1])
def test_forward_modes_are_recognised(reported: object) -> None:
    assert is_forward_gear(reported) is True


@pytest.mark.parametrize("reported", ["P", "N", "R", 0, -2, None])
def test_non_forward_modes_are_recognised(reported: object) -> None:
    assert is_forward_gear(reported) is False


@pytest.mark.parametrize("reported", ["R", "r", -1, -2])
def test_reverse_modes_are_recognised(reported: object) -> None:
    assert is_reverse_gear(reported) is True


@pytest.mark.parametrize("reported", ["P", "N", "D", "M1", 0, 1, None])
def test_non_reverse_modes_are_recognised(reported: object) -> None:
    assert is_reverse_gear(reported) is False


def test_a_gearbox_sitting_in_park_is_commanded_into_drive() -> None:
    """Regression: this commanded gear 1, which is PARK, so nothing happened."""
    command = DrivingController().step(_plan(), 0.0, DT, reported_gear="P")

    assert command.gear == 2
    assert command.throttle > 0.0


def test_a_manual_gearbox_is_commanded_into_first() -> None:
    command = DrivingController().step(_plan(), 0.0, DT, reported_gear=0)

    assert command.gear == 1


@pytest.mark.parametrize("reported", ["P", 0])
def test_reverse_is_the_same_index_for_either_gearbox(reported: object) -> None:
    controller = DrivingController()
    blocked = _plan(free_distance_m=1.0)
    for _ in range(int(BLOCKED_HOLD_S / DT) + 2):
        command = controller.step(
            blocked, 0.0, DT, rear_free_distance_m=OPEN_REAR, reported_gear=reported
        )

    assert command.mode == "REVERSING"
    assert command.gear == REVERSE_GEAR == -1


# --- longitudinal ------------------------------------------------------------


def test_target_speed_never_exceeds_the_cap() -> None:
    command = DrivingController().step(_plan(), 0.0, DT)

    assert command.target_speed_mps == pytest.approx(MAX_SPEED_MPS)


def test_a_tight_arc_lowers_the_target_speed() -> None:
    curvature = 1.0 / 8.0

    command = DrivingController().step(_plan(curvature=curvature), 0.0, DT)

    assert command.target_speed_mps == pytest.approx(
        np.sqrt(CORNERING_ACCEL_MPS2 / curvature)
    )
    assert command.target_speed_mps < MAX_SPEED_MPS


def test_a_short_free_distance_lowers_the_target_speed() -> None:
    command = DrivingController().step(_plan(free_distance_m=8.0), 0.0, DT)

    assert command.target_speed_mps < MAX_SPEED_MPS
    assert command.target_speed_mps > 0.0


def test_it_accelerates_from_rest_on_an_open_road() -> None:
    command = DrivingController().step(_plan(), 0.0, DT)

    assert command.throttle > 0.0
    assert command.brake == 0.0
    assert command.gear != REVERSE_GEAR


def test_it_brakes_when_above_the_target_speed() -> None:
    command = DrivingController().step(_plan(), MAX_SPEED_MPS + 3.0, DT)

    assert command.brake > 0.0
    assert command.throttle == 0.0


def test_throttle_and_brake_are_never_both_applied() -> None:
    controller = DrivingController()

    for speed in (0.0, 2.0, MAX_SPEED_MPS, MAX_SPEED_MPS + 5.0):
        command = controller.step(_plan(), speed, DT)
        assert command.throttle == 0.0 or command.brake == 0.0


def test_target_speed_brakes_to_corner_entry_not_to_zero() -> None:
    plan = ArcPlan(
        curvature=0.0,
        free_distance_m=30.0,
        clearance_m=3.0,
        keep_right_target_m=None,
        nav_heading_rad=None,
        candidate_curvatures=_EMPTY,
        candidate_costs=_EMPTY,
        candidate_free_distances=_EMPTY,
        next_curvature=1.0 / 6.0,
        transition_distance_m=10.0,
    )

    command = DrivingController().step(plan, 8.0, DT)

    entry = np.sqrt(
        CORNERING_ACCEL_MPS2 * 6.0 + 2.0 * COMFORT_DECEL_MPS2 * 10.0
    )
    assert command.target_speed_mps == pytest.approx(entry, rel=1e-6)
    assert command.target_speed_mps > np.sqrt(CORNERING_ACCEL_MPS2 * 6.0)


def test_a_one_tick_free_distance_dip_never_reaches_the_brake() -> None:
    """
    Regression pin for "it brakes for no reason".

    Free distance is a nearest-point measurement over a cloud rebuilt every
    40 ms, so single-tick dips are noise. One stray return at 20 m is measured
    to take a clear road's free distance to 19.9 m, and unfiltered anything
    under ~25 m put the pedal down at the speed cap.
    """
    controller = DrivingController()
    open_road = _plan()
    _run(controller, open_road, MAX_SPEED_MPS, 50)

    dipped = controller.step(_plan(free_distance_m=22.0), MAX_SPEED_MPS, DT)
    assert dipped.brake == 0.0

    recovered = controller.step(open_road, MAX_SPEED_MPS, DT)
    assert recovered.brake == 0.0
    assert recovered.target_speed_mps > MAX_SPEED_MPS - 0.5


def test_a_sustained_obstruction_still_brakes() -> None:
    """The filter must delay noise, never a real slowdown."""
    controller = DrivingController()
    _run(controller, _plan(), MAX_SPEED_MPS, 50)

    command = _run(controller, _plan(free_distance_m=20.0), MAX_SPEED_MPS, 25)

    assert command.brake > 0.0
    assert command.target_speed_mps < MAX_SPEED_MPS


def test_a_collapsed_free_distance_bypasses_the_target_filter() -> None:
    """A genuine emergency must not wait out the low-pass."""
    controller = DrivingController()
    _run(controller, _plan(), MAX_SPEED_MPS, 50)

    command = controller.step(_plan(free_distance_m=6.0), MAX_SPEED_MPS, DT)

    assert command.brake > 0.0


@pytest.mark.parametrize("radius_m", (25.0, 17.0))
def test_a_turn_in_stays_inside_the_lateral_acceleration_limit(
    radius_m: float,
) -> None:
    """
    Regression pin against crediting the steering wind-on to the speed law.

    "Hold speed, the wheel is not wound on yet" reads as smoothness and is
    circular: the credit keeps the target high, the high target commands no
    braking, and the wind-on finishes with the car still at speed. Measured at
    6.7 m/s^2 through a 17 m bend against a 3.5 limit. Anticipation belongs to
    the planned path -- the deferred families -- not to the controller.
    """
    curvature = 1.0 / radius_m
    controller = DrivingController()
    speed = MAX_SPEED_MPS
    _run(controller, _plan(), speed, 50)

    peak = 0.0
    for _ in range(500):  # 20 s: through turn-in and well into the corner
        command = controller.step(_plan(curvature=curvature), speed, DT)
        # Deliberately drag-poor plant, so the speed law has to do the slowing
        # rather than the model quietly doing it for free.
        speed = max(
            0.0,
            speed + (command.throttle * 3.5 - command.brake * 6.0 - 0.15) * DT,
        )
        peak = max(peak, speed**2 * abs(controller.current_curvature))

    # Never past the grip ceiling, and settling on the comfort figure with
    # steering authority still in reserve -- that reserve is what lets the
    # car actually track a tight corner instead of running wide out of it.
    assert peak <= MAX_LATERAL_ACCEL_MPS2
    assert speed**2 * curvature == pytest.approx(CORNERING_ACCEL_MPS2, rel=0.15)


def test_the_brake_releases_with_hysteresis_not_at_the_same_threshold() -> None:
    """One shared threshold makes the pedal chatter across the boundary."""
    controller = DrivingController()
    # Well over target: solidly on the brake.
    _run(controller, _plan(), MAX_SPEED_MPS + 5.0, 20)
    assert controller._braking

    # Back inside the coast band, but not yet past the release threshold.
    a_des_target = -COAST_DECEL_MPS2 * 0.75
    speed = MAX_SPEED_MPS - a_des_target / SPEED_KV
    controller.step(_plan(), speed, DT)
    assert controller._braking

    # Past it, the pedal comes off.
    controller.step(_plan(), MAX_SPEED_MPS + 0.1, DT)
    assert not controller._braking


def test_slightly_over_target_coasts_instead_of_braking() -> None:
    """The lift-off band: neither pedal moves for a small overspeed."""
    command = _run(DrivingController(), _plan(), MAX_SPEED_MPS + 0.5, 10)

    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_throttle_ramps_at_the_slew_limit_from_rest() -> None:
    controller = DrivingController()

    first = controller.step(_plan(), 0.0, DT)
    second = controller.step(_plan(), 0.0, DT)

    assert first.throttle == pytest.approx(THROTTLE_SLEW_UP_PER_S * DT)
    assert second.throttle == pytest.approx(2 * THROTTLE_SLEW_UP_PER_S * DT)


def test_an_emergency_stop_brakes_immediately_and_fully() -> None:
    command = DrivingController().step(_plan(free_distance_m=1.0), 8.0, DT)

    assert command.mode == "BLOCKED"
    assert command.brake == 1.0


def test_the_brake_relaxes_to_a_hold_once_stopped() -> None:
    command = _run(DrivingController(), _plan(free_distance_m=1.0), 0.0, 5)

    assert command.brake == pytest.approx(BRAKE_HOLD_FRACTION)


# --- lateral -----------------------------------------------------------------


def test_steering_is_rate_limited() -> None:
    """Regression guard: the arc fan is discrete, so raw steps reach the wheels."""
    controller = DrivingController()

    command = controller.step(_plan(curvature=1.0 / 6.0), 0.0, DT)

    # At standstill the curvature slew runs at its manoeuvring ceiling.
    assert abs(command.steering) == pytest.approx(
        (K_RATE_CEIL_PER_S * DT) / (1.0 / MIN_TURN_RADIUS_M)
    )


def test_curvature_slew_is_speed_scheduled() -> None:
    """The same steering step is a lurch at 40 km/h and nothing at parking pace."""
    fast, slow = DrivingController(), DrivingController()

    fast.step(_plan(curvature=1.0 / 6.0), 10.0, DT)
    slow.step(_plan(curvature=1.0 / 6.0), 0.0, DT)

    assert fast.current_curvature == pytest.approx(
        (LAT_JERK_MAX_MPS3 / 100.0) * DT
    )
    assert slow.current_curvature == pytest.approx(K_RATE_CEIL_PER_S * DT)


def test_steering_gain_adapts_to_an_understeering_plant() -> None:
    """
    Closed loop against a car that yaws at 70% of the commanded curvature:
    the gain must climb toward 1/0.7 so the driven path matches the plan.
    """
    controller = DrivingController()
    heading, speed, plant_gain = 0.0, 6.0, 0.7
    for _ in range(3000):  # 120 simulated seconds
        command = controller.step(
            _plan(curvature=0.08), speed, DT, heading_rad=heading
        )
        real_curvature = plant_gain * (command.steering / STEERING_SIGN) * (
            1.0 / MIN_TURN_RADIUS_M
        )
        heading += speed * real_curvature * DT

    assert 1.2 < controller.steering_gain < 1.6


def test_the_gain_never_leaves_its_clamps() -> None:
    controller = DrivingController()
    heading = 0.0
    for _ in range(2000):
        command = controller.step(
            _plan(curvature=0.1), 6.0, DT, heading_rad=heading
        )
        # A wildly oversteering plant: the gain must pin at the clamp rather
        # than running away or flipping sign.
        heading += 6.0 * 5.0 * (command.steering / STEERING_SIGN) * DT

    assert controller.steering_gain >= STEER_GAIN_MIN


def test_heading_wrap_does_not_spike_the_yaw_estimate() -> None:
    controller = DrivingController()

    controller.step(_plan(), 5.0, DT, heading_rad=3.10)
    controller.step(_plan(), 5.0, DT, heading_rad=-3.10)  # crossed +/- pi

    measured = controller.measured_curvature
    assert measured is not None
    # Unwrapped the heading moved 0.083 rad; the naive diff would read 6.2 rad
    # in 40 ms and everything downstream would see a phantom spin.
    assert abs(measured) < 1.0


def test_without_a_heading_the_gain_stays_nominal() -> None:
    controller = DrivingController()

    _run(controller, _plan(curvature=0.1), 6.0, 100)

    assert controller.steering_gain == pytest.approx(1.0)
    assert controller.measured_curvature is None


def test_a_left_arc_commands_negative_steering() -> None:
    """
    Regression: the car steered the opposite way to the drawn path.

    BeamNG's steering input is `kbdSteerRight - kbdSteerLeft` (input.lua), so
    **positive is RIGHT** -- the opposite of what BeamNGpy's Vehicle.control
    docstring claims. Positive curvature is left, so it must come out negative.
    """
    left = _run(DrivingController(), _plan(curvature=1.0 / 6.0), 3.0, 40)

    assert left.steering < 0.0


def test_a_right_arc_commands_positive_steering() -> None:
    right = _run(DrivingController(), _plan(curvature=-1.0 / 6.0), 3.0, 40)

    assert right.steering > 0.0


def test_steering_stays_inside_the_control_range() -> None:
    command = _run(DrivingController(), _plan(curvature=1.0 / 6.0), 3.0, 200)

    assert -1.0 <= command.steering <= 1.0


def test_reversing_steers_straight_back() -> None:
    controller = DrivingController()
    blocked = _plan(free_distance_m=1.0, curvature=1.0 / 6.0)
    _run(controller, blocked, 0.0, int(BLOCKED_HOLD_S / DT) + 2)

    command = controller.step(blocked, 0.0, DT, rear_free_distance_m=OPEN_REAR)

    assert command.mode == "REVERSING"
    assert command.steering == pytest.approx(0.0)


# --- blocked / recovery ------------------------------------------------------


def test_a_blocked_plan_stops_the_car() -> None:
    command = DrivingController().step(_plan(free_distance_m=1.0), 4.0, DT)

    assert command.mode == "BLOCKED"
    assert command.brake > 0.0
    assert command.throttle == 0.0


def test_it_resumes_driving_when_the_way_clears() -> None:
    controller = DrivingController()
    controller.step(_plan(free_distance_m=1.0), 0.0, DT)

    command = controller.step(_plan(), 0.0, DT)

    assert command.mode == "DRIVING"
    assert command.throttle > 0.0


def test_it_holds_before_reversing() -> None:
    """Traffic moves. Reversing immediately would be wrong far more often."""
    controller = DrivingController()

    command = _run(controller, _plan(free_distance_m=1.0), 0.0, 2)

    assert command.mode == "BLOCKED"


def test_it_reverses_after_holding_at_a_blockage() -> None:
    controller = DrivingController()

    command = _run(
        controller, _plan(free_distance_m=1.0), 0.0, int(BLOCKED_HOLD_S / DT) + 2
    )

    assert command.mode == "REVERSING"
    assert command.gear == -1
    assert command.throttle > 0.0


def test_it_does_not_reverse_into_an_obstacle_behind() -> None:
    controller = DrivingController()

    command = _run(
        controller,
        _plan(free_distance_m=1.0),
        0.0,
        int(BLOCKED_HOLD_S / DT) + 2,
        rear_free_distance_m=0.5,
    )

    assert command.mode != "REVERSING"
    assert command.throttle == 0.0


def test_reversing_ends_after_the_reverse_distance() -> None:
    controller = DrivingController()
    blocked = _plan(free_distance_m=1.0)
    _run(controller, blocked, 0.0, int(BLOCKED_HOLD_S / DT) + 2)

    ticks = int(REVERSE_DISTANCE_M / (2.0 * DT)) + 5
    command = _run(controller, blocked, -2.0, ticks)

    assert command.mode != "REVERSING"


def test_it_gives_up_after_repeated_recovery_attempts() -> None:
    controller = DrivingController()
    blocked = _plan(free_distance_m=1.0)

    command = _run(
        controller,
        blocked,
        0.0,
        (int(BLOCKED_HOLD_S / DT) + 2) * (MAX_RECOVERY_ATTEMPTS + 2),
        rear_free_distance_m=0.5,
    )

    assert command.mode == "STUCK"
    assert command.brake > 0.0


def test_commanded_throttle_with_no_motion_counts_as_blocked() -> None:
    """Kerbed or wedged: the arc looks clear but the car is not moving."""
    controller = DrivingController()

    command = _run(controller, _plan(), 0.0, int(STUCK_TIMEOUT_S / DT) + 2)

    assert command.mode != "DRIVING"


def test_a_moving_car_is_never_treated_as_stalled() -> None:
    controller = DrivingController()

    command = _run(controller, _plan(), 3.0, int(STUCK_TIMEOUT_S / DT) + 20)

    assert command.mode == "DRIVING"


def test_reset_returns_the_controller_to_driving() -> None:
    controller = DrivingController()
    _run(controller, _plan(free_distance_m=1.0), 0.0, int(BLOCKED_HOLD_S / DT) + 2)

    controller.reset()
    command = controller.step(_plan(), 0.0, DT)

    assert command.mode == "DRIVING"
    assert command.steering == pytest.approx(0.0)


def test_a_blocked_plan_is_reported_with_a_reason() -> None:
    command = DrivingController().step(_plan(free_distance_m=1.0), 0.0, DT)

    assert command.reason


def test_free_distance_at_the_stop_margin_is_blocked() -> None:
    command = DrivingController().step(
        _plan(free_distance_m=STOP_MARGIN_M), 0.0, DT
    )

    assert command.mode == "BLOCKED"
