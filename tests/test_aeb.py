from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.aeb import (
    FORWARD,
    REVERSE,
    BrakeEvent,
    EmergencyBraking,
    corridor_cross_section,
    mirror_points,
    mirrored,
    predicted_corridor,
    stopping_distance,
)
from beamng_lidar_bev.config import (
    AEB_BRAKING_DECEL_MPS2,
    AEB_CONFIRM_S,
    AEB_HOLD_BRAKE,
    AEB_LATENCY_S,
    AEB_MIN_ENGAGED_S,
    AEB_MIN_HITS,
    AEB_MIN_SPEED_MPS,
    AEB_REVERSE_STANDOFF_M,
    AEB_STANDOFF_M,
    AEB_STOPPED_SPEED_MPS,
    AEB_TRIGGER_MARGIN,
)
from beamng_lidar_bev.models import ARMED, BRAKING, STANDBY, VehicleGeometry

# A small hatchback: 1.8 m wide, 2.0 m of nose ahead of the reference node.
GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.55,
    left_m=0.9,
    right_m=0.9,
    front_m=2.0,
    rear_m=2.4,
    height_m=1.45,
    mounts={},
)
STANDOFF_M = GEOMETRY.front_m + AEB_STANDOFF_M
DT = 0.04


def wall(distance_m: float, half_width_m: float = 1.5) -> np.ndarray:
    """A row of returns across the road at ``distance_m`` ahead."""
    lateral = np.linspace(-half_width_m, half_width_m, 15)
    return np.column_stack((lateral, np.full_like(lateral, distance_m)))


# Long enough to clear AEB_CONFIRM_S with a tick to spare, so tests that are
# about the brake law are not accidentally about the confirmation timer.
CONFIRM_TICKS = int(AEB_CONFIRM_S / DT) + 1


def _fires_at(speed_mps: float, inside_m: float = 1.0) -> float:
    """A threat distance comfortably inside the last point to brake."""
    return (
        STANDOFF_M
        + AEB_TRIGGER_MARGIN * stopping_distance(speed_mps)
        - inside_m
    )


def hold(
    brake: EmergencyBraking,
    cloud: np.ndarray,
    speed: float,
    ticks: int = CONFIRM_TICKS,
    curvature: float = 0.0,
    heading: float = 0.0,
):
    """
    Present the same scene for several ticks and return the final state.

    The heading advances at ``curvature`` throughout, because a step that
    repeats the previous heading reads as a straight moment and drags the yaw
    filter back toward zero -- which straightens the corridor under the test.
    """
    for _ in range(ticks):
        state = brake.step(cloud, GEOMETRY, speed, DT, heading_rad=heading)
        heading += curvature * speed * DT
    return state


def _prime_yaw(
    brake: EmergencyBraking, speed: float, curvature: float, ticks: int = 4
) -> float:
    """
    Drive an empty road at a constant curvature until the yaw filter has settled.

    Returns the heading for the caller's NEXT step. The cloud is empty on
    purpose: what is being set up is the path prediction, and presenting the
    obstacle before the filter has converged would just test the straight
    corridor.
    """
    hold(brake, np.empty((0, 2)), speed, ticks, curvature=curvature)
    return curvature * speed * DT * ticks


def test_a_clear_road_does_not_brake() -> None:
    brake = EmergencyBraking()

    state = brake.step(np.empty((0, 2)), GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.status == ARMED
    assert state.brake == 0.0
    assert state.threat_m is None


def test_a_wall_at_the_last_point_to_brake_fires() -> None:
    brake = EmergencyBraking()
    # Just inside the last point a full stop from 11 m/s still works.
    threat = STANDOFF_M + AEB_TRIGGER_MARGIN * stopping_distance(11.0) - 0.5

    state = hold(brake, wall(threat), 11.0)

    assert state.status == BRAKING
    assert state.brake == pytest.approx(1.0)
    assert state.threat_m == pytest.approx(threat, abs=0.2)
    assert state.time_to_collision_s == pytest.approx(
        (threat - STANDOFF_M) / 11.0, rel=0.05
    )


def test_the_same_wall_a_metre_further_out_does_not() -> None:
    """The whole point: it does nothing until a full stop is the last option."""
    brake = EmergencyBraking()
    threat = STANDOFF_M + AEB_TRIGGER_MARGIN * stopping_distance(11.0) + 1.0

    state = hold(brake, wall(threat), 11.0)

    assert state.status == ARMED
    assert state.brake == 0.0
    assert state.threat_m == pytest.approx(threat, abs=0.2)


def test_an_obstacle_beyond_the_scan_horizon_is_not_even_watched() -> None:
    """The corridor is only scanned as far as this speed could still need."""
    brake = EmergencyBraking()

    state = hold(brake, wall(34.0), 8.0)

    assert state.horizon_m < 34.0
    assert state.threat_m is None
    assert state.status == ARMED


@pytest.mark.parametrize("kph", (30, 50, 70, 100))
def test_it_brakes_full_whenever_it_brakes_at_all(kph: int) -> None:
    """
    Regression: the pedal used to be graded off the required deceleration, so
    at 50 km/h it fired 22.9 m out on 0.52 of brake and stopped with 7.2 m to
    spare -- early and gentle, which is a driver-assist rather than an
    emergency brake. The grading now lives in the trigger; by the time the
    pedal moves, nothing less than all of it works.
    """
    speed = kph / 3.6
    threat = STANDOFF_M + AEB_TRIGGER_MARGIN * stopping_distance(speed) - 0.5

    state = hold(EmergencyBraking(), wall(threat), speed)

    assert state.status == BRAKING
    assert state.brake == pytest.approx(1.0)


@pytest.mark.parametrize("kph", (30, 50, 70, 100))
def test_it_fires_late_enough_to_be_a_last_resort(kph: int) -> None:
    """
    It must not act while there is still comfortable room. Firing further out
    than ~1.5x the distance a full stop needs is nannying, not emergency
    braking.
    """
    speed = kph / 3.6
    generous = STANDOFF_M + 1.6 * stopping_distance(speed)

    state = hold(EmergencyBraking(), wall(generous), speed)

    assert state.status == ARMED
    assert state.brake == 0.0


@pytest.mark.parametrize("kph", (30, 50, 70, 100))
def test_it_fires_early_enough_to_actually_stop(kph: int) -> None:
    """
    The other side of the same coin, and the one that was reported: at 100 km/h
    it braked only 10 m before the wall because the corridor was capped at the
    planner's 35 m horizon while a stop from that speed needs 48 m. Whatever
    the speed, the trigger has to sit outside the real braking envelope.
    """
    speed = kph / 3.6
    # The wall is placed at the last point it should still act, and the
    # physical stop from there has to fit in the room left.
    state = hold(EmergencyBraking(), wall(80.0), speed, ticks=1)
    fires_at = state.brake_now_m

    assert fires_at <= state.horizon_m, "cannot fire on what it cannot see"
    room = fires_at - STANDOFF_M
    assert room >= speed * speed / (2.0 * AEB_BRAKING_DECEL_MPS2)


# Full-pedal braking distances measured on the vehicle in use. These are the
# ground truth AEB_BRAKING_DECEL_MPS2 is fitted to, so they belong in the suite:
# a plausible-looking edit to that constant is exactly the change that would
# make AEB fire too late to stop, and nothing else here would catch it.
MEASURED_BRAKING_M = {
    11: 0.40,
    20: 1.47,
    40: 6.01,
    60: 13.23,
    80: 24.22,
    100: 37.36,
    120: 55.04,
    160: 98.25,
    200: 160.00,
}


@pytest.mark.parametrize("kph, measured_m", sorted(MEASURED_BRAKING_M.items()))
def test_the_model_never_under_predicts_the_real_stopping_distance(
    kph: int, measured_m: float
) -> None:
    """
    `stopping_distance` has to be pessimistic against the car, or the trigger
    fires after the last point the stop could have worked.

    Compared at the trigger, margin included, and against the measured distance
    plus the same latency the model charges -- the measurements are pure
    braking distance with no reaction time in them (a least-squares fit puts the
    linear term at -0.14 s, i.e. zero).
    """
    speed = kph / 3.6
    fires_at = AEB_TRIGGER_MARGIN * stopping_distance(speed)
    really_needs = speed * AEB_LATENCY_S + measured_m

    assert fires_at > really_needs, (
        f"at {kph} km/h AEB commits {fires_at:.1f} m to a stop that really "
        f"takes {really_needs:.1f} m"
    )


def test_the_assumed_deceleration_is_not_optimistic_about_the_car() -> None:
    """The same check stated as the constant rather than as the distances."""
    worst = min(
        (kph / 3.6) ** 2 / (2.0 * metres)
        # 200 km/h excluded: AEB cannot stop from there whatever it assumes,
        # and the car itself falls off to 9.65 m/s^2 at that speed.
        for kph, metres in MEASURED_BRAKING_M.items()
        if kph <= 160
    )
    assert AEB_BRAKING_DECEL_MPS2 <= worst


def test_it_stays_inert_below_the_arming_speed() -> None:
    """Parking manoeuvres are a stream of near misses, not a stream of crashes."""
    brake = EmergencyBraking()

    state = hold(brake, wall(3.5), 1.0)

    assert state.status == STANDBY
    assert state.brake == 0.0


def test_reversing_never_arms_it() -> None:
    brake = EmergencyBraking()

    state = hold(brake, wall(5.0), -3.0)

    assert state.status == STANDBY
    assert state.brake == 0.0


def test_an_event_already_under_way_brakes_below_the_arming_speed() -> None:
    """The speed gate blocks arming only, or AEB would let go at 7 km/h."""
    brake = EmergencyBraking()
    hold(brake, wall(10.0), 11.0)

    state = brake.step(wall(4.0), GEOMETRY, 1.2, DT, heading_rad=0.0)

    assert state.status == BRAKING
    assert state.brake > 0.0


def test_stopped_against_the_obstacle_it_holds_rather_than_stands_on_the_pedal(
) -> None:
    brake = EmergencyBraking()
    hold(brake, wall(10.0), 11.0)

    state = brake.step(wall(3.6), GEOMETRY, 0.0, DT, heading_rad=0.0)

    assert state.status == BRAKING
    assert state.brake == pytest.approx(AEB_HOLD_BRAKE)


def test_the_brake_is_handed_back_the_moment_the_car_stops() -> None:
    """
    Reaching a standstill IS the objective, so the event ends when the car
    stops. There is no timed hold: it would leave a window in which neither the
    system nor the driver is clearly in charge of the pedal, and it never held
    the car against a gradient anyway.

    The wall is still a wall throughout, which is the point -- the release is
    the car having stopped, not the threat having gone.
    """
    brake = EmergencyBraking()
    hold(brake, wall(10.0), 11.0)

    # Still rolling, nose-to-nose, and for longer than AEB_MIN_ENGAGED_S so the
    # release below is the standstill rule rather than the blip guard expiring.
    # 0.2 m/s is inside STALL_SPEED_MPS, which is exactly the threshold it would
    # have been wrong to release on: 0.7 km/h is still moving.
    for _ in range(4):
        moving = brake.step(wall(3.5), GEOMETRY, 0.2, 0.1, heading_rad=0.0)
    assert moving.status == BRAKING
    assert moving.brake == pytest.approx(AEB_HOLD_BRAKE)

    stopped = brake.step(wall(3.5), GEOMETRY, 0.0, 0.1, heading_rad=0.0)

    # The pedal is up on the very tick the car reached rest -- no hold, and no
    # part of a second of one.
    assert stopped.status != BRAKING
    assert stopped.brake == 0.0

    # ...and it stays up with the wall still there. STANDBY rather than ARMED
    # because a stationary car is below the arming speed; the brake does not
    # come back until the car is moving at something again.
    settled = brake.step(wall(3.5), GEOMETRY, 0.0, 0.1, heading_rad=0.0)
    assert settled.status == STANDBY
    assert settled.brake == 0.0


def test_a_single_tick_blip_is_still_barred() -> None:
    """
    Releasing at rest must not become "release immediately". AEB_MIN_ENGAGED_S
    is a separate rule from the hold that was removed, and it is what stops a
    one-tick pulse: a car that is already stationary when the brake fires still
    holds it for the minimum rather than flickering.
    """
    brake = EmergencyBraking()
    state = hold(brake, wall(_fires_at(5.0)), 5.0)
    assert state.status == BRAKING

    elapsed = 0.0
    while elapsed < AEB_MIN_ENGAGED_S - 0.05:
        state = brake.step(
            wall(_fires_at(5.0)), GEOMETRY, 0.0, 0.05, heading_rad=0.0
        )
        elapsed += 0.05
        assert state.status == BRAKING, "a stop released inside the blip guard"


def test_an_emergency_stop_is_one_continuous_brake_application() -> None:
    """
    Regression, found by tracing a 50 km/h stop: `needed` falls as v^2 while
    the distance to the obstacle falls only as v, so partway through the stop
    the release ratio recovered even though the car was CLOSER than when it
    fired. The brake let go at 11.6 m and again at 6.4 m, delivering an
    emergency stop as three pulses. The release is now latched against the room
    the event started with, which braking cannot manufacture.
    """
    releases, stopped_at, _ = _close_on_a_wall(50.0 / 3.6)

    assert releases == 0, "the brake pulsed instead of stopping the car"
    assert stopped_at is not None, "the car never stopped"
    # Stopped short of the wall, with the bumper clear of it.
    assert stopped_at > GEOMETRY.front_m


def _close_on_a_wall(
    speed_mps: float, ticks: int = 3000
) -> tuple[int, float | None, float]:
    """
    Drive at a wall until stopped, returning ``(releases, gap left, peak pedal)``.

    The plant is the one AEB_BRAKING_DECEL_MPS2 promises -- deceleration
    proportional to the pedal -- which matters: modelling the brake as on/off
    ignores the AEB_HOLD_BRAKE tail and lets a stopped car creep forever.
    Latency is deliberately NOT modelled, so what this measures is the geometry
    of the trigger rather than the fidelity of the plant.
    """
    brake = EmergencyBraking()
    speed, travelled, wall_at = float(speed_mps), 0.0, 200.0
    releases, engaged, peak = 0, False, 0.0
    for _ in range(ticks):
        gap = wall_at - travelled
        cloud = wall(gap) if gap > 0.2 else np.empty((0, 2))
        state = brake.step(cloud, GEOMETRY, speed, DT, heading_rad=0.0)
        if state.status == BRAKING:
            engaged = True
        elif engaged and speed > 0.5:
            releases += 1
            engaged = False
        peak = max(peak, state.brake)
        speed = max(0.0, speed - AEB_BRAKING_DECEL_MPS2 * state.brake * DT)
        travelled += speed * DT
        # Stopped by the same definition AEB releases on. The plant models no
        # drag at all, so once the pedal comes up the car coasts forever at
        # whatever residual it had; asking for a tighter zero than the system
        # itself uses would be measuring the plant's missing friction.
        if speed <= AEB_STOPPED_SPEED_MPS:
            return releases, wall_at - travelled, peak
    return releases, None, peak


@pytest.mark.parametrize("kph", (30, 50, 70, 100))
def test_it_stops_the_car_clear_of_the_wall_at_every_speed(kph: int) -> None:
    """
    The end-to-end claim, closed loop. Both halves of the reported complaint
    live here: fire late and hard enough to be an emergency brake, but early
    enough to still work at motorway speed.
    """
    releases, stopped_at, peak = _close_on_a_wall(kph / 3.6)

    assert releases == 0
    assert peak == pytest.approx(1.0)
    assert stopped_at is not None, "the car never stopped"
    assert stopped_at > GEOMETRY.front_m, "the bumper reached the wall"


def test_a_single_tick_of_clear_road_does_not_release_it() -> None:
    brake = EmergencyBraking()
    hold(brake, wall(_fires_at(11.0)), 11.0)

    state = brake.step(np.empty((0, 2)), GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.status == BRAKING


def test_the_predicted_path_follows_the_measured_yaw() -> None:
    """
    A left-hand bend has to move the corridor left with the car, or AEB brakes
    for the outside kerb of every corner it takes.
    """
    brake = EmergencyBraking()
    heading = _prime_yaw(brake, 11.0, 0.05)  # a 20 m radius

    state = hold(
        brake, np.empty((0, 2)), 11.0, ticks=1, curvature=0.05, heading=heading
    )

    assert state.curvature == pytest.approx(0.05, rel=0.05)


def test_an_obstacle_straight_ahead_is_ignored_while_turning_away() -> None:
    brake = EmergencyBraking()
    heading = _prime_yaw(brake, 11.0, 0.05)

    state = hold(
        brake,
        # A wall straight ahead: at 14 m a 20 m-radius left-hander is already
        # 4.7 m to the left of it, well clear of the ~1.05 m corridor.
        wall(14.0),
        11.0,
        curvature=0.05,
        heading=heading,
    )

    assert state.threat_m is None
    assert state.status == ARMED


def test_an_obstacle_on_the_curved_path_is_caught() -> None:
    brake = EmergencyBraking()
    curvature = 0.05
    heading = _prime_yaw(brake, 11.0, curvature)
    # A wall across the arc 14 m along it: the arc's own point at that distance,
    # spread along the normal there.
    along = _fires_at(11.0)
    theta = curvature * along
    centre = np.asarray(
        ((math.cos(theta) - 1.0) / curvature, math.sin(theta) / curvature)
    )
    normal = np.asarray((math.cos(theta), math.sin(theta)))
    on_path = centre + np.linspace(-0.6, 0.6, 8)[:, None] * normal

    state = hold(brake, on_path, 11.0, curvature=curvature, heading=heading)

    assert state.threat_m == pytest.approx(along, abs=0.3)
    assert state.status == BRAKING


def test_a_gap_in_the_returns_cannot_arm_it() -> None:
    """An empty cloud looks exactly like a clear road; braking on it is noise."""
    brake = EmergencyBraking()

    state = brake.step(
        np.empty((0, 2)), GEOMETRY, 11.0, DT, heading_rad=0.0, has_returns=False
    )

    assert state.status == ARMED
    assert state.brake == 0.0


def test_a_gap_in_the_returns_cannot_disarm_it_either() -> None:
    """Releasing the pedal mid-event because a scan was missed is a crash."""
    brake = EmergencyBraking()
    hold(brake, wall(10.0), 11.0)

    state = brake.step(
        np.empty((0, 2)), GEOMETRY, 10.0, DT, heading_rad=0.0, has_returns=False
    )

    assert state.status == BRAKING
    assert state.brake > 0.0


def test_the_corridor_is_the_body_plus_a_slim_tolerance() -> None:
    """Wider than this and every kerb is a full brake application."""
    brake = EmergencyBraking()

    state = brake.step(np.empty((0, 2)), GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.corridor_half_width_m == pytest.approx(
        GEOMETRY.width_m / 2.0 + 0.15
    )
    assert state.standoff_m == pytest.approx(STANDOFF_M)


def test_a_kerb_beside_the_car_is_not_an_obstacle() -> None:
    kerb = np.column_stack(
        (np.full(40, 1.6), np.linspace(2.0, 30.0, 40))
    )
    brake = EmergencyBraking()

    state = hold(brake, kerb, 11.0)

    assert state.threat_m is None
    assert state.brake == 0.0


# --- no phantom braking ------------------------------------------------------
#
# AEB fires a full-authority brake, so a false positive costs far more here than
# anywhere else in the app. These pin the three filters that stand between a
# flat, empty road and the pedal.


@pytest.mark.parametrize("kph", (20, 40, 50, 64, 80, 110, 160))
def test_an_empty_map_never_brakes_at_any_speed(kph: int) -> None:
    """
    Regression, and the one that was actually reported: on flat gridmap with
    nothing anywhere near, AEB braked itself from 64 km/h down to 45, released,
    and did it again.

    A clear corridor was being scored as though an obstacle sat at the horizon,
    so the required deceleration on an empty road was
    v^2 / (2 * (horizon - standoff)). Below 40 km/h the horizon grows with
    speed and that lands at a harmless constant, but past the point where it
    clamps at PLANNER_HORIZON_M the ratio climbs with v^2 and crosses the
    trigger on its own -- no obstacle involved.
    """
    state = hold(EmergencyBraking(), np.empty((0, 2)), kph / 3.6, ticks=40)

    assert state.status == ARMED
    assert state.brake == 0.0
    assert state.threat_m is None
    assert state.required_decel_mps2 == 0.0


def test_a_clear_corridor_is_not_an_obstacle_at_the_horizon() -> None:
    """The same bug stated directly: nothing to hit means nothing to compute."""
    fast = hold(EmergencyBraking(), np.empty((0, 2)), 30.0, ticks=10)

    assert fast.threat_m is None
    assert fast.required_decel_mps2 == 0.0
    assert fast.time_to_collision_s == math.inf


def test_returns_outside_the_corridor_leave_it_clear_at_speed() -> None:
    """Walls either side of a wide open road, driven fast, are not a threat."""
    walls = np.concatenate(
        [
            np.column_stack((np.full(200, side * 4.0), np.linspace(0.0, 60.0, 200)))
            for side in (-1, 1)
        ]
    )

    state = hold(EmergencyBraking(), walls, 25.0, ticks=20)

    assert state.status == ARMED
    assert state.threat_m is None


def test_a_single_stray_return_in_the_corridor_is_not_a_wall() -> None:
    """
    One speck used to be a full brake application: the corridor scan is a
    nearest-return measurement, so a lone point ends it on its own. See
    AEB_MIN_HITS.
    """
    brake = EmergencyBraking()

    state = hold(brake, np.asarray(((0.1, 9.0),)), 11.0)

    assert state.threat_m is None
    assert state.status == ARMED
    assert state.brake == 0.0


def test_a_handful_of_specks_short_of_the_support_threshold_is_not_a_wall(
) -> None:
    specks = np.column_stack(
        (
            np.linspace(-0.4, 0.4, AEB_MIN_HITS - 1),
            np.linspace(8.0, 12.0, AEB_MIN_HITS - 1),
        )
    )
    brake = EmergencyBraking()

    state = hold(brake, specks, 11.0)

    assert state.threat_m is None
    assert state.brake == 0.0


def test_a_surface_at_the_support_threshold_is_a_wall() -> None:
    """The other side of the same rule: enough returns and it must brake."""
    surface = np.column_stack(
        (np.linspace(-0.3, 0.3, AEB_MIN_HITS), np.full(AEB_MIN_HITS, 9.0))
    )
    brake = EmergencyBraking()

    state = hold(brake, surface, 11.0)

    assert state.threat_m == pytest.approx(9.0, abs=0.2)
    assert state.status == BRAKING


def test_a_one_tick_phantom_never_reaches_the_pedal() -> None:
    """A phantom lasts a frame; a wall does not. See AEB_CONFIRM_S."""
    brake = EmergencyBraking()
    hold(brake, np.empty((0, 2)), 11.0)

    flash = brake.step(wall(8.0), GEOMETRY, 11.0, DT, heading_rad=0.0)
    after = hold(brake, np.empty((0, 2)), 11.0)

    assert flash.status == ARMED
    assert flash.brake == 0.0
    assert after.brake == 0.0


def test_the_confirmation_window_resets_on_any_tick_that_disagrees() -> None:
    brake = EmergencyBraking()
    clear = np.empty((0, 2))

    for _ in range(20):
        brake.step(wall(8.0), GEOMETRY, 11.0, DT, heading_rad=0.0)
        state = brake.step(clear, GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.status == ARMED
    assert state.brake == 0.0


def test_flat_road_returns_below_the_obstacle_floor_are_invisible() -> None:
    """
    The floor is applied by `planner.geometric_obstacles` before AEB ever sees
    the cloud, with AEB's own much higher AEB_OBSTACLE_MIN_HEIGHT_M -- so a road
    surface lifted into the planner's band by a brake dive never reaches here.
    This pins the pairing rather than re-testing the height band.
    """
    from beamng_lidar_bev.config import (
        AEB_OBSTACLE_MIN_HEIGHT_M,
        OBSTACLE_MIN_HEIGHT_M,
    )
    from beamng_lidar_bev.planner import geometric_obstacles

    # A flat road 15 m long, reading 0.18 m high: a plausible brake-dive
    # offset, and above the planner's 0.12 m floor.
    lateral, forward = np.meshgrid(
        np.linspace(-2.0, 2.0, 12), np.linspace(4.0, 15.0, 24)
    )
    road = np.column_stack((lateral.ravel(), forward.ravel()))
    heights = np.full(len(road), 0.18, dtype=np.float32)

    to_the_planner = geometric_obstacles(road, heights, 0.0)
    to_aeb = geometric_obstacles(
        road, heights, 0.0, min_height_m=AEB_OBSTACLE_MIN_HEIGHT_M
    )

    assert AEB_OBSTACLE_MIN_HEIGHT_M > OBSTACLE_MIN_HEIGHT_M
    assert len(to_the_planner) > 0  # the planner does see it...
    assert len(to_aeb) == 0  # ...and AEB does not

    assert hold(EmergencyBraking(), to_aeb, 11.0).brake == 0.0


# --- overlay geometry --------------------------------------------------------


def test_the_corridor_polygon_starts_at_the_ego_width() -> None:
    polygon = predicted_corridor(0.0, 20.0, 1.05, samples=8)

    assert len(polygon) == 16
    # First point is the right-hand edge at s = 0, last is the left-hand edge.
    assert polygon[0] == pytest.approx((1.05, 0.0))
    assert polygon[-1] == pytest.approx((-1.05, 0.0))


def test_the_corridor_bends_the_way_the_curvature_says() -> None:
    """Positive curvature turns LEFT, matching planner and the BEV overlay."""
    left = predicted_corridor(0.05, 14.0, 1.0, samples=4)

    # Midpoint of the far chord, which is the centreline at s = 14.
    far = (left[3] + left[4]) / 2.0
    assert far[0] < -3.0


def test_the_cross_section_spans_the_corridor_at_that_distance() -> None:
    ends = corridor_cross_section(0.0, 12.0, 1.05)

    assert ends[0] == pytest.approx((1.05, 12.0))
    assert ends[1] == pytest.approx((-1.05, 12.0))


def test_the_arming_speed_is_a_walking_pace_not_a_driving_one() -> None:
    assert 1.0 < AEB_MIN_SPEED_MPS < 4.0


# --- reverse ------------------------------------------------------------------
#
# The rear system is the SAME machine on a 180-degree-rotated cloud, so these
# pin the two things that are genuinely different: the mirroring, and the
# measured plant.

REAR_GEOMETRY = mirrored(GEOMETRY)
REAR_STANDOFF_M = GEOMETRY.rear_m + AEB_REVERSE_STANDOFF_M

# Full-pedal braking distances measured on the same vehicle, reversing.
MEASURED_REVERSE_BRAKING_M = {10: 0.56, 20: 2.06, 30: 4.46, 50: 13.21}


def rear(brake, cloud, speed, ticks=CONFIRM_TICKS, heading=0.0):
    """Reverse at ``speed`` m/s toward a cloud given in ordinary BEV metres."""
    for _ in range(ticks):
        state = brake.step(
            mirror_points(cloud), REAR_GEOMETRY, speed, DT, heading_rad=heading
        )
    return state


def test_mirroring_swaps_the_overhangs_and_keeps_the_width() -> None:
    assert REAR_GEOMETRY.front_m == GEOMETRY.rear_m
    assert REAR_GEOMETRY.rear_m == GEOMETRY.front_m
    assert REAR_GEOMETRY.width_m == pytest.approx(GEOMETRY.width_m)
    # A rotation, not a reflection: left and right swap together with front and
    # rear, so handedness survives and the curvature convention holds.
    assert REAR_GEOMETRY.left_m == GEOMETRY.right_m


def test_mirroring_puts_what_was_behind_in_front() -> None:
    behind = np.asarray(((1.5, -8.0), (-1.5, -8.0)))

    assert mirror_points(behind).tolist() == [[-1.5, 8.0], [1.5, 8.0]]
    assert mirror_points(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize(
    "kph, measured_m", sorted(MEASURED_REVERSE_BRAKING_M.items())
)
def test_the_reverse_model_never_under_predicts_the_real_stop(
    kph: int, measured_m: float
) -> None:
    """
    The car stops ~30% worse backwards -- braking in reverse loads the rear
    axle, which carries the smaller brakes. Sharing the forward figure would
    make the rear system fire far too late.
    """
    speed = kph / 3.6
    fires_at = AEB_TRIGGER_MARGIN * stopping_distance(speed, REVERSE)
    really_needs = speed * AEB_LATENCY_S + measured_m

    assert fires_at > really_needs


def test_the_reverse_plant_is_genuinely_worse_than_the_forward_one() -> None:
    worst = min(
        (kph / 3.6) ** 2 / (2.0 * metres)
        for kph, metres in MEASURED_REVERSE_BRAKING_M.items()
    )
    assert REVERSE.braking_decel_mps2 <= worst
    assert REVERSE.braking_decel_mps2 < FORWARD.braking_decel_mps2


def test_reversing_into_a_wall_brakes() -> None:
    brake = EmergencyBraking(REVERSE)
    speed = 20.0 / 3.6
    behind = (
        REAR_STANDOFF_M
        + AEB_TRIGGER_MARGIN * stopping_distance(speed, REVERSE)
        - 0.4
    )
    # Negated y: the wall is behind the car, in ordinary BEV coordinates.
    cloud = wall(behind) * np.asarray((1.0, -1.0))

    state = rear(brake, cloud, speed)

    assert state.status == BRAKING
    assert state.brake == pytest.approx(1.0)
    assert state.rearward is True
    assert state.threat_m == pytest.approx(behind, abs=0.2)


def test_the_rear_system_ignores_what_is_in_front() -> None:
    """Driving at a wall is the forward system's job, not this one's."""
    brake = EmergencyBraking(REVERSE)

    state = rear(brake, wall(4.0), 20.0 / 3.6)

    assert state.threat_m is None
    assert state.brake == 0.0


def test_the_rear_system_stands_by_while_driving_forwards() -> None:
    """It is fed the negated forward speed, so going forwards reads negative."""
    brake = EmergencyBraking(REVERSE)
    behind = wall(5.0) * np.asarray((1.0, -1.0))

    state = rear(brake, behind, -11.0)

    assert state.status == STANDBY
    assert state.brake == 0.0


def test_the_rear_system_is_awake_at_parking_speed() -> None:
    """
    Where the forward system deliberately stands by. Backing into a wall at a
    crawl is exactly what a reverse AEB is for, and the forward 7.2 km/h gate
    would sleep through all of it.
    """
    assert REVERSE.min_speed_mps < FORWARD.min_speed_mps

    brake = EmergencyBraking(REVERSE)
    cloud = wall(REAR_STANDOFF_M + 0.3) * np.asarray((1.0, -1.0))

    state = rear(brake, cloud, 1.2)

    assert state.status == BRAKING


# --- Plant measurement --------------------------------------------------------
#
# Diagnostics: `BrakeEvent` measures what the car actually delivered, and nothing
# reads the answer. It exists because every braking figure in config is a
# property of ONE vehicle and applying them to a heavier one is a question that
# needs a number rather than an opinion.


def _constant_decel_trace(
    event: BrakeEvent, from_speed: float, decel: float, dt: float = DT
) -> None:
    """Brake from `from_speed` to rest at exactly `decel`, one tick at a time."""
    speed = from_speed
    event.start(speed, 0.0, "MANUAL")
    while speed > 0.0:
        speed = max(0.0, speed - decel * dt)
        event.sample(speed, dt)


def test_a_brake_event_recovers_the_deceleration_the_car_delivered() -> None:
    """
    The whole instrument in one assertion. Fed a stop at a known rate it must
    report that rate, or it cannot be trusted to report an unknown one.
    """
    event = BrakeEvent(FORWARD)

    _constant_decel_trace(event, 25.0, 7.4)
    measurement = event.finish()

    assert measurement is not None
    assert measurement.mean_decel_mps2 == pytest.approx(7.4, rel=0.02)
    assert measurement.peak_decel_mps2 == pytest.approx(7.4, rel=0.02)
    assert measurement.distance_m == pytest.approx(25.0**2 / (2 * 7.4), rel=0.02)
    assert measurement.to_speed_mps == pytest.approx(0.0, abs=1e-6)


def test_a_brake_event_reports_what_the_model_promised_alongside_it() -> None:
    """
    The comparison is the point: a car that brakes worse than the configured
    plant takes further to stop than AEB assumed when it chose its trigger
    distance, and `optimism` is that ratio.

    The modelled distance is the PURE braking distance the config tables are
    made of, without `stopping_distance`'s latency term -- the two are different
    quantities and comparing a measured stop against the wrong one would read as
    a plant error that is not there.
    """
    event = BrakeEvent(FORWARD)

    # Half the configured deceleration: twice the distance, optimism 2.
    _constant_decel_trace(event, 20.0, AEB_BRAKING_DECEL_MPS2 / 2.0)
    measurement = event.finish()

    assert measurement is not None
    assert measurement.modelled_decel_mps2 == AEB_BRAKING_DECEL_MPS2
    assert measurement.modelled_distance_m == pytest.approx(
        20.0**2 / (2 * AEB_BRAKING_DECEL_MPS2)
    )
    assert measurement.optimism == pytest.approx(2.0, rel=0.02)
    assert measurement.modelled_distance_m < measurement.distance_m


def test_the_reverse_plant_is_measured_against_the_reverse_model() -> None:
    """Reversing loads the smaller brakes, so it has its own figure."""
    event = BrakeEvent(REVERSE)

    # Speed arrives signed from the worker; a stop is a stop either way.
    event.start(-8.0, 0.0, "MANUAL")
    for speed in (-6.0, -4.0, -2.0, 0.0):
        event.sample(speed, DT)
    measurement = event.finish()

    assert measurement is not None
    assert measurement.from_speed_mps == pytest.approx(8.0)
    assert measurement.modelled_decel_mps2 == REVERSE.braking_decel_mps2
    assert measurement.mean_decel_mps2 > 0.0


def test_an_event_with_nothing_to_measure_reports_nothing() -> None:
    """
    Rather than a deceleration divided by no elapsed time. A single tick has no
    trace, and a car that never lost any speed did not brake.
    """
    empty = BrakeEvent(FORWARD)
    empty.start(20.0, 0.0, "AEB")
    assert empty.finish() is None

    coasting = BrakeEvent(FORWARD)
    coasting.start(20.0, 0.0, "AEB")
    coasting.sample(20.0, DT)
    assert coasting.finish() is None

    assert BrakeEvent(FORWARD).finish() is None


def test_a_finished_event_goes_inactive_so_one_stop_is_reported_once() -> None:
    event = BrakeEvent(FORWARD)
    _constant_decel_trace(event, 15.0, 9.0)

    assert event.active
    assert event.finish() is not None
    assert not event.active
    assert event.finish() is None
