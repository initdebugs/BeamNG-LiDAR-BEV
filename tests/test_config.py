from __future__ import annotations

import math

from beamng_lidar_bev.config import (
    AEB_BRAKING_DECEL_MPS2,
    AEB_CLEARANCE_MARGIN_M,
    AEB_CONFIRM_S,
    AEB_HORIZON_MARGIN,
    AEB_LATENCY_S,
    AEB_MAX_HORIZON_M,
    AEB_MIN_HITS,
    AEB_MIN_HORIZON_M,
    AEB_OBSTACLE_MIN_HEIGHT_M,
    AEB_RELEASE_MARGIN,
    AEB_STANDOFF_M,
    AEB_TRIGGER_MARGIN,
    BRIDGE_PROBE_INTERVAL_MS,
    BRIDGE_PROBE_TIMEOUT_S,
    CLEARANCE_MARGIN_M,
    COMFORT_DECEL_MPS2,
    CONTROL_INTERVAL_MS,
    DISPLAY_INTERVAL_MS,
    GROUND_FALLBACK_ABOVE_M,
    HARD_DECEL_MPS2,
    LIDAR_DENSITY,
    LIDAR_FRONT_DENSITY,
    LIDAR_FRONT_HORIZONTAL_FOV_DEG,
    LIDAR_FRONT_MAX_DISTANCE_M,
    LIDAR_HORIZONTAL_FOV_DEG,
    LIDAR_MAX_DISTANCE_M,
    LIDAR_RANGE_M,
    LIDAR_VERTICAL_FOV_DEG,
    LIDAR_VERTICAL_RESOLUTION,
    LOOKAHEAD_MAX_M,
    LOOKAHEAD_MIN_M,
    MAX_SPEED_MPS,
    MIN_TURN_RADIUS_M,
    NAV_POLL_INTERVAL_MS,
    OBSTACLE_MAX_HEIGHT_M,
    OBSTACLE_MIN_HEIGHT_M,
    OBSTACLE_MIN_SUPPORT,
    PLANNER_ARC_SAMPLES,
    PLANNER_HORIZON_M,
    PLANNER_LOOKAHEAD_M,
    REQUIRED_FREE_DISTANCE_M,
    SENSOR_HEIGHT_ABOVE_GROUND_M,
    STOP_MARGIN_M,
    TRANSITION_DISTANCES_M,
)

# These pin decisions that were established by measuring against a live
# simulator. They are cheap guard rails against a plausible-looking edit.


def test_slant_range_exceeds_the_horizontal_cull_radius() -> None:
    # max_distance is a slant range from a sensor mounted up to ~2 m off the
    # reference node; LIDAR_RANGE_M is a horizontal cull from that node. Merging
    # them silently truncates the outer annulus. The cull is sized for the FRONT
    # unit, which is the only one that reaches that far -- the other three stop
    # at their own shorter range, which truncates nothing.
    assert LIDAR_FRONT_MAX_DISTANCE_M > LIDAR_RANGE_M
    assert LIDAR_FRONT_MAX_DISTANCE_M > LIDAR_MAX_DISTANCE_M


def test_horizontal_fov_stays_off_the_tan_cliff() -> None:
    # Measured: 179 deg returns 7389 points but only 3053 unique, against
    # 7826 / 5613 at 170 deg. Four wedges this wide still cover a full circle.
    assert 1.0 <= LIDAR_HORIZONTAL_FOV_DEG <= 175.0
    assert LIDAR_HORIZONTAL_FOV_DEG * 4 >= 360.0


def test_ground_reach_covers_the_display_radius() -> None:
    """
    The channel nearest horizontal sets how far the ground is sampled from a
    low mount. This is what keeps a 0.20 m kerb-height sensor useful at 100 m.
    """
    half_step = LIDAR_VERTICAL_FOV_DEG / (LIDAR_VERTICAL_RESOLUTION - 1) / 2.0
    reach = SENSOR_HEIGHT_ABOVE_GROUND_M / math.tan(math.radians(half_step))
    assert reach >= 90.0


def test_lowest_channel_still_sees_kerbs_beside_the_car() -> None:
    lowest = LIDAR_VERTICAL_FOV_DEG / 2.0
    nearest = SENSOR_HEIGHT_ABOVE_GROUND_M / math.tan(math.radians(lowest))
    assert nearest <= 1.0


def test_bridge_probes_cannot_overlap() -> None:
    assert BRIDGE_PROBE_TIMEOUT_S * 1000.0 < BRIDGE_PROBE_INTERVAL_MS


# --- self-driving ------------------------------------------------------------


def test_the_speed_cap_is_the_requested_40_kph() -> None:
    assert MAX_SPEED_MPS * 3.6 == 40.0


def test_the_lookahead_window_is_ordered_and_visible() -> None:
    assert LOOKAHEAD_MIN_M < LOOKAHEAD_MAX_M < PLANNER_HORIZON_M
    # PLANNER_LOOKAHEAD_M remains the static default inside the window.
    assert LOOKAHEAD_MIN_M <= PLANNER_LOOKAHEAD_M <= LOOKAHEAD_MAX_M


def test_transition_distances_cover_the_braking_envelope() -> None:
    distances = TRANSITION_DISTANCES_M
    assert distances[0] == 0.0  # today's fan is always a candidate family
    assert list(distances) == sorted(distances)
    # A turn deferred further than the car can brake toward is a turn the
    # speed law cannot prepare for.
    assert distances[-1] <= REQUIRED_FREE_DISTANCE_M


def test_the_braking_envelope_fits_inside_the_planning_horizon() -> None:
    """
    The planner has to be able to see far enough to stop from the speed cap, or
    it commits to arcs it cannot brake within.
    """
    assert REQUIRED_FREE_DISTANCE_M <= PLANNER_HORIZON_M
    assert REQUIRED_FREE_DISTANCE_M == (
        MAX_SPEED_MPS**2 / (2.0 * COMFORT_DECEL_MPS2) + STOP_MARGIN_M
    )


def test_the_planner_looks_no_further_than_it_can_see() -> None:
    assert PLANNER_LOOKAHEAD_M < PLANNER_HORIZON_M


def test_the_planning_horizon_stays_inside_the_lidar_cull_radius() -> None:
    assert PLANNER_HORIZON_M < LIDAR_RANGE_M


def test_a_straight_arc_is_always_a_candidate() -> None:
    # An even sample count would straddle zero curvature, so the car could never
    # choose to simply go straight.
    assert PLANNER_ARC_SAMPLES % 2 == 1


def test_the_obstacle_band_is_ordered_and_clears_the_roof() -> None:
    assert OBSTACLE_MIN_HEIGHT_M < OBSTACLE_MAX_HEIGHT_M
    # Above the ego roofline, so bridges and gantries are not walls.
    assert OBSTACLE_MAX_HEIGHT_M >= 2.0


def test_a_kerb_counts_as_an_obstacle_to_the_planner() -> None:
    # The display's ground band and the planner's obstacle floor must agree, or
    # a return can be drawn as road while being planned around, or vice versa.
    assert OBSTACLE_MIN_HEIGHT_M == GROUND_FALLBACK_ABOVE_M
    assert OBSTACLE_MIN_HEIGHT_M < SENSOR_HEIGHT_ABOVE_GROUND_M


def test_full_lock_is_a_plausible_car_turning_circle() -> None:
    assert 4.0 <= MIN_TURN_RADIUS_M <= 12.0


def test_the_navigation_route_is_read_far_slower_than_the_display_loop() -> None:
    # It is a blocking Lua round-trip and the route only changes when the player
    # sets a new destination.
    assert NAV_POLL_INTERVAL_MS >= DISPLAY_INTERVAL_MS * 10


def test_actuation_keeps_up_with_the_display_loop() -> None:
    assert CONTROL_INTERVAL_MS >= DISPLAY_INTERVAL_MS


# --- emergency braking --------------------------------------------------------


def test_aeb_brakes_far_harder_than_the_comfort_law_ever_asks_for() -> None:
    """
    An AEB that goes off while the ordinary brake law is still coping is one
    the user switches off. It intervenes at the last point a stop still works,
    and then uses an authority the comfort controller never reaches.
    """
    assert AEB_BRAKING_DECEL_MPS2 > HARD_DECEL_MPS2 > COMFORT_DECEL_MPS2


def test_the_trigger_is_a_last_resort_rather_than_a_safety_blanket() -> None:
    # It fires at AEB_TRIGGER_MARGIN times the distance a full stop needs.
    # Much above ~1.3 and it is braking while there is still comfortable room,
    # which is a driver-assist rather than an emergency brake; below 1.0 it
    # would be firing after the last moment the stop could have worked.
    assert 1.0 < AEB_TRIGGER_MARGIN <= 1.3
    assert AEB_RELEASE_MARGIN > AEB_TRIGGER_MARGIN


def test_the_latency_term_is_real_hardware_delay_not_padding() -> None:
    """
    Display tick + the blocking control ack + pad bite and weight transfer.

    AEB_CONFIRM_S is deliberately NOT in it -- including it was worth 1.4 m of
    unnecessarily early braking at 50 km/h, because the confirmation window
    elapses while the threat is still far away and adds no delay at the moment
    the trigger fires.
    """
    assert AEB_LATENCY_S >= 2.0 * DISPLAY_INTERVAL_MS / 1000.0
    assert AEB_LATENCY_S < AEB_CONFIRM_S + DISPLAY_INTERVAL_MS / 1000.0


def test_aeb_can_act_from_far_enough_for_any_speed_these_cars_reach() -> None:
    """
    The 70 m ceiling made 125 km/h look right for the wrong reason: it fired AT
    the horizon, which happened to land near the correct distance, rather than
    because the geometry said so. Anything the trigger can ask for has to be
    inside what the front sensor returns.
    """
    v = 170.0 / 3.6
    needed = v * AEB_LATENCY_S + v * v / (2.0 * AEB_BRAKING_DECEL_MPS2)
    assert AEB_MAX_HORIZON_M >= AEB_TRIGGER_MARGIN * needed
    # ...and inside the cull, which is inside the front unit's slant range.
    assert AEB_MAX_HORIZON_M < LIDAR_RANGE_M < LIDAR_FRONT_MAX_DISTANCE_M


def test_the_front_sensor_trades_sweep_for_azimuth_density() -> None:
    """
    Narrowing is what makes long range work: at the shared 170 deg sweep the
    azimuth spacing is ~3.5 deg, which is 9.3 m between ray columns at 150 m,
    against a 2 m-wide car. Vertical spacing over the same cloud is 0.118 deg,
    thirty times finer, so azimuth is the whole bottleneck.
    """
    assert LIDAR_FRONT_HORIZONTAL_FOV_DEG < LIDAR_HORIZONTAL_FOV_DEG
    assert LIDAR_FRONT_DENSITY < LIDAR_DENSITY  # lower divisor = more rays
    # Still wide enough to cover the wedge the left and right units leave: each
    # sweeps 170 deg about its own axis, reaching to within 5 deg of dead ahead.
    assert LIDAR_FRONT_HORIZONTAL_FOV_DEG >= 2.0 * (
        90.0 - LIDAR_HORIZONTAL_FOV_DEG / 2.0
    )


def test_the_aeb_corridor_is_tighter_than_the_planners() -> None:
    # The planner's margin buys comfortable paths; this one answers "will I
    # actually hit it", and every extra centimetre is another kerb that fires.
    assert AEB_CLEARANCE_MARGIN_M < CLEARANCE_MARGIN_M


def test_aeb_ignores_what_the_planner_merely_steers_around() -> None:
    """
    The floors answer different questions. AEB's is high enough that the road
    surface cannot reach it under a brake dive, which is where phantom braking
    comes from -- planner.ground_rise cannot absorb that inside
    SLOPE_ALLOWANCE_START_M, and the near field is where AEB works.
    """
    assert AEB_OBSTACLE_MIN_HEIGHT_M > OBSTACLE_MIN_HEIGHT_M
    assert AEB_OBSTACLE_MIN_HEIGHT_M >= 0.25
    # ...and still far below anything worth braking for.
    assert AEB_OBSTACLE_MIN_HEIGHT_M < OBSTACLE_MAX_HEIGHT_M / 2.0


def test_an_obstacle_is_seen_well_before_the_trigger_can_fire() -> None:
    """
    The confirmation window counts how long a threat has been in the corridor,
    so the corridor has to reach past the distance the trigger fires from or
    the window would elapse only after the situation had become critical --
    delaying every real brake by AEB_CONFIRM_S.
    """
    # Both are multiples of the same stopping distance, so the lead is a plain
    # ratio and holds at every speed the horizon is not clamped at.
    assert AEB_HORIZON_MARGIN > AEB_TRIGGER_MARGIN
    assert AEB_HORIZON_MARGIN >= AEB_RELEASE_MARGIN


def test_aeb_can_see_far_enough_to_stop_from_motorway_speed() -> None:
    """
    Regression, reported: at 100 km/h AEB braked only 10 m before the wall. It
    was sharing the planner's 35 m horizon while a stop from that speed needs
    48 m, so the brake could not come on until 13 m after the last point it
    would have worked.
    """
    for kph in (50, 70, 100):
        v = kph / 3.6
        needed = v * AEB_LATENCY_S + v * v / (2.0 * AEB_BRAKING_DECEL_MPS2)
        assert AEB_MAX_HORIZON_M >= AEB_TRIGGER_MARGIN * needed + STOP_MARGIN_M
    # ...but still inside what the sensors actually return.
    assert AEB_MAX_HORIZON_M < LIDAR_RANGE_M
    assert AEB_MAX_HORIZON_M > PLANNER_HORIZON_M


def test_the_confirmation_window_costs_less_than_a_car_length() -> None:
    # Long enough to reject a one-frame phantom, short enough that a real
    # obstacle appearing inside it was never avoidable anyway.
    assert AEB_CONFIRM_S >= DISPLAY_INTERVAL_MS / 1000.0 * 2
    assert AEB_CONFIRM_S * MAX_SPEED_MPS < 2.0


def test_a_threat_has_to_be_a_surface_rather_than_a_speck() -> None:
    # The corridor scan is a nearest-return measurement, so one stray point
    # ends it -- with a full brake application on the end of it.
    assert AEB_MIN_HITS > OBSTACLE_MIN_SUPPORT


def test_aeb_never_reaches_past_what_the_obstacle_band_can_be_trusted_for(
) -> None:
    assert AEB_MIN_HORIZON_M < PLANNER_HORIZON_M
    # Stopping short of the bumper, not of the reference node the free distance
    # is measured from -- the geometry's front overhang is added on top.
    assert 0.5 <= AEB_STANDOFF_M <= STOP_MARGIN_M
