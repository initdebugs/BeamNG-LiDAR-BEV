from __future__ import annotations

import time

import numpy as np
import pytest

from beamng_lidar_bev import planner
from beamng_lidar_bev.config import (
    AEB_MIN_VERTICAL_EXTENT_M,
    AEB_OBSTACLE_MIN_HEIGHT_M,
    CLEARANCE_MARGIN_M,
    KEEP_RIGHT_MARGIN_M,
    MIN_TURN_RADIUS_M,
    OBSTACLE_CELL_M,
    OBSTACLE_MAX_HEIGHT_M,
    OBSTACLE_MIN_HEIGHT_M,
    PLANNER_HORIZON_M,
    PLANNER_MAX_OBSTACLE_POINTS,
)
from beamng_lidar_bev.models import RoadGrid, RoutePath, VehicleGeometry
from beamng_lidar_bev.planner import (
    ObstacleBand,
    arc_polyline,
    corridor_edges,
    corridor_return_profile,
    despeckle,
    geometric_obstacle_sets,
    geometric_obstacles,
    path_polyline,
    plan_arc,
    rear_free_distance,
)

GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.5,
    left_m=0.9,
    right_m=0.9,
    front_m=2.0,
    rear_m=2.4,
    height_m=1.5,
    mounts={},
)
NO_OBSTACLES = np.empty((0, 2), dtype=np.float32)


def _wall(right_from: float, right_to: float, forward_m: float) -> np.ndarray:
    """A dense lateral line of returns at a fixed forward distance."""
    xs = np.arange(right_from, right_to, 0.1, dtype=np.float32)
    return np.column_stack((xs, np.full_like(xs, forward_m)))


def _rail(right_m: float, forward_from: float, forward_to: float) -> np.ndarray:
    """A dense longitudinal line of returns at a fixed lateral offset."""
    ys = np.arange(forward_from, forward_to, 0.1, dtype=np.float32)
    return np.column_stack((np.full_like(ys, right_m), ys))


def _patch(right_m: float, forward_m: float, count: int = 4) -> np.ndarray:
    """
    A small cluster of returns, as any real surface produces.

    Isolated single returns are rejected by `despeckle`, so a test about the
    height band has to present something the height band can actually be asked
    about. Four sensors at 256 channels put far more than this on a kerb face.
    """
    step = 0.05  # well inside one despeckle cell
    return np.column_stack(
        (
            np.full(count, right_m, dtype=np.float32),
            forward_m + np.arange(count, dtype=np.float32) * step,
        )
    )


# --- geometric_obstacles -----------------------------------------------------


def test_ground_returns_are_not_obstacles() -> None:
    bev = np.asarray(((0.0, 10.0), (2.0, 8.0)), dtype=np.float32)
    heights = np.asarray(
        (GEOMETRY.ground_z_vehicle, GEOMETRY.ground_z_vehicle + 0.02),
        dtype=np.float32,
    )

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert obstacles.shape == (0, 2)


def test_a_kerb_height_return_is_an_obstacle() -> None:
    bev = _patch(3.0, 6.0)
    heights = np.full(len(bev), GEOMETRY.ground_z_vehicle + 0.15, dtype=np.float32)

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert len(obstacles) == len(bev)


def test_an_overhead_gantry_is_not_an_obstacle() -> None:
    """Regression guard: without a ceiling, every bridge reads as a wall."""
    bev = np.asarray(((0.0, 20.0),), dtype=np.float32)
    heights = np.asarray(
        (GEOMETRY.ground_z_vehicle + OBSTACLE_MAX_HEIGHT_M + 1.0,), dtype=np.float32
    )

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert obstacles.shape == (0, 2)


def test_the_obstacle_set_is_capped_for_the_arc_scan() -> None:
    """The scan is an (obstacles x arcs) matrix, so the row count is bounded."""
    count = PLANNER_MAX_OBSTACLE_POINTS * 3
    bev = np.column_stack(
        (
            np.linspace(-8.0, 8.0, count, dtype=np.float32),
            np.full(count, 10.0, dtype=np.float32),
        )
    )
    heights = np.full(count, GEOMETRY.ground_z_vehicle + 0.8, dtype=np.float32)

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert len(obstacles) == PLANNER_MAX_OBSTACLE_POINTS


def test_returns_beyond_the_planning_horizon_are_dropped() -> None:
    bev = np.asarray(((0.0, PLANNER_HORIZON_M + 5.0),), dtype=np.float32)
    heights = np.asarray((GEOMETRY.ground_z_vehicle + 0.8,), dtype=np.float32)

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert obstacles.shape == (0, 2)


def _ground_disc(grade: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    A dense drivable surface out to the horizon, optionally rising with range.

    Dense enough that `ground_rise` has the samples per ring it needs; without
    a surface to measure, the estimate abstains and the old cone stands.
    """
    radii = np.arange(1.0, 36.0, 0.25)
    angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    r = np.repeat(radii, len(angles))
    a = np.tile(angles, len(radii))
    bev = np.column_stack((r * np.sin(a), r * np.cos(a))).astype(np.float32)
    heights = (GEOMETRY.ground_z_vehicle + grade * r).astype(np.float32)
    return bev, heights


def test_a_kerb_stays_visible_to_the_horizon_on_flat_ground() -> None:
    """
    Regression pin for the blindness that made the car run wide and block.

    The slope cone alone put the obstacle floor at 0.27 m by 20 m and 0.50 m by
    35 m, so a 0.15 m kerb stopped being an obstacle beyond about 12 m -- 1.1 s
    of road-edge information at the speed cap. Where the ground is measurably
    flat the floor must stay at OBSTACLE_MIN_HEIGHT_M all the way out.
    """
    ground_bev, ground_heights = _ground_disc()
    kerb = _patch(3.0, 30.0)
    bev = np.concatenate((ground_bev, kerb))
    heights = np.concatenate(
        (
            ground_heights,
            np.full(len(kerb), GEOMETRY.ground_z_vehicle + 0.15, dtype=np.float32),
        )
    )

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert len(obstacles) == len(kerb)
    assert obstacles[:, 1].min() == pytest.approx(30.0)


def test_a_measured_rise_is_terrain_not_a_wall() -> None:
    """The protection the cone existed for, now driven by the measurement."""
    grade = 0.008  # 0.24 m of rise by 30 m, inside the cone that bounds it
    bev, heights = _ground_disc(grade=grade)

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert obstacles.shape == (0, 2)


def test_the_ground_estimate_can_never_read_below_the_ego_plane() -> None:
    """
    A ditch beside the road must not drag the floor under the road surface --
    that would turn the tarmac itself into a wall in front of the car.
    """
    bev, heights = _ground_disc()
    ditch = np.abs(bev[:, 0]) > 6.0
    heights = heights.copy()
    heights[ditch] -= 1.5

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    # The road surface either side of the ditch stays drivable.
    assert not np.any(np.abs(obstacles[:, 0]) <= 6.0)


def test_an_isolated_return_is_not_an_obstacle() -> None:
    """
    Regression pin. The arc scan takes the NEAREST blocking point, so a lone
    spurious return ends that arc by itself: measured, one stray point at 10 m
    took a clear road's free distance from 33.2 m to 10.0 m, which the speed
    law turns into a full brake application for nothing.
    """
    bev = np.asarray(((0.4, 10.0),), dtype=np.float32)
    heights = np.asarray((GEOMETRY.ground_z_vehicle + 0.8,), dtype=np.float32)

    obstacles = geometric_obstacles(bev, heights, GEOMETRY.ground_z_vehicle)

    assert obstacles.shape == (0, 2)


def test_despeckling_keeps_neighbours_across_adjacent_cells() -> None:
    """Support is a 3x3 cell box, so a pair astride a cell edge still counts."""
    pair = np.asarray(
        ((0.0, 10.0), (0.0, 10.0 + OBSTACLE_CELL_M)), dtype=np.float32
    )

    assert len(despeckle(pair)) == 2
    assert len(despeckle(np.asarray(((0.0, 10.0),), dtype=np.float32))) == 0


def test_a_stray_return_no_longer_collapses_the_free_distance() -> None:
    """The whole point of the filter, measured end to end through plan_arc."""
    road = np.concatenate((_rail(-2.6, 2.0, 35.0), _rail(2.6, 2.0, 35.0)))
    speck = np.concatenate((road, np.asarray(((0.4, 10.0),), dtype=np.float32)))
    heights = np.full(len(speck), GEOMETRY.ground_z_vehicle + 0.8, dtype=np.float32)

    obstacles = geometric_obstacles(speck, heights, GEOMETRY.ground_z_vehicle)
    plan = plan_arc(obstacles, GEOMETRY, lookahead_m=25.0)

    assert plan.free_distance_m > 30.0


# --- plan_arc ----------------------------------------------------------------


def test_open_road_runs_the_free_distance_out_to_the_horizon() -> None:
    plan = plan_arc(NO_OBSTACLES, GEOMETRY, nav_heading_rad=None)

    assert plan.free_distance_m == pytest.approx(PLANNER_HORIZON_M)
    assert plan.curvature == pytest.approx(0.0, abs=1e-6)


def test_a_wall_ahead_shortens_the_straight_arc_free_distance() -> None:
    obstacles = _wall(-6.0, 6.0, 15.0)

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    straight = np.argmin(np.abs(plan.candidate_curvatures))
    assert plan.candidate_free_distances[straight] == pytest.approx(15.0, abs=0.6)


def test_steers_towards_the_gap_in_a_wall() -> None:
    """A wall spanning everything except a gap on the left must be steered into."""
    obstacles = np.concatenate((_wall(-12.0, -4.0, 14.0), _wall(0.0, 12.0, 14.0)))

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    # The path may legitimately hold course briefly before turning in, but it
    # must bend left through the gap and clear the wall.
    assert plan.next_curvature > 0.0  # positive curvature is left
    assert plan.free_distance_m > 14.0


def test_a_deferred_gap_plan_commits_to_the_turn_in_time() -> None:
    """
    Deferral must be time-consistent: re-planning while the car advances has
    to resolve into an immediate left command well before the wall, not keep
    promising a turn that never comes.
    """
    obstacles = np.concatenate((_wall(-12.0, -4.0, 14.0), _wall(0.0, 12.0, 14.0)))
    previous = 0.0
    committed_at = None
    for advance in range(10):  # metre steps while the command stays straight
        shifted = obstacles - (0.0, float(advance))
        plan = plan_arc(shifted, GEOMETRY, previous_curvature=previous)
        previous = plan.curvature
        if plan.curvature > 0.005:
            committed_at = 14.0 - advance
            break

    assert committed_at is not None  # it turned at all
    assert committed_at > 6.0  # and with room to spare, not at the wall


def test_a_blocked_road_reports_a_short_free_distance() -> None:
    obstacles = _wall(-40.0, 40.0, 5.0)

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    assert plan.free_distance_m < 6.0


def test_curvature_never_exceeds_the_minimum_turn_radius() -> None:
    obstacles = _wall(-40.0, 40.0, 8.0)

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    assert abs(plan.curvature) <= 1.0 / MIN_TURN_RADIUS_M + 1e-9


def test_an_empty_cloud_is_safe() -> None:
    plan = plan_arc(NO_OBSTACLES, GEOMETRY, nav_heading_rad=None)

    assert plan.candidate_curvatures.shape == plan.candidate_costs.shape
    assert plan.keep_right_target_m is None
    assert np.isfinite(plan.free_distance_m)


def test_the_nav_hint_decides_between_two_equally_open_directions() -> None:
    """Open ground is geometrically symmetric, so the turn hint alone decides."""
    left = plan_arc(NO_OBSTACLES, GEOMETRY, nav_heading_rad=0.6)
    right = plan_arc(NO_OBSTACLES, GEOMETRY, nav_heading_rad=-0.6)

    assert left.curvature > 0.0
    assert right.curvature < 0.0
    assert left.nav_heading_rad == pytest.approx(0.6)


def test_the_nav_hint_never_overrides_a_blocked_arc() -> None:
    """LiDAR wins: a wall to the left is not driven into because nav says left."""
    obstacles = _rail(-2.6, 2.0, PLANNER_HORIZON_M)

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=1.0)

    assert plan.free_distance_m > 20.0


def test_smoothness_penalty_favours_holding_the_previous_curvature() -> None:
    obstacles = NO_OBSTACLES

    held = plan_arc(
        obstacles, GEOMETRY, nav_heading_rad=0.25, previous_curvature=0.10
    )
    fresh = plan_arc(
        obstacles, GEOMETRY, nav_heading_rad=0.25, previous_curvature=-0.10
    )

    assert held.curvature > fresh.curvature


# --- the route reference path ------------------------------------------------


def _route(
    points: list[tuple[float, float]],
    half_width: float = 3.0,
    speed_limit: float = 11.0,
    cross_track: float = 0.0,
) -> RoutePath:
    """A RoutePath built straight from BEV points, the way plan_arc sees it."""
    pts = np.asarray(points, dtype=np.float64)
    chords = np.hypot(*np.diff(pts, axis=0).T)
    arc = np.concatenate(([0.0], np.cumsum(chords)))
    dx, dy = np.diff(pts[:, 0]), np.diff(pts[:, 1])
    seg_theta = np.unwrap(np.arctan2(-dx, dy))
    theta = np.empty(len(pts))
    theta[0] = seg_theta[0]
    theta[-1] = seg_theta[-1]
    if len(pts) > 2:
        theta[1:-1] = 0.5 * (seg_theta[:-1] + seg_theta[1:])
    return RoutePath(
        points=pts.astype(np.float32),
        arc_s=arc,
        headings=theta,
        curvatures=np.gradient(theta, arc),
        half_width_m=np.full(len(pts), half_width),
        junction_turn=np.zeros(len(pts), dtype=bool),
        remaining_m=float(arc[-1]),
        cross_track_m=cross_track,
        speed_limit_mps=speed_limit,
    )


def test_the_route_terms_pull_the_car_toward_the_reference_path() -> None:
    """A route running parallel 2 m to the LEFT: open ground, so the
    cross-track term alone must bend the plan toward the lane."""
    offset_route = _route([(-2.0, 0.0), (-2.0, 60.0)])

    plan = plan_arc(NO_OBSTACLES, GEOMETRY, route=offset_route)

    assert plan.curvature > 0.001  # positive is left, toward the path


def test_the_route_lane_target_sits_right_of_the_centreline() -> None:
    """On a two-way road the reference is the RIGHT lane, not the centreline
    the navgraph actually encodes."""
    centred_route = _route([(0.0, 0.0), (0.0, 60.0)])

    plan = plan_arc(NO_OBSTACLES, GEOMETRY, route=centred_route)

    assert plan.curvature < -0.001  # negative is right, into the lane


def test_a_narrow_route_road_centres_instead_of_hugging_the_kerb() -> None:
    narrow_route = _route([(0.0, 0.0), (0.0, 60.0)], half_width=1.2)

    plan = plan_arc(NO_OBSTACLES, GEOMETRY, route=narrow_route)

    assert plan.curvature == pytest.approx(0.0, abs=1e-3)


def test_the_route_never_overrides_a_blocked_arc() -> None:
    """LiDAR wins: a wall to the left is not driven into because the route
    bends left -- the mirror of the nav-hint pin above."""
    obstacles = _rail(-2.6, 2.0, PLANNER_HORIZON_M)
    left_route = _route([(0.0, 0.0), (-1.0, 10.0), (-4.0, 20.0), (-9.0, 30.0)])

    plan = plan_arc(obstacles, GEOMETRY, route=left_route)

    assert plan.free_distance_m > 20.0


def test_a_route_disables_the_legacy_terms() -> None:
    """The gate is structural: with a route, a stray bearing hint changes
    nothing, because two lateral guidance sources must never fight."""
    route = _route([(0.0, 0.0), (0.0, 60.0)])
    kerbed = np.concatenate(
        (_rail(-2.6, 2.0, PLANNER_HORIZON_M), _rail(2.6, 2.0, PLANNER_HORIZON_M))
    )

    bare = plan_arc(kerbed, GEOMETRY, route=route)
    with_hint = plan_arc(kerbed, GEOMETRY, nav_heading_rad=1.0, route=route)

    assert with_hint.curvature == bare.curvature
    assert with_hint.keep_right_target_m is None
    np.testing.assert_array_equal(with_hint.candidate_costs, bare.candidate_costs)


def test_route_terms_carry_no_deferral_discount() -> None:
    """A straight route on open ground scores every family identically, so
    the transition tie-break must still resolve to acting now."""
    plan = plan_arc(NO_OBSTACLES, GEOMETRY, route=_route([(0.0, 0.0), (0.0, 60.0)]))

    assert plan.transition_distance_m == 0.0


def test_a_route_plan_reports_its_route_fields() -> None:
    route = _route([(0.0, 0.0), (0.0, 60.0)], speed_limit=6.5, cross_track=0.8)

    routed = plan_arc(NO_OBSTACLES, GEOMETRY, route=route)
    unrouted = plan_arc(NO_OBSTACLES, GEOMETRY)

    assert routed.route_speed_limit_mps == pytest.approx(6.5)
    assert routed.route_cross_track_m == pytest.approx(0.8)
    assert routed.route_heading_rad == pytest.approx(0.0, abs=1e-9)
    assert unrouted.route_speed_limit_mps is None
    assert unrouted.route_cross_track_m is None
    assert unrouted.route_heading_rad is None


def test_keep_right_can_be_disabled_for_the_reverse_planner() -> None:
    """Reversing wants clearance and free distance only."""
    kerbed = np.concatenate(
        (_rail(-2.6, 2.0, PLANNER_HORIZON_M), _rail(2.6, 2.0, PLANNER_HORIZON_M))
    )

    with_term = plan_arc(kerbed, GEOMETRY)
    without = plan_arc(kerbed, GEOMETRY, keep_right=False)

    assert with_term.keep_right_target_m is not None
    assert without.keep_right_target_m is None


# --- the road-coverage bonus -------------------------------------------------


def _road_strip(right_from: float, right_to: float) -> RoadGrid:
    """A straight road strip, occupancy 1 between the two lateral bounds."""
    occupancy = np.zeros((40, 40), dtype=np.uint8)
    col_from = int((right_from + 16.0) / 0.8)
    col_to = int((right_to + 16.0) / 0.8)
    occupancy[:, col_from:col_to] = 1
    return RoadGrid(
        occupancy=occupancy,
        cell_m=0.8,
        origin_right_m=-16.0,
        origin_forward_m=0.0,
    )


def test_the_road_bonus_pulls_the_car_onto_the_tarmac() -> None:
    """A kerbless annotated road beside the car: nothing geometric prefers
    it, so only the bonus can."""
    beside = _road_strip(1.0, 9.0)

    drawn = plan_arc(NO_OBSTACLES, GEOMETRY, road_grid=beside)
    baseline = plan_arc(NO_OBSTACLES, GEOMETRY)

    assert baseline.curvature == pytest.approx(0.0, abs=1e-6)
    assert drawn.curvature < -0.001  # bends right, onto the road


def test_the_road_bonus_never_outbids_free_distance() -> None:
    """A blocked road arc loses to open grass: geometry keeps deciding what
    is drivable, semantics only make the tarmac preferable."""
    road = _road_strip(-3.0, 3.0)
    wall = _wall(-4.0, 4.0, 12.0)  # blocks the road, open ground beyond

    plan = plan_arc(wall, GEOMETRY, road_grid=road)

    assert plan.free_distance_m > 20.0  # went around, off the road


def test_uniform_road_coverage_shifts_the_fan_without_reordering_it() -> None:
    """Full coverage everywhere must change no decision: the shift is equal
    across the candidates whose samples stay inside the grid, and the tie
    still resolves to the straight immediate arc."""
    everywhere = RoadGrid(
        occupancy=np.ones((40, 40), dtype=np.uint8),
        cell_m=0.8,
        origin_right_m=-16.0,
        origin_forward_m=0.0,
    )

    covered = plan_arc(NO_OBSTACLES, GEOMETRY, road_grid=everywhere)
    bare = plan_arc(NO_OBSTACLES, GEOMETRY)

    delta = bare.candidate_costs - covered.candidate_costs
    centre = delta[8:-8]  # full-lock arcs curl out of the grid, honestly
    assert np.allclose(centre, centre[0], atol=1e-5)
    assert covered.transition_distance_m == 0.0
    assert covered.curvature == pytest.approx(0.0, abs=1e-6)


def test_no_road_grid_is_bit_identical_to_before() -> None:
    kerbed = np.concatenate(
        (_rail(-2.6, 2.0, PLANNER_HORIZON_M), _rail(2.6, 2.0, PLANNER_HORIZON_M))
    )

    first = plan_arc(kerbed, GEOMETRY, road_grid=None)
    second = plan_arc(kerbed, GEOMETRY)

    np.testing.assert_array_equal(first.candidate_costs, second.candidate_costs)
    assert first.curvature == second.curvature


# --- two-segment candidates --------------------------------------------------


def _bend_corridor() -> np.ndarray:
    """A straight that flows into a tight left bend of ~14 m radius."""
    parts = [_rail(-3.0, 1.0, 8.0), _rail(3.0, 1.0, 8.0)]
    centre = np.asarray((-14.0, 8.0))
    angles = np.arange(0.0, 1.3, 0.01)
    for radius in (11.0, 17.0):  # inner and outer kerbs of the bend
        arc = centre + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        parts.append(arc.astype(np.float32))
    return np.concatenate(parts)


def test_a_bend_ahead_is_planned_as_hold_then_turn() -> None:
    """
    Turning into the bend immediately clips the inner kerb and going straight
    runs into the outer one, so the winner must hold course and bend later --
    which is exactly what corner-entry braking needs to know about.
    """
    plan = plan_arc(_bend_corridor(), GEOMETRY, previous_curvature=0.0)

    assert plan.transition_distance_m > 0.0
    assert plan.next_curvature > 0.03  # the bend goes left, tightly
    assert plan.curvature == pytest.approx(0.0, abs=1e-6)  # hold course now
    # Going straight is blocked by the outer kerb at ~16 m; the composite
    # path sweeps well past it.
    assert plan.free_distance_m > 30.0


def test_the_chosen_path_clears_every_obstacle() -> None:
    rng = np.random.default_rng(7)
    obstacles = np.column_stack(
        (rng.uniform(-10.0, 10.0, 60), rng.uniform(2.0, 30.0, 60))
    ).astype(np.float32)

    plan = plan_arc(obstacles, GEOMETRY, previous_curvature=0.05)

    half_width = GEOMETRY.width_m / 2.0 + CLEARANCE_MARGIN_M
    # Stop half_width short of the free distance: the obstacle that ENDS the
    # free distance sits at that arc length, and sampling right up to it would
    # measure the straight-ahead gap to the path's endpoint, not clearance.
    driven = max(plan.free_distance_m - half_width, 0.1)
    path = path_polyline(
        plan.curvature,
        plan.transition_distance_m,
        plan.next_curvature,
        driven,
        samples=96,
    )
    gaps = np.hypot(
        obstacles[:, 0:1] - path[:, 0][None, :],
        obstacles[:, 1:2] - path[:, 1][None, :],
    ).min(axis=1)
    # 96 samples leave ~0.2 m chord gaps at most; 0.25 m covers it.
    assert gaps.min() >= half_width - 0.25


def test_path_polyline_reduces_to_a_single_arc() -> None:
    np.testing.assert_allclose(
        path_polyline(0.1, 0.0, 0.1, 15.0, samples=12),
        arc_polyline(0.1, 15.0, samples=12),
        atol=1e-5,
    )


def test_path_polyline_straight_then_turn_offsets_the_turn() -> None:
    path = path_polyline(0.0, 10.0, 0.1, 20.0, samples=64)

    early = path[path[:, 1] <= 9.9]
    np.testing.assert_allclose(early[:, 0], 0.0, atol=1e-6)  # straight prefix
    assert path[-1, 0] < -1.0  # the turn bends left after the prefix
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert float(segments.sum()) == pytest.approx(20.0, rel=5e-3)


def test_curvature_interpolates_between_fan_steps() -> None:
    # A slightly offset obstacle should move the answer by less than one fan
    # step, which is only possible with sub-step interpolation. Both offsets
    # keep the clearance term unsaturated so the optima genuinely differ.
    base = np.asarray(((1.55, 12.0),), dtype=np.float32)
    nudged = np.asarray(((1.70, 12.0),), dtype=np.float32)

    plan = plan_arc(base, GEOMETRY)
    coarsest_step = float(np.diff(plan.candidate_curvatures).max())
    delta = abs(plan.next_curvature - plan_arc(nudged, GEOMETRY).next_curvature)
    assert 0.0 < delta < coarsest_step


# --- corridor_edges / keep right ---------------------------------------------


def test_corridor_edges_finds_both_kerbs() -> None:
    obstacles = np.concatenate((_rail(-4.0, 2.0, 20.0), _rail(3.0, 2.0, 20.0)))

    left_edge, right_edge = corridor_edges(obstacles, lookahead_m=12.0)

    assert left_edge == pytest.approx(-4.0, abs=0.05)
    assert right_edge == pytest.approx(3.0, abs=0.05)


def test_keep_right_target_sits_inside_the_right_kerb() -> None:
    obstacles = np.concatenate((_rail(-4.0, 2.0, 20.0), _rail(3.0, 2.0, 20.0)))

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    assert plan.keep_right_target_m == pytest.approx(
        3.0 - (GEOMETRY.width_m / 2.0 + KEEP_RIGHT_MARGIN_M), abs=0.05
    )


def test_keep_right_is_dropped_on_an_open_surface() -> None:
    """No right edge within range is a car park, not a road. Do not guess one."""
    obstacles = _rail(-4.0, 2.0, 20.0)

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    assert plan.keep_right_target_m is None


def test_keep_right_biases_the_chosen_arc_to_the_right_of_centre() -> None:
    obstacles = np.concatenate((_rail(-5.0, 2.0, 25.0), _rail(5.0, 2.0, 25.0)))

    plan = plan_arc(obstacles, GEOMETRY, nav_heading_rad=None)

    assert plan.curvature < 0.0  # negative curvature is right


# --- rear_free_distance ------------------------------------------------------


def test_rear_free_distance_is_the_horizon_on_open_ground() -> None:
    assert rear_free_distance(NO_OBSTACLES, GEOMETRY) == pytest.approx(
        PLANNER_HORIZON_M
    )


def test_rear_free_distance_measures_back_to_the_nearest_obstacle() -> None:
    obstacles = _wall(-3.0, 3.0, -9.0)

    assert rear_free_distance(obstacles, GEOMETRY) == pytest.approx(9.0, abs=0.2)


def test_obstacles_ahead_do_not_limit_the_rear_free_distance() -> None:
    obstacles = _wall(-3.0, 3.0, 6.0)

    assert rear_free_distance(obstacles, GEOMETRY) == pytest.approx(
        PLANNER_HORIZON_M
    )


def test_obstacles_beside_the_car_do_not_limit_the_rear_free_distance() -> None:
    obstacles = _rail(4.0, -20.0, 20.0)

    assert rear_free_distance(obstacles, GEOMETRY) == pytest.approx(
        PLANNER_HORIZON_M
    )


# --- arc_polyline ------------------------------------------------------------


def test_a_straight_arc_polyline_runs_along_the_forward_axis() -> None:
    points = arc_polyline(0.0, 10.0, samples=5)

    np.testing.assert_allclose(points[:, 0], 0.0, atol=1e-6)
    assert points[-1, 1] == pytest.approx(10.0)


def test_a_left_arc_polyline_bends_to_negative_right() -> None:
    points = arc_polyline(1.0 / 10.0, 10.0, samples=9)

    assert points[-1, 0] < 0.0
    assert points[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert points[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_arc_polyline_length_matches_the_requested_arc_length() -> None:
    points = arc_polyline(1.0 / 8.0, 12.0, samples=200)

    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert float(segments.sum()) == pytest.approx(12.0, rel=1e-3)


# --- Diagnostics --------------------------------------------------------------


def _graded_road(grade: float, ground_z: float) -> tuple[np.ndarray, np.ndarray]:
    """A road surface climbing at `grade`, sampled the way ground rings arrive."""
    xs = np.arange(-4.0, 4.0, 0.25, dtype=np.float32)
    ys = np.arange(4.0, 40.0, 0.25, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    bev = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    heights = (ground_z + grade * bev[:, 1]).astype(np.float32)
    return bev, heights


def test_the_corridor_profile_separates_a_graded_road_from_a_wall() -> None:
    """
    "AEB braked and there was nothing there" cannot be answered from the
    trigger's own numbers: the corridor scan reduces a surface to one distance,
    and a distance cannot say whether that surface was a wall or the road
    arriving in the height band because the slope allowance refused to believe
    the grade. The VERTICAL EXTENT can, and it is invariant to both.
    """
    ground_z = -0.5
    road, road_heights = _graded_road(0.05, ground_z)

    # A wall standing on that same road, at the same range.
    wall_xs = np.arange(-1.5, 1.5, 0.1, dtype=np.float32)
    wall_zs = np.arange(0.0, 3.0, 0.1, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(wall_xs, wall_zs)
    wall = np.column_stack(
        (grid_x.ravel(), np.full(grid_x.size, 25.0, dtype=np.float32))
    ).astype(np.float32)
    wall_heights = (ground_z + 0.05 * 25.0 + grid_z.ravel()).astype(np.float32)

    open_road = corridor_return_profile(
        road, road_heights, ground_z, 0.0, 1.05, 25.0
    )
    with_wall = corridor_return_profile(
        np.concatenate((road, wall)),
        np.concatenate((road_heights, wall_heights)),
        ground_z,
        0.0,
        1.05,
        25.0,
    )

    # The hillside is well into AEB's obstacle band -- 1.25 m above the ego
    # plane at 25 m -- so the height test alone cannot tell it from the wall.
    assert open_road["height_median_m"] > 1.0
    assert open_road["spread_m"] < 0.15
    # ...and it climbs at the grade, over metres of range rather than none.
    assert open_road["range_span_m"] > 1.0
    assert open_road["spread_m"] / open_road["range_span_m"] == pytest.approx(
        0.05, abs=0.02
    )

    assert with_wall["spread_m"] > 2.0
    assert with_wall["spread_m"] > 10.0 * open_road["spread_m"]


def test_the_corridor_profile_reports_the_rise_the_slope_cone_threw_away() -> None:
    """
    The other half of the answer. `geometric_obstacle_sets` clamps the measured
    ground rise into a 1.5% cone, so on any real grade the estimate and the
    figure actually used part company -- and that gap is the difference between
    an obstacle and a hill.
    """
    ground_z = -0.5
    road, heights = _graded_road(0.05, ground_z)

    profile = corridor_return_profile(road, heights, ground_z, 0.0, 1.05, 25.0)

    assert profile["ground_rise_m"] > 1.0
    assert profile["cone_bound_m"] < 0.3
    assert profile["cone_bound_m"] < profile["ground_rise_m"]


def test_the_corridor_profile_is_empty_when_nothing_is_in_the_corridor() -> None:
    beside = np.asarray(((8.0, 25.0), (8.2, 25.1)), dtype=np.float32)
    heights = np.asarray((0.4, 0.4), dtype=np.float32)

    profile = corridor_return_profile(beside, heights, -0.5, 0.0, 1.05, 25.0)

    assert profile["count"] == 0.0
    assert profile["spread_m"] == 0.0


# --- AEB's two shape tests ----------------------------------------------------
#
# The height floor answers "how high above the estimated ground is this return",
# and on a grade the road itself climbs through any fixed value of it. These two
# answer "what SHAPE is the thing standing here" instead, which neither a
# gradient, a brake dive nor suspension travel can fake.

ROOF_H = 1.58
AEB_BAND = ObstacleBand(
    AEB_OBSTACLE_MIN_HEIGHT_M,
    60.0,
    min_vertical_extent_m=AEB_MIN_VERTICAL_EXTENT_M,
    porosity=True,
)
PLANNER_BAND = ObstacleBand(OBSTACLE_MIN_HEIGHT_M, PLANNER_HORIZON_M)


def _ground(
    grade: float, ground_z: float, to_m: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """A surface climbing at `grade`, sampled densely enough to be meshable."""
    xs = np.arange(-6.0, 6.0, 0.25, dtype=np.float32)
    ys = np.arange(3.0, to_m, 0.25, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    bev = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    heights = (ground_z + grade * bev[:, 1]).astype(np.float32)
    return bev, heights


def _standing(
    at_m: float, height_m: float, ground_z: float, grade: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A solid vertical face: one range, returns all the way up it."""
    xs = np.arange(-1.2, 1.2, 0.08, dtype=np.float32)
    zs = np.arange(0.0, height_m, 0.05, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    bev = np.column_stack(
        (grid_x.ravel(), np.full(grid_x.size, at_m, dtype=np.float32))
    ).astype(np.float32)
    heights = (ground_z + grade * at_m + grid_z.ravel()).astype(np.float32)
    return bev, heights


def _bush(
    at_m: float, height_m: float, ground_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Foliage: the same vertical extent as a post, but rays go through it."""
    xs = np.arange(-0.5, 0.5, 0.05, dtype=np.float32)
    zs = np.arange(0.0, height_m, 0.04, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    bev = np.column_stack(
        (grid_x.ravel(), np.full(grid_x.size, at_m, dtype=np.float32))
    ).astype(np.float32)
    return bev, (ground_z + grid_z.ravel()).astype(np.float32)


def _occlude(
    bev: np.ndarray,
    heights: np.ndarray,
    at_m: float,
    height_m: float,
    half_width_m: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delete the ground a SOLID object of this height would have hidden.

    The simulator's LiDAR is a real ray cast, so an opaque object genuinely
    shadows the ground behind it for r*a/(h - a). A synthetic scene that draws
    the surface straight through that shadow is describing something the rays
    could see past -- which is a bush, and the porosity test is right to say so.
    """
    shadow = at_m * height_m / max(ROOF_H - height_m, 1e-6)
    hidden = (
        (bev[:, 1] > at_m)
        & (bev[:, 1] < at_m + shadow)
        & (np.abs(bev[:, 0]) < half_width_m)
    )
    return bev[~hidden], heights[~hidden]


def _aeb_obstacles(
    bev: np.ndarray, heights: np.ndarray, ground_z: float
) -> np.ndarray:
    return geometric_obstacle_sets(
        bev, heights, ground_z, (AEB_BAND,), sensor_height_m=ROOF_H
    )[0]


@pytest.mark.parametrize("grade", (0.0, 0.05, 0.10, 0.20))
def test_a_hill_is_never_an_obstacle_however_steep(grade: float) -> None:
    """
    The point-5 phantom, and it is structural rather than a tuning slip. The
    local ground estimate is clamped into a 1.5% cone to protect the planner's
    kerb detection, so on a 5% climb at 25 m the road sits 1.25 m above the ego
    plane against 0.53 m of allowance -- a dense, persistent surface right in
    AEB's obstacle band, which sails through the height, support and persistence
    filters alike and reads as a wall across the road.
    """
    ground_z = -0.5
    bev, heights = _ground(grade, ground_z)

    assert not len(_aeb_obstacles(bev, heights, ground_z))


@pytest.mark.parametrize("grade", (0.0, 0.05, 0.10, 0.20))
def test_a_wall_on_that_same_hill_is_still_an_obstacle(grade: float) -> None:
    """The other half, and the half that matters: AEB must still fire."""
    ground_z = -0.5
    road, road_h = _ground(grade, ground_z)
    wall, wall_h = _standing(30.0, 3.0, ground_z, grade)

    obstacles = _aeb_obstacles(
        np.concatenate((road, wall)),
        np.concatenate((road_h, wall_h)),
        ground_z,
    )

    assert len(obstacles)
    assert obstacles[:, 1].min() == pytest.approx(30.0, abs=0.5)


def test_reversing_up_a_ramp_is_not_an_obstacle() -> None:
    """
    The case the slope cone can never reach. It is zero inside
    SLOPE_ALLOWANCE_START_M and the whole rear system lives at 2-8 m, so every
    driveway ramp was a phantom by construction.
    """
    ground_z = -0.5
    xs = np.arange(-4.0, 4.0, 0.2, dtype=np.float32)
    ys = np.arange(2.0, 9.0, 0.2, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    bev = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    heights = (ground_z + 0.08 * bev[:, 1]).astype(np.float32)

    assert not len(_aeb_obstacles(bev, heights, ground_z))


def test_a_short_solid_object_survives_because_extent_is_measured_whole() -> None:
    """
    A 0.35 m rock puts only 0.05 m of returns above the 0.30 m floor. Measuring
    the spread over the SURVIVORS would call that 0.05 m tall and delete it --
    and with it every kerbstone, bollard and low post there is. The extent is
    measured over the whole cell, floor-rejected returns included.
    """
    ground_z = -0.5
    road, road_h = _occlude(*_ground(0.0, ground_z), 20.0, 0.35)
    rock, rock_h = _standing(20.0, 0.35, ground_z)

    obstacles = _aeb_obstacles(
        np.concatenate((road, rock)),
        np.concatenate((road_h, rock_h)),
        ground_z,
    )

    assert len(obstacles)
    assert obstacles[:, 1].min() == pytest.approx(20.0, abs=0.5)


def test_a_bush_is_seen_through_and_a_post_of_the_same_height_is_not() -> None:
    """
    The bush complaint, decided on geometry alone: the ground BEHIND a bush
    comes back because rays pass between the leaves, and behind a solid object
    of the same height it does not. The window is the shadow a solid twin would
    cast, r*a/(h - a), so the two clouds below differ in exactly one thing --
    whether ground returns exist inside that shadow -- and get opposite verdicts.
    """
    ground_z, at_m, height = -0.5, 12.0, 0.6
    shadow = at_m * height / (ROOF_H - height)
    assert shadow > 5.0, "the test is meaningless if the shadow is short"

    foliage, foliage_h = _bush(at_m, height, ground_z)
    open_ground, open_h = _ground(0.0, ground_z)
    # The same scene with the shadow actually empty, as a solid object leaves it.
    shaded, shaded_h = _occlude(open_ground, open_h, at_m, height)

    porous = _aeb_obstacles(
        np.concatenate((open_ground, foliage)),
        np.concatenate((open_h, foliage_h)),
        ground_z,
    )
    solid = _aeb_obstacles(
        np.concatenate((shaded, foliage)),
        np.concatenate((shaded_h, foliage_h)),
        ground_z,
    )

    assert not len(porous), "a bush braked the car"
    assert len(solid), "a solid post of the same height was ignored"


def test_a_wall_can_never_be_vetoed_by_the_porosity_test() -> None:
    """
    The safety property, and it is derived rather than imposed: an object at or
    above the sensor height shadows the ground for ever, so its evidence window
    is EMPTY and no quantity of returns beyond it can clear it. Without that,
    parallax between five mount positions could talk AEB out of braking for a
    wall. Here the ground is drawn straight through the wall to force the issue.
    """
    ground_z = -0.5
    through, through_h = _ground(0.0, ground_z)
    wall, wall_h = _standing(15.0, 3.0, ground_z)

    obstacles = _aeb_obstacles(
        np.concatenate((through, wall)),
        np.concatenate((through_h, wall_h)),
        ground_z,
    )

    assert len(obstacles)
    assert obstacles[:, 1].min() == pytest.approx(15.0, abs=0.5)


def test_the_planner_still_steers_around_what_aeb_now_ignores() -> None:
    """
    The two bands answer different questions and only AEB's got pickier. The
    planner has to keep seeing the bush: steering around one is free, and
    braking for it is not.
    """
    ground_z = -0.5
    road, road_h = _ground(0.0, ground_z)
    foliage, foliage_h = _bush(12.0, 0.6, ground_z)

    planner_set, aeb_set = geometric_obstacle_sets(
        np.concatenate((road, foliage)),
        np.concatenate((road_h, foliage_h)),
        ground_z,
        (PLANNER_BAND, AEB_BAND),
        sensor_height_m=ROOF_H,
    )

    assert len(planner_set), "the planner went blind to the bush"
    assert not len(aeb_set)


def _rings(
    horizon_m: float, ground_z: float, az_deg: float = 0.26
) -> tuple[np.ndarray, np.ndarray]:
    """Ground as the sensors lay it down -- concentric rings, not a grid.

    Uniform noise is the wrong shape for any of the porosity arithmetic: the
    evidence is ground returns at a RANGE beyond a candidate and at the same
    BEARING, and rings and columns are what decide whether any exist.
    """
    parts = []
    for height, near, far in ((ROOF_H, 6.0, 80.0), (0.20, 3.0, 11.6)):
        for theta in np.linspace(
            np.arctan(height / far), np.arctan(height / near), 256
        ):
            radius = height / np.tan(theta)
            if radius < 2.0 or radius > horizon_m:
                continue
            bearings = np.arange(
                np.radians(-25.0), np.radians(25.0), np.radians(az_deg)
            )
            parts.append(
                np.column_stack(
                    (radius * np.sin(bearings), radius * np.cos(bearings))
                )
            )
    bev = np.concatenate(parts).astype(np.float32)
    return bev, np.full(len(bev), ground_z, dtype=np.float32)


def _shadow_wedge(
    bev: np.ndarray,
    heights: np.ndarray,
    at_m: float,
    height_m: float,
    half_width_m: float,
    offset_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Delete the ground a solid object of this size really hides.

    The umbra of an object seen from a point sensor is the angular WEDGE it
    subtends, which widens with range. `_occlude` uses a constant-width lateral
    strip, which is narrower than the truth at range and so leaves ground
    visible that a real ray cast would never have returned.
    """
    bearing = np.arctan2(bev[:, 0], bev[:, 1])
    hidden = (
        (bev[:, 1] > at_m)
        & (bev[:, 1] < at_m + at_m * height_m / max(ROOF_H - height_m, 1e-6))
        & (bearing > np.arctan2(offset_x - half_width_m, at_m))
        & (bearing < np.arctan2(offset_x + half_width_m, at_m))
    )
    return bev[~hidden], heights[~hidden]


def _car(
    at_m: float, ground_z: float, offset_x: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """The back of a stopped car, sampled at the sensors' angular resolution.

    A real target thins with range -- 0.118 deg between channels vertically and
    0.26 deg between the front unit's columns -- which is the whole point: a
    dense synthetic face hides every failure that only appears once an object
    stops filling the structures that measure it.
    """
    zs = np.arange(0.25, 1.45, at_m * np.radians(0.118), dtype=np.float32)
    xs = np.arange(
        offset_x - 0.9, offset_x + 0.9, at_m * np.radians(0.26), dtype=np.float32
    )
    grid_x, grid_z = np.meshgrid(xs, zs)
    bev = np.column_stack(
        (grid_x.ravel(), np.full(grid_x.size, at_m, dtype=np.float32))
    ).astype(np.float32)
    return bev, (ground_z + grid_z.ravel()).astype(np.float32)


@pytest.mark.parametrize("offset_x", (0.0, 0.35, 0.7))
@pytest.mark.parametrize("at_m", (15.0, 20.0, 30.0, 40.0, 50.0, 60.0))
def test_a_stopped_car_survives_both_shape_tests_at_every_firing_range(
    at_m: float, offset_x: float
) -> None:
    """
    The regression that made AEB brake late, and it is two defects at once.

    Both live in the porosity window, and both come from treating a fixed ANGLE
    as though it described an object of fixed WIDTH:

    - the (bin, range) key was not clamped to its bin, so a shadow longer than
      the key stride ran into the NEXT bin and counted the road in FRONT of the
      car as proof of seeing through it. A car's shadow is over 3x its range, so
      this fired for every car past ~15 m;
    - the bin outgrew the car with range, so the road BESIDE it -- which no part
      of it ever stood in front of -- became evidence against it.

    Together they made AEB blind to a stopped car beyond about 10 m. The ranges
    here are the ones the trigger actually fires from: 20 m at 60 km/h, 33 at
    80, 49 at 100. The lateral offsets matter because a fixed bin grid makes the
    verdict depend on where the car happens to sit against it, which is not a
    property any of this may have.
    """
    ground_z = -0.5
    horizon = at_m + 30.0
    road, road_h = _shadow_wedge(
        *_rings(horizon, ground_z), at_m, 1.45, 0.9, offset_x
    )
    car, car_h = _car(at_m, ground_z, offset_x)

    obstacles = geometric_obstacle_sets(
        np.concatenate((road, car)),
        np.concatenate((road_h, car_h)),
        ground_z,
        (ObstacleBand(
            AEB_OBSTACLE_MIN_HEIGHT_M,
            horizon,
            min_vertical_extent_m=AEB_MIN_VERTICAL_EXTENT_M,
            porosity=True,
        ),),
        sensor_height_m=ROOF_H,
    )[0]

    assert len(obstacles), f"AEB went blind to a car at {at_m:.0f} m"
    assert obstacles[:, 1].min() == pytest.approx(at_m, abs=0.5)


def _sensor_cloud(horizon_m: float, ground_z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    A street as the sensors actually lay it down, for the timing bound.

    Ground arrives as RINGS whose spacing goes as r^2/h, not as uniform noise --
    which matters, because the shape tests are per-cell and a cloud with one
    return per cell is the worst case for them and nothing like a real scene.
    """
    parts = []
    radius = 3.0
    while radius < horizon_m:
        count = max(24, int(2.0 * radius / 0.25))
        bearing = np.linspace(-1.4, 1.4, count)
        parts.append(
            np.column_stack(
                (
                    radius * np.sin(bearing),
                    radius * np.cos(bearing),
                    np.full(count, ground_z),
                )
            )
        )
        radius += max(0.05, (radius * radius / 1.58) * np.radians(0.118))
    for side in (-6.0, 6.0):
        ys = np.arange(5.0, horizon_m, 0.6)
        zs = np.arange(ground_z, ground_z + 4.5, 0.05)
        grid_y, grid_z = np.meshgrid(ys, zs)
        parts.append(
            np.column_stack(
                (
                    np.full(grid_y.size, side),
                    grid_y.ravel(),
                    grid_z.ravel(),
                )
            )
        )
    cloud = np.concatenate(parts)
    return cloud[:, :2].astype(np.float32), cloud[:, 2].astype(np.float32)


def test_the_shape_tests_stay_inside_the_tick() -> None:
    """
    Both are O(cloud) and ride the 40 ms display loop, next to a blocking
    `poll_sensors` round trip that already costs most of it.
    """
    ground_z = -0.5
    horizon = 90.0  # ~125 km/h, the furthest AEB ever scans in practice
    bev, heights = _sensor_cloud(horizon, ground_z)
    band = ObstacleBand(
        AEB_OBSTACLE_MIN_HEIGHT_M,
        horizon,
        min_vertical_extent_m=AEB_MIN_VERTICAL_EXTENT_M,
        porosity=True,
    )

    started = time.perf_counter()
    for _ in range(5):
        geometric_obstacle_sets(
            bev, heights, ground_z, (PLANNER_BAND, band), sensor_height_m=ROOF_H
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / 5

    assert len(bev) > 40_000, "the timing scene stopped being a worst case"
    assert elapsed_ms < 18.0, f"both bands cost {elapsed_ms:.1f} ms of a 40 ms tick"


# --- the planner's band is cell-referenced, and what that costs and buys ------
#
# The slope cone bounds the ground estimate at SLOPE_ALLOWANCE_PER_M = 1.5%/m,
# so on any road steeper than 1.5% the surface climbed into the planner's own
# obstacle band. That is "it brakes for hills", and because the controller
# blocks at STOP_MARGIN_M and then reverses, it is most of "it keeps reversing"
# too. The band is now referenced to each cell's own base, exactly as AEB's is.

_PLANNER_CELL_BAND = ObstacleBand(
    OBSTACLE_MIN_HEIGHT_M,
    PLANNER_HORIZON_M,
    cell_referenced=True,
    reduce_to_cells=True,
)


def _stripe_sampled_scene(
    grade: float = 0.0,
    *,
    soffit_at: float | None = None,
    wall_at: float | None = None,
    kerb_at: float | None = None,
    kerb_height: float = 0.12,
    ground_z: float = -0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ground laid down the way the sensors really lay it down.

    Rings are dense radially and azimuth STRIPES are `r * 2.45 deg` apart --
    0.81 m at 19 m, twice `OBSTACLE_CELL_M`. That anisotropy is not a detail
    here: it is the whole reason a cell can hold an overhead surface and no
    ground return, which is what the coarse ceiling exists for.
    """
    points: list[tuple[float, float, float]] = []
    for forward in np.arange(1.0, 36.0, 0.12):
        step = max(0.05, float(np.radians(2.45) * forward))
        for lateral in np.arange(-5.0, 5.01, step):
            points.append((float(lateral), float(forward), grade * float(forward)))
    if soffit_at is not None:
        for forward in np.arange(soffit_at, soffit_at + 1.0, 0.04):
            for lateral in np.arange(-5.0, 5.01, 0.04):
                for height in np.arange(2.4, 3.0, 0.05):
                    points.append(
                        (float(lateral), float(forward), grade * forward + height)
                    )
    if wall_at is not None:
        # Wider than the ground on purpose. The fan can reach 14 m of lateral
        # offset at a 30 m lookahead, so a wall that stopped where the sampled
        # ground stops would leave "drive off the edge of the test scene" as a
        # free path, and the scene rather than the planner would be what the
        # assertion measured.
        for forward in np.arange(wall_at, wall_at + 0.3, 0.05):
            for lateral in np.arange(-20.0, 20.01, 0.05):
                for height in np.arange(0.0, 2.5, 0.05):
                    points.append(
                        (float(lateral), float(forward), grade * forward + height)
                    )
    if kerb_at is not None:
        for forward in np.arange(1.0, 36.0, 0.06):
            for side in (-1.0, 1.0):
                for height in np.arange(0.0, kerb_height + 1e-9, 0.02):
                    points.append(
                        (
                            side * kerb_at,
                            float(forward),
                            grade * forward + float(height),
                        )
                    )
    array = np.asarray(points)
    return (
        array[:, :2].astype(np.float32),
        (array[:, 2] + ground_z).astype(np.float32),
    )


def _planner_free(bev: np.ndarray, heights: np.ndarray) -> float:
    obstacles = geometric_obstacle_sets(
        bev, heights, -0.30, (_PLANNER_CELL_BAND,)
    )[0]
    return float(plan_arc(obstacles, GEOMETRY).free_distance_m)


@pytest.mark.parametrize("grade", [0.0, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30])
def test_an_empty_graded_road_is_not_an_obstacle_at_any_ordinary_grade(
    grade: float,
) -> None:
    """
    The headline hill fix. Against the slope cone this collapsed at a 2% grade
    -- free distance 35.0 m to 6.0 m, and by 3% to exactly STOP_MARGIN_M, which
    is the blocked entry and then the reverse recovery.
    """
    bev, heights = _stripe_sampled_scene(grade)

    assert _planner_free(bev, heights) == pytest.approx(PLANNER_HORIZON_M)


def test_a_kerb_still_blocks_a_swerve_into_it_on_a_hill() -> None:
    """
    Grade-proofing must not cost kerb detection -- the kerb face is the only
    thing keeping the car on the tarmac on an unannotated map. It does not: the
    cone had itself begun swallowing kerbs on grades, so this improves.
    """
    for grade in (0.0, 0.05, 0.12):
        bev, heights = _stripe_sampled_scene(grade, kerb_at=3.5)
        obstacles = geometric_obstacle_sets(
            bev, heights, -0.30, (_PLANNER_CELL_BAND,)
        )[0]
        near_kerb = np.count_nonzero(
            np.abs(np.abs(obstacles[:, 0]) - 3.5) < 0.3
        )
        assert near_kerb > 20, f"the kerb vanished at a {grade:.0%} grade"


def test_the_car_drives_under_a_bridge_rather_than_braking_for_the_soffit() -> None:
    """
    The cost of cell-referencing, and what the COARSE ceiling pays it with. A
    fine cell need not contain the ground under it -- azimuth stripes are 0.81 m
    apart at 19 m against a 0.4 m cell -- so referencing the ceiling to the fine
    base makes a soffit its own floor and reads 2.4-3.0 m of bridge as 0.6 m of
    solid wall. Measured before the coarse reference: free distance 19.0 m.
    """
    for grade in (0.0, 0.08):
        bev, heights = _stripe_sampled_scene(grade, soffit_at=19.0)

        assert _planner_free(bev, heights) == pytest.approx(PLANNER_HORIZON_M), (
            f"braked for an overhead bridge at a {grade:.0%} grade"
        )


def test_a_wall_still_blocks_on_a_steep_hill() -> None:
    """The other side of the coarse ceiling: it must not let a real wall through
    just because the hill it stands on is high above the ego's own plane."""
    for grade in (0.0, 0.20):
        bev, heights = _stripe_sampled_scene(grade, wall_at=19.0)

        free = _planner_free(bev, heights)
        assert free < 20.0, f"missed a wall at a {grade:.0%} grade (free {free})"


def test_the_planner_drives_through_a_bush_and_round_a_post() -> None:
    """
    Roadside vegetation is a thing to IGNORE, not a thing to steer around. The
    planner's band was deliberately built without the porosity veto on the
    reasoning that "the planner should steer around a bush even if AEB should
    not brake for one", and in practice that made the car flinch at every verge
    and refuse gaps that were actually open.

    The discriminator is geometric, not semantic, and it is the same one AEB
    uses: an object of height a at range r hides the ground behind it for
    r*a/(h - a), so ground returns inside that shadow mean the rays went
    THROUGH it. A post of the same height as a bush is still an obstacle.
    """
    band = ObstacleBand(
        OBSTACLE_MIN_HEIGHT_M,
        PLANNER_HORIZON_M,
        cell_referenced=True,
        reduce_to_cells=True,
        porosity=True,
    )
    roof = 1.6

    def occupied(bev: np.ndarray, heights: np.ndarray) -> int:
        found = geometric_obstacle_sets(
            bev, heights, -0.30, (band,), sensor_height_m=roof
        )[0]
        return len(found)

    # Ground the rays get through, plus a see-through clump at 15 m.
    ground: list[tuple[float, float, float]] = []
    for forward in np.arange(1.0, 36.0, 0.12):
        step = max(0.05, float(np.radians(2.45) * forward))
        for lateral in np.arange(-6.0, 6.01, step):
            ground.append((float(lateral), float(forward), 0.0))
    bush = [
        (float(x), float(y), float(z))
        for x in np.arange(-0.7, 0.71, 0.06)
        for y in np.arange(15.0, 15.7, 0.06)
        for z in np.arange(0.15, 0.65, 0.05)
        if hash((round(float(x), 2), round(float(y), 2), round(float(z), 2))) % 3
    ]
    porous = np.asarray(ground + bush)
    assert (
        occupied(
            porous[:, :2].astype(np.float32),
            (porous[:, 2] - 0.30).astype(np.float32),
        )
        == 0
    ), "the planner still treats a see-through bush as solid"

    # The same height, but SOLID: it shadows the ground behind it, so the
    # evidence that vetoed the bush simply is not there.
    shadow_to = 15.0 + 15.0 * 0.6 / (roof - 0.6)
    solid = [
        p
        for p in ground
        if not (15.0 < p[1] < shadow_to and abs(p[0]) < 0.4)
    ] + [
        (float(x), float(y), float(z))
        for x in np.arange(-0.15, 0.16, 0.03)
        for y in np.arange(15.0, 15.31, 0.03)
        for z in np.arange(0.0, 0.61, 0.03)
    ]
    post = np.asarray(solid)
    assert (
        occupied(
            post[:, :2].astype(np.float32),
            (post[:, 2] - 0.30).astype(np.float32),
        )
        > 0
    ), "a solid post of bush height stopped being an obstacle"


def test_reducing_to_cells_keeps_every_place_something_is_standing() -> None:
    """
    `PLANNER_MAX_OBSTACLE_POINTS` bounded POINTS over a cloud whose information
    is CELLS, so the sample landed on a fraction of the occupied cells and
    overstated free distance -- the unsafe direction. The reduction is complete
    by construction: one point per occupied cell, and no cell lost.
    """
    bev, heights = _stripe_sampled_scene(0.0, wall_at=19.0, kerb_at=3.5)
    plain = ObstacleBand(
        OBSTACLE_MIN_HEIGHT_M, PLANNER_HORIZON_M, cell_referenced=True
    )
    full, reduced = geometric_obstacle_sets(
        bev, heights, -0.30, (plain, _PLANNER_CELL_BAND)
    )

    capped = np.unique(planner._cell_keys(full, OBSTACLE_CELL_M))
    cells = np.unique(planner._cell_keys(reduced, OBSTACLE_CELL_M))

    # One point per occupied cell, and every cell exactly once.
    assert len(reduced) == len(cells), "a cell survived twice"
    # NOTHING the point cap kept is lost -- the reduction is a superset.
    assert not np.setdiff1d(capped, cells).size, "the reduction dropped a cell"
    # ...and it recovers the places the cap had deleted. This is the defect,
    # measured on this very scene: the cap spent 4,000 points covering 127
    # distinct cells where 218 are occupied, so 42% of the places the planner
    # knew something was standing were simply absent from the arc scan.
    assert len(cells) > len(capped), "the cap was not losing anything here"
    assert len(reduced) < len(full), "the reduction saved nothing"


# --- guidance yields to sensing when its own line is blocked ------------------
#
# CLAUDE.md has always promised that "guidance never becomes authority ... a
# blocked arc is never chosen just because the route asked for it -- LiDAR
# always wins". It was not true. Both lateral terms SATURATE at 2.0 m of error
# while passing a 1.8 m car needs 2.3-3.0 m of offset, so every way round paid
# the full weight as a flat toll with no gradient at all -- and against that,
# the free-distance term can only offer the difference between two clipped
# values. Measured, with a destination set the planner would not leave the lane
# at ANY obstacle distance: it drove at the car until free distance crossed
# STOP_MARGIN_M, then blocked, then reversed.


def _open_road_with_a_parked_car(
    gap_m: float | None, *, road_half: float = 5.5, car_x: float = 1.8
) -> np.ndarray:
    """
    Kerbs 11 m apart and a stopped car in the ego's lane.

    The gap from the car's near flank to the far kerb is 6.4 m against the
    2.6 m the ego needs, so a pass is unambiguously available -- if the cost
    stack will let the planner take it.
    """
    points: list[tuple[float, float]] = []
    for forward in np.arange(0.5, 50.0, 0.10):
        for side in (-1.0, 1.0):
            points.append((side * road_half, float(forward)))
    if gap_m is not None:
        for lateral in np.arange(car_x - 0.9, car_x + 0.9 + 1e-9, 0.08):
            for forward in np.arange(gap_m, gap_m + 4.5, 0.08):
                points.append((float(lateral), float(forward)))
    return np.asarray(points, dtype=np.float32)


def _straight_route(count: int = 40, step: float = 3.0) -> RoutePath:
    return _route([(0.0, index * step) for index in range(count)], half_width=5.5)


def test_a_route_still_holds_the_lane_on_a_clear_road() -> None:
    """
    The identity guard. The taper is multiplicative and is exactly 1.0 while
    the reference line is clear, so nothing about ordinary route following
    changes -- if this fails, the authority is being read off the wrong
    candidate and lane discipline has been given away.
    """
    clear = _open_road_with_a_parked_car(None)
    # A route 2.5 m to the LEFT of the car, on an ordinary 6 m road. The lane
    # rule sits the ego 1.5 m right of that centreline, so the target is 1.0 m
    # left of where the car is now and it must steer for it.
    offset = _route([(-2.5, index * 3.0) for index in range(40)], half_width=3.0)

    plan = plan_arc(clear, GEOMETRY, lookahead_m=30.0, route=offset)

    assert plan.curvature > 1e-3, "the route stopped pulling the car to its lane"
    assert plan.free_distance_m == pytest.approx(PLANNER_HORIZON_M)


def test_a_stopped_car_in_the_lane_is_passed_even_with_a_route_set() -> None:
    """
    The headline. Before the taper this held the lane at every gap and drove
    into the car; the route's flat 0.9 cross-track toll simply outbid the
    whole free-distance term.
    """
    route = _straight_route()

    for gap in (20.0, 15.0):
        plan = plan_arc(
            _open_road_with_a_parked_car(gap),
            GEOMETRY,
            lookahead_m=30.0,
            route=route,
        )
        assert plan.free_distance_m > gap + 6.0, (
            f"drove at the car instead of round it with a {gap:.0f} m gap "
            f"(free {plan.free_distance_m:.1f} m)"
        )
        # ...and it went round the OPEN side, not into the kerb.
        assert plan.curvature > 0.0


def test_keep_right_also_yields_when_its_own_line_is_blocked() -> None:
    """The same defect without a destination set: keep-right saturates at
    KEEP_RIGHT_SCALE_M too, so it was the term vetoing the detour."""

    plan = plan_arc(
        _open_road_with_a_parked_car(15.0),
        GEOMETRY,
        lookahead_m=30.0,
    )

    assert plan.free_distance_m > 21.0


