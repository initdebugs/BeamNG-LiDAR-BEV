from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    CLEARANCE_MARGIN_M,
    KEEP_RIGHT_MARGIN_M,
    MIN_TURN_RADIUS_M,
    OBSTACLE_CELL_M,
    OBSTACLE_MAX_HEIGHT_M,
    PLANNER_HORIZON_M,
    PLANNER_MAX_OBSTACLE_POINTS,
)
from beamng_lidar_bev.models import VehicleGeometry
from beamng_lidar_bev.planner import (
    arc_polyline,
    corridor_edges,
    despeckle,
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
