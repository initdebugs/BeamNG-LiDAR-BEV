"""
The two pure halves of the view controls: the per-unit coverage wedge the RAW
BEV debug toggles draw, and the mouse orbit laid over the WORLD chase camera.
Both are plain arithmetic, so the offline suite can pin them completely; only
the mouse plumbing itself needs a running Qt.
"""

from __future__ import annotations

import math

import pytest

from beamng_lidar_bev.config import LIDAR_ROOF_FAR_M, LIDAR_ROOF_NEAR_M
from beamng_lidar_bev.geometry import sensor_coverage
from beamng_lidar_bev.models import SensorMount
from beamng_lidar_bev.world_scene import apply_view_orbit


def test_the_front_wedge_points_forward_with_its_own_range_and_fov() -> None:
    mount = SensorMount(
        "front",
        (0.0, -2.2, 0.20),
        (0.0, -1.0, 0.0),
        max_distance_m=200.0,
        horizontal_fov_deg=50.0,
    )
    coverage = sensor_coverage(mount)
    assert coverage.right_m == pytest.approx(0.0)
    assert coverage.forward_m == pytest.approx(2.2)
    assert coverage.heading_deg == pytest.approx(90.0)
    assert coverage.fov_deg == pytest.approx(50.0)
    assert coverage.near_m == 0.0
    assert coverage.far_m == pytest.approx(200.0)


def test_side_and_rear_wedges_point_out_of_their_own_sides() -> None:
    # Vehicle frame is +X left, +Y rearward; BEV headings are anticlockwise
    # from +right, so left is 180, right is 0 and rearward is -90.
    left = sensor_coverage(SensorMount("left", (0.9, 0.0, 0.2), (1.0, 0.0, 0.0)))
    right = sensor_coverage(
        SensorMount("right", (-0.9, 0.0, 0.2), (-1.0, 0.0, 0.0))
    )
    rear = sensor_coverage(SensorMount("rear", (0.0, 2.3, 0.2), (0.0, 1.0, 0.0)))
    assert left.right_m == pytest.approx(-0.9)
    assert abs(left.heading_deg) == pytest.approx(180.0)
    assert right.right_m == pytest.approx(0.9)
    assert right.heading_deg == pytest.approx(0.0)
    assert rear.forward_m == pytest.approx(-2.3)
    assert rear.heading_deg == pytest.approx(-90.0)


def test_the_roof_reach_is_its_ground_annulus_not_its_slant_range() -> None:
    depression = math.radians(10.0)
    mount = SensorMount(
        "roof",
        (0.0, 0.0, 1.6),
        (0.0, -math.cos(depression), -math.sin(depression)),
        max_distance_m=100.0,
        horizontal_fov_deg=170.0,
    )
    coverage = sensor_coverage(mount)
    # The downward component must not bend the bearing: only the horizontal
    # part of the direction defines the wedge.
    assert coverage.heading_deg == pytest.approx(90.0)
    assert coverage.near_m == pytest.approx(LIDAR_ROOF_NEAR_M)
    assert coverage.far_m == pytest.approx(LIDAR_ROOF_FAR_M)
    assert coverage.far_m != pytest.approx(mount.max_distance_m)


def test_a_straight_down_direction_still_yields_a_forward_bearing() -> None:
    coverage = sensor_coverage(
        SensorMount("roof", (0.0, 0.0, 1.6), (0.0, 0.0, -1.0))
    )
    assert coverage.heading_deg == pytest.approx(90.0)


_POSE = ((0.0, 12.0, 20.0), (-21.0, 0.0, 0.0))


def test_an_identity_orbit_returns_the_chase_pose_untouched() -> None:
    position, euler = apply_view_orbit(*_POSE, 0.0, 0.0, 1.0)
    assert position == _POSE[0]
    assert euler == _POSE[1]


def test_a_yaw_orbit_swings_the_position_and_the_aim_together() -> None:
    position, euler = apply_view_orbit(*_POSE, 90.0, 0.0, 1.0)
    assert position[0] == pytest.approx(20.0)
    assert position[1] == pytest.approx(12.0)
    assert position[2] == pytest.approx(0.0, abs=1e-9)
    assert euler == (pytest.approx(-21.0), pytest.approx(90.0), 0.0)


def test_zoom_divides_the_orbit_radius_and_leaves_the_aim_alone() -> None:
    position, euler = apply_view_orbit(*_POSE, 0.0, 0.0, 2.0)
    assert position == (
        pytest.approx(0.0),
        pytest.approx(6.0),
        pytest.approx(10.0),
    )
    assert euler == _POSE[1]


def test_the_pitch_clamp_moves_the_aim_only_as_far_as_the_position() -> None:
    position, euler = apply_view_orbit(*_POSE, 0.0, 200.0, 1.0)
    radius = math.hypot(math.hypot(position[0], position[2]), position[1])
    elevation = math.degrees(
        math.atan2(position[1], math.hypot(position[0], position[2]))
    )
    assert elevation == pytest.approx(85.0)
    assert radius == pytest.approx(math.hypot(12.0, 20.0))
    original_elevation = math.degrees(math.atan2(12.0, 20.0))
    assert euler[0] == pytest.approx(-21.0 - (85.0 - original_elevation))


def test_the_orbit_can_never_dive_under_the_road() -> None:
    position, _ = apply_view_orbit(*_POSE, 0.0, -200.0, 1.0)
    elevation = math.degrees(
        math.atan2(position[1], math.hypot(position[0], position[2]))
    )
    assert elevation == pytest.approx(3.0)
    assert position[1] > 0.0


def test_extreme_zoom_is_clamped_to_the_camera_range() -> None:
    far, _ = apply_view_orbit(*_POSE, 0.0, 0.0, 1e-6)
    near, _ = apply_view_orbit(*_POSE, 0.0, 0.0, 1e6)

    def radius(position: tuple[float, float, float]) -> float:
        return math.hypot(math.hypot(position[0], position[2]), position[1])

    assert radius(far) == pytest.approx(240.0)
    assert radius(near) == pytest.approx(6.0)
