from __future__ import annotations

import itertools

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    LIDAR_ROOF_FAR_M,
    LIDAR_ROOF_NEAR_M,
    SENSOR_HEIGHT_ABOVE_GROUND_M,
)
from beamng_lidar_bev.geometry import (
    derive_vehicle_geometry,
    world_points_to_bev,
)


def _state() -> dict[str, tuple[float, ...]]:
    return {
        "pos": (10.0, 20.0, 1.0),
        "dir": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "vel": (0.0, 0.0, 0.0),
    }


def _bbox() -> dict[str, tuple[float, float, float]]:
    names = (
        "front_bottom_left",
        "front_bottom_right",
        "front_top_left",
        "front_top_right",
        "rear_bottom_left",
        "rear_bottom_right",
        "rear_top_left",
        "rear_top_right",
    )
    local_corners = itertools.product((-1.0, 1.0), (-2.0, 2.0), (-0.5, 1.5))
    # local +X is world left (-X); local +Y is world rearward (-Y).
    world_corners = [(10.0 - x, 20.0 - y, 1.0 + z) for x, y, z in local_corners]
    return dict(zip(names, world_corners, strict=True))


def test_derives_outside_mounts_from_world_bbox() -> None:
    geometry = derive_vehicle_geometry(_state(), _bbox())

    assert geometry.width_m == pytest.approx(2.0)
    assert geometry.length_m == pytest.approx(4.0)
    assert geometry.height_m == pytest.approx(2.0)
    assert geometry.ground_z_vehicle == pytest.approx(-0.5)

    height = SENSOR_HEIGHT_ABOVE_GROUND_M
    assert geometry.mounts["front"].position_vehicle == pytest.approx(
        (0.0, -2.08, height)
    )
    assert geometry.mounts["rear"].position_vehicle == pytest.approx(
        (0.0, 2.08, height)
    )
    assert geometry.mounts["left"].position_vehicle == pytest.approx(
        (1.08, 0.0, height)
    )
    assert geometry.mounts["right"].position_vehicle == pytest.approx(
        (-1.08, 0.0, height)
    )


@pytest.mark.parametrize("min_z", (-0.5, -0.2431, 0.0, 0.3))
def test_mount_height_ignores_the_bbox_bottom(min_z: float) -> None:
    """
    The simulator references sensor pos to the vehicle ground plane, so every
    mount Z must be independent of where the bbox bottom sits. Adding min_z on
    top buried the sensors underground, which killed every downward ray and
    collapsed the horizontal sweep to 37 degrees.

    The roof unit is measured from the same plane, so it is the bbox HEIGHT
    plus clearance -- never max_z, which moves with the reference node.
    """
    shift = min_z + 0.5  # _bbox() is built with min_z == -0.5
    bbox = {name: (x, y, z + shift) for name, (x, y, z) in _bbox().items()}

    geometry = derive_vehicle_geometry(
        _state(), bbox, sensor_height_m=0.2, roof_clearance_m=0.12
    )

    assert geometry.ground_z_vehicle == pytest.approx(min_z)
    for name, mount in geometry.mounts.items():
        if name == "roof":
            continue
        assert mount.position_vehicle[2] == pytest.approx(0.2)
    # _bbox() is 2.0 m tall at every shift.
    assert geometry.mounts["roof"].position_vehicle[2] == pytest.approx(2.12)


@pytest.mark.parametrize("vehicle_height_m", (1.28, 1.45, 2.0, 2.9, 4.1))
def test_roof_aperture_holds_its_ground_annulus_at_any_vehicle_height(
    vehicle_height_m: float,
) -> None:
    """
    The roof unit exists to fill the road surface, so what has to be right is
    the ground ANNULUS it lands on, not the angle it subtends.

    A fixed aperture does not deliver that. At a 1.45 m saloon roof, 13 deg
    aimed 7.5 deg down starts at 6.0 m; bolted to a 2.0 m van the same angles
    start at 8.5 m -- outside the ~7 m the 0.20 m mounts resolve in one frame,
    so the two sets leave a blind RING around the car.
    """
    half = vehicle_height_m / 2.0
    corners = itertools.product((-1.0, 1.0), (-2.0, 2.0), (-half, half))
    bbox = dict(
        zip(
            _bbox().keys(),
            [(10.0 - x, 20.0 - y, 1.0 + z) for x, y, z in corners],
            strict=True,
        )
    )
    geometry = derive_vehicle_geometry(_state(), bbox, roof_clearance_m=0.12)
    roof = geometry.mounts["roof"]

    height = roof.position_vehicle[2]
    assert height == pytest.approx(vehicle_height_m + 0.12)
    # +Y is rearward, so a forward-and-down aim is -Y with a negative Z.
    x, y, z = roof.direction_vehicle
    assert x == pytest.approx(0.0)
    assert y < 0.0 and z < 0.0
    assert np.linalg.norm(roof.direction_vehicle) == pytest.approx(1.0)

    aim_deg = np.degrees(np.arctan2(-z, -y))
    top_deg = aim_deg - roof.vertical_fov_deg / 2.0
    bottom_deg = aim_deg + roof.vertical_fov_deg / 2.0
    # Every channel points below the horizon, so every channel meets the ground.
    assert top_deg > 0.0

    near_m = height / np.tan(np.radians(bottom_deg))
    far_m = height / np.tan(np.radians(top_deg))
    assert near_m == pytest.approx(LIDAR_ROOF_NEAR_M)
    assert far_m == pytest.approx(LIDAR_ROOF_FAR_M)

    # And the payoff, which pinning the annulus makes height-independent:
    # dtheta scales with h, so dr = (r^2 / h) * dtheta cancels it. Sub-cell at
    # 20 m against the 4.11 m the 0.20 m mounts manage there.
    d_theta = np.radians(roof.vertical_fov_deg / (roof.vertical_resolution - 1))
    assert (20.0**2 / height) * d_theta < 0.3


def test_transforms_world_points_to_ego_right_forward_frame() -> None:
    points = np.asarray(
        [
            (10.0, 30.0, 1.0),  # 10 m forward
            (11.0, 20.0, 1.0),  # 1 m right
            (9.0, 15.0, 2.0),  # left, rearward, and above
        ],
        dtype=np.float32,
    )

    bev, heights = world_points_to_bev(points, _state())

    np.testing.assert_allclose(bev, ((0.0, 10.0), (1.0, 0.0), (-1.0, -5.0)))
    np.testing.assert_allclose(heights, (0.0, 0.0, 1.0))


@pytest.mark.parametrize("pitch_deg", (-3.0, -1.0, 1.0, 3.0))
def test_heights_are_referenced_to_gravity_not_the_body_axis(
    pitch_deg: float,
) -> None:
    """
    Regression pin for the phantom-wall latch.

    Measured against the body up axis, a nose-down pitch lifts the road ahead
    in the height band: 1 degree puts flat ground 15 m out at 0.26 m against a
    0.20 m obstacle floor, and 3 degrees puts 25 m out at 1.31 m. Ordinary
    braking pitches a road car 1-3 degrees, so the planner saw a wall, braked,
    pitched further and saw more wall. Heights must not move when the body
    does.
    """
    theta = np.radians(pitch_deg)
    state = {
        "pos": (10.0, 20.0, 1.0),
        "dir": (0.0, float(np.cos(theta)), float(-np.sin(theta))),
        "up": (0.0, float(np.sin(theta)), float(np.cos(theta))),
        "vel": (0.0, 0.0, 0.0),
    }
    # Flat ground straight ahead, 0.5 m below the reference node.
    ahead = np.column_stack(
        (
            np.full(4, 10.0),
            20.0 + np.asarray((5.0, 15.0, 25.0, 35.0)),
            np.full(4, 0.5),
        )
    )

    _, heights = world_points_to_bev(ahead, state)

    np.testing.assert_allclose(heights, np.full(4, -0.5), atol=1e-5)
