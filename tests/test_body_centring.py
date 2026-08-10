"""
The BEV origin is the vehicle's REFERENCE NODE, not the body centre, and the
bounding box extends unequal distances either side of it. Everything anchored
to the origin without accounting for that asymmetry misplaces the car: the
AEB corridor swept a band partly beside the body (measured on the D-Series,
backing into a garage doorway it was centred in fired the reverse brake), and
the WORLD ego model stood beside the walls the scene drew correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev.aeb import EmergencyBraking, mirror_points, mirrored
from beamng_lidar_bev.config import AEB_CLEARANCE_MARGIN_M
from beamng_lidar_bev.models import PerceptionSnapshot, VehicleGeometry
from beamng_lidar_bev.world_scene import WorldSceneAssembler

# The node sits 0.5 m right of the body centre: the box reaches 1.5 m to the
# left of it and only 0.5 m to the right, so the body centre is at BEV -0.5.
OFFSET_GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.55,
    left_m=1.5,
    right_m=0.5,
    front_m=2.0,
    rear_m=2.4,
    height_m=1.45,
    mounts={},
)
_BODY_CENTRE = (OFFSET_GEOMETRY.right_m - OFFSET_GEOMETRY.left_m) / 2.0
_HALF_WIDTH = OFFSET_GEOMETRY.width_m / 2.0 + AEB_CLEARANCE_MARGIN_M
DT = 0.04


def _fence(lateral_m: float) -> np.ndarray:
    """A line of returns parallel to the path at one lateral offset."""
    return np.asarray(
        [(lateral_m, 8.0 + 0.2 * index) for index in range(12)],
        dtype=np.float64,
    )


def test_a_wall_beside_the_body_but_astride_the_node_is_not_a_threat() -> None:
    """
    The garage case: a wall 1.4 m from the BODY centre is outside the 1.15 m
    corridor, but only 0.9 m from the NODE -- the node-centred scan called it
    a threat and fired the brake in a doorway the car fit through.
    """
    brake = EmergencyBraking()
    wall = _fence(_BODY_CENTRE + _HALF_WIDTH + 0.25)

    state = brake.step(wall, OFFSET_GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.lateral_offset_m == pytest.approx(_BODY_CENTRE)
    assert state.threat_m is None


def test_a_wall_the_body_would_hit_is_a_threat_even_across_the_node() -> None:
    """The converse blindness: 1.35 m from the node but 0.85 m from the body."""
    brake = EmergencyBraking()
    wall = _fence(_BODY_CENTRE - _HALF_WIDTH + 0.3)
    assert abs(float(wall[0, 0])) > _HALF_WIDTH  # the node-centred scan missed it

    state = brake.step(wall, OFFSET_GEOMETRY, 11.0, DT, heading_rad=0.0)

    assert state.threat_m is not None


def test_the_rear_system_centres_on_the_same_body_through_the_mirror() -> None:
    """
    `mirrored` swaps left and right and `mirror_points` rotates the cloud 180
    degrees, so the same physical wall must resolve the same way backwards.
    """
    mirror_geometry = mirrored(OFFSET_GEOMETRY)
    assert (
        mirror_geometry.right_m - mirror_geometry.left_m
    ) / 2.0 == pytest.approx(-_BODY_CENTRE)

    brake = EmergencyBraking()
    # The same clear wall as the forward test, standing BEHIND the car.
    clear_wall = _fence(_BODY_CENTRE + _HALF_WIDTH + 0.25)
    clear_wall[:, 1] *= -1.0
    state = brake.step(
        mirror_points(clear_wall), mirror_geometry, 11.0, DT, heading_rad=0.0
    )
    assert state.threat_m is None


def test_the_world_frame_carries_the_body_centre() -> None:
    assembler = WorldSceneAssembler()
    snapshot = PerceptionSnapshot(
        points_world=np.zeros((0, 3), dtype=np.float32),
        semantic_groups=np.zeros(0, dtype=np.uint8),
        ego_pos_world=(0.0, 0.0, 0.0),
        ego_dir_world=(0.0, 1.0, 0.0),
        ego_up_world=(0.0, 0.0, 1.0),
        timestamp=0.0,
        speed_mps=0.0,
        vehicle_geometry=OFFSET_GEOMETRY,
    )

    frame = assembler.update(snapshot)

    # Render x is BEV right; render z is -forward, so a node ahead of the body
    # centre pushes the drawn car toward +z (behind the origin).
    assert frame.ego_centre[0] == pytest.approx(_BODY_CENTRE)
    assert frame.ego_centre[1] == pytest.approx(
        -(OFFSET_GEOMETRY.front_m - OFFSET_GEOMETRY.rear_m) / 2.0
    )
