from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .config import (
    LIDAR_FRONT_DENSITY,
    LIDAR_FRONT_HORIZONTAL_FOV_DEG,
    LIDAR_FRONT_MAX_DISTANCE_M,
    LIDAR_ROOF_DENSITY,
    LIDAR_ROOF_FAR_M,
    LIDAR_ROOF_HORIZONTAL_FOV_DEG,
    LIDAR_ROOF_MAX_DISTANCE_M,
    LIDAR_ROOF_NEAR_M,
    ROOF_SENSOR_CLEARANCE_M,
    SENSOR_BODY_CLEARANCE_M,
    SENSOR_HEIGHT_ABOVE_GROUND_M,
)
from .models import SensorMount, VehicleGeometry


def vec3(value: Any) -> np.ndarray:
    """Convert BeamNG vector dictionaries/sequences to a float64 vector."""
    if isinstance(value, Mapping):
        return np.asarray((value["x"], value["y"], value["z"]), dtype=np.float64)
    if isinstance(value, Sequence) or isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (3,):
            return array
    raise ValueError(f"Expected a three-component vector, got {value!r}")


def _normalise(vector: np.ndarray, label: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-8:
        raise ValueError(f"BeamNG returned a zero-length {label} vector")
    return vector / magnitude


def vehicle_axes(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal world-space right, forward and up unit vectors."""
    forward = _normalise(vec3(state["dir"]), "forward")
    up = _normalise(vec3(state["up"]), "up")
    right = _normalise(np.cross(forward, up), "right")
    forward = _normalise(np.cross(up, right), "forward")
    return right, forward, up


def derive_vehicle_geometry(
    state: Mapping[str, Any],
    bbox: Mapping[str, Any],
    sensor_height_m: float = SENSOR_HEIGHT_ABOVE_GROUND_M,
    body_clearance_m: float = SENSOR_BODY_CLEARANCE_M,
    roof_clearance_m: float = ROOF_SENSOR_CLEARANCE_M,
) -> VehicleGeometry:
    """
    Derive vehicle-space extents and five mounts from a live world OOBB.

    BeamNG vehicle coordinates use +X left, +Y rearward and +Z up.
    """
    if len(bbox) < 8:
        raise ValueError("BeamNG did not return a complete vehicle bounding box")

    origin = vec3(state["pos"])
    right, forward, up = vehicle_axes(state)
    left = -right
    rearward = -forward

    world_corners = np.asarray([vec3(point) for point in bbox.values()])
    offsets = world_corners - origin
    local = np.column_stack(
        (
            offsets @ left,
            offsets @ rearward,
            offsets @ up,
        )
    )

    min_x, min_y, min_z = np.min(local, axis=0)
    max_x, max_y, max_z = np.max(local, axis=0)
    width = float(max_x - min_x)
    length = float(max_y - min_y)
    height = float(max_z - min_z)
    if not (0.5 <= width <= 10.0 and 1.0 <= length <= 35.0 and 0.4 <= height <= 12.0):
        raise ValueError(
            "The selected vehicle bounding box is implausible "
            f"({width:.2f} x {length:.2f} x {height:.2f} m)"
        )

    # The simulator already measures vehicle-space sensor `pos` from the
    # vehicle's ground plane, so adding the bbox bottom again buries the sensor
    # below the terrain -- which kills every downward ray and collapses the
    # horizontal sweep. Verified live: passing pos.z=0.20 puts
    # Lidar.get_position() at bbox_bottom + 0.21 m, while passing
    # min_z + 0.20 put it 0.03 m underground. ground_z_vehicle below is a
    # different quantity and is still the bbox bottom.
    mount_z = float(sensor_height_m)
    # Same reference plane, so this is the bbox HEIGHT plus clearance and never
    # max_z (which is measured from the reference node, not the ground). Derived
    # per vehicle rather than fixed, so a van puts the unit on the van's roof.
    roof_z = float(height + roof_clearance_m)
    # Fit the roof aperture to a ground ANNULUS in metres rather than to a fixed
    # angle, so it tracks the roof it is bolted to: a fixed 13/7.5 deg starts at
    # 6.0 m on a saloon but 8.5 m on a van, which opens a blind ring outside the
    # ~7 m the low mounts resolve. The aperture is centred on the mount
    # direction, so aiming that down is what puts every channel on the ground
    # instead of throwing half of them at the sky.
    near_rad = float(np.arctan2(roof_z, LIDAR_ROOF_NEAR_M))
    far_rad = float(np.arctan2(roof_z, LIDAR_ROOF_FAR_M))
    roof_fov_deg = float(np.degrees(near_rad - far_rad))
    depression = 0.5 * (near_rad + far_rad)
    mounts = {
        # The long-range unit. Further, narrower and denser than the other
        # three, because AEB has to act from far enough out to stop from
        # motorway speed and that is a question about one direction only.
        "front": SensorMount(
            "front",
            (0.0, float(min_y - body_clearance_m), mount_z),
            (0.0, -1.0, 0.0),
            max_distance_m=LIDAR_FRONT_MAX_DISTANCE_M,
            horizontal_fov_deg=LIDAR_FRONT_HORIZONTAL_FOV_DEG,
            density=LIDAR_FRONT_DENSITY,
        ),
        "left": SensorMount(
            "left",
            (float(max_x + body_clearance_m), 0.0, mount_z),
            (1.0, 0.0, 0.0),
        ),
        "right": SensorMount(
            "right",
            (float(min_x - body_clearance_m), 0.0, mount_z),
            (-1.0, 0.0, 0.0),
        ),
        "rear": SensorMount(
            "rear",
            (0.0, float(max_y + body_clearance_m), mount_z),
            (0.0, 1.0, 0.0),
        ),
        # The ground-fill unit. Above the bodywork and aimed down, because ring
        # spacing on the ground goes as (r^2 / h) * dtheta and the four low
        # mounts therefore cannot resolve the surface past ~7 m in one frame.
        # Forward-facing: the 170-degree wedge covers left-abeam through
        # right-abeam, which is everything the WORLD camera looks at, and a
        # second rearward copy is a one-entry addition if the rear ever matters.
        "roof": SensorMount(
            "roof",
            (0.0, 0.0, roof_z),
            (0.0, -float(np.cos(depression)), -float(np.sin(depression))),
            max_distance_m=LIDAR_ROOF_MAX_DISTANCE_M,
            horizontal_fov_deg=LIDAR_ROOF_HORIZONTAL_FOV_DEG,
            density=LIDAR_ROOF_DENSITY,
            vertical_fov_deg=roof_fov_deg,
        ),
    }

    # Measured along world Z, to match world_points_to_bev's gravity-referenced
    # heights -- the two are compared directly by geometric_obstacles and
    # classify_road_points, so they must share a frame. min_z is the same
    # quantity in the body frame and stays what the box dimensions are built
    # from; the pair only diverge while the car is pitched or rolled.
    ground_z = float(np.min(world_corners[:, 2] - origin[2]))

    return VehicleGeometry(
        ground_z_vehicle=ground_z,
        left_m=float(max_x),
        right_m=float(-min_x),
        front_m=float(-min_y),
        rear_m=float(max_y),
        height_m=height,
        mounts=mounts,
    )


def world_points_to_bev(
    points_world: np.ndarray, state: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform BeamNG world points to EGO right/forward BEV coordinates.

    Returns an Nx2 array (right, forward) plus each point's height in the
    vehicle up-axis, relative to the vehicle reference node.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 point cloud, got {points.shape}")

    origin = vec3(state["pos"])
    right, forward, _ = vehicle_axes(state)
    offsets = points - origin
    bev = np.column_stack((offsets @ right, offsets @ forward)).astype(
        np.float32, copy=False
    )
    # Heights are referenced to GRAVITY, not to the vehicle's own up axis, and
    # this is load-bearing. BeamNG's world Z is up, so the offset's Z component
    # already is the gravity-referenced height.
    #
    # Measured against the body axis instead, every degree of pitch tips the
    # whole cloud: at 1 deg nose-down the flat road 15 m ahead reads 0.26 m
    # high against a 0.20 m obstacle floor, and at 25 m it reads 0.44 m against
    # 0.34 m. A road car pitches 1-3 deg under ordinary braking, so the planner
    # saw a wall across the road, braked, pitched further, and saw more wall --
    # a latch that presents exactly as "it brakes for no reason", and as
    # "it gets stuck" once the phantom crossed STOP_MARGIN_M.
    #
    # The x/y projection stays body-referenced: cos(3 deg) shortens a 30 m
    # range by 0.04 m, which is far under the point spacing.
    heights = np.ascontiguousarray(offsets[:, 2]).astype(np.float32, copy=False)
    return bev, heights
