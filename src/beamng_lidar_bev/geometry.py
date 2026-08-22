from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    CAMERA_FRONT_BUMPER_HFOV_DEG,
    CAMERA_FRONT_MAIN_HFOV_DEG,
    CAMERA_FRONT_WIDE_HFOV_DEG,
    CAMERA_PILLAR_HFOV_DEG,
    CAMERA_PILLAR_YAW_DEG,
    CAMERA_REAR_HFOV_DEG,
    CAMERA_REPEATER_HFOV_DEG,
    CAMERA_REPEATER_YAW_DEG,
    CAMERA_RESOLUTION,
    LIDAR_FRONT_DENSITY,
    LIDAR_FRONT_HORIZONTAL_FOV_DEG,
    LIDAR_FRONT_MAX_DISTANCE_M,
    LIDAR_ROAD_DENSITY,
    LIDAR_ROAD_FAR_M,
    LIDAR_ROAD_HORIZONTAL_FOV_DEG,
    LIDAR_ROAD_MAX_DISTANCE_M,
    LIDAR_ROAD_NEAR_M,
    LIDAR_ROAD_VERTICAL_RESOLUTION,
    LIDAR_ROOF_DENSITY,
    LIDAR_ROOF_FAR_M,
    LIDAR_ROOF_HORIZONTAL_FOV_DEG,
    LIDAR_ROOF_MAX_DISTANCE_M,
    LIDAR_ROOF_NEAR_M,
    LIDAR_ROOF_VERTICAL_RESOLUTION,
    ROOF_SENSOR_CLEARANCE_M,
    SENSOR_BODY_CLEARANCE_M,
    SENSOR_HEIGHT_ABOVE_GROUND_M,
)
from .models import CameraMount, SensorMount, VehicleGeometry


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
    # The road-scan unit's aperture, fitted the same way to its own annulus.
    road_near_rad = float(np.arctan2(roof_z, LIDAR_ROAD_NEAR_M))
    road_far_rad = float(np.arctan2(roof_z, LIDAR_ROAD_FAR_M))
    road_fov_deg = float(np.degrees(road_near_rad - road_far_rad))
    road_depression = 0.5 * (road_near_rad + road_far_rad)
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
            # Twice the global figure, because ring spacing on the ground is
            # (r^2/h) * dtheta and halving dtheta is what pushed the road
            # surface from 70 m to WORLD_ROAD_RADIUS_M. Channels spread the
            # ray budget rather than adding to it, so this is nearly free.
            vertical_resolution=LIDAR_ROOF_VERTICAL_RESOLUTION,
        ),
        # The road-scan unit: the far half of the ground, ahead only. An
        # equal-angle aperture starves far rings quadratically, so the roof
        # unit alone could never resolve the road past ~55 m however many
        # channels it got -- this one spends its whole budget on the 20-100 m
        # annulus through a narrow forward wedge. Slightly ahead of the roof
        # unit so the two never coincide.
        "road": SensorMount(
            "road",
            (0.0, -0.25, roof_z),
            (
                0.0,
                -float(np.cos(road_depression)),
                -float(np.sin(road_depression)),
            ),
            max_distance_m=LIDAR_ROAD_MAX_DISTANCE_M,
            horizontal_fov_deg=LIDAR_ROAD_HORIZONTAL_FOV_DEG,
            density=LIDAR_ROAD_DENSITY,
            vertical_fov_deg=road_fov_deg,
            vertical_resolution=LIDAR_ROAD_VERTICAL_RESOLUTION,
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


# Fixed rig order, front row first, so the GUI grid and every log line agree
# on which camera is which without consulting the worker.
CAMERA_NAMES = (
    "front_main",
    "front_wide",
    "front_bumper",
    "pillar_left",
    "pillar_right",
    "repeater_left",
    "repeater_right",
    "rear",
)


def camera_vertical_fov_deg(
    horizontal_fov_deg: float, resolution: tuple[int, int]
) -> float:
    """
    The `field_of_view_y` that yields a designed HORIZONTAL aperture.

    The Camera constructor takes only the vertical field of view and derives
    the horizontal one from the aspect ratio, so the rig -- which is designed
    around horizontal coverage exactly as the LiDAR wedges are -- has to run
    the rectilinear projection backwards: tan(h/2) scales by height/width.
    """
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError(f"Implausible camera resolution {resolution!r}")
    half_h = math.radians(float(horizontal_fov_deg)) / 2.0
    return math.degrees(2.0 * math.atan(math.tan(half_h) * height / width))


def derive_camera_rig(
    geometry: VehicleGeometry,
    resolution: tuple[int, int] = CAMERA_RESOLUTION,
    body_clearance_m: float = SENSOR_BODY_CLEARANCE_M,
) -> dict[str, CameraMount]:
    """
    The eight-camera Vision rig, derived from the same live bounding box the
    LiDAR mounts are (Tesla HW4 layout: windshield main + wide, front bumper,
    two B-pillars looking forward-outboard, two fender repeaters looking
    rear-outboard, one rear).

    Same conventions as the LiDAR mounts: vehicle frame +X left, +Y rearward,
    +Z up; FORWARD IS (0, -1, 0) -- the intuitive (0, 1, 0) renders the rear
    seats -- and `pos.z` is measured from the vehicle ground plane, which the
    simulator references sensor positions to (see derive_vehicle_geometry).

    Mounts sit ON or just outside the body shell because there is no hide-ego
    flag: a camera inside the glasshouse films the cabin (measured: a
    windshield mount placed inboard came back 68% CAR). The stations are
    plausible fractions of the bounding box, not measured body features -- a
    later rung can refine the B-pillar station from the Mesh sensor's
    per-part nodes. Some bodywork in frame is correct: a real bumper camera
    sees bonnet.
    """
    height = geometry.height_m
    windshield_z = 0.90 * height
    pillar_z = 0.80 * height
    repeater_z = 0.60 * height
    bumper_z = max(0.45, 0.32 * height)
    rear_z = 0.75 * height
    # Just outside each surface, reusing the LiDAR body clearance.
    front_y = -(geometry.front_m + body_clearance_m)
    rear_y = geometry.rear_m + body_clearance_m
    left_x = geometry.left_m + body_clearance_m
    right_x = -(geometry.right_m + body_clearance_m)
    # The windshield pair sits ahead of the reference node, roughly at the
    # screen header; the repeaters ride the front fenders; the B-pillars sit
    # slightly behind the node, mid-cabin.
    windshield_y = -0.30 * geometry.front_m
    repeater_y = -0.55 * geometry.front_m
    pillar_y = 0.10 * geometry.rear_m

    pillar_yaw = math.radians(CAMERA_PILLAR_YAW_DEG)
    repeater_yaw = math.radians(CAMERA_REPEATER_YAW_DEG)
    forward = (0.0, -1.0, 0.0)
    rearward = (0.0, 1.0, 0.0)

    def mount(
        name: str,
        position: tuple[float, float, float],
        direction: tuple[float, float, float],
        hfov: float,
    ) -> CameraMount:
        return CameraMount(
            name=name,
            position_vehicle=position,
            direction_vehicle=direction,
            horizontal_fov_deg=hfov,
            vertical_fov_deg=camera_vertical_fov_deg(hfov, resolution),
            resolution=resolution,
        )

    return {
        # The windshield pair, offset either side of the mirror so the two
        # never coincide. Main is the long-range narrow view, wide the
        # context view.
        "front_main": mount(
            "front_main",
            (0.08, windshield_y, windshield_z),
            forward,
            CAMERA_FRONT_MAIN_HFOV_DEG,
        ),
        "front_wide": mount(
            "front_wide",
            (-0.08, windshield_y, windshield_z),
            forward,
            CAMERA_FRONT_WIDE_HFOV_DEG,
        ),
        "front_bumper": mount(
            "front_bumper",
            (0.0, front_y, bumper_z),
            forward,
            CAMERA_FRONT_BUMPER_HFOV_DEG,
        ),
        # B-pillars: forward-outboard. +X is LEFT, so the left camera's
        # direction gains a positive X component.
        "pillar_left": mount(
            "pillar_left",
            (left_x, pillar_y, pillar_z),
            (math.sin(pillar_yaw), -math.cos(pillar_yaw), 0.0),
            CAMERA_PILLAR_HFOV_DEG,
        ),
        "pillar_right": mount(
            "pillar_right",
            (right_x, pillar_y, pillar_z),
            (-math.sin(pillar_yaw), -math.cos(pillar_yaw), 0.0),
            CAMERA_PILLAR_HFOV_DEG,
        ),
        # Fender repeaters: rear-outboard, the blind-spot view.
        "repeater_left": mount(
            "repeater_left",
            (left_x, repeater_y, repeater_z),
            (math.sin(repeater_yaw), math.cos(repeater_yaw), 0.0),
            CAMERA_REPEATER_HFOV_DEG,
        ),
        "repeater_right": mount(
            "repeater_right",
            (right_x, repeater_y, repeater_z),
            (-math.sin(repeater_yaw), math.cos(repeater_yaw), 0.0),
            CAMERA_REPEATER_HFOV_DEG,
        ),
        "rear": mount(
            "rear",
            (0.0, rear_y, rear_z),
            rearward,
            CAMERA_REAR_HFOV_DEG,
        ),
    }


# The ground-annulus units: every ray points below the horizon, so their reach
# is the ring of ground they were fitted to, not their slant range.
_GROUND_ANNULI = {
    "roof": (LIDAR_ROOF_NEAR_M, LIDAR_ROOF_FAR_M),
    "road": (LIDAR_ROAD_NEAR_M, LIDAR_ROAD_FAR_M),
}


@dataclass(frozen=True)
class SensorCoverage:
    """One unit's horizontal footprint, in BEV (right, forward) metres."""

    right_m: float
    forward_m: float
    heading_deg: float
    """Boresight bearing, measured anticlockwise from +right (so ahead is 90)."""
    fov_deg: float
    near_m: float
    far_m: float


def sensor_coverage(mount: SensorMount) -> SensorCoverage:
    """
    What patch of ground a unit can put returns on, for the debug overlay.

    The roof unit's reach is its ground ANNULUS rather than its slant range:
    every one of its channels points below the horizon, so the slant figure
    describes air it never samples, while LIDAR_ROOF_NEAR_M..FAR_M is the ring
    of road the aperture was fitted to. The other four report a plain wedge
    from the mount to their own max distance.
    """
    local_x, local_y, _ = mount.position_vehicle
    dir_x, dir_y, _ = mount.direction_vehicle
    # Vehicle frame is +X left, +Y rearward; BEV is (right, forward). The roof
    # unit's direction has a downward component, so only the horizontal part
    # defines the wedge's bearing.
    boresight_right = -float(dir_x)
    boresight_forward = -float(dir_y)
    if math.hypot(boresight_right, boresight_forward) < 1e-9:
        boresight_right, boresight_forward = 0.0, 1.0
    near_m, far_m = _GROUND_ANNULI.get(
        mount.name, (0.0, float(mount.max_distance_m))
    )
    return SensorCoverage(
        right_m=-float(local_x),
        forward_m=-float(local_y),
        heading_deg=math.degrees(
            math.atan2(boresight_forward, boresight_right)
        ),
        fov_deg=float(mount.horizontal_fov_deg),
        near_m=float(near_m),
        far_m=float(far_m),
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
