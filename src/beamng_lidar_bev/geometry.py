from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    HYBRID_CAMERA_BODY_CLEARANCE_M,
    HYBRID_CAMERA_FRONT_FRACTION,
    HYBRID_CAMERA_HEIGHT_FRACTION,
    HYBRID_CAMERA_HFOV_DEG,
    HYBRID_CAMERA_PITCH_DEG,
    HYBRID_CAMERA_RESOLUTION,
    HYBRID_CAMERA_YAW_DEG,
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


def rotate_about_up(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a world vector about world Z (up) by `angle_rad`, positive left."""
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = (float(value) for value in vector)
    return np.asarray((cos * x - sin * y, sin * x + cos * y, z), dtype=np.float64)


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
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VehicleGeometry:
    """
    Derive vehicle-space extents and six mounts from a live world OOBB.

    BeamNG vehicle coordinates use +X left, +Y rearward and +Z up.

    **The extents are measured from the REFERENCE NODE and a mount `pos` is
    NOT** -- see `VehicleGeometry.sensor_origin_vehicle`. Measured live on the
    vivace with `tools/mount_origin_probe.py`, a Camera AND a Lidar both asked
    for (0, 0, 0) land at (+0.160, +0.362, -0.233) in the node frame, and the
    offset is a pure translation (identical at four probe positions, stable to
    0.1 mm across trials). So every mount built from an extent -- the four
    perimeter units, and both A-pillar cameras -- has `sensor_origin`
    subtracted from that extent, which is what puts it against the body face
    it names. The RIGHT unit was 0.11 m INSIDE the shell before this and the
    front unit 0.36 m back inside the bonnet.

    Two things deliberately do NOT get the correction. `pos.z` is measured
    from the vehicle ground plane, which is the sensor frame's own z origin
    (the bbox bottom sits at -0.010 m in it), so heights are already right and
    are the one axis this project had checked. And the centreline mounts stay
    at x = 0, because 0 in the sensor frame IS the body centreline -- the
    lateral offset above equals the body centre to a millimetre.
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

    # The body faces, expressed in the frame the simulator reads a mount `pos`
    # in. Identical to the node-frame extents when the origin has not been
    # measured, which is what keeps the offline suite's numbers unchanged.
    origin_x, origin_y, origin_z = (float(value) for value in sensor_origin)
    face_left = float(max_x) - origin_x
    face_right = float(min_x) - origin_x
    face_front = float(min_y) - origin_y
    face_rear = float(max_y) - origin_y
    # The simulator already measures vehicle-space sensor `pos` from the
    # vehicle's ground plane, so adding the bbox bottom again buries the sensor
    # below the terrain -- which kills every downward ray and collapses the
    # horizontal sweep. Verified live: passing pos.z=0.20 puts
    # Lidar.get_position() at bbox_bottom + 0.21 m, while passing
    # min_z + 0.20 put it 0.03 m underground. ground_z_vehicle below is a
    # different quantity and is still the bbox bottom. This is the one axis
    # `sensor_origin` does NOT correct, because the two frames already agree
    # on it (the bbox bottom measures -0.010 m in the sensor frame).
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
            (0.0, float(face_front - body_clearance_m), mount_z),
            (0.0, -1.0, 0.0),
            max_distance_m=LIDAR_FRONT_MAX_DISTANCE_M,
            horizontal_fov_deg=LIDAR_FRONT_HORIZONTAL_FOV_DEG,
            density=LIDAR_FRONT_DENSITY,
        ),
        "left": SensorMount(
            "left",
            (float(face_left + body_clearance_m), 0.0, mount_z),
            (1.0, 0.0, 0.0),
        ),
        "right": SensorMount(
            "right",
            (float(face_right - body_clearance_m), 0.0, mount_z),
            (-1.0, 0.0, 0.0),
        ),
        "rear": SensorMount(
            "rear",
            (0.0, float(face_rear + body_clearance_m), mount_z),
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
        body_floor_z=float(min_z),
        sensor_origin_vehicle=(origin_x, origin_y, origin_z),
    )


# Fixed rig order, front row first, so the GUI grid and every log line agree
# on which camera is which without consulting the worker.
HYBRID_CAMERA_NAMES = ("a_pillar_left", "a_pillar_right")


def camera_vertical_fov_deg(
    horizontal_fov_deg: float, resolution: tuple[int, int]
) -> float:
    """
    The `field_of_view_y` that yields a designed HORIZONTAL aperture.

    The Camera constructor takes only the vertical field of view and derives
    the horizontal one from the aspect ratio, so a camera rig -- which is
    designed around horizontal coverage exactly as the LiDAR wedges are --
    has to run the rectilinear projection backwards: tan(h/2) scales by
    height/width.
    """
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError(f"Implausible camera resolution {resolution!r}")
    half_h = math.radians(float(horizontal_fov_deg)) / 2.0
    return math.degrees(2.0 * math.atan(math.tan(half_h) * height / width))


def derive_hybrid_camera_rig(
    geometry: VehicleGeometry,
    resolution: tuple[int, int] = HYBRID_CAMERA_RESOLUTION,
) -> dict[str, CameraMount]:
    """
    HYBRID's two A-pillar colour cameras, one per side, mirrored about the
    BODY.

    The stations are built from the body faces expressed in the simulator's
    own sensor frame (`VehicleGeometry.sensor_origin_vehicle`), not from the
    node-referenced extents. Placed from the node instead, the pair is
    symmetric on paper and lands displaced by the whole origin offset: on the
    vivace the left camera ended up 0.28 m outboard of the body with the car
    entirely out of frame, and the right camera 0.04 m INSIDE the shell, where
    6.6% of its pixels were ego bodywork against the left camera's 0.65%
    (measured, `tools/hybrid_rig_probe.py`). That is the whole of the live
    "one cam shows more bodywork than the other" report, and it also drags the
    two cameras' independent auto-exposure apart, because a frame with a slab
    of dark car in it adapts brighter.
    """
    yaw = math.radians(HYBRID_CAMERA_YAW_DEG)
    pitch = math.radians(HYBRID_CAMERA_PITCH_DEG)
    horizontal = math.cos(pitch)
    origin_x, origin_y, _ = geometry.sensor_origin_vehicle
    face_left = geometry.left_m - origin_x
    face_right = -geometry.right_m - origin_x
    face_front = -geometry.front_m - origin_y
    y = HYBRID_CAMERA_FRONT_FRACTION * face_front
    z = HYBRID_CAMERA_HEIGHT_FRACTION * geometry.height_m
    left = CameraMount(
        name="a_pillar_left",
        position_vehicle=(
            face_left + HYBRID_CAMERA_BODY_CLEARANCE_M,
            y,
            z,
        ),
        direction_vehicle=(
            math.sin(yaw) * horizontal,
            -math.cos(yaw) * horizontal,
            -math.sin(pitch),
        ),
        horizontal_fov_deg=HYBRID_CAMERA_HFOV_DEG,
        vertical_fov_deg=camera_vertical_fov_deg(
            HYBRID_CAMERA_HFOV_DEG, resolution
        ),
        resolution=resolution,
    )
    right = CameraMount(
        name="a_pillar_right",
        position_vehicle=(
            face_right - HYBRID_CAMERA_BODY_CLEARANCE_M,
            y,
            z,
        ),
        direction_vehicle=(
            -math.sin(yaw) * horizontal,
            -math.cos(yaw) * horizontal,
            -math.sin(pitch),
        ),
        horizontal_fov_deg=HYBRID_CAMERA_HFOV_DEG,
        vertical_fov_deg=camera_vertical_fov_deg(
            HYBRID_CAMERA_HFOV_DEG, resolution
        ),
        resolution=resolution,
    )
    return {left.name: left, right.name: right}


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


# How far outside the bounding box a return still counts as the ego's own
# bodywork: mirrors, the shell's curvature past the OOBB face, and the
# reference-node/box misalignment the low mounts sit inside.
EGO_BODY_MARGIN_M = 0.18


def outside_ego_body(
    bev: np.ndarray, geometry: VehicleGeometry, margin_m: float = EGO_BODY_MARGIN_M
) -> np.ndarray:
    """
    Which BEV returns lie OUTSIDE the ego's own footprint (plus a margin).

    Shared by the worker's tick and the oracle tool so both clouds are culled
    identically -- the camera rig films its own bodywork by design (the
    bumper camera sees bonnet, the pitched rear camera sees the boot), and a
    cull that differed between the two would show up as obstacle cells on
    the car itself.
    """
    if not len(bev):
        return np.ones(0, dtype=bool)
    inside = (
        (bev[:, 0] >= -geometry.left_m - margin_m)
        & (bev[:, 0] <= geometry.right_m + margin_m)
        & (bev[:, 1] >= -geometry.rear_m - margin_m)
        & (bev[:, 1] <= geometry.front_m + margin_m)
    )
    return ~inside


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
