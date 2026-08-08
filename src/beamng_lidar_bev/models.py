from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import (
    LIDAR_DENSITY,
    LIDAR_HORIZONTAL_FOV_DEG,
    LIDAR_MAX_DISTANCE_M,
    LIDAR_VERTICAL_FOV_DEG,
    LIDAR_VERTICAL_RESOLUTION,
)


@dataclass(frozen=True)
class SensorMount:
    name: str
    position_vehicle: tuple[float, float, float]
    direction_vehicle: tuple[float, float, float]
    # Every optical property is per-unit, because two of the five are different
    # instruments rather than extra copies. The FRONT one reaches far enough for
    # AEB to stop from motorway speed and pays for it with a narrow, dense sweep
    # (LIDAR_FRONT_MAX_DISTANCE_M). The ROOF one exists only to fill the road
    # surface, and pays for it with a narrow aperture aimed down from above the
    # bodywork (LIDAR_ROOF_VERTICAL_FOV_DEG). The other three are the plain
    # 170-degree wedges.
    max_distance_m: float = LIDAR_MAX_DISTANCE_M
    horizontal_fov_deg: float = LIDAR_HORIZONTAL_FOV_DEG
    density: float = LIDAR_DENSITY
    vertical_fov_deg: float = LIDAR_VERTICAL_FOV_DEG
    vertical_resolution: int = LIDAR_VERTICAL_RESOLUTION


@dataclass(frozen=True)
class VehicleGeometry:
    ground_z_vehicle: float
    left_m: float
    right_m: float
    front_m: float
    rear_m: float
    height_m: float
    mounts: Mapping[str, SensorMount]

    @property
    def width_m(self) -> float:
        return self.left_m + self.right_m

    @property
    def length_m(self) -> float:
        return self.front_m + self.rear_m


@dataclass(frozen=True)
class RouteHint:
    """The destination the player set in-game, used only to pick turns."""

    path_world: np.ndarray
    """(N, 3) world-space route nodes, ordered from the car outward."""
    remaining_m: float


@dataclass(frozen=True)
class ArcPlan:
    """The planner's geometric verdict, in BEV (right, forward) metres."""

    curvature: float
    """Immediate curvature command, 1/m. Positive turns left."""
    free_distance_m: float
    clearance_m: float
    keep_right_target_m: float | None
    """Desired lateral offset at the lookahead, or None when no corridor edge."""
    nav_heading_rad: float | None
    """Turn hint from the in-game route. Positive is left. None = no destination."""
    candidate_curvatures: np.ndarray
    candidate_costs: np.ndarray
    candidate_free_distances: np.ndarray
    next_curvature: float = 0.0
    """Segment-B curvature: what the path bends to after the transition."""
    transition_distance_m: float = 0.0
    """Metres of `curvature` driven before bending to `next_curvature`."""
    lookahead_m: float = 20.0
    """Where keep-right/nav were evaluated; the worker scales it with speed."""


@dataclass(frozen=True)
class ControlCommand:
    steering: float
    throttle: float
    brake: float
    gear: int
    mode: str
    """DRIVING | BLOCKED | REVERSING | STUCK."""
    target_speed_mps: float
    reason: str


@dataclass(frozen=True)
class DrivingPlan:
    arc: ArcPlan
    command: ControlCommand
    forward_speed_mps: float
    """Signed: vel projected onto the vehicle forward axis, so reverse is < 0."""
    reported_gear: object = None
    """
    What the gearbox says it is in, as opposed to what was commanded. A mode
    string ("P"/"R"/"N"/"D") for automatic-family boxes, a number for manuals,
    None when electrics could not be read.
    """


STANDBY = "STANDBY"
ARMED = "ARMED"
BRAKING = "BRAKING"


@dataclass(frozen=True)
class AebState:
    """What the emergency brake sees and is doing, in BEV metres."""

    status: str
    """STANDBY (below the arming speed) | ARMED | BRAKING."""
    brake: float
    curvature: float
    """
    Predicted path curvature, 1/m, in the frame of travel. Positive turns left
    as in `planner` -- and for the rear system that means left of the REVERSING
    direction, because `aeb` runs it on a 180-degree-rotated cloud.
    """
    rearward: bool
    """Whether this is the rear system, so the overlay knows which way to draw."""
    threat_m: float | None
    """
    Distance to the blockage, or **None when the corridor is clear**.

    Explicitly nullable rather than "the free distance, which equals the horizon
    when clear": conflating the two made a clear road read as an obstacle
    parked at the horizon, and above 64 km/h -- where the horizon is clamped by
    PLANNER_HORIZON_M -- that alone exceeded the trigger. An empty map braked
    itself down to 45 km/h, released, and did it again.
    """
    horizon_m: float
    """How far the corridor was scanned -- only as far as this speed needs."""
    standoff_m: float
    """Distance from the reference node the car must stop by."""
    brake_now_m: float
    """
    The last point to brake: a threat closer than this fires the pedal.

    The trigger is this DISTANCE, not a deceleration threshold. Scoring "how
    hard would I have to brake" and serving a proportional pedal made AEB fire
    22.9 m out at 50 km/h on 0.52 of brake -- early and gentle, which is a
    driver-assist rather than an emergency brake.
    """
    corridor_half_width_m: float
    required_decel_mps2: float
    """Deceleration needed to stop short of the standoff. 0 with no threat."""
    time_to_collision_s: float
    reason: str

    @property
    def engaged(self) -> bool:
        return self.status == BRAKING


@dataclass(frozen=True)
class BevFrame:
    road_points: np.ndarray
    obstacle_points: np.ndarray
    raw_point_count: int
    acquisition_fps: float
    poll_ms: float
    speed_mps: float
    vehicle_geometry: VehicleGeometry
    # Defaulted so the viewer path and every existing construction site are
    # untouched when self-driving is off.
    plan: DrivingPlan | None = None
    control_ms: float = 0.0
    # None whenever the matching toggle is off. Independent of `plan`: they
    # toggle separately, and AEB is the one that also runs under a human driver.
    aeb: AebState | None = None
    rear_aeb: AebState | None = None


@dataclass(frozen=True)
class ActorObservation:
    """Simulator actor state used only to enrich LiDAR-corroborated returns."""

    actor_id: str
    kind: str
    pos_world: tuple[float, float, float]
    dir_world: tuple[float, float, float]
    velocity_world: tuple[float, float, float]
    # Width, height, length of the generic visual model.
    dimensions_m: tuple[float, float, float]


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Qt-free, immutable input consumed by the asynchronous scene builder."""

    points_world: np.ndarray
    semantic_groups: np.ndarray
    ego_pos_world: tuple[float, float, float]
    ego_dir_world: tuple[float, float, float]
    ego_up_world: tuple[float, float, float]
    timestamp: float
    speed_mps: float
    """Unsigned road speed, `norm(vel)`."""
    vehicle_geometry: VehicleGeometry
    actors: tuple[ActorObservation, ...] = ()
    plan: DrivingPlan | None = None
    aeb: AebState | None = None
    rear_aeb: AebState | None = None
    forward_speed_mps: float = 0.0
    """
    Velocity projected onto the vehicle's forward axis, so reverse is < 0.

    Carried separately because `speed_mps` is a magnitude and cannot express
    direction. The scene camera needs the sign to swing round when the car
    reverses, and it cannot get it from the plan: `plan` is None whenever
    self-driving is off, which is exactly when a human is doing the reversing.
    """

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world, dtype=np.float32)
        groups = np.asarray(self.semantic_groups, dtype=np.uint8).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected an Nx3 world point array, got {points.shape}")
        if len(points) != len(groups):
            raise ValueError("Perception point and semantic-group counts do not match")
        for name, value in (
            ("ego_pos_world", self.ego_pos_world),
            ("ego_dir_world", self.ego_dir_world),
            ("ego_up_world", self.ego_up_world),
        ):
            if len(value) != 3:
                raise ValueError(f"{name} must contain three components")
        object.__setattr__(self, "points_world", np.ascontiguousarray(points))
        object.__setattr__(self, "semantic_groups", np.ascontiguousarray(groups))


@dataclass(frozen=True)
class WorldActor:
    """One corroborated generic actor in Qt Quick 3D render coordinates."""

    actor_id: str
    kind: str
    position: tuple[float, float, float]
    yaw_deg: float
    scale: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class WorldFrame:
    """
    Complete immutable payload for one WORLD view update.

    Every mesh carries its own `(N, 4)` float32 vertex colours alongside its
    positions. They are LINEAR RGBA and they hold the whole palette: the QML
    materials ship a white base and multiply, so the colours computed in
    `world_scene` are what reaches the screen. That is where the depth tint and
    the slab face shading live, both baked per vertex because the shader-side
    alternative -- SceneEnvironment Fog -- was measured to do nothing at all on
    the NoLighting materials this scene is built from.
    """

    road_vertices: np.ndarray
    road_colors: np.ndarray
    road_indices: np.ndarray
    boundary_vertices: np.ndarray
    boundary_colors: np.ndarray
    boundary_indices: np.ndarray
    # Traffic as the LiDAR actually saw it, extruded like any other solid, and
    # entirely independent of `actors`. A car has to be visible as an obstacle
    # whether or not the simulator will tell us it is there -- see
    # `_vehicle_mesh`.
    vehicle_vertices: np.ndarray
    vehicle_colors: np.ndarray
    vehicle_indices: np.ndarray
    # The emergency-braking overlay: corridor wash, then the threat plane and
    # brake-now line, split because they cannot share an opacity.
    aeb_vertices: np.ndarray
    aeb_colors: np.ndarray
    aeb_indices: np.ndarray
    aeb_marker_vertices: np.ndarray
    aeb_marker_colors: np.ndarray
    aeb_marker_indices: np.ndarray
    path_vertices: np.ndarray
    path_colors: np.ndarray
    path_indices: np.ndarray
    uncertain_points: np.ndarray
    uncertain_colors: np.ndarray
    actors: tuple[WorldActor, ...]
    ego_scale: tuple[float, float, float]
    speed_kph: float
    target_speed_kph: float
    autonomy_mode: str
    alert: str
    camera_position: tuple[float, float, float]
    camera_euler: tuple[float, float, float]
    timestamp: float
    perception_available: bool
