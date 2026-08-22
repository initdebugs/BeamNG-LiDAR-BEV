from __future__ import annotations

from dataclasses import dataclass, field
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
    """The destination the player set in-game, and the route to it."""

    path_world: np.ndarray
    """(N, 3) world-space route nodes, ordered from the car outward."""
    remaining_m: float
    dist_to_target_m: np.ndarray | None = None
    """(N,) metres of route left at each node; -1 where the game had none."""
    link_counts: np.ndarray | None = None
    """(N,) navgraph degree at each node -- the junction detector; 0 = unknown."""
    half_width_m: np.ndarray | None = None
    """(N,) navgraph half road width at each node; -1 where the node had none."""


@dataclass(frozen=True)
class RoutePath:
    """
    The route ahead as an ego-frame reference path, resampled and previewed.

    Built fresh each tick by `route_model.build_route_path` (the ego frame
    moves every tick), consumed by `planner.plan_arc` as guidance cost terms.
    Everything here is derived from `RouteHint` plus the current pose; the
    planner sees only this type, which is what keeps it free of any dependency
    on `navigation` or `route_model`.
    """

    points: np.ndarray
    """(M, 2) float32 BEV (right, forward) samples at uniform arc spacing."""
    arc_s: np.ndarray
    """(M,) arc length from the ego's projection onto the path, metres."""
    headings: np.ndarray
    """(M,) path heading, radians; 0 is straight ahead, positive is LEFT."""
    curvatures: np.ndarray
    """(M,) signed 1/m, positive left, smoothed over ROUTE_CURVATURE_SMOOTH_M."""
    half_width_m: np.ndarray
    """(M,) road half width, default-filled where the navgraph had none."""
    junction_turn: np.ndarray
    """(M,) bool: near a junction node AND the route actually turns there."""
    remaining_m: float
    """Metres of route left, measured along this polyline from the ego."""
    cross_track_m: float
    """The ego's signed offset from the path: positive when right of it."""
    speed_limit_mps: float
    """The backward speed pass evaluated at the ego -- the allowed speed now."""


@dataclass(frozen=True)
class RoadGrid:
    """
    Coarse BEV occupancy of road-classified returns, for the road bonus.

    Built by the worker from the semantic road mask (plus remembered road
    cells); consumed by `planner.plan_arc` as a negative cost. Like
    `RoutePath` it is a `models` type on purpose: the planner sees only this,
    never the semantics that produced it.
    """

    occupancy: np.ndarray
    """(H, W) uint8, 1 where a cell held road returns; row = forward index."""
    cell_m: float
    origin_right_m: float
    """BEV right coordinate of column 0's near edge."""
    origin_forward_m: float
    """BEV forward coordinate of row 0's near edge."""


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
    route_speed_limit_mps: float | None = None
    """
    The route preview's allowed speed now, or None with no reference path.

    Computed by `route_model`'s backward pass over upcoming curvature,
    turning junctions and the destination. This field is the SANCTIONED
    channel for anticipation: the controller only ever takes min() with it,
    the same category of path knowledge as `next_curvature` -- it must never
    grow controller-side lookahead of its own.
    """
    route_cross_track_m: float | None = None
    """The ego's signed offset from the route, positive right. Diagnostics."""
    route_heading_rad: float | None = None
    """The route TANGENT at the lookahead (not a bearing to a node)."""


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
    parking_brake: float = 0.0


@dataclass(frozen=True)
class DrivingPlan:
    arc: ArcPlan
    command: ControlCommand
    forward_speed_mps: float
    """Signed: vel projected onto the vehicle forward axis, so reverse is < 0."""
    reverse_arc: ArcPlan | None = None
    """
    The steered-reverse plan, in the 180-degree-rotated TRAVEL frame
    (`aeb.mirror_points`' frame, exactly as rear AEB reasons). Present only
    while the recovery is reversing or about to; overlays must un-rotate it
    the same way they un-rotate the rear AEB corridor.
    """
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
    lateral_offset_m: float = 0.0
    """
    Where the corridor's centreline sits, as a lateral offset from the origin
    of this system's own travel frame.

    The BEV origin is the vehicle's REFERENCE NODE, which is not the body
    centre: the bounding box extends `left_m` one way and `right_m` the other,
    and on a real vehicle the two differ. A corridor centred on the node sweeps
    a band partly beside the body -- measured on the D-Series backing into a
    centred garage doorway, the edge reached into the wall and fired the brake.
    The scan shifts its cloud by this value, so the overlays must shift the
    drawn corridor by it too, or what is on screen stops being what was
    scanned.
    """

    @property
    def engaged(self) -> bool:
        return self.status == BRAKING


@dataclass(frozen=True)
class ParkingBay:
    """
    One candidate parking bay in WORLD XY metres -- the anchored form.

    Lives here rather than in `parking` because `PerceptionSnapshot` carries
    these to the scene thread, and `models` sits below both. The BEV-frame
    `ParkingSlot` below is what gets drawn; this is what persists between the
    scans that rebuild the set.
    """

    centre: tuple[float, float]
    axis: tuple[float, float]
    """Unit vector from the bay's MOUTH toward its head."""
    width_m: float
    depth_m: float
    occupied: bool
    stripe_cells: int


@dataclass(frozen=True)
class ParkingJob:
    """Immutable world-space goal plus the externally meaningful job state."""

    bay: ParkingBay
    status: str = "PLANNING"


@dataclass(frozen=True)
class ParkingSlot:
    """
    One candidate parking bay, carried in BOTH frames on purpose.

    The BEV fields are what gets drawn and hit-tested; `centre_world` is the
    bay's IDENTITY. A selection has to survive both the ego moving and the bay
    set being rebuilt every `PARKING_SCAN_INTERVAL_S`, and an index into the
    last scan survives neither -- so the widget reports the world centre of
    what was clicked and the worker re-matches it, rather than passing a
    subscript that means something different by the time it arrives.
    """

    centre_right_m: float
    centre_forward_m: float
    heading_rad: float
    """
    Direction pointing INTO the bay from its mouth, in BEV (right, forward).

    Measured from +forward toward +right, so it is a compass-style bearing in
    the display frame rather than the planner's `arctan2(forward, right)`. The
    widget draws the entry chevron from it; nothing steers by it yet.
    """
    width_m: float
    """Divider to divider, across the bay."""
    depth_m: float
    """Mouth to head, along the dividers."""
    occupied: bool
    stripe_cells: int
    """Marking cells backing the two dividers -- the evidence, for the label."""
    centre_world: tuple[float, float]
    selected: bool = False


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
    route_points: np.ndarray | None = None
    """
    (M, 2) BEV samples of the reference path being followed, or None.

    Populated only while self-driving follows a route -- the overlay
    disappearing when disengaged is the honest reading, because the car is
    not following it then.
    """


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
    surface_materials: np.ndarray | None = None
    """
    Per point, what the surface it belongs to is made of (a `semantics.SURFACE_*`
    code), or None when the caller has none to offer.

    Optional because it is display-only enrichment: the scene assembler decides
    what IS a surface from shape alone, and this only decides what colour the
    surface it found should be. None fills in as zeros, which is
    `SURFACE_UNKNOWN` -- the value that renders as unidentified ground. That
    constant is not imported here because `semantics` sits ABOVE `models` in the
    layering, so the zero is a contract between the two.
    """
    forward_speed_mps: float = 0.0
    """
    Velocity projected onto the vehicle's forward axis, so reverse is < 0.

    Carried separately because `speed_mps` is a magnitude and cannot express
    direction. The scene camera needs the sign to swing round when the car
    reverses, and it cannot get it from the plan: `plan` is None whenever
    self-driving is off, which is exactly when a human is doing the reversing.
    """
    route_world: np.ndarray | None = None
    """
    (N, 3) float32 world nodes of the route being followed, or None.

    The route reaches WORLD's compose thread through this snapshot ONLY --
    no store, no refresh-thread contact -- so the two-rate confinement
    contract is untouched. Populated only while self-driving follows a route.
    """
    parking_path: np.ndarray | None = None
    """
    (N, 2) BEV samples of the manoeuvre being driven into a bay, or None.

    Present only while the park is actually running, so the overlay appearing
    IS the manoeuvre being under way -- the same honesty rule the route ribbon
    follows.
    """
    parking_slots: tuple[ParkingSlot, ...] = ()
    """
    Candidate bays in this snapshot's BEV frame, or empty with the scan off.

    Already projected by the worker rather than re-derived here: WORLD draws
    the same bays the worker found, and BEV to render is a fixed relabelling
    (`right, height, -forward`) needing no pose of its own. Reaches the scene
    thread through the frozen snapshot only, the `route_world` precedent.
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
        materials = (
            np.zeros(len(points), dtype=np.uint8)
            if self.surface_materials is None
            else np.asarray(self.surface_materials, dtype=np.uint8).reshape(-1)
        )
        if len(materials) != len(points):
            raise ValueError("Perception point and surface-material counts differ")
        object.__setattr__(self, "points_world", np.ascontiguousarray(points))
        object.__setattr__(self, "semantic_groups", np.ascontiguousarray(groups))
        object.__setattr__(
            self, "surface_materials", np.ascontiguousarray(materials)
        )
        if self.route_world is not None:
            route = np.asarray(self.route_world, dtype=np.float32)
            if route.ndim != 2 or route.shape[1] != 3:
                raise ValueError(
                    f"Expected an Nx3 route node array, got {route.shape}"
                )
            object.__setattr__(
                self, "route_world", np.ascontiguousarray(route)
            )


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
    ego_centre: tuple[float, float] = (0.0, 0.0)
    """
    Render-space (x, z) of the BODY centre. The render origin is the vehicle's
    reference node, which sits off-centre in the bounding box; an ego model
    drawn centred on the origin therefore stands beside where the scene says
    the car is -- the whole width of the node offset, against walls that are
    drawn correctly.
    """
    # The route ribbon: dashed, subordinate to the plan path by chroma and
    # alpha. Defaulted empty because it exists only while a destination is
    # being followed.
    route_vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    route_colors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    route_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )
    # Parking bays: one mesh for all of them, because vertex alpha multiplies
    # the material's opacity exactly (measured on the real GPU), so a single
    # opaque material carries the wash, the border and the selected bay's
    # stronger fill without needing a second pass.
    parking_vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    parking_colors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    parking_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )
    parking_slots: tuple[ParkingSlot, ...] = ()
    """
    The bays this frame drew, in BEV metres, for HIT-TESTING what was picked.

    Kept beside the mesh rather than derived from it: the renderer's own
    `View3D.pick` answers where in the scene the click landed, and turning
    that point into "which bay" is a containment test this tuple is the
    input to. A picked triangle index could not say which bay it belonged
    to, because they share one mesh.
    """
