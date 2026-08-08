from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    WORLD_ACTOR_COAST_S,
    WORLD_ACTOR_FADE_S,
    WORLD_CELL_SIZE_M,
    WORLD_CELL_TTL_S,
    WORLD_MAX_BOUNDARY_MARKS,
    WORLD_MAX_UNCERTAIN_POINTS,
    WORLD_POSE_JUMP_RESET_M,
    WORLD_RADIUS_M,
)
from .models import (
    BRAKING,
    ActorObservation,
    AebState,
    PerceptionSnapshot,
    WorldActor,
    WorldFrame,
)
from .planner import path_polyline
from .semantics import (
    SCENE_BOUNDARY,
    SCENE_ROAD,
    SCENE_UNKNOWN,
    SCENE_VEHICLE,
    SCENE_VULNERABLE,
)

_EMPTY_VERTICES = np.empty((0, 3), dtype=np.float32)
_EMPTY_INDICES = np.empty(0, dtype=np.uint32)


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-8:
        raise ValueError(f"Scene snapshot contains a zero-length {label} vector")
    return vector / magnitude


def _basis(snapshot: PerceptionSnapshot) -> tuple[np.ndarray, np.ndarray]:
    forward = _unit(np.asarray(snapshot.ego_dir_world, dtype=np.float64), "forward")
    up = _unit(np.asarray(snapshot.ego_up_world, dtype=np.float64), "up")
    right = _unit(np.cross(forward, up), "right")
    forward = _unit(np.cross(up, right), "forward")
    return right, forward


def world_to_render(
    points_world: np.ndarray, snapshot: PerceptionSnapshot
) -> np.ndarray:
    """Convert BeamNG world coordinates to Qt Quick 3D right/up/-forward."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 world point array, got {points.shape}")
    if not len(points):
        return _EMPTY_VERTICES.copy()

    origin = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
    right, forward = _basis(snapshot)
    offsets = points - origin
    return np.ascontiguousarray(
        np.column_stack(
            (
                offsets @ right,
                offsets[:, 2],
                -(offsets @ forward),
            )
        ),
        dtype=np.float32,
    )


def path_ribbon(
    points_bev: np.ndarray, half_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a constant-width ribbon around a BEV path polyline."""
    points = np.asarray(points_bev, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 path, got {points.shape}")
    if len(points) < 2:
        return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()
    if half_width_m <= 0.0:
        raise ValueError("Path half-width must be positive")

    segment = np.diff(points, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    valid = segment_length > 1e-6
    if not valid.any():
        return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()
    segment[valid] /= segment_length[valid, None]
    if not valid.all():
        valid_indices = np.flatnonzero(valid)
        for index in np.flatnonzero(~valid):
            nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
            segment[index] = segment[nearest]

    tangent = np.empty_like(points)
    tangent[0] = segment[0]
    tangent[-1] = segment[-1]
    if len(points) > 2:
        tangent[1:-1] = segment[:-1] + segment[1:]
        lengths = np.linalg.norm(tangent[1:-1], axis=1)
        nonzero = lengths > 1e-6
        tangent[1:-1][nonzero] /= lengths[nonzero, None]
        tangent[1:-1][~nonzero] = segment[:-1][~nonzero]

    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = points + normal * float(half_width_m)
    right = points - normal * float(half_width_m)
    vertices = np.empty((len(points) * 2, 3), dtype=np.float32)
    vertices[0::2, 0] = left[:, 0]
    vertices[0::2, 1] = 0.03
    vertices[0::2, 2] = -left[:, 1]
    vertices[1::2, 0] = right[:, 0]
    vertices[1::2, 1] = 0.03
    vertices[1::2, 2] = -right[:, 1]

    base = np.arange(len(points) - 1, dtype=np.uint32) * 2
    indices = np.column_stack(
        (
            base,
            base + 1,
            base + 2,
            base + 1,
            base + 3,
            base + 2,
        )
    ).reshape(-1)
    return np.ascontiguousarray(vertices), np.ascontiguousarray(indices)


@dataclass
class _RoadCell:
    centre_world: tuple[float, float, float]
    last_seen: float


@dataclass
class _ActorTrack:
    observation: ActorObservation
    last_evidence: float
    confidence: float


class WorldSceneAssembler:
    """Build bounded, temporally stable render frames from perception snapshots."""

    def __init__(self) -> None:
        self._road_cells: dict[tuple[int, int, int], _RoadCell] = {}
        self._actor_tracks: dict[str, _ActorTrack] = {}
        self._last_ego_pos: np.ndarray | None = None

    def clear(self) -> None:
        self._road_cells.clear()
        self._actor_tracks.clear()
        self._last_ego_pos = None

    def update(self, snapshot: PerceptionSnapshot) -> WorldFrame:
        self._reset_after_pose_jump(snapshot)
        self._update_road_cells(snapshot)
        self._expire_road_cells(snapshot.timestamp)

        road_vertices, road_indices = self._road_mesh(snapshot)
        boundary_vertices, boundary_indices = self._boundary_mesh(snapshot)
        uncertain = self._uncertain_points(snapshot)
        actors = self._update_actors(snapshot)
        path_vertices, path_indices = self._planned_path(snapshot)
        camera_position, camera_euler = self._camera(snapshot)
        alert = self._alert(snapshot.aeb, snapshot.rear_aeb)
        plan = snapshot.plan

        return WorldFrame(
            road_vertices=road_vertices,
            road_indices=road_indices,
            boundary_vertices=boundary_vertices,
            boundary_indices=boundary_indices,
            path_vertices=path_vertices,
            path_indices=path_indices,
            uncertain_points=uncertain,
            actors=actors,
            ego_scale=(
                snapshot.vehicle_geometry.width_m,
                snapshot.vehicle_geometry.height_m,
                snapshot.vehicle_geometry.length_m,
            ),
            speed_kph=abs(snapshot.speed_mps) * 3.6,
            target_speed_kph=(
                plan.command.target_speed_mps * 3.6 if plan is not None else 0.0
            ),
            autonomy_mode=plan.command.mode if plan is not None else "OFF",
            alert=alert,
            camera_position=camera_position,
            camera_euler=camera_euler,
            timestamp=snapshot.timestamp,
            perception_available=bool(len(snapshot.points_world)),
        )

    def _reset_after_pose_jump(self, snapshot: PerceptionSnapshot) -> None:
        position = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        if (
            self._last_ego_pos is not None
            and np.linalg.norm(position - self._last_ego_pos)
            > WORLD_POSE_JUMP_RESET_M
        ):
            self.clear()
        self._last_ego_pos = position

    def _update_road_cells(self, snapshot: PerceptionSnapshot) -> None:
        points = snapshot.points_world[
            snapshot.semantic_groups == SCENE_ROAD
        ].astype(np.float64, copy=False)
        if not len(points):
            return

        keys = np.column_stack(
            (
                np.floor(points[:, 0] / WORLD_CELL_SIZE_M),
                np.floor(points[:, 1] / WORLD_CELL_SIZE_M),
                np.floor(points[:, 2] / 0.75),
            )
        ).astype(np.int32)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        mean_z = np.bincount(inverse, weights=points[:, 2]) / counts
        for index, key_values in enumerate(unique):
            key = tuple(int(value) for value in key_values)
            centre = (
                (key[0] + 0.5) * WORLD_CELL_SIZE_M,
                (key[1] + 0.5) * WORLD_CELL_SIZE_M,
                float(mean_z[index]),
            )
            self._road_cells[key] = _RoadCell(centre, snapshot.timestamp)

    def _expire_road_cells(self, now: float) -> None:
        expired = [
            key
            for key, cell in self._road_cells.items()
            if now - cell.last_seen > WORLD_CELL_TTL_S
        ]
        for key in expired:
            del self._road_cells[key]

    def _road_mesh(
        self, snapshot: PerceptionSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._road_cells:
            return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()
        ego = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        cells = [
            (key, cell)
            for key, cell in self._road_cells.items()
            if np.linalg.norm(np.asarray(cell.centre_world[:2]) - ego[:2])
            <= WORLD_RADIUS_M
        ]
        if not cells:
            return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()

        rectangles = self._merge_road_cells(cells)
        world_vertices = np.empty((len(rectangles) * 4, 3), dtype=np.float32)
        for index, (x_min, x_max, y_min, y_max, z) in enumerate(rectangles):
            world_vertices[index * 4 : index * 4 + 4] = (
                (x_min, y_min, z),
                (x_max, y_min, z),
                (x_max, y_max, z),
                (x_min, y_max, z),
            )
        vertices = world_to_render(world_vertices, snapshot)
        base = np.arange(len(rectangles), dtype=np.uint32) * 4
        indices = np.column_stack(
            (base, base + 1, base + 2, base, base + 2, base + 3)
        ).reshape(-1)
        return vertices, np.ascontiguousarray(indices)

    @staticmethod
    def _merge_road_cells(
        cells: list[tuple[tuple[int, int, int], _RoadCell]],
    ) -> list[tuple[float, float, float, float, float]]:
        """Greedily merge grid cells into rectangles for a compact QML mesh."""
        layers: dict[int, dict[int, list[tuple[int, float]]]] = {}
        for (cell_x, cell_y, cell_z), cell in cells:
            layers.setdefault(cell_z, {}).setdefault(cell_y, []).append(
                (cell_x, cell.centre_world[2])
            )

        merged: list[tuple[float, float, float, float, float]] = []
        for rows in layers.values():
            # run -> [first row, last row, height sum, contributing cells]
            active: dict[tuple[int, int], list[float]] = {}
            previous_y: int | None = None
            for cell_y in sorted(rows):
                if previous_y is None or cell_y != previous_y + 1:
                    WorldSceneAssembler._finish_road_runs(active, merged)
                    active = {}

                row = sorted(rows[cell_y])
                runs: list[tuple[int, int, float, int]] = []
                start = row[0][0]
                end = start
                height_sum = row[0][1]
                count = 1
                for cell_x, height in row[1:]:
                    if cell_x == end + 1:
                        end = cell_x
                        height_sum += height
                        count += 1
                    else:
                        runs.append((start, end, height_sum, count))
                        start = end = cell_x
                        height_sum = height
                        count = 1
                runs.append((start, end, height_sum, count))

                current_keys = {(start, end) for start, end, _, _ in runs}
                finished = {
                    key: value
                    for key, value in active.items()
                    if key not in current_keys
                }
                WorldSceneAssembler._finish_road_runs(finished, merged)
                active = {
                    key: value
                    for key, value in active.items()
                    if key in current_keys
                }
                for start, end, height_sum, count in runs:
                    key = (start, end)
                    if key in active:
                        active[key][1] = float(cell_y)
                        active[key][2] += height_sum
                        active[key][3] += count
                    else:
                        active[key] = [
                            float(cell_y),
                            float(cell_y),
                            height_sum,
                            float(count),
                        ]
                previous_y = cell_y
            WorldSceneAssembler._finish_road_runs(active, merged)
        return merged

    @staticmethod
    def _finish_road_runs(
        active: dict[tuple[int, int], list[float]],
        destination: list[tuple[float, float, float, float, float]],
    ) -> None:
        for (start_x, end_x), (
            start_y,
            end_y,
            height_sum,
            count,
        ) in active.items():
            destination.append(
                (
                    start_x * WORLD_CELL_SIZE_M,
                    (end_x + 1) * WORLD_CELL_SIZE_M,
                    start_y * WORLD_CELL_SIZE_M,
                    (end_y + 1) * WORLD_CELL_SIZE_M,
                    height_sum / count,
                )
            )

    def _boundary_mesh(
        self, snapshot: PerceptionSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (snapshot.semantic_groups == SCENE_BOUNDARY) | (
            snapshot.semantic_groups == SCENE_VULNERABLE
        )
        points = self._limit(snapshot.points_world[mask], WORLD_MAX_BOUNDARY_MARKS)
        if not len(points):
            return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()
        rendered = world_to_render(points, snapshot)
        width = 0.08
        height = 0.32
        vertices = np.empty((len(rendered) * 4, 3), dtype=np.float32)
        vertices[0::4] = rendered + (-width, 0.0, 0.0)
        vertices[1::4] = rendered + (width, 0.0, 0.0)
        vertices[2::4] = rendered + (width, height, 0.0)
        vertices[3::4] = rendered + (-width, height, 0.0)
        base = np.arange(len(rendered), dtype=np.uint32) * 4
        indices = np.column_stack(
            (base, base + 1, base + 2, base, base + 2, base + 3)
        ).reshape(-1)
        return np.ascontiguousarray(vertices), np.ascontiguousarray(indices)

    def _uncertain_points(self, snapshot: PerceptionSnapshot) -> np.ndarray:
        points = self._limit(
            snapshot.points_world[snapshot.semantic_groups == SCENE_UNKNOWN],
            WORLD_MAX_UNCERTAIN_POINTS,
        )
        return world_to_render(points, snapshot)

    def _update_actors(
        self, snapshot: PerceptionSnapshot
    ) -> tuple[WorldActor, ...]:
        vehicle_points = snapshot.points_world[
            snapshot.semantic_groups == SCENE_VEHICLE
        ].astype(np.float64, copy=False)
        observations = {actor.actor_id: actor for actor in snapshot.actors}
        for actor in snapshot.actors:
            hit_count = self._actor_hit_count(actor, vehicle_points)
            track = self._actor_tracks.get(actor.actor_id)
            if hit_count >= 3:
                confidence = min(
                    1.0,
                    (track.confidence if track is not None else 0.35) + 0.45,
                )
                self._actor_tracks[actor.actor_id] = _ActorTrack(
                    actor,
                    snapshot.timestamp,
                    confidence,
                )
            elif track is not None:
                track.observation = actor

        rendered: list[WorldActor] = []
        expired: list[str] = []
        for actor_id, track in self._actor_tracks.items():
            age = snapshot.timestamp - track.last_evidence
            if age > WORLD_ACTOR_FADE_S:
                expired.append(actor_id)
                continue
            if age <= WORLD_ACTOR_COAST_S:
                confidence = track.confidence
            else:
                fade_span = WORLD_ACTOR_FADE_S - WORLD_ACTOR_COAST_S
                confidence = track.confidence * (
                    1.0 - (age - WORLD_ACTOR_COAST_S) / fade_span
                )
            observation = observations.get(actor_id, track.observation)
            rendered.append(self._render_actor(observation, confidence, snapshot))
        for actor_id in expired:
            del self._actor_tracks[actor_id]
        return tuple(rendered)

    @staticmethod
    def _actor_hit_count(
        actor: ActorObservation, vehicle_points: np.ndarray
    ) -> int:
        if not len(vehicle_points):
            return 0
        position = np.asarray(actor.pos_world, dtype=np.float64)
        forward = np.asarray(actor.dir_world, dtype=np.float64)
        forward[2] = 0.0
        length = float(np.linalg.norm(forward))
        if length < 1e-8:
            return 0
        forward /= length
        right = np.asarray((forward[1], -forward[0], 0.0))
        offsets = vehicle_points - position
        lateral = offsets @ right
        longitudinal = offsets @ forward
        width, height, actor_length = actor.dimensions_m
        inside = (
            (np.abs(lateral) <= width * 0.5 + 0.5)
            & (np.abs(longitudinal) <= actor_length * 0.5 + 0.7)
            & (np.abs(offsets[:, 2]) <= height + 0.8)
        )
        return int(np.count_nonzero(inside))

    @staticmethod
    def _render_actor(
        actor: ActorObservation,
        confidence: float,
        snapshot: PerceptionSnapshot,
    ) -> WorldActor:
        position = world_to_render(
            np.asarray((actor.pos_world,), dtype=np.float32),
            snapshot,
        )[0]
        right, forward = _basis(snapshot)
        actor_forward = _unit(
            np.asarray(actor.dir_world, dtype=np.float64), "actor forward"
        )
        local_right = float(actor_forward @ right)
        local_forward = float(actor_forward @ forward)
        yaw = math.degrees(math.atan2(local_right, local_forward))
        return WorldActor(
            actor_id=actor.actor_id,
            kind=actor.kind,
            position=tuple(float(value) for value in position),
            yaw_deg=yaw,
            scale=actor.dimensions_m,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    @staticmethod
    def _planned_path(
        snapshot: PerceptionSnapshot,
    ) -> tuple[np.ndarray, np.ndarray]:
        plan = snapshot.plan
        if plan is None:
            return _EMPTY_VERTICES.copy(), _EMPTY_INDICES.copy()
        arc = plan.arc
        length = max(4.0, float(arc.free_distance_m))
        points = path_polyline(
            arc.curvature,
            arc.transition_distance_m,
            arc.next_curvature,
            length,
            samples=64,
        )
        vertices, indices = path_ribbon(
            points,
            snapshot.vehicle_geometry.width_m * 0.5 + 0.18,
        )
        vertices[:, 1] += snapshot.vehicle_geometry.ground_z_vehicle
        return vertices, indices

    @staticmethod
    def _camera(
        snapshot: PerceptionSnapshot,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        plan = snapshot.plan
        reversing = plan is not None and plan.command.mode == "REVERSING"
        speed = abs(snapshot.speed_mps)
        curvature = (
            abs(plan.arc.next_curvature) if plan is not None else 0.0
        )
        height = float(np.clip(9.5 + speed * 0.18 + curvature * 18.0, 9.5, 15.5))
        distance = float(np.clip(14.5 + speed * 0.62, 14.5, 33.0))
        if reversing:
            return (0.0, height, -distance), (-21.0, 180.0, 0.0)
        return (0.0, height, distance), (-21.0, 0.0, 0.0)

    @staticmethod
    def _alert(aeb: AebState | None, rear_aeb: AebState | None) -> str:
        for state in (aeb, rear_aeb):
            if state is not None and state.status == BRAKING:
                direction = "REAR AEB" if state.rearward else "AEB"
                if math.isfinite(state.time_to_collision_s):
                    return (
                        f"{direction} FULL BRAKE · "
                        f"{state.time_to_collision_s:.1f} s"
                    )
                return f"{direction} FULL BRAKE"
        return ""

    @staticmethod
    def _limit(points: np.ndarray, limit: int) -> np.ndarray:
        if len(points) <= limit:
            return points
        stride = max(1, math.ceil(len(points) / limit))
        return points[::stride][:limit]
