from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .aeb import corridor_cross_section, predicted_corridor
from .config import (
    AEB_BRAKING_DECEL_MPS2,
    WORLD_ACTOR_COAST_S,
    WORLD_ACTOR_FADE_S,
    WORLD_AEB_ARMED_RGB,
    WORLD_AEB_BAR_ALPHA,
    WORLD_AEB_BAR_THICKNESS_M,
    WORLD_AEB_BRAKING_BOOST,
    WORLD_AEB_BRAKING_RGB,
    WORLD_AEB_DETAIL_OFFSET_M,
    WORLD_AEB_FRAME_ALPHA,
    WORLD_AEB_FRAME_WIDTH_M,
    WORLD_AEB_GROUND_OFFSET_M,
    WORLD_AEB_MARKER_ALPHA_BASE,
    WORLD_AEB_MARKER_ALPHA_TOP,
    WORLD_AEB_MARKER_HEIGHT_M,
    WORLD_AEB_POOL_ALPHA,
    WORLD_AEB_POOL_LENGTH_M,
    WORLD_AEB_RAIL_ALPHA,
    WORLD_AEB_RAIL_FADE_M,
    WORLD_AEB_RAIL_WIDTH_M,
    WORLD_AEB_URGENCY_FLOOR,
    WORLD_AEB_WASH_ALPHA_FAR,
    WORLD_AEB_WASH_ALPHA_NEAR,
    WORLD_AIR_RGB,
    WORLD_BOUNDARY_LIT_RGB,
    WORLD_BOUNDARY_RGB,
    WORLD_CAM_ALERT_LIFT_M,
    WORLD_CAM_ALERT_PULLBACK_M,
    WORLD_CAM_CORNER_LIFT_M,
    WORLD_CAM_CORNER_YAW_DEG,
    WORLD_CAM_CORNER_YAW_PER_CURVATURE,
    WORLD_CAM_DISTANCE_BASE_M,
    WORLD_CAM_DISTANCE_MAX_M,
    WORLD_CAM_DISTANCE_PER_MPS,
    WORLD_CAM_HEIGHT_BASE_M,
    WORLD_CAM_HEIGHT_MAX_M,
    WORLD_CAM_HEIGHT_PER_MPS,
    WORLD_CAM_PITCH_DEG,
    WORLD_CAM_PITCH_LIMIT_DEG,
    WORLD_CAM_REVERSE_SPEED_MPS,
    WORLD_CAM_TAU_S,
    WORLD_CAM_YAW_TAU_S,
    WORLD_CELL_MEMORY_M,
    WORLD_CELL_SIZE_M,
    WORLD_COLLISION_CEILING_M,
    WORLD_COLUMN_BRIDGE_CELLS,
    WORLD_COLUMN_HEIGHT_M,
    WORLD_COLUMN_MEMORY_M,
    WORLD_COLUMN_SIZE_M,
    WORLD_COLUMN_VERTICAL_BRIDGE_BINS,
    WORLD_DEPTH_HAZE,
    WORLD_DEPTH_NEAR_M,
    WORLD_DEPTH_SCALE_M,
    WORLD_EDGE_FADE_M,
    WORLD_MAX_BOUNDARY_POINTS,
    WORLD_MAX_COLUMNS,
    WORLD_MAX_ROAD_CELLS,
    WORLD_MAX_UNCERTAIN_POINTS,
    WORLD_MIN_SLAB_HEIGHT_M,
    WORLD_PATH_ALERT_RGB,
    WORLD_PATH_RGB,
    WORLD_POSE_JUMP_RESET_M,
    WORLD_RADIUS_M,
    WORLD_ROAD_BRIDGE_CELLS,
    WORLD_ROAD_RADIUS_M,
    WORLD_ROAD_RGB,
    WORLD_SLAB_BOTTOM_SHADE,
    WORLD_SLAB_HEIGHT_BUCKET_M,
    WORLD_SLAB_LIGHT_DIR,
    WORLD_SLAB_SIDE_SHADE_RANGE,
    WORLD_SLAB_TOP_SHADE,
    WORLD_SURFACE_BARE_RGB,
    WORLD_SURFACE_PAVED_RGB,
    WORLD_SURFACE_RADIUS_M,
    WORLD_SURFACE_SIDEWALK_RGB,
    WORLD_SURFACE_UNKNOWN_RGB,
    WORLD_SURFACE_VEGETATION_RGB,
    WORLD_SURFACE_WATER_RGB,
    WORLD_UNCERTAIN_RGB,
    WORLD_VEHICLE_LIT_RGB,
    WORLD_VEHICLE_RGB,
    WORLD_VEHICLE_TTL_S,
)
from .models import (
    BRAKING,
    STANDBY,
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
    SURFACE_BARE,
    SURFACE_PAVED,
    SURFACE_SIDEWALK,
    SURFACE_UNKNOWN,
    SURFACE_VEGETATION,
    SURFACE_WATER,
)

_EMPTY_VERTICES = np.empty((0, 3), dtype=np.float32)
_EMPTY_COLOURS = np.empty((0, 4), dtype=np.float32)
_EMPTY_INDICES = np.empty(0, dtype=np.uint32)
_EMPTY_CELL_KEYS = np.empty((0, 3), dtype=np.int32)
_EMPTY_CELL_VALUES = np.empty(0, dtype=np.float64)

# A slab is built face by face rather than as eight shared corners, because
# each face carries its OWN shade and a shared corner can only hold one colour.
# Twenty-four vertices a box against eight is a real cost and it buys the thing
# the view was missing: unlit boxes of a single flat colour have no edges, so a
# wall standing in front of a building was literally one silhouette. Corners
# 0-3 are the base ring and 4-7 the top, both counter-clockwise from
# (x_min, y_min).
_BOX_FACE_CORNERS = np.asarray(
    (
        (4, 5, 6, 7),  # top       +z
        (0, 3, 2, 1),  # bottom    -z
        (1, 2, 6, 5),  # east      +x
        (3, 0, 4, 7),  # west      -x
        (2, 3, 7, 6),  # north     +y
        (0, 1, 5, 4),  # south     -y
    ),
    dtype=np.intp,
)
_BOX_FACE_NORMALS_WORLD = np.asarray(
    (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    dtype=np.float64,
)
# Two triangles per quad, in the face's own 4-vertex block.
_FACE_TRIANGLES = np.asarray((0, 1, 2, 0, 2, 3), dtype=np.uint32)


def srgb_to_linear(channels: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer curve, elementwise."""
    values = np.asarray(channels, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_rgb(colour: str) -> np.ndarray:
    """
    An "#rrggbb" string as a LINEAR RGB triple.

    Linear because that is the space the GPU multiplies in. Measured against
    Qt 6.7.1 on D3D11: a NoLighting DefaultMaterial with a white base and a
    vertex colour of `linear_rgb(target)` renders exactly `target`. Writing the
    sRGB values straight into the buffer instead would darken the whole palette
    -- #6a7176 would land on #3f4448.
    """
    digits = colour.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"Expected an #rrggbb colour, got {colour!r}")
    channels = np.asarray(
        [int(digits[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    )
    return srgb_to_linear(channels)


_AIR_LINEAR = linear_rgb(WORLD_AIR_RGB)
_ROAD_LINEAR = linear_rgb(WORLD_ROAD_RGB)
_BOUNDARY_LINEAR = linear_rgb(WORLD_BOUNDARY_RGB)
_BOUNDARY_LIT_LINEAR = linear_rgb(WORLD_BOUNDARY_LIT_RGB)
_PATH_LINEAR = linear_rgb(WORLD_PATH_RGB)
_PATH_ALERT_LINEAR = linear_rgb(WORLD_PATH_ALERT_RGB)
_UNCERTAIN_LINEAR = linear_rgb(WORLD_UNCERTAIN_RGB)
# Indexed by SURFACE_* code, so the lookup is one fancy-index over the cells.
# Every entry sits on the road's rung of the contrast ladder and separates by
# hue -- see the WORLD_SURFACE_* block in config for why it cannot be otherwise,
# and `test_world_palette.py` for the recomputation.
_SURFACE_LINEAR = np.stack(
    [
        linear_rgb(colour)
        for _, colour in sorted(
            (
                (int(SURFACE_UNKNOWN), WORLD_SURFACE_UNKNOWN_RGB),
                (int(SURFACE_PAVED), WORLD_SURFACE_PAVED_RGB),
                (int(SURFACE_SIDEWALK), WORLD_SURFACE_SIDEWALK_RGB),
                (int(SURFACE_VEGETATION), WORLD_SURFACE_VEGETATION_RGB),
                (int(SURFACE_BARE), WORLD_SURFACE_BARE_RGB),
                (int(SURFACE_WATER), WORLD_SURFACE_WATER_RGB),
            )
        )
    ]
)
# The height field of a ground cell's key, so a bridge deck and the road beneath
# it stay separate surfaces instead of averaging into one ramp. Shared by the
# road store and by the runs promoted out of the voxel store, which is the whole
# reason it is a named constant rather than a literal in each.
_GROUND_LAYER_M = 0.75
_VEHICLE_LINEAR = linear_rgb(WORLD_VEHICLE_RGB)
_VEHICLE_LIT_LINEAR = linear_rgb(WORLD_VEHICLE_LIT_RGB)
_AEB_ARMED_LINEAR = linear_rgb(WORLD_AEB_ARMED_RGB)
_AEB_BRAKING_LINEAR = linear_rgb(WORLD_AEB_BRAKING_RGB)

# Voxel classes. _TRAFFIC is the higher value on purpose: both the per-voxel and
# the per-run reductions are `np.maximum`, so any traffic evidence in a box or a
# run promotes the whole of it rather than being averaged away.
_STATIC = np.uint8(0)
_TRAFFIC = np.uint8(1)


def depth_mix(distance_m: np.ndarray) -> np.ndarray:
    """
    How far toward the air colour something at this range should sit, in [0, 1].

    Flat zero inside WORLD_DEPTH_NEAR_M so the near field keeps the full
    contrast ladder, then exponential extinction over WORLD_DEPTH_SCALE_M.

    Exponential rather than a ramp to a far distance, because a ramp has to
    spend its gradient somewhere and the view now reaches 150 m: normalised to
    that, a wall at 12 m and a building at 20 m came back to 1.17:1, which is
    the complaint this exists to fix. An exponential puts its strongest relative
    gradient right after the cutoff and then asymptotes, so the band where two
    objects need telling apart keeps its separation and the rim still fades.
    """
    distance = np.asarray(distance_m, dtype=np.float64)
    beyond = np.maximum(distance - WORLD_DEPTH_NEAR_M, 0.0)
    return WORLD_DEPTH_HAZE * (
        1.0 - np.exp(-beyond / max(WORLD_DEPTH_SCALE_M, 1e-6))
    )


def depth_tint(colour_linear: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Bake aerial perspective into per-vertex RGBA for a render-space mesh.

    `colour_linear` is either one linear RGB triple for the whole mesh or one
    per vertex. Distance is horizontal range from the ego, which sits at the
    render origin, so the tint does not change when the camera rises with speed.

    This is a BAKED fade rather than the SceneEnvironment Fog the QML used to
    declare, because that fog was measured to do nothing at all: it is a no-op
    on NoLighting materials, and every large surface in this scene is one.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if not len(points):
        return _EMPTY_COLOURS.copy()
    colour = np.asarray(colour_linear, dtype=np.float64)
    if colour.ndim == 1:
        colour = np.broadcast_to(colour, (len(points), 3))

    distance = np.hypot(points[:, 0], points[:, 2])
    mix = depth_mix(distance)[:, None]
    blended = colour + mix * (_AIR_LINEAR - colour)
    return np.ascontiguousarray(
        np.column_stack((blended, np.ones(len(points)))), dtype=np.float32
    )


def _fade_to_air(
    colours: np.ndarray, vertices: np.ndarray, radius_m: float
) -> np.ndarray:
    """Blend the outer `WORLD_EDGE_FADE_M` of a mesh into the air colour."""
    if not len(colours) or WORLD_EDGE_FADE_M <= 0.0:
        return colours
    distance = np.hypot(
        vertices[:, 0].astype(np.float64), vertices[:, 2].astype(np.float64)
    )
    edge = np.clip(
        (distance - (radius_m - WORLD_EDGE_FADE_M)) / WORLD_EDGE_FADE_M, 0.0, 1.0
    )[:, None]
    faded = colours.astype(np.float64)
    faded[:, :3] += edge * (_AIR_LINEAR - faded[:, :3])
    return np.ascontiguousarray(faded, dtype=np.float32)


def face_shades(right: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """
    Where each of a slab's six faces sits on the boundary shadow->lit ramp.

    The world-axis face normals are rotated into RENDER space first, so the key
    light stays fixed relative to the camera: shade a box by its world normals
    instead and every wall in the scene changes brightness as the car turns.

    The two flat faces are pinned and only the four vertical ones follow the
    light, which is a stylisation rather than a simulation -- see the comment on
    WORLD_SLAB_LIGHT_DIR for why one 3D light cannot separate the three faces a
    trailing camera actually sees. Returns shades in `_BOX_FACE_CORNERS` order.
    """
    light = np.asarray(WORLD_SLAB_LIGHT_DIR, dtype=np.float64)
    light = light / np.linalg.norm(light)
    normals = np.column_stack(
        (
            _BOX_FACE_NORMALS_WORLD @ right,
            -(_BOX_FACE_NORMALS_WORLD @ forward),
        )
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    facing = (normals / np.maximum(lengths, 1e-9)) @ light
    low, high = WORLD_SLAB_SIDE_SHADE_RANGE
    shades = low + (high - low) * (0.5 + 0.5 * facing)
    shades[0] = WORLD_SLAB_TOP_SHADE
    shades[1] = WORLD_SLAB_BOTTOM_SHADE
    return np.clip(shades, 0.0, 1.0)


def _aeb_bev_to_render(
    points_bev: np.ndarray, height_m: np.ndarray | float, rearward: bool
) -> np.ndarray:
    """
    An AEB system's own travel frame to render coordinates.

    The rear system reasons in a 180-degree-ROTATED frame -- that is what lets
    it share every arc helper in `planner` unchanged -- so un-rotating is the
    whole of what drawing it backwards takes. A rotation, not a reflection, so
    handedness survives and the curvature convention holds. Mirrors
    `bev_widget._aeb_to_screen`.
    """
    points = np.asarray(points_bev, dtype=np.float64).reshape(-1, 2)
    sign = -1.0 if rearward else 1.0
    heights = np.broadcast_to(
        np.asarray(height_m, dtype=np.float64), (len(points),)
    )
    return np.ascontiguousarray(
        np.column_stack(
            (sign * points[:, 0], heights, -(sign * points[:, 1]))
        ),
        dtype=np.float32,
    )


def corridor_edges(
    curvature: float, length_m: float, half_width_m: float, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    The right and left edges of a swept corridor, as matched point sequences.

    `predicted_corridor` returns a closed polygon -- up the right edge and back
    down the left -- so splitting it in half and reversing the second gives two
    sequences that pair off index by index. Every element of the overlay is a
    strip between two such sequences, which is what keeps all of it provably the
    corridor `aeb` actually scanned rather than a redrawing of it.
    """
    outline = predicted_corridor(
        curvature, max(float(length_m), 1e-3), half_width_m, samples
    )
    half = len(outline) // 2
    return outline[:half], outline[half:][::-1]


def _strip(
    right_edge: np.ndarray,
    left_edge: np.ndarray,
    heights_m: tuple[float, float],
    colour: np.ndarray,
    alpha: np.ndarray,
    rearward: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate a quad strip between two matched edges.

    `heights_m` is `(right, left)`, which is what lets the same builder make a
    flat ground band and a vertical panel: give both edges the same height for a
    band, or two different heights for a wall standing on the chord between
    them. `alpha` is per SAMPLE and applies to both edges, so a strip can fade
    along its length or up its height with one array.
    """
    count = len(right_edge)
    interleaved = np.empty((count * 2, 2), dtype=np.float64)
    interleaved[0::2] = right_edge
    interleaved[1::2] = left_edge
    heights = np.empty(count * 2, dtype=np.float64)
    heights[0::2] = heights_m[0]
    heights[1::2] = heights_m[1]

    vertices = _aeb_bev_to_render(interleaved, heights, rearward)
    base = np.arange(count - 1, dtype=np.uint32) * 2
    indices = np.column_stack(
        (base, base + 1, base + 2, base + 1, base + 3, base + 2)
    ).reshape(-1)
    per_vertex = np.repeat(np.asarray(alpha, dtype=np.float32), 2)
    colours = np.column_stack(
        (np.tile(np.asarray(colour, dtype=np.float32), (count * 2, 1)), per_vertex)
    )
    return vertices, np.ascontiguousarray(colours), np.ascontiguousarray(indices)


def _band(
    state: AebState,
    ground_m: float,
    colour: np.ndarray,
    from_m: float,
    to_m: float,
    half_width_m: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A flat band across the corridor between two distances along the path.

    Built from the exact chords at each end rather than by slicing a sampled
    arc, because these are all short -- a 0.4 m trigger bar, a 1.4 m pool -- and
    would otherwise fall between two samples of the corridor polyline and vanish.
    """
    near = corridor_cross_section(state.curvature, float(from_m), half_width_m)
    far = corridor_cross_section(state.curvature, float(to_m), half_width_m)
    return _strip(
        np.stack((near[0], far[0])),
        np.stack((near[1], far[1])),
        (ground_m, ground_m),
        colour,
        np.full(2, alpha, dtype=np.float32),
        state.rearward,
    )


def _rail(
    state: AebState,
    ground_m: float,
    colour: np.ndarray,
    length_m: float,
    side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One edge of the corridor, as a thin strip fading away with distance."""
    half_width = state.corridor_half_width_m
    outer_r, outer_l = corridor_edges(state.curvature, length_m, half_width, 64)
    inner_r, inner_l = corridor_edges(
        state.curvature, length_m, max(half_width - WORLD_AEB_RAIL_WIDTH_M, 0.02), 64
    )
    outer, inner = (outer_r, inner_r) if side > 0 else (outer_l, inner_l)
    progress = np.linspace(0.0, length_m, len(outer))
    fade = np.exp(-progress / max(WORLD_AEB_RAIL_FADE_M, 1e-6))
    return _strip(
        outer,
        inner,
        (ground_m, ground_m),
        colour,
        (WORLD_AEB_RAIL_ALPHA * fade).astype(np.float32),
        state.rearward,
    )


def _panel(
    state: AebState,
    ground_m: float,
    colour: np.ndarray,
    distance_m: float,
    height_m: float,
    alpha: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A wall standing on the corridor chord at `distance_m`.

    `alpha` is `(base, top)`: held at the road and fading upward, so the panel
    reads as standing ON the surface rather than as a card floating over it, and
    so it never masks whatever is behind it.
    """
    chord = corridor_cross_section(
        state.curvature, float(distance_m), state.corridor_half_width_m
    )
    vertices = np.concatenate(
        (
            _aeb_bev_to_render(chord, ground_m, state.rearward),
            _aeb_bev_to_render(chord, ground_m + height_m, state.rearward),
        )
    )
    indices = np.asarray((0, 1, 3, 0, 3, 2), dtype=np.uint32)
    ramp = np.asarray((alpha[0], alpha[0], alpha[1], alpha[1]), dtype=np.float32)
    colours = np.column_stack(
        (np.tile(np.asarray(colour, dtype=np.float32), (4, 1)), ramp)
    )
    return vertices, np.ascontiguousarray(colours), indices


def _post(
    state: AebState,
    ground_m: float,
    colour: np.ndarray,
    distance_m: float,
    height_m: float,
    side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One upright of the frame around the threat panel."""
    outer = corridor_cross_section(
        state.curvature, float(distance_m), state.corridor_half_width_m
    )
    inner = corridor_cross_section(
        state.curvature,
        float(distance_m),
        max(state.corridor_half_width_m - WORLD_AEB_FRAME_WIDTH_M, 0.02),
    )
    index = 0 if side > 0 else 1
    chord = np.stack((outer[index], inner[index]))
    vertices = np.concatenate(
        (
            _aeb_bev_to_render(chord, ground_m, state.rearward),
            _aeb_bev_to_render(chord, ground_m + height_m, state.rearward),
        )
    )
    indices = np.asarray((0, 1, 3, 0, 3, 2), dtype=np.uint32)
    colours = np.column_stack(
        (
            np.tile(np.asarray(colour, dtype=np.float32), (4, 1)),
            np.full(4, WORLD_AEB_FRAME_ALPHA, dtype=np.float32),
        )
    )
    return vertices, np.ascontiguousarray(colours), indices


def _lintel(
    state: AebState,
    ground_m: float,
    colour: np.ndarray,
    distance_m: float,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bar across the top of the frame, closing the reticle."""
    chord = corridor_cross_section(
        state.curvature, float(distance_m), state.corridor_half_width_m
    )
    vertices = np.concatenate(
        (
            _aeb_bev_to_render(chord, ground_m + height_m, state.rearward),
            _aeb_bev_to_render(
                chord,
                ground_m + height_m - WORLD_AEB_FRAME_WIDTH_M,
                state.rearward,
            ),
        )
    )
    indices = np.asarray((0, 1, 3, 0, 3, 2), dtype=np.uint32)
    colours = np.column_stack(
        (
            np.tile(np.asarray(colour, dtype=np.float32), (4, 1)),
            np.full(4, WORLD_AEB_FRAME_ALPHA, dtype=np.float32),
        )
    )
    return vertices, np.ascontiguousarray(colours), indices


def _combine(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate meshes into one buffer, rebasing each one's indices."""
    if not meshes:
        return _EMPTY_VERTICES.copy(), _EMPTY_COLOURS.copy(), _EMPTY_INDICES.copy()
    vertices, colours, indices, offset = [], [], [], 0
    for mesh_vertices, mesh_colours, mesh_indices in meshes:
        vertices.append(mesh_vertices)
        colours.append(mesh_colours)
        indices.append(mesh_indices + offset)
        offset += len(mesh_vertices)
    return (
        np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(colours), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(indices), dtype=np.uint32),
    )


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


_CELL_FIELD_BITS = 21
_CELL_FIELD_OFFSET = 1 << (_CELL_FIELD_BITS - 1)


def pack_cell_keys(keys: np.ndarray) -> np.ndarray:
    """
    Pack integer cell keys into one int64 apiece.

    `np.unique(..., axis=0)` sorts a void view of each row and is dramatically
    slower than sorting plain integers -- measured 41 ms against 6 ms for the
    same road cells. Three 21-bit fields fit an int64, which covers +/- 524 km
    of cell index at any grid size this code uses.
    """
    keys = np.asarray(keys)
    packed = np.zeros(len(keys), dtype=np.int64)
    for column in range(keys.shape[1]):
        packed = (packed << _CELL_FIELD_BITS) | (
            keys[:, column].astype(np.int64) + _CELL_FIELD_OFFSET
        )
    return packed


def bridge_gaps(
    keys: np.ndarray, values: np.ndarray, axis: int, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill short gaps between two observed cells on the same line, interpolating.

    `keys` is `(N, 3)` integer `(x, y, layer)` and `values` is `(N, V)`; only
    cells sharing the other axis and the layer are ever joined. A gap wider than
    `max_gap` cells is left alone, which is what keeps the inference honest --
    it can never close an opening the car could drive through, nor extend a
    surface past its own edge, because both ends have to be observed.

    Both accumulators need this and for the same underlying reason: the sensors
    sample far more finely in elevation than in azimuth, so anything at range
    arrives as lines with holes between them. On a WALL that reads as vertical
    stripes -- 1.24 m apart at 20 m -- and extruding stripes gives striped
    buildings. On the ROAD it reads as a lattice of disconnected quads: ground
    sampling thins as r^2 radially and as r in azimuth, so past about 20 m the
    returns no longer reach every quarter-metre cell and the surface breaks up
    into a checkerboard. Interpolating the height across the gap is exactly
    right there -- a return either side means the ray reached both, so the
    surface between them was there to be hit.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if max_gap < 1 or len(keys) < 2:
        return keys, values

    other = 1 - axis
    order = np.lexsort((keys[:, axis], keys[:, other], keys[:, 2]))
    sorted_keys = keys[order]
    step = np.diff(sorted_keys[:, axis])
    same_line = (np.diff(sorted_keys[:, other]) == 0) & (
        np.diff(sorted_keys[:, 2]) == 0
    )
    fillable = same_line & (step >= 2) & (step <= max_gap + 1)
    if not fillable.any():
        return keys, values

    left = np.flatnonzero(fillable)
    counts = step[left] - 1
    repeat = np.repeat(np.arange(len(left)), counts)
    within = (
        np.arange(int(counts.sum()))
        - np.repeat(np.cumsum(counts) - counts, counts)
        + 1
    )
    fraction = (within / (counts[repeat] + 1))[:, None]

    filled = np.empty((len(repeat), 3), dtype=keys.dtype)
    filled[:, other] = sorted_keys[left[repeat], other]
    filled[:, 2] = sorted_keys[left[repeat], 2]
    filled[:, axis] = sorted_keys[left[repeat], axis] + within

    ordered = values[order]
    blended = (
        ordered[left[repeat]] * (1.0 - fraction)
        + ordered[left[repeat] + 1] * fraction
    )
    return np.concatenate((keys, filled)), np.concatenate((values, blended))


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """Index of the first element of each run of equal values."""
    if not len(sorted_keys):
        return np.empty(0, dtype=np.intp)
    breaks = np.empty(len(sorted_keys), dtype=bool)
    breaks[0] = True
    breaks[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(breaks)


def _group_ends(sorted_keys: np.ndarray) -> np.ndarray:
    """Index of the last element of each run of equal values."""
    if not len(sorted_keys):
        return np.empty(0, dtype=np.intp)
    breaks = np.empty(len(sorted_keys), dtype=bool)
    breaks[-1] = True
    breaks[:-1] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(breaks)


def _run_count(keys: np.ndarray, axis: int) -> int:
    """
    How many contiguous runs the cells form along `axis`, without merging them.

    Used only to pick the cheaper scan direction, so it does the minimum: one
    lexsort and one comparison pass, no allocation of the runs themselves.
    """
    other = 1 - axis
    order = np.lexsort((keys[:, axis], keys[:, other], keys[:, 2]))
    sorted_keys = keys[order]
    breaks = (
        (sorted_keys[1:, 2] != sorted_keys[:-1, 2])
        | (sorted_keys[1:, other] != sorted_keys[:-1, other])
        | (sorted_keys[1:, axis] != sorted_keys[:-1, axis] + 1)
    )
    return int(np.count_nonzero(breaks)) + 1


def merge_cell_runs(
    keys: np.ndarray,
    values: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily merge occupied grid cells into rectangles.

    `keys` is `(N, 3)` integer `(x, y, layer)` and `values` is `(N, V)`. Only
    cells sharing a layer ever merge, so the caller controls what may be
    averaged together -- an altitude-and-height bucket for boundary slabs. Every
    rectangle reports the mean of the cells that formed it, as
    `((M, 4) x_min/x_max/y_min/y_max, (M, V) values)`.

    The road no longer comes through here: it is meshed as a shared-corner
    surface instead, because merged rectangles share no vertices and every
    difference in mean height between two of them was a visible step. Slabs
    still merge, and for a different reason -- a facade wants to be ONE box
    rather than forty stacked ones, and a box's six faces each carry their own
    shade, so the vertex cost is 24 apiece rather than 8.

    **Runs along one axis are found with numpy and only the merge across the
    other is a Python loop.** The whole thing used to iterate cells, which put a
    dict lookup and a tuple build on every occupied 0.5 m square: measured on a
    40 m-radius open area (17.6k road cells) that was 66 ms of meshing against a
    40 ms tick, so WORLD ran at 9 Hz. Iterating runs instead makes the loop
    proportional to structure rather than area.

    **Which axis runs along is chosen per call, and for walls it is worth an
    order of magnitude.** The loop costs one iteration per run, so scanning
    across a long straight wall is the worst possible orientation: a 200 m wall
    parallel to Y is 800 single-cell X-runs, one per row, but only a handful of
    Y-runs. Fixing the axis at X put 7.4 ms a tick into merging four such walls
    once the view reached 150 m. Counting both ways first is two vectorised
    passes and picks the cheaper by an order of magnitude on exactly the
    geometry a street scene is made of.
    """
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not len(keys):
        return np.empty((0, 4), dtype=np.float64), np.empty(
            (0, values.shape[1]), dtype=np.float64
        )

    if _run_count(keys, 1) < _run_count(keys, 0):
        # Scan along Y instead. Swapping the two key columns is the whole of it
        # -- the algorithm is symmetric -- and the rectangles are swapped back
        # before returning.
        flipped, extents = merge_cell_runs(
            keys[:, (1, 0, 2)], values, cell_size_m
        )
        return flipped[:, (2, 3, 0, 1)], extents

    order = np.lexsort((keys[:, 0], keys[:, 1], keys[:, 2]))
    sorted_keys = keys[order]
    # A run breaks wherever the layer or row changes, or X stops being
    # contiguous. Everything up to here is one vectorised pass.
    breaks = np.empty(len(sorted_keys), dtype=bool)
    breaks[0] = True
    breaks[1:] = (
        (sorted_keys[1:, 2] != sorted_keys[:-1, 2])
        | (sorted_keys[1:, 1] != sorted_keys[:-1, 1])
        | (sorted_keys[1:, 0] != sorted_keys[:-1, 0] + 1)
    )
    starts = np.flatnonzero(breaks)
    lengths = np.diff(np.append(starts, len(sorted_keys)))
    run_sums = np.add.reduceat(values[order], starts, axis=0)
    run_x0 = sorted_keys[starts, 0]
    run_x1 = run_x0 + lengths - 1
    run_y = sorted_keys[starts, 1]
    run_layer = sorted_keys[starts, 2]

    rectangles: list[tuple[float, float, float, float]] = []
    merged_values: list[np.ndarray] = []

    def flush(runs: dict[tuple[int, int], list[Any]]) -> None:
        for (x_start, x_end), (y_start, y_end, count, total) in runs.items():
            rectangles.append(
                (
                    x_start * cell_size_m,
                    (x_end + 1) * cell_size_m,
                    y_start * cell_size_m,
                    (y_end + 1) * cell_size_m,
                )
            )
            merged_values.append(total / count)

    active: dict[tuple[int, int], list[Any]] = {}
    previous_layer: int | None = None
    previous_y: int | None = None
    index = 0
    while index < len(starts):
        row_end = index
        while (
            row_end < len(starts)
            and run_layer[row_end] == run_layer[index]
            and run_y[row_end] == run_y[index]
        ):
            row_end += 1

        layer = int(run_layer[index])
        row_y = int(run_y[index])
        if layer != previous_layer or previous_y is None or row_y != previous_y + 1:
            flush(active)
            active = {}

        spans = {
            (int(run_x0[position]), int(run_x1[position])): position
            for position in range(index, row_end)
        }
        flush({key: run for key, run in active.items() if key not in spans})
        active = {key: run for key, run in active.items() if key in spans}
        for span, position in spans.items():
            if span in active:
                run = active[span]
                run[1] = row_y
                run[2] += int(lengths[position])
                run[3] = run[3] + run_sums[position]
            else:
                active[span] = [
                    row_y,
                    row_y,
                    int(lengths[position]),
                    run_sums[position].copy(),
                ]
        previous_layer = layer
        previous_y = row_y
        index = row_end
    flush(active)

    if not rectangles:
        return np.empty((0, 4), dtype=np.float64), np.empty(
            (0, values.shape[1]), dtype=np.float64
        )
    return (
        np.asarray(rectangles, dtype=np.float64),
        np.asarray(merged_values, dtype=np.float64),
    )


@dataclass(frozen=True)
class CameraPose:
    """Where the chase camera is, as four independently damped scalars."""

    height_m: float
    distance_m: float
    pitch_deg: float
    yaw_deg: float
    """
    Orbit angle about the ego, 0 behind and 180 ahead. The camera POSITION is
    derived from it, so damping this one number sweeps the reverse swing round
    the side instead of teleporting through the car.

    Kept as a plain scalar rather than wrapped: the only targets are ~0 and ~180
    give or take the corner offset, so the value never leaves [-30, 210] and
    there is no shortest-path ambiguity to resolve.
    """


def damp(current: float, target: float, dt: float, tau: float) -> float:
    """
    One exponential step toward a target, independent of the frame rate.

    ``1 - exp(-dt/tau)`` rather than a fixed fraction per frame, because the
    scene thread's rate varies with the scene: a per-frame constant would make
    the camera lazier exactly when the build is slow.
    """
    if tau <= 0.0 or dt <= 0.0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


def camera_target(snapshot: PerceptionSnapshot, alerting: bool) -> CameraPose:
    """
    Where the camera wants to be. Pure: the damping toward it lives in the
    assembler, which is the thing that has state.

    There is ONE framing, and standing still is not a special case of it. A
    top-down tilt at a standstill was tried and removed: the speed terms already
    close the view in as the car slows, and every threshold that could switch
    framings sits inside the range ordinary driving spends time in -- junctions,
    queues, give-way lines -- so the view changed shape while nothing about the
    situation had. Distance is cued here by depth tint and by a stable frame,
    and both are worth more than a second framing.
    """
    speed = abs(snapshot.speed_mps)
    plan = snapshot.plan
    reversing = snapshot.forward_speed_mps < -WORLD_CAM_REVERSE_SPEED_MPS or (
        plan is not None and plan.command.mode == "REVERSING"
    )
    curvature = _travel_curvature(snapshot, reversing)

    height = WORLD_CAM_HEIGHT_BASE_M + speed * WORLD_CAM_HEIGHT_PER_MPS
    height += abs(curvature) * WORLD_CAM_CORNER_LIFT_M
    distance = WORLD_CAM_DISTANCE_BASE_M + speed * WORLD_CAM_DISTANCE_PER_MPS
    if alerting:
        height += WORLD_CAM_ALERT_LIFT_M
        distance += WORLD_CAM_ALERT_PULLBACK_M

    # Positive curvature turns left, and the camera orbits toward the outside of
    # the bend, which is what puts the inside of it in view past the ego.
    corner = float(
        np.clip(
            -curvature * WORLD_CAM_CORNER_YAW_PER_CURVATURE,
            -WORLD_CAM_CORNER_YAW_DEG,
            WORLD_CAM_CORNER_YAW_DEG,
        )
    )
    return CameraPose(
        height_m=float(
            np.clip(height, WORLD_CAM_HEIGHT_BASE_M, WORLD_CAM_HEIGHT_MAX_M)
        ),
        distance_m=float(
            np.clip(distance, WORLD_CAM_DISTANCE_BASE_M, WORLD_CAM_DISTANCE_MAX_M)
        ),
        # Kept as a guard rather than as a working limit: at exactly -90 the
        # euler yaw is degenerate and the view spins on its own, so any future
        # pitch term has to run into this rather than into that.
        pitch_deg=float(max(WORLD_CAM_PITCH_DEG, WORLD_CAM_PITCH_LIMIT_DEG)),
        yaw_deg=(180.0 if reversing else 0.0) + corner,
    )


def _travel_curvature(snapshot: PerceptionSnapshot, reversing: bool) -> float:
    """
    How hard the car is turning, however it is being driven.

    The plan carries it when self-driving; otherwise the armed AEB state does,
    because that one derives curvature from MEASURED yaw and runs under a human
    driver -- which is exactly when the camera most needs to know.
    """
    plan = snapshot.plan
    if plan is not None:
        return float(plan.arc.next_curvature)
    state = snapshot.rear_aeb if reversing else snapshot.aeb
    return float(state.curvature) if state is not None else 0.0


@dataclass
class _ActorTrack:
    observation: ActorObservation
    last_evidence: float
    confidence: float


class WorldSceneAssembler:
    """Build bounded, temporally stable render frames from perception snapshots."""

    def __init__(self) -> None:
        self._actor_tracks: dict[str, _ActorTrack] = {}
        self._last_ego_pos: np.ndarray | None = None
        # Camera state. None means "no pose yet", which makes the first frame
        # land exactly on its target instead of easing in from a guess.
        self._camera_pose: CameraPose | None = None
        self._camera_at: float | None = None
        # Parallel numpy arrays rather than dicts of dataclasses: both stores
        # are bin-and-reduce over the cloud and stay vectorised end to end.
        self._clear_geometry()

    def clear(self) -> None:
        self._actor_tracks.clear()
        self._last_ego_pos = None
        self._clear_geometry()

    def _clear_geometry(self) -> None:
        # Metres of ego travel since the store was last cleared. Static geometry
        # expires against THIS rather than against the wall clock -- see
        # WORLD_CELL_MEMORY_M for why, and `_track_ego_motion` for where it comes
        # from. Reset here so the odometer and the stamps written against it can
        # never disagree.
        self._travelled_m = 0.0
        self._road_keys = _EMPTY_CELL_KEYS.copy()
        self._road_height = _EMPTY_CELL_VALUES.copy()
        # Odometer readings, not timestamps.
        self._road_seen = _EMPTY_CELL_VALUES.copy()
        # What the surface is made of, for colour only. A value rather than a
        # key field, exactly like `_voxel_class`: a cell must not be able to
        # exist twice because a lane marking and the tarmac beside it were
        # annotated differently.
        self._road_material = np.empty(0, dtype=np.uint8)
        # (x, y, height-bin) voxels, not (x, y) columns holding one span. The
        # third field is what lets a tree be a canopy over a gap over a trunk.
        self._voxel_keys = _EMPTY_CELL_KEYS.copy()
        self._voxel_low = _EMPTY_CELL_VALUES.copy()
        self._voxel_high = _EMPTY_CELL_VALUES.copy()
        # TWO clocks, because the two classes are different kinds of thing.
        # Scenery is remembered for a distance driven; traffic is remembered for
        # a duration, so a car crossing in front of a STOPPED ego still fades.
        self._voxel_seen = _EMPTY_CELL_VALUES.copy()
        self._voxel_travel = _EMPTY_CELL_VALUES.copy()
        # _STATIC or _TRAFFIC, parallel to the keys. Kept as a value rather than
        # a fourth key field so a voxel cannot exist twice, and so the class can
        # be promoted in place when a parked car starts moving.
        self._voxel_class = np.empty(0, dtype=np.uint8)
        # ...and what it is made of, for the runs `_column_runs` promotes to the
        # ground surface. Meaningless on a wall or a car, and never read there.
        self._voxel_material = np.empty(0, dtype=np.uint8)

    def update(self, snapshot: PerceptionSnapshot) -> WorldFrame:
        self._track_ego_motion(snapshot)
        self._update_road_cells(snapshot)
        self._expire_road_cells(snapshot)
        self._update_boundary_columns(snapshot)
        self._expire_boundary_columns(snapshot)

        alert = self._alert(snapshot.aeb, snapshot.rear_aeb)
        # Collapsed ONCE and handed to both consumers. The pass is a reduceat
        # over the whole voxel store, and the ground surface and the slabs are
        # the two halves of its one answer -- what is flat enough to stand on,
        # and what is not.
        (
            hazard_keys,
            hazard_base,
            hazard_top,
            hazard_classes,
            surface_keys,
            surface_height,
            surface_material,
        ) = self._column_runs(snapshot)
        road_vertices, road_colors, road_indices = self._ground_mesh(
            snapshot, surface_keys, surface_height, surface_material
        )
        (
            (boundary_vertices, boundary_colors, boundary_indices),
            (vehicle_vertices, vehicle_colors, vehicle_indices),
        ) = self._solid_meshes(
            snapshot, hazard_keys, hazard_base, hazard_top, hazard_classes
        )
        (
            (aeb_vertices, aeb_colors, aeb_indices),
            (marker_vertices, marker_colors, marker_indices),
        ) = self._aeb_meshes(snapshot)
        uncertain, uncertain_colors = self._uncertain_points(snapshot)
        actors = self._update_actors(snapshot)
        path_vertices, path_colors, path_indices = self._planned_path(snapshot, alert)
        camera_position, camera_euler = self._camera(snapshot, alert)
        plan = snapshot.plan

        return WorldFrame(
            road_vertices=road_vertices,
            road_colors=road_colors,
            road_indices=road_indices,
            boundary_vertices=boundary_vertices,
            boundary_colors=boundary_colors,
            boundary_indices=boundary_indices,
            vehicle_vertices=vehicle_vertices,
            vehicle_colors=vehicle_colors,
            vehicle_indices=vehicle_indices,
            aeb_vertices=aeb_vertices,
            aeb_colors=aeb_colors,
            aeb_indices=aeb_indices,
            aeb_marker_vertices=marker_vertices,
            aeb_marker_colors=marker_colors,
            aeb_marker_indices=marker_indices,
            path_vertices=path_vertices,
            path_colors=path_colors,
            path_indices=path_indices,
            uncertain_points=uncertain,
            uncertain_colors=uncertain_colors,
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

    def _track_ego_motion(self, snapshot: PerceptionSnapshot) -> None:
        """
        Advance the odometer, and drop everything on a teleport.

        The odometer is summed from successive ego positions rather than
        integrated from speed, because the positions ARE the ground truth the
        whole store is anchored on -- there is nothing to drift against. A
        snapshot dropped by `SceneWorker`'s one-slot handoff costs a chord
        instead of an arc, which at these speeds is nothing.
        """
        position = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        if self._last_ego_pos is not None:
            step = float(np.linalg.norm(position - self._last_ego_pos))
            if step > WORLD_POSE_JUMP_RESET_M:
                # A teleport, not a drive. Clearing resets the odometer too, so
                # the jump is never counted as travel.
                self.clear()
            else:
                self._travelled_m += step
        self._last_ego_pos = position

    def _update_road_cells(self, snapshot: PerceptionSnapshot) -> None:
        """
        Fold this snapshot's road returns into the world-anchored cells.

        Parallel numpy arrays for the same reason the boundary columns are: a
        Python loop that touched every occupied 0.5 m square cost 33 ms on an
        open 40 m radius, before any meshing had happened.
        """
        road = snapshot.semantic_groups == SCENE_ROAD
        points = snapshot.points_world[road].astype(np.float64, copy=False)
        if not len(points):
            return
        # Anything the road rule accepted but no material named is PAVED, which
        # keeps the geometric fallback band -- unannotated ground near the ego
        # plane -- looking exactly as it always has rather than switching to the
        # unidentified-surface colour on every community map.
        material = np.where(
            snapshot.surface_materials[road] == SURFACE_UNKNOWN,
            SURFACE_PAVED,
            snapshot.surface_materials[road],
        ).astype(np.uint8)

        keys = np.column_stack(
            (
                np.floor(points[:, 0] / WORLD_CELL_SIZE_M),
                np.floor(points[:, 1] / WORLD_CELL_SIZE_M),
                np.floor(points[:, 2] / _GROUND_LAYER_M),
            )
        ).astype(np.int32)
        # ONE sort, not np.unique's sort plus a lexsort on top of it. Sorting is
        # the whole cost of this function and the quarter-metre grid multiplied
        # the cell count by four, so the second pass stopped being affordable.
        # Packed ONCE and then indexed by the sort order. Packing is three
        # shifts and three ors over the whole cloud, so re-deriving it from the
        # reordered keys was a second full pass for nothing.
        packed = pack_cell_keys(keys)
        order = np.argsort(packed, kind="stable")
        starts = _group_starts(packed[order])
        counts = np.diff(np.append(starts, len(order)))
        fresh_height = np.add.reduceat(points[order, 2], starts) / counts
        # Height is the MEAN of the cell's returns; material cannot be averaged,
        # so the cell takes the highest code present. Ties inside one 0.25 m
        # square are between two materials that genuinely meet there, and at
        # that size either answer is right.
        fresh_material = np.maximum.reduceat(material[order], starts)

        all_keys = np.concatenate((self._road_keys, keys[order][starts]))
        all_height = np.concatenate((self._road_height, fresh_height))
        all_material = np.concatenate((self._road_material, fresh_material))
        all_seen = np.concatenate(
            (self._road_seen, np.full(len(starts), self._travelled_m))
        )
        # Newest observation wins, which is exactly what a dict write did. The
        # sort is STABLE and this frame's cells were appended after the stored
        # ones, so the last entry of each cell's block is the freshest reading.
        # Note this compares NOTHING: it is the append order that decides, which
        # is why the stamp being an odometer -- a value that stalls whenever the
        # car is parked, instead of a clock that always advances -- changes
        # nothing here. A pose jump clears the whole store anyway.
        combined_packed = pack_cell_keys(all_keys)
        combined_order = np.argsort(combined_packed, kind="stable")
        newest = combined_order[_group_ends(combined_packed[combined_order])]
        self._road_keys = all_keys[newest]
        self._road_height = all_height[newest]
        self._road_material = all_material[newest]
        self._road_seen = all_seen[newest]

    def _expire_road_cells(self, snapshot: PerceptionSnapshot) -> None:
        """
        Forget road the car has DRIVEN PAST, and road it can no longer draw.

        Both bounds live here rather than in `_road_mesh`. The radius test used
        to sit in the mesh and decide only what was drawn, which was survivable
        while a 1.2 s TTL bounded the store by itself; a distance-stamped store
        parked in one spot would otherwise keep every cell the sensors ever
        reached. `_expire_boundary_columns` has always owned its own cull for
        the same reason, and `update` expires before it meshes, so the surface
        drawn is unchanged.
        """
        if not len(self._road_keys):
            return
        ego = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        centres = (self._road_keys[:, :2] + 0.5) * WORLD_CELL_SIZE_M
        offsets = centres - ego[:2]
        keep = (self._travelled_m - self._road_seen <= WORLD_CELL_MEMORY_M) & (
            np.einsum("ij,ij->i", offsets, offsets) <= WORLD_ROAD_RADIUS_M**2
        )
        if keep.all() and len(self._road_keys) <= WORLD_MAX_ROAD_CELLS:
            return
        indices = np.flatnonzero(keep)
        if len(indices) > WORLD_MAX_ROAD_CELLS:
            # Drop what was seen longest ago, as the voxel store does.
            freshest = np.argsort(self._road_seen[indices])[-WORLD_MAX_ROAD_CELLS:]
            indices = np.sort(indices[freshest])
        self._road_keys = self._road_keys[indices]
        self._road_height = self._road_height[indices]
        self._road_material = self._road_material[indices]
        self._road_seen = self._road_seen[indices]

    def _ground_cells(
        self,
        snapshot: PerceptionSnapshot,
        surface_keys: np.ndarray,
        surface_height: np.ndarray,
        surface_material: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Every ground cell to be drawn: road store plus the promoted surface.

        Two sources, one surface. The road store holds what the semantics called
        drivable; the promoted runs hold everything else the sensors found lying
        flat on the floor, which before they were kept was simply not drawn --
        grass, dirt, a gravel yard, and the whole of any map without
        annotations. Where both have a cell the ROAD wins, because "the car may
        drive here" is the more specific claim and it is the one the view exists
        to make.

        The two stores are keyed on their own grids (WORLD_CELL_SIZE_M and
        WORLD_COLUMN_SIZE_M, equal today), so the promoted keys are converted
        through world coordinates rather than reused, and a future divergence
        between the two costs an arithmetic pass instead of a silent misalignment
        of half a cell.
        """
        ego = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        if len(surface_keys):
            # The road store was culled to WORLD_ROAD_RADIUS_M in
            # `_expire_road_cells`; the voxel store runs on to WORLD_RADIUS_M,
            # so the promoted half needs its own cull here -- and a TIGHTER one.
            # See WORLD_SURFACE_RADIUS_M: past ~36 m the ground rings are
            # further apart than WORLD_ROAD_BRIDGE_CELLS can close, so a
            # quarter-metre lattice out there is disconnected rings rather than
            # a surface. The road reaches further because it is driven along and
            # accumulation fills it in; the terrain beside it is never swept
            # that way.
            centres = (surface_keys + 0.5) * WORLD_COLUMN_SIZE_M
            offsets = centres - ego[:2]
            inside = (
                np.einsum("ij,ij->i", offsets, offsets)
                <= WORLD_SURFACE_RADIUS_M**2
            )
            centres = centres[inside]
            surface_height = surface_height[inside]
            surface_material = surface_material[inside]
        else:
            centres = np.empty((0, 2))
            surface_height = np.empty(0)
            surface_material = np.empty(0, dtype=np.uint8)

        promoted = np.column_stack(
            (
                np.floor(centres[:, 0] / WORLD_CELL_SIZE_M),
                np.floor(centres[:, 1] / WORLD_CELL_SIZE_M),
                np.floor(surface_height / _GROUND_LAYER_M),
            )
        ).astype(np.int32)

        keys = np.concatenate((promoted, self._road_keys))
        heights = np.concatenate((surface_height, self._road_height))
        materials = np.concatenate((surface_material, self._road_material))
        # Each half carries the radius it will dissolve at, because they end in
        # different places and a mesh that ended abruptly at either would read
        # as a cliff. Carried per cell rather than derived from the material,
        # so the two questions -- what is this made of, how far can it be seen --
        # stay independent.
        limits = np.concatenate(
            (
                np.full(len(promoted), WORLD_SURFACE_RADIUS_M),
                np.full(len(self._road_keys), WORLD_ROAD_RADIUS_M),
            )
        )
        if not len(keys):
            return keys, heights, materials, limits

        # Road appended LAST, so the stable sort's final entry per cell is the
        # road reading -- the same "newest wins" mechanic `_update_road_cells`
        # uses, doing duty here as "more specific wins".
        packed = pack_cell_keys(keys)
        order = np.argsort(packed, kind="stable")
        winner = order[_group_ends(packed[order])]
        return keys[winner], heights[winner], materials[winner], limits[winner]

    def _ground_mesh(
        self,
        snapshot: PerceptionSnapshot,
        surface_keys: np.ndarray,
        surface_height: np.ndarray,
        surface_material: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mesh the ground cells as a CONTINUOUS surface with shared corners.

        Still the WorldFrame's `road_*` channel, and still one draw: the QML
        binds that buffer once and the material shows up as vertex colour, so
        surfacing the rest of the world costs no extra geometry object.

        The road used to be `merge_cell_runs` rectangles, each drawn flat at the
        mean height of the cells that formed it. Neighbouring rectangles have no
        vertices in common, so every difference in mean height was a hard step
        and the surface read as a quilt of flat plates -- the "big pixels" a
        road is not made of. Sharing corners between cells makes the surface
        continuous by construction: a corner has ONE height, so the two cells
        that meet there cannot disagree.

        The merge is what kept the old vertex count survivable, and dropping it
        is only affordable because `world_view.SceneGeometry` now hands Qt the
        numpy buffer verbatim instead of building a QVector3D per vertex. There
        is no Python loop left here at all, which is why this is cheaper than
        the merge it replaces despite emitting far more geometry.
        """
        # No radius test here: `_expire_road_cells` culled to exactly
        # WORLD_ROAD_RADIUS_M earlier in `update`, so everything still in the
        # store is drawable.
        cells, heights, materials, limits = self._ground_cells(
            snapshot, surface_keys, surface_height, surface_material
        )
        if not len(cells):
            return _EMPTY_VERTICES.copy(), _EMPTY_COLOURS.copy(), _EMPTY_INDICES.copy()

        # Colour is resolved per CELL, before bridging, so the interpolation
        # below blends the two materials either side of a gap instead of trying
        # to interpolate a material CODE -- which has no meaningful midpoint.
        # It also means a material boundary is a gradient across one cell rather
        # than a hard seam, which is what a grass verge actually looks like.
        colour = _SURFACE_LINEAR[materials]
        # Close the sampling lattice before meshing. Ground returns thin as r^2
        # radially and as r in azimuth, so past roughly 20 m they stop reaching
        # every quarter-metre cell and the road breaks into a checkerboard of
        # disconnected quads -- which reads as a far worse "big pixels" than the
        # coarse grid it replaced. Bridging is applied to the MESH, never to the
        # store, so an inference is never accumulated as though it were an
        # observation.
        values = np.column_stack((heights, colour, limits))
        for axis in (0, 1):
            cells, values = bridge_gaps(
                cells, values, axis, WORLD_ROAD_BRIDGE_CELLS
            )
        heights, colour, limits = values[:, 0], values[:, 1:4], values[:, 4]

        # Each cell contributes its height to its own four lattice corners; a
        # corner's height is the mean of the cells touching it. The layer stays
        # in the corner key so a bridge deck and the road under it keep separate
        # surfaces instead of averaging into one ramp.
        offsets_xy = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int32)
        corner_keys = np.empty((len(cells), 4, 3), dtype=np.int32)
        corner_keys[:, :, :2] = cells[:, None, :2] + offsets_xy[None, :, :]
        corner_keys[:, :, 2] = cells[:, None, 2]
        flat_keys = corner_keys.reshape(-1, 3)

        # A corner holds ONE of each of these, which is the whole reason the
        # surface is continuous: two cells meeting there cannot disagree about
        # where it is, what it is made of, or how far away it fades. All five
        # quantities average over exactly the same cells.
        #
        # `bincount` per channel, and NOT the sort-once-and-reduceat idiom the
        # two accumulators use. That was tried here and is SLOWER -- 43.4 ms
        # against 37.4 on a 40k-cell disc -- because reducing five channels
        # together means materialising a (4N, 5) repeat and then gathering it
        # into sort order, and those two copies cost more than the extra sweeps
        # they save. The accumulators win with it because there the sort is the
        # thing being avoided; here `np.unique` has to sort anyway.
        #
        # Averaging colour in LINEAR space is not incidental: the buffer is
        # linear and the GPU multiplies there, so blending sRGB triples would
        # darken every material boundary rather than crossfade it.
        unique, first, inverse = np.unique(
            pack_cell_keys(flat_keys), return_index=True, return_inverse=True
        )
        counts = np.bincount(inverse, minlength=len(unique))

        def _corner_mean(values: np.ndarray) -> np.ndarray:
            return (
                np.bincount(
                    inverse, weights=np.repeat(values, 4), minlength=len(unique)
                )
                / counts
            )

        corner_height = _corner_mean(heights)
        corner_colour = np.stack(
            [_corner_mean(colour[:, channel]) for channel in range(3)], axis=1
        )
        corner_limit = _corner_mean(limits)
        corner_lattice = flat_keys[first]

        world = np.column_stack(
            (
                corner_lattice[:, 0] * WORLD_CELL_SIZE_M,
                corner_lattice[:, 1] * WORLD_CELL_SIZE_M,
                corner_height,
            )
        )
        vertices = world_to_render(world, snapshot)
        colours = depth_tint(corner_colour, vertices)
        # Dissolve the last few metres into the air. The road stops at its own
        # radius while everything else runs on to WORLD_RADIUS_M, so without
        # this the surface ends on a hard rim that reads as a cliff edge --
        # a drawn boundary where there is only the end of what was measured.
        colours = _fade_to_air(colours, vertices, corner_limit)

        quad = inverse.reshape(-1, 4).astype(np.uint32)
        indices = np.column_stack(
            (
                quad[:, 0],
                quad[:, 1],
                quad[:, 2],
                quad[:, 0],
                quad[:, 2],
                quad[:, 3],
            )
        ).reshape(-1)
        return vertices, colours, np.ascontiguousarray(indices)

    def _update_boundary_columns(self, snapshot: PerceptionSnapshot) -> None:
        """
        Fold this snapshot's boundary returns into world-anchored VOXELS.

        Held as parallel numpy arrays rather than a dict of dataclasses because
        the whole update is a bin-and-reduce: a Python loop over ten thousand
        voxels every frame would cost more than the meshing it feeds.

        Each voxel keeps the true min and max height of the returns inside it,
        not just the bin edges, so a 0.15 m kerb is still measured at 0.15 m
        after being binned at 0.25 m.
        """
        groups = snapshot.semantic_groups
        # SCENE_VEHICLE is folded in here rather than left to the actor path.
        # It used to be excluded, on the understanding that traffic would be
        # drawn as corroborated actor models instead -- but that path depends on
        # `vehicles.get_states()`, which BeamNG.tech REJECTS in free-roam (see
        # the comment on `worker._get_vehicle_state`). In the normal workflow no
        # actor is ever confirmed, so a car was drawn neither as a model nor as
        # a solid: it was invisible. Perception first -- a car is a solid that
        # the LiDAR saw, and the ground-truth model is enrichment on top.
        traffic_mask = groups == SCENE_VEHICLE
        static_mask = (groups == SCENE_BOUNDARY) | (groups == SCENE_VULNERABLE)
        selected = static_mask | traffic_mask
        # Class and material ride the same decimation, packed into one uint16 so
        # `_limit_pair` still takes a single companion array. They are two
        # different questions -- what kind of thing this is, and what the ground
        # here is made of -- and only runs the shape test promotes ever read the
        # second one.
        labels = (
            np.where(traffic_mask[selected], _TRAFFIC, _STATIC).astype(np.uint16)
            << 8
        ) | snapshot.surface_materials[selected].astype(np.uint16)
        points, labels = self._limit_pair(
            snapshot.points_world[selected], labels, WORLD_MAX_BOUNDARY_POINTS
        )
        classes = (labels >> 8).astype(np.uint8)
        materials = (labels & 0xFF).astype(np.uint8)
        points = points.astype(np.float64, copy=False)

        if len(points):
            keys = np.column_stack(
                (
                    np.floor(points[:, :2] / WORLD_COLUMN_SIZE_M),
                    np.floor(points[:, 2] / WORLD_COLUMN_HEIGHT_M),
                )
            ).astype(np.int32)
            heights = points[:, 2]
            seen = np.full(len(points), snapshot.timestamp)
            travel = np.full(len(points), self._travelled_m)
        else:
            keys = np.empty((0, 3), dtype=np.int32)
            heights = np.empty(0)
            seen = np.empty(0)
            travel = np.empty(0)
            classes = np.empty(0, dtype=np.uint8)
            materials = np.empty(0, dtype=np.uint8)

        all_keys = np.concatenate((self._voxel_keys, keys))
        if not len(all_keys):
            return
        all_low = np.concatenate((self._voxel_low, heights))
        all_high = np.concatenate((self._voxel_high, heights))
        all_seen = np.concatenate((self._voxel_seen, seen))
        all_travel = np.concatenate((self._voxel_travel, travel))
        all_class = np.concatenate((self._voxel_class, classes))
        all_material = np.concatenate((self._voxel_material, materials))

        # One sort, and the store is LEFT in that sorted order. pack_cell_keys
        # puts x in the high bits, then y, then the height bin, so sorted-by-key
        # is grouped by column with the bins ascending inside each one -- which
        # is exactly the order `_column_runs` needs, so it can find vertical
        # runs without sorting anything again.
        packed = pack_cell_keys(all_keys)
        order = np.argsort(packed, kind="stable")
        ordered_keys = all_keys[order]
        starts = _group_starts(packed[order])
        # Low and high are the extremes ever seen in the window, not this
        # frame's: the vertical FOV means a wall's observed top falls as you
        # approach it, and forgetting the earlier, taller look would make
        # buildings shrink as you drive at them.
        self._voxel_keys = ordered_keys[starts]
        self._voxel_low = np.minimum.reduceat(all_low[order], starts)
        self._voxel_high = np.maximum.reduceat(all_high[order], starts)
        # Both clocks, reduced the same way. Each is non-decreasing -- the
        # odometer stalls while parked but never runs backwards -- so the
        # maximum is still "most recently seen" under either measure.
        self._voxel_seen = np.maximum.reduceat(all_seen[order], starts)
        self._voxel_travel = np.maximum.reduceat(all_travel[order], starts)
        # _TRAFFIC outranks _STATIC, so a voxel a car has just moved into takes
        # the short vehicle TTL and cannot be pinned in place by an older static
        # reading of the same box.
        self._voxel_class = np.maximum.reduceat(all_class[order], starts)
        # Highest code wins, so any identified material beats SURFACE_UNKNOWN
        # (0) and a voxel that has ever been named keeps its name.
        self._voxel_material = np.maximum.reduceat(all_material[order], starts)

    def _expire_boundary_columns(self, snapshot: PerceptionSnapshot) -> None:
        if not len(self._voxel_keys):
            return
        ego = np.asarray(snapshot.ego_pos_world, dtype=np.float64)
        centres = (self._voxel_keys[:, :2] + 0.5) * WORLD_COLUMN_SIZE_M
        offsets = centres - ego[:2]
        # Per CLASS, and on different CLOCKS, because the two are different
        # kinds of thing. Scenery only improves with more looks and is forgotten
        # by the metre, so a stopped car keeps everything it has seen. Traffic
        # is forgotten by the second: accumulated over the scenery window a car
        # is drawn as a streak of itself, and one crossing in front of a STOPPED
        # ego would never fade at all if it shared the odometer.
        stale = np.where(
            self._voxel_class == _TRAFFIC,
            snapshot.timestamp - self._voxel_seen > WORLD_VEHICLE_TTL_S,
            self._travelled_m - self._voxel_travel > WORLD_COLUMN_MEMORY_M,
        )
        keep = (~stale) & (
            np.einsum("ij,ij->i", offsets, offsets) <= WORLD_RADIUS_M**2
        )
        if keep.all() and len(self._voxel_keys) <= WORLD_MAX_COLUMNS:
            return
        indices = np.flatnonzero(keep)
        if len(indices) > WORLD_MAX_COLUMNS:
            # Drop the oldest first, so what survives is what was seen most
            # recently rather than an arbitrary slice of the map.
            freshest = np.argsort(self._voxel_seen[indices])[-WORLD_MAX_COLUMNS:]
            indices = np.sort(indices[freshest])
        self._voxel_keys = self._voxel_keys[indices]
        self._voxel_material = self._voxel_material[indices]
        self._voxel_low = self._voxel_low[indices]
        self._voxel_high = self._voxel_high[indices]
        self._voxel_seen = self._voxel_seen[indices]
        self._voxel_travel = self._voxel_travel[indices]
        self._voxel_class = self._voxel_class[indices]

    def _column_runs(
        self, snapshot: PerceptionSnapshot
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collapse the voxel store into vertical RUNS of occupied height.

        This is where a tree stops being a solid block. Grass and terrain are
        boundary returns like everything that is not road, so the column under a
        canopy holds returns at ankle height and again at 3-7 m. The old store
        kept one (min, max) span per column and extruded it, which drew the
        canopy smeared down into the grass; runs keep the void, so the canopy is
        its own object and can be recognised as something you drive under.

        Returns the collision hazards -- `(keys (N, 2) column, lows, highs,
        classes)` -- followed by the GROUND SURFACE the short runs describe:
        `(keys (M, 2), heights (M,), materials (M,))`. Overhead structure is
        dropped; nothing else is.

        **Separating the short ones HERE rather than after bridging is what
        keeps the build inside the tick.** `_slab_mesh` used to bridge every run
        and
        then discard the ones under WORLD_MIN_SLAB_HEIGHT_M, so on open ground --
        where the whole surface is boundary-classified rather than road -- it
        bridged tens of thousands of flat ground runs across two axes and threw
        nearly all of them away. Measured on a synthetic street with striped
        facades, kerbs and open ground: **69.1 ms -> 34.7 ms**, with every
        structure over a metre tall preserved exactly (41 boxes either way).
        What disappears is short ground fragments that bridging was inventing --
        interpolating between a flat ground column and a kerb manufactures a
        ramp nobody observed, so bridging only ever joining structure to
        structure is also the more honest rule.
        """
        if not len(self._voxel_keys):
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty(0),
                np.empty(0),
                np.empty(0, dtype=np.uint8),
                np.empty((0, 2), dtype=np.int32),
                np.empty(0),
                np.empty(0, dtype=np.uint8),
            )

        # No sort here: `_update_boundary_columns` leaves the store ordered by
        # packed (x, y, bin), which groups it by column with the bins ascending
        # inside each one, and expiry only ever drops rows.
        column = pack_cell_keys(self._voxel_keys[:, :2])
        height_bin = self._voxel_keys[:, 2]

        # A run breaks at a new column, or at a vertical gap too wide to be
        # sampling noise. One vectorised pass, as everywhere else here.
        breaks = np.empty(len(column), dtype=bool)
        breaks[0] = True
        breaks[1:] = (column[1:] != column[:-1]) | (
            height_bin[1:] - height_bin[:-1] > WORLD_COLUMN_VERTICAL_BRIDGE_BINS + 1
        )
        starts = np.flatnonzero(breaks)
        lows = np.minimum.reduceat(self._voxel_low, starts)
        highs = np.maximum.reduceat(self._voxel_high, starts)
        keys = self._voxel_keys[starts][:, :2]
        # A run holding any traffic voxel is traffic: a car standing in front of
        # a wall must not be absorbed into the wall's colour.
        classes = np.maximum.reduceat(self._voxel_class, starts)

        # Ground reference per column: the lowest return the sensors put in it,
        # so the test follows terrain instead of assuming the map is flat. Where
        # a structure hid its own footing -- under a canopy, or over a road,
        # which is not a boundary return at all -- there is nothing local to use
        # and the ego's ground plane stands in.
        run_column = column[starts]
        column_breaks = np.empty(len(starts), dtype=bool)
        column_breaks[0] = True
        column_breaks[1:] = run_column[1:] != run_column[:-1]
        column_starts = np.flatnonzero(column_breaks)
        column_floor = np.minimum.reduceat(lows, column_starts)
        floor_per_run = np.repeat(
            column_floor, np.diff(np.append(column_starts, len(starts)))
        )

        ego_ground = (
            snapshot.ego_pos_world[2] + snapshot.vehicle_geometry.ground_z_vehicle
        )
        grounded = floor_per_run <= ego_ground + WORLD_COLLISION_CEILING_M
        reference = np.where(grounded, floor_per_run, ego_ground)
        tall = highs - lows >= WORLD_MIN_SLAB_HEIGHT_M
        keep = (lows - reference < WORLD_COLLISION_CEILING_M) & tall

        # ...and the runs too short to be structure are the GROUND, not rubbish.
        # They were discarded here, which is why anything the annotations did not
        # call road -- grass, dirt, sand, a gravel yard, the whole of an
        # unannotated map -- rendered as nothing at all rather than as a surface:
        # too flat to be a slab, and not road, so no store wanted it.
        #
        # Two conditions, and the second is what stops a rooftop or the underside
        # of a canopy becoming ground: the run must be the LOWEST in its column.
        # There is deliberately no ceiling test -- a hillside 10 m above the ego
        # plane is still a surface, and `grounded` is asking whether something
        # could be driven into rather than whether it can be stood on.
        #
        # The threshold is WORLD_MIN_SLAB_HEIGHT_M itself rather than a second
        # constant, so this promotes EXACTLY what was being thrown away and
        # nothing that currently draws as a slab changes. A 0.12 m kerb is still
        # structure the planner steers around, not floor.
        surface = (~tall) & (lows == floor_per_run)
        return (
            keys[keep],
            lows[keep],
            highs[keep],
            classes[keep],
            keys[surface],
            0.5 * (lows[surface] + highs[surface]),
            self._voxel_material[starts][surface],
        )

    @staticmethod
    def _bridge_column_gaps(
        keys: np.ndarray, base: np.ndarray, top: np.ndarray, axis: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bridge azimuth stripe gaps between boundary columns. See `bridge_gaps`."""
        bridged, values = bridge_gaps(
            keys,
            np.column_stack((base, top)),
            axis,
            WORLD_COLUMN_BRIDGE_CELLS,
        )
        return bridged, values[:, 0], values[:, 1]

    def _solid_meshes(
        self,
        snapshot: PerceptionSnapshot,
        keys_2d: np.ndarray,
        base: np.ndarray,
        top: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """
        The static scenery and the traffic, as two independently coloured meshes.

        Split at the RUN level rather than meshed together and recoloured,
        because merging must not cross the boundary: a car parked against a wall
        would otherwise merge into it and take one colour for both, which is the
        one thing hue is carrying here.
        """
        return tuple(
            self._slab_mesh(
                snapshot,
                keys_2d[classes == kind],
                base[classes == kind],
                top[classes == kind],
                shadow,
                lit,
            )
            for kind, shadow, lit in (
                (_STATIC, _BOUNDARY_LINEAR, _BOUNDARY_LIT_LINEAR),
                (_TRAFFIC, _VEHICLE_LINEAR, _VEHICLE_LIT_LINEAR),
            )
        )

    def _slab_mesh(
        self,
        snapshot: PerceptionSnapshot,
        keys_2d: np.ndarray,
        base: np.ndarray,
        top: np.ndarray,
        shadow_linear: np.ndarray,
        lit_linear: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extrude one class of voxel runs into merged, face-shaded slabs."""
        if not len(keys_2d):
            return _EMPTY_VERTICES.copy(), _EMPTY_COLOURS.copy(), _EMPTY_INDICES.copy()

        # Layer by BOTH altitude and height so neither a facade and the kerb in
        # front of it, nor a wall and a balcony above it, can average together.
        # `merge_cell_runs` only ever merges within one layer value, so any
        # injective combination of the two buckets works as the key.
        altitude = np.floor(base / WORLD_SLAB_HEIGHT_BUCKET_M).astype(np.int64)
        span = np.floor((top - base) / WORLD_SLAB_HEIGHT_BUCKET_M).astype(np.int64)
        layers = (
            (altitude - altitude.min()) * 4096 + np.clip(span, 0, 4095)
        ).astype(np.int32)

        keys = np.column_stack((keys_2d, layers))
        for axis in (0, 1):
            keys, base, top = self._bridge_column_gaps(keys, base, top, axis)

        # No height cull here: `_column_runs` already dropped everything under
        # WORLD_MIN_SLAB_HEIGHT_M, and doing it there rather than here is what
        # keeps bridging off the flat ground -- see its docstring for the
        # measurement.
        rectangles, extents = merge_cell_runs(
            keys, np.column_stack((base, top)), WORLD_COLUMN_SIZE_M
        )
        if not len(rectangles):
            return _EMPTY_VERTICES.copy(), _EMPTY_COLOURS.copy(), _EMPTY_INDICES.copy()

        x_min, x_max, y_min, y_max = rectangles.T
        low, high = extents.T
        corners_x = np.stack(
            (x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min), axis=1
        )
        corners_y = np.stack(
            (y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max), axis=1
        )
        corners_z = np.stack((low, low, low, low, high, high, high, high), axis=1)
        corners = np.stack((corners_x, corners_y, corners_z), axis=2)

        # Explode each box into six independent quads so every face can hold its
        # own shade; a shared corner belongs to three faces and can only carry
        # one colour.
        faces = corners[:, _BOX_FACE_CORNERS.reshape(-1), :]
        world = faces.reshape(-1, 3)
        vertices = world_to_render(world, snapshot)

        right, forward = _basis(snapshot)
        shade = np.repeat(face_shades(right, forward), 4)
        shade = np.tile(shade, len(rectangles))[:, None]
        colour = shadow_linear + shade * (lit_linear - shadow_linear)
        colours = depth_tint(colour, vertices)

        quad = np.arange(len(rectangles) * 6, dtype=np.uint32) * 4
        indices = (quad[:, None] + _FACE_TRIANGLES[None, :]).reshape(-1)
        return vertices, colours, np.ascontiguousarray(indices)

    @staticmethod
    def _aeb_meshes(
        snapshot: PerceptionSnapshot,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        The emergency-braking overlay: a corridor wash, then the threat marker.

        Both are built from `AebState`'s OWN curvature and half-width through
        `aeb.predicted_corridor`, so what is drawn is provably the corridor that
        was scanned rather than a redrawing of it -- the same reason
        `bev_widget` imports the function instead of reimplementing the arc.

        Filled only as far as the blockage, never to the horizon: filling to the
        horizon says "all 34 m of this is dangerous" when the thing that matters
        is the one point about to be hit.

        Returned as two meshes because a wash and a wall cannot share an
        opacity -- at one value either the corridor hides the road under it or
        the marker is too faint to read as an impact.
        """
        corridor: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        markers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        ground = snapshot.vehicle_geometry.ground_z_vehicle + WORLD_AEB_GROUND_OFFSET_M
        detail = ground + WORLD_AEB_DETAIL_OFFSET_M

        for state in (snapshot.aeb, snapshot.rear_aeb):
            if state is None or state.status == STANDBY:
                continue
            braking = state.status == BRAKING
            colour = _AEB_BRAKING_LINEAR if braking else _AEB_ARMED_LINEAR
            half_width = state.corridor_half_width_m

            # RAILS run the full scanned length, not to the threat: the extent of
            # the scan is information in its own right, and a filled wash cannot
            # express "clear, and I checked this far".
            scanned = float(max(state.horizon_m, 1.0))
            for side in (1, -1):
                corridor.append(_rail(state, detail, colour, scanned, side))

            # The WASH exists only when there is something to brake FOR, and it
            # stops at that thing. A clear corridor gets rails and nothing else:
            # filling the scanned length says "all 48 m of this is dangerous",
            # which is the same mistake as scoring an empty road as an obstacle
            # parked at the horizon -- and it read as a solid white band down the
            # whole road, drowning the scene the overlay sits on.
            if state.threat_m is not None:
                filled = float(max(state.threat_m, 1.0))
                right, left = corridor_edges(
                    state.curvature, filled, half_width, 48
                )
                progress = np.linspace(0.0, filled, len(right)) / filled
                # Urgency scales the wash, so the corridor visibly builds as a
                # threat closes instead of snapping on at the moment of firing.
                urgency = (
                    1.0
                    if braking
                    else float(
                        np.clip(
                            state.required_decel_mps2 / AEB_BRAKING_DECEL_MPS2,
                            0.0,
                            1.0,
                        )
                    )
                )
                floor = WORLD_AEB_URGENCY_FLOOR
                strength = floor + (1.0 - floor) * urgency
                if braking:
                    strength *= WORLD_AEB_BRAKING_BOOST
                span = WORLD_AEB_WASH_ALPHA_FAR - WORLD_AEB_WASH_ALPHA_NEAR
                wash = np.clip(
                    (WORLD_AEB_WASH_ALPHA_NEAR + span * progress) * strength,
                    0.0,
                    1.0,
                )
                corridor.append(
                    _strip(
                        right,
                        left,
                        (ground, ground),
                        colour,
                        wash.astype(np.float32),
                        state.rearward,
                    )
                )

            # The BRAKE-NOW bar: the last point at which braking still works, and
            # the trigger the whole system turns on. The gap between it and the
            # threat IS the margin left, which no other element shows.
            if 1.0 < state.brake_now_m < scanned:
                corridor.append(
                    _band(
                        state,
                        detail,
                        colour,
                        state.brake_now_m,
                        state.brake_now_m + WORLD_AEB_BAR_THICKNESS_M,
                        half_width,
                        WORLD_AEB_BAR_ALPHA,
                    )
                )

            if state.threat_m is None:
                continue
            threat = float(state.threat_m)
            height = WORLD_AEB_MARKER_HEIGHT_M
            markers.extend(
                (
                    _panel(
                        state,
                        ground,
                        colour,
                        threat,
                        height,
                        (WORLD_AEB_MARKER_ALPHA_BASE, WORLD_AEB_MARKER_ALPHA_TOP),
                    ),
                    _post(state, ground, colour, threat, height, 1),
                    _post(state, ground, colour, threat, height, -1),
                    _lintel(state, ground, colour, threat, height),
                    # A pool of light where the panel meets the road, so the
                    # marker is anchored to a place on the surface rather than
                    # hanging in the air in front of one.
                    _band(
                        state,
                        detail,
                        colour,
                        max(threat - WORLD_AEB_POOL_LENGTH_M, 0.05),
                        threat,
                        half_width,
                        WORLD_AEB_POOL_ALPHA,
                    ),
                )
            )
        return _combine(corridor), _combine(markers)

    def _uncertain_points(
        self, snapshot: PerceptionSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self._limit(
            snapshot.points_world[snapshot.semantic_groups == SCENE_UNKNOWN],
            WORLD_MAX_UNCERTAIN_POINTS,
        )
        vertices = world_to_render(points, snapshot)
        return vertices, depth_tint(_UNCERTAIN_LINEAR, vertices)

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
        snapshot: PerceptionSnapshot, alert: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plan = snapshot.plan
        if plan is None:
            return _EMPTY_VERTICES.copy(), _EMPTY_COLOURS.copy(), _EMPTY_INDICES.copy()
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
        # Deliberately NOT depth-tinted, and the one surface that is not. The
        # path is a guidance overlay rather than something perceived -- hazing
        # its far end would fade out exactly the part that says where the car is
        # going -- so it keeps one flat colour over its whole length, the same
        # reason it breaks the luminance ramp.
        colour = _PATH_ALERT_LINEAR if alert else _PATH_LINEAR
        colours = np.tile(
            np.asarray((*colour, 1.0), dtype=np.float32), (len(vertices), 1)
        )
        return vertices, np.ascontiguousarray(colours), indices

    def _camera(
        self, snapshot: PerceptionSnapshot, alert: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Chase camera: swings round when reversing, pulls back with speed, tilts
        toward top-down at a standstill, leans into a bend, and stands off on a
        full brake so the threat and the ego are framed together.

        Reversing is read from the SIGNED forward speed, not from the plan.
        `plan` is None whenever self-driving is off -- which is exactly when a
        human is doing the reversing -- so keying the swing on
        `plan.command.mode == "REVERSING"` meant the camera only ever turned
        round for the autonomous reverse recovery, and never for a driver
        selecting reverse themselves.

        Everything is DAMPED toward its target. Nothing was, which is why the
        reverse swing teleported, and why any of the states above would have
        snapped in and out. The first frame still lands exactly on its target,
        so the pose is never a blend of a real scene and an initial guess.

        The alert gate is `_alert`'s own string, so the framing move and the
        overlay agree by construction about what an event is: it is non-empty
        only while a pedal is actually down.
        """
        dt = 0.0
        if self._camera_at is not None:
            dt = max(0.0, float(snapshot.timestamp) - self._camera_at)
        self._camera_at = float(snapshot.timestamp)

        target = camera_target(snapshot, bool(alert))
        if self._camera_pose is None:
            pose = target
        else:
            current = self._camera_pose
            pose = CameraPose(
                height_m=damp(
                    current.height_m, target.height_m, dt, WORLD_CAM_TAU_S
                ),
                distance_m=damp(
                    current.distance_m, target.distance_m, dt, WORLD_CAM_TAU_S
                ),
                pitch_deg=damp(
                    current.pitch_deg, target.pitch_deg, dt, WORLD_CAM_TAU_S
                ),
                yaw_deg=damp(
                    current.yaw_deg, target.yaw_deg, dt, WORLD_CAM_YAW_TAU_S
                ),
            )
        self._camera_pose = pose

        # The position is DERIVED from the orbit angle, which is what turns the
        # reverse flip into a sweep round the side of the car rather than a jump
        # through it. At yaw 0 this is (0, h, +d) and at 180 it is (0, h, -d),
        # exactly the two poses the fixed version had.
        yaw = math.radians(pose.yaw_deg)
        return (
            (
                pose.distance_m * math.sin(yaw),
                pose.height_m,
                pose.distance_m * math.cos(yaw),
            ),
            (pose.pitch_deg, pose.yaw_deg, 0.0),
        )

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

    @classmethod
    def _limit_pair(
        cls, points: np.ndarray, labels: np.ndarray, limit: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decimate points and their per-point labels with the SAME stride.

        Separate arrays rather than one structured one, but they must be thinned
        together or every label lands on the wrong point -- traffic voxels would
        be scattered arbitrarily through the scenery.
        """
        if len(points) <= limit:
            return points, labels
        stride = max(1, math.ceil(len(points) / limit))
        return points[::stride][:limit], labels[::stride][:limit]
