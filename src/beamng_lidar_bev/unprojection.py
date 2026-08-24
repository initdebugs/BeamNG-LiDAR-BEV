"""
Rung 0.5 of the vision ladder: the engine's depth image, unprojected.

Turns one camera's depth and annotation buffers into the perception waist the
whole downstream stack already consumes -- `points_world (N, 3)` plus
`colours (N, 3)` -- through a per-camera ray lookup table built once at
attach. Pure: config, models, geometry and numpy only, no Qt and no BeamNGpy,
exactly like `planner` and `aeb`, so every piece of the arithmetic is pinned
offline against synthetic depth images in `tests/test_unprojection.py`.

Phase 1's verdict (docs/VISION_ROADMAP.md) made this the PERMANENT source of
the ground band: computed stereo resolved a kerb at 15 m and nowhere beyond
it, so kerbs and the road surface keep coming from here whatever later rungs
do about obstacles. Build quality accordingly -- this is not scaffolding.

Four facts the module is built on, all measured live:

* **Depth is PLANAR Z, not radial range**, decoded as `raw_float32 x far
  plane` in linear metres. A planar depth multiplies the UNNORMALISED ray
  `(x, y, 1)`; multiplying a unit ray instead pulls every off-axis sample
  toward the lens and the ground bows into a bowl. `test_a_flat_floor_comes_
  back_flat_to_the_edge_of_the_frame` pins it.
* **Sky and anything past the far plane come back AT the far plane**, so a
  sample near it is not a surface and is culled before any transform.
* **`pos.z` is referenced to the vehicle's ground plane** (the body-frame
  bounding-box bottom), not to the reference node -- the same rule every
  LiDAR mount follows, see `derive_vehicle_geometry`.
* **The per-camera frames carry no timestamp** and are staged a frame or two
  behind. The worker measures the part of each frame's age it can see (how
  long since the buffer changed) and this module places the cloud from the
  pose the car had THEN, so a 40 km/h car does not smear a wall 0.4 m down
  the road between two cameras. The fixed staging part is
  `CAMERA_FRAME_STAGING_S`, unmeasured as of 2026-08-23 and zero until
  tools/unprojection_oracle.py measures it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    CAMERA_DEPTH_FAR_FRACTION,
    CAMERA_DEPTH_MIN_M,
    CAMERA_NEAR_FAR_PLANES,
    LIDAR_RANGE_M,
)
from .geometry import camera_basis, rotate_about_up, vec3, vehicle_axes
from .models import CameraMount, VehicleGeometry

_EMPTY_POINTS = np.empty((0, 3), dtype=np.float32)
_EMPTY_COLOURS = np.empty((0, 3), dtype=np.uint8)


@dataclass(frozen=True)
class CameraRays:
    """
    One camera's sample lattice and ray table, built once per attach.

    `directions` are in the VEHICLE frame and are deliberately NOT unit
    vectors: each is `axis + x * right - y * up` with the optical-axis
    component exactly 1, so a planar depth `Z` places the point at
    `origin + Z * direction` with no per-sample normalisation or cosine
    divide -- the divide is baked into the table.
    """

    name: str
    pixel_index: np.ndarray
    """(K,) int64 flat indices into the row-major H x W image."""
    directions: np.ndarray
    """(K, 3) float32 rays, vehicle frame, optical-axis component 1."""
    origin_vehicle: np.ndarray
    """(3,) float64 mount position, vehicle frame, z from the ground plane."""
    resolution: tuple[int, int]
    focal_px: float
    sample_stride: tuple[int, int]
    image_right_vehicle: np.ndarray
    """(3,) float64 unit vector the image's column index increases along."""

    @property
    def sample_count(self) -> int:
        return int(self.pixel_index.shape[0])


def focal_length_px(horizontal_fov_deg: float, width: int) -> float:
    """Pinhole focal length in pixels from a horizontal aperture."""
    return (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)


def build_camera_rays(mount: CameraMount) -> CameraRays:
    """
    The strided sample lattice and its ray table for one mount.

    Sampled at pixel CENTRES (`index + 0.5`) against a principal point at the
    image centre: the simulator's camera is a plain pinhole with no lens
    model, which the kerb experiment relied on (a pair is rectified by
    construction) and which is what makes a closed-form table correct.

    The lattice starts half a stride in so that it is centred in the image
    rather than hugging its top-left edge -- a detail that matters only for
    the one camera where rows are range (front_main), where it keeps the
    lowest sampled row the same distance from the frame edge as the highest.
    """
    width, height = mount.resolution
    col_stride, row_stride = mount.sample_stride
    if width <= 0 or height <= 0:
        raise ValueError(f"Implausible camera resolution {mount.resolution!r}")
    if col_stride <= 0 or row_stride <= 0:
        raise ValueError(f"Implausible sample stride {mount.sample_stride!r}")

    focal = float(focal_length_px(mount.horizontal_fov_deg, width))
    cols = np.arange(col_stride // 2, width, col_stride, dtype=np.int64)
    rows = np.arange(row_stride // 2, height, row_stride, dtype=np.int64)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    flat_rows = grid_rows.reshape(-1)
    flat_cols = grid_cols.reshape(-1)
    pixel_index = flat_rows * width + flat_cols

    # Normalised image-plane coordinates: x right, y DOWN (rows grow
    # downward), at unit distance along the optical axis.
    x = (flat_cols.astype(np.float64) + 0.5 - width / 2.0) / focal
    y = (flat_rows.astype(np.float64) + 0.5 - height / 2.0) / focal

    right, up, axis = camera_basis(mount)
    directions = (
        axis[None, :]
        + x[:, None] * right[None, :]
        - y[:, None] * up[None, :]
    ).astype(np.float32)

    return CameraRays(
        name=mount.name,
        pixel_index=np.ascontiguousarray(pixel_index),
        directions=np.ascontiguousarray(directions),
        origin_vehicle=np.asarray(mount.position_vehicle, dtype=np.float64),
        resolution=(int(width), int(height)),
        focal_px=focal,
        sample_stride=(int(col_stride), int(row_stride)),
        image_right_vehicle=right,
    )


def build_rig_rays(rig: Mapping[str, CameraMount]) -> dict[str, CameraRays]:
    return {name: build_camera_rays(mount) for name, mount in rig.items()}


def sample_depth(
    raw: Any,
    rays: CameraRays,
    far_plane_m: float = CAMERA_NEAR_FAR_PLANES[1],
) -> np.ndarray | None:
    """
    Planar depth in metres at the lattice pixels, gathered straight from the
    live buffer -- one vectorised read, never a full-frame copy.

    None when the buffer is absent, the wrong size, or still zero-filled (a
    streaming camera whose first frame has not landed yet).
    """
    if raw is None:
        return None
    width, height = rays.resolution
    expected = width * height * 4
    if len(raw) != expected:
        return None
    values = np.frombuffer(raw, dtype=np.float32)
    depth = values[rays.pixel_index] * np.float32(far_plane_m)
    if not depth.any():
        return None
    return depth


def sample_annotation(
    raw: Any, rays: CameraRays, keep: np.ndarray | None = None
) -> np.ndarray | None:
    """
    RGB at the lattice pixels -- or only at the `keep` subset of them --
    as (K, 3) uint8; None if the buffer is unusable.

    Gathered as ONE 32-bit word per pixel and split into bytes afterwards:
    a byte-wise gather of three channels through a (N, 4) view was measured
    at 0.66 ms per camera against 0.16 here, and the tick pays it eight
    times.
    """
    if raw is None:
        return None
    width, height = rays.resolution
    if len(raw) != width * height * 4:
        return None
    index = rays.pixel_index if keep is None else rays.pixel_index[keep]
    words = np.frombuffer(raw, dtype=np.uint32)[index]
    return np.ascontiguousarray(words.view(np.uint8).reshape((-1, 4))[:, :3])


def surface_mask(
    depth_m: np.ndarray,
    far_plane_m: float = CAMERA_NEAR_FAR_PLANES[1],
    max_range_m: float = LIDAR_RANGE_M,
) -> np.ndarray:
    """
    Which samples are a surface the camera actually saw.

    Sky and everything beyond the far plane arrive AT the far plane; a
    sample inside `CAMERA_DEPTH_MIN_M` is the car's own bodywork (the bumper
    camera sees bonnet, the repeaters see fender). Anything past the LiDAR
    cull radius is dropped here too so it is never transformed at all.
    """
    ceiling = min(float(far_plane_m) * CAMERA_DEPTH_FAR_FRACTION, float(max_range_m))
    return (
        np.isfinite(depth_m)
        & (depth_m > CAMERA_DEPTH_MIN_M)
        & (depth_m < ceiling)
    )


@dataclass(frozen=True)
class VehiclePose:
    """Where the car was when a frame was rendered, in world coordinates."""

    origin_world: np.ndarray
    right_world: np.ndarray
    forward_world: np.ndarray
    up_world: np.ndarray


def pose_from_state(
    state: Mapping[str, Any], age_s: float = 0.0, yaw_rate_rps: float = 0.0
) -> VehiclePose:
    """
    The vehicle pose `age_s` ago, by the same linear extrapolation the
    worker's prefetched state poll already uses in the other direction:
    the position rewound by velocity x age, the heading by yaw rate x age.

    The heading half matters more than it looks. A frame placed with a
    heading it was not rendered at is rotated about the car before it is
    stamped into WORLD's world-anchored stores, and a turn is precisely when
    a frame's age is worth the most: 100 ms at 30 deg/s is 3 degrees, which
    at 30 m is 1.6 m of sideways error that the store then remembers for 25 m
    of travel. Pitch and roll are left as they are.
    """
    right, forward, up = vehicle_axes(state)
    origin = vec3(state["pos"])
    if age_s > 0.0:
        origin = origin - vec3(state.get("vel", (0.0, 0.0, 0.0))) * float(age_s)
        if yaw_rate_rps != 0.0:
            rewind = -float(yaw_rate_rps) * float(age_s)
            right = rotate_about_up(right, rewind)
            forward = rotate_about_up(forward, rewind)
    return VehiclePose(origin, right, forward, up)


def unproject_camera(
    rays: CameraRays,
    depth_m: np.ndarray,
    pose: VehiclePose,
    geometry: VehicleGeometry,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """
    World-space points for one camera's sampled depth, (N, 3) float32.

    `keep` selects which lattice samples to place (the surface mask); the
    caller gathers the matching annotation rows with the same mask so the
    two stay parallel. Vehicle frame is +X left, +Y rearward, +Z up, so the
    world basis is `(-right, -forward, up)`; the mount's `pos.z` is measured
    from the body-frame floor, which `geometry.sensor_floor_z` places
    relative to the reference node.
    """
    if keep is not None:
        directions = rays.directions[keep]
        depth = depth_m[keep]
    else:
        directions = rays.directions
        depth = depth_m
    if not len(depth):
        return _EMPTY_POINTS

    # float32 end to end, like the LiDAR cloud the simulator hands over in
    # float32 world coordinates: a direction component is under 3 and a depth
    # under 200 m, so the per-point product resolves to ~1e-5 relative, and
    # the only float64 quantity -- the world origin -- is added once at the
    # end. Promoting the whole gather to float64 first was measured at 2.4 ms
    # for one camera against 1.3 here, on 43k surviving samples; the whole
    # eight-camera rig went 12.6 ms -> 5.5 per tick with this and the
    # one-word annotation gather together.
    local = depth[:, None] * directions
    local += (
        rays.origin_vehicle + (0.0, 0.0, geometry.sensor_floor_z)
    ).astype(np.float32)[None, :]
    # Vehicle frame -> world: +X left, +Y rearward, +Z up, with z measured
    # from the floor plane rather than the node.
    basis = np.stack(
        (-pose.right_world, -pose.forward_world, pose.up_world)
    ).astype(np.float32)
    world = local @ basis
    world += pose.origin_world.astype(np.float32)[None, :]
    return np.ascontiguousarray(world, dtype=np.float32)


def unproject_frame(
    rays: CameraRays,
    depth_raw: Any,
    annotation_raw: Any,
    pose: VehiclePose,
    geometry: VehicleGeometry,
    unknown_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """
    One camera, one tick: `(points_world, colours, sampled)` or None when
    the depth buffer is unusable. `sampled` is the lattice size that passed
    the surface mask, for the reach line.

    A missing annotation buffer paints every point `unknown_rgb`, which the
    semantic classifier treats as unannotated -- the height-band fallback
    then decides road from shape, exactly as it does on an unannotated map.
    """
    depth = sample_depth(depth_raw, rays)
    if depth is None:
        return None
    keep = surface_mask(depth)
    if not keep.any():
        return _EMPTY_POINTS, _EMPTY_COLOURS, 0
    points = unproject_camera(rays, depth, pose, geometry, keep)
    colours = sample_annotation(annotation_raw, rays, keep)
    if colours is None:
        colours = np.tile(unknown_rgb.astype(np.uint8), (len(points), 1))
    return points, colours, int(len(points))
