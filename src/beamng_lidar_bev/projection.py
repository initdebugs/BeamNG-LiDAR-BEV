"""
Between a camera pixel and a point on the ground, both ways.

This is the piece the whole vision plan waits on. A model that reads bay paint
out of the A-pillar images is worth nothing until a predicted pixel can be
placed on the road, and the 177 hand-clicked bays are unverifiable until they
can be drawn back onto the frames that saw them. One module answers both:

- `project` takes world points to pixels. That is what draws a label onto a
  recorded frame, which is the ONLY proof the labels are where they are
  believed to be.
- `ground_points` takes pixels to world XY through a ground plane. That is the
  direction a paint model's output travels.

**It is not the deleted `unprojection.py`.** That module turned the engine's
DEPTH buffer into a cloud and went with the rest of VISION mode; what is
recovered from it here is the part that was about the CAMERA rather than about
depth -- `camera_basis` and the pinhole focal length, both of which were
measured live against the LiDAR cloud before they were deleted. Everything
about depth decoding is gone, deliberately: HYBRID's cameras render colour
only, and there is no depth to decode.

Three facts it is built on, and each has a measurement behind it in CLAUDE.md
rather than an assumption:

* **A vehicle-frame vector is `-x * right - y * forward + z * up` in world.**
  BeamNG's vehicle convention is +X LEFT, +Y REARWARD, +Z up, which is exactly
  what `derive_vehicle_geometry` builds its extents from (`left = -right`,
  `rearward = -forward`). Both flips, or neither.
* **A mount does not sit where its `pos` says.** The simulator reads a
  vehicle-space sensor `pos` in its own frame, offset from the reference node
  by `VehicleGeometry.sensor_origin_vehicle` -- measured live at
  (+0.160, +0.362, -0.233) on the vivace, a pure translation stable to 0.1 mm.
  `place_camera` adds it back on ALL THREE axes, because that is what the probe
  measured; note `derive_hybrid_camera_rig` subtracts it on x and y only, so
  the two together put the camera exactly where the rig asked.
* **Image v runs DOWN.** The camera basis is built with image-up as a vector,
  so the row index is `cy - f * y/z`, not `cy + f * y/z`. A sign error here
  flips every projection about the horizon and is obvious the moment a label is
  drawn -- which is why the label overlay exists before anything is trained.

Pure like `planner`, `aeb`, `parking` and `capture`: config, models, geometry
and numpy. No Qt, no BeamNGpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import vec3, vehicle_axes
from .models import CameraMount


def focal_length_px(horizontal_fov_deg: float, width: int) -> float:
    """Pinhole focal length in pixels from a horizontal aperture.

    Recovered verbatim from the deleted `unprojection.py`, where it was checked
    against the LiDAR cloud on a live scene.
    """
    return (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)


def camera_basis_vehicle(
    direction_vehicle: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (image right, image up, optical axis) in the VEHICLE frame.

    Recovered from the deleted `unprojection.py`. Image right is
    `axis x up`, which gives the same handedness `vehicle_axes` uses, so a
    point to the camera's right lands on the vehicle's right: for an axis of
    (0, -1, 0) -- dead ahead, since +Y is rearward -- this returns (-1, 0, 0),
    and -x is the vehicle's right.

    The vehicle basis (left, rearward, up) is right-handed, the same as
    (right, forward, up), so an ordinary cross product is correct in it.
    """
    axis = _unit(np.asarray(direction_vehicle, dtype=np.float64), "camera dir")
    up_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = _unit(np.cross(axis, up_axis), "camera right")
    # Re-orthogonalised against a PITCHED axis, exactly as `vehicle_axes` does
    # for the body: the A-pillar cameras are pitched down, and skipping this
    # shears every column of the image.
    up = _unit(np.cross(right, axis), "camera up")
    return right, up, axis


@dataclass(frozen=True)
class CameraPlacement:
    """One camera's pinhole model, fully in WORLD coordinates.

    Built per frame, because the ego moves. Everything downstream reads only
    this, so nothing else has to know about vehicle frames or sensor origins.
    """

    origin: np.ndarray
    """(3,) the lens position in world metres."""
    right: np.ndarray
    """(3,) unit, the direction image COLUMNS increase along."""
    up: np.ndarray
    """(3,) unit, image up -- so the row index runs the other way."""
    axis: np.ndarray
    """(3,) unit optical axis."""
    focal_px: float
    resolution: tuple[int, int]

    @property
    def centre_px(self) -> tuple[float, float]:
        width, height = self.resolution
        return (width / 2.0, height / 2.0)


def place_camera(
    state: Mapping[str, Any],
    mount: CameraMount,
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraPlacement:
    """
    Put a mount into the world, from the pose the car had when it was shot.

    `state` is BeamNG's own dict -- `pos`, `dir`, `up` -- which is exactly what
    `capture.EgoPose` records per sample, so a recorded frame and a live tick
    are placed by the same code.

    `sensor_origin` is `VehicleGeometry.sensor_origin_vehicle`, recorded into
    each session's `meta.json`. It is added on all three axes: the probe
    measured a pure translation, and `derive_hybrid_camera_rig` has already
    subtracted it on x and y, so the pair cancels to the intended station.
    """
    origin_world = vec3(state["pos"])
    right_w, forward_w, up_w = vehicle_axes(state)

    def to_world(vector: np.ndarray) -> np.ndarray:
        # +X left, +Y rearward, +Z up -- both of the first two flip.
        return (
            -float(vector[0]) * right_w
            - float(vector[1]) * forward_w
            + float(vector[2]) * up_w
        )

    station = np.asarray(mount.position_vehicle, dtype=np.float64) + np.asarray(
        sensor_origin, dtype=np.float64
    )
    right_v, up_v, axis_v = camera_basis_vehicle(mount.direction_vehicle)
    width, _height = mount.resolution
    return CameraPlacement(
        origin=origin_world + to_world(station),
        right=_unit(to_world(right_v), "world camera right"),
        up=_unit(to_world(up_v), "world camera up"),
        axis=_unit(to_world(axis_v), "world camera axis"),
        focal_px=focal_length_px(mount.horizontal_fov_deg, width),
        resolution=mount.resolution,
    )


def project(
    placement: CameraPlacement,
    points_world: np.ndarray,
    near_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    World points to pixel coordinates.

    Returns `(uv, visible)`: an `(N, 2)` float array of (column, row) and an
    `(N,)` bool mask. A point is visible when it is in FRONT of the lens and
    inside the frame -- both, because a point behind the camera still yields
    finite pixel coordinates and they are the mirror image of the truth.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 array of world points, got {points.shape}")
    if not len(points):
        return np.empty((0, 2)), np.empty(0, dtype=bool)

    delta = points - placement.origin
    depth = delta @ placement.axis
    lateral = delta @ placement.right
    vertical = delta @ placement.up

    safe = np.where(depth > near_m, depth, 1.0)
    cx, cy = placement.centre_px
    # Image v runs DOWN, so the vertical term is subtracted.
    uv = np.column_stack(
        (
            cx + placement.focal_px * lateral / safe,
            cy - placement.focal_px * vertical / safe,
        )
    )
    width, height = placement.resolution
    visible = (
        (depth > near_m)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    return uv, visible


def pixel_rays(placement: CameraPlacement, uv: np.ndarray) -> np.ndarray:
    """
    `(N, 3)` world ray directions through the given pixels.

    NOT normalised, deliberately: the optical-axis component is exactly 1, so
    a planar distance along the axis places a point with no cosine divide. The
    convention the deleted module used, kept because it is the one that makes
    a ground-plane intersection a single division.
    """
    pixels = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    cx, cy = placement.centre_px
    x = (pixels[:, 0] - cx) / placement.focal_px
    y = -(pixels[:, 1] - cy) / placement.focal_px
    return (
        placement.axis[None, :]
        + x[:, None] * placement.right[None, :]
        + y[:, None] * placement.up[None, :]
    )


def ground_points(
    placement: CameraPlacement,
    uv: np.ndarray,
    plane_z: float,
    max_range_m: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixels to the world points where their rays meet a horizontal plane.

    Returns `(points, hit)`. A ray that climbs, runs parallel to the plane, or
    meets it beyond `max_range_m` reports no hit rather than a point: above the
    horizon the intersection is BEHIND the camera and is arithmetically
    perfect nonsense, which is exactly the kind of value that silently poisons
    a dataset.

    A single plane is the honest model for a car park and it is not the honest
    model for a road. When this runs live it should intersect
    `world_scene.GroundField` instead, which is the surface actually drawn and
    already carries a confidence mask; the plane is what the OFFLINE dataset
    has, because a recorded sample carries the ego pose and no terrain.
    """
    rays = pixel_rays(placement, uv)
    drop = float(placement.origin[2]) - plane_z
    # Downward means the ray's world Z is negative while the camera is above
    # the plane; the sign of `drop` handles a camera below it too.
    denominator = -rays[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(
            np.abs(denominator) > 1e-9, drop / denominator, np.inf
        )
    hit = np.isfinite(distance) & (distance > 0.0) & (distance <= max_range_m)
    points = placement.origin[None, :] + np.where(
        hit[:, None], distance[:, None], 0.0
    ) * rays
    return points, hit


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError(f"Degenerate {name} vector: {vector!r}")
    return vector / length
