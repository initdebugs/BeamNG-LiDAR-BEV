"""
Fit oriented boxes to this snapshot's vehicle returns.

Pure Python/numpy: no Qt, no BeamNGpy, no wall clock and no state, so the whole
thing is a function of one cloud and a pose. It sits beside `parking` in the
pure layer and, like `parking`, it answers a question about SHAPE.

**Why a car is not accumulated and meshed like everything else.** The voxel store
forgets scenery by the metre, so a stopped ego keeps every look it ever got and a
wall fills in over a drive. Traffic is forgotten by the second, because a car
moves and the scenery window would draw it as a streak of itself -- and at a
standstill that leaves ONE snapshot of evidence. One snapshot of a car at 15 m is
four or five azimuth stripes over a metre apart, and `_column_runs` meshes
exactly what it is handed: confetti. No amount of waiting fixes it either, since
a stationary ego re-samples the same rays.

So the returns are not meshed. Accumulation is how you build a surface whose
shape you do not know; a car is an object whose shape you DO know. Five stripes
cannot mesh a surface and are ample to fit a footprint, so this fits the
footprint and the existing actor delegate draws a car over it.

Two properties are load-bearing and both are about erring toward the solids:

- **This is additive.** Nothing here removes a return from the voxel store. A
  cluster that does not fit a plausible vehicle is simply not claimed, and it
  still draws as a solid exactly as it does today -- so the failure mode is the
  current picture, never a car that vanished because the fit was wrong.
- **Every inference goes BEHIND the evidence.** Where only one face was seen the
  unobserved dimension is assumed and the box is pushed away from the ego by the
  amount it was extended, so the drawn near face stays on the returns that were
  actually measured. `confidence` carries how much was resolved and rides the
  delegate's opacity, which is how the view says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    VEHICLE_FIT_ANGLE_STEP_DEG,
    VEHICLE_FIT_DEFAULT_LENGTH_M,
    VEHICLE_FIT_DEFAULT_WIDTH_M,
    VEHICLE_FIT_EDGE_FLOOR_M,
    VEHICLE_FIT_FULL_POINTS,
    VEHICLE_FIT_LINK_AZIMUTH_CELLS,
    VEHICLE_FIT_LINK_RANGE_CELLS,
    VEHICLE_FIT_MAX_CLUSTERS,
    VEHICLE_FIT_MIN_POINTS,
    VEHICLE_FIT_ONE_FACE_CONFIDENCE,
    VEHICLE_FIT_RANGE_CELL_M,
    VEHICLE_FIT_SIDE_LENGTH_M,
    VEHICLE_FIT_SPLIT_GAP_M,
    VEHICLE_FIT_SPLIT_LENGTH_M,
    VEHICLE_FIT_STRIPE_RAD,
    VEHICLE_MAX_HEIGHT_M,
    VEHICLE_MAX_LENGTH_M,
    VEHICLE_MAX_WIDTH_M,
    VEHICLE_MIN_ASPECT,
    VEHICLE_MIN_HEIGHT_M,
    VEHICLE_MIN_LENGTH_M,
    VEHICLE_MIN_WIDTH_M,
)

# Enough to hold every range cell inside any sensible world radius, so the
# packed (azimuth, range) key stays a plain int64 and every `unique` stays 1-D.
# Same reasoning as `world_scene.pack_cell_keys`.
_RANGE_STRIDE = 1 << 20
# A cluster that splits this many times is not a row of cars, it is noise.
_MAX_SPLIT_DEPTH = 4


@dataclass(frozen=True)
class VehicleBox:
    """One oriented vehicle box fitted to a cluster of vehicle returns."""

    # Footprint centre in world XY, and the BASE height -- the lowest return in
    # the cluster, which is what the delegate stands on. Not the box centre in
    # Z: the model is built upward from its node.
    centre_world: tuple[float, float, float]
    # Unit world XY heading along the vehicle's length. Which END is the front
    # is not observable from a point cloud, so this is an axis, not a direction;
    # the drawn model is symmetric enough that it does not matter.
    forward_world: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    """Width, height, length -- the order `WorldActor.scale` uses."""
    kind: str
    confidence: float
    point_count: int
    inferred_depth: bool
    """True when only one face was seen and a dimension was assumed."""


def fit_vehicle_boxes(
    points_world: np.ndarray,
    ego_pos_world: tuple[float, float, float],
) -> tuple[VehicleBox, ...]:
    """
    Cluster vehicle returns and fit an oriented box to each plausible vehicle.

    Returns only what it is confident enough to call a vehicle. Everything else
    is left alone; it is already drawn as solid geometry by the voxel store.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 world point array, got {points.shape}")
    if len(points) < VEHICLE_FIT_MIN_POINTS:
        return ()

    ego_xy = np.asarray(ego_pos_world, dtype=np.float64)[:2]
    labels = _cluster_labels(points, ego_xy)
    boxes: list[VehicleBox] = []
    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(grouped)]):
        member = points[order[start:stop]]
        boxes.extend(_fit_cluster(member, ego_xy, 0))

    if len(boxes) <= VEHICLE_FIT_MAX_CLUSTERS:
        return tuple(boxes)
    # Keep the nearest, which are the ones that matter and the ones the fit is
    # most confident about -- never an arbitrary slice.
    ranked = sorted(
        boxes,
        key=lambda box: float(
            np.hypot(
                box.centre_world[0] - ego_xy[0], box.centre_world[1] - ego_xy[1]
            )
        ),
    )
    return tuple(ranked[:VEHICLE_FIT_MAX_CLUSTERS])


def _cluster_labels(points: np.ndarray, ego_xy: np.ndarray) -> np.ndarray:
    """
    Connected components in the SENSOR's lattice: azimuth by range.

    Clustering in world XY needs a distance threshold, and there is no value
    that works: azimuth stripes spread as `r`, so anything wide enough to hold a
    car together at 30 m welds two parked cars together at 8 m. In polar
    coordinates the stripe spacing is one cell everywhere by construction, which
    is the same reasoning `WORLD_COLUMN_BRIDGE_CELLS` is sized by.
    """
    offsets = points[:, :2] - ego_xy
    ranges = np.hypot(offsets[:, 0], offsets[:, 1])
    azimuth = np.arctan2(offsets[:, 1], offsets[:, 0])
    azimuth_cells = int(math.ceil(2.0 * math.pi / VEHICLE_FIT_STRIPE_RAD))

    # Wrapped into [0, azimuth_cells) so a cluster straddling the +/-pi seam is
    # one cluster; the neighbour walk wraps with the same modulo.
    cell_a = (
        np.floor((azimuth + math.pi) / VEHICLE_FIT_STRIPE_RAD).astype(np.int64)
        % azimuth_cells
    )
    cell_r = np.floor(ranges / VEHICLE_FIT_RANGE_CELL_M).astype(np.int64)
    packed = cell_a * _RANGE_STRIDE + cell_r
    keys, inverse = np.unique(packed, return_inverse=True)
    key_a = keys // _RANGE_STRIDE
    key_r = keys - key_a * _RANGE_STRIDE

    parent = np.arange(len(keys), dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    index = np.arange(len(keys), dtype=np.int64)
    azimuth_reach = range(
        -VEHICLE_FIT_LINK_AZIMUTH_CELLS, VEHICLE_FIT_LINK_AZIMUTH_CELLS + 1
    )
    range_reach = range(
        -VEHICLE_FIT_LINK_RANGE_CELLS, VEHICLE_FIT_LINK_RANGE_CELLS + 1
    )
    for delta_a in azimuth_reach:
        for delta_r in range_reach:
            if delta_a == 0 and delta_r == 0:
                continue
            neighbour = (
                (key_a + delta_a) % azimuth_cells
            ) * _RANGE_STRIDE + (key_r + delta_r)
            slot = np.searchsorted(keys, neighbour)
            np.clip(slot, 0, len(keys) - 1, out=slot)
            hit = keys[slot] == neighbour
            # The loop runs over CELLS, of which a cloud of vehicle returns has
            # a few hundred -- not over points.
            for left, right in zip(index[hit], slot[hit]):
                root_left, root_right = find(int(left)), find(int(right))
                if root_left != root_right:
                    parent[root_left] = root_right

    roots = np.array([find(int(node)) for node in index], dtype=np.int64)
    return roots[inverse]


def _fit_cluster(
    member: np.ndarray, ego_xy: np.ndarray, depth: int
) -> list[VehicleBox]:
    if len(member) < VEHICLE_FIT_MIN_POINTS:
        return []

    angle, span_u, span_v, proj_u, proj_v = _footprint_frame(member[:, :2])
    along_u = span_u >= span_v
    long_span = span_u if along_u else span_v
    short_span = span_v if along_u else span_u
    long_proj = proj_u if along_u else proj_v

    if long_span > VEHICLE_FIT_SPLIT_LENGTH_M:
        # Longer than an ordinary car, so ask whether it is really two of them.
        #
        # **The test is whether the split is PLAUSIBLE, not whether a gap
        # exists**, and it has to be: at range the stripes are metres apart, so
        # one car's own end face and flank are separated by a gap of exactly the
        # kind two parked cars leave between them. Splitting on the gap alone
        # cuts single cars in half. Splitting only when both halves fit whole
        # vehicles keeps a bus whole (no gap to take), keeps an L-shaped car
        # whole (one half would be a sliver), and separates a queue.
        split = _split_cluster(member, long_proj, ego_xy, depth)
        if split is not None:
            return split
        if long_span > VEHICLE_MAX_LENGTH_M:
            # Longer than anything that drives and not divisible into things
            # that do. Claim nothing; the solids already draw it.
            return []

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    axis_u = np.array((cos_a, sin_a))
    axis_v = np.array((-sin_a, cos_a))
    centre = axis_u * (0.5 * (proj_u.max() + proj_u.min())) + axis_v * (
        0.5 * (proj_v.max() + proj_v.min())
    )
    # **Stripe sampling under-measures every horizontal extent, and by enough to
    # matter.** A surface is only sampled where an azimuth stripe crosses it, so
    # its edges lie somewhere between the outermost stripe that hit and the next
    # one that missed: the true extent is `measured + up to two spacings`, and
    # one spacing is the expected correction. Measured on a 1.9 m car at 12.75 m
    # the raw span is 0.87 m -- narrow enough to be REJECTED as too thin to be a
    # vehicle, which is the whole width of the car thrown away by the sampling
    # rather than by the geometry.
    #
    # The short span is corrected only when it already spans a stripe. That is
    # what stops the correction MANUFACTURING depth: a single flat face measures
    # ~0 across, and at 30 m one spacing is 1.86 m, so correcting it
    # unconditionally would invent a second face and report an assumed length as
    # a measured one.
    #
    # **The spacing is MEASURED from the cluster, not assumed from the range.**
    # Applying the nominal stripe spacing unconditionally over-reads whatever is
    # densely sampled -- a car close enough to be crossed by many stripes, or any
    # surface the near-field units resolve properly -- and a 1.7 m face came back
    # as 2.6 m wide. The largest hole between consecutive azimuths IS the
    # sampling interval when the cluster is striped and is near zero when it is
    # dense, so it self-calibrates. Capped at the nominal spacing so an occlusion
    # hole inside one object cannot inflate it.
    seen_from = member[:, :2] - ego_xy
    azimuths = np.sort(np.arctan2(seen_from[:, 1], seen_from[:, 0]))
    sampling = float(np.diff(azimuths).max()) if len(azimuths) > 1 else 0.0
    stripe = float(np.hypot(*(centre - ego_xy))) * min(
        sampling, VEHICLE_FIT_STRIPE_RAD
    )
    if along_u:
        span_u += stripe
        span_v += stripe if span_v >= stripe else 0.0
    else:
        span_v += stripe
        span_u += stripe if span_u >= stripe else 0.0
    long_span, short_span = max(span_u, span_v), min(span_u, span_v)
    long_axis = axis_u if along_u else axis_v
    short_axis = axis_v if along_u else axis_u

    # **The question is only ever "was a FLANK seen", and nothing else settles
    # it.** A vehicle's length is observable exactly when something longer than
    # any vehicle is wide came back; below that the returns are an end face, or
    # a corner, and either way the length was not measured. Believing a short
    # measurement instead is what drew a 2.4 x 2.4 m car -- a parked car
    # alongside the ego shows a badly foreshortened flank, so its L has two
    # short arms and taking them at face value asserts a square vehicle.
    inferred = False
    if long_span >= VEHICLE_FIT_SIDE_LENGTH_M:
        length, forward = long_span, long_axis
        if short_span >= VEHICLE_MIN_WIDTH_M:
            # Both dimensions measured: a flank and an end.
            width = short_span
            inferred_axis, inferred_extent, observed = short_axis, 0.0, 0.0
        else:
            # A flank with no depth to it. The width is assumed.
            width = VEHICLE_FIT_DEFAULT_WIDTH_M
            inferred = True
            inferred_axis, inferred_extent, observed = (
                short_axis,
                width,
                short_span,
            )
    else:
        # Nothing long enough to be a flank, so the length is not observed at
        # all and is assumed.
        #
        # **Which arm is the WIDTH cannot be settled by asking which is longer,
        # and assuming the longer one is the length drew a car ahead of the ego
        # lying across the road.** Both arms are under a flank's length by the
        # test above, so length tells you nothing; geometry does. The face
        # turned toward the ego is the one lying ACROSS the line of sight, and
        # the other arm is the foreshortened stub of a flank running away from
        # it. That resolves the pure end face and the corner with one rule, and
        # it agrees with the old behaviour on the end face exactly.
        length = VEHICLE_FIT_DEFAULT_LENGTH_M
        inferred = True
        sight = centre - ego_xy
        sight = sight / max(float(np.hypot(sight[0], sight[1])), 1e-9)
        if abs(float(axis_u @ sight)) <= abs(float(axis_v @ sight)):
            width, forward, observed = span_u, axis_v, span_v
        else:
            width, forward, observed = span_v, axis_u, span_u
        inferred_axis, inferred_extent = forward, length

    if inferred:
        # Push by exactly what was ADDED, so the near face stays on the
        # measured returns and the whole inference sits behind them.
        away = centre - ego_xy
        sign = 1.0 if float(inferred_axis @ away) >= 0.0 else -1.0
        centre = centre + inferred_axis * (
            sign * max(0.0, 0.5 * (inferred_extent - observed))
        )

    # **Road vehicles are a remarkably consistent shape, and that is usable
    # evidence about the dimension the returns resolve WORST.** A car alongside
    # the ego is crossed by two or three stripes in total, so its width is
    # measured between the outermost two and then corrected by a whole spacing --
    # and small errors in the frame land almost entirely on the short axis.
    # Measured on a synthetic street the parked cars came back 2.4 and 2.8 m
    # wide against a true 1.85. Everything that drives is at least this much
    # longer than it is wide, so the ratio only ever NARROWS an over-read: a van
    # at 5.1 x 2.1 and an artic at 9 x 2.6 are both untouched.
    width = min(width, length / VEHICLE_MIN_ASPECT)

    base = float(member[:, 2].min())
    height = float(member[:, 2].max()) - base
    if not VEHICLE_MIN_WIDTH_M <= width <= VEHICLE_MAX_WIDTH_M:
        return []
    if not VEHICLE_MIN_LENGTH_M <= length <= VEHICLE_MAX_LENGTH_M:
        return []
    if not VEHICLE_MIN_HEIGHT_M <= height <= VEHICLE_MAX_HEIGHT_M:
        return []

    support = min(1.0, len(member) / VEHICLE_FIT_FULL_POINTS)
    confidence = 0.30 + 0.70 * support
    if inferred:
        confidence *= VEHICLE_FIT_ONE_FACE_CONFIDENCE

    return [
        VehicleBox(
            centre_world=(float(centre[0]), float(centre[1]), base),
            forward_world=(float(forward[0]), float(forward[1]), 0.0),
            dimensions_m=(float(width), float(height), float(length)),
            kind=_kind_for(length, height),
            confidence=float(confidence),
            point_count=int(len(member)),
            inferred_depth=inferred,
        )
    ]


def _split_cluster(
    member: np.ndarray,
    long_proj: np.ndarray,
    ego_xy: np.ndarray,
    depth: int,
) -> list[VehicleBox] | None:
    """
    Cut at the largest hole along the long axis, but only if BOTH halves are
    vehicles. Returns None when the cluster is better left whole.
    """
    if depth >= _MAX_SPLIT_DEPTH:
        return None
    order = np.argsort(long_proj)
    gaps = np.diff(long_proj[order])
    if not len(gaps):
        return None
    widest = int(np.argmax(gaps))
    if float(gaps[widest]) < VEHICLE_FIT_SPLIT_GAP_M:
        return None
    near = _fit_cluster(member[order[: widest + 1]], ego_xy, depth + 1)
    far = _fit_cluster(member[order[widest + 1 :]], ego_xy, depth + 1)
    return near + far if near and far else None


def _footprint_frame(
    xy: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Sweep the angle and take the frame whose EDGES the returns hug.

    Two wrong answers were measured on the way here, and both are the obvious
    thing to reach for.

    **PCA** answers a different question. A car seen from behind and to one side
    comes back as an L -- a rear face and one flank -- and the principal axis of
    an L runs diagonally across it, so a parked car is drawn at a large angle to
    the kerb it is parked against.

    **Minimum-area rectangle** is the textbook fix and it is DEGENERATE on
    exactly this shape. The convex hull of a clean L is a triangle, and for a
    triangle every rectangle supported on a hull edge has area `2 x area` --
    identical. So the criterion ties across the aligned frame and the diagonal
    one and `argmin` takes whichever came first: measured on an L with 1.2 m and
    2.7 m arms it chose 24 degrees, drawing a car 24 degrees off the kerb.

    Closeness has no such tie. Each point is scored by its distance to the
    NEARER of the four edges, and the score is the sum of reciprocals, so a
    frame the returns lie ON scores unboundedly better than one they lie inside.
    Both real cases have every return on an edge by construction -- an L is two
    faces, a single face is one -- while the diagonal frame puts them through
    the middle. The floor keeps the reciprocal finite and sets how much better
    "on the edge" can score than "a few centimetres off it".

    The sweep only needs [0, 90) because a rectangle is symmetric under a
    quarter turn. Same trick, and the same reason, as `parking`'s angle sweep.
    """
    angles = np.deg2rad(np.arange(0.0, 90.0, VEHICLE_FIT_ANGLE_STEP_DEG))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    proj_u = xy[:, :1] * cos_a + xy[:, 1:2] * sin_a
    proj_v = -xy[:, :1] * sin_a + xy[:, 1:2] * cos_a
    low_u, high_u = proj_u.min(axis=0), proj_u.max(axis=0)
    low_v, high_v = proj_v.min(axis=0), proj_v.max(axis=0)
    to_edge = np.minimum(
        np.minimum(proj_u - low_u, high_u - proj_u),
        np.minimum(proj_v - low_v, high_v - proj_v),
    )
    closeness = (
        1.0 / np.maximum(to_edge, VEHICLE_FIT_EDGE_FLOOR_M)
    ).sum(axis=0)
    best = int(np.argmax(closeness))
    span_u = high_u - low_u
    span_v = high_v - low_v
    return (
        float(angles[best]),
        float(span_u[best]),
        float(span_v[best]),
        proj_u[:, best],
        proj_v[:, best],
    )


def _kind_for(length: float, height: float) -> str:
    """
    The same three visual classes `worker._actor_visual_type` names, so a car
    drawn from LiDAR and the same car drawn from ground truth agree about what
    it is. Decided from the fitted size because that is all a cloud carries --
    there is no model name to read.
    """
    if length >= 7.0 or height >= 2.6:
        return "truck"
    if length >= 4.9 or height >= 1.8:
        return "utility"
    return "car"
