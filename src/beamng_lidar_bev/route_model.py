"""
Turns the bigmap route into an ego-frame reference path.

`navigation` fetches and parses the route; this module is what makes it
followable. `build_route_path` projects the cached world polyline into the
current BEV frame, resamples it (navgraph spacing is tens of metres on
straights, so nothing downstream may measure the raw nodes), derives heading
and smoothed curvature, and runs the backward speed pass that lets the speed
law brake for corners, turning junctions and the destination before the
LiDAR's obstacle work would force it.

Guidance, never authority: everything produced here enters `planner.plan_arc`
as cost terms and enters the controller only through `ArcPlan` fields, so a
blocked arc still outranks any route conformance and the controller invents no
anticipation of its own.

Layering: imports config/models/geometry/numpy -- the exact footprint
`navigation` already has. `planner` never imports this module; it sees only
the `models.RoutePath` the worker hands it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import (
    COMFORT_DECEL_MPS2,
    CORNERING_ACCEL_MPS2,
    MAX_SPEED_MPS,
    ROUTE_ARRIVAL_MARGIN_M,
    ROUTE_CURVATURE_SMOOTH_M,
    ROUTE_DEFAULT_HALF_WIDTH_M,
    ROUTE_JUNCTION_SPEED_MPS,
    ROUTE_JUNCTION_TURN_RAD,
    ROUTE_JUNCTION_WINDOW_M,
    ROUTE_MAX_HALF_WIDTH_M,
    ROUTE_MIN_HALF_WIDTH_M,
    ROUTE_PREVIEW_LEAD_S,
    ROUTE_PREVIEW_M,
    ROUTE_SAMPLE_STEP_M,
)
from .geometry import vec3, vehicle_axes
from .models import RouteHint, RoutePath

# Below this much polyline ahead of the ego there is nothing to follow -- the
# caller falls back to the legacy bearing hint.
_MIN_AHEAD_M = ROUTE_SAMPLE_STEP_M


def _speed_pass(
    arc_s: np.ndarray,
    curvatures: np.ndarray,
    junction_turn: np.ndarray,
    remaining_m: float,
) -> np.ndarray:
    """
    The backward speed pass: the fastest the ego may be going AT each sample
    such that every upcoming curvature, turning junction and the destination
    stays reachable at COMFORT_DECEL_MPS2.

    Uses the same deceleration the controller's free-distance headroom term
    uses, so there is one deceleration model end to end. Beyond the preview
    nothing constrains -- by construction nothing out there can require action
    yet.
    """
    v_cap = np.minimum(
        MAX_SPEED_MPS,
        np.sqrt(CORNERING_ACCEL_MPS2 / np.maximum(np.abs(curvatures), 1e-4)),
    )
    v_cap = np.where(
        junction_turn, np.minimum(v_cap, ROUTE_JUNCTION_SPEED_MPS), v_cap
    )
    # The destination is a v = 0 point at remaining - margin, folded in closed
    # form rather than as a virtual sample: v <= sqrt(2 a (stop - s)) at every
    # sample is exactly the backward pass from a standing stop there.
    stop_s = remaining_m - ROUTE_ARRIVAL_MARGIN_M
    v = np.minimum(
        v_cap,
        np.sqrt(2.0 * COMFORT_DECEL_MPS2 * np.maximum(stop_s - arc_s, 0.0)),
    )
    # Each sample may only be as fast as it can shed to the next one's
    # allowance. <= 41 samples at the display rate, so the scalar loop is
    # microseconds; a vectorised scan of a running recurrence would be less
    # clear for no measurable win.
    for index in range(len(v) - 2, -1, -1):
        step = float(arc_s[index + 1] - arc_s[index])
        limit = math.sqrt(
            float(v[index + 1]) ** 2 + 2.0 * COMFORT_DECEL_MPS2 * step
        )
        if v[index] > limit:
            v[index] = limit
    return v


def route_speed_limit(
    arc_s: np.ndarray,
    curvatures: np.ndarray,
    junction_turn: np.ndarray,
    remaining_m: float,
    speed_mps: float = 0.0,
) -> float:
    """
    The allowed speed handed to the controller: the MINIMUM of the backward
    pass over the next ROUTE_PREVIEW_LEAD_S of travel down the path.

    The lead is what cancels the speed loop's standing error against a
    ramping target (see the config comment); at standstill, or wherever the
    pass is flat, it changes nothing. It must be a minimum over the window
    and never a point sample: the pass only constrains deceleration, so it
    rises again right after every corner dip, and a point sample at the lead
    lands on that rising limb whenever the dip is closer than the lead --
    measured on a single-vertex 90-degree corner at the cap, the sampled
    limit read 9.3 m/s where the dip allowed 3.2, INVERTING with speed (a
    faster ego got a weaker limit) and accelerating the car mid-corner. The
    window minimum is identical on flat and monotone-falling stretches, so
    the documented lead behaviour is untouched.
    """
    if len(arc_s) == 0:
        return 0.0
    v = _speed_pass(arc_s, curvatures, junction_turn, remaining_m)
    lead_s = max(0.0, float(speed_mps)) * ROUTE_PREVIEW_LEAD_S
    window = v[arc_s <= lead_s]
    at_lead = float(np.interp(lead_s, arc_s, v))
    if len(window):
        return float(min(window.min(), at_lead))
    return at_lead


def build_route_path(
    route: RouteHint | None,
    state: Mapping[str, Any],
    preview_m: float = ROUTE_PREVIEW_M,
    step_m: float = ROUTE_SAMPLE_STEP_M,
) -> RoutePath | None:
    """
    Project the cached route into the current ego frame and preview it.

    Returns None whenever a reference path cannot honestly be built (no route,
    fewer than two usable nodes ahead of the ego) -- the caller then falls back
    to `navigation.route_heading`, the coarse bearing hint.
    """
    if route is None or len(route.path_world) < 2:
        return None

    origin = vec3(state["pos"])
    right, forward, _ = vehicle_axes(state)
    offsets = np.asarray(route.path_world, dtype=np.float64) - origin
    nodes = np.column_stack((offsets @ right, offsets @ forward))

    # --- project the ego onto the polyline -------------------------------
    # The game snaps path[1] to the vehicle, but the cache can be a few polls
    # old and the car has moved along the route since; the projection is what
    # keeps arc lengths and the cross-track honest under that staleness.
    segments = np.diff(nodes, axis=0)
    seg_len = np.hypot(segments[:, 0], segments[:, 1])
    usable = seg_len > 1e-6
    if not usable.any():
        return None
    t = np.zeros(len(segments))
    t[usable] = np.clip(
        -(nodes[:-1][usable] * segments[usable]).sum(axis=1)
        / seg_len[usable] ** 2,
        0.0,
        1.0,
    )
    closest = nodes[:-1] + segments * t[:, None]
    distance_sq = (closest**2).sum(axis=1)
    distance_sq[~usable] = np.inf
    match = int(np.argmin(distance_sq))
    foot = closest[match]

    # Signed cross-track, positive when the ego sits RIGHT of the path. The
    # right normal of direction (dx, dy) is (dy, -dx); the ego is the origin,
    # so its offset from the foot is simply -foot.
    direction = segments[match] / seg_len[match]
    normal = np.array((direction[1], -direction[0]))
    cross_track = float(-(foot @ normal))

    # --- the polyline ahead, with node extras aligned to it --------------
    ahead = np.vstack((foot, nodes[match + 1 :]))
    chords = np.hypot(*np.diff(ahead, axis=0).T)
    keep = chords > 1e-6
    ahead = np.vstack((foot, ahead[1:][keep]))
    if len(ahead) < 2:
        return None
    arc_nodes = np.concatenate(([0.0], np.cumsum(chords[keep])))
    total_m = float(arc_nodes[-1])
    if total_m < _MIN_AHEAD_M:
        return None

    def _aligned(values: np.ndarray | None) -> np.ndarray | None:
        # Per-node extras for the ahead polyline: the foot inherits the node
        # it projects from, then the surviving downstream nodes in order.
        if values is None:
            return None
        trimmed = np.asarray(values, dtype=np.float64)[match:]
        return np.concatenate((trimmed[:1], trimmed[1:][keep]))

    node_widths = _aligned(route.half_width_m)
    node_links = _aligned(route.link_counts)

    # --- resample at uniform arc spacing ---------------------------------
    # Mandatory before any curvature is measured: navgraph spacing can be tens
    # of metres on straights, and curvature from raw chords is meaningless.
    samples = np.arange(0.0, min(total_m, preview_m) + step_m / 2.0, step_m)
    if len(samples) < 2:
        return None
    xs = np.interp(samples, arc_nodes, ahead[:, 0])
    ys = np.interp(samples, arc_nodes, ahead[:, 1])

    # Heading: 0 straight ahead, positive LEFT (the planner's convention).
    dx = np.diff(xs)
    dy = np.diff(ys)
    segment_theta = np.unwrap(np.arctan2(-dx, dy))
    theta = np.empty(len(samples))
    theta[0] = segment_theta[0]
    theta[-1] = segment_theta[-1]
    theta[1:-1] = 0.5 * (segment_theta[:-1] + segment_theta[1:])

    # Curvature, smoothed over ROUTE_CURVATURE_SMOOTH_M. The window is sized
    # against a 90-degree junction encoded as a SINGLE VERTEX between long
    # chords: smearing pi/2 over the window reads k ~= 0.17 -> a ~4 m/s creep
    # through the turn, conservative because over-reading curvature
    # under-reads speed. A real R = 25 m bend spans 39 m of arc and is
    # untouched. This shape is load-bearing -- see the config comment.
    curvature = np.gradient(theta, samples)
    window = max(1, int(round(ROUTE_CURVATURE_SMOOTH_M / step_m)))
    if window % 2 == 0:
        window += 1
    if window > 1:
        padded = np.pad(curvature, window // 2, mode="edge")
        curvature = np.convolve(padded, np.ones(window) / window, mode="valid")

    # Half width: interpolate across the nodes that carried one, clamp to
    # ordinary road geometry (a plaza's navgraph radius is not a lane
    # statement), and fall back to the default when none did.
    if node_widths is not None and bool((node_widths >= 0.0).any()):
        carried = node_widths >= 0.0
        half_width = np.interp(
            samples, arc_nodes[carried], node_widths[carried]
        )
        half_width = np.clip(
            half_width, ROUTE_MIN_HALF_WIDTH_M, ROUTE_MAX_HALF_WIDTH_M
        )
    else:
        half_width = np.full(len(samples), ROUTE_DEFAULT_HALF_WIDTH_M)

    # A junction slows the car only when the route TURNS there: near a
    # linkCount > 2 node AND the smoothed heading actually changes across the
    # window. Either test alone fails -- degree alone brakes at every
    # crossroads driven straight through, curvature alone misses the
    # single-vertex turns the smoothing dilutes.
    junction_turn = np.zeros(len(samples), dtype=bool)
    if node_links is not None:
        junction_s = arc_nodes[node_links > 2.5]
        if len(junction_s):
            near = (
                np.abs(samples[:, None] - junction_s[None, :]).min(axis=1)
                <= ROUTE_JUNCTION_WINDOW_M
            )
            half_window = ROUTE_JUNCTION_WINDOW_M / 2.0
            theta_ahead = np.interp(
                np.minimum(samples + half_window, samples[-1]), samples, theta
            )
            theta_behind = np.interp(
                np.maximum(samples - half_window, 0.0), samples, theta
            )
            junction_turn = near & (
                np.abs(theta_ahead - theta_behind) > ROUTE_JUNCTION_TURN_RAD
            )

    # Remaining distance measured along this polyline rather than taken from
    # the game's figure: chord sums under-read on curves, which errs toward
    # stopping early, and it stays consistent with arc_s under a stale cache.
    # The ego speed (when the state carries it) drives the lead sampling that
    # cancels the speed loop's tracking lag.
    velocity = state.get("vel")
    ego_speed = 0.0 if velocity is None else float(np.linalg.norm(vec3(velocity)))
    speed_limit = route_speed_limit(
        samples, curvature, junction_turn, total_m, speed_mps=ego_speed
    )

    return RoutePath(
        points=np.column_stack((xs, ys)).astype(np.float32),
        arc_s=samples,
        headings=theta,
        curvatures=curvature,
        half_width_m=half_width,
        junction_turn=junction_turn,
        remaining_m=total_m,
        cross_track_m=cross_track,
        speed_limit_mps=speed_limit,
    )
