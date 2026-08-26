"""
Per-leg path smoothing: the stage between the search and the tracker.

A raw planned leg is pieces of constant curvature -- straights, arcs,
Reeds-Shepp words -- with a curvature STEP at every join. A car cannot steer a
step: the wheel takes real time to wind, so the tracker chases every join
late, develops cross-track, and corrects with the very gains that then look
"unstable" when raised. Every production parking pipeline puts a smoothing
stage here (Dolgov's conjugate-gradient smoother, Apollo's DL-IAPS), run per
GEAR SEGMENT with the endpoints held fixed -- a cusp is where the car is
stopped, so a curvature discontinuity there is free, and only there.

This is the small, safe version of that stage: gradient descent on the
interior vertices toward their neighbours' midpoint (which minimises bending
energy), with every vertex hard-clamped to `PARKING_SMOOTH_MAX_DEVIATION_M`
of the path the collision and bay-keepout checks approved. The caller
re-validates the smoothed leg against both checks and keeps the raw leg when
either fails, so the clamp is a belt and the validation is the braces.

Qt-free and BeamNGpy-free: config + numpy only, like every planning module.
"""

from __future__ import annotations

import numpy as np

from .config import (
    PARKING_PATH_STEP_M,
    PARKING_SMOOTH_MAX_DEVIATION_M,
    PARKING_SMOOTH_WINDOW_M,
)


def discrete_curvature(points: np.ndarray) -> np.ndarray:
    """
    Signed curvature along a polyline, by the circumscribed-circle rule.

    Positive is a LEFT bend relative to the traversal order, which makes it
    the travel-frame curvature whichever way the leg is driven: negating every
    point (the reverse-leg convention) is a rotation, and signed curvature is
    rotation-invariant.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros(len(pts))
    previous, current, following = pts[:-2], pts[1:-1], pts[2:]
    first = current - previous
    second = following - current
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    lengths = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(following - previous, axis=1)
    )
    curvature = np.zeros(len(pts))
    safe = lengths > 1e-9
    curvature[1:-1][safe] = 2.0 * cross[safe] / lengths[safe]
    return curvature


def smooth_path(
    points: np.ndarray,
    window_m: float = PARKING_SMOOTH_WINDOW_M,
    max_deviation_m: float = PARKING_SMOOTH_MAX_DEVIATION_M,
) -> np.ndarray:
    """
    The path with its curvature steps spread into trackable transitions.

    A windowed moving average along arc length, on the executor's uniform
    sample grid: a tangent kink of angle theta becomes a bend of curvature
    roughly theta / window over the window's length, and a corner is cut by
    only theta * window / 8 -- comfortably inside the deviation clamp for
    every kink the planners emit. A fixed linear operator on purpose: the
    Laplacian descent tried first CLUSTERS vertices at the corners it rounds
    (the classic shrinkage artifact), and clustered samples read as higher
    discrete curvature than the kink they replaced.

    The two samples at EACH end are pinned, not just the endpoints. The end
    TANGENT is the pose-heading contract between legs -- two legs meeting at
    a cusp agree on the heading there -- and rotating it hands the next leg
    a phantom initial error: measured at 19 degrees, fought at full lock.
    Interior samples near the ends take a symmetrically shrunk window, so
    the smoothing fades toward the pinned tangents instead of stepping.
    """
    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < 7:
        return pts
    original = pts.copy()
    half = max(1, int(round(0.5 * window_m / PARKING_PATH_STEP_M)))
    prefix = np.concatenate(
        (np.zeros((1, 2)), np.cumsum(pts, axis=0))
    )
    smoothed = pts.copy()
    # Only the endpoints are hard-pinned; every interior sample takes a
    # symmetrically SHRUNK window, so the smoothing fades to nothing at the
    # ends instead of stepping from raw to filtered -- a hard transition two
    # samples in read as a curvature spike of its own. The end tangent still
    # cannot rotate meaningfully: the first movable sample averages over one
    # neighbour each side, a millimetre-scale move at this spacing.
    for index in range(1, len(pts) - 1):
        reach = min(half, index, len(pts) - 1 - index)
        lo = index - reach
        hi = index + reach + 1
        smoothed[index] = (prefix[hi] - prefix[lo]) / (hi - lo)
    drift = smoothed - original
    distance_sq = np.einsum("ij,ij->i", drift, drift)
    over = distance_sq > float(max_deviation_m) ** 2
    if over.any():
        scale = max_deviation_m / np.sqrt(distance_sq[over])
        smoothed[over] = original[over] + drift[over] * scale[:, None]
    return smoothed
