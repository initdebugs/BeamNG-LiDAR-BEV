"""Parking-only world-cell memory for observed free and blocked space.

Fed every measured tick from the moment the parking SCAN is armed, not from
engage: the first search used to run against a map cleared seconds earlier
and see one tick of returns in a sea of UNKNOWN, so it planned through space
the corridor check then vetoed mid-drive -- the live block/resume cycling.
Accumulating while the user is still choosing a bay means the plan starts
from what the car has actually seen of the lot.

That promotion is why the hot path is vectorized: the free set is the whole
road-classified cloud (tens of thousands of points a tick), and the old
per-point Python loop priced at several milliseconds of a 40 ms tick. Points
are reduced to one representative per cell in numpy first; only the surviving
few thousand cells touch the dicts, and the radius expiry is amortized.
"""

from __future__ import annotations

import numpy as np

from .config import (
    HYBRID_CELL_M,
    MEMORY_MAX_CELLS,
    MEMORY_MAX_ROAD_CELLS,
    MEMORY_POSE_JUMP_RESET_M,
    MEMORY_RADIUS_M,
)
from .hybrid_astar import Occupancy

_EXPIRE_EVERY = 25
"""Updates between full radius sweeps; the caps are also checked per update."""


def _horizontal_axes(
    right: np.ndarray, forward: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r_xy = np.asarray(right, dtype=np.float64)[:2]
    f_xy = np.asarray(forward, dtype=np.float64)[:2]
    r_xy /= max(float(np.linalg.norm(r_xy)), 1e-9)
    f_xy /= max(float(np.linalg.norm(f_xy)), 1e-9)
    return r_xy, f_xy


def _cells_and_points(
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One representative world point per occupied cell, newest kept."""
    if not len(world):
        return np.empty((0, 2), dtype=np.int64), world
    cells = np.floor(world / HYBRID_CELL_M).astype(np.int64)
    # Pack to one integer per cell so `unique` stays 1-D (the scene stores'
    # idiom); keep the LAST point of each cell, matching what the old
    # overwrite-in-order loop kept.
    packed = (cells[:, 0] + (1 << 21)) * (1 << 22) + (cells[:, 1] + (1 << 21))
    reversed_packed = packed[::-1]
    _, first_of_reversed = np.unique(reversed_packed, return_index=True)
    keep = len(world) - 1 - first_of_reversed
    return cells[keep], world[keep]


class ParkingMap:
    """A bounded, mutable map owned by the BeamNG worker thread."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._blocked: dict[tuple[int, int], np.ndarray] = {}
        self._free: dict[tuple[int, int], np.ndarray] = {}
        self._last_pos: np.ndarray | None = None
        self._updates_since_expiry = 0

    def update(
        self,
        pos_world: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        blocked_bev: np.ndarray,
        free_bev: np.ndarray,
    ) -> None:
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        if self._last_pos is not None and float(
            np.linalg.norm(ego - self._last_pos)
        ) > MEMORY_POSE_JUMP_RESET_M:
            self.clear()
        self._last_pos = ego.copy()
        r_xy, f_xy = _horizontal_axes(right, forward)

        def world(points: np.ndarray) -> np.ndarray:
            local = np.asarray(points, dtype=np.float64)
            if not local.size:
                return np.empty((0, 2), dtype=np.float64)
            return ego + local[:, [0]] * r_xy + local[:, [1]] * f_xy

        free_cells, free_points = _cells_and_points(world(free_bev))
        blocked_cells, blocked_points = _cells_and_points(world(blocked_bev))
        blocked_keys = set(map(tuple, blocked_cells.tolist()))
        for key, point in zip(
            map(tuple, free_cells.tolist()), free_points
        ):
            if key not in blocked_keys:
                self._blocked.pop(key, None)
                self._free[key] = point
        for key, point in zip(
            map(tuple, blocked_cells.tolist()), blocked_points
        ):
            self._free.pop(key, None)
            self._blocked[key] = point
        self._updates_since_expiry += 1
        if (
            self._updates_since_expiry >= _EXPIRE_EVERY
            or len(self._blocked) > MEMORY_MAX_CELLS
            or len(self._free) > MEMORY_MAX_ROAD_CELLS
        ):
            self._updates_since_expiry = 0
            self._expire(ego)

    def _expire(self, ego: np.ndarray) -> None:
        radius_sq = MEMORY_RADIUS_M**2

        def bounded(
            cells: dict[tuple[int, int], np.ndarray], maximum: int
        ) -> dict[tuple[int, int], np.ndarray]:
            if not cells:
                return cells
            keys = tuple(cells.keys())
            points = np.asarray(tuple(cells.values()), dtype=np.float64)
            distance_sq = np.einsum(
                "ij,ij->i", points - ego, points - ego
            )
            inside = distance_sq <= radius_sq
            if int(inside.sum()) > maximum:
                order = np.argsort(distance_sq)
                chosen = order[inside[order]][:maximum]
            else:
                chosen = np.flatnonzero(inside)
            return {keys[i]: points[i] for i in chosen.tolist()}

        self._blocked = bounded(self._blocked, MEMORY_MAX_CELLS)
        self._free = bounded(self._free, MEMORY_MAX_ROAD_CELLS)

    def occupancy_bev(
        self,
        pos_world: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        body: tuple[float, float, float, float] | None = None,
    ) -> Occupancy:
        """
        The map as a BEV occupancy, with the ego's own footprint TRUSTED.

        `body` is (left, right, front, rear) about the reference node. A
        blocked cell inside it is stale by definition -- the car is standing
        there -- and it can never be re-observed while the body covers it,
        because the ego cull removes every return inside the body before
        anything reaches this map. Persisting the map across engage (which
        is what lets the first search know the lot) is what made this
        possible: measured live, four engagements refused UNREACHABLE in a
        row from a spot the car had just driven to. The store is untouched;
        once the car moves away the ground is observed again and the cell
        clears or re-blocks on real evidence.
        """
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        r_xy, f_xy = _horizontal_axes(right, forward)

        def project(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
            if not cells:
                return np.empty((0, 2), dtype=np.float64)
            rel = np.asarray(tuple(cells.values()), dtype=np.float64) - ego
            return np.column_stack((rel @ r_xy, rel @ f_xy))

        blocked = project(self._blocked)
        if body is not None and len(blocked):
            left_m, right_m, front_m, rear_m = body
            margin = 0.10
            inside = (
                (blocked[:, 0] >= -left_m - margin)
                & (blocked[:, 0] <= right_m + margin)
                & (blocked[:, 1] >= -rear_m - margin)
                & (blocked[:, 1] <= front_m + margin)
            )
            blocked = blocked[~inside]
        return Occupancy(
            blocked,
            project(self._free),
            cell_m=HYBRID_CELL_M,
        )
