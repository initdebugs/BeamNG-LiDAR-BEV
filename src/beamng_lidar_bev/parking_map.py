"""Parking-only world-cell memory for observed free and blocked space."""

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


def _horizontal_axes(
    right: np.ndarray, forward: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r_xy = np.asarray(right, dtype=np.float64)[:2]
    f_xy = np.asarray(forward, dtype=np.float64)[:2]
    r_xy /= max(float(np.linalg.norm(r_xy)), 1e-9)
    f_xy /= max(float(np.linalg.norm(f_xy)), 1e-9)
    return r_xy, f_xy


class ParkingMap:
    """A bounded, mutable map owned by the BeamNG worker thread."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._blocked: dict[tuple[int, int], np.ndarray] = {}
        self._free: dict[tuple[int, int], np.ndarray] = {}
        self._last_pos: np.ndarray | None = None

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

        free_world = world(free_bev)
        blocked_world = world(blocked_bev)
        blocked_keys = {
            tuple(cell)
            for cell in np.floor(blocked_world / HYBRID_CELL_M).astype(np.int64)
        }
        for point in free_world:
            key = tuple(np.floor(point / HYBRID_CELL_M).astype(np.int64))
            if key not in blocked_keys:
                self._blocked.pop(key, None)
                self._free[key] = point.copy()
        for point in blocked_world:
            key = tuple(np.floor(point / HYBRID_CELL_M).astype(np.int64))
            self._free.pop(key, None)
            self._blocked[key] = point.copy()
        self._expire(ego)

    def _expire(self, ego: np.ndarray) -> None:
        radius_sq = MEMORY_RADIUS_M**2

        def bounded(
            cells: dict[tuple[int, int], np.ndarray], maximum: int
        ) -> dict[tuple[int, int], np.ndarray]:
            inside = [
                (key, point)
                for key, point in cells.items()
                if float(np.sum((point - ego) ** 2)) <= radius_sq
            ]
            if len(inside) > maximum:
                inside.sort(key=lambda item: float(np.sum((item[1] - ego) ** 2)))
                inside = inside[:maximum]
            return dict(inside)

        self._blocked = bounded(self._blocked, MEMORY_MAX_CELLS)
        self._free = bounded(self._free, MEMORY_MAX_ROAD_CELLS)

    def occupancy_bev(
        self, pos_world: np.ndarray, right: np.ndarray, forward: np.ndarray
    ) -> Occupancy:
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        r_xy, f_xy = _horizontal_axes(right, forward)

        def project(cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
            if not cells:
                return np.empty((0, 2), dtype=np.float64)
            rel = np.asarray(tuple(cells.values()), dtype=np.float64) - ego
            return np.column_stack((rel @ r_xy, rel @ f_xy))

        return Occupancy(
            project(self._blocked),
            project(self._free),
            cell_m=HYBRID_CELL_M,
        )
