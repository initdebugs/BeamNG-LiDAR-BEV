"""
Worker-thread obstacle/road memory for the PLANNER ONLY.

Accumulates the planner band's obstacle output as world-anchored cells so
kerbs persist through occlusion, the rear corridor keeps what the sensors saw
a moment ago, and single-frame sampling noise stops dithering the free
distance. Static cells are forgotten by the METRE travelled and anything the
semantics call a vehicle by the WALL CLOCK -- the WORLD stores' two-clocks
design, for the same reasons.

Two hard rules, both load-bearing:

- **AEB never reads this.** A full-authority brake on a remembered ghost is
  the unacceptable failure; both AEB bands stay current-frame. The layering
  enforces it structurally -- this module sits below `planner`, and the only
  consumer is the worker's plan step.
- **Memory can never unblind the planner.** The worker merges the query only
  on ticks that HAVE returns; an empty cloud still plans `_BLIND_ARC` and
  brakes, exactly as before.

Thread confinement: this is a mutable store owned by the BeamNG worker
thread, updated and queried inside `_poll_once` only -- the same confinement
argument as the controller's state, not the scene stores' pool-thread
contract. Nothing here is shared across threads.

The packed-key idiom is deliberately duplicated from `planner`/`world_scene`
rather than imported: importing either would close a layering cycle, and the
three lines it costs buy keeping this module on config + numpy alone.
"""

from __future__ import annotations

import numpy as np

from .config import (
    MEMORY_CELL_M,
    MEMORY_DISTANCE_M,
    MEMORY_MAX_CELLS,
    MEMORY_MAX_QUERY_POINTS,
    MEMORY_MAX_ROAD_CELLS,
    MEMORY_MIN_SUPPORT,
    MEMORY_POSE_JUMP_RESET_M,
    MEMORY_RADIUS_M,
    MEMORY_VEHICLE_TTL_S,
)

_EMPTY_BEV = np.empty((0, 2), dtype=np.float32)
# 21-bit fields, x high: the same packing planner._cell_keys uses, covering
# +-419 km of world at the 0.4 m cell.
_FIELD = 1 << 21
_BIAS = 1 << 20
# Sentinel for "never seen a vehicle return": -inf survives maximum-reduction
# untouched, which is what lets the merge propagate vehicle stamps with the
# same reduceat every other column uses.
_NEVER = -np.inf


def _pack(cells: np.ndarray) -> np.ndarray:
    return (cells[:, 0].astype(np.int64) + _BIAS) * _FIELD + (
        cells[:, 1].astype(np.int64) + _BIAS
    )


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    flags = np.empty(len(sorted_keys), dtype=bool)
    flags[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=flags[1:])
    return np.flatnonzero(flags)


def _axes_xy(
    right: np.ndarray, forward: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    The horizontal projections of the vehicle axes, renormalised.

    Raw projections shrink by cos(pitch) on a grade, which would scale every
    remembered point toward the car; renormalising leaves a pure rotation,
    exact in yaw and second-order small in grade.
    """
    r_xy = np.asarray(right, dtype=np.float64)[:2]
    f_xy = np.asarray(forward, dtype=np.float64)[:2]
    r_xy = r_xy / max(float(np.hypot(*r_xy)), 1e-9)
    f_xy = f_xy / max(float(np.hypot(*f_xy)), 1e-9)
    return r_xy, f_xy


class PlanningMemory:
    """Accumulated obstacle and road cells, world-anchored, ego-queried."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        # Per-cell NEWEST per-tick mean position, world XY. The mean rather
        # than the cell centre: 0.4 m centre quantisation is up to 0.28 m
        # diagonally, which eats most of the 0.35 m clearance margin.
        self._pos = np.empty((0, 2), dtype=np.float64)
        self._support = np.empty(0, dtype=np.int64)
        self._stamp = np.empty(0, dtype=np.float64)
        self._vehicle_at = np.empty(0, dtype=np.float64)
        self._road_keys = np.empty(0, dtype=np.int64)
        self._road_pos = np.empty((0, 2), dtype=np.float64)
        self._road_stamp = np.empty(0, dtype=np.float64)
        self._odometer = 0.0
        self._last_pos: np.ndarray | None = None

    @property
    def cell_count(self) -> int:
        return len(self._keys)

    @property
    def road_cell_count(self) -> int:
        return len(self._road_keys)

    @property
    def vehicle_cell_count(self) -> int:
        return int(np.count_nonzero(~np.isneginf(self._vehicle_at)))

    def oldest_age_m(self) -> float:
        """Metres travelled since the oldest surviving static observation."""
        static = np.isneginf(self._vehicle_at)
        if not static.any():
            return 0.0
        return float(self._odometer - self._stamp[static].min())

    def update(
        self,
        pos_world: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        obstacles_bev: np.ndarray,
        vehicle_bev: np.ndarray,
        road_bev: np.ndarray,
        now: float,
    ) -> None:
        """
        Fold one tick's planner-band obstacles (and road returns) in.

        `vehicle_bev` marks cells, it does not add them: a vehicle return in
        the band is already in `obstacles_bev`, and the mark is what moves
        its cell onto the wall clock so a mover never smears.
        """
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        if self._last_pos is not None:
            step = float(np.hypot(*(ego - self._last_pos)))
            if step > MEMORY_POSE_JUMP_RESET_M:
                self.clear()
            else:
                self._odometer += step
        self._last_pos = ego

        r_xy, f_xy = _axes_xy(right, forward)

        def to_world(bev: np.ndarray) -> np.ndarray:
            bev = np.asarray(bev, dtype=np.float64)
            if bev.size == 0:
                return np.empty((0, 2), dtype=np.float64)
            return ego + bev[:, [0]] * r_xy + bev[:, [1]] * f_xy

        vehicle_keys = _pack(
            np.floor(to_world(vehicle_bev) / MEMORY_CELL_M).astype(np.int64)
        )
        vehicle_keys = np.unique(vehicle_keys)
        obstacle_world = to_world(obstacles_bev)
        road_world = to_world(road_bev)
        self._ingest_obstacles(obstacle_world, vehicle_keys, now)
        self._evict_contradicted(road_world, obstacle_world)
        self._ingest_road(road_world)
        self._expire(ego, now)

    # --- ingestion -----------------------------------------------------------

    def _ingest_obstacles(
        self, world: np.ndarray, vehicle_keys: np.ndarray, now: float
    ) -> None:
        if len(world) == 0:
            return
        raw_keys = _pack(np.floor(world / MEMORY_CELL_M).astype(np.int64))
        order = np.argsort(raw_keys, kind="stable")
        sorted_keys = raw_keys[order]
        sorted_pts = world[order]
        starts = _group_starts(sorted_keys)
        bounds = np.append(starts, len(sorted_keys))
        counts = np.diff(bounds)
        cell_keys = sorted_keys[starts]
        cell_mean = np.column_stack(
            (
                np.add.reduceat(sorted_pts[:, 0], starts) / counts,
                np.add.reduceat(sorted_pts[:, 1], starts) / counts,
            )
        )
        # A cell containing any vehicle-classified return this tick goes on
        # the wall clock; maximum-reduction in the merge below carries the
        # newest mark forward.
        hit = np.searchsorted(vehicle_keys, cell_keys)
        hit = np.minimum(hit, max(len(vehicle_keys) - 1, 0))
        is_vehicle = (
            vehicle_keys[hit] == cell_keys
            if len(vehicle_keys)
            else np.zeros(len(cell_keys), dtype=bool)
        )
        cell_vehicle_at = np.where(is_vehicle, now, _NEVER)

        # Sort-merge, this tick's cells appended LAST: a stable sort makes
        # "last of each group" mean "newest", so position needs no stamp
        # comparison -- the accumulators' idiom.
        merged_keys = np.concatenate((self._keys, cell_keys))
        merged_pos = np.concatenate((self._pos, cell_mean))
        merged_support = np.concatenate((self._support, counts))
        merged_stamp = np.concatenate(
            (self._stamp, np.full(len(cell_keys), self._odometer))
        )
        merged_vehicle = np.concatenate((self._vehicle_at, cell_vehicle_at))
        order = np.argsort(merged_keys, kind="stable")
        keys = merged_keys[order]
        starts = _group_starts(keys)
        newest = np.append(starts[1:], len(keys)) - 1
        self._keys = keys[starts]
        self._pos = merged_pos[order][newest]
        self._support = np.add.reduceat(merged_support[order], starts)
        self._stamp = np.maximum.reduceat(merged_stamp[order], starts)
        self._vehicle_at = np.maximum.reduceat(merged_vehicle[order], starts)
        # WORLD's semantics for the wall clock: ANY observation of a marked
        # cell refreshes its stamp -- the class stays sticky, and the cell
        # fades only after MEMORY_VEHICLE_TTL_S of NON-observation. Without
        # this, a once-marked cell still being observed as a static surface
        # expired mid-observation and lost its accumulated support.
        seen = np.searchsorted(self._keys, cell_keys)
        marked = ~np.isneginf(self._vehicle_at[seen])
        self._vehicle_at[seen[marked]] = now

    def _evict_contradicted(
        self, road_world: np.ndarray, obstacle_world: np.ndarray
    ) -> None:
        """
        Seeing the GROUND somewhere outranks remembering an obstacle there.

        The vehicle TTL depends on the semantic mark, and unannotated maps
        never produce one -- a crossing car's returns entered as static
        scenery and painted a believed phantom wall across the lane for 20 m
        of travel (and while the ego then stood braked before it, the
        odometer never expired it). So any remembered cell that this tick's
        ROAD-classified returns cover with at least MEMORY_MIN_SUPPORT hits,
        while contributing not a single obstacle return, is dropped: the
        geometric road fallback classifies vacated tarmac as road on any
        map, annotated or not, and the sensors keep re-sampling it while the
        ego is parked. A genuinely occluded kerb receives neither kind of
        return and persists; a visible kerb face receives obstacle returns
        and is protected.
        """
        if len(self._keys) == 0 or len(road_world) == 0:
            return
        road_keys = _pack(
            np.floor(road_world / MEMORY_CELL_M).astype(np.int64)
        )
        road_keys.sort()
        starts = _group_starts(road_keys)
        counts = np.diff(np.append(starts, len(road_keys)))
        covered = road_keys[starts][counts >= MEMORY_MIN_SUPPORT]
        if len(covered) and len(obstacle_world):
            blocked = np.unique(
                _pack(np.floor(obstacle_world / MEMORY_CELL_M).astype(np.int64))
            )
            position = np.searchsorted(blocked, covered)
            position = np.minimum(position, len(blocked) - 1)
            covered = covered[blocked[position] != covered]
        if not len(covered):
            return
        position = np.searchsorted(covered, self._keys)
        position = np.minimum(position, len(covered) - 1)
        contradicted = covered[position] == self._keys
        if contradicted.any():
            self._gather(np.flatnonzero(~contradicted))

    def _ingest_road(self, world: np.ndarray) -> None:
        if len(world) == 0:
            return
        raw_keys = _pack(np.floor(world / MEMORY_CELL_M).astype(np.int64))
        order = np.argsort(raw_keys, kind="stable")
        sorted_keys = raw_keys[order]
        sorted_pts = world[order]
        starts = _group_starts(sorted_keys)
        merged_keys = np.concatenate((self._road_keys, sorted_keys[starts]))
        merged_pos = np.concatenate((self._road_pos, sorted_pts[starts]))
        merged_stamp = np.concatenate(
            (self._road_stamp, np.full(len(starts), self._odometer))
        )
        order = np.argsort(merged_keys, kind="stable")
        keys = merged_keys[order]
        starts = _group_starts(keys)
        newest = np.append(starts[1:], len(keys)) - 1
        self._road_keys = keys[starts]
        self._road_pos = merged_pos[order][newest]
        self._road_stamp = np.maximum.reduceat(merged_stamp[order], starts)

    # --- expiry --------------------------------------------------------------

    def _expire(self, ego: np.ndarray, now: float) -> None:
        if len(self._keys):
            static = np.isneginf(self._vehicle_at)
            fresh = np.where(
                static,
                self._odometer - self._stamp <= MEMORY_DISTANCE_M,
                now - self._vehicle_at <= MEMORY_VEHICLE_TTL_S,
            )
            rel = self._pos - ego
            inside = rel[:, 0] ** 2 + rel[:, 1] ** 2 <= MEMORY_RADIUS_M**2
            self._gather(np.flatnonzero(fresh & inside))
            if len(self._keys) > MEMORY_MAX_CELLS:
                # Newest-K by stamp; re-sorting the selected indices is what
                # keeps the store in key order, which every merge depends on.
                picked = np.argpartition(self._stamp, -MEMORY_MAX_CELLS)[
                    -MEMORY_MAX_CELLS:
                ]
                self._gather(np.sort(picked))
        if len(self._road_keys):
            fresh = self._odometer - self._road_stamp <= MEMORY_DISTANCE_M
            rel = self._road_pos - ego
            inside = rel[:, 0] ** 2 + rel[:, 1] ** 2 <= MEMORY_RADIUS_M**2
            self._gather_road(np.flatnonzero(fresh & inside))
            if len(self._road_keys) > MEMORY_MAX_ROAD_CELLS:
                picked = np.argpartition(
                    self._road_stamp, -MEMORY_MAX_ROAD_CELLS
                )[-MEMORY_MAX_ROAD_CELLS:]
                self._gather_road(np.sort(picked))

    def _gather(self, indices: np.ndarray) -> None:
        self._keys = self._keys[indices]
        self._pos = self._pos[indices]
        self._support = self._support[indices]
        self._stamp = self._stamp[indices]
        self._vehicle_at = self._vehicle_at[indices]

    def _gather_road(self, indices: np.ndarray) -> None:
        self._road_keys = self._road_keys[indices]
        self._road_pos = self._road_pos[indices]
        self._road_stamp = self._road_stamp[indices]

    # --- queries -------------------------------------------------------------

    def obstacles_bev(
        self, pos_world: np.ndarray, right: np.ndarray, forward: np.ndarray
    ) -> np.ndarray:
        """Believed cells in the current BEV frame, decimated for the scan."""
        believed = self._support >= MEMORY_MIN_SUPPORT
        return self._project(self._pos[believed], pos_world, right, forward)

    def road_bev(
        self, pos_world: np.ndarray, right: np.ndarray, forward: np.ndarray
    ) -> np.ndarray:
        """Remembered road cells in the current BEV frame."""
        return self._project(self._road_pos, pos_world, right, forward)

    @staticmethod
    def _project(
        world: np.ndarray,
        pos_world: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
    ) -> np.ndarray:
        if len(world) == 0:
            return _EMPTY_BEV
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        r_xy, f_xy = _axes_xy(right, forward)
        rel = world - ego
        bev = np.column_stack((rel @ r_xy, rel @ f_xy)).astype(np.float32)
        if len(bev) > MEMORY_MAX_QUERY_POINTS:
            # Evenly spaced indices, the planner-cap idiom: an integer stride
            # would systematically favour one end of the key order.
            picked = np.linspace(
                0, len(bev) - 1, MEMORY_MAX_QUERY_POINTS
            ).astype(np.intp)
            bev = bev[picked]
        return bev
