"""
Parking bay detection from road paint, and the selection the user clicks.

Bays are found from the PAINT, not from gaps between parked cars. A gap-based
finder -- measure the hole between two solid things, accept it if the car fits
-- is what production parking assists do, works on unannotated maps, and needs
no new sensing. It also has one fatal property for the case this was asked
for: an empty lot is one enormous gap and offers nothing to find. An empty bay
is *defined* by its dividers, so paint is the only signal that survives the
empty case.

The pipeline is three steps and each is a plain geometric question:

1. **Which way do the dividers run?** All bay dividers in a row are parallel,
   so the answer is the angle whose PERPENDICULAR projection concentrates the
   cells into narrow peaks. Swept, not fitted: a global PCA finds the axis of
   the ROW (tens of metres) rather than of a stripe (a few), which is the
   wrong answer by 90 degrees on exactly the geometry this exists for.
2. **Where are they?** Runs of occupied bins in that projection.
3. **Which pairs bound a bay?** Adjacent stripes at a plausible spacing whose
   lengths overlap by a plausible depth.

Nothing here reaches the planner or either AEB band, and it deliberately has
no route back into them: a bay is a suggestion the user can see and decline,
so a wrong one costs a bad suggestion rather than a phantom brake. The store
is a worker-thread mutable owned by `_poll_once`, the same confinement
argument as `PlanningMemory` and the controller's state.

Qt-free and BeamNGpy-free, like `planner` and `aeb`: config + models + numpy.
The packed-key and renormalised-axes idioms are duplicated from
`planning_map` rather than imported, for the reason its own docstring gives --
importing it would close a sibling edge to buy three lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    MEMORY_POSE_JUMP_RESET_M,
    PARKING_ANGLE_COARSE_DEG,
    PARKING_ANGLE_FINE_DEG,
    PARKING_BAY_MATCH_M,
    PARKING_BAY_MAX_DEPTH_M,
    PARKING_BAY_MEMORY_M,
    PARKING_BAY_MIN_DEPTH_M,
    PARKING_BAY_WIDTH_MAX_M,
    PARKING_BAY_WIDTH_MIN_M,
    PARKING_MARKING_CELL_M,
    PARKING_MARKING_MAX_CELLS,
    PARKING_MARKING_MEMORY_M,
    PARKING_MARKING_RADIUS_M,
    PARKING_MAX_ROWS,
    PARKING_MAX_SLOTS,
    PARKING_MIN_BIN_CELLS,
    PARKING_MIN_MARKING_CELLS,
    PARKING_MIN_STRIPE_CELLS,
    PARKING_OCCUPANCY_MARGIN_M,
    PARKING_OCCUPANCY_MIN_CELLS,
    PARKING_OFFSET_BIN_M,
    PARKING_SCAN_RADIUS_M,
    PARKING_SELECT_MATCH_M,
    PARKING_SLAB_COLUMN_DEPTH_FRACTION,
    PARKING_SLAB_MAX_BAYS,
    PARKING_SLAB_MIN_FILL,
    PARKING_SLAB_MIN_WIDTH_M,
    PARKING_SLAB_NOMINAL_WIDTH_M,
    PARKING_STRIPE_ANGLE_TOL_DEG,
    PARKING_STRIPE_GAP_M,
    PARKING_STRIPE_MAX_WIDTH_M,
    PARKING_STRIPE_MIN_OVERLAP_M,
)
from .models import ParkingBay, ParkingSlot

# 21-bit fields, x high -- planner._cell_keys' packing, covering +-209 km of
# world at the 0.2 m marking cell.
_FIELD = 1 << 21
_BIAS = 1 << 20


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
    stored point toward the car; renormalising leaves a pure rotation.
    """
    r_xy = np.asarray(right, dtype=np.float64)[:2]
    f_xy = np.asarray(forward, dtype=np.float64)[:2]
    r_xy = r_xy / max(float(np.hypot(*r_xy)), 1e-9)
    f_xy = f_xy / max(float(np.hypot(*f_xy)), 1e-9)
    return r_xy, f_xy


class MarkingMemory:
    """World-anchored road-marking cells, forgotten by the metre travelled."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        # Newest per-tick MEAN position rather than the cell centre: the
        # quantity every downstream step measures is a line's OFFSET, and
        # 0.2 m of centre quantisation is a seventh of the narrowest bay
        # tolerance.
        self._pos = np.empty((0, 2), dtype=np.float64)
        self._stamp = np.empty(0, dtype=np.float64)
        self._odometer = 0.0
        self._last_pos: np.ndarray | None = None

    @property
    def cell_count(self) -> int:
        return len(self._keys)

    @property
    def travelled_m(self) -> float:
        """Metres of ego travel, the clock every parking store ages on."""
        return self._odometer

    def update(
        self,
        pos_world: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        marking_bev: np.ndarray,
    ) -> None:
        """Fold one tick's marking returns in, and expire what has aged out."""
        ego = np.asarray(pos_world, dtype=np.float64)[:2]
        if self._last_pos is not None:
            step = float(np.hypot(*(ego - self._last_pos)))
            if step > MEMORY_POSE_JUMP_RESET_M:
                # The same teleport guard as every other store: a respawn must
                # never leave the old lot's bays standing in the new map.
                self.clear()
            else:
                self._odometer += step
        self._last_pos = ego

        bev = np.asarray(marking_bev, dtype=np.float64)
        if bev.size:
            r_xy, f_xy = _axes_xy(right, forward)
            world = ego + bev[:, [0]] * r_xy + bev[:, [1]] * f_xy
            self._ingest(world)
        self._expire(ego)

    def cells_world(self) -> np.ndarray:
        """Every surviving marking cell, world XY."""
        return self._pos

    def _ingest(self, world: np.ndarray) -> None:
        raw_keys = _pack(
            np.floor(world / PARKING_MARKING_CELL_M).astype(np.int64)
        )
        order = np.argsort(raw_keys, kind="stable")
        sorted_keys = raw_keys[order]
        sorted_pts = world[order]
        starts = _group_starts(sorted_keys)
        bounds = np.append(starts, len(sorted_keys))
        counts = np.diff(bounds)
        cell_mean = np.column_stack(
            (
                np.add.reduceat(sorted_pts[:, 0], starts) / counts,
                np.add.reduceat(sorted_pts[:, 1], starts) / counts,
            )
        )
        # Sort-merge with this tick appended LAST, so a stable sort makes
        # "last of each group" mean "newest" with no stamp comparison -- the
        # accumulators' idiom throughout this codebase.
        merged_keys = np.concatenate((self._keys, sorted_keys[starts]))
        merged_pos = np.concatenate((self._pos, cell_mean))
        merged_stamp = np.concatenate(
            (self._stamp, np.full(len(starts), self._odometer))
        )
        order = np.argsort(merged_keys, kind="stable")
        keys = merged_keys[order]
        starts = _group_starts(keys)
        newest = np.append(starts[1:], len(keys)) - 1
        self._keys = keys[starts]
        self._pos = merged_pos[order][newest]
        self._stamp = np.maximum.reduceat(merged_stamp[order], starts)

    def _expire(self, ego: np.ndarray) -> None:
        if not len(self._keys):
            return
        fresh = self._odometer - self._stamp <= PARKING_MARKING_MEMORY_M
        rel = self._pos - ego
        inside = (
            rel[:, 0] ** 2 + rel[:, 1] ** 2 <= PARKING_MARKING_RADIUS_M**2
        )
        self._gather(np.flatnonzero(fresh & inside))
        if len(self._keys) > PARKING_MARKING_MAX_CELLS:
            # Newest-K by stamp; re-sorting the picked indices is what keeps
            # the store in key order, which the merge depends on.
            picked = np.argpartition(self._stamp, -PARKING_MARKING_MAX_CELLS)[
                -PARKING_MARKING_MAX_CELLS:
            ]
            self._gather(np.sort(picked))

    def _gather(self, indices: np.ndarray) -> None:
        self._keys = self._keys[indices]
        self._pos = self._pos[indices]
        self._stamp = self._stamp[indices]


def _stripe_sharpness(counts: np.ndarray, max_width_bins: int) -> np.ndarray:
    """
    Per row, the sweep's score: cells per bin, summed over narrow runs only.

    Two separate things have to be true of this score, and each was found by
    an angle it got wrong.

    **Wide runs must score nothing.** A lot often paints one long line across
    the HEADS of its bays, perpendicular to the dividers. The obvious
    concentration measure -- sum of squared bin counts -- piles that line into
    one bin at its own angle and, squared, rewards the single tall peak more
    than eight genuine stripes: 8 stripes of 26 cells score 5,408 against the
    head line's 7,744, so the row comes back measured 90 degrees out.
    Restricting the sum to runs no wider than a stripe fixes that, because at
    the head line's angle every divider is seen along its length and spans
    twenty-odd bins.

    **And the score must not PLATEAU.** Mass alone does not distinguish
    angles once every cell already lies in some narrow run: a 6 m divider
    stays inside a 0.75 m cap for +-7 degrees, so mass was flat across a
    14-degree band and `argmax` simply took its first element -- measured, it
    returned 83 degrees for a row of dividers lying at exactly 90, and the
    stripes then merged and no bay survived. Dividing each run's mass by its
    width makes the score keep climbing as the projection tightens, so the
    true angle wins the band outright: on that same pair, 62.0 at 90 degrees
    against 20.7 at 83.
    """
    rows, bins = counts.shape
    occupied = counts > 0
    pad = np.zeros((rows, 1), dtype=bool)
    padded = np.concatenate((pad, occupied, pad), axis=1)
    # Padding per ROW is what stops a run spanning a row boundary, which is
    # what lets the flattened starts and ends pair up positionally below.
    starts = np.flatnonzero((padded[:, 1:-1] & ~padded[:, :-2]).reshape(-1))
    ends = np.flatnonzero((padded[:, 1:-1] & ~padded[:, 2:]).reshape(-1))
    if not len(starts):
        return np.zeros(rows, dtype=np.float64)
    flat = np.concatenate(([0], np.cumsum(counts.reshape(-1))))
    mass = (flat[ends + 1] - flat[starts]).astype(np.float64)
    widths = (ends - starts + 1).astype(np.float64)
    narrow = widths <= max_width_bins
    return np.bincount(
        starts[narrow] // bins,
        weights=mass[narrow] / widths[narrow],
        minlength=rows,
    )


def _projection_counts(
    points: np.ndarray, angles: np.ndarray, half_span_m: float
) -> np.ndarray:
    """Offset-bin histograms of `points` for every angle, as one (A, B) grid."""
    # The perpendicular of the along-stripe unit vector (cos a, sin a).
    normals = np.column_stack((-np.sin(angles), np.cos(angles)))
    across = points @ normals.T
    bins = int(math.ceil(2.0 * half_span_m / PARKING_OFFSET_BIN_M)) + 1
    index = np.floor(
        (across + half_span_m) / PARKING_OFFSET_BIN_M
    ).astype(np.int64)
    np.clip(index, 0, bins - 1, out=index)
    offsets = np.arange(len(angles), dtype=np.int64)[None, :] * bins
    counts = np.bincount(
        (index + offsets).reshape(-1), minlength=len(angles) * bins
    )
    return _clear_thin_bins(counts.reshape(len(angles), bins))


def _clear_thin_bins(counts: np.ndarray) -> np.ndarray:
    """Drop bins too thinly populated to be a divider seen end-on."""
    return np.where(counts >= PARKING_MIN_BIN_CELLS, counts, 0)


def dominant_axis(points: np.ndarray) -> float:
    """
    The angle in [0, pi) along which the marking stripes run.

    Swept rather than fitted. A global PCA over every marking cell in a bay
    row returns the direction of the ROW -- eight 5 m stripes spread over
    20 m of frontage have their greatest variance across the frontage, not
    along a stripe -- which is the wrong answer by exactly 90 degrees on the
    one geometry this function exists for.
    """
    half_span = float(np.abs(points).max()) + PARKING_OFFSET_BIN_M
    width_bins = max(
        1, int(round(PARKING_STRIPE_MAX_WIDTH_M / PARKING_OFFSET_BIN_M))
    )
    coarse = np.radians(np.arange(0.0, 180.0, PARKING_ANGLE_COARSE_DEG))
    counts = _projection_counts(points, coarse, half_span)
    best = float(coarse[int(np.argmax(_stripe_sharpness(counts, width_bins)))])

    fine = best + np.radians(
        np.arange(
            -PARKING_ANGLE_COARSE_DEG,
            PARKING_ANGLE_COARSE_DEG + PARKING_ANGLE_FINE_DEG * 0.5,
            PARKING_ANGLE_FINE_DEG,
        )
    )
    counts = _projection_counts(points, fine, half_span)
    refined = float(fine[int(np.argmax(_stripe_sharpness(counts, width_bins)))])
    return refined % math.pi


@dataclass(frozen=True)
class _Stripe:
    """
    One believed divider, carried in its OWN geometry rather than in the
    sweep's frame.

    The offset/lo/hi triple only means anything relative to the pass that
    produced it, and pairing has to work ACROSS passes -- a row whose
    dividers are split between two sweeps otherwise loses every bay at the
    seam, because neighbours found on different passes could never meet.
    Centre, direction and half-length are frame-free, so one pairing step can
    consider every divider the scan found however it found it.
    """

    centre: tuple[float, float]
    direction: tuple[float, float]
    half_length_m: float
    cells: int


def _stripes(
    points: np.ndarray, angle: float
) -> tuple[list[_Stripe], np.ndarray]:
    """
    Every believed divider line at `angle`, plus which cells belong to one.

    The membership mask is what lets `find_bays` sweep more than once: a lot
    is rarely one row, and the sweep can only return ONE angle. Removing the
    cells this angle claimed leaves the other rows to be found by the next
    pass.
    """
    along_axis = np.array([math.cos(angle), math.sin(angle)])
    across_axis = np.array([-math.sin(angle), math.cos(angle)])
    along = points @ along_axis
    across = points @ across_axis

    half_span = float(np.abs(across).max()) + PARKING_OFFSET_BIN_M
    bins = int(math.ceil(2.0 * half_span / PARKING_OFFSET_BIN_M)) + 1
    index = np.floor((across + half_span) / PARKING_OFFSET_BIN_M).astype(
        np.int64
    )
    np.clip(index, 0, bins - 1, out=index)
    # The same floor the sweep uses, for the same reason: a line crossing the
    # projection must not bridge the dividers into one unusable run.
    counts = _clear_thin_bins(np.bincount(index, minlength=bins))

    occupied = counts > 0
    padded = np.concatenate(([False], occupied, [False]))
    starts = np.flatnonzero(padded[1:-1] & ~padded[:-2])
    ends = np.flatnonzero(padded[1:-1] & ~padded[2:])
    width_bins = max(
        1, int(round(PARKING_STRIPE_MAX_WIDTH_M / PARKING_OFFSET_BIN_M))
    )
    # Only the WIDTH cap here; the cell-count floor moved to the per-segment
    # test below, because an offset run may hold several dividers and it is
    # each one that has to carry its own evidence.
    run_bins = ends - starts + 1
    keep = run_bins <= width_bins
    # A run TOO WIDE to be a divider is not automatically rubbish. Some lots
    # annotate whole bay quads rather than the lines between them, and there
    # the entire row arrives as one solid run -- so discarding everything wide
    # meant those lots produced no dividers, no bays, and no counter saying
    # why. See PARKING_SLAB_MIN_WIDTH_M.
    slab = run_bins * PARKING_OFFSET_BIN_M >= PARKING_SLAB_MIN_WIDTH_M
    slab_stripes, slab_member, slab_count = _slab_stripes(
        points,
        along,
        across,
        index,
        starts[slab],
        ends[slab],
        along_axis,
        across_axis,
    )
    starts, ends = starts[keep], ends[keep]
    if not len(starts):
        return slab_stripes, slab_member, slab_count

    # bin -> stripe, so every point finds its stripe in one gather. The loop
    # runs once per STRIPE (tens), never per point -- `merge_cell_runs`' rule.
    label_of_bin = np.full(bins, -1, dtype=np.int64)
    for stripe, (lo, hi) in enumerate(zip(starts, ends)):
        label_of_bin[lo : hi + 1] = stripe
    labels = label_of_bin[index]
    candidate = labels >= 0
    cell_index = np.flatnonzero(candidate)
    order = np.argsort(labels[candidate], kind="stable")
    cell_index = cell_index[order]
    grouped = labels[candidate][order]
    group = _group_starts(grouped)
    bounds = np.append(group, len(grouped))

    # An offset run is not yet a divider: it is every cell at this
    # perpendicular offset, and in a lot with TWO FACING ROWS across an aisle
    # that is one divider from each row, at the same offset, tens of metres
    # apart. Left merged, the pair spans both rows and every bay between them
    # fails PARKING_BAY_MAX_DEPTH_M -- measured, 5 bays + 5 bays came back as
    # 0. So each run is split into segments along its own length wherever it
    # gaps by more than PARKING_STRIPE_GAP_M.
    #
    # The loop runs once per offset run (tens), never per point.
    stripes: list[_Stripe] = []
    member = np.zeros(len(points), dtype=bool)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        rows = cell_index[start:stop]
        run_along = along[rows]
        run_order = np.argsort(run_along, kind="stable")
        rows = rows[run_order]
        run_along = run_along[run_order]
        splits = np.flatnonzero(np.diff(run_along) > PARKING_STRIPE_GAP_M) + 1
        for segment in np.split(np.arange(len(rows)), splits):
            if len(segment) < PARKING_MIN_STRIPE_CELLS:
                continue
            picked = rows[segment]
            offset = float(across[picked].mean())
            lo = float(run_along[segment[0]])
            hi = float(run_along[segment[-1]])
            centre = across_axis * offset + along_axis * (lo + hi) * 0.5
            stripes.append(
                _Stripe(
                    (float(centre[0]), float(centre[1])),
                    (float(along_axis[0]), float(along_axis[1])),
                    (hi - lo) * 0.5,
                    int(len(segment)),
                )
            )
            member[picked] = True
    return stripes + slab_stripes, member | slab_member, slab_count


def _slab_stripes(
    points: np.ndarray,
    along: np.ndarray,
    across: np.ndarray,
    index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    along_axis: np.ndarray,
    across_axis: np.ndarray,
) -> tuple[list[_Stripe], np.ndarray, int]:
    """
    Synthetic dividers across a filled bay ROW, for lots that annotate quads.

    A row of bays and one solid slab are the same painted region as far as
    this sensor is concerned, so the row own extents are all there is to go
    on: one of them is a bay DEPTH and the other is the row length, and which
    is which is settled by PARKING_BAY_MIN/MAX_DEPTH_M rather than by assuming
    the longer side. Dividing the length by a nominal bay width gives a count.

    **The count is a considered guess and cannot be anything else** -- the
    dividers are exactly what the annotation did not draw, and a 17.5 m row
    divides into 6, 7 or 8 bays all of which are plausible widths. Emitting
    synthetic stripes rather than finished bays is what keeps that honest:
    they pass through the same pairing, width, depth and occupancy filters
    every real divider does, and what comes out is a CANDIDATE to be clicked.

    **Not everything annotated is a bay.** Hatched keep-clear zones, chevron
    end caps and aisle markings can share a slab with the row they adjoin, and
    a solid quad carries nothing to tell them apart. The per-column depth test
    in `_row_dividers` trims a zone of a DIFFERENT depth from the row; one of
    the same depth is indistinguishable here and will be offered. That is the
    cost of reading a row off a shape, and it is why these are candidates
    rather than destinations.
    """
    stripes: list[_Stripe] = []
    member = np.zeros(len(points), dtype=bool)
    slabs = 0
    for lo_bin, hi_bin in zip(starts, ends):
        picked = np.flatnonzero((index >= lo_bin) & (index <= hi_bin))
        if len(picked) < PARKING_MIN_MARKING_CELLS:
            continue
        run_across, run_along = across[picked], along[picked]
        span_across = float(run_across.max() - run_across.min())
        span_along = float(run_along.max() - run_along.min())
        # Is this actually a FILLED region, or several things that merely
        # project onto the same wide band of offsets? See
        # PARKING_SLAB_MIN_FILL -- getting this wrong does not merely add
        # wrong bays, it CONSUMES the cells of every row a later sweep would
        # have found.
        area = span_across * span_along
        if area <= 0.0 or len(picked) < PARKING_SLAB_MIN_FILL * area / (
            PARKING_MARKING_CELL_M**2
        ):
            continue
        slabs += 1
        # Which extent is the bay depth. Both are tried; assuming the longer
        # side is the row is the trap CLAUDE.md already records for the hand
        # labeller, where two 2.4 m bays measure 4.8 m across against a 5.5 m
        # depth, so the longer side is the DEPTH and slicing it invents bays
        # no lot has.
        wide_enough = 2.0 * PARKING_BAY_WIDTH_MIN_M
        if _is_bay_depth(span_across) and span_along >= wide_enough:
            row, perp, depth = run_along, run_across, span_across
            row_axis, cut_axis = along_axis, across_axis
            fixed = float(run_across.mean())
        elif _is_bay_depth(span_along) and span_across >= wide_enough:
            row, perp, depth = run_across, run_along, span_along
            row_axis, cut_axis = across_axis, along_axis
            fixed = float(run_along.mean())
        else:
            continue
        found = _row_dividers(row, perp, depth, fixed, row_axis, cut_axis)
        if not found:
            continue
        # Claimed only once it has actually yielded dividers. A slab that
        # produced nothing must be left in the cloud for the next sweep, the
        # same way an offset run bounding no bay is.
        member[picked] = True
        stripes.extend(found)
    return stripes, member, slabs


def _is_bay_depth(extent: float) -> bool:
    return PARKING_BAY_MIN_DEPTH_M <= extent <= PARKING_BAY_MAX_DEPTH_M


def _row_dividers(
    row: np.ndarray,
    perp: np.ndarray,
    depth: float,
    fixed: float,
    row_axis: np.ndarray,
    cut_axis: np.ndarray,
) -> list[_Stripe]:
    """Evenly spaced dividers over the stretch of a row that is one bay deep."""
    low, high = float(row.min()), float(row.max())
    length = high - low
    count = max(
        1,
        min(
            int(round(length / PARKING_SLAB_NOMINAL_WIDTH_M)),
            PARKING_SLAB_MAX_BAYS,
        ),
    )
    width = length / count
    if not PARKING_BAY_WIDTH_MIN_M <= width <= PARKING_BAY_WIDTH_MAX_M:
        return []
    # Columns that are not the row own depth are something else painted
    # alongside it -- a hatched zone, a chevron cap, an aisle marking. Only a
    # contiguous stretch of full-depth columns is divided, so a differently
    # shaped neighbour is trimmed rather than turned into bays.
    edges = low + np.arange(count + 1) * width
    column = np.clip(
        np.searchsorted(edges, row, side="right") - 1, 0, count - 1
    )
    # Full DEPTH, not merely populated. Checking only that a column holds
    # cells lets a shallow hatched area at the end of a row through: it is
    # painted, so its columns are populated, and it became bays.
    wanted = depth * PARKING_SLAB_COLUMN_DEPTH_FRACTION
    deep = np.zeros(count, dtype=bool)
    for slot in range(count):
        inside = perp[column == slot]
        deep[slot] = (
            len(inside) >= PARKING_MIN_BIN_CELLS
            and float(inside.max() - inside.min()) >= wanted
        )
    span = _longest_run(deep)
    if span is None:
        return []
    first, last = span
    return [
        _Stripe(
            centre=(
                float(row_axis[0] * edges[cut] + cut_axis[0] * fixed),
                float(row_axis[1] * edges[cut] + cut_axis[1] * fixed),
            ),
            direction=(float(cut_axis[0]), float(cut_axis[1])),
            half_length_m=depth * 0.5,
            cells=PARKING_MIN_STRIPE_CELLS,
        )
        for cut in range(first, last + 2)
    ]


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """The longest contiguous True span as (first, last), or None."""
    best = current = None
    for position, flag in enumerate(flags):
        if not flag:
            current = None
            continue
        current = (
            (position, position) if current is None else (current[0], position)
        )
        if best is None or current[1] - current[0] > best[1] - best[0]:
            best = current
    return best


def _occupancy(
    obstacles_rebased: np.ndarray | None,
    centre: np.ndarray,
    axis: np.ndarray,
    normal: np.ndarray,
    depth_m: float,
    width_m: float,
) -> bool:
    """Whether anything is standing inside the bay, shrunk by the margin."""
    if obstacles_rebased is None or not len(obstacles_rebased):
        return False
    half_depth = max(depth_m * 0.5 - PARKING_OCCUPANCY_MARGIN_M, 0.05)
    half_width = max(width_m * 0.5 - PARKING_OCCUPANCY_MARGIN_M, 0.05)
    delta = obstacles_rebased - centre
    inside = (np.abs(delta @ axis) <= half_depth) & (
        np.abs(delta @ normal) <= half_width
    )
    return bool(np.count_nonzero(inside) >= PARKING_OCCUPANCY_MIN_CELLS)


@dataclass
class ScanReport:
    """
    Why the bays that were not offered were not offered.

    Detection is a chain of geometric filters and a screenshot only shows the
    survivors, so "why is there no outline on that painted bay" is
    unanswerable from the picture. Each counter names the filter that
    consumed a candidate, which turns the question into one log line.
    """

    cells_in_store: int = 0
    cells_in_range: int = 0
    rows_swept: int = 0
    stripes_found: int = 0
    rejected_width: int = 0
    """Adjacent stripe pairs outside PARKING_BAY_WIDTH_MIN/MAX_M."""
    rejected_depth: int = 0
    """Pairs whose dividers overlapped too little, or far too much (an aisle)."""
    unpaired_stripes: int = 0
    """Dividers that found no partner at all -- a row's outer edges, mostly."""
    bays_found: int = 0
    bays_capped: int = 0
    slabs_found: int = 0
    """
    Painted runs too wide to be dividers -- whole bay QUADS rather than the
    lines between them.

    Counted because a lot annotated that way used to report plenty of marking
    cells, zero dividers and zero of every rejection reason, so the log could
    not distinguish "the paint never annotated" from "the paint annotated as
    something this scan discards". Those are opposite problems.
    """

    def summary(self) -> str:
        return (
            f"{self.cells_in_store} marking cells "
            f"({self.cells_in_range} in range), {self.rows_swept} row "
            f"sweep(s), {self.stripes_found} dividers -> {self.bays_found} "
            f"bays; rejected {self.rejected_width} on width, "
            f"{self.rejected_depth} on depth, {self.unpaired_stripes} "
            f"unpaired, {self.bays_capped} over the cap"
            + (
                f"; {self.slabs_found} filled slab(s)"
                if self.slabs_found
                else ""
            )
        )


def find_bays(
    marking_world: np.ndarray,
    obstacles_world: np.ndarray,
    ego_xy: np.ndarray,
    report: ScanReport | None = None,
) -> tuple[ParkingBay, ...]:
    """
    Candidate bays from accumulated paint, nearest to the ego first.

    **Swept more than once, because a lot is rarely one row.** The sweep can
    only return ONE angle, so a single pass finds whichever row carries the
    most paint and silently drops every row at a different orientation --
    which in practice is the row you are facing keeping the bays and the rest
    of the lot having none. That reads exactly like a sensor that only looks
    forward, and it is not: nothing here filters by bearing, and a row
    directly behind the car is found perfectly when it is the only one.
    Each pass removes the cells its stripes claimed and re-sweeps the rest,
    so rows at 90 degrees to each other are both found.

    `obstacles_world` decides only whether a bay is drawn as occupied; it
    never adds or removes one. A bay whose paint is there is a bay, and
    saying "that one has a car in it" is a different claim from "that one
    does not exist".
    """
    marking = np.asarray(marking_world, dtype=np.float64)
    ego = np.asarray(ego_xy, dtype=np.float64)[:2]
    tally = report if report is not None else ScanReport()
    if marking.ndim != 2 or marking.shape[1] != 2:
        return ()
    tally.cells_in_store = len(marking)
    if len(marking):
        rel = marking - ego
        marking = marking[
            rel[:, 0] ** 2 + rel[:, 1] ** 2 <= PARKING_SCAN_RADIUS_M**2
        ]
    tally.cells_in_range = len(marking)
    if len(marking) < PARKING_MIN_MARKING_CELLS:
        return ()

    # Centred so the sweep's offset span is bounded by the cloud's own extent
    # rather than by the world origin, which can be kilometres away. Held
    # fixed across passes so every row lands in one coordinate system.
    origin = marking.mean(axis=0)
    local = marking - origin
    obstacles = np.asarray(obstacles_world, dtype=np.float64)
    have_obstacles = (
        obstacles.ndim == 2 and obstacles.shape[1] == 2 and len(obstacles) > 0
    )
    rebased = obstacles - origin if have_obstacles else None

    all_stripes: list[_Stripe] = []
    remaining = local
    for _ in range(PARKING_MAX_ROWS):
        if len(remaining) < PARKING_MIN_MARKING_CELLS:
            break
        angle = dominant_axis(remaining)
        stripes, claimed, slabs = _stripes(remaining, angle)
        tally.slabs_found += slabs
        if not len(stripes):
            break
        tally.rows_swept += 1
        tally.stripes_found += len(stripes)
        all_stripes.extend(stripes)
        # Whether or not this angle yielded a BAY, its stripes are consumed:
        # leaving them in would make the next pass re-find the same angle for
        # ever, and the loop bound would be the only thing ending it.
        remaining = remaining[~claimed]

    # ONE pairing step over every divider the scan found, whichever pass
    # found it. Pairing per pass is what a real lot broke on: a row that is
    # slightly curved, or two rows at similar angles, get their dividers
    # split between sweeps, and neighbours on different passes could never
    # meet -- measured live at 18 dividers -> 12 bays with 6 unpaired, and
    # 10 dividers -> 5 bays with 5 unpaired on a single sweep.
    bays = _pair_stripes(all_stripes, origin, ego, rebased, tally)
    bays.sort(key=lambda item: item[0])
    tally.bays_found = len(bays)
    tally.bays_capped = max(0, len(bays) - PARKING_MAX_SLOTS)
    return tuple(bay for _, bay in bays[:PARKING_MAX_SLOTS])


def _pair_stripes(
    stripes: list[_Stripe],
    origin: np.ndarray,
    ego: np.ndarray,
    obstacles_rebased: np.ndarray | None,
    tally: ScanReport,
) -> list[tuple[float, ParkingBay]]:
    """
    Every pair of dividers that bounds a plausible bay, measured in 2D.

    Each divider takes its NEAREST valid partner and a pair is emitted once,
    so both members of an adjacent pair nominate each other and a divider
    between two bays serves both. Everything is measured from the two
    dividers' own geometry -- no sweep frame, no sort order -- which is what
    lets dividers found on different passes pair with each other.
    """
    if len(stripes) < 2:
        tally.unpaired_stripes += len(stripes)
        return []
    ego_local = ego - origin
    tolerance = math.cos(math.radians(PARKING_STRIPE_ANGLE_TOL_DEG))
    accepted: dict[tuple[int, int], tuple[float, ParkingBay]] = {}
    for index, left in enumerate(stripes):
        # The nearest valid partner on EACH SIDE, not one nearest overall.
        # A divider in the middle of a row bounds the bay to its left AND the
        # one to its right, and nominating once loses whichever it did not
        # pick -- measured, that cost two bays of ten on a pair of facing
        # rows. Both members of an adjacent pair then nominate each other,
        # which is what makes the dedupe exact rather than lossy.
        best: dict[bool, tuple[float, int, tuple[float, ParkingBay]]] = {}
        width_rejects = depth_rejects = 0
        for other, right in enumerate(stripes):
            if other == index:
                continue
            left_dir = np.asarray(left.direction)
            right_dir = np.asarray(right.direction)
            # Direction is an AXIS, not an arrow: a divider found on another
            # pass may be described the other way round.
            alignment = float(left_dir @ right_dir)
            if abs(alignment) < tolerance:
                continue
            if alignment < 0.0:
                right_dir = -right_dir
            axis = left_dir + right_dir
            axis = axis / max(float(np.hypot(*axis)), 1e-9)
            normal = np.asarray((-axis[1], axis[0]))

            delta = np.asarray(right.centre) - np.asarray(left.centre)
            offset = float(delta @ normal)
            side = offset > 0.0
            width = abs(offset)
            if not PARKING_BAY_WIDTH_MIN_M <= width <= PARKING_BAY_WIDTH_MAX_M:
                width_rejects += 1
                continue
            # The overlap is an ADJACENCY test, not the depth. In an ANGLED
            # lot neighbours are staggered along their own direction, and
            # subtracting that stagger from the depth rejected a 60-degree
            # bay by 4 cm. Depth is the shorter divider's own length.
            left_mid = float(np.asarray(left.centre) @ axis)
            right_mid = float(np.asarray(right.centre) @ axis)
            overlap = min(
                left_mid + left.half_length_m, right_mid + right.half_length_m
            ) - max(
                left_mid - left.half_length_m, right_mid - right.half_length_m
            )
            depth = 2.0 * min(left.half_length_m, right.half_length_m)
            if (
                overlap < PARKING_STRIPE_MIN_OVERLAP_M
                or not PARKING_BAY_MIN_DEPTH_M
                <= depth
                <= PARKING_BAY_MAX_DEPTH_M
            ):
                depth_rejects += 1
                continue

            centre = (np.asarray(left.centre) + np.asarray(right.centre)) * 0.5
            # The MOUTH is whichever end of the centreline is nearer the ego,
            # so the axis runs from there into the bay -- a heuristic that
            # reads the aisle's position off where the car actually is.
            towards = centre - ego_local
            head = axis if float(towards @ axis) > 0.0 else -axis
            occupied = _occupancy(
                obstacles_rebased, centre, head, normal, depth, width
            )
            world_centre = origin + centre
            bay = (
                float(np.hypot(*(world_centre - ego))),
                ParkingBay(
                    centre=(float(world_centre[0]), float(world_centre[1])),
                    axis=(float(head[0]), float(head[1])),
                    width_m=float(width),
                    depth_m=float(depth),
                    occupied=occupied,
                    stripe_cells=left.cells + right.cells,
                ),
            )
            if side not in best or width < best[side][0]:
                best[side] = (width, other, bay)
        if not best:
            tally.rejected_width += width_rejects
            tally.rejected_depth += depth_rejects
            tally.unpaired_stripes += 1
            continue
        for _, partner_index, bay in best.values():
            key = (min(index, partner_index), max(index, partner_index))
            accepted.setdefault(key, bay)
    return list(accepted.values())


def remember_bays(
    previous: tuple[ParkingBay, ...],
    found: tuple[ParkingBay, ...],
    travelled_m: float,
    seen_at: dict[tuple[int, int], float],
    ego_xy: np.ndarray,
) -> tuple[ParkingBay, ...]:
    """
    This scan's bays, plus recently-seen ones the scan happened to miss.

    Detection is a chain of geometric filters over an accumulating cloud, so a
    bay sits near several thresholds at once and a single scan can drop it and
    find it again a moment later. Reported from the app as bays flashing --
    and, far worse, as the SELECTION going away with them, because a selection
    is matched against the offered set every scan.

    Bays are remembered by the METRE travelled, the two-clocks rule the WORLD
    stores and `PlanningMemory` already use, and for the same reason: paint
    does not move and the ego pose is ground truth, so a bay seen 5 m ago is
    exactly as valid as one seen now -- what makes it stale is the car leaving,
    not the clock. A freshly found bay always replaces its remembered twin, so
    memory only ever fills gaps and never overrides what the sensors say now.
    """
    keep: list[ParkingBay] = []
    fresh_keys = set()
    for bay in found:
        key = _bay_key(bay)
        fresh_keys.add(key)
        seen_at[key] = travelled_m
        keep.append(bay)
    for bay in previous:
        key = _bay_key(bay)
        if key in fresh_keys:
            continue
        last = seen_at.get(key)
        if last is None or travelled_m - last > PARKING_BAY_MEMORY_M:
            seen_at.pop(key, None)
            continue
        rel = np.asarray(bay.centre, dtype=np.float64) - np.asarray(
            ego_xy, dtype=np.float64
        )[:2]
        if float(np.hypot(*rel)) > PARKING_SCAN_RADIUS_M:
            seen_at.pop(key, None)
            continue
        keep.append(bay)
    keep.sort(
        key=lambda bay: float(
            np.hypot(*(np.asarray(bay.centre) - np.asarray(ego_xy)[:2]))
        )
    )
    return tuple(keep[:PARKING_MAX_SLOTS])


def _bay_key(bay: ParkingBay) -> tuple[int, int]:
    """
    A bay's identity: its centre on a coarse grid.

    Coarse on purpose. The same physical bay is re-measured every scan and its
    centre wanders by a few centimetres as cells come and go, so an exact key
    would make every scan a different bay and remember nothing.
    """
    return (
        int(round(bay.centre[0] / PARKING_BAY_MATCH_M)),
        int(round(bay.centre[1] / PARKING_BAY_MATCH_M)),
    )


def match_selection(
    bays: tuple[ParkingBay, ...], selected_world: tuple[float, float] | None
) -> ParkingBay | None:
    """
    The bay a held selection now refers to, or None if it matches nothing.

    Indices are not stable across a rescan, so a selection is held as a world
    pose and re-matched here. Returning None rather than the nearest bay
    regardless is the point: a selection that has drifted out of the scan --
    or whose paint has aged out of the store -- must go quiet rather than
    silently become a different bay.
    """
    if selected_world is None or not bays:
        return None
    target = np.asarray(selected_world, dtype=np.float64)
    centres = np.array([bay.centre for bay in bays], dtype=np.float64)
    distances = np.hypot(*(centres - target).T)
    nearest = int(np.argmin(distances))
    if distances[nearest] > PARKING_SELECT_MATCH_M:
        return None
    return bays[nearest]


def project_bays(
    bays: tuple[ParkingBay, ...],
    pos_world: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    selected_world: tuple[float, float] | None = None,
) -> tuple[ParkingSlot, ...]:
    """
    World bays in the current BEV frame, so they stay glued between scans.

    The set is rebuilt only every `PARKING_SCAN_INTERVAL_S`; projecting the
    world geometry every tick is what stops the overlay lagging the car.
    """
    if not bays:
        return ()
    ego = np.asarray(pos_world, dtype=np.float64)[:2]
    r_xy, f_xy = _axes_xy(right, forward)
    selected = match_selection(bays, selected_world)
    slots: list[ParkingSlot] = []
    for bay in bays:
        rel = np.asarray(bay.centre, dtype=np.float64) - ego
        axis = np.asarray(bay.axis, dtype=np.float64)
        axis_right = float(axis @ r_xy)
        axis_forward = float(axis @ f_xy)
        slots.append(
            ParkingSlot(
                centre_right_m=float(rel @ r_xy),
                centre_forward_m=float(rel @ f_xy),
                # Bearing from +forward toward +right, the display frame's
                # own convention -- see ParkingSlot.heading_rad.
                heading_rad=float(math.atan2(axis_right, axis_forward)),
                width_m=bay.width_m,
                depth_m=bay.depth_m,
                occupied=bay.occupied,
                stripe_cells=bay.stripe_cells,
                centre_world=bay.centre,
                selected=bay is selected,
            )
        )
    return tuple(slots)


def slot_contains(slot: ParkingSlot, right_m: float, forward_m: float) -> bool:
    """Whether a BEV point falls inside a bay's rectangle. Hit-testing."""
    dr = right_m - slot.centre_right_m
    df = forward_m - slot.centre_forward_m
    sin_h, cos_h = math.sin(slot.heading_rad), math.cos(slot.heading_rad)
    # Into the bay, and across it: the inverse of the heading rotation.
    depth = df * cos_h + dr * sin_h
    across = dr * cos_h - df * sin_h
    return (
        abs(depth) <= slot.depth_m * 0.5 and abs(across) <= slot.width_m * 0.5
    )
