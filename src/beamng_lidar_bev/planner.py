"""
Geometric path selection from a BEV obstacle cloud.

Pure numpy: no Qt, no BeamNGpy, no state. Everything works in the BEV display
frame produced by ``geometry.world_points_to_bev`` -- ``(right, forward)`` metres
about the vehicle reference node.

Sign convention, used consistently throughout this module and the BEV overlay:
**positive curvature turns left**. Note this is the opposite of BeamNG's
steering input, where positive is RIGHT -- ``controller.STEERING_SIGN`` is what
reconciles the two, and it is the only place that conversion happens.

An arc of curvature ``k`` is the circle of radius ``R = 1/|k|`` centred at
``(-1/k, 0)``: to the left of the car when ``k > 0``, to the right when ``k < 0``.
That single expression is what lets the whole candidate fan be scanned against
the whole obstacle cloud as one ``(obstacles x arcs)`` matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import (
    AEB_POROSITY_AZIMUTH_DEG,
    AEB_POROSITY_GAP_M,
    AEB_POROSITY_MIN_HITS,
    CLEARANCE_MARGIN_M,
    CORRIDOR_BAND_M,
    COST_CLEARANCE,
    COST_FREE_DISTANCE,
    COST_KEEP_RIGHT,
    COST_NAV_HEADING,
    COST_ROAD_BONUS,
    COST_ROUTE_HEADING,
    COST_ROUTE_XTRACK,
    COST_SMOOTHNESS,
    COST_TRANSITION,
    DESIRED_CLEARANCE_M,
    GROUND_BIN_M,
    GROUND_MIN_SAMPLES,
    GROUND_PERCENTILE,
    GROUND_SAMPLE_STRIDE,
    KEEP_RIGHT_MARGIN_M,
    KEEP_RIGHT_SCALE_M,
    MAX_CORRIDOR_HALF_WIDTH_M,
    MIN_TURN_RADIUS_M,
    OBSTACLE_CELL_M,
    OBSTACLE_MAX_HEIGHT_M,
    OBSTACLE_MIN_HEIGHT_M,
    OBSTACLE_MIN_SUPPORT,
    PLANNER_ARC_SAMPLES,
    PLANNER_HORIZON_M,
    PLANNER_LOOKAHEAD_M,
    PLANNER_MAX_OBSTACLE_POINTS,
    REQUIRED_FREE_DISTANCE_M,
    ROAD_BONUS_SAMPLES,
    ROUTE_MATCH_SAMPLES,
    ROUTE_XTRACK_SCALE_M,
    SLOPE_ALLOWANCE_PER_M,
    SLOPE_ALLOWANCE_START_M,
    TRANSITION_DISTANCES_M,
)
from .models import ArcPlan, RoadGrid, RoutePath, VehicleGeometry

MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M
# Quadratically spaced (|k| = K_MAX * u^2): dense around straight ahead, coarse
# only toward full lock where collision dominates anyway. The spacing IS the
# lateral resolution: endpoint offset goes as k * L^2 / 2, so at a 30 m
# lookahead one step of a UNIFORM 41-arc fan is 3.75 m of offset -- "correct
# by a metre" was literally not a candidate. The planner then reached for a
# deferred family every tick as a finer-offset workaround ("hold course, bend
# later"), the immediate command never corrected, and the car wove kerb to
# kerb. The first quadratic step is ~K_MAX/400: 0.19 m of offset at 30 m.
_FAN_POSITIONS = np.linspace(-1.0, 1.0, PLANNER_ARC_SAMPLES)
CANDIDATE_CURVATURES = MAX_CURVATURE * _FAN_POSITIONS * np.abs(_FAN_POSITIONS)
# Below this a candidate is straight for every practical purpose. Substituting
# it keeps the arc maths free of a divide-by-zero special case: R = 10 km bends
# 7 mm over 20 m, far under the point spacing.
_MIN_CURVATURE = 1e-4
_EMPTY_BEV = np.empty((0, 2), dtype=np.float32)
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ObstacleBand:
    """
    One consumer's definition of "solid", for `geometric_obstacle_sets`.

    The bands differ because the consumers ask different questions -- the
    planner steers around a 0.12 m kerb inside 35 m, AEB brakes only for what
    would be a crash and has to see far enough to stop from motorway speed --
    but the ground underneath them is one measurement, so they are built
    together.

    The two shape tests default OFF, so a band that names only a floor and a
    horizon behaves exactly as it did before they existed. That is the planner's
    band: it *should* steer around bushes and kerbs. Only the full-authority
    brake gets pickier.
    """

    min_height_m: float
    horizon_m: float
    min_vertical_extent_m: float = 0.0
    """How tall the thing standing in a cell must be. 0 disables the test."""
    porosity: bool = False
    """Whether see-through candidates are vetoed. See `_porous`."""


def _as_bev(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return array.reshape((-1, 2))


def despeckle(
    points: np.ndarray,
    cell_m: float = OBSTACLE_CELL_M,
    min_support: int = OBSTACLE_MIN_SUPPORT,
) -> np.ndarray:
    """
    Drop returns with no neighbours, which the arc scan cannot tolerate.

    ``_scan_arcs`` takes the nearest blocking point per arc, so one spurious
    return ends that arc on its own -- measured, a single stray point at 10 m
    took a clear road's free distance from 33 m to 10 m, which is a full brake
    application. A point survives only if its 3x3 cell neighbourhood holds at
    least ``min_support`` returns including itself.

    Counting is a ``bincount`` over flattened cell indices plus a nine-term box
    sum, so it stays vectorized and runs on the full pre-decimation cloud.
    """
    array = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if len(array) < min_support or min_support <= 1:
        return array if min_support <= 1 else array[:0]

    ix = np.floor(array[:, 0] / cell_m).astype(np.intp)
    iy = np.floor(array[:, 1] / cell_m).astype(np.intp)
    ix -= ix.min()
    iy -= iy.min()
    width = int(ix.max()) + 1
    height = int(iy.max()) + 1
    # One cell of halo on every side, so the box sum is pure slicing.
    stride = width + 2
    counts = np.bincount(
        (iy + 1) * stride + (ix + 1), minlength=(height + 2) * stride
    ).reshape((height + 2, stride))
    box = (
        counts[0:height, 0:width]
        + counts[0:height, 1 : width + 1]
        + counts[0:height, 2 : width + 2]
        + counts[1 : height + 1, 0:width]
        + counts[1 : height + 1, 1 : width + 1]
        + counts[1 : height + 1, 2 : width + 2]
        + counts[2 : height + 2, 0:width]
        + counts[2 : height + 2, 1 : width + 1]
        + counts[2 : height + 2, 2 : width + 2]
    )
    return array[box[iy, ix] >= min_support]


def ground_rise(ranges: np.ndarray, above_ego: np.ndarray) -> np.ndarray:
    """
    Local ground height above the ego's own plane, per point, from the returns.

    The ego's ground plane is sampled under the car, so extending it flat makes
    terrain drift away from it with distance. The cone in
    ``SLOPE_ALLOWANCE_PER_M`` covered that by simply raising the obstacle floor
    with range, which also raised it past every kerb: nothing under 0.27 m was
    an obstacle at 20 m. Measuring the ground instead keeps the floor tight.

    Estimated as a low percentile of height per range ring, then interpolated
    between ring centres -- interpolating rather than stepping cancels the
    first-order within-ring grade error, leaving only terrain curvature. Rings
    too sparse to trust are dropped and interpolated across; ``np.interp`` holds
    the end values, which is the conservative choice beyond the last good ring.

    The caller is responsible for clamping the result. Returned relative to the
    ego plane, in the same units as ``above_ego``.
    """
    flat = np.zeros(len(ranges), dtype=np.float64)
    if len(ranges) < GROUND_MIN_SAMPLES:
        return flat
    sample_r = ranges[::GROUND_SAMPLE_STRIDE]
    sample_h = above_ego[::GROUND_SAMPLE_STRIDE]
    bin_count = max(1, int(np.ceil(float(sample_r.max()) / GROUND_BIN_M)))
    bins = np.clip(
        (sample_r / GROUND_BIN_M).astype(np.intp), 0, bin_count - 1
    )
    # Sorted by ring, then by height inside each ring, so the percentile of a
    # ring is one index into the sorted heights.
    order = np.lexsort((sample_h, bins))
    sorted_heights = sample_h[order]
    counts = np.bincount(bins, minlength=bin_count)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    valid = counts >= GROUND_MIN_SAMPLES
    if not valid.any():
        return flat
    picks = starts[valid] + (
        counts[valid] * (GROUND_PERCENTILE / 100.0)
    ).astype(np.intp)
    centres = (np.flatnonzero(valid) + 0.5) * GROUND_BIN_M
    return np.interp(ranges, centres, sorted_heights[picks].astype(np.float64))


def geometric_obstacles(
    bev: np.ndarray,
    heights: np.ndarray,
    ground_z_vehicle: float,
    min_height_m: float = OBSTACLE_MIN_HEIGHT_M,
    horizon_m: float = PLANNER_HORIZON_M,
) -> np.ndarray:
    """
    Select the returns treated as solid, on geometry alone.

    An obstacle is a return standing between ``min_height_m`` and
    ``OBSTACLE_MAX_HEIGHT_M`` above the LOCAL ground, inside the planning
    horizon, with at least one neighbour. Semantics are deliberately not
    consulted: flat grass and car parks read as drivable, and a kerb face reads
    as a wall.

    Both the floor and the horizon are parameters because they are questions
    about the consumer, not about the cloud: the planner steers around a 0.12 m
    kerb inside 35 m, while `aeb` only brakes for what would be a crash and has
    to see far enough to stop from 100 km/h. Callers that want more than one
    band should use `geometric_obstacle_sets` rather than calling this twice.
    """
    return geometric_obstacle_sets(
        bev,
        heights,
        ground_z_vehicle,
        (ObstacleBand(min_height_m, horizon_m),),
    )[0]


def _group_starts(sorted_keys: np.ndarray) -> np.ndarray:
    """Index of the first element of each run of equal keys.

    Duplicated from `world_scene` rather than shared: that module sits far above
    this one (world_scene -> aeb -> planner), so importing it here would close a
    cycle, and this is three lines.
    """
    if not len(sorted_keys):
        return np.empty(0, dtype=np.intp)
    breaks = np.empty(len(sorted_keys), dtype=bool)
    breaks[0] = True
    breaks[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(breaks)


# A cell key is two 21-bit fields; the height packed under it in `_cell_profile`
# takes a third. 21 bits of millimetres covers +/- 1000 m of height, and 21 bits
# of cell index covers +/- 400 km at this cell size.
_CELL_FIELD = 1 << 21
_HEIGHT_FIELD = 1 << 21
_HEIGHT_BIAS_M = 1000.0

# How many evidence bins one porosity candidate may consult. A cell's wedge is
# cell_m/r, so this only ever binds in the near field -- 16 bins of 0.5 deg is
# the whole wedge beyond 2.9 m at OBSTACLE_CELL_M. Under-counting evidence means
# fewer vetoes, so the bound errs toward braking.
_POROSITY_MAX_BINS = 16


def _cell_keys(points: np.ndarray, cell_m: float) -> np.ndarray:
    """Pack XY cell indices into one int64 so grouping stays a 1-D sort."""
    ix = np.floor(points[:, 0] / cell_m).astype(np.int64)
    iy = np.floor(points[:, 1] / cell_m).astype(np.int64)
    return (ix + (_CELL_FIELD >> 1)) * _CELL_FIELD + (iy + (_CELL_FIELD >> 1))


def _cell_profile(
    points: np.ndarray, above_ego: np.ndarray, cell_m: float = OBSTACLE_CELL_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per point, the base and the height of whatever occupies its XY cell.

    The answer to "is this an obstacle or a hill", and the reason it works where
    a height floor cannot: a 0.4 m cell on a 20% slope holds 0.08 m of spread
    while a wall holds metres. Being a difference it is equally immune to brake
    dive and to suspension heave, so it fixes the near field the slope cone
    never reached -- which is the entire operating range of the rear system.

    The BASE matters as much as the spread, because it is the local ground
    reference the whole point of this is to get right. Measuring height from it
    makes both the floor and the ceiling grade-proof at once: on a 20% climb the
    surface at 30 m is 6 m above the ego plane, so a wall standing on it starts
    *above* OBSTACLE_MAX_HEIGHT_M and was discarded as overhead -- AEB went
    blind to real walls on exactly the hills it used to brake at nothing for.

    Computed over EVERY return in the cell, not just those above the obstacle
    floor. A 0.35 m rock puts 0.05 m above a 0.30 m floor; it is still 0.35 m
    tall, and measuring the survivors alone would delete every short solid thing
    there is.

    Returned in CELL space -- ``(index per point, base, extent, centre)`` -- not
    broadcast back over the cloud. Everything downstream is a property of the
    object rather than of the individual return, and a cloud has one or two
    orders of magnitude more returns than occupied cells.
    """
    if len(points) == 0:
        empty = np.empty(0)
        return np.empty(0, dtype=np.intp), empty, empty, np.empty((0, 2))
    keys = _cell_keys(points, cell_m)
    # SORT, not argsort. Packing the height into the low bits of the same
    # integer key makes the group minimum and maximum the first and last entry
    # of each run, so the permutation itself is never needed -- and numpy sorts
    # integers by radix. Measured on a 110k cloud, argsort was 5.9 ms of a
    # 10.6 ms profile; this is a plain sort plus one searchsorted to map back.
    #
    # Heights are quantised to the millimetre and biased positive, which is far
    # finer than the returns resolve and comfortably inside the 21 bits the
    # packing leaves. Anything outside +/-1000 m is clipped rather than allowed
    # to wrap into the cell field.
    height_key = np.clip(
        ((above_ego + _HEIGHT_BIAS_M) * 1000.0).astype(np.int64),
        0,
        _HEIGHT_FIELD - 1,
    )
    combined = np.sort(keys * _HEIGHT_FIELD + height_key)
    cells = combined // _HEIGHT_FIELD
    starts = _group_starts(cells)
    unique_cells = cells[starts]
    ends = np.append(starts[1:], len(combined)) - 1
    lows = (combined[starts] % _HEIGHT_FIELD) / 1000.0 - _HEIGHT_BIAS_M
    highs = (combined[ends] % _HEIGHT_FIELD) / 1000.0 - _HEIGHT_BIAS_M
    index = np.searchsorted(unique_cells, keys)
    counts = np.diff(np.append(starts, len(combined)))
    centre = np.column_stack(
        (
            np.bincount(index, weights=points[:, 0], minlength=len(starts)),
            np.bincount(index, weights=points[:, 1], minlength=len(starts)),
        )
    ) / counts[:, None]
    return index, lows, highs - lows, centre


def _porous(
    centres: np.ndarray,
    heights: np.ndarray,
    ground_bev: np.ndarray,
    sensor_height_m: float,
    cell_m: float = OBSTACLE_CELL_M,
) -> np.ndarray:
    """
    Which candidates have been SEEN THROUGH, and so are not solid.

    A bush passes rays between its leaves and the ground behind it comes back; a
    wall stops them. The window that separates the two is not a guess -- an
    object of height ``a`` at range ``r``, seen from a sensor at height ``h``,
    hides the ground behind it for exactly ``r * a / (h - a)``. Ground returns
    inside that shadow mean the rays got through.

    ``a >= h`` makes the shadow infinite and the window empty, so anything as
    tall as the sensor can NEVER be vetoed here. That is the safety property,
    and it falls out of the geometry rather than being imposed on top: no
    parallax between the five mount positions can talk AEB out of braking for a
    wall, a car or a person. For short things the failure direction is "no
    evidence, so treat it as solid".

    Answered per CANDIDATE CELL rather than per return: being see-through is a
    property of the object, and a cloud holds far more returns than cells.
    ``ground_bev`` is the evidence -- every return the height floor rejected,
    which is a geometric test rather than a semantic one and so survives an
    unannotated map.

    **Evidence only counts if the candidate would have BLOCKED it**, which is
    what makes the azimuth window the candidate's OWN angular width rather than
    a fixed angle. A ray passing beside an object was never stopped by it, so it
    says nothing about whether the object is porous. A fixed bin is a fixed
    angle against an object of fixed width, so it outgrows the object with
    range: at a 2 deg bin a 1.8 m car stops covering one at all past ~52 m, and
    stops reliably covering one past ~26 m, whereupon the ground BESIDE it vetoes
    it. Measured on a ring-sampled scene, that made AEB blind to a stopped car
    through the whole 30-60 m band -- which is exactly where it must fire at
    60-100 km/h, so the brake came on late.
    """
    if not len(centres) or not len(ground_bev) or sensor_height_m <= 0.0:
        return np.zeros(len(centres), dtype=bool)

    # One sort over a composite key, so a window lookup is two vectorised
    # searchsorted calls rather than a scan. The key is an INTEGER -- range
    # quantised to the centimetre, far finer than anything here resolves --
    # because numpy sorts integers by radix and floats by comparison, and the
    # evidence side of this is most of the cloud.
    ground_ranges = np.hypot(ground_bev[:, 0], ground_bev[:, 1])
    ranges = np.hypot(centres[:, 0], centres[:, 1])
    span = int(max(ground_ranges.max(), ranges.max()) * 100.0) + 200

    bin_rad = np.radians(AEB_POROSITY_AZIMUTH_DEG)
    evidence = (
        np.floor(
            np.arctan2(ground_bev[:, 0], ground_bev[:, 1]) / bin_rad
        ).astype(np.int64)
        * span
        + (ground_ranges * 100.0).astype(np.int64)
    )
    evidence.sort(kind="stable")

    # Anything at or above the sensor shadows the ground for ever, so its window
    # is empty and it can never be vetoed. Written as an explicit negative
    # shadow rather than left to the arithmetic, so the guarantee is structural
    # -- it must not depend on the cloud happening to contain no returns beyond
    # a wall, which parallax between five mount positions could break.
    shadow = np.where(
        heights >= sensor_height_m,
        -1.0,
        ranges * heights / np.maximum(sensor_height_m - heights, 1e-6),
    )

    # The bins lying ENTIRELY inside this cell's own wedge. Partly-covered bins
    # are discarded rather than shared, because a bin that pokes out past the
    # cell holds rays the cell never stood in front of. None fully inside means
    # no evidence at all, which reads as solid -- the conservative direction,
    # and the one a candidate narrower than the grid should get.
    bearing = np.arctan2(centres[:, 0], centres[:, 1])
    half = 0.5 * cell_m / np.maximum(ranges, 1e-6)
    first = np.ceil((bearing - half) / bin_rad).astype(np.int64)
    covered = np.maximum(
        np.floor((bearing + half) / bin_rad).astype(np.int64) - first, 0
    )

    # Offsets within a bin, and `far` is clamped to the END of that bin. The key
    # packs (bin, range) into one integer, so an unclamped window runs straight
    # off the end of its own bin and into the next one -- where it counts the
    # ground in FRONT of the object as evidence of seeing through it. A car's
    # shadow is over three times its range, so this overflowed for every car
    # past ~15 m and vetoed the lot.
    range_cm = (ranges * 100.0).astype(np.int64)
    near_offset = range_cm + int(AEB_POROSITY_GAP_M * 100.0)
    far_offset = np.minimum(
        range_cm + (shadow * 100.0).astype(np.int64), span - 1
    )

    hits = np.zeros(len(centres), dtype=np.int64)
    for step in range(min(int(covered.max()), _POROSITY_MAX_BINS)):
        active = covered > step
        base = (first[active] + step) * span
        hits[active] += np.searchsorted(
            evidence, base + far_offset[active], side="right"
        ) - np.searchsorted(evidence, base + near_offset[active], side="left")
    return hits >= AEB_POROSITY_MIN_HITS


def geometric_obstacle_sets(
    bev: np.ndarray,
    heights: np.ndarray,
    ground_z_vehicle: float,
    bands: Sequence[ObstacleBand],
    sensor_height_m: float = 0.0,
) -> list[np.ndarray]:
    """
    One obstacle set per `ObstacleBand`, sharing a single local ground estimate.

    The bands differ per consumer but the ground underneath them is one
    measurement, and it is the expensive part of this function: measured on a
    60k-point worst case, ``ground_rise`` plus the ranges cost 2.6 ms of 3.3,
    against 0.7 ms for the mask and despeckle that a second band adds. Calling
    the whole thing twice put the both-features-on tick at 7.8 ms against 4.0.

    ``sensor_height_m`` is the ROOF mount height above the vehicle's ground
    plane, and only the porosity test uses it -- that is the ``h`` in the shadow
    length ``r*a/(h - a)``. Left at 0 no candidate is ever vetoed, which is the
    conservative direction and what a band without porosity gets anyway.

    Returned in the order the bands were given.
    """
    bands = list(bands)
    points = np.asarray(bev, dtype=np.float32).reshape((-1, 2))
    if len(points) == 0:
        return [_EMPTY_BEV for _ in bands]
    above = np.asarray(heights, dtype=np.float32).reshape(-1) - float(
        ground_z_vehicle
    )
    ranges = np.hypot(points[:, 0], points[:, 1])
    # Everything below is O(cloud), and the cloud now reaches LIDAR_RANGE_M so
    # the long-range front unit can feed AEB -- most of which no band wants.
    # Culling to the furthest band FIRST is what keeps that reach affordable:
    # measured on a 110k cloud, the full-radius version cost 24.5 ms of the
    # 40 ms tick against 5.5 ms once the ground estimate only sees the points
    # some band will actually use.
    furthest = max(band.horizon_m for band in bands)
    if furthest < float(ranges.max()):
        inside = ranges <= furthest
        points = points[inside]
        above = above[inside]
        ranges = ranges[inside]
        if len(points) == 0:
            return [_EMPTY_BEV for _ in bands]
    # The cone bounds the measured rise rather than replacing it. Clamped below
    # at zero so a ditch cannot drag the floor under the road surface, and
    # above at the old cone so terrain can never read as more drivable than it
    # used to. On flat ground the rise is ~0 and the floor stays at
    # OBSTACLE_MIN_HEIGHT_M all the way out, which is what keeps kerbs visible.
    cone = SLOPE_ALLOWANCE_PER_M * np.maximum(
        ranges - SLOPE_ALLOWANCE_START_M, 0.0
    )
    rise = np.clip(ground_rise(ranges, above), 0.0, cone)
    below_ceiling = above <= OBSTACLE_MAX_HEIGHT_M + rise

    # Shared by both shape tests, and computed at most once. The cell BASE is
    # the local ground the shape bands measure everything from, which is what
    # makes them independent of the slope clamp in both directions -- see
    # `_cell_profile`.
    cell = None
    if any(band.min_vertical_extent_m > 0.0 for band in bands):
        cell = _cell_profile(points, above)

    sets: list[np.ndarray] = []
    for band in bands:
        if band.min_vertical_extent_m > 0.0 and cell is not None:
            index, cell_base, cell_extent, cell_centre = cell
            # Heights measured from the CELL's own base rather than from the
            # clamped ground estimate, so a gradient moves the floor and the
            # ceiling together and neither can be fooled by one.
            local = above - cell_base[index]
            tall_enough = cell_extent >= band.min_vertical_extent_m
            selected = (
                (ranges <= band.horizon_m)
                & tall_enough[index]
                & (local >= band.min_height_m)
                & (local <= OBSTACLE_MAX_HEIGHT_M)
            )
            # Both shape tests only ever REMOVE candidates, so neither can
            # invent an obstacle. That is what makes them safe to add under a
            # full-authority brake, and why the live checklist only has to
            # re-prove that it still fires.
            if band.porosity and selected.any():
                # Decided per CELL, because being see-through is a property of
                # the object. The shadow is likewise cast by the whole object,
                # so its length comes from the cell's extent and not from an
                # individual return's height up it -- which would both shorten
                # every window and make a wall's lower half testable when the
                # wall as a whole must never be.
                #
                # Only cells SHORTER than the sensor are even asked: taller ones
                # shadow the ground for ever and are immune by construction, so
                # filtering them here changes no verdict and skips building the
                # evidence set entirely in the common case where everything
                # AEB can see is a wall, a car or a person.
                suspect = np.unique(index[selected])
                suspect = suspect[cell_extent[suspect] < sensor_height_m]
                if len(suspect):
                    seen_through = _porous(
                        cell_centre[suspect],
                        cell_extent[suspect],
                        points[local < band.min_height_m],
                        sensor_height_m,
                    )
                    if seen_through.any():
                        vetoed = np.zeros(len(cell_extent), dtype=bool)
                        vetoed[suspect[seen_through]] = True
                        selected &= ~vetoed[index]
        else:
            selected = (
                (ranges <= band.horizon_m)
                & below_ceiling
                & (above >= band.min_height_m + rise)
            )
        # Despeckled before decimation: decimating first would thin real
        # surfaces toward the support threshold and make the filter's verdict
        # depend on the cloud size.
        obstacles = despeckle(points[selected])
        if len(obstacles) > PLANNER_MAX_OBSTACLE_POINTS:
            # Evenly spaced indices, matching the decimation the display uses:
            # an integer stride would halve the output the moment the count
            # exceeds the cap by one.
            index = np.linspace(
                0, len(obstacles) - 1, PLANNER_MAX_OBSTACLE_POINTS
            ).astype(np.intp)
            obstacles = obstacles[index]
        sets.append(obstacles)
    return sets


def corridor_return_profile(
    bev: np.ndarray,
    heights: np.ndarray,
    ground_z_vehicle: float,
    curvature: float,
    half_width_m: float,
    threat_m: float,
    window_m: float = 1.0,
) -> dict[str, float]:
    """
    Describe what is standing in the corridor around a threat. Diagnostics only.

    Nothing acts on this. It exists because "AEB braked and there was nothing
    there" cannot be answered from the trigger's own numbers: the corridor scan
    reduces a surface to one distance, and a distance cannot say whether that
    surface was a wall, a kerb face, or the road itself arriving in the height
    band. The VERTICAL EXTENT can, and it is the one measurement that is
    invariant to both grade and ride height:

        spread < ~0.10 m   a surface lying along the ground -- flat road under a
                           brake dive, or a hillside crossing the floor because
                           `ground_rise` is clamped into the slope cone
        ~0.10-0.20 m       a kerb or low lip: real, but not a crash
        tens of cm         genuinely solid, and the brake was right

    Read the spread against ``range_span_m``, because that ratio is the local
    SLOPE and it separates the first case from the others outright: ground rises
    a few centimetres per metre of range however steep the hill, while a wall
    puts metres of height into a range span of nearly nothing.

    Reports the local ground estimate and the cone that bounds it at the same
    range, because "the ground under it" is the other half of the answer.

    Costs a full `ground_rise` pass over the cloud, so call it on the ARMED to
    BRAKING transition and nowhere else.
    """
    points = np.asarray(bev, dtype=np.float32).reshape((-1, 2))
    above = np.asarray(heights, dtype=np.float32).reshape(-1) - float(
        ground_z_vehicle
    )
    empty = {
        "count": 0.0,
        "height_min_m": 0.0,
        "height_median_m": 0.0,
        "height_max_m": 0.0,
        "spread_m": 0.0,
        "range_span_m": 0.0,
        "ground_rise_m": 0.0,
        "cone_bound_m": 0.0,
    }
    if len(points) == 0 or len(points) != len(above):
        return empty

    ranges = np.hypot(points[:, 0], points[:, 1])
    near_threat = np.abs(ranges - float(threat_m)) <= float(window_m)
    if not near_threat.any():
        return empty

    # The same corridor the scan used, so this describes what actually blocked
    # it rather than a differently-shaped guess at it.
    k = float(curvature)
    safe = k if abs(k) >= _MIN_CURVATURE else _MIN_CURVATURE
    radius = abs(1.0 / safe)
    offset_x = points[:, 0] + 1.0 / safe
    inside = (
        np.abs(np.hypot(offset_x, points[:, 1]) - radius) < float(half_width_m)
    ) & near_threat & (points[:, 1] > 0.0)
    if not inside.any():
        return empty

    selected = above[inside]
    spanned = ranges[inside]
    rise = ground_rise(ranges, above)[inside]
    cone = SLOPE_ALLOWANCE_PER_M * np.maximum(
        spanned - SLOPE_ALLOWANCE_START_M, 0.0
    )
    return {
        "count": float(len(selected)),
        "height_min_m": float(np.min(selected)),
        "height_median_m": float(np.median(selected)),
        "height_max_m": float(np.max(selected)),
        "spread_m": float(np.max(selected) - np.min(selected)),
        "range_span_m": float(np.max(spanned) - np.min(spanned)),
        "ground_rise_m": float(np.median(rise)),
        "cone_bound_m": float(np.median(cone)),
    }


def arc_polyline(
    curvature: float, length_m: float, samples: int = 24
) -> np.ndarray:
    """Sample an arc of the given curvature as a BEV ``(right, forward)`` path."""
    count = max(2, int(samples))
    progress = np.linspace(0.0, float(length_m), count)
    k = float(curvature)
    if abs(k) < _MIN_CURVATURE:
        return np.column_stack((np.zeros_like(progress), progress)).astype(
            np.float32
        )
    radius = 1.0 / abs(k)
    centre_x = -1.0 / k
    start_angle = 0.0 if k > 0.0 else np.pi
    angle = start_angle + np.sign(k) * (progress / radius)
    return np.column_stack(
        (centre_x + radius * np.cos(angle), radius * np.sin(angle))
    ).astype(np.float32)


def _arc_endpoint(curvature: float, length_m: float) -> tuple[float, float, float]:
    """``(x, y, heading)`` after driving ``length_m`` along a single arc."""
    k = float(curvature)
    if abs(k) < _MIN_CURVATURE:
        return 0.0, float(length_m), 0.0
    return (
        -(1.0 - float(np.cos(k * length_m))) / k,
        float(np.sin(k * length_m)) / k,
        k * float(length_m),
    )


def path_polyline(
    curvature0: float,
    transition_m: float,
    curvature1: float,
    length_m: float,
    samples: int = 32,
) -> np.ndarray:
    """
    Sample a two-segment path: ``curvature0`` held for ``transition_m``, then
    ``curvature1`` for the remainder. Collapses to a single arc when the
    transition is empty, so a plain `ArcPlan` draws exactly as before.
    """
    length = float(length_m)
    d1 = float(min(max(transition_m, 0.0), length))
    if d1 <= 1e-6 or abs(float(curvature0) - float(curvature1)) < 1e-9:
        return arc_polyline(curvature1, length, samples)
    count = max(4, int(samples))
    n_a = min(count - 2, max(2, int(round(count * d1 / length))))
    part_a = arc_polyline(curvature0, d1, n_a)
    x1, y1, theta1 = _arc_endpoint(curvature0, d1)
    part_b = arc_polyline(curvature1, length - d1, max(2, count - n_a))
    # A heading change of theta (positive = left) is the standard CCW rotation
    # in (right, forward) coordinates.
    cos_t, sin_t = float(np.cos(theta1)), float(np.sin(theta1))
    rotated = np.column_stack(
        (
            cos_t * part_b[:, 0] - sin_t * part_b[:, 1],
            sin_t * part_b[:, 0] + cos_t * part_b[:, 1],
        )
    )
    return np.concatenate((part_a, (rotated + (x1, y1))[1:])).astype(np.float32)


def corridor_edges(
    obstacles: np.ndarray, lookahead_m: float
) -> tuple[float | None, float | None]:
    """
    Measure the nearest obstacle either side of the centreline at the lookahead.

    Returns ``(left_edge, right_edge)`` as lateral offsets, each ``None`` when
    nothing bounds that side inside ``MAX_CORRIDOR_HALF_WIDTH_M``.
    """
    points = _as_bev(obstacles)
    if len(points) == 0:
        return None, None
    half_band = CORRIDOR_BAND_M / 2.0
    in_band = (points[:, 1] >= lookahead_m - half_band) & (
        points[:, 1] <= lookahead_m + half_band
    )
    lateral = points[in_band, 0]
    if len(lateral) == 0:
        return None, None
    left = lateral[(lateral < 0.0) & (lateral >= -MAX_CORRIDOR_HALF_WIDTH_M)]
    right = lateral[(lateral > 0.0) & (lateral <= MAX_CORRIDOR_HALF_WIDTH_M)]
    return (
        float(left.max()) if len(left) else None,
        float(right.min()) if len(right) else None,
    )


def corridor_blocking_distances(
    obstacles: np.ndarray,
    curvature: float,
    half_width_m: float,
    horizon_m: float = PLANNER_HORIZON_M,
) -> np.ndarray:
    """
    Every return inside ONE arc's corridor, as sorted arc-length distances.

    The single-arc counterpart to ``_scan_arcs``, which stays the matrix scanner
    for the candidate fan. The two differ in what they hand back rather than in
    how they measure: ``_scan_arcs`` reduces straight to a per-arc minimum,
    which is all the planner's cost function needs, while `aeb` needs the *Nth*
    nearest -- a full brake application must be backed by a surface, and taking
    the nearest single return would fire on any speck that survived despeckling.

    Distances are along the arc, in the direction of travel; returns behind the
    car land near ``2 * pi * R`` and are dropped against the horizon on their
    own, exactly as in the matrix scan.
    """
    points = _as_bev(obstacles)
    if len(points) == 0:
        return np.empty(0)
    k = float(curvature)
    safe = k if abs(k) >= _MIN_CURVATURE else _MIN_CURVATURE
    radius = abs(1.0 / safe)
    offset_x = points[:, 0] + 1.0 / safe
    blocking = (
        np.abs(np.hypot(offset_x, points[:, 1]) - radius) < float(half_width_m)
    )
    if not blocking.any():
        return np.empty(0)
    start_angle = 0.0 if safe > 0.0 else np.pi
    swept = np.mod(
        np.sign(safe)
        * (np.arctan2(points[blocking, 1], offset_x[blocking]) - start_angle),
        _TWO_PI,
    )
    progress = radius * swept
    return np.sort(progress[(progress > 0.0) & (progress <= float(horizon_m))])


def rear_free_distance(
    obstacles: np.ndarray,
    geometry: VehicleGeometry,
    horizon_m: float = PLANNER_HORIZON_M,
) -> float:
    """
    Clear distance straight behind the car, for the reverse recovery.

    Reversing is always straight, so this is a simple rearward corridor test
    rather than a second arc scan.
    """
    points = _as_bev(obstacles)
    if len(points) == 0:
        return float(horizon_m)
    half_width = geometry.width_m / 2.0 + CLEARANCE_MARGIN_M
    behind = (points[:, 1] < 0.0) & (np.abs(points[:, 0]) <= half_width)
    if not behind.any():
        return float(horizon_m)
    return float(min(np.abs(points[behind, 1]).min(), horizon_m))


def _lateral_offsets(
    curvatures: np.ndarray, safe: np.ndarray, distance_m: float
) -> np.ndarray:
    """Lateral position each arc reaches after ``distance_m`` of arc length."""
    radius = np.abs(1.0 / safe)
    centre_x = -1.0 / safe
    start_angle = np.where(safe > 0.0, 0.0, np.pi)
    angle = start_angle + np.sign(safe) * (distance_m / radius)
    return centre_x + radius * np.cos(angle)


def _scan_arcs(
    points: np.ndarray,
    safe_curvatures: np.ndarray,
    half_width: float,
    horizon_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Free distance per arc plus the ``(points x arcs)`` deviation matrix.

    The swept-angle transcendentals run only on the blocking pairs
    (``deviation < half_width``) -- a few percent of the matrix -- which is
    what keeps five family scans inside the display tick. Free distance stays
    exact: the swept angle is measured in the direction of travel, so points
    behind land near ``2*pi*R`` and drop out against the horizon on their own.
    Points must be non-empty.
    """
    radius = np.abs(1.0 / safe_curvatures)
    centre_x = -1.0 / safe_curvatures
    x = points[:, 0:1]
    y = points[:, 1:2]
    offset_x = x - centre_x[None, :]
    deviation = np.abs(np.sqrt(offset_x * offset_x + y * y) - radius[None, :])

    free = np.full(len(safe_curvatures), float(horizon_m))
    hit_point, hit_arc = np.nonzero(deviation < half_width)
    if len(hit_point):
        start_angle = np.where(safe_curvatures[hit_arc] > 0.0, 0.0, np.pi)
        swept = np.mod(
            np.sign(safe_curvatures[hit_arc])
            * (
                np.arctan2(y[hit_point, 0], offset_x[hit_point, hit_arc])
                - start_angle
            ),
            _TWO_PI,
        )
        progress = radius[hit_arc] * swept
        forward = progress > 0.0
        np.minimum.at(free, hit_arc[forward], progress[forward])
    return free, deviation


def _windowed_clearance(
    deviation: np.ndarray,
    range_from_origin: np.ndarray,
    forward: np.ndarray,
    radius: np.ndarray,
    window: np.ndarray,
) -> np.ndarray:
    """
    Minimum deviation over the driven window, per arc.

    Uses the exact chord equivalence ``progress <= w  <=>  y > 0 and
    |P| <= 2R sin(w / 2R)`` on the front half-circle, so no transcendental
    runs over the full matrix. For points off the arc the boundary blurs by
    up to their deviation, which only shades a cost term; a negative window
    (composite window shorter than the transition) selects nothing and falls
    through to "unbounded" at the call site.
    """
    capped = np.minimum(window, np.pi * radius)
    chord = 2.0 * radius * np.sin(capped / (2.0 * radius))
    inside = (forward[:, None] > 0.0) & (
        range_from_origin[:, None] <= chord[None, :]
    )
    return np.where(inside, deviation, np.inf).min(axis=0)


def plan_arc(
    obstacles: np.ndarray,
    geometry: VehicleGeometry,
    nav_heading_rad: float | None = None,
    previous_curvature: float = 0.0,
    lookahead_m: float = PLANNER_LOOKAHEAD_M,
    horizon_m: float = PLANNER_HORIZON_M,
    *,
    route: RoutePath | None = None,
    keep_right: bool = True,
    road_grid: RoadGrid | None = None,
    required_free_m: float | None = None,
    smoothness_weight: float | None = None,
) -> ArcPlan:
    """
    Score the two-segment candidate fan against the obstacle cloud.

    Each candidate holds ``previous_curvature`` -- the curvature the car is
    actually driving right now -- for one of ``TRANSITION_DISTANCES_M``, then
    bends to a target curvature. The zero-transition family is the classic
    immediate fan (and is what the widget displays); the deferred families are
    what let the planner say "keep going, the turn comes later", which is what
    corner-entry braking needs to know.

    ``route`` is the reference path built by `route_model` (seen here only as
    its `models` type, which is what keeps this module free of any dependency
    on navigation). When present it REPLACES both the nav-heading and
    keep-right terms with a tangent and a cross-track term; when absent the
    legacy terms run byte-identically. Either way guidance never becomes
    authority: every route term is cost among several, so a blocked arc is
    never chosen just because the route asked for it -- LiDAR always wins.

    ``nav_heading_rad`` is the legacy bearing hint, used only when no
    reference path could be built. ``keep_right``, ``required_free_m`` and
    ``smoothness_weight`` exist for the reverse planner: reversing wants
    clearance and free distance only, scores free room against the
    manoeuvre's own needs rather than the 40 km/h braking envelope, and
    treats "steer least" as a tie-break rather than passenger comfort.
    """
    points = _as_bev(obstacles)
    curvatures = CANDIDATE_CURVATURES
    safe = np.where(
        np.abs(curvatures) < _MIN_CURVATURE, _MIN_CURVATURE, curvatures
    )
    collision_half_width = geometry.width_m / 2.0 + CLEARANCE_MARGIN_M
    lookahead = float(lookahead_m)
    horizon = float(horizon_m)
    k0 = float(np.clip(previous_curvature, -MAX_CURVATURE, MAX_CURVATURE))
    safe_k0 = k0 if abs(k0) >= _MIN_CURVATURE else _MIN_CURVATURE

    families = TRANSITION_DISTANCES_M
    d1_array = np.asarray(families, dtype=np.float64)
    arc_count = len(curvatures)
    free_all = np.full((len(families), arc_count), horizon)
    clear_all = np.full(
        (len(families), arc_count), float(MAX_CORRIDOR_HALF_WIDTH_M)
    )
    # Segment-A endpoints per family, reused for the keep-right offsets.
    ends = [_arc_endpoint(k0, d1) for d1 in families]

    if len(points):
        scan_points = points.astype(np.float32)
        origin_range = np.sqrt(
            scan_points[:, 0] ** 2 + scan_points[:, 1] ** 2
        )
        radii = np.abs(1.0 / safe)
        radius_k0 = np.asarray([abs(1.0 / safe_k0)])

        # Segment A is one arc at the current curvature, scanned once.
        free_a, dev_a = _scan_arcs(
            scan_points, np.asarray([safe_k0]), collision_half_width, horizon
        )
        free_a = float(free_a[0])

        for index, d1 in enumerate(families):
            if d1 <= 0.0:
                free, deviation = _scan_arcs(
                    scan_points, safe, collision_half_width, horizon
                )
                # Clearance is judged only over the stretch that will actually
                # be driven before the next re-plan. Scoring it to the full
                # free distance would punish every arc for the pinch point
                # that ends it.
                window = np.minimum(free, lookahead)
                clearance = _windowed_clearance(
                    deviation, origin_range, scan_points[:, 1], radii, window
                )
            else:
                x1, y1, theta1 = ends[index]
                cos_t, sin_t = np.cos(theta1), np.sin(theta1)
                # Half-decimated cloud for the deferred scans: kerb and wall
                # returns arrive as dense lines, so every second point still
                # samples them far finer than the collision corridor is wide.
                # The immediate family keeps the full cloud.
                sparse = scan_points[::2]
                dx = sparse[:, 0] - np.float32(x1)
                dy = sparse[:, 1] - np.float32(y1)
                # Rotate by -theta1: express the cloud in segment A's endpoint
                # frame so segment B is a plain arc scan again.
                local = np.column_stack(
                    (cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy)
                ).astype(np.float32)
                free_b, dev_b = _scan_arcs(
                    local, safe, collision_half_width, horizon
                )
                if free_a < d1:
                    free = np.full(arc_count, free_a)
                else:
                    free = np.minimum(d1 + free_b, horizon)
                window = np.minimum(free, lookahead)
                # Segment A's clearance window is one scalar per family: the
                # per-candidate free-truncation only matters when the path is
                # already blocked inside the transition, where the free term
                # dominates the cost anyway.
                clear_near = _windowed_clearance(
                    dev_a,
                    origin_range,
                    scan_points[:, 1],
                    radius_k0,
                    np.asarray([min(lookahead, d1)]),
                )[0]
                clear_far = _windowed_clearance(
                    dev_b,
                    np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2),
                    local[:, 1],
                    radii,
                    window - d1,
                )
                clearance = np.minimum(clear_near, clear_far)
            free_all[index] = free
            clear_all[index] = np.where(
                np.isfinite(clearance),
                clearance,
                float(MAX_CORRIDOR_HALF_WIDTH_M),
            )

    required_free = (
        REQUIRED_FREE_DISTANCE_M
        if required_free_m is None
        else float(required_free_m)
    )
    smoothness = (
        COST_SMOOTHNESS if smoothness_weight is None else float(smoothness_weight)
    )
    # Free distance is scored against what the braking envelope actually needs,
    # not against the horizon: 20 m of clear road is not worse than 35 m, and
    # treating it as worse is what makes a planner refuse to ever turn.
    cost = COST_FREE_DISTANCE * np.clip(
        1.0 - free_all / required_free, 0.0, 1.0
    )
    cost = cost + COST_CLEARANCE * (
        1.0
        - np.clip(
            (clear_all - collision_half_width) / DESIRED_CLEARANCE_M, 0.0, 1.0
        )
    )
    # Smoothness scores the eventual curvature change identically in every
    # family. A deferral discount would make "turn later" always cheaper than
    # "turn now", and under per-tick re-planning "later" never arrives -- so
    # deferral must win on geometry (an immediate turn collides, a deferred one
    # does not), never on comfort. Also the tie-break that keeps an empty scene
    # pointing straight ahead: ties resolve to the first row, the immediate fan.
    cost = cost + smoothness * (
        ((curvatures - k0) / MAX_CURVATURE) ** 2
    )[None, :]
    # The deferral tie-break: near-equal plans act now rather than promising a
    # turn that per-tick re-planning would keep postponing.
    cost = cost + COST_TRANSITION * (d1_array / max(d1_array[-1], 1.0))[:, None]

    def _family_offsets() -> np.ndarray:
        """
        Candidate lateral offsets at the lookahead, per family.

        The shared geometry under keep-right (no route) and the route
        cross-track term (route present) -- only ever one of the two runs.
        """
        offsets = np.empty((len(families), arc_count))
        for index, d1 in enumerate(families):
            if d1 <= 0.0:
                offsets[index] = _lateral_offsets(curvatures, safe, lookahead)
            elif d1 >= lookahead:
                # Still on segment A at the lookahead: one offset for the row.
                offsets[index] = _lateral_offsets(
                    np.asarray([k0]), np.asarray([safe_k0]), lookahead
                )[0]
            else:
                x1, _, theta1 = ends[index]
                cos_t, sin_t = np.cos(theta1), np.sin(theta1)
                remaining = lookahead - d1
                local_x = -(1.0 - np.cos(safe * remaining)) / safe
                local_y = np.sin(safe * remaining) / safe
                offsets[index] = x1 + cos_t * local_x - sin_t * local_y
        return offsets

    def _composite_headings() -> np.ndarray:
        # The composite path's heading at the lookahead, identical units for
        # every family (for the immediate fan this is algebraically the old
        # curvature-error form).
        return (
            k0 * np.minimum(d1_array, lookahead)[:, None]
            + curvatures[None, :]
            * np.maximum(lookahead - d1_array, 0.0)[:, None]
        )

    def _family_endpoints() -> tuple[np.ndarray, np.ndarray]:
        """
        EXACT composite-path endpoints at the lookahead, per family.

        The route terms need real positions, not the small-angle offsets:
        matching an approximated endpoint against a curved reference is what
        let candidates cut a 90-degree bend by seven metres in the closed
        loop, because at a mid-corner lookahead the straight-line model
        cannot represent where a curved path actually ends up.
        """
        end_x = np.empty((len(families), arc_count))
        end_y = np.empty((len(families), arc_count))
        for index, d1 in enumerate(families):
            if d1 >= lookahead:
                # Still on segment A at the lookahead: one point per row.
                ax, ay, _ = _arc_endpoint(k0, lookahead)
                end_x[index] = ax
                end_y[index] = ay
                continue
            remaining = lookahead - d1
            local_x = -(1.0 - np.cos(safe * remaining)) / safe
            local_y = np.sin(safe * remaining) / safe
            if d1 <= 0.0:
                end_x[index] = local_x
                end_y[index] = local_y
            else:
                x1, y1, theta1 = ends[index]
                cos_t, sin_t = np.cos(theta1), np.sin(theta1)
                end_x[index] = x1 + cos_t * local_x - sin_t * local_y
                end_y[index] = y1 + sin_t * local_x + cos_t * local_y
        return end_x, end_y

    keep_right_target: float | None = None
    route_heading_out: float | None = None
    if route is not None:
        # ROUTE PRESENT: the reference path replaces BOTH lateral guidance
        # terms. Two lateral targets fighting -- kerb band versus route
        # centreline -- is the failure this gate avoids, and the kerb band is
        # measured over a STRAIGHT forward slice (`corridor_edges`), which
        # degrades on exactly the bends the route tangent is strongest on.
        route_pts = route.points.astype(np.float64)
        route_theta = route.headings
        route_width = route.half_width_m
        half_width = geometry.width_m / 2.0

        def _priced_deviation(px: np.ndarray, py: np.ndarray) -> np.ndarray:
            """
            Cross-track cost of candidate positions against their NEAREST
            route samples: signed offset along the matched sample's right
            normal, against a lane target right of the centreline by half
            the right lane -- clamped so a road too narrow to lane-keep in
            centres rather than aiming the car at the kerb (keep-right's own
            clamp philosophy).
            """
            dx = px[:, None] - route_pts[None, :, 0]
            dy = py[:, None] - route_pts[None, :, 1]
            nearest = np.argmin(dx * dx + dy * dy, axis=-1)
            theta_n = route_theta[nearest]
            width_n = route_width[nearest]
            lane = np.clip(
                width_n / 2.0,
                0.0,
                np.maximum(width_n - half_width - KEEP_RIGHT_MARGIN_M, 0.0),
            )
            cross = (px - route_pts[nearest, 0]) * np.cos(theta_n) + (
                py - route_pts[nearest, 1]
            ) * np.sin(theta_n)
            return np.clip(
                ((cross - lane) / ROUTE_XTRACK_SCALE_M) ** 2, 0.0, 1.0
            )

        # Conformance is the MEAN priced deviation over samples along each
        # candidate's composite path. One endpoint cannot measure a path
        # through a bend: a shallow arc that cuts a 90-degree corner lands
        # its endpoint ON the ribbon at the apex, and priced there alone it
        # read as perfect -- measured 7.4 m inside the bend in the closed
        # loop. Every family is sampled by the same rule at the same arc
        # lengths, so no deferral discount enters here: a deferred family
        # whose held segment drifts off a bending route pays for exactly
        # that drift, which is geometry, not comfort.
        ribbon = np.zeros((len(families), arc_count))
        fractions = (
            np.arange(1, ROUTE_MATCH_SAMPLES + 1) / ROUTE_MATCH_SAMPLES
        )
        for index, d1 in enumerate(families):
            if d1 > 0.0:
                x1, y1, theta1 = ends[index]
                cos_t, sin_t = np.cos(theta1), np.sin(theta1)
            for fraction in fractions:
                s = fraction * lookahead
                if s <= d1:
                    # Still on segment A: one shared point for the row.
                    ax, ay, _ = _arc_endpoint(k0, s)
                    ribbon[index] += _priced_deviation(
                        np.full(arc_count, ax), np.full(arc_count, ay)
                    )
                    continue
                remaining = s - d1
                local_x = -(1.0 - np.cos(safe * remaining)) / safe
                local_y = np.sin(safe * remaining) / safe
                if d1 <= 0.0:
                    px, py = local_x, local_y
                else:
                    px = x1 + cos_t * local_x - sin_t * local_y
                    py = y1 + sin_t * local_x + cos_t * local_y
                ribbon[index] += _priced_deviation(px, py)
        cost = cost + COST_ROUTE_XTRACK * ribbon / ROUTE_MATCH_SAMPLES

        # The tangent term: each candidate's EXACT endpoint matched to its
        # nearest route sample, compared there -- never at one fixed arc
        # distance, which rewarded turning toward a mid-corner tangent
        # early. Rides COST_NAV_HEADING's exact normalisation so its tuned
        # rank carries over.
        end_x, end_y = _family_endpoints()
        dx = end_x[..., None] - route_pts[None, None, :, 0]
        dy = end_y[..., None] - route_pts[None, None, :, 1]
        nearest_end = np.argmin(dx * dx + dy * dy, axis=-1)
        cost = cost + COST_ROUTE_HEADING * (
            (_composite_headings() - route_theta[nearest_end])
            / (MAX_CURVATURE * lookahead)
        ) ** 2
        # The scalar tangent at the lookahead, for the overlay spoke.
        route_heading_out = float(
            np.interp(lookahead, route.arc_s, route.headings)
        )
    elif keep_right:
        left_edge, right_edge = corridor_edges(points, lookahead)
        if right_edge is not None:
            half_width = geometry.width_m / 2.0
            target = right_edge - (half_width + KEEP_RIGHT_MARGIN_M)
            if left_edge is not None:
                # A corridor too narrow to keep right in: centre it rather
                # than aiming the car at the left kerb.
                target = max(
                    target, left_edge + half_width + KEEP_RIGHT_MARGIN_M
                )
            keep_right_target = float(target)
            cost = cost + COST_KEEP_RIGHT * np.clip(
                ((_family_offsets() - keep_right_target) / KEEP_RIGHT_SCALE_M)
                ** 2,
                0.0,
                1.0,
            )

    if nav_heading_rad is not None and route is None:
        cost = cost + COST_NAV_HEADING * (
            (_composite_headings() - float(nav_heading_rad))
            / (MAX_CURVATURE * lookahead)
        ) ** 2

    if road_grid is not None:
        # The road-coverage BONUS: mean occupancy under samples of each
        # composite path, over the stretch actually about to be driven --
        # min(free, lookahead), the same window clearance uses, so an arc is
        # never rewarded for road it will brake before reaching. A NEGATIVE
        # term, per the long-standing design note: geometry keeps deciding
        # what is drivable; semantics only make the tarmac preferable.
        # Out-of-grid samples read as off-road. Sampled by the same rule for
        # every family: no deferral discount enters here either.
        occupancy = road_grid.occupancy
        grid_h, grid_w = occupancy.shape
        coverage = np.zeros((len(families), arc_count))
        fractions = np.arange(1, ROAD_BONUS_SAMPLES + 1) / ROAD_BONUS_SAMPLES
        safe_k0_arr = np.asarray([safe_k0])
        for index, d1 in enumerate(families):
            window = np.minimum(free_all[index], lookahead)
            if d1 > 0.0:
                x1, y1, theta1 = ends[index]
                cos_t, sin_t = np.cos(theta1), np.sin(theta1)
            for fraction in fractions:
                s = fraction * window  # per-candidate arc lengths
                s_a = np.minimum(s, d1)
                s_b = np.maximum(s - d1, 0.0)
                if d1 <= 0.0:
                    px = -(1.0 - np.cos(safe * s)) / safe
                    py = np.sin(safe * s) / safe
                else:
                    ax = -(1.0 - np.cos(safe_k0_arr * s_a)) / safe_k0_arr
                    ay = np.sin(safe_k0_arr * s_a) / safe_k0_arr
                    local_x = -(1.0 - np.cos(safe * s_b)) / safe
                    local_y = np.sin(safe * s_b) / safe
                    on_b = s_b > 0.0
                    px = np.where(
                        on_b, x1 + cos_t * local_x - sin_t * local_y, ax
                    )
                    py = np.where(
                        on_b, y1 + sin_t * local_x + cos_t * local_y, ay
                    )
                col = np.floor(
                    (px - road_grid.origin_right_m) / road_grid.cell_m
                ).astype(np.intp)
                row = np.floor(
                    (py - road_grid.origin_forward_m) / road_grid.cell_m
                ).astype(np.intp)
                inside = (
                    (col >= 0) & (col < grid_w) & (row >= 0) & (row < grid_h)
                )
                hit = np.zeros(arc_count)
                hit[inside] = occupancy[row[inside], col[inside]]
                coverage[index] += hit
        cost = cost - COST_ROAD_BONUS * coverage / ROAD_BONUS_SAMPLES

    family_index, best = divmod(int(np.argmin(cost)), arc_count)
    transition = float(families[family_index])
    row = cost[family_index]
    next_curvature = float(curvatures[best])
    free_best = float(free_all[family_index, best])
    if 0 < best < arc_count - 1:
        # Parabolic interpolation between fan steps. Free distance takes the
        # neighbourhood minimum: adjacent collision corridors overlap through
        # the horizon, so anything blocking an in-between arc blocked a
        # neighbour too, and the minimum is rigorous rather than hopeful.
        c0, c1, c2 = float(row[best - 1]), float(row[best]), float(row[best + 1])
        denom = c0 - 2.0 * c1 + c2
        if np.isfinite(denom) and denom > 1e-12:
            # Index-space parabola; the grid is non-uniform, so the shift maps
            # through the spacing on the side it moves toward.
            shift = float(np.clip(0.5 * (c0 - c2) / denom, -0.5, 0.5))
            if shift > 0.0:
                step = float(curvatures[best + 1] - curvatures[best])
            else:
                step = float(curvatures[best] - curvatures[best - 1])
            next_curvature += shift * step
            # The interpolated arc lies between the winner and ONE neighbour;
            # in the dense region where interpolation actually operates their
            # collision corridors overlap through the horizon, so the pairwise
            # minimum is a sound free distance for it.
            if shift > 0.0:
                free_best = min(free_best, float(free_all[family_index, best + 1]))
            elif shift < 0.0:
                free_best = min(free_best, float(free_all[family_index, best - 1]))
    return ArcPlan(
        curvature=k0 if transition > 0.0 else next_curvature,
        free_distance_m=free_best,
        clearance_m=float(clear_all[family_index, best]),
        keep_right_target_m=keep_right_target,
        nav_heading_rad=(
            None if nav_heading_rad is None else float(nav_heading_rad)
        ),
        candidate_curvatures=curvatures.astype(np.float32),
        candidate_costs=cost[0].astype(np.float32),
        candidate_free_distances=free_all[0].astype(np.float32),
        next_curvature=next_curvature,
        transition_distance_m=transition,
        lookahead_m=lookahead,
        route_speed_limit_mps=(
            None if route is None else float(route.speed_limit_mps)
        ),
        route_cross_track_m=(
            None if route is None else float(route.cross_track_m)
        ),
        route_heading_rad=route_heading_out,
    )
