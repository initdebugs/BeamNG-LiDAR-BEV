"""
Parking bay detection from paint, offline over synthetic marking layouts.

Every scene here is built the way the sensors actually lay paint down: a
divider is a run of cells along a line, sampled at the marking cell pitch,
with the row of them parallel and evenly spaced. The cases that matter are
the ones where a plausible-looking rule gets the wrong answer -- a global PCA
returning the row's axis instead of a stripe's, a squared-count score letting
one long head line outvote eight real dividers, an aisle's two edge lines
looking exactly like a very deep bay -- so each has a test of its own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    PARKING_BAY_MAX_DEPTH_M,
    PARKING_BAY_MEMORY_M,
    PARKING_BAY_MIN_DEPTH_M,
    PARKING_BAY_WIDTH_MAX_M,
    PARKING_BAY_WIDTH_MIN_M,
    PARKING_MARKING_CELL_M,
    PARKING_MARKING_MEMORY_M,
    PARKING_MARKING_RADIUS_M,
    PARKING_MAX_SLOTS,
    PARKING_OFFSET_BIN_M,
    PARKING_SCAN_RADIUS_M,
    PARKING_SELECT_MATCH_M,
    PARKING_STRIPE_GAP_M,
    PARKING_STRIPE_MAX_WIDTH_M,
)
from beamng_lidar_bev.parking import (
    MarkingMemory,
    ScanReport,
    dominant_axis,
    find_bays,
    match_selection,
    project_bays,
    remember_bays,
    slot_contains,
)

RIGHT = np.asarray((1.0, 0.0, 0.0))
FORWARD = np.asarray((0.0, 1.0, 0.0))
# Facing world +X, so the car's right is world -Y.
TURNED_RIGHT = np.asarray((0.0, -1.0, 0.0))
TURNED_FORWARD = np.asarray((1.0, 0.0, 0.0))
_EMPTY = np.empty((0, 2), dtype=np.float64)


def _line(
    start: tuple[float, float], along: tuple[float, float], length: float
) -> np.ndarray:
    """One painted divider, sampled at the marking cell pitch."""
    steps = int(length / PARKING_MARKING_CELL_M) + 1
    t = np.linspace(0.0, length, steps)[:, None]
    return np.asarray(start)[None, :] + t * np.asarray(along)[None, :]


def _bay_row(
    count: int = 8,
    spacing: float = 2.5,
    depth: float = 5.0,
    angle_deg: float = 0.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    `count` parallel dividers, so `count - 1` bays between them.

    At 0 degrees the dividers run along +Y and the row advances along +X --
    a car driving the aisle eastward sees the bays on its left. `angle_deg`
    rotates the whole row.
    """
    angle = math.radians(angle_deg)
    along = (-math.sin(angle), math.cos(angle))
    across = (math.cos(angle), math.sin(angle))
    lines = [
        _line(
            (
                origin[0] + across[0] * spacing * i,
                origin[1] + across[1] * spacing * i,
            ),
            along,
            depth,
        )
        for i in range(count)
    ]
    return np.concatenate(lines, axis=0)


def test_the_marking_store_reaches_further_than_the_scan() -> None:
    """
    Otherwise the STORE bounds how far bays are found rather than the scan,
    and it does it silently -- the scan just receives fewer cells.
    """
    assert PARKING_MARKING_RADIUS_M > PARKING_SCAN_RADIUS_M


def test_the_stripe_gap_sits_between_a_sampling_hole_and_an_aisle() -> None:
    """
    Bounded on both sides and neither bound is slack: too small and one
    divider fragments into pieces too short to bound a bay, too large and two
    facing rows merge back into one stripe and the whole lot disappears.
    """
    assert PARKING_STRIPE_GAP_M > 1.0
    assert PARKING_STRIPE_GAP_M < PARKING_BAY_MIN_DEPTH_M


def test_the_offset_bin_is_never_finer_than_the_marking_cell() -> None:
    """
    The sweep's score depends on this and the failure is total, not gradual.

    Below the cell pitch a divider seen along its length lands in alternating
    bins, reads as a row of narrow one-bin stripes, and ties the score of the
    same divider seen end-on -- so the angle 90 degrees from the right one
    scores equally and nothing is found.
    """
    assert PARKING_OFFSET_BIN_M >= PARKING_MARKING_CELL_M


def test_a_row_of_bays_is_found_with_the_right_width_and_depth() -> None:
    marking = _bay_row(count=8, spacing=2.5, depth=5.0)

    bays = find_bays(marking, _EMPTY, np.asarray((0.0, -6.0)))

    assert len(bays) == 7
    assert all(abs(bay.width_m - 2.5) < 0.15 for bay in bays)
    assert all(abs(bay.depth_m - 5.0) < 0.3 for bay in bays)
    assert not any(bay.occupied for bay in bays)


def test_the_stripe_axis_is_not_the_rows_axis() -> None:
    """
    The defect a global PCA would have.

    Eight 5 m dividers spread over 17.5 m of frontage have their greatest
    variance ACROSS the row, so a covariance fit returns a direction 90
    degrees from the one wanted. The sweep has to return the divider's.
    """
    marking = _bay_row(count=8, spacing=2.5, depth=5.0)
    centred = marking - marking.mean(axis=0)

    _, vectors = np.linalg.eigh(np.cov(centred.T))
    principal = vectors[:, -1]
    pca_angle = math.atan2(principal[1], principal[0]) % math.pi

    swept = dominant_axis(centred)

    # The dividers run along +Y, so pi/2 is the answer wanted. PCA returns
    # the frontage instead, which is 0 -- a full 90 degrees wrong.
    assert min(pca_angle, math.pi - pca_angle) < math.radians(5.0)
    assert abs(swept - math.pi / 2.0) < math.radians(2.0)


def test_bays_are_found_at_any_row_angle() -> None:
    for angle_deg in (0.0, 17.0, 33.0, 61.0, 90.0, 128.0):
        marking = _bay_row(count=6, spacing=2.6, depth=5.0, angle_deg=angle_deg)
        bays = find_bays(marking, _EMPTY, np.asarray((0.0, -8.0)))

        assert len(bays) == 5, f"{angle_deg} deg produced {len(bays)} bays"
        assert all(abs(bay.width_m - 2.6) < 0.2 for bay in bays)


def test_a_head_line_across_the_bays_does_not_capture_the_sweep() -> None:
    """
    A line across the heads of the bays, which many real lots paint.

    It threatens the sweep twice, and the second way is the dangerous one.
    At its OWN angle it piles into a single bin, which a squared-count score
    would rank above eight genuine dividers. And at the DIVIDERS' angle it
    smears along the whole row, bridging every divider's bin into one run too
    wide to be a stripe -- measured, that took this scene from seven bays to
    none, which is a worse failure than being outvoted. The per-bin floor is
    what removes the smear; the width cap is what removes the pile.
    """
    dividers = _bay_row(count=8, spacing=2.5, depth=5.0)
    # Across the heads of all eight, which is where a lot really paints it.
    head = _line((0.0, 5.0), (1.0, 0.0), 17.5)

    bays = find_bays(
        np.concatenate((dividers, head)), _EMPTY, np.asarray((0.0, -6.0))
    )

    assert len(bays) == 7
    assert all(abs(bay.width_m - 2.5) < 0.15 for bay in bays)


def test_two_facing_rows_across_an_aisle_are_both_found() -> None:
    """
    What every real lot looks like, and it used to detect NOTHING.

    Facing rows put their dividers at the SAME perpendicular offsets, so
    before the stripe-gap split each pair merged into one stripe spanning
    both rows and every bay failed PARKING_BAY_MAX_DEPTH_M. And once split,
    the offsets REPEAT, so pairing by adjacency in sorted-offset order saw
    only zero-width pairs and cross-row pairs. Measured: 5 bays + 5 bays came
    back as 0 under either defect alone.
    """
    near = _bay_row(count=6, spacing=2.5, depth=5.0, origin=(-6.0, 6.0))
    far = _bay_row(count=6, spacing=2.5, depth=5.0, origin=(-6.0, 16.0))
    ego = np.asarray((0.0, 0.0))

    assert len(find_bays(near, _EMPTY, ego)) == 5
    assert len(find_bays(far, _EMPTY, ego)) == 5

    both = find_bays(np.concatenate((near, far)), _EMPTY, ego)

    assert len(both) == 10
    # Five in each row, so the two rows really are both represented rather
    # than one of them being found twice.
    assert sum(1 for bay in both if bay.centre[1] < 12.0) == 5
    assert sum(1 for bay in both if bay.centre[1] > 12.0) == 5


def test_a_stray_stripe_between_dividers_does_not_kill_its_neighbours() -> None:
    """
    Pairing looks for the nearest OVERLAPPING partner, not the next stripe in
    offset order. A fragment of paint interleaved between two real dividers
    -- a kerb line, a worn patch, part of a head line -- broke the adjacency
    and took out the bay on both sides of it.
    """
    dividers = _bay_row(count=4, spacing=2.5, depth=5.0)
    # A short fragment sitting between the second and third divider, well
    # clear of them along the bay so it overlaps neither.
    stray = _line((3.6, 9.0), (0.0, 1.0), 2.4)

    bays = find_bays(
        np.concatenate((dividers, stray)), _EMPTY, np.asarray((3.75, -8.0))
    )

    assert len(bays) == 3


def test_bays_are_offered_out_to_the_scan_radius() -> None:
    """
    WORLD draws paint to WORLD_ROAD_RADIUS_M, so a scan radius far short of
    it shows a full row of painted bays with outlines on only the near few.
    """
    row = np.concatenate(
        [
            _line((8.0 + 2.5 * i, 6.0), (0.0, 1.0), 5.0)
            for i in range(13)
        ]
    )

    bays = find_bays(row, _EMPTY, np.asarray((0.0, 0.0)))

    assert len(bays) == 12
    furthest = max(math.hypot(*bay.centre) for bay in bays)
    assert furthest > 35.0, "detection used to stop dead at 32.9 m"


def test_an_aisles_edge_lines_are_not_a_very_deep_bay() -> None:
    """Parallel, plausibly spaced, and only their LENGTH gives them away."""
    aisle = _bay_row(count=2, spacing=3.2, depth=PARKING_BAY_MAX_DEPTH_M + 12.0)

    assert find_bays(aisle, _EMPTY, np.asarray((0.0, -5.0))) == ()


def test_the_two_halves_of_a_double_divider_are_not_a_bay() -> None:
    """
    Some lots paint each divider as a close pair. Nobody parks in the 0.4 m
    between the halves, and the property that matters is that it is never
    offered -- whether the pair merges into one divider or is rejected as too
    narrow a gap is an implementation detail, and it does merge.
    """
    marking = np.concatenate(
        (
            _line((0.0, 0.0), (0.0, 1.0), 5.0),
            _line((0.4, 0.0), (0.0, 1.0), 5.0),
            _line((2.9, 0.0), (0.0, 1.0), 5.0),
            _line((3.3, 0.0), (0.0, 1.0), 5.0),
        )
    )

    bays = find_bays(marking, _EMPTY, np.asarray((1.5, -6.0)))

    # Whether the two halves merge into one divider or stay separate is an
    # implementation detail that the stripe width cap decides. What must
    # always hold is that the 0.4 m slot between them is never offered, and
    # that every bay returned spans the real gap.
    assert bays
    assert all(bay.width_m >= PARKING_BAY_WIDTH_MIN_M for bay in bays)
    assert all(2.4 < bay.width_m < 3.4 for bay in bays)


def _angled_lot(
    count: int,
    bay_angle_deg: float,
    width: float = 2.5,
    depth: float = 5.0,
    origin: tuple[float, float] = (-12.0, 4.0),
) -> np.ndarray:
    """
    A real angled (herringbone) lot.

    Dividers start along a common aisle edge and run off at `bay_angle_deg`,
    spaced so the PERPENDICULAR gap between them is the bay width -- which is
    how such a lot is actually laid out. Neighbours are therefore STAGGERED
    along their own direction by `width / tan(angle)`.
    """
    angle = math.radians(bay_angle_deg)
    pitch = width / math.sin(angle)
    along = (math.cos(angle), math.sin(angle))
    return np.concatenate(
        [
            _line((origin[0] + pitch * i, origin[1]), along, depth)
            for i in range(count)
        ]
    )


@pytest.mark.parametrize("bay_angle_deg", (90.0, 75.0, 60.0, 45.0, 30.0))
def test_angled_lots_are_found_at_every_bay_angle(bay_angle_deg: float) -> None:
    """
    Depth is the shorter divider's LENGTH, not the overlap, and this is why.

    In an angled lot neighbouring dividers are staggered along their own
    direction, so measuring depth as their overlap subtracts a stagger the
    bay's real depth does not depend on. Measured on a proper 2.5 x 5.0 m
    lot, the overlap runs 5.00 / 4.33 / 3.56 / 2.50 / 0.67 m across these
    angles -- so everything from 60 degrees down was rejected against the
    3.6 m floor, a 60-degree bay missing it by 4 cm.
    """
    marking = _angled_lot(count=7, bay_angle_deg=bay_angle_deg)

    bays = find_bays(marking, _EMPTY, np.asarray((0.0, -6.0)))

    assert len(bays) == 6
    assert all(abs(bay.width_m - 2.5) < 0.15 for bay in bays)
    assert all(abs(bay.depth_m - 5.0) < 0.3 for bay in bays)


def test_a_staggered_pair_is_as_deep_as_its_dividers() -> None:
    """
    Two 6 m dividers offset by 2 m bound a 6 m bay lying at a stagger, not a
    4 m one: the offset is where the bay SITS, not how deep it is.
    """
    marking = np.concatenate(
        (
            _line((0.0, 0.0), (0.0, 1.0), 6.0),
            _line((2.5, 2.0), (0.0, 1.0), 6.0),
        )
    )

    bays = find_bays(marking, _EMPTY, np.asarray((1.25, -6.0)))

    assert len(bays) == 1
    assert abs(bays[0].depth_m - 6.0) < 0.3
    # Centred between the two dividers' own centres (3.0 and 5.0), not on
    # the slice of them that happens to overlap.
    assert abs(bays[0].centre[1] - 4.0) < 0.3


def test_a_short_divider_still_bounds_only_what_was_seen() -> None:
    """The depth floor is what rejects a pair with too little evidence."""
    marking = np.concatenate(
        (
            _line((0.0, 0.0), (0.0, 1.0), 6.0),
            _line((2.5, 0.0), (0.0, 1.0), 2.0),
        )
    )

    assert find_bays(marking, _EMPTY, np.asarray((1.25, -6.0))) == ()


def test_a_car_in_one_bay_marks_only_that_bay_occupied() -> None:
    marking = _bay_row(count=4, spacing=2.5, depth=5.0)
    # A car's returns filling the middle bay, whose centre is at x = 3.75.
    car = np.column_stack(
        (
            np.random.default_rng(0).uniform(3.0, 4.5, 60),
            np.random.default_rng(1).uniform(1.0, 4.0, 60),
        )
    )

    bays = find_bays(marking, car, np.asarray((3.75, -6.0)))

    assert len(bays) == 3
    occupied = [bay for bay in bays if bay.occupied]
    assert len(occupied) == 1
    assert abs(occupied[0].centre[0] - 3.75) < 0.3


def test_paint_on_the_bay_boundary_does_not_read_as_occupied() -> None:
    """The shrink margin exists for the dividers, kerbs and wing mirrors."""
    marking = _bay_row(count=3, spacing=2.5, depth=5.0)
    # Returns sitting exactly on the dividers themselves.
    on_the_lines = marking.copy()

    bays = find_bays(marking, on_the_lines, np.asarray((1.25, -6.0)))

    assert len(bays) == 2
    assert not any(bay.occupied for bay in bays)


def test_the_mouth_faces_the_car() -> None:
    """The axis runs from the end nearer the ego into the bay."""
    marking = _bay_row(count=3, spacing=2.5, depth=5.0)

    from_below = find_bays(marking, _EMPTY, np.asarray((1.25, -8.0)))
    from_above = find_bays(marking, _EMPTY, np.asarray((1.25, 13.0)))

    assert from_below[0].axis[1] > 0.9
    assert from_above[0].axis[1] < -0.9


def test_rows_are_found_all_round_the_car_not_just_ahead() -> None:
    """
    Nothing in the detector filters by BEARING, and this pins it.

    Reported from the app as "the scan only uses the front sensor". The cause
    was not directional sensing at all: the sweep returns ONE angle, so a
    single pass kept whichever row carried the most paint and dropped every
    row at a different orientation. In a lot that is the row you happen to be
    facing, which is indistinguishable from a forward-only scan.
    """
    ahead = _bay_row(count=5, spacing=2.5, depth=5.0, origin=(-5.0, 8.0))
    behind = _bay_row(
        count=5, spacing=2.5, depth=5.0, angle_deg=90.0, origin=(-14.0, -12.0)
    )
    beside = _bay_row(
        count=4, spacing=2.5, depth=5.0, angle_deg=40.0, origin=(-22.0, 2.0)
    )
    ego = np.asarray((0.0, 0.0))

    # Each row alone is found in full, wherever it sits relative to the car.
    assert len(find_bays(ahead, _EMPTY, ego)) == 4
    assert len(find_bays(behind, _EMPTY, ego)) == 4
    assert len(find_bays(beside, _EMPTY, ego)) == 3

    every = np.concatenate((ahead, behind, beside))
    bays = find_bays(every, _EMPTY, ego)

    assert len(bays) == 11, "a second row must not cost the first one its bays"
    bearings = sorted(
        math.degrees(math.atan2(bay.centre[0], bay.centre[1])) % 360.0
        for bay in bays
    )
    # Bays behind the car (bearings near 180-300) and ahead of it (near 0)
    # both survive, which is the property the single-pass sweep lost.
    assert any(bearing < 45.0 or bearing > 315.0 for bearing in bearings)
    assert any(180.0 < bearing < 320.0 for bearing in bearings)


def test_the_row_sweep_terminates_on_paint_that_bounds_no_bay() -> None:
    """
    Each pass consumes the stripes it found whether or not they yielded a
    bay. Without that the next pass re-finds the same angle for ever and only
    the loop bound ends it.
    """
    # Dividers far too widely spaced to bound a bay, at two orientations.
    sparse = np.concatenate(
        (
            _bay_row(count=4, spacing=9.0, depth=5.0),
            _bay_row(count=4, spacing=9.0, depth=5.0, angle_deg=90.0),
        )
    )

    assert find_bays(sparse, _EMPTY, np.asarray((0.0, 0.0))) == ()


def test_an_empty_or_sparse_scene_offers_nothing() -> None:
    ego = np.asarray((0.0, 0.0))

    assert find_bays(_EMPTY, _EMPTY, ego) == ()
    assert find_bays(_line((0.0, 0.0), (0.0, 1.0), 1.0), _EMPTY, ego) == ()
    # Paint far outside the scan radius is not a bay near the car.
    assert find_bays(_bay_row(origin=(400.0, 400.0)), _EMPTY, ego) == ()


def test_only_the_nearest_bays_are_offered() -> None:
    """
    A COMPACT lot rather than one long row: past the cap the row itself would
    outrun the scan radius, so the cull rather than the cap would be what
    bounded the answer and the test would stop testing the cap.
    """
    rows = [
        _bay_row(count=16, spacing=2.5, depth=5.0, origin=(-20.0, y))
        for y in (-18.0, -6.0, 6.0, 18.0)
    ]
    ego = np.asarray((0.0, 0.0))

    bays = find_bays(np.concatenate(rows), _EMPTY, ego)

    assert len(bays) == PARKING_MAX_SLOTS
    distances = [
        math.hypot(bay.centre[0] - ego[0], bay.centre[1] - ego[1])
        for bay in bays
    ]
    # Recomputed here rather than read back, so compare with a tolerance: the
    # sort key is the same arithmetic but not the same rounding.
    assert all(
        later >= earlier - 1e-9
        for earlier, later in zip(distances, distances[1:])
    )


def test_a_bay_projects_into_the_bev_frame_and_hit_tests() -> None:
    """World-anchored: the drawn rectangle follows the car and the turn."""
    marking = _bay_row(count=3, spacing=2.5, depth=5.0)
    bays = find_bays(marking, _EMPTY, np.asarray((1.25, -8.0)))

    ahead = project_bays(bays, np.asarray((1.25, -8.0, 0.0)), RIGHT, FORWARD)
    turned = project_bays(
        bays, np.asarray((1.25, -8.0, 0.0)), TURNED_RIGHT, TURNED_FORWARD
    )

    assert len(ahead) == 2
    # Facing world +Y, a bay dead ahead sits at forward > 0, right ~ 0.
    assert ahead[0].centre_forward_m > 8.0
    assert abs(ahead[0].centre_right_m) < 0.3
    # Facing world +X, the same bay is off to the left.
    assert turned[0].centre_right_m < -8.0
    assert abs(turned[0].centre_forward_m) < 0.3

    slot = ahead[0]
    assert slot_contains(slot, slot.centre_right_m, slot.centre_forward_m)
    assert not slot_contains(
        slot, slot.centre_right_m + slot.width_m, slot.centre_forward_m
    )
    assert not slot_contains(
        slot, slot.centre_right_m, slot.centre_forward_m + slot.depth_m
    )


def test_a_selection_survives_a_rescan_but_never_jumps_to_another_bay() -> None:
    marking = _bay_row(count=4, spacing=2.5, depth=5.0)
    bays = find_bays(marking, _EMPTY, np.asarray((1.25, -8.0)))
    chosen = bays[1]

    # A rescan an instant later, the centre recovered to within a cell.
    nudged = (chosen.centre[0] + 0.1, chosen.centre[1] - 0.1)
    assert match_selection(bays, nudged) is chosen

    # A selection that has drifted past the match radius goes quiet rather
    # than silently becoming whichever bay is now closest. Displaced ALONG
    # the dividers, where the next bay along the row cannot claim it.
    adrift = (
        chosen.centre[0],
        chosen.centre[1] + PARKING_SELECT_MATCH_M + 25.0,
    )
    assert match_selection(bays, adrift) is None
    assert match_selection((), chosen.centre) is None
    assert match_selection(bays, None) is None


def test_selection_is_marked_on_the_projected_slot() -> None:
    marking = _bay_row(count=4, spacing=2.5, depth=5.0)
    bays = find_bays(marking, _EMPTY, np.asarray((1.25, -8.0)))

    slots = project_bays(
        bays,
        np.asarray((1.25, -8.0, 0.0)),
        RIGHT,
        FORWARD,
        selected_world=bays[1].centre,
    )

    assert [slot.selected for slot in slots].count(True) == 1
    assert slots[1].selected


def test_marking_memory_accumulates_across_ticks_and_survives_motion() -> None:
    """
    The store is what makes an empty lot legible at all.

    A single frame lays ground rings ACROSS the dividers rather than along
    them; the continuous lines the detector needs only exist because cells
    accumulate as the car moves.
    """
    memory = MarkingMemory()
    # Two passes, each seeing alternate halves of the same two dividers.
    first = np.concatenate(
        (_line((0.0, 0.0), (0.0, 1.0), 2.4), _line((2.5, 0.0), (0.0, 1.0), 2.4))
    )
    memory.update(np.asarray((1.25, -5.0, 0.0)), RIGHT, FORWARD, first - (1.25, -5.0))
    assert memory.cell_count > 0

    second = np.concatenate(
        (_line((0.0, 2.6), (0.0, 1.0), 2.4), _line((2.5, 2.6), (0.0, 1.0), 2.4))
    )
    memory.update(np.asarray((1.25, -3.0, 0.0)), RIGHT, FORWARD, second - (1.25, -3.0))

    cells = memory.cells_world()
    bays = find_bays(cells, _EMPTY, np.asarray((1.25, -5.0)))

    assert len(bays) == 1
    assert abs(bays[0].width_m - 2.5) < 0.2
    # The two passes joined into one 5 m divider rather than staying apart.
    assert bays[0].depth_m > 4.0


def test_marking_memory_forgets_by_the_metre_and_clears_on_a_teleport() -> None:
    memory = MarkingMemory()
    paint = _line((0.0, 0.0), (0.0, 1.0), 3.0)
    memory.update(np.asarray((0.0, 0.0, 0.0)), RIGHT, FORWARD, paint)
    seeded = memory.cell_count
    assert seeded > 0

    # Driven away in short steps, past the distance window.
    for step in range(1, int(PARKING_MARKING_MEMORY_M) + 12):
        memory.update(
            np.asarray((0.0, float(step), 0.0)), RIGHT, FORWARD, _EMPTY
        )
    assert memory.cell_count == 0

    memory.update(np.asarray((0.0, 0.0, 0.0)), RIGHT, FORWARD, paint)
    assert memory.cell_count == seeded
    # A respawn must never leave the old lot's paint standing in the new map.
    memory.update(np.asarray((5000.0, 5000.0, 0.0)), RIGHT, FORWARD, _EMPTY)
    assert memory.cell_count == 0


def _curved_row(
    count: int, radius_m: float, width: float = 3.2, depth: float = 5.0
) -> np.ndarray:
    """A row of bays following a curved wall, which is what a real lot does."""
    lines = []
    for index in range(count):
        phi = (width / radius_m) * index
        lines.append(
            _line(
                (radius_m * math.sin(phi), 10.0 - radius_m * (1.0 - math.cos(phi))),
                (math.sin(phi), math.cos(phi)),
                depth,
            )
        )
    return np.concatenate(lines)


def test_a_row_following_a_curved_wall_keeps_its_bays() -> None:
    """
    The live defect, and it needed BOTH halves of the pairing rewrite.

    A curved row's dividers are not parallel, so one swept angle fits only
    part of it and the rest is claimed by later sweeps. Pairing WITHIN a pass
    then meant neighbours found on different passes could never meet --
    measured live at 18 dividers -> 12 bays with 6 unpaired, and 10 dividers
    -> 5 bays with 5 unpaired on a single sweep. Pairing now runs once over
    every divider the scan found, matched on their own 2D geometry.
    """
    for radius_m in (150.0, 80.0):
        bays = find_bays(
            _curved_row(10, radius_m), _EMPTY, np.asarray((0.0, -6.0))
        )
        assert len(bays) == 9, f"radius {radius_m} gave {len(bays)} bays"


def test_a_divider_off_the_swept_angle_is_still_a_divider() -> None:
    """
    The stripe width cap doubles as the tolerance on the SWEEP ANGLE: a
    divider `t` off the sweep spreads over `L * sin(t)`, so at 5 m long a
    0.7 m cap silently dropped anything past 8 degrees -- which on a gently
    curved row cost a divider and with it a bay.
    """
    assert PARKING_STRIPE_MAX_WIDTH_M >= 5.0 * math.sin(math.radians(10.0))
    # And still under half the narrowest bay, so two adjacent dividers can
    # never merge into one run.
    assert PARKING_STRIPE_MAX_WIDTH_M < PARKING_BAY_WIDTH_MIN_M * 0.5


# --- bays flicker, and a selection must not flicker with them -----------------


def _bay(x: float, y: float) -> object:
    from beamng_lidar_bev.models import ParkingBay

    return ParkingBay(
        centre=(x, y),
        axis=(0.0, 1.0),
        width_m=2.5,
        depth_m=5.0,
        occupied=False,
        stripe_cells=52,
    )


def test_a_bay_a_scan_misses_survives_for_a_short_distance() -> None:
    """
    Reported live as bays flashing, and worse, as the SELECTION going away
    with them. Each bay sits near several thresholds at once, so a scan can
    drop it and find it again a moment later.
    """
    seen: dict = {}
    ego = np.asarray((0.0, 0.0))
    first = remember_bays((), (_bay(2.0, 10.0), _bay(5.0, 10.0)), 0.0, seen, ego)
    assert len(first) == 2

    # The next scan finds only one of them.
    blinked = remember_bays(first, (_bay(2.0, 10.0),), 1.0, seen, ego)

    assert len(blinked) == 2, "the missed bay must survive the blink"
    # And the selection still matches the one that blinked.
    assert match_selection(blinked, (5.0, 10.0)) is not None


def test_a_freshly_found_bay_always_replaces_its_remembered_twin() -> None:
    """Memory fills gaps; it never overrides what the sensors say now."""
    seen: dict = {}
    ego = np.asarray((0.0, 0.0))
    remembered = remember_bays((), (_bay(2.0, 10.0),), 0.0, seen, ego)

    moved = remember_bays(remembered, (_bay(2.1, 10.05),), 0.5, seen, ego)

    assert len(moved) == 1
    assert abs(moved[0].centre[0] - 2.1) < 1e-9


def test_a_bay_left_behind_is_forgotten() -> None:
    """Forgotten by the METRE, the two-clocks rule every other store uses."""
    seen: dict = {}
    ego = np.asarray((0.0, 0.0))
    remembered = remember_bays((), (_bay(2.0, 10.0),), 0.0, seen, ego)

    gone = remember_bays(
        remembered, (), PARKING_BAY_MEMORY_M + 1.0, seen, ego
    )

    assert gone == ()


# --- lots that annotate whole bay quads instead of the divider lines ----------


def _filled_row(
    count: int = 6,
    width: float = 3.0,
    depth: float = 5.3,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """A bay row whose whole QUADS annotate, so no divider line is visible."""
    step = PARKING_MARKING_CELL_M
    xs = np.arange(0.0, count * width, step) + origin[0]
    ys = np.arange(0.0, depth, step) + origin[1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def test_a_row_annotated_as_solid_quads_still_yields_bays() -> None:
    """
    The case that used to yield NOTHING, with no counter saying why.

    Two lots in the same session annotate differently: one gives its dividers
    as thin lines, the other whole bay quads as one solid slab. A slab is not
    a stripe, so `PARKING_STRIPE_MAX_WIDTH_M` rejected every run of it and the
    lot produced no dividers, no bays, and zero of every rejection reason.
    """
    lot = _filled_row(count=6)
    report = ScanReport()

    bays = find_bays(lot, _EMPTY, np.asarray((9.0, -6.0)), report)

    assert report.slabs_found == 1, "the slab must be counted, not silently dropped"
    assert bays, "a filled bay row must still offer bays"
    # The COUNT is a guess -- a filled quad carries no dividers to read it off,
    # and 18 m of frontage divides plausibly several ways -- so the assertion
    # is that every offered bay is a believable size, not that there are six.
    for bay in bays:
        assert PARKING_BAY_WIDTH_MIN_M <= bay.width_m <= PARKING_BAY_WIDTH_MAX_M
        assert PARKING_BAY_MIN_DEPTH_M <= bay.depth_m <= PARKING_BAY_MAX_DEPTH_M


def test_paint_beside_the_row_at_a_different_depth_is_trimmed_off() -> None:
    """
    A hatched keep-clear zone can share a slab with the row it adjoins.

    Reported live: parking in the outermost bay of a row, with the annotation
    running on past it over a chevron area that is not a bay at all. Nothing
    in a solid quad distinguishes them, so what CAN be used is the shape --
    only the contiguous stretch that is one bay deep is divided.
    """
    row = _filled_row(count=4, width=3.0, depth=5.3)
    # A shallower painted area butted onto the end of the row.
    hatch = _filled_row(count=2, width=3.0, depth=2.0, origin=(12.0, 0.0))
    report = ScanReport()

    bays = find_bays(
        np.concatenate((row, hatch)), _EMPTY, np.asarray((6.0, -6.0)), report
    )

    assert bays
    # No bay may sit out over the shallow paint: the row ends at 12 m.
    for bay in bays:
        assert bay.centre[0] < 12.0, "a bay was invented over the hatched area"


def test_widely_spaced_dividers_are_not_read_as_a_filled_row() -> None:
    """
    A wide run is only a bay row when it is actually FILLED.

    Separate rows, and dividers too far apart to bound a bay, project onto the
    same wide band of offsets at some sweep angle -- and reading that as a slab
    does not merely add wrong bays, it CONSUMES the cells every later sweep
    needed. Measured before the fill test: three rows worth 4 + 4 + 3 bays
    apart came back as 4 together.
    """
    sparse = _bay_row(count=4, spacing=9.0, depth=5.0)
    report = ScanReport()

    find_bays(sparse, _EMPTY, np.asarray((0.0, 0.0)), report)

    assert report.slabs_found == 0
