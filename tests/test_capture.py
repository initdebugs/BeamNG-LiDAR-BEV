"""
The training-capture path: the writer, the label store, and the one piece of
arithmetic between a click and a label.

Offline like the rest of the suite -- no BeamNG.tech, no QApplication. The
session is driven directly and the worker slot is called unbound against a
SimpleNamespace, the idiom `test_worker_state.py` established.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_lidar_bev.capture import (
    BayLabelStore,
    CameraPose,
    CaptureSession,
    EgoPose,
)
from beamng_lidar_bev.config import (
    CAPTURE_QUEUE_DEPTH,
    LABEL_BAY_CORNERS,
    LABEL_DUPLICATE_M,
    LABEL_MAX_RANGE_M,
    LABEL_MAX_ROW_BAYS,
    LABEL_SNAP_M,
    PARKING_BAY_WIDTH_MIN_M,
)
from beamng_lidar_bev.worker import BeamNgWorker

_CAMERA = CameraPose(
    name="a_pillar_left",
    position_vehicle=(0.8, -1.2, 1.0),
    direction_vehicle=(0.3, -0.95, -0.12),
    horizontal_fov_deg=105.0,
    vertical_fov_deg=85.0,
    resolution=(64, 48),
)


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(48, 64, 4), dtype=np.uint8)


def _pose() -> EgoPose:
    return EgoPose(
        pos=(10.0, 20.0, 30.0),
        dir=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        speed_mps=1.5,
    )


def _drain(session: CaptureSession, expected: int, timeout_s: float = 10.0):
    """Wait for the writer, bounded -- a hung writer must fail, not hang."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if session.stats.saved + session.stats.errors >= expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"writer never reached {expected}: {session.stats}")


def test_a_sample_lands_on_disk_as_jpeg_plus_one_index_line(tmp_path: Path):
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    try:
        assert session.offer(_pose(), [("a_pillar_left", _frame())])
        _drain(session, 1)
    finally:
        session.close()

    root = tmp_path / "run"
    written = sorted(p.name for p in (root / "frames").glob("*.jpg"))
    assert written == ["000000_a_pillar_left.jpg"]

    lines = (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["index"] == 0
    assert record["images"] == {
        "a_pillar_left": "frames/000000_a_pillar_left.jpg"
    }
    # The full world frame, not a yaw: this project has been wrong twice about
    # a convention it guessed, so the raw triples are what gets stored.
    assert record["ego"]["pos"] == [10.0, 20.0, 30.0]
    assert record["ego"]["dir"] == [0.0, 1.0, 0.0]
    assert record["ego"]["up"] == [0.0, 0.0, 1.0]

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    # The rig is derived from the VEHICLE, so a dataset that assumed today's
    # constants would mislabel every frame shot in a different car.
    assert meta["cameras"][0]["horizontal_fov_deg"] == 105.0
    assert meta["cameras"][0]["resolution"] == [64, 48]


def test_a_full_queue_drops_the_sample_rather_than_blocking_the_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    The property the whole design hangs on.

    `offer` is called from the 40 ms worker tick, ahead of `_actuate`. A disk
    that cannot keep up must cost frames, never latency -- the same rule the
    camera digest was written to after whole-buffer hashing cost a measured
    9.8 ms of the tick and delayed every control command.
    """
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    # Wedge the writer so nothing can leave the queue.
    monkeypatch.setattr(session, "_write_sample", lambda *_: time.sleep(30))
    try:
        accepted = 0
        started = time.perf_counter()
        for _ in range(CAPTURE_QUEUE_DEPTH + 40):
            if session.offer(_pose(), [("a_pillar_left", _frame())]):
                accepted += 1
        elapsed = time.perf_counter() - started
        # At most the queue plus the one the writer pulled off it.
        assert accepted <= CAPTURE_QUEUE_DEPTH + 1
        assert session.stats.dropped >= 39
        # Nowhere near a tick, let alone the 30 s the writer is asleep for.
        assert elapsed < 1.0
    finally:
        session._closed = True  # the writer is wedged; do not join it


def test_offer_after_close_is_refused_rather_than_queued(tmp_path: Path):
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    session.close()
    assert not session.offer(_pose(), [("a_pillar_left", _frame())])
    # Idempotent: every teardown path in `worker` funnels through one call, and
    # `_cleanup_sensors` runs on re-attach as well as on stop.
    assert session.close().saved == 0


def test_four_corners_make_a_bay_and_the_file_is_valid_json_throughout(
    tmp_path: Path,
):
    store = BayLabelStore(tmp_path / "bays.json")
    corners = [(0.0, 0.0), (2.5, 0.0), (2.5, 5.0), (0.0, 5.0)]
    for corner in corners[:-1]:
        assert store.add_corner(*corner) is None
        assert store.bay_count == 0
    result = store.add_corner(*corners[-1])
    assert result is not None
    assert not result.replaced_indices
    assert sorted(result.bay.corners) == sorted(corners)
    assert store.bay_count == 1
    assert store.pending_count == 0

    payload = json.loads((tmp_path / "bays.json").read_text(encoding="utf-8"))
    assert sorted(map(tuple, payload["bays"][0]["corners"])) == sorted(corners)
    assert not (tmp_path / "bays.tmp").exists()


def test_undo_takes_the_corner_in_progress_before_the_last_whole_bay(
    tmp_path: Path,
):
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]:
        store.add_corner(*corner)
    store.add_corner(9.0, 9.0)
    assert (store.bay_count, store.pending_count) == (1, 1)

    assert store.undo()  # the stray corner
    assert (store.bay_count, store.pending_count) == (1, 0)
    assert store.undo()  # now the bay
    assert store.bay_count == 0
    assert not store.undo()
    assert json.loads((tmp_path / "bays.json").read_text())["bays"] == []


def _labelling_worker(heading_rad: float) -> SimpleNamespace:
    """A worker stub facing `heading_rad` from world +Y, at a known place."""
    right = np.array([math.cos(heading_rad), -math.sin(heading_rad), 0.0])
    forward = np.array([math.sin(heading_rad), math.cos(heading_rad), 0.0])
    return SimpleNamespace(
        _labelling=True,
        _labels=None,
        _last_pose_world=(np.array([100.0, 200.0, 5.0]), right, forward),
        _label_row_bays=1,
        status_changed=SimpleNamespace(emit=lambda *_: None),
        label_progress=SimpleNamespace(emit=lambda *_: None),
        _emit_label_progress=lambda: None,
    )


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, -35.0, 200.0])
def test_a_click_becomes_a_world_point_in_the_car_s_own_frame(
    tmp_path: Path, heading_deg: float
):
    """
    The one piece of arithmetic between a click and a label.

    A sign error here is invisible on screen and silently corrupts every label
    in the dataset, so it is checked against an independently written rotation
    rather than against the implementation's own axes.
    """
    heading = math.radians(heading_deg)
    stub = _labelling_worker(heading)
    stub._labels = BayLabelStore(tmp_path / "bays.json")

    right_m, forward_m = 3.0, 7.0
    BeamNgWorker.add_bay_label(stub, right_m, forward_m)  # type: ignore[arg-type]

    got = stub._labels.pending_corners[0]
    expected = (
        100.0 + right_m * math.cos(heading) + forward_m * math.sin(heading),
        200.0 - right_m * math.sin(heading) + forward_m * math.cos(heading),
    )
    assert got == pytest.approx(expected, abs=1e-9)


def test_a_click_beyond_the_dense_ground_is_refused(tmp_path: Path):
    """
    The raycast will happily answer 150 m out on a surface built from a
    handful of returns. A label there records the accumulated store's error
    rather than the paint's, so it is refused with a message instead.
    """
    stub = _labelling_worker(0.0)
    stub._labels = BayLabelStore(tmp_path / "bays.json")
    messages: list[tuple[str, str]] = []
    stub.status_changed = SimpleNamespace(
        emit=lambda *args: messages.append(args)
    )

    BeamNgWorker.add_bay_label(stub, 0.0, LABEL_MAX_RANGE_M + 0.5)  # type: ignore[arg-type]
    assert stub._labels.pending_count == 0
    assert messages and "away" in messages[0][1]

    BeamNgWorker.add_bay_label(stub, 0.0, LABEL_MAX_RANGE_M - 0.5)  # type: ignore[arg-type]
    assert stub._labels.pending_count == 1


def test_a_click_is_ignored_when_labelling_is_off_or_the_pose_is_unknown(
    tmp_path: Path,
):
    store = BayLabelStore(tmp_path / "bays.json")

    off = _labelling_worker(0.0)
    off._labels = store
    off._labelling = False
    BeamNgWorker.add_bay_label(off, 1.0, 1.0)  # type: ignore[arg-type]

    # Before the first tick there is no pose, so there is nothing to convert
    # WITH -- and inventing one would put the label at the world origin.
    poseless = _labelling_worker(0.0)
    poseless._labels = store
    poseless._last_pose_world = None
    BeamNgWorker.add_bay_label(poseless, 1.0, 1.0)  # type: ignore[arg-type]

    assert store.pending_count == 0


def test_a_bay_needs_exactly_the_configured_number_of_corners(tmp_path: Path):
    """Pins the store against the constant, so a bay can never be half a quad."""
    store = BayLabelStore(tmp_path / "bays.json")
    for index in range(LABEL_BAY_CORNERS - 1):
        assert store.add_corner(float(index), 0.0) is None
    assert store.add_corner(0.0, 1.0) is not None


def _stopping_worker(tmp_path: Path, session, labels) -> SimpleNamespace:
    """A worker stub carrying only what `_stop_capture` may legitimately touch."""
    return SimpleNamespace(
        _capture=session,
        _labels=labels,
        _labelling=True,
        recording_changed=SimpleNamespace(emit=lambda *_: None),
        labelling_changed=SimpleNamespace(emit=lambda *_: None),
        labels_complete_changed=SimpleNamespace(emit=lambda *_: None),
        label_progress=SimpleNamespace(emit=lambda *_: None),
        _emit_label_progress=lambda: None,
        status_changed=SimpleNamespace(emit=lambda *_: None),
    )


def test_stopping_a_session_touches_nothing_the_worker_does_not_own(
    tmp_path: Path,
):
    """
    `_stop_capture` runs inside `_cleanup_sensors`, the single teardown funnel
    every fault path goes through -- so an AttributeError here would surface as
    a lost bridge rather than as the bug it is. Driven against a stub carrying
    ONLY the attributes it may use, which is what catches a reach for state
    that lives on some other object.
    """
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    stub = _stopping_worker(tmp_path, session, BayLabelStore(tmp_path / "b.json"))

    BeamNgWorker._stop_capture(stub, "sensors stopped")  # type: ignore[arg-type]

    assert stub._capture is None
    assert stub._labels is None
    assert not stub._labelling
    # Idempotent: `_cleanup_sensors` also runs on re-attach, so a second pass
    # over an already-closed session must be a no-op rather than a crash.
    BeamNgWorker._stop_capture(stub, "sensors stopped")  # type: ignore[arg-type]


def test_the_teardown_stop_reports_no_badge_and_the_user_stop_does(
    tmp_path: Path,
):
    """
    The badge is an argument because the worker tracks no status of its own,
    and because `_cleanup_sensors` is about to emit READY -- a STREAMING badge
    in front of it would flicker the wrong state over a dead session.
    """
    emitted: list[tuple[str, str]] = []

    def run(badge: str | None) -> None:
        session = CaptureSession(tmp_path / f"run_{badge}", [_CAMERA])
        stub = _stopping_worker(tmp_path, session, None)
        stub.status_changed = SimpleNamespace(
            emit=lambda *args: emitted.append(args)
        )
        BeamNgWorker._stop_capture(stub, "stopped", badge=badge)  # type: ignore[arg-type]

    run(None)
    assert emitted == []
    run("STREAMING")
    assert len(emitted) == 1
    assert emitted[0][0] == "STREAMING"
    assert "Recording stopped" in emitted[0][1]


def _recording_worker(session, position: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        _capture=session,
        _last_capture_at=0.0,
        _last_capture_pos=None,
        _last_capture_log_at=0.0,
        _cameras_fresh_since_capture=True,
        _last_speed=0.0,
        _stop_capture=lambda *_, **__: None,
    )


def _state(x: float, y: float) -> dict:
    return {"pos": (x, y, 0.0), "dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)}


def test_a_parked_car_stops_recording_the_same_view_over_and_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Parking up to label bays is the normal workflow, and at the capture cadence
    a five-minute labelling session would otherwise write ~600 samples of one
    identical picture. Near-duplicates inflate a training set without adding
    anything to it, which is the whole reason `CAPTURE_INTERVAL_S` is as slow
    as it is.
    """
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    offered: list[tuple[float, float]] = []
    monkeypatch.setattr(
        session,
        "offer",
        lambda ego, images: offered.append(ego.pos[:2]) or True,
    )
    stub = _recording_worker(session, (0.0, 0.0))
    images = [SimpleNamespace(name="a_pillar_left", rgba=_frame())]

    # Clear the cadence gate every time, so the only thing under test is travel.
    def tick(x: float, y: float) -> None:
        stub._last_capture_at = -1e6
        stub._cameras_fresh_since_capture = True
        BeamNgWorker._record_sample(stub, _state(x, y), images)  # type: ignore[arg-type]

    tick(0.0, 0.0)  # the first sample of a session always lands
    for _ in range(20):  # parked, labelling
        tick(0.0, 0.0)
    assert offered == [(0.0, 0.0)]

    tick(0.0, 0.1)  # a shuffle, still the same viewpoint
    assert len(offered) == 1

    tick(0.0, 0.4)  # driven on: a new viewpoint
    assert offered[-1] == (0.0, 0.4)
    assert len(offered) == 2
    session.close()


def test_a_held_camera_frame_is_never_paired_with_a_new_pose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    A re-read of a held frame beside a moved pose asserts that the car moved
    and the picture did not -- which is exactly the relation the unprojection
    would be trained on, taught backwards.
    """
    session = CaptureSession(tmp_path / "run", [_CAMERA])
    offered: list[tuple[float, float]] = []
    monkeypatch.setattr(
        session,
        "offer",
        lambda ego, images: offered.append(ego.pos[:2]) or True,
    )
    stub = _recording_worker(session, (0.0, 0.0))
    images = [SimpleNamespace(name="a_pillar_left", rgba=_frame())]

    stub._last_capture_at = -1e6
    BeamNgWorker._record_sample(stub, _state(0.0, 0.0), images)  # type: ignore[arg-type]
    assert len(offered) == 1

    # The camera has not refreshed since, so a moved car records nothing.
    stub._last_capture_at = -1e6
    BeamNgWorker._record_sample(stub, _state(0.0, 5.0), images)  # type: ignore[arg-type]
    assert len(offered) == 1

    stub._cameras_fresh_since_capture = True
    stub._last_capture_at = -1e6
    BeamNgWorker._record_sample(stub, _state(0.0, 5.0), images)  # type: ignore[arg-type]
    assert offered[-1] == (0.0, 5.0)
    session.close()


def _shoelace(corners) -> float:
    total = 0.0
    for index, (x, y) in enumerate(corners):
        nx, ny = corners[(index + 1) % len(corners)]
        total += x * ny - nx * y
    return abs(total) / 2.0


def test_corners_clicked_in_a_z_still_make_a_rectangle(tmp_path: Path):
    """
    A bowtie is the one malformed label a coordinate list looks fine in.

    Measured on the first real session: one bay of twelve came back with a
    shoelace area of 0.2 m2 against a true 17.5 -- every corner in the right
    place, the polygon between them crossing itself. Sorting by bearing about
    the centroid removes the failure rather than detecting it, because a
    parking bay is convex.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    # Bottom-left, bottom-right, TOP-LEFT, top-right: the Z that did it.
    for corner in [(0.0, 0.0), (3.0, 0.0), (0.0, 5.0), (3.0, 5.0)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert _shoelace(result.bay.corners) == pytest.approx(15.0)


def test_re_clicking_a_bay_replaces_it_instead_of_adding_a_second(
    tmp_path: Path,
):
    """
    3 of the first session's 12 labels were re-clicks 0.10-0.16 m apart, so the
    count read 12 over 9 places. A re-click almost always means the first
    attempt was bad, so it replaces rather than being refused.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)
    assert store.bay_count == 1

    # The same bay, clicked a few centimetres out on a second pass.
    for corner in [(0.1, 0.1), (3.1, 0.1), (3.1, 5.1), (0.1, 5.1)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert result.replaced_indices == (0,)
    assert store.bay_count == 1

    # And the corners snapped back to the original, which is deliberate:
    # `LABEL_SNAP_M` IS the labelling precision. The first real session's bays
    # measured 3.08-3.28 m across, so click scatter is already ~0.2 m -- a
    # "correction" smaller than the snap radius is smaller than the noise the
    # labels carry anyway, and inside a row the shared-divider case is the one
    # worth serving. A corner moved further than that does not snap and wins.
    assert sorted(store.bays[0].corners) == sorted(
        [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]
    )
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (-1.2, 5.0)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert result.replaced_indices == (0,)
    assert min(c[0] for c in store.bays[0].corners) == pytest.approx(-1.2)


def test_the_bay_next_door_is_not_treated_as_a_duplicate(tmp_path: Path):
    """The other half of the rule: `LABEL_DUPLICATE_M` sits well under a bay
    width, so a genuine neighbour is never merged into its neighbour."""
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)
    for corner in [(3.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert not result.replaced_indices
    assert store.bay_count == 2
    assert LABEL_DUPLICATE_M < PARKING_BAY_WIDTH_MIN_M


def test_a_row_of_bays_shares_its_dividers_exactly(tmp_path: Path):
    """
    Bay 1's right line IS bay 2's left line, so every interior divider gets
    clicked twice -- and twice at slightly different places, which describes one
    painted line as two. A click near a corner already labelled lands ON it, so
    the shared edge is shared exactly and the second bay can be clicked roughly.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)

    # The neighbour, its shared edge clicked a sloppy 20 cm out both times.
    for corner in [(3.2, 0.15), (6.0, 0.0), (6.0, 5.0), (2.85, 4.9)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert result.snapped_corners == 2

    shared = {c for c in store.bays[0].corners} & {
        c for c in store.bays[1].corners
    }
    assert shared == {(3.0, 0.0), (3.0, 5.0)}
    # And the bay is still the right shape -- snapping must not distort it.
    assert _shoelace(result.bay.corners) == pytest.approx(15.0)


def test_snapping_never_welds_two_genuinely_distinct_corners(tmp_path: Path):
    """The other half: `LABEL_SNAP_M` sits well under the closest real corner
    spacing, which is one bay width."""
    assert LABEL_SNAP_M < PARKING_BAY_WIDTH_MIN_M / 2.0
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)
    # A corner a whole bay away is a different corner, however the row runs.
    for corner in [(6.0, 0.0), (9.0, 0.0), (9.0, 5.0), (6.0, 5.0)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert result.snapped_corners == 0


def test_abandoning_a_quad_forgets_its_snap_tally_too(tmp_path: Path):
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)
    store.add_corner(0.05, 0.05)  # snaps, then is abandoned
    assert store.undo()
    for corner in [(9.0, 0.0), (12.0, 0.0), (12.0, 5.0), (9.0, 5.0)]:
        result = store.add_corner(*corner)
    assert result is not None
    assert result.snapped_corners == 0


def test_one_quad_can_be_a_whole_row_of_bays(tmp_path: Path):
    """
    The case that made this necessary: a lot whose annotation covers whole bay
    quads rather than the divider lines. The interior dividers are not in the
    sensor data at all -- WORLD draws one solid slab -- so there is nothing to
    click. The slab's OUTLINE is visible and the count is readable from the
    simulator's own window, and between them the row is fully determined.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    # A 9-bay row: 28.8 m along the kerb, 5.5 m deep.
    for corner in [(0.0, 0.0), (28.8, 0.0), (28.8, 5.5), (0.0, 5.5)]:
        result = store.add_corner(*corner, row_bays=9)
    assert result is not None
    assert store.bay_count == 9
    for bay in result.bays:
        sides = [
            math.dist(bay.corners[i], bay.corners[(i + 1) % 4])
            for i in range(4)
        ]
        assert sorted(sides)[:2] == pytest.approx([3.2, 3.2])
        assert sorted(sides)[2:] == pytest.approx([5.5, 5.5])
    # Adjacent bays share their divider exactly, by construction rather than by
    # snapping -- they were cut from one quad.
    centres = sorted(bay.centre[0] for bay in result.bays)
    assert np.allclose(np.diff(centres), 3.2)


def test_the_split_axis_is_chosen_by_SIZE_not_by_which_side_is_longer(
    tmp_path: Path,
):
    """
    The obvious rule -- divide the longer side -- is wrong, and a real row
    proves it: 7.5 m across by 5.0 m deep is three 2.5 m bays, and the longer
    side happens to be the one that must be divided here, but at other counts
    it is not. What decides is which orientation yields bays the DETECTOR would
    accept: slicing a bay's depth three ways leaves something 1.67 m wide, well
    under `PARKING_BAY_WIDTH_MIN_M`.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (7.5, 0.0), (7.5, 5.0), (0.0, 5.0)]:
        result = store.add_corner(*corner, row_bays=3)
    assert result is not None
    assert store.bay_count == 3
    for bay in result.bays:
        sides = sorted(
            math.dist(bay.corners[i], bay.corners[(i + 1) % 4])
            for i in range(4)
        )
        assert sides[:2] == pytest.approx([2.5, 2.5])
        assert sides[2:] == pytest.approx([5.0, 5.0])


@pytest.mark.parametrize(
    "across,deep", [(4.8, 5.5), (6.0, 5.0), (6.4, 5.5)]
)
def test_two_bays_is_ambiguous_and_is_refused_rather_than_guessed(
    tmp_path: Path, across: float, deep: float
):
    """
    Measured over the detector's bounds: at TWO bays, these quads divide two
    ways that are each entirely plausible, and no tie-break survives contact --
    prefer the longer side and 4.8x5.5 goes wrong, prefer the squarer bays and
    6.0x5.0 does. Guessing would invent a pair of bays that are not the ones
    painted on the ground, so the quad is refused and says why. Two bays is
    eight clicks.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    for corner in [(0.0, 0.0), (across, 0.0), (across, deep), (0.0, deep)]:
        result = store.add_corner(*corner, row_bays=2)
    assert result is None
    assert store.bay_count == 0
    assert "either way" in (store.split_refusal or "")
    # And the corners survive, so only the count has to change.
    assert store.pending_count == LABEL_BAY_CORNERS


def test_a_row_count_that_does_not_fit_is_refused_not_invented(tmp_path: Path):
    """
    A wrong count must cost a retry, never a plausible-looking set of bays that
    are not the ones painted on the ground. The corners are KEPT, because it is
    the number that was wrong and not the clicking.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    # 28.8 m of row divided 3 ways is 9.6 m per bay -- no bay is that wide.
    for corner in [(0.0, 0.0), (28.8, 0.0), (28.8, 5.5), (0.0, 5.5)]:
        result = store.add_corner(*corner, row_bays=3)
    assert result is None
    assert store.bay_count == 0
    assert "do not fit" in (store.split_refusal or "")
    assert store.pending_count == LABEL_BAY_CORNERS

    # The corners survive, so fixing the count is all that is needed... after
    # abandoning the quad, which is what `undo` is for.
    assert store.undo()
    assert store.pending_count == 0


def test_a_row_of_one_is_byte_identical_to_labelling_a_single_bay(
    tmp_path: Path,
):
    """The default path must be untouched by the row tool existing."""
    corners = [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]
    plain = BayLabelStore(tmp_path / "plain.json")
    rowed = BayLabelStore(tmp_path / "rowed.json")
    for corner in corners:
        plain.add_corner(*corner)
        rowed.add_corner(*corner, row_bays=1)
    assert plain.bays[0].corners == rowed.bays[0].corners
    assert LABEL_MAX_ROW_BAYS >= 9


def test_a_session_is_partial_until_it_is_marked_complete(tmp_path: Path):
    """
    The flag that decides what the training data MEANS.

    Partial labelling is why `build_dataset` has to ignore a third of every
    mask, and the ignored third is exactly the tarmac where the unlabelled bays
    are -- so the model is never told that any particular patch of tarmac is
    NOT a bay. Marked complete, all of it becomes supervised background.
    """
    store = BayLabelStore(tmp_path / "bays.json")
    assert not store.complete
    for corner in [(0.0, 0.0), (3.0, 0.0), (3.0, 5.0), (0.0, 5.0)]:
        store.add_corner(*corner)
    # Absent means partial, so every session recorded before the flag existed
    # reads as what it actually was.
    assert "complete" in json.loads((tmp_path / "bays.json").read_text())
    assert json.loads((tmp_path / "bays.json").read_text())["complete"] is False

    store.set_complete(True)
    assert store.complete
    assert json.loads((tmp_path / "bays.json").read_text())["complete"] is True
    # And it survives further labelling, which is when it would matter.
    for corner in [(3.0, 0.0), (6.0, 0.0), (6.0, 5.0), (3.0, 5.0)]:
        store.add_corner(*corner)
    assert json.loads((tmp_path / "bays.json").read_text())["complete"] is True

    store.set_complete(False)
    assert not store.complete


def test_marking_complete_without_a_session_is_refused_and_says_so(
    tmp_path: Path,
):
    """Same rule as labelling: the claim is stored beside the frames it
    describes, so there has to be somewhere to store it."""
    messages: list[tuple[str, str]] = []
    stub = SimpleNamespace(
        _labels=None,
        labels_complete_changed=SimpleNamespace(emit=lambda *_: None),
        status_changed=SimpleNamespace(
            emit=lambda *args: messages.append(args)
        ),
    )
    BeamNgWorker.set_labels_complete(stub, True)  # type: ignore[arg-type]
    assert messages and "Start recording" in messages[0][1]
