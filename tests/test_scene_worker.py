from __future__ import annotations

from types import SimpleNamespace

import pytest

from beamng_lidar_bev.scene_worker import SceneWorker


class AssemblerStub:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.composed: list[object] = []
        self.refreshed: list[object] = []
        self.clear_calls = 0

    def refresh(self, snapshot: object) -> None:
        self.refreshed.append(snapshot)

    def compose(self, snapshot: object) -> str:
        self.composed.append(snapshot)
        if self.error is not None:
            raise self.error
        return f"frame-{snapshot.timestamp}"  # type: ignore[attr-defined]

    def clear(self) -> None:
        self.clear_calls += 1


def test_scene_worker_keeps_only_the_latest_pending_snapshot() -> None:
    assembler = AssemblerStub()
    worker = SceneWorker(assembler=assembler)
    frames: list[object] = []
    worker.world_frame_ready.connect(frames.append)

    older = SimpleNamespace(timestamp=1.0)
    newest = SimpleNamespace(timestamp=2.0)
    worker.submit(older)  # type: ignore[arg-type]
    worker.submit(newest)  # type: ignore[arg-type]
    worker._process_pending()

    assert assembler.composed == [newest]
    assert frames == ["frame-2.0"]


def test_scene_worker_isolates_builder_errors_and_clears_state() -> None:
    assembler = AssemblerStub(error=RuntimeError("broken geometry"))
    worker = SceneWorker(assembler=assembler)
    errors: list[str] = []
    worker.scene_error.connect(errors.append)

    worker.submit(SimpleNamespace(timestamp=3.0))  # type: ignore[arg-type]
    worker._process_pending()

    assert assembler.clear_calls == 1
    assert errors == ["3D scene build failed: broken geometry"]


def test_scene_worker_refreshes_the_stores_on_its_own_clock() -> None:
    assembler = AssemblerStub()
    worker = SceneWorker(assembler=assembler)

    for tick in range(3):
        worker.submit(SimpleNamespace(timestamp=float(tick)))  # type: ignore[arg-type]
        worker._process_pending()

    # Three back-to-back snapshots land inside one refresh interval: the
    # first refreshes the stores (inline, so the first frame carries a
    # world), the ones behind it only compose.
    assert len(assembler.refreshed) == 1
    assert len(assembler.composed) == 3

    worker.clear()
    worker.submit(SimpleNamespace(timestamp=9.0))  # type: ignore[arg-type]
    worker._process_pending()
    assert len(assembler.refreshed) == 2


def test_a_slow_refresh_never_blocks_the_composes_behind_it() -> None:
    """
    The reason the refresh moved to its own thread: run inline, a 48 ms
    refresh stalled the view for its whole duration on every cadence tick.
    """
    import threading

    release = threading.Event()
    started = threading.Event()

    class SlowAssembler(AssemblerStub):
        def refresh(self, snapshot: object) -> None:
            super().refresh(snapshot)
            if len(self.refreshed) > 1:
                started.set()
                release.wait(timeout=5.0)

    assembler = SlowAssembler()
    worker = SceneWorker(assembler=assembler)
    frames: list[object] = []
    worker.world_frame_ready.connect(frames.append)

    # First tick refreshes inline; force the cadence to be due again so the
    # second tick submits the slow ASYNC refresh.
    worker.submit(SimpleNamespace(timestamp=0.0))  # type: ignore[arg-type]
    worker._process_pending()
    worker._last_refresh_at = -float("inf")
    worker.submit(SimpleNamespace(timestamp=1.0))  # type: ignore[arg-type]
    worker._process_pending()
    assert started.wait(timeout=5.0)

    # The refresh is hanging on its own thread; composes keep flowing.
    for tick in (2.0, 3.0):
        worker.submit(SimpleNamespace(timestamp=tick))  # type: ignore[arg-type]
        worker._process_pending()
    assert frames == ["frame-0.0", "frame-1.0", "frame-2.0", "frame-3.0"]

    release.set()
    worker.shutdown()


def test_an_overrunning_refresh_stretches_its_own_cadence() -> None:
    """
    The refresh pool shares the process with the worker tick and the compose
    thread; a build slower than the cadence used to run back-to-back and tax
    both through the GIL. The interval must stretch to last build / duty.
    """
    from beamng_lidar_bev.config import (
        WORLD_STORE_REFRESH_DUTY,
        WORLD_STORE_REFRESH_INTERVAL_S,
    )

    worker = SceneWorker.__new__(SceneWorker)
    worker._last_build_ms = 300.0
    interval = max(
        WORLD_STORE_REFRESH_INTERVAL_S,
        worker._last_build_ms / 1000.0 / WORLD_STORE_REFRESH_DUTY,
    )
    assert interval == pytest.approx(0.5)
    worker._last_build_ms = 30.0
    interval = max(
        WORLD_STORE_REFRESH_INTERVAL_S,
        worker._last_build_ms / 1000.0 / WORLD_STORE_REFRESH_DUTY,
    )
    assert interval == pytest.approx(WORLD_STORE_REFRESH_INTERVAL_S)
