from __future__ import annotations

from types import SimpleNamespace

from beamng_lidar_bev.scene_worker import SceneWorker


class AssemblerStub:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.snapshots: list[object] = []
        self.refreshes: list[bool] = []
        self.clear_calls = 0

    def update(self, snapshot: object, *, refresh_stores: bool = True) -> str:
        self.snapshots.append(snapshot)
        self.refreshes.append(refresh_stores)
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

    assert assembler.snapshots == [newest]
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
