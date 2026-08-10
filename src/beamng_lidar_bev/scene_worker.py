from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

from PyQt6.QtCore import (
    QCoreApplication,
    QMetaObject,
    QObject,
    Qt,
    pyqtSignal,
    pyqtSlot,
)

from .config import WORLD_STORE_REFRESH_INTERVAL_S
from .models import PerceptionSnapshot
from .world_scene import WorldSceneAssembler

LOGGER = logging.getLogger(__name__)


class SceneWorker(QObject):
    """
    Latest-frame-only scene construction isolated from BeamNG I/O.

    Two rates on TWO threads. Every snapshot is composed on this worker's own
    thread -- re-projecting the cached world meshes into the current ego frame,
    a few milliseconds -- while the store refresh (folding clouds in and
    rebuilding the meshes, tens of ms) runs on a one-thread pool at
    WORLD_STORE_REFRESH_INTERVAL_S. Composes therefore never wait on a
    refresh: before this split the refresh ran inline and stalled the view for
    its whole duration on every cadence tick, which read as a rhythmic hitch.

    The safety argument is confinement, not locking (see
    `WorldSceneAssembler.refresh`): the stores belong to the refresh pool
    (single-flight), the view state to this thread, and the only shared value
    is the immutable mesh cache, swapped in one assignment. The first build
    after construction or `clear` runs INLINE so the first frame carries a
    world rather than a blank.
    """

    world_frame_ready = pyqtSignal(object)
    scene_error = pyqtSignal(str)
    build_time_changed = pyqtSignal(float)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        assembler: WorldSceneAssembler | Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._assembler = assembler or WorldSceneAssembler()
        self._pending: PerceptionSnapshot | None = None
        self._scheduled = False
        self._stopped = False
        self._mailbox_lock = threading.Lock()
        self._refresh_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scene-refresh"
        )
        self._refresh_future: Future[float] | None = None
        # -inf so the first snapshot after construction or clear() always
        # refreshes; the flag makes that first refresh synchronous.
        self._last_refresh_at = -float("inf")
        self._refresh_inline = True

    @pyqtSlot(object)
    def submit(self, snapshot: PerceptionSnapshot) -> None:
        schedule = False
        with self._mailbox_lock:
            if self._stopped:
                return
            self._pending = snapshot
            if not self._scheduled:
                self._scheduled = True
                schedule = True
        if schedule:
            self._schedule()

    def _schedule(self) -> None:
        if QCoreApplication.instance() is not None:
            QMetaObject.invokeMethod(
                self,
                "_process_pending",
                Qt.ConnectionType.QueuedConnection,
            )

    def _timed_refresh(self, snapshot: PerceptionSnapshot) -> float:
        started = time.perf_counter()
        self._assembler.refresh(snapshot)
        return (time.perf_counter() - started) * 1000.0

    def _collect_refresh(self) -> None:
        """Reap a finished async refresh: report its time, surface its errors."""
        future = self._refresh_future
        if future is None or not future.done():
            return
        self._refresh_future = None
        try:
            self.build_time_changed.emit(future.result())
        except Exception as exc:
            LOGGER.exception("3D scene store refresh failed")
            # No refresh is in flight now, so clearing from this thread is
            # safe; the stores may be half-written and cannot be trusted.
            self._assembler.clear()
            self._refresh_inline = True
            self.scene_error.emit(f"3D scene build failed: {exc}")

    @pyqtSlot()
    def _process_pending(self) -> None:
        with self._mailbox_lock:
            snapshot = self._pending
            self._pending = None
            stopped = self._stopped
            if snapshot is None or stopped:
                self._scheduled = False
        if snapshot is None or stopped:
            return

        self._collect_refresh()
        now = time.perf_counter()
        try:
            if (
                now - self._last_refresh_at >= WORLD_STORE_REFRESH_INTERVAL_S
                and self._refresh_future is None
            ):
                self._last_refresh_at = now
                if self._refresh_inline:
                    # First world after construction or clear: build it here
                    # so this very frame carries it.
                    self.build_time_changed.emit(self._timed_refresh(snapshot))
                    self._refresh_inline = False
                else:
                    self._refresh_future = self._refresh_pool.submit(
                        self._timed_refresh, snapshot
                    )
            frame = self._assembler.compose(snapshot)
        except Exception as exc:
            LOGGER.exception("3D scene build failed")
            self._drop_refresh()
            self._assembler.clear()
            self._refresh_inline = True
            self.scene_error.emit(f"3D scene build failed: {exc}")
        else:
            self.world_frame_ready.emit(frame)
        with self._mailbox_lock:
            schedule = self._pending is not None and not self._stopped
            if not schedule:
                self._scheduled = False
        if schedule:
            self._schedule()

    def _drop_refresh(self) -> None:
        """
        Wait out any in-flight refresh before touching the stores from here.

        Bounded, because the refresh is pure numpy with no I/O -- if it has
        not finished in two seconds something is deeply wrong and abandoning
        it beats hanging a teardown.
        """
        future = self._refresh_future
        self._refresh_future = None
        if future is not None and not future.cancel():
            futures_wait((future,), timeout=2.0)

    @pyqtSlot()
    def clear(self) -> None:
        with self._mailbox_lock:
            self._pending = None
        self._drop_refresh()
        self._assembler.clear()
        self._last_refresh_at = -float("inf")
        self._refresh_inline = True

    @pyqtSlot()
    def shutdown(self) -> None:
        with self._mailbox_lock:
            self._stopped = True
            self._pending = None
            self._scheduled = False
        self._drop_refresh()
        self._refresh_pool.shutdown(wait=False)
        self._assembler.clear()
