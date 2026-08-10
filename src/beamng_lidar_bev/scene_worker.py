from __future__ import annotations

import logging
import threading
import time
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
    """Latest-frame-only scene construction isolated from BeamNG I/O."""

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
        # When the stores were last refreshed. -inf so the first snapshot after
        # construction or clear() always runs a full build.
        self._last_refresh_at = -float("inf")

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

        # Two rates on one thread: the stores refresh on their own clock, and
        # every snapshot in between re-presents the cached world meshes into
        # its own ego frame -- which is the cheap part, and the part that must
        # track the car or the whole scene visibly lags. See
        # WORLD_STORE_REFRESH_INTERVAL_S.
        started = time.perf_counter()
        refresh = (
            started - self._last_refresh_at >= WORLD_STORE_REFRESH_INTERVAL_S
        )
        try:
            frame = self._assembler.update(snapshot, refresh_stores=refresh)
        except Exception as exc:
            LOGGER.exception("3D scene build failed")
            self._assembler.clear()
            self.scene_error.emit(f"3D scene build failed: {exc}")
        else:
            if refresh:
                self._last_refresh_at = started
                # SCENE BUILD keeps meaning "the store refresh", which is the
                # figure the over-budget warning watches; compose-only ticks
                # are a few milliseconds and reporting them would bury it.
                self.build_time_changed.emit(
                    (time.perf_counter() - started) * 1000.0
                )
            self.world_frame_ready.emit(frame)
        with self._mailbox_lock:
            schedule = self._pending is not None and not self._stopped
            if not schedule:
                self._scheduled = False
        if schedule:
            self._schedule()

    @pyqtSlot()
    def clear(self) -> None:
        with self._mailbox_lock:
            self._pending = None
        self._assembler.clear()
        self._last_refresh_at = -float("inf")

    @pyqtSlot()
    def shutdown(self) -> None:
        with self._mailbox_lock:
            self._stopped = True
            self._pending = None
            self._scheduled = False
        self._assembler.clear()
