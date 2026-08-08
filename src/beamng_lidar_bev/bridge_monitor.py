from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

from .config import (
    BRIDGE_LOSS_CONFIRMATIONS,
    BRIDGE_PROBE_INTERVAL_MS,
    BRIDGE_PROBE_STREAM_INTERVAL_MS,
    BRIDGE_PROBE_TIMEOUT_S,
)
from .launcher import bridge_is_reachable

LOGGER = logging.getLogger(__name__)


class BridgeMonitor(QObject):
    """
    Edge-triggered liveness probe for the BeamNG.tech communication bridge.

    Lives on its own QThread. The GUI thread would stutter on a blocking
    connect, and the worker thread is already saturated by the 33 ms poll loop
    and would serialise the probe behind a multi-second attach.

    Signals fire only on transitions, never once per probe, so the UI can
    connect them directly without debouncing.
    """

    bridge_up = pyqtSignal()
    bridge_down = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._is_up: bool | None = None
        self._misses = 0
        self._idle = True
        self._timer = QTimer(self)
        # Single-shot and rearmed by the probe itself, so a slow probe can never
        # stack ticks behind itself.
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.timeout.connect(self._probe)

    @pyqtSlot()
    def start(self) -> None:
        self._timer.start(0)

    @pyqtSlot()
    def stop(self) -> None:
        self._timer.stop()

    @pyqtSlot(bool)
    def set_streaming(self, streaming: bool) -> None:
        """Back the probe off once attached; it is only a death detector then."""
        self._idle = not streaming

    @pyqtSlot()
    def _probe(self) -> None:
        try:
            if bridge_is_reachable(timeout_s=BRIDGE_PROBE_TIMEOUT_S):
                self._misses = 0
                if self._is_up is not True:
                    self._is_up = True
                    self.bridge_up.emit()
            else:
                self._misses += 1
                # Debounced: a map reload may briefly refuse connections.
                if (
                    self._is_up is not False
                    and self._misses >= BRIDGE_LOSS_CONFIRMATIONS
                ):
                    self._is_up = False
                    self.bridge_down.emit()
        except Exception:
            LOGGER.debug("Bridge probe failed", exc_info=True)
        finally:
            self._timer.start(
                BRIDGE_PROBE_INTERVAL_MS
                if self._idle
                else BRIDGE_PROBE_STREAM_INTERVAL_MS
            )
