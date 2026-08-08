from __future__ import annotations

from typing import Iterator

import pytest

from beamng_lidar_bev import bridge_monitor as monitor_module
from beamng_lidar_bev.bridge_monitor import BridgeMonitor
from beamng_lidar_bev.config import (
    BRIDGE_PROBE_INTERVAL_MS,
    BRIDGE_PROBE_STREAM_INTERVAL_MS,
)


def _scripted(monkeypatch: pytest.MonkeyPatch, results: list[bool]) -> None:
    stream: Iterator[bool] = iter(results)
    monkeypatch.setattr(
        monitor_module,
        "bridge_is_reachable",
        lambda **_kwargs: next(stream),
    )


def _run(monitor: BridgeMonitor, ticks: int) -> tuple[list[str], list[str]]:
    ups: list[str] = []
    downs: list[str] = []
    monitor.bridge_up.connect(lambda: ups.append("up"))
    monitor.bridge_down.connect(lambda: downs.append("down"))
    for _ in range(ticks):
        monitor._probe()
    return ups, downs


def test_signals_fire_on_transitions_not_on_every_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted(monkeypatch, [True] * 5)
    monitor = BridgeMonitor()

    ups, downs = _run(monitor, 5)

    assert ups == ["up"]
    assert downs == []


def test_a_single_missed_probe_does_not_report_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A map reload may briefly refuse connections; one miss must be tolerated.
    _scripted(monkeypatch, [True, False, True, True])
    monitor = BridgeMonitor()

    ups, downs = _run(monitor, 4)

    assert ups == ["up"]
    assert downs == []


def test_consecutive_misses_report_loss_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted(monkeypatch, [True, False, False, False, False])
    monitor = BridgeMonitor()

    ups, downs = _run(monitor, 5)

    assert ups == ["up"]
    assert downs == ["down"]


def test_recovery_after_loss_reports_up_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scripted(monkeypatch, [True, False, False, True])
    monitor = BridgeMonitor()

    ups, downs = _run(monitor, 4)

    assert ups == ["up", "up"]
    assert downs == ["down"]


class _TimerSpy:
    """QTimer.start() is a no-op outside a QThread, so record the intent."""

    def __init__(self) -> None:
        self.intervals: list[int] = []

    def start(self, interval: int) -> None:
        self.intervals.append(interval)


def test_a_failing_probe_does_not_stop_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(**_kwargs: object) -> bool:
        raise OSError("winsock exploded")

    monkeypatch.setattr(monitor_module, "bridge_is_reachable", explode)
    monitor = BridgeMonitor()
    spy = _TimerSpy()
    monitor._timer = spy  # type: ignore[assignment]

    monitor._probe()

    assert spy.intervals == [BRIDGE_PROBE_INTERVAL_MS]


def test_probe_backs_off_once_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, [True])
    monitor = BridgeMonitor()
    spy = _TimerSpy()
    monitor._timer = spy  # type: ignore[assignment]
    monitor.set_streaming(True)

    monitor._probe()

    assert spy.intervals == [BRIDGE_PROBE_STREAM_INTERVAL_MS]
