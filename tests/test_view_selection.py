from __future__ import annotations

from beamng_lidar_bev.main_window import (
    _BRIDGE_WAIT_GRACE_S,
    bridge_wait_message,
    resolve_visualization,
    sensor_mode_for_visualization,
)
from beamng_lidar_bev.worker import SENSOR_MODE_LIDAR, SENSOR_MODE_VISION


def test_world_is_the_default_when_the_renderer_is_available() -> None:
    assert resolve_visualization(None, world_available=True) == "WORLD"


def test_raw_bev_selection_is_preserved() -> None:
    assert resolve_visualization("RAW BEV", world_available=True) == "RAW BEV"


def test_unavailable_world_renderer_forces_raw_bev() -> None:
    assert resolve_visualization("WORLD", world_available=False) == "RAW BEV"


def test_vision_selection_is_preserved() -> None:
    assert resolve_visualization("VISION", world_available=True) == "VISION"


def test_vision_survives_a_broken_world_renderer() -> None:
    """VISION is a plain QWidget grid with no Qt Quick 3D dependency, so a
    broken WORLD renderer must not steal the selection from it."""
    assert resolve_visualization("VISION", world_available=False) == "VISION"


def test_only_the_vision_view_asks_for_the_camera_rig() -> None:
    assert sensor_mode_for_visualization("VISION") == SENSOR_MODE_VISION
    assert sensor_mode_for_visualization("WORLD") == SENSOR_MODE_LIDAR
    assert sensor_mode_for_visualization("RAW BEV") == SENSOR_MODE_LIDAR


def test_a_booting_simulator_is_reported_as_starting_not_as_absent() -> None:
    """
    The reported defect: press Launch, and one second later the badge reads
    "BeamNG.tech is not running" with Launch clickable again -- because the
    monitor's 2 s tick finds the phase back at IDLE while the simulator is
    still booting. It reads as a dead button, and answering it with a second
    click puts two instances on port 64256.

    `BeamNG.tech.exe` is a shim that spawns the real `BeamNG.tech.x64`, so the
    gap between "process alive" and "port listening" is a minute or more.
    """
    assert bridge_wait_message(0.0) is not None
    assert bridge_wait_message(1.0) is not None
    assert bridge_wait_message(60.0) is not None
    assert "starting" in bridge_wait_message(1.0)


def test_the_wait_reports_progress_so_it_does_not_look_frozen() -> None:
    assert "(45s)" in bridge_wait_message(45.0)
    assert "(120s)" in bridge_wait_message(120.4)


def test_the_wait_is_bounded_so_a_failed_launch_can_be_retried() -> None:
    """Holding Launch disabled forever would be the opposite failure."""
    assert bridge_wait_message(_BRIDGE_WAIT_GRACE_S) is None
    assert bridge_wait_message(_BRIDGE_WAIT_GRACE_S + 1.0) is None


def test_the_grace_window_clears_a_first_run_shader_compile() -> None:
    """
    Measured ~60 s warm on this machine; a first run of a new version compiles
    shaders and takes considerably longer, so the window is minutes, not
    seconds. Pinned because shrinking it re-creates the double-instance bug.
    """
    assert _BRIDGE_WAIT_GRACE_S >= 180.0
