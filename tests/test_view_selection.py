from __future__ import annotations

from beamng_lidar_bev import main_window
from beamng_lidar_bev.main_window import (
    _BRIDGE_WAIT_GRACE_S,
    VIEW_CAMERAS,
    VIEW_RAW_BEV,
    VIEW_WORLD,
    bridge_wait_message,
    controls_offered,
    resolve_sensor_mode,
    resolve_visualization,
)
from beamng_lidar_bev.worker import SENSOR_MODE_LIDAR, SENSOR_MODE_VISION


def test_world_is_the_default_when_the_renderer_is_available() -> None:
    assert resolve_visualization(None, world_available=True) == VIEW_WORLD


def test_raw_bev_selection_is_preserved() -> None:
    assert resolve_visualization(VIEW_RAW_BEV, world_available=True) == VIEW_RAW_BEV


def test_unavailable_world_renderer_forces_raw_bev() -> None:
    assert resolve_visualization(VIEW_WORLD, world_available=False) == VIEW_RAW_BEV


def test_the_camera_grid_is_preserved_while_the_camera_rig_is_streaming() -> None:
    assert (
        resolve_visualization(
            VIEW_CAMERAS, world_available=True, cameras_available=True
        )
        == VIEW_CAMERAS
    )


def test_the_camera_grid_survives_a_broken_world_renderer() -> None:
    """CAMERAS is a plain QWidget grid with no Qt Quick 3D dependency, so a
    broken WORLD renderer must not steal the selection from it."""
    assert (
        resolve_visualization(
            VIEW_CAMERAS, world_available=False, cameras_available=True
        )
        == VIEW_CAMERAS
    )


def test_the_camera_grid_falls_back_when_the_lidar_set_is_the_instrument() -> None:
    """The grid draws the rig's images; the LiDAR set has none. Since the
    view and the instrument set were separated the view follows the set."""
    assert (
        resolve_visualization(
            VIEW_CAMERAS, world_available=True, cameras_available=False
        )
        == VIEW_WORLD
    )
    assert (
        resolve_visualization(
            VIEW_CAMERAS, world_available=False, cameras_available=False
        )
        == VIEW_RAW_BEV
    )


def test_a_persisted_legacy_vision_view_means_the_camera_grid() -> None:
    """Settings written before the split stored "VISION" for the grid."""
    assert (
        resolve_visualization("VISION", world_available=True, cameras_available=True)
        == VIEW_CAMERAS
    )
    assert (
        resolve_visualization("VISION", world_available=True, cameras_available=False)
        == VIEW_WORLD
    )


def test_the_instrument_set_is_its_own_setting() -> None:
    assert resolve_sensor_mode("VISION") == SENSOR_MODE_VISION
    assert resolve_sensor_mode("vision") == SENSOR_MODE_VISION
    assert resolve_sensor_mode("LIDAR") == SENSOR_MODE_LIDAR
    assert resolve_sensor_mode(None) == SENSOR_MODE_LIDAR
    assert resolve_sensor_mode("garbage") == SENSOR_MODE_LIDAR


def test_the_driving_controls_follow_the_workers_vision_gate(
    monkeypatch,
) -> None:
    """The GUI offers exactly what the worker will accept, and nothing the
    worker would bounce: one constant gates both."""
    assert controls_offered(SENSOR_MODE_LIDAR) is True
    monkeypatch.setattr(main_window, "VISION_DRIVING_ENABLED", False)
    assert controls_offered(SENSOR_MODE_VISION) is False
    monkeypatch.setattr(main_window, "VISION_DRIVING_ENABLED", True)
    assert controls_offered(SENSOR_MODE_VISION) is True


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
