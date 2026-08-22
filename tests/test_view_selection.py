from __future__ import annotations

from beamng_lidar_bev.main_window import (
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
