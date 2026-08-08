from __future__ import annotations

from beamng_lidar_bev.main_window import resolve_visualization


def test_world_is_the_default_when_the_renderer_is_available() -> None:
    assert resolve_visualization(None, world_available=True) == "WORLD"


def test_raw_bev_selection_is_preserved() -> None:
    assert resolve_visualization("RAW BEV", world_available=True) == "RAW BEV"


def test_unavailable_world_renderer_forces_raw_bev() -> None:
    assert resolve_visualization("WORLD", world_available=False) == "RAW BEV"
