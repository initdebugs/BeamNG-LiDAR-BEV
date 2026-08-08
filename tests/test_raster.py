from __future__ import annotations

import numpy as np

from beamng_lidar_bev.config import OBSTACLE_RGBA, ROAD_RGBA
from beamng_lidar_bev.raster import rasterize_points


def test_rasterizes_road_and_obstacle_colours() -> None:
    road = np.asarray(((0.0, 0.0),), dtype=np.float32)
    obstacles = np.asarray(((10.0, 0.0),), dtype=np.float32)

    image = rasterize_points(road, obstacles, 101, 101)

    assert tuple(image[50, 50]) == ROAD_RGBA
    obstacle_x = round(50 + 10 * (101 / 210))
    assert tuple(image[50, obstacle_x]) == OBSTACLE_RGBA


def test_empty_raster_has_expected_shape() -> None:
    empty = np.empty((0, 2), dtype=np.float32)
    image = rasterize_points(empty, empty, 80, 60)
    assert image.shape == (60, 80, 4)
    assert not image.any()
