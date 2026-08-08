from __future__ import annotations

import numpy as np

from .config import DISPLAY_RADIUS_M, OBSTACLE_RGBA, ROAD_RGBA


def rasterize_points(
    road_points: np.ndarray,
    obstacle_points: np.ndarray,
    width: int,
    height: int,
    radius_m: float = DISPLAY_RADIUS_M,
) -> np.ndarray:
    """Rasterize right/forward BEV points into a transparent RGBA image."""
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 4), dtype=np.uint8)

    image = np.zeros((height, width, 4), dtype=np.uint8)
    scale = min(width, height) / (2.0 * radius_m)
    centre_x = (width - 1) * 0.5
    centre_y = (height - 1) * 0.5

    _paint_points(image, road_points, centre_x, centre_y, scale, ROAD_RGBA, 1)
    _paint_points(
        image,
        obstacle_points,
        centre_x,
        centre_y,
        scale,
        OBSTACLE_RGBA,
        2,
    )
    return image


def _paint_points(
    image: np.ndarray,
    points: np.ndarray,
    centre_x: float,
    centre_y: float,
    scale: float,
    rgba: tuple[int, int, int, int],
    size: int,
) -> None:
    values = np.asarray(points)
    if values.size == 0:
        return
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 BEV array, got {values.shape}")

    px = np.rint(centre_x + values[:, 0] * scale).astype(np.int32)
    py = np.rint(centre_y - values[:, 1] * scale).astype(np.int32)

    offsets = ((0, 0),) if size == 1 else ((0, 0), (-1, 0), (0, -1), (-1, -1))
    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        image[y[inside], x[inside]] = rgba
