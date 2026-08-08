from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import (
    GROUND_FALLBACK_ABOVE_M,
    GROUND_FALLBACK_BELOW_M,
    GROUND_FALLBACK_CLASSES,
    ROAD_CLASSES,
)

SCENE_ROAD = np.uint8(0)
SCENE_VEHICLE = np.uint8(1)
SCENE_VULNERABLE = np.uint8(2)
SCENE_BOUNDARY = np.uint8(3)
SCENE_UNKNOWN = np.uint8(4)

_VEHICLE_CLASSES = frozenset(
    {
        "BUS",
        "CAR",
        "MOTORCYCLE",
        "PICKUP",
        "TRAILER",
        "TRUCK",
        "VAN",
    }
)
_VULNERABLE_CLASSES = frozenset(
    {
        "BICYCLE",
        "CYCLIST",
        "PEDESTRIAN",
        "PERSON",
    }
)


def _normalise_rgb(rgb: Sequence[int]) -> tuple[int, int, int]:
    # BeamNG 0.37's annotations.json contains one value of 256. The renderer
    # stores channels as uint8, so normalise it using the same byte semantics.
    return tuple(int(channel) % 256 for channel in rgb[:3])  # type: ignore[return-value]


def pack_rgb(rgb: Sequence[int]) -> int:
    red, green, blue = _normalise_rgb(rgb)
    return (red << 16) | (green << 8) | blue


def pack_rgb_rows(colours: np.ndarray) -> np.ndarray:
    values = np.asarray(colours, dtype=np.uint32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected an Nx3 colour array, got {values.shape}")
    return (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]


@dataclass(frozen=True)
class SemanticPalette:
    road_codes: np.ndarray
    known_codes: np.ndarray
    fallback_codes: np.ndarray
    vehicle_codes: np.ndarray
    vulnerable_codes: np.ndarray
    boundary_codes: np.ndarray

    @classmethod
    def from_annotations(
        cls, annotations: Mapping[str, Sequence[int]]
    ) -> "SemanticPalette":
        road_codes = [
            pack_rgb(rgb)
            for name, rgb in annotations.items()
            if name.upper() in ROAD_CLASSES
        ]
        fallback_codes = [
            pack_rgb(rgb)
            for name, rgb in annotations.items()
            if name.upper() in GROUND_FALLBACK_CLASSES
        ]
        vehicle_codes = [
            pack_rgb(rgb)
            for name, rgb in annotations.items()
            if name.upper() in _VEHICLE_CLASSES
        ]
        vulnerable_codes = [
            pack_rgb(rgb)
            for name, rgb in annotations.items()
            if name.upper() in _VULNERABLE_CLASSES
        ]
        boundary_codes = [
            pack_rgb(rgb)
            for name, rgb in annotations.items()
            if name.upper()
            not in ROAD_CLASSES
            | GROUND_FALLBACK_CLASSES
            | _VEHICLE_CLASSES
            | _VULNERABLE_CLASSES
        ]
        if not road_codes:
            # Without this the isin() below is all-False and the entire BEV
            # renders red, which looks like a sensor fault rather than a
            # palette problem.
            raise ValueError(
                "BeamNG's annotation palette contains none of the expected road "
                f"classes ({', '.join(sorted(ROAD_CLASSES))}). "
                f"Got {len(annotations)} annotations."
            )
        return cls(
            road_codes=np.asarray(sorted(set(road_codes)), dtype=np.uint32),
            known_codes=np.asarray(
                sorted({pack_rgb(rgb) for rgb in annotations.values()}),
                dtype=np.uint32,
            ),
            fallback_codes=np.asarray(sorted(set(fallback_codes)), dtype=np.uint32),
            vehicle_codes=np.asarray(sorted(set(vehicle_codes)), dtype=np.uint32),
            vulnerable_codes=np.asarray(
                sorted(set(vulnerable_codes)), dtype=np.uint32
            ),
            boundary_codes=np.asarray(sorted(set(boundary_codes)), dtype=np.uint32),
        )


def _checked_heights(
    colours: np.ndarray, heights_vehicle: np.ndarray
) -> np.ndarray:
    heights = np.asarray(heights_vehicle, dtype=np.float32).reshape(-1)
    if len(colours) != len(heights):
        raise ValueError("Point and semantic-colour counts do not match")
    return heights


def _road_mask(
    packed: np.ndarray,
    heights: np.ndarray,
    ground_z_vehicle: float,
    palette: SemanticPalette,
) -> np.ndarray:
    """
    The road rule, over an already-packed cloud.

    Split out so `classify_scene_groups` can share one `pack_rgb_rows` pass
    with it instead of re-packing and re-classifying the whole cloud. Both are
    O(cloud) and the caller wants both every tick.
    """
    road = np.isin(packed, palette.road_codes)
    known = np.isin(packed, palette.known_codes)
    fallback_label = np.isin(packed, palette.fallback_codes)
    ground_band = (heights >= ground_z_vehicle - GROUND_FALLBACK_BELOW_M) & (
        heights <= ground_z_vehicle + GROUND_FALLBACK_ABOVE_M
    )
    return road | (ground_band & (~known | fallback_label))


def classify_road_points(
    colours: np.ndarray,
    heights_vehicle: np.ndarray,
    ground_z_vehicle: float,
    palette: SemanticPalette,
) -> np.ndarray:
    """
    Return a mask for drivable/road returns.

    Semantic labels take precedence. A narrow geometric ground band is used
    only for unknown or BACKGROUND labels so unannotated community maps still
    produce a useful BEV without turning known grass/sidewalk classes grey.
    """
    heights = _checked_heights(colours, heights_vehicle)
    return _road_mask(pack_rgb_rows(colours), heights, ground_z_vehicle, palette)


def classify_scene_groups(
    colours: np.ndarray,
    heights_vehicle: np.ndarray,
    ground_z_vehicle: float,
    palette: SemanticPalette,
) -> np.ndarray:
    """
    Classify semantic returns into the small vocabulary used by WORLD.

    `groups == SCENE_ROAD` is the same mask `classify_road_points` returns, so
    the worker takes it from here rather than paying for a second pass -- the
    four code sets are disjoint by construction (`boundary_codes` is defined as
    everything not road, fallback, vehicle or vulnerable), so nothing the road
    rule accepts can be overwritten below. `test_semantics.py` pins it.
    """
    heights = _checked_heights(colours, heights_vehicle)
    packed = pack_rgb_rows(colours)
    groups = np.full(len(packed), SCENE_UNKNOWN, dtype=np.uint8)
    groups[_road_mask(packed, heights, ground_z_vehicle, palette)] = SCENE_ROAD

    vehicle = np.isin(packed, palette.vehicle_codes)
    vulnerable = np.isin(packed, palette.vulnerable_codes)
    boundary = np.isin(packed, palette.boundary_codes)
    groups[boundary] = SCENE_BOUNDARY
    groups[vulnerable] = SCENE_VULNERABLE
    groups[vehicle] = SCENE_VEHICLE
    return groups
