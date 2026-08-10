from __future__ import annotations

import numpy as np

from beamng_lidar_bev.semantics import (
    SCENE_BOUNDARY,
    SCENE_ROAD,
    SCENE_UNKNOWN,
    SCENE_VEHICLE,
    SURFACE_MARKING,
    SURFACE_PAVED,
    SURFACE_SIDEWALK,
    SemanticPalette,
    classify_road_points,
    classify_scene_groups,
    classify_surface_materials,
    pack_rgb,
)


def test_annotation_value_256_uses_uint8_byte_semantics() -> None:
    assert pack_rgb((196, 196, 256)) == pack_rgb((196, 196, 0))


def test_semantic_classes_override_ground_fallback() -> None:
    annotations = {
        "STREET": (255, 0, 0),
        "CAR": (0, 255, 0),
        "BACKGROUND": (16, 16, 16),
        "GRASS": (64, 128, 64),
    }
    palette = SemanticPalette.from_annotations(annotations)
    colours = np.asarray(
        [
            annotations["STREET"],
            annotations["CAR"],
            annotations["BACKGROUND"],
            annotations["BACKGROUND"],
            (4, 5, 6),  # Unknown map annotation.
            annotations["GRASS"],
        ],
        dtype=np.uint8,
    )
    heights = np.asarray((-0.5, -0.5, -0.45, 0.0, -0.5, -0.5))

    mask = classify_road_points(colours, heights, -0.5, palette)

    np.testing.assert_array_equal(mask, (True, False, True, False, True, False))


def test_scene_groups_preserve_road_vehicle_boundary_and_unknown() -> None:
    annotations = {
        "STREET": (1, 2, 3),
        "CAR": (4, 5, 6),
        "GRASS": (7, 8, 9),
        "BACKGROUND": (10, 11, 12),
    }
    palette = SemanticPalette.from_annotations(annotations)
    colours = np.asarray(
        (
            annotations["STREET"],
            annotations["CAR"],
            annotations["GRASS"],
            (13, 14, 15),
        ),
        dtype=np.uint8,
    )

    groups = classify_scene_groups(
        colours,
        np.asarray((0.0, 0.6, 0.0, 0.6), dtype=np.float32),
        0.0,
        palette,
    )

    np.testing.assert_array_equal(
        groups,
        (SCENE_ROAD, SCENE_VEHICLE, SCENE_BOUNDARY, SCENE_UNKNOWN),
    )


def test_scene_road_group_is_exactly_the_road_mask() -> None:
    """
    The worker takes its BEV road/obstacle split from `groups == SCENE_ROAD`
    rather than calling `classify_road_points` a second time, so the two must
    agree point for point. They do because the four code sets are disjoint by
    construction -- `boundary_codes` is defined as everything that is not road,
    fallback, vehicle or vulnerable -- so nothing the road rule accepts can be
    overwritten by the group assignments that follow it.
    """
    annotations = {
        "STREET": (1, 2, 3),
        "ASPHALT": (21, 22, 23),
        "CAR": (4, 5, 6),
        "PEDESTRIAN": (16, 17, 18),
        "GRASS": (7, 8, 9),
        "BACKGROUND": (10, 11, 12),
        "BUILDING": (19, 20, 21),
    }
    palette = SemanticPalette.from_annotations(annotations)
    known = list(annotations.values())
    unknown = [(13, 14, 15), (200, 201, 202)]
    colours = np.asarray(known + unknown, dtype=np.uint8)

    # Sweep the height band so the geometric fallback is exercised on both
    # sides of the ground plane, not just at it.
    for offset in (-2.0, -0.4, -0.05, 0.0, 0.05, 0.4, 2.0):
        heights = np.full(len(colours), offset, dtype=np.float32)
        groups = classify_scene_groups(colours, heights, 0.0, palette)
        road = classify_road_points(colours, heights, 0.0, palette)
        np.testing.assert_array_equal(groups == SCENE_ROAD, road)


def test_road_markings_are_road_to_drive_on_but_paint_to_look_at() -> None:
    """
    Lane lines, arrows and crossings sit in BOTH vocabularies on purpose: the
    road rule must keep accepting them (paint is drivable tarmac, and the road
    store is what draws the surface they lie on) while the material table gives
    them their own colour. The material sets are ordered so MARKING_CLASSES
    wins that second question -- last match takes the cell.
    """
    annotations = {
        "STREET": (255, 0, 0),
        "SOLID_LINE": (255, 196, 128),
        "DASHED_LINE": (196, 196, 256),  # the uint8-wraparound class
        "ZEBRA_CROSSING": (255, 128, 128),
        "DRIVING_INSTRUCTIONS": (132, 226, 244),
        "SIDEWALK": (89, 118, 155),
    }
    palette = SemanticPalette.from_annotations(annotations)
    colours = np.asarray(
        [
            annotations["STREET"],
            annotations["SOLID_LINE"],
            [196, 196, 0],  # DASHED_LINE as the renderer actually emits it
            annotations["ZEBRA_CROSSING"],
            annotations["DRIVING_INSTRUCTIONS"],
            annotations["SIDEWALK"],
        ],
        dtype=np.uint8,
    )
    heights = np.full(len(colours), -0.5)

    road = classify_road_points(colours, heights, -0.5, palette)
    materials = classify_surface_materials(colours, palette)

    np.testing.assert_array_equal(
        road, (True, True, True, True, True, False)
    )
    np.testing.assert_array_equal(
        materials,
        (
            SURFACE_PAVED,
            SURFACE_MARKING,
            SURFACE_MARKING,
            SURFACE_MARKING,
            SURFACE_MARKING,
            SURFACE_SIDEWALK,
        ),
    )
