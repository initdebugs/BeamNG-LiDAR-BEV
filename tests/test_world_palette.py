from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from beamng_lidar_bev import config
from beamng_lidar_bev.world_scene import (
    _AIR_LINEAR,
    _BOUNDARY_LINEAR,
    _BOUNDARY_LIT_LINEAR,
    _ROAD_LINEAR,
    depth_mix,
    face_shades,
    linear_rgb,
)

QML = (
    Path(__file__).parents[1]
    / "src"
    / "beamng_lidar_bev"
    / "qml"
    / "WorldScene.qml"
)

_LUMINANCE_COEFFICIENTS = np.asarray((0.2126, 0.7152, 0.0722))

# Pairs that have to separate, the floor each needs, and why it matters.
# 3:1 is the WCAG minimum for a graphical object; the lower floors are where
# hue, face shading and depth carry the separation instead, which is a
# deliberate choice and is argued in the QML comment.
REQUIRED = (
    ("road", "air", 3.0, "the drivable surface against empty air"),
    ("boundary", "road", 3.0, "where the drivable surface ends"),
    ("boundary", "air", 3.0, "a building silhouetted against the sky"),
    ("ego", "road", 3.0, "the car itself"),
    ("alert", "air", 3.0, "the AEB alert"),
    ("actor", "road", 2.0, "traffic on the road ahead"),
    ("boundary_lit", "boundary", 1.8, "the top of a slab against its own sides"),
    ("path", "road", 1.8, "the planned path over the road"),
    ("boundary_lit", "road", 1.5, "a slab roof against the surface below it"),
    ("actor", "boundary", 1.5, "a moving car against a static wall"),
)


def luminance(linear: np.ndarray) -> float:
    return float(np.asarray(linear, dtype=np.float64) @ _LUMINANCE_COEFFICIENTS)


def contrast_ratio(first: np.ndarray, second: np.ndarray) -> float:
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _qml_source() -> str:
    return QML.read_text(encoding="utf-8")


def _lit_material_colour(name: str) -> str:
    """Read a LIT material's colour from the QML, where it really lives."""
    block = re.search(
        rf"id:\s*{name}Material\b(?P<body>.*?)\n\s{{8}}\}}",
        _qml_source(),
        re.DOTALL,
    )
    assert block is not None, f"no {name}Material in WorldScene.qml"
    found = re.search(
        r'(?:diffuseColor|baseColor):\s*"(#[0-9a-fA-F]{6})"', block.group("body")
    )
    assert found is not None, f"{name}Material has no literal colour"
    return found.group(1)


def _palette() -> dict[str, np.ndarray]:
    """
    The colours actually shipped, from wherever each one really lives.

    Unlit surfaces are vertex-coloured, so their palette is in config.py; the
    lit ones (ego, actors) keep theirs in the QML where the shading path uses
    them. Reading both is the point -- a copy of either would let them drift.
    """
    return {
        "air": _AIR_LINEAR,
        "road": _ROAD_LINEAR,
        "boundary": _BOUNDARY_LINEAR,
        "boundary_lit": _BOUNDARY_LIT_LINEAR,
        "uncertain": linear_rgb(config.WORLD_UNCERTAIN_RGB),
        "path": linear_rgb(config.WORLD_PATH_RGB),
        "alert": linear_rgb(config.WORLD_PATH_ALERT_RGB),
        "ego": linear_rgb(_lit_material_colour("ego")),
        "actor": linear_rgb(_lit_material_colour("actor")),
    }


def test_the_qml_clear_colour_is_the_air_the_depth_tint_mixes_toward() -> None:
    """
    The air colour is now in two places and they have to agree.

    QML owns `clearColor` because that is what paints the background, but
    `depth_tint` mixes distant geometry toward `config.WORLD_AIR_RGB`. If those
    drift apart, far geometry fades toward a colour that is not the background
    and the horizon grows a visible band instead of dissolving.
    """
    found = re.search(r'clearColor:\s*"(#[0-9a-fA-F]{6})"', _qml_source())
    assert found is not None, "no clearColor in WorldScene.qml"
    assert found.group(1).lower() == config.WORLD_AIR_RGB.lower()


def test_the_unlit_materials_carry_no_colour_of_their_own() -> None:
    """
    Every vertex-coloured material must ship white, or it tints the palette.

    The GPU multiplies base by vertex colour, so a leftover `#6a7176` base on
    the road would square the road colour and land it at `#3f4448` -- dark
    enough to lose the road-vs-boundary step entirely, and silent.
    """
    source = _qml_source()
    for name in ("road", "boundary", "path", "uncertain"):
        block = re.search(
            rf"id:\s*{name}Material\b(?P<body>.*?)\n\s{{8}}\}}", source, re.DOTALL
        )
        assert block is not None, f"no {name}Material in WorldScene.qml"
        body = block.group("body")
        assert 'diffuseColor: "#ffffff"' in body, f"{name}Material is not white"
        assert "vertexColorsEnabled: true" in body, (
            f"{name}Material ships white but does not read vertex colours, so it "
            f"would render as a white silhouette"
        )


@pytest.mark.parametrize(("first", "second", "floor", "why"), REQUIRED)
def test_world_palette_separates_what_has_to_separate(
    first: str, second: str, floor: float, why: str
) -> None:
    """
    Recompute every contrast claim against the colours really shipped.

    This test exists because the ratios were once written into the QML and
    CLAUDE.md from an estimate rather than a calculation, and the estimate was
    wrong in a way that mattered: actors and boundaries shipped at 1.05:1, so a
    traffic car and a wall were the same colour. A number in a comment is not
    evidence; this is.
    """
    palette = _palette()
    ratio = contrast_ratio(palette[first], palette[second])
    assert ratio >= floor, (
        f"{first} vs {second} is {ratio:.2f}:1, under the {floor}:1 needed "
        f"for {why}"
    )


def test_air_is_the_palest_thing_in_the_scene() -> None:
    """
    The whole point of the light-inverted ramp. Before it, empty air was the
    BRIGHTEST surface drawn, which is what put a building against the sky at
    1.35:1 -- the thing that reads as "no contrast between air and obstacles".
    """
    palette = _palette()
    air = luminance(palette["air"])
    for name, colour in palette.items():
        if name in {"air", "path"}:  # the path is a guidance overlay, not a surface
            continue
        assert luminance(colour) < air, f"{name} is no darker than air"


def test_obstacles_sit_below_the_road_in_luminance() -> None:
    """Solid things are darker than the surface you drive on, without exception."""
    palette = _palette()
    road = luminance(palette["road"])
    for name in ("boundary", "boundary_lit", "actor", "ego"):
        assert luminance(palette[name]) < road, f"{name} is lighter than road"


def _hazed(colour: np.ndarray, distance_m: float) -> np.ndarray:
    mix = depth_mix(np.asarray([distance_m]))[0]
    return colour + mix * (_AIR_LINEAR - colour)


@pytest.mark.parametrize(
    ("near_m", "far_m", "floor"),
    ((10.0, 18.0, 1.8), (12.0, 20.0, 1.6), (15.0, 25.0, 1.5), (20.0, 30.0, 1.35)),
)
def test_the_same_surface_reads_differently_near_and_far(
    near_m: float, far_m: float, floor: float
) -> None:
    """
    The reported symptom, as a number: a wall standing in front of a building
    was the same colour as the building, so the two read as one blob.

    Depth is what actually separates them -- face shading gives a slab its
    creases but cannot distinguish two slabs of the same orientation.

    The floors FALL with range on purpose. Extinction saturates, so a fixed
    separation buys less contrast the further out it happens, and that is the
    correct behaviour rather than a shortfall: two buildings 10 m apart at 100 m
    genuinely are harder to tell apart than two 10 m apart at 15 m, and the
    sampling out there is sparser too.
    """
    ratio = contrast_ratio(
        _hazed(_BOUNDARY_LINEAR, near_m), _hazed(_BOUNDARY_LINEAR, far_m)
    )
    assert ratio >= floor, (
        f"a boundary at {near_m:.0f} m and one at {far_m:.0f} m are only "
        f"{ratio:.2f}:1 apart, so a wall in front of a building still reads "
        f"as one surface"
    )


def test_depth_separation_decays_with_range_rather_than_inverting() -> None:
    """
    A near pair must always separate at least as well as a far pair of the same
    span. A ramp normalised to the far rim gets this backwards in the near
    field, which is how the 150 m reach first broke the tint.
    """
    span = 8.0
    ratios = [
        contrast_ratio(
            _hazed(_BOUNDARY_LINEAR, start), _hazed(_BOUNDARY_LINEAR, start + span)
        )
        for start in (10.0, 20.0, 40.0, 80.0)
    ]
    assert ratios == sorted(ratios, reverse=True), ratios


def test_the_near_field_is_not_tinted_at_all() -> None:
    """
    Everything inside WORLD_DEPTH_NEAR_M keeps the full contrast ladder.

    The tint is a cue for distance, not a filter over the whole view: the band
    the planner and AEB work in has to stay at the ratios the palette was built
    around, or the depth cue is bought by washing out the obstacles.
    """
    assert depth_mix(np.asarray([0.0, 4.0, config.WORLD_DEPTH_NEAR_M])).max() == 0.0
    assert contrast_ratio(_hazed(_ROAD_LINEAR, 8.0), _AIR_LINEAR) >= 3.0
    assert contrast_ratio(
        _hazed(_BOUNDARY_LINEAR, 8.0), _hazed(_ROAD_LINEAR, 8.0)
    ) >= 3.0


def test_depth_tint_is_monotone_and_bounded_by_the_haze_ceiling() -> None:
    """
    Extinction asymptotes toward the ceiling and must never cross it.

    The ceiling is short of 1.0 deliberately: at 1.0 the rim dissolves
    completely and the world reads as ending there rather than as continuing
    out of sensor range. Nothing may exceed it at any distance, including well
    past WORLD_RADIUS_M, since the camera itself sits outside the scene.
    """
    mix = depth_mix(np.linspace(0.0, 600.0, 500))
    assert np.all(np.diff(mix) >= 0.0)
    assert mix.max() < config.WORLD_DEPTH_HAZE
    # Still essentially saturated by the time anything reaches the rim.
    at_rim = float(depth_mix(np.asarray([float(config.WORLD_RADIUS_M)]))[0])
    assert at_rim > config.WORLD_DEPTH_HAZE * 0.9


def test_every_visible_slab_face_gets_its_own_shade() -> None:
    """
    An unlit box of one flat colour has no edges at all, which is half of why a
    wall and the building behind it read as one shape. The three faces a
    trailing camera can see -- roof, near side, flank -- must all differ, and
    no wall side may be as bright as a roof or "lit from above" stops reading.
    """
    right = np.asarray((1.0, 0.0, 0.0))
    forward = np.asarray((0.0, 1.0, 0.0))
    shades = face_shades(right, forward)
    top, bottom, east, west, north, south = shades

    assert top == config.WORLD_SLAB_TOP_SHADE
    assert bottom == config.WORLD_SLAB_BOTTOM_SHADE
    for side in (east, west, north, south):
        assert side < top, "a wall side is as bright as a roof"

    visible = sorted((top, south, east))
    gaps = np.diff(visible)
    assert gaps.min() > 0.05, f"two visible faces share a shade: {visible}"


def test_slab_shading_is_fixed_to_the_camera_not_to_the_world() -> None:
    """
    The key light lives in RENDER space, so it must not follow the car round a
    corner. Shading by world normals instead would repaint every wall in the
    scene as the ego turned, which reads as the world flickering.
    """
    straight = face_shades(
        np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))
    )
    turned = face_shades(
        np.asarray((0.0, -1.0, 0.0)), np.asarray((1.0, 0.0, 0.0))
    )

    # The same six shades are handed out either way; which world-facing side
    # receives which one is exactly what should change with heading.
    np.testing.assert_allclose(sorted(straight), sorted(turned), atol=1e-9)
