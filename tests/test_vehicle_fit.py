"""
Fitting a box to the returns a car actually gives back.

Every cloud here is built the way the sensors lay one down -- dense vertically,
STRIPED in azimuth -- because that is the whole difficulty. A car sampled
uniformly is easy and is not what arrives.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    VEHICLE_FIT_DEFAULT_LENGTH_M,
    VEHICLE_FIT_STRIPE_RAD,
)
from beamng_lidar_bev.vehicle_fit import fit_vehicle_boxes

EGO = (0.0, 0.0, 0.0)


def _face(start: np.ndarray, end: np.ndarray, base: float, top: float) -> np.ndarray:
    """A dense vertical surface between two XY points."""
    span = float(np.linalg.norm(end - start))
    along = np.linspace(0.0, 1.0, max(2, int(span / 0.02)))
    heights = np.arange(base, top, 0.05)
    xy = start + np.outer(along, end - start)
    return np.column_stack(
        (
            np.repeat(xy[:, 0], len(heights)),
            np.repeat(xy[:, 1], len(heights)),
            np.tile(heights, len(xy)),
        )
    )


def _striped(points: np.ndarray, ego=EGO) -> np.ndarray:
    """
    Keep only what lands on an azimuth stripe.

    This is the sampling that makes a car unmeshable: at 15 m the stripes are
    nearly a metre apart, so a whole car comes back as a handful of vertical
    slices with nothing between them.
    """
    offsets = points[:, :2] - np.asarray(ego)[:2]
    azimuth = np.arctan2(offsets[:, 1], offsets[:, 0]) / VEHICLE_FIT_STRIPE_RAD
    return points[np.abs(azimuth - np.round(azimuth)) < 0.06]


def _car(
    centre: tuple[float, float],
    yaw_deg: float,
    *,
    width: float = 1.9,
    length: float = 4.5,
    height: float = 1.5,
    base: float = 0.0,
    faces: str = "both",
    ego=EGO,
) -> np.ndarray:
    """The visible faces of a box: the end nearest the ego, and one flank."""
    yaw = math.radians(yaw_deg)
    forward = np.array((math.cos(yaw), math.sin(yaw)))
    right = np.array((math.sin(yaw), -math.cos(yaw)))
    origin = np.asarray(centre, dtype=np.float64)

    def corner(along: float, across: float) -> np.ndarray:
        return origin + forward * (along * length / 2.0) + right * (
            across * width / 2.0
        )

    # Only the faces TURNED TOWARD the ego return anything, which is what makes
    # a car an L rather than a rectangle. Which two those are depends on where
    # the car is, so it is chosen here rather than fixed.
    eye = np.asarray(ego, dtype=np.float64)[:2]
    near_along = -1.0 if np.dot(forward, origin - eye) > 0.0 else 1.0
    near_across = -1.0 if np.dot(right, origin - eye) > 0.0 else 1.0

    parts = []
    if faces in ("both", "end"):
        parts.append(
            _face(
                corner(near_along, -1),
                corner(near_along, 1),
                base,
                base + height,
            )
        )
    if faces in ("both", "flank"):
        parts.append(
            _face(
                corner(-1, near_across),
                corner(1, near_across),
                base,
                base + height,
            )
        )
    return _striped(np.concatenate(parts), ego)


def test_a_striped_car_is_one_box_not_a_handful_of_fragments() -> None:
    """
    The headline case, and the reported one: at a standstill a car is four or
    five azimuth stripes over a metre apart. Clustering in world XY needs a
    threshold wide enough to bridge that, which is why this clusters in the
    sensor's own lattice instead.
    """
    cloud = _car((0.0, 15.0), 90.0)

    boxes = fit_vehicle_boxes(cloud, EGO)

    assert len(boxes) == 1, "one car came back as fragments"
    assert boxes[0].point_count == len(cloud)


def test_the_footprint_lies_along_the_car_not_across_its_corner() -> None:
    """
    Why minimum-area and not PCA. A car seen from behind and to one side is an
    L, and an L's principal axis runs diagonally across it -- so PCA draws a
    parked car at a large angle to the kerb it is parked against.
    """
    cloud = _car((3.0, 14.0), 90.0)

    box = fit_vehicle_boxes(cloud, EGO)[0]

    heading = math.degrees(
        math.atan2(box.forward_world[1], box.forward_world[0])
    )
    assert min(abs(heading - 90.0), abs(heading + 90.0)) < 6.0
    # Both dimensions are MEASURED here -- but the length reads short, and that
    # is the sampling rather than the fit: a flank alongside the ego is nearly
    # edge-on, so only its near part is crossed by any stripe at all. Under-
    # reading what was seen is the honest direction; nothing is invented.
    width, _, length = box.dimensions_m
    assert not box.inferred_depth
    assert 3.0 <= length <= 5.5
    assert width == pytest.approx(1.9, abs=0.5)


def test_an_angled_car_keeps_its_angle() -> None:
    cloud = _car((6.0, 16.0), 35.0)

    box = fit_vehicle_boxes(cloud, EGO)[0]

    heading = math.degrees(
        math.atan2(box.forward_world[1], box.forward_world[0])
    )
    assert min(abs(heading - 35.0), abs(heading + 145.0)) < 6.0


def test_one_face_infers_the_depth_BEHIND_the_returns_that_were_measured()  -> None:
    """
    A car dead ahead shows only its end. The width is real, the length is not --
    and the inference must sit behind the evidence, so the drawn near face stays
    on the measured returns rather than floating in front of them.
    """
    cloud = _car((0.0, 15.0), 90.0, faces="end")

    box = fit_vehicle_boxes(cloud, EGO)[0]

    assert box.inferred_depth
    # Two stripes cross a 1.9 m face at this range, so the width is resolved to
    # within about 0.4 m and no correction can do better -- the tolerance is the
    # sampling, not slack in the fit.
    width, _, length = box.dimensions_m
    assert width == pytest.approx(1.9, abs=0.4)
    assert length == pytest.approx(VEHICLE_FIT_DEFAULT_LENGTH_M, abs=0.1)
    near_face = box.centre_world[1] - length / 2.0
    assert near_face == pytest.approx(cloud[:, 1].min(), abs=0.25)
    # Drawn fainter than a box measured on two faces, which reaches 1.0.
    assert box.confidence < 1.0, "an inferred box is not drawn as a measured one"


def test_a_flank_alone_measures_the_length_and_assumes_the_width() -> None:
    # Well off to the side, so the flank is crossed by several stripes rather
    # than grazed by one. A flank seen nearly edge-on returns a short segment
    # and is correctly read as an end instead -- ambiguous from a cloud, and
    # resolved the conservative way.
    cloud = _car((14.0, 5.0), 90.0, faces="flank")

    box = fit_vehicle_boxes(cloud, EGO)[0]

    assert box.inferred_depth
    width, _, length = box.dimensions_m
    assert 3.5 <= length <= 5.5
    assert width == pytest.approx(1.9, abs=0.1)
    # Pushed away from the ego, never toward it.
    assert box.centre_world[0] > cloud[:, 0].min()


def test_two_cars_parked_a_metre_apart_stay_two_cars() -> None:
    """
    The failure a world-XY threshold cannot avoid: wide enough to hold one car
    together at range is wide enough to weld two together up close.
    """
    cloud = np.concatenate(
        (_car((6.0, 12.0), 90.0), _car((6.0, 18.0), 90.0))
    )

    boxes = fit_vehicle_boxes(cloud, EGO)

    assert len(boxes) == 2
    assert abs(boxes[0].centre_world[1] - boxes[1].centre_world[1]) > 4.0


def test_a_long_wall_of_returns_is_never_claimed_as_a_vehicle() -> None:
    """
    Nothing here may INVENT a vehicle. A run of returns longer than any vehicle
    with no gap to split at is left to the solids, which already draw it.
    """
    wall = _striped(
        _face(np.array((-12.0, 20.0)), np.array((12.0, 20.0)), 0.0, 2.0)
    )

    assert fit_vehicle_boxes(wall, EGO) == ()


def test_a_kerb_height_clump_is_not_a_vehicle() -> None:
    bush = _striped(
        _face(np.array((3.0, 9.0)), np.array((5.0, 9.0)), 0.0, 0.35)
    )

    assert fit_vehicle_boxes(bush, EGO) == ()


def test_too_few_returns_claims_nothing() -> None:
    assert fit_vehicle_boxes(np.zeros((0, 3)), EGO) == ()
    assert fit_vehicle_boxes(np.ones((4, 3)), EGO) == ()


def test_the_box_stands_on_the_lowest_return_so_the_model_sits_on_the_road() -> None:
    cloud = _car((0.0, 14.0), 90.0, base=3.25, height=1.5)

    box = fit_vehicle_boxes(cloud, EGO)[0]

    assert box.centre_world[2] == pytest.approx(3.25, abs=0.05)
    assert box.dimensions_m[1] == pytest.approx(1.5, abs=0.1)


def test_a_car_behind_the_ego_is_found_exactly_as_one_ahead_is() -> None:
    """The azimuth lattice wraps, so the +/-pi seam is not a blind spot."""
    behind = _car((0.0, -15.0), 90.0)

    assert len(fit_vehicle_boxes(behind, EGO)) == 1


def test_a_far_car_is_still_one_box_where_the_stripes_are_widest() -> None:
    cloud = _car((0.0, 38.0), 90.0)

    boxes = fit_vehicle_boxes(cloud, EGO)

    assert len(boxes) == 1
    assert boxes[0].confidence > 0.0
