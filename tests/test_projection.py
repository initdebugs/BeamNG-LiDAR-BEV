"""
The arithmetic between a camera pixel and a point on the ground.

Every sign in here is a convention this project has been wrong about before --
BeamNG's vehicle frame is +X LEFT and +Y REARWARD, the simulator reads a mount
`pos` in a frame offset from the reference node, and image rows run DOWN. A
mistake in any of them still produces plausible-looking numbers, so each is
pinned against a case worked out by hand rather than against the module's own
helpers.

Offline like the rest of the suite. The LIVE confirmation is
`tools/label_overlay.py --measure`, which found zero bias (mean +0.00 px,
median -0.25 px over 234 bay dividers inside 20 m) on the real corpus.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.models import CameraMount
from beamng_lidar_bev.projection import (
    camera_basis_vehicle,
    focal_length_px,
    ground_points,
    pixel_rays,
    place_camera,
    project,
)

# Facing world +Y, level, at the origin.
_LEVEL = {"pos": (0.0, 0.0, 0.0), "dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)}


def _mount(
    position=(0.0, 0.0, 1.3),
    direction=(0.0, -1.0, 0.0),
    hfov=90.0,
    resolution=(1280, 960),
) -> CameraMount:
    return CameraMount(
        name="probe",
        position_vehicle=position,
        direction_vehicle=direction,
        horizontal_fov_deg=hfov,
        vertical_fov_deg=73.7,
        resolution=resolution,
    )


def test_the_focal_length_is_the_pinhole_one():
    # A 90 degree aperture puts the frame edge exactly one focal length out.
    assert focal_length_px(90.0, 1280) == pytest.approx(640.0)
    assert focal_length_px(105.0, 1280) == pytest.approx(
        640.0 / math.tan(math.radians(52.5))
    )


def test_a_forward_camera_looks_along_the_vehicle_s_own_forward():
    """
    +Y is REARWARD in BeamNG's vehicle frame, so a camera looking ahead has
    `dir = (0, -1, 0)`. Getting the flip wrong points every camera backwards
    and is invisible in any single number.
    """
    right, up, axis = camera_basis_vehicle((0.0, -1.0, 0.0))
    assert np.allclose(axis, (0.0, -1.0, 0.0))
    # -x is the vehicle's RIGHT, so a forward camera's image right is -x.
    assert np.allclose(right, (-1.0, 0.0, 0.0))
    assert np.allclose(up, (0.0, 0.0, 1.0))


def test_a_pitched_camera_keeps_its_up_perpendicular_to_its_axis():
    """The A-pillar pair is pitched down; skipping the re-orthogonalisation
    shears every column of the image."""
    right, up, axis = camera_basis_vehicle((0.3, -0.9, -0.2))
    for a, b in ((right, up), (up, axis), (axis, right)):
        assert float(a @ b) == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.norm(up)) == pytest.approx(1.0)
    # Pitched DOWN means the axis dips and image-up tips forward, never back.
    assert axis[2] < 0.0 and up[2] > 0.0


def test_the_vehicle_frame_flips_both_lateral_and_longitudinal():
    """
    A mount at +X sits to the vehicle's LEFT and one at +Y sits BEHIND it, and
    that is what `derive_vehicle_geometry` builds its extents from
    (`left = -right`, `rearward = -forward`). Both flips, or neither.
    """
    left = place_camera(_LEVEL, _mount(position=(1.0, 0.0, 0.0)))
    behind = place_camera(_LEVEL, _mount(position=(0.0, 1.0, 0.0)))
    # Facing world +Y, so the vehicle's left is world -X and behind is world -Y.
    assert np.allclose(left.origin, (-1.0, 0.0, 0.0))
    assert np.allclose(behind.origin, (0.0, -1.0, 0.0))


def test_the_sensor_origin_moves_the_camera_and_is_a_pure_translation():
    """
    The simulator does not read a mount `pos` from the reference node: measured
    live at (+0.160, +0.362, -0.233) on the vivace, identical at four probe
    positions. It applies on all three axes, which is what
    `derive_hybrid_camera_rig` subtracts on x and y so the pair cancels.
    """
    origin = (0.16, 0.36, -0.23)
    plain = place_camera(_LEVEL, _mount(position=(0.0, 0.0, 1.3)))
    moved = place_camera(_LEVEL, _mount(position=(0.0, 0.0, 1.3)), origin)
    shift = moved.origin - plain.origin
    # +x is left (world -X here), +y is rearward (world -Y), +z is up.
    assert np.allclose(shift, (-0.16, -0.36, -0.23))
    # A pure translation: it must not turn the camera at all.
    for field in ("right", "up", "axis"):
        assert np.allclose(getattr(plain, field), getattr(moved, field))


def test_image_rows_run_down_and_columns_run_right():
    """
    A sign error on v mirrors every projection about the horizon, and one on u
    mirrors it left to right. Both still produce finite, plausible pixels.
    """
    placement = place_camera(_LEVEL, _mount())
    ahead = np.array([[0.0, 20.0, 0.0]])       # on the ground, dead ahead
    higher = np.array([[0.0, 20.0, 3.0]])      # the same place, 3 m up
    righter = np.array([[4.0, 20.0, 0.0]])     # 4 m to the world +X

    (u0, v0), = project(placement, ahead)[0]
    (_, v1), = project(placement, higher)[0]
    (u2, _), = project(placement, righter)[0]

    assert u0 == pytest.approx(640.0)
    assert v0 > 480.0        # the ground is BELOW the optical axis
    assert v1 < v0           # higher in the world is a SMALLER row index
    assert u2 > u0           # the vehicle's right is a LARGER column index


def test_a_point_behind_the_camera_is_never_visible():
    """It still yields finite pixels -- the mirror image of the truth -- so the
    depth test is what keeps it out, not the frame bounds."""
    placement = place_camera(_LEVEL, _mount())
    uv, visible = project(placement, np.array([[0.0, -20.0, 0.0]]))
    assert np.isfinite(uv).all()
    assert not visible.any()


def test_a_pixel_becomes_the_ground_point_that_projects_back_to_it():
    """The round trip is the strongest single check: it exercises the basis,
    the focal length and both sign conventions at once."""
    placement = place_camera(_LEVEL, _mount())
    grid = np.array(
        [[120.0, 700.0], [640.0, 600.0], [1150.0, 940.0], [400.0, 520.0]]
    )
    points, hit = ground_points(placement, grid, plane_z=0.0)
    assert hit.all()
    assert np.allclose(points[:, 2], 0.0)
    back, visible = project(placement, points)
    assert visible.all()
    assert np.allclose(back, grid, atol=1e-6)


def test_nothing_above_the_horizon_lands_on_the_ground():
    """
    Above the horizon the plane intersection is BEHIND the camera. It is
    arithmetically perfect nonsense, which is precisely the kind of value that
    poisons a dataset silently, so it reports no hit instead.
    """
    placement = place_camera(_LEVEL, _mount())
    sky = np.array([[640.0, 60.0], [200.0, 10.0], [640.0, 479.0]])
    _, hit = ground_points(placement, sky, plane_z=0.0)
    assert not hit.any()


def test_the_ground_hit_is_bounded_by_range():
    """A ray a whisker under the horizon meets the plane kilometres away, and
    a point out there is noise however exact the arithmetic is."""
    placement = place_camera(_LEVEL, _mount())
    grazing = np.array([[640.0, 481.0]])
    _, near_horizon = ground_points(placement, grazing, plane_z=0.0)
    assert not near_horizon.any()
    _, allowed = ground_points(
        placement, grazing, plane_z=0.0, max_range_m=1e6
    )
    assert allowed.all()


def test_the_rays_carry_a_unit_axis_component_rather_than_unit_length():
    """Deliberate: a planar distance along the axis then places a point with no
    cosine divide, which is the convention the deleted module used."""
    placement = place_camera(_LEVEL, _mount())
    rays = pixel_rays(placement, np.array([[640.0, 480.0], [100.0, 900.0]]))
    assert np.allclose(rays @ placement.axis, 1.0)
    assert float(np.linalg.norm(rays[1])) > 1.0
    assert np.allclose(rays[0], placement.axis)


@pytest.mark.parametrize("heading_deg", [0.0, 37.0, 90.0, -128.0, 180.0])
def test_the_whole_thing_rides_the_ego_heading(heading_deg: float):
    """
    A bay does not move when the car turns, so the pixel a fixed world point
    lands on must depend on the pose and on nothing else. Checked by rotating
    the ego and the point together and requiring the SAME pixel.
    """
    heading = math.radians(heading_deg)
    forward = (math.sin(heading), math.cos(heading), 0.0)
    state = {"pos": (0.0, 0.0, 0.0), "dir": forward, "up": (0.0, 0.0, 1.0)}

    base = np.array([3.0, 18.0, 0.0])
    turned = np.array(
        [
            base[0] * math.cos(heading) + base[1] * math.sin(heading),
            -base[0] * math.sin(heading) + base[1] * math.cos(heading),
            0.0,
        ]
    )
    level = project(place_camera(_LEVEL, _mount()), base[None, :])
    rotated = project(place_camera(state, _mount()), turned[None, :])
    assert np.allclose(level[0], rotated[0], atol=1e-9)
    assert level[1] == rotated[1]


def test_a_ground_point_moves_the_right_way_when_the_car_does():
    """Driving forward must bring a fixed bay NEARER, which is DOWN the frame
    for anything on the ground. A sign error in the pose subtraction inverts
    this and nothing else notices."""
    placement_a = place_camera(_LEVEL, _mount())
    placement_b = place_camera(
        {**_LEVEL, "pos": (0.0, 5.0, 0.0)}, _mount()
    )
    bay = np.array([[0.0, 25.0, 0.0]])
    (_, v_far), = project(placement_a, bay)[0]
    (_, v_near), = project(placement_b, bay)[0]
    assert v_near > v_far
