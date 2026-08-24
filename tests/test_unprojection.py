"""
Rung 0.5: the depth image unprojected, pinned against synthetic depth images
whose geometry is known exactly. The suite is offline, so every test here is
arithmetic; what the simulator actually writes into the buffer is the
`Unprojection check:` line's and tools/unprojection_oracle.py's job.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    CAMERA_NEAR_FAR_PLANES,
    CAMERA_RESOLUTION,
    LIDAR_RANGE_M,
)
from beamng_lidar_bev.geometry import camera_basis, derive_camera_rig
from beamng_lidar_bev.models import CameraMount, VehicleGeometry
from beamng_lidar_bev.unprojection import (
    CameraRays,
    build_camera_rays,
    build_rig_rays,
    pose_from_state,
    sample_annotation,
    sample_depth,
    surface_mask,
    unproject_camera,
    unproject_frame,
)

FAR = CAMERA_NEAR_FAR_PLANES[1]


def _geometry() -> VehicleGeometry:
    return VehicleGeometry(
        ground_z_vehicle=-0.30,
        left_m=0.9,
        right_m=0.9,
        front_m=2.2,
        rear_m=2.3,
        height_m=1.45,
        mounts={},
        body_floor_z=-0.30,
    )


def _mount(
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0),
    position: tuple[float, float, float] = (0.0, -0.6, 1.3),
    hfov: float = 60.0,
    resolution: tuple[int, int] = (64, 48),
    stride: tuple[int, int] = (1, 1),
) -> CameraMount:
    half_h = math.radians(hfov) / 2.0
    aspect = resolution[1] / resolution[0] if resolution[0] else 0.75
    vfov = math.degrees(2.0 * math.atan(math.tan(half_h) * aspect))
    return CameraMount(
        name="probe",
        position_vehicle=position,
        direction_vehicle=direction,
        horizontal_fov_deg=hfov,
        vertical_fov_deg=vfov,
        resolution=resolution,
        sample_stride=stride,
    )


def _level_state(pos=(100.0, 200.0, 50.0), forward=(0.0, 1.0, 0.0)) -> dict:
    return {
        "pos": pos,
        "dir": forward,
        "up": (0.0, 0.0, 1.0),
        "vel": (0.0, 0.0, 0.0),
    }


def _flat_floor_depth(
    rays: CameraRays, mount: CameraMount, floor_z: float
) -> np.ndarray:
    """
    Planar depth of a horizontal plane at vehicle-frame height `floor_z` for
    every lattice sample, NaN where the ray never reaches it.

    Built from the mount's own basis, not from the ray table under test: a
    ray at unit optical depth descends by `-(direction . up_world)` per metre,
    so the plane is hit at Z = (camera height above plane) / descent.
    """
    del rays
    width, height = mount.resolution
    focal = (width / 2.0) / math.tan(math.radians(mount.horizontal_fov_deg) / 2.0)
    cs, rs = mount.sample_stride
    cols = np.arange(cs // 2, width, cs)
    rows = np.arange(rs // 2, height, rs)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    x = (grid_cols.reshape(-1) + 0.5 - width / 2.0) / focal
    y = (grid_rows.reshape(-1) + 0.5 - height / 2.0) / focal
    right, up, axis = camera_basis(mount)
    world_up = np.asarray((0.0, 0.0, 1.0))
    descent = -((axis + x[:, None] * right - y[:, None] * up) @ world_up)
    drop = mount.position_vehicle[2] - floor_z
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(descent > 1e-6, drop / descent, np.nan)
    return depth.astype(np.float32)


def _raw_depth(depth_m: np.ndarray, rays: CameraRays, fill_m: float = FAR) -> bytes:
    """A whole-frame float32 buffer holding `depth_m` at the lattice pixels."""
    width, height = rays.resolution
    frame = np.full(width * height, fill_m / FAR, dtype=np.float32)
    values = np.where(np.isfinite(depth_m), depth_m, fill_m) / FAR
    frame[rays.pixel_index] = values.astype(np.float32)
    return frame.tobytes()


def _raw_annotation(rgb: np.ndarray, rays: CameraRays) -> bytes:
    width, height = rays.resolution
    frame = np.zeros((width * height, 4), dtype=np.uint8)
    frame[rays.pixel_index, :3] = rgb
    frame[:, 3] = 255
    return frame.tobytes()


# --- The ray table --------------------------------------------------------------


def test_the_table_holds_one_ray_per_lattice_pixel_at_unit_optical_depth() -> None:
    mount = _mount(resolution=(64, 48), stride=(4, 2))
    rays = build_camera_rays(mount)

    assert rays.sample_count == (64 // 4) * (48 // 2)
    assert rays.directions.shape == (rays.sample_count, 3)
    assert rays.directions.dtype == np.float32
    # Optical-axis component exactly 1: planar depth multiplies the
    # UNNORMALISED ray, the cosine divide is baked in.
    _, _, axis = camera_basis(mount)
    along = rays.directions.astype(np.float64) @ axis
    assert np.allclose(along, 1.0, atol=1e-6)
    assert np.all(rays.pixel_index < 64 * 48)
    assert len(np.unique(rays.pixel_index)) == rays.sample_count


def test_the_lattice_is_centred_rather_than_hugging_the_top_left() -> None:
    rays = build_camera_rays(_mount(resolution=(64, 48), stride=(4, 4)))
    cols = rays.pixel_index % 64
    rows = rays.pixel_index // 64
    assert cols.min() == 2 and cols.max() == 62
    assert rows.min() == 2 and rows.max() == 46


def test_the_centre_pixel_looks_straight_down_the_axis() -> None:
    mount = _mount(resolution=(64, 48), stride=(1, 1))
    rays = build_camera_rays(mount)
    # Pixel centres sit at +0.5, so the four pixels around the principal point
    # are each half a pixel off it; the one at (31, 23) is offset (-0.5, -0.5).
    index = int(np.flatnonzero(rays.pixel_index == 23 * 64 + 31)[0])
    direction = rays.directions[index].astype(np.float64)
    _, _, axis = camera_basis(mount)
    off_axis = direction - axis
    assert np.linalg.norm(off_axis) == pytest.approx(
        math.hypot(0.5, 0.5) / rays.focal_px, rel=1e-5
    )


def test_a_bad_stride_or_resolution_is_refused() -> None:
    with pytest.raises(ValueError):
        build_camera_rays(_mount(stride=(0, 1)))
    with pytest.raises(ValueError):
        build_camera_rays(_mount(resolution=(0, 48)))


# --- Decoding the buffers ---------------------------------------------------------


def test_depth_decodes_as_raw_float_times_the_far_plane() -> None:
    rays = build_camera_rays(_mount(resolution=(8, 6)))
    truth = np.full(rays.sample_count, 25.0, dtype=np.float32)
    depth = sample_depth(_raw_depth(truth, rays), rays)
    assert depth is not None
    assert np.allclose(depth, 25.0, atol=1e-3)


def test_an_absent_short_or_unfilled_depth_buffer_reads_as_no_frame() -> None:
    rays = build_camera_rays(_mount(resolution=(8, 6)))
    assert sample_depth(None, rays) is None
    assert sample_depth(b"\x00" * 7, rays) is None
    assert sample_depth(b"\x00" * (8 * 6 * 4), rays) is None


def test_annotation_is_gathered_at_the_same_pixels_fourth_byte_dropped() -> None:
    rays = build_camera_rays(_mount(resolution=(8, 6), stride=(2, 2)))
    rgb = np.arange(rays.sample_count * 3, dtype=np.uint8).reshape((-1, 3))
    colours = sample_annotation(_raw_annotation(rgb, rays), rays)
    assert colours is not None
    assert colours.shape == (rays.sample_count, 3)
    assert np.array_equal(colours, rgb)
    assert sample_annotation(b"\x00" * 5, rays) is None


def test_sky_bodywork_and_beyond_the_cull_radius_are_not_surfaces() -> None:
    depth = np.asarray(
        [FAR, FAR * 0.99, 0.1, 0.5, 20.0, LIDAR_RANGE_M + 1.0, np.nan],
        dtype=np.float32,
    )
    assert surface_mask(depth).tolist() == [
        False, False, False, True, True, False, False
    ]


# --- Placing the points ------------------------------------------------------------


def test_a_flat_floor_comes_back_flat_to_the_edge_of_the_frame() -> None:
    """
    THE planar-Z test. Depth is Z along the optical axis, not range along the
    ray; multiplying a unit ray by it instead pulls every off-axis sample
    toward the lens, and a flat road renders as a bowl that deepens toward
    the frame edges. Every unprojected ground sample here must land on the
    floor plane to well under a millimetre, centre and corners alike.
    """
    geometry = _geometry()
    mount = _mount(hfov=100.0, resolution=(96, 72))
    rays = build_camera_rays(mount)
    floor_vehicle = 0.0  # the vehicle ground plane, where pos.z = 0
    depth = _flat_floor_depth(rays, mount, floor_vehicle)
    keep = np.isfinite(depth) & (depth < 150.0)
    assert keep.sum() > 1000

    state = _level_state()
    points = unproject_camera(rays, depth, pose_from_state(state), geometry, keep)

    # The floor plane sits at node z + sensor_floor_z in world.
    expected_z = state["pos"][2] + geometry.sensor_floor_z
    assert np.abs(points[:, 2] - expected_z).max() < 5e-4
    # And it is in FRONT of the car: forward is world +Y here.
    assert (points[:, 1] > state["pos"][1]).all()


def test_a_wall_at_known_planar_depth_lands_at_that_distance_ahead() -> None:
    geometry = _geometry()
    mount = _mount(hfov=60.0, resolution=(32, 24))
    rays = build_camera_rays(mount)
    depth = np.full(rays.sample_count, 20.0, dtype=np.float32)
    state = _level_state()

    points = unproject_camera(rays, depth, pose_from_state(state), geometry)

    # Mount y is -0.6 (0.6 m ahead of the node): the wall is 20.6 m ahead.
    ahead = points[:, 1] - state["pos"][1]
    assert np.allclose(ahead, 20.6, atol=1e-3)


def test_image_right_is_vehicle_right_for_a_forward_camera() -> None:
    """
    Handedness. A pixel to the right of the principal point must land on
    the vehicle's RIGHT (vehicle -X; BEV +right). Vehicle +X is LEFT, which is
    the one sign in this frame that is easy to get backwards.
    """
    mount = _mount(resolution=(32, 24), stride=(1, 1))
    rays = build_camera_rays(mount)
    right_col = int(np.flatnonzero(rays.pixel_index == 12 * 32 + 30)[0])
    left_col = int(np.flatnonzero(rays.pixel_index == 12 * 32 + 1)[0])
    assert rays.directions[right_col, 0] < 0.0  # vehicle -X is right
    assert rays.directions[left_col, 0] > 0.0
    # In the world, with forward = +Y, vehicle right is world +X.
    state = _level_state()
    depth = np.full(rays.sample_count, 10.0, dtype=np.float32)
    points = unproject_camera(rays, depth, pose_from_state(state), _geometry())
    assert points[right_col, 0] > state["pos"][0]
    assert points[left_col, 0] < state["pos"][0]


def test_image_down_is_toward_the_ground() -> None:
    mount = _mount(resolution=(32, 24), stride=(1, 1))
    rays = build_camera_rays(mount)
    bottom = int(np.flatnonzero(rays.pixel_index == 23 * 32 + 16)[0])
    top = int(np.flatnonzero(rays.pixel_index == 0 * 32 + 16)[0])
    assert rays.directions[bottom, 2] < 0.0
    assert rays.directions[top, 2] > 0.0


def test_the_pitched_rear_camera_still_reconstructs_a_level_floor() -> None:
    """
    The reversing camera looks down as well as back. Its basis has to be
    re-orthogonalised against the tilted axis or the columns shear and the
    floor tilts with the camera instead of staying where the ground is.
    """
    geometry = _geometry()
    rig = derive_camera_rig(geometry)
    rear = rig["rear"]
    assert rear.direction_vehicle[2] < 0.0, "the rear camera must pitch down"
    rays = build_camera_rays(rear)
    depth = _flat_floor_depth(rays, rear, 0.0)
    keep = np.isfinite(depth) & (depth > 0.3) & (depth < 60.0)
    assert keep.sum() > 500
    state = _level_state()

    points = unproject_camera(rays, depth, pose_from_state(state), geometry, keep)

    expected_z = state["pos"][2] + geometry.sensor_floor_z
    assert np.abs(points[:, 2] - expected_z).max() < 1e-3
    # Behind the car, not in front of it.
    assert (points[:, 1] < state["pos"][1]).all()
    # And image right is vehicle LEFT for a camera looking backwards: a
    # column right of centre lands at world -X when forward is +Y.
    width = rear.resolution[0]
    cols = rays.pixel_index % width
    right_half = keep & (cols > width * 0.75)
    left_half = keep & (cols < width * 0.25)
    assert (points[right_half[keep], 0] < state["pos"][0]).all()
    assert (points[left_half[keep], 0] > state["pos"][0]).all()


def test_the_cloud_rotates_with_the_car() -> None:
    geometry = _geometry()
    mount = _mount(resolution=(16, 12))
    rays = build_camera_rays(mount)
    depth = np.full(rays.sample_count, 10.0, dtype=np.float32)
    facing_west = _level_state(forward=(-1.0, 0.0, 0.0))

    points = unproject_camera(rays, depth, pose_from_state(facing_west), geometry)

    assert np.allclose(points[:, 0] - facing_west["pos"][0], -10.6, atol=1e-3)


def test_a_frames_age_places_it_from_where_the_car_was() -> None:
    """
    The per-camera frames carry no timestamp and are staged behind. A frame
    that is 60 ms old at 11 m/s was rendered from 0.66 m back, and placing it
    from the current pose would put every wall 0.66 m too far ahead -- the
    late direction for AEB. The pose is rewound by velocity x age.
    """
    state = _level_state()
    state["vel"] = (0.0, 11.0, 0.0)
    now = pose_from_state(state, age_s=0.0)
    then = pose_from_state(state, age_s=0.06)
    assert np.allclose(then.origin_world - now.origin_world, (0.0, -0.66, 0.0))
    assert np.array_equal(then.forward_world, now.forward_world)


def test_the_body_floor_places_the_camera_not_the_gravity_referenced_one() -> None:
    """
    `pos.z` is referenced to the BODY's floor plane. On a grade the gravity-
    referenced bbox bottom is lower than the body-frame one, and using it
    would sink every camera by the difference. `sensor_floor_z` prefers the
    body figure and falls back to the other only for callers that lack it.
    """
    with_both = VehicleGeometry(
        ground_z_vehicle=-0.42,
        left_m=0.9,
        right_m=0.9,
        front_m=2.2,
        rear_m=2.3,
        height_m=1.45,
        mounts={},
        body_floor_z=-0.30,
    )
    assert with_both.sensor_floor_z == pytest.approx(-0.30)
    assert _geometry().sensor_floor_z == pytest.approx(-0.30)
    legacy = VehicleGeometry(
        ground_z_vehicle=-0.42,
        left_m=0.9,
        right_m=0.9,
        front_m=2.2,
        rear_m=2.3,
        height_m=1.45,
        mounts={},
    )
    assert legacy.sensor_floor_z == pytest.approx(-0.42)


# --- One camera, one tick ---------------------------------------------------------


def test_a_frame_yields_parallel_points_and_colours_with_sky_removed() -> None:
    geometry = _geometry()
    mount = _mount(resolution=(16, 12), stride=(2, 2))
    rays = build_camera_rays(mount)
    depth = np.full(rays.sample_count, 15.0, dtype=np.float32)
    depth[:10] = FAR  # sky
    depth[10:12] = 0.1  # bonnet
    rgb = np.zeros((rays.sample_count, 3), dtype=np.uint8)
    rgb[:, 0] = np.arange(rays.sample_count, dtype=np.uint8)
    unknown = np.asarray((1, 2, 3), dtype=np.uint8)

    result = unproject_frame(
        rays,
        _raw_depth(depth, rays),
        _raw_annotation(rgb, rays),
        pose_from_state(_level_state()),
        geometry,
        unknown,
    )

    assert result is not None
    points, colours, sampled = result
    assert sampled == rays.sample_count - 12
    assert points.shape == (sampled, 3)
    assert colours.shape == (sampled, 3)
    assert points.dtype == np.float32 and colours.dtype == np.uint8
    # Parallel: the surviving colours are exactly the surviving samples'.
    assert np.array_equal(colours[:, 0], np.arange(12, rays.sample_count))


def test_a_missing_annotation_buffer_paints_unknown_not_nothing() -> None:
    mount = _mount(resolution=(8, 6))
    rays = build_camera_rays(mount)
    depth = np.full(rays.sample_count, 12.0, dtype=np.float32)
    unknown = np.asarray((1, 2, 3), dtype=np.uint8)

    result = unproject_frame(
        rays, _raw_depth(depth, rays), None, pose_from_state(_level_state()),
        _geometry(), unknown,
    )

    assert result is not None
    points, colours, _ = result
    assert len(points) == rays.sample_count
    assert (colours == unknown).all()


def test_an_unfilled_depth_buffer_is_no_frame_and_an_all_sky_one_is_empty() -> None:
    mount = _mount(resolution=(8, 6))
    rays = build_camera_rays(mount)
    assert unproject_frame(
        rays, None, None, pose_from_state(_level_state()), _geometry(),
        np.zeros(3, np.uint8),
    ) is None
    sky = np.full(rays.sample_count, FAR, dtype=np.float32)
    result = unproject_frame(
        rays, _raw_depth(sky, rays), None, pose_from_state(_level_state()),
        _geometry(), np.zeros(3, np.uint8),
    )
    assert result is not None
    assert result[2] == 0 and len(result[0]) == 0


# --- The rig's budget ---------------------------------------------------------------


def test_the_rig_sample_budget_is_bounded() -> None:
    """
    Every downstream stage is O(cloud) and was budgeted against a 100-150k
    LiDAR cloud. The lattice is the ONLY thing that sets the camera cloud's
    size, so a stride edit that blew the tick would otherwise be silent.
    Roughly half of each frame is sky and gets culled, so ~300k sampled
    pixels is the ceiling that lands the live cloud in the LiDAR band.
    """
    rig = derive_camera_rig(_geometry(), resolution=CAMERA_RESOLUTION)
    rays = build_rig_rays(rig)
    total = sum(r.sample_count for r in rays.values())
    assert 150_000 <= total <= 320_000, total
    # The long-range camera gets the finest ROW stride of all: rows are the
    # range axis for ground seen from a camera.
    assert rays["front_main"].sample_stride[1] <= min(
        r.sample_stride[1] for r in rays.values()
    )


def test_the_rear_camera_reaches_the_ground_close_behind_the_bumper() -> None:
    """
    The reason it is wide and pitched: a reversing camera's job is the
    ground immediately behind the car. The bottom of the frame must meet the
    floor within a metre of the lens.
    """
    geometry = _geometry()
    rear = derive_camera_rig(geometry)["rear"]
    rays = build_camera_rays(rear)
    depth = _flat_floor_depth(rays, rear, 0.0)
    keep = np.isfinite(depth) & (depth > 0.0)
    points = unproject_camera(
        rays, depth, pose_from_state(_level_state()), geometry, keep
    )
    behind_lens = -(points[:, 1] - _level_state()["pos"][1]) - rear.position_vehicle[1]
    assert behind_lens.min() < 1.0, behind_lens.min()


def test_the_far_road_band_rows_are_sampled_at_full_density() -> None:
    """
    The street oracle capture (2026-08-24) measured the camera ground band
    ACCURATE to -1..-2 cm against the LiDAR floor on every ring out to 60 m
    but STARVED past 20 m: ~175 returns per 4 m ring at 20-24 m against the
    road-scan unit's ~1300, because rows are the range axis and stride-2
    rows land rings 2.3 m apart at 40 m against WORLD's 1.5 m road bridge.
    All ground from 20 to 100 m lives in a ~54-row band just under the
    horizon (planar geometry, image y = h/r), so that band is sampled at
    full density: consecutive rings must stay inside the bridge out to
    40 m, which is what moves the single-frame road edge from ~30 to ~45 m.
    """
    import numpy as np

    from beamng_lidar_bev.config import (
        WORLD_CELL_SIZE_M,
        WORLD_ROAD_BRIDGE_CELLS,
    )
    from beamng_lidar_bev.geometry import camera_vertical_fov_deg
    from beamng_lidar_bev.models import CameraMount
    from beamng_lidar_bev.unprojection import (
        build_camera_rays,
        focal_length_px,
    )

    eye = 1.3
    resolution = (960, 720)

    def make(band):
        return CameraMount(
            name="front_main",
            position_vehicle=(0.0, -1.0, eye),
            direction_vehicle=(0.0, -1.0, 0.0),
            horizontal_fov_deg=50.0,
            vertical_fov_deg=camera_vertical_fov_deg(50.0, resolution),
            resolution=resolution,
            sample_stride=(4, 2),
            far_road_band_m=band,
        )

    rays = build_camera_rays(make((20.0, 100.0)))
    rows = np.unique(rays.pixel_index // resolution[0])
    focal = focal_length_px(50.0, resolution[0])
    centre = resolution[1] / 2.0 - 0.5
    in_band = rows[
        (rows >= centre + eye / 100.0 * focal)
        & (rows <= centre + eye / 20.0 * focal)
    ]
    assert len(in_band) >= 40
    assert np.all(np.diff(in_band) == 1), "the band is full density"

    # The rings those rows lay on level ground: every gap out to 40 m must
    # stay inside the road bridge or the far road still fragments.
    ranges = np.sort(eye * focal / (in_band + 0.5 - resolution[1] / 2.0))
    gaps = np.diff(ranges)
    bridge_m = WORLD_CELL_SIZE_M * WORLD_ROAD_BRIDGE_CELLS
    assert gaps[ranges[:-1] <= 40.0].max() < bridge_m

    # A mount with no band keeps the plain strided lattice bit for bit.
    plain = build_camera_rays(make(None))
    assert np.array_equal(
        np.unique(plain.pixel_index // resolution[0]),
        np.arange(1, resolution[1], 2),
    )
