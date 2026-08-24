from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.config import CAMERA_NEAR_FAR_PLANES
from beamng_lidar_bev.geometry import (
    CAMERA_NAMES,
    HYBRID_CAMERA_NAMES,
    camera_vertical_fov_deg,
    derive_camera_rig,
    derive_hybrid_camera_rig,
)
from beamng_lidar_bev.models import (
    BevFrame,
    CameraMount,
    VehicleGeometry,
    VisionFrame,
)
from beamng_lidar_bev.semantics import SemanticPalette
from beamng_lidar_bev.unprojection import build_camera_rays
from beamng_lidar_bev.vision_view import (
    grid_dimensions,
    toggle_focus,
    wants_prescale,
)
from beamng_lidar_bev.worker import (
    SENSOR_MODE_HYBRID,
    SENSOR_MODE_LIDAR,
    SENSOR_MODE_VISION,
    BeamNgWorker,
)


def _geometry() -> VehicleGeometry:
    return VehicleGeometry(
        ground_z_vehicle=-0.3,
        left_m=0.9,
        right_m=0.9,
        front_m=2.2,
        rear_m=2.3,
        height_m=1.45,
        mounts={},
    )


def _offset_geometry() -> VehicleGeometry:
    """
    A vehicle whose REFERENCE NODE is off the body centre, as real ones are.

    `_geometry` above is symmetric (left_m == right_m), which is exactly why it
    could never catch a camera placed at x = 0 instead of the body centreline.
    These extents are the vivace's, measured live: 2.02 m wide with the node
    0.16 m off centre.
    """
    return VehicleGeometry(
        ground_z_vehicle=-0.248,
        left_m=1.17,
        right_m=0.85,
        front_m=1.84,
        rear_m=2.49,
        height_m=1.45,
        mounts={},
    )


# --- The rig ------------------------------------------------------------------


def test_the_rig_is_the_eight_hw4_cameras_in_a_stable_order() -> None:
    rig = derive_camera_rig(_geometry())
    assert tuple(rig) == CAMERA_NAMES
    assert len(rig) == 8


def test_the_hybrid_rig_is_exactly_the_two_a_pillar_cameras() -> None:
    rig = derive_hybrid_camera_rig(_offset_geometry())
    assert tuple(rig) == HYBRID_CAMERA_NAMES
    assert tuple(rig) == ("a_pillar_left", "a_pillar_right")


def test_hybrid_cameras_are_mirrored_outboard_a_pillar_mounts() -> None:
    geometry = _offset_geometry()
    rig = derive_hybrid_camera_rig(geometry)
    left, right = rig.values()
    centre_x = (geometry.left_m - geometry.right_m) / 2.0

    assert left.position_vehicle[0] > geometry.left_m
    assert right.position_vehicle[0] < -geometry.right_m
    assert left.position_vehicle[1] == pytest.approx(-0.25 * geometry.front_m)
    assert right.position_vehicle[1] == pytest.approx(left.position_vehicle[1])
    assert left.position_vehicle[2] == pytest.approx(0.88 * geometry.height_m)
    assert right.position_vehicle[2] == pytest.approx(left.position_vehicle[2])
    assert left.position_vehicle[0] - centre_x == pytest.approx(
        -(right.position_vehicle[0] - centre_x)
    )


def test_hybrid_pair_covers_the_front_half_circle_with_overlap() -> None:
    left, right = derive_hybrid_camera_rig(_geometry()).values()
    left_centre = _bearing_deg(left.direction_vehicle)
    right_centre = _bearing_deg(right.direction_vehicle)
    assert left_centre == pytest.approx(37.0)
    assert right_centre == pytest.approx(-37.0)
    assert left.horizontal_fov_deg == pytest.approx(105.0)
    assert right.horizontal_fov_deg == pytest.approx(105.0)
    assert left_centre + left.horizontal_fov_deg / 2.0 == pytest.approx(89.5)
    assert right_centre - right.horizontal_fov_deg / 2.0 == pytest.approx(-89.5)
    overlap = (
        right_centre + right.horizontal_fov_deg / 2.0
        - (left_centre - left.horizontal_fov_deg / 2.0)
    )
    assert overlap == pytest.approx(31.0)


def test_hybrid_pair_is_high_quality_and_pitched_toward_the_road() -> None:
    for mount in derive_hybrid_camera_rig(_geometry()).values():
        assert mount.resolution == (1280, 960)
        assert mount.direction_vehicle[2] < 0.0
        pitch = math.degrees(math.asin(-mount.direction_vehicle[2]))
        assert pitch == pytest.approx(7.0)


def test_forward_is_negative_y_never_the_intuitive_positive() -> None:
    """
    Vehicle space is +X left, +Y REARWARD: a camera built with the intuitive
    dir=(0, 1, 0) films the rear seats. Verified live in the feasibility work
    across all mounts, so the sign is pinned here.
    """
    rig = derive_camera_rig(_geometry())
    for name in ("front_main", "front_wide", "front_bumper"):
        assert rig[name].direction_vehicle == (0.0, -1.0, 0.0)
    # The rear camera looks back (+Y) and DOWN: it is the reversing camera,
    # and its job is the ground immediately behind the bumper.
    rear = rig["rear"].direction_vehicle
    assert rear[0] == 0.0 and rear[1] > 0.9 and rear[2] < 0.0


def test_side_cameras_sit_outside_the_body_shell() -> None:
    """There is no hide-ego flag: a mount inside the glasshouse films the
    cabin (measured 68% CAR on the first windshield attempt)."""
    geometry = _geometry()
    rig = derive_camera_rig(geometry)
    for name in ("pillar_left", "repeater_left"):
        assert rig[name].position_vehicle[0] > geometry.left_m
    for name in ("pillar_right", "repeater_right"):
        assert rig[name].position_vehicle[0] < -geometry.right_m
    # The bumper camera needs a GENEROUS standoff, not just "outside the
    # bbox": the ordinary clearance landed inside the bumper shell live,
    # because the box face is the car's widest point, not the bumper face.
    assert rig["front_bumper"].position_vehicle[1] <= -(
        geometry.front_m + 0.25
    )
    assert rig["rear"].position_vehicle[1] > geometry.rear_m


def test_pillars_look_forward_outboard_and_repeaters_rear_outboard() -> None:
    rig = derive_camera_rig(_geometry())
    # +X is LEFT: the left pillar's direction gains a positive X component,
    # and forward means a negative Y one.
    assert rig["pillar_left"].direction_vehicle[0] > 0.0
    assert rig["pillar_left"].direction_vehicle[1] < 0.0
    assert rig["pillar_right"].direction_vehicle[0] < 0.0
    assert rig["pillar_right"].direction_vehicle[1] < 0.0
    assert rig["repeater_left"].direction_vehicle[0] > 0.0
    assert rig["repeater_left"].direction_vehicle[1] > 0.0
    assert rig["repeater_right"].direction_vehicle[0] < 0.0
    assert rig["repeater_right"].direction_vehicle[1] > 0.0


def test_every_mount_sits_between_the_ground_plane_and_the_roof() -> None:
    geometry = _geometry()
    for mount in derive_camera_rig(geometry).values():
        assert 0.0 < mount.position_vehicle[2] <= geometry.height_m


def test_the_vertical_fov_is_derived_from_the_horizontal_and_aspect() -> None:
    # The constructor takes field_of_view_y; the rig is designed in horizontal
    # apertures, so the projection has to run backwards through the aspect.
    vfov = camera_vertical_fov_deg(90.0, (640, 480))
    assert vfov == pytest.approx(
        math.degrees(2.0 * math.atan(math.tan(math.radians(45.0)) * 0.75))
    )
    for mount in derive_camera_rig(_geometry()).values():
        assert mount.vertical_fov_deg < mount.horizontal_fov_deg


# --- The grid layout ----------------------------------------------------------


def test_the_grid_always_holds_every_camera() -> None:
    for count in range(1, 10):
        rows, cols = grid_dimensions(count, 1600.0, 900.0)
        assert rows * cols >= count


def test_a_wide_window_lays_out_in_more_columns_than_rows() -> None:
    rows, cols = grid_dimensions(8, 1600.0, 700.0)
    assert cols > rows


def test_a_tall_window_lays_out_in_more_rows_than_columns() -> None:
    rows, cols = grid_dimensions(8, 700.0, 1600.0)
    assert rows > cols


def test_an_empty_grid_is_zero_by_zero() -> None:
    assert grid_dimensions(0, 800.0, 600.0) == (0, 0)


def test_clicking_a_tile_focuses_it_and_any_click_returns_to_the_grid() -> None:
    assert toggle_focus(None, "front_main") == "front_main"
    # While focused there are no other tiles on screen, so EVERY click goes
    # back to the grid -- including one that would have hit another tile.
    assert toggle_focus("front_main", "front_main") is None
    assert toggle_focus("front_main", "rear") is None
    assert toggle_focus("front_main", None) is None
    # Clicking the gap between tiles does nothing.
    assert toggle_focus(None, None) is None


# --- The worker's vision path -------------------------------------------------


class StreamingCameraStub:
    """
    A Camera the way the worker drives one: stream_raw only, never poll.

    Three channels, as the rung-0.5 rig renders: colour for the grid, DEPTH
    (planar Z as raw float32 / far plane) and ANNOTATION (a palette colour)
    for the unprojection. A repaint changes the depth by a hair, which is
    what a new simulator frame looks like to the worker's digest.
    """

    def __init__(
        self,
        width: int = 4,
        height: int = 3,
        depth_m: float = 20.0,
        annotation: tuple[int, int, int] = (255, 0, 0),
        with_depth: bool = True,
    ) -> None:
        self.resolution = (width, height)
        self.stream_raw_calls = 0
        self._pixel = 17
        self._depth_m = depth_m
        self._annotation = annotation
        self._with_depth = with_depth
        self._frame = 0

    def repaint(self) -> None:
        """Change the buffers, as a new simulator frame would."""
        self._pixel = (self._pixel + 41) % 256
        self._frame += 1

    def stream_raw(self) -> dict[str, bytes]:
        self.stream_raw_calls += 1
        width, height = self.resolution
        count = width * height
        raw = {"colour": bytes([self._pixel]) * (count * 4)}
        if self._with_depth:
            depth = np.full(
                count,
                (self._depth_m + 1e-3 * self._frame) / CAMERA_NEAR_FAR_PLANES[1],
                dtype=np.float32,
            )
            raw["depth"] = depth.tobytes()
            annotation = np.zeros((count, 4), dtype=np.uint8)
            annotation[:, :3] = self._annotation
            annotation[:, 3] = 255
            raw["annotation"] = annotation.tobytes()
        return raw

    def poll(self) -> None:
        raise AssertionError(
            "The display loop must not issue blocking camera polls"
        )


class RemovableCameraStub(StreamingCameraStub):
    def __init__(self, raises_on_remove: bool = False) -> None:
        super().__init__()
        self.raises_on_remove = raises_on_remove
        self.remove_calls = 0

    def remove(self) -> None:
        self.remove_calls += 1
        if self.raises_on_remove:
            raise RuntimeError("camera removal failed")


def _stub_mount(name: str, width: int, height: int) -> CameraMount:
    return CameraMount(
        name=name,
        position_vehicle=(0.0, -0.6, 1.3),
        direction_vehicle=(0.0, -1.0, 0.0),
        horizontal_fov_deg=60.0,
        vertical_fov_deg=camera_vertical_fov_deg(60.0, (width, height)),
        resolution=(width, height),
        sample_stride=(1, 1),
    )


def test_hybrid_camera_constructor_is_rgb_only_streaming_shared_memory() -> None:
    mount = derive_hybrid_camera_rig(_geometry())["a_pillar_left"]

    kwargs = BeamNgWorker.hybrid_camera_sensor_kwargs(mount)

    assert kwargs["requested_update_time"] == pytest.approx(0.10)
    assert kwargs["resolution"] == (1280, 960)
    assert kwargs["is_using_shared_memory"] is True
    assert kwargs["is_streaming"] is True
    assert kwargs["is_render_colours"] is True
    assert kwargs["is_render_depth"] is False
    assert kwargs["is_render_annotations"] is False
    assert kwargs["is_render_instance"] is False


def test_hybrid_camera_attach_constructs_and_owns_only_the_ordered_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HYBRID attach must leave its six LiDARs out of the Camera path."""

    constructed: list[object] = []

    class CameraStub:
        def __init__(
            self, name: str, bng: object, vehicle: object, **kwargs: object
        ) -> None:
            self.name = name
            self.bng = bng
            self.vehicle = vehicle
            self.kwargs = kwargs
            constructed.append(self)

    monkeypatch.setattr("beamngpy.sensors.Camera", CameraStub)
    worker = BeamNgWorker()
    lidar_sensors = [object() for _ in range(6)]
    worker._sensors = lidar_sensors  # type: ignore[assignment]
    worker._sensor_names = [f"lidar_{index}" for index in range(6)]
    worker._bng = object()  # type: ignore[assignment]
    vehicle = object()

    attached = worker._attach_hybrid_camera_rig(vehicle, _geometry(), "hybrid")

    assert attached == 2
    assert [camera.name for camera in constructed] == [
        "hybrid_a_pillar_left",
        "hybrid_a_pillar_right",
    ]
    assert worker._hybrid_cameras == constructed
    assert worker._hybrid_camera_names == ["a_pillar_left", "a_pillar_right"]
    assert worker._sensors == lidar_sensors
    assert worker._sensor_names == [f"lidar_{index}" for index in range(6)]


def test_cleanup_removes_and_forgets_every_hybrid_camera() -> None:
    worker = BeamNgWorker()
    cameras = (
        RemovableCameraStub(),
        RemovableCameraStub(raises_on_remove=True),
    )
    worker._hybrid_cameras = list(cameras)  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES)
    worker._hybrid_camera_digests = {"a_pillar_left": b"old"}
    worker._hybrid_camera_failures = {"a_pillar_right"}

    worker._cleanup_sensors()

    assert [camera.remove_calls for camera in cameras] == [1, 1]
    assert worker._hybrid_cameras == []
    assert worker._hybrid_camera_names == []
    assert worker._hybrid_camera_digests == {}
    assert worker._hybrid_camera_failures == set()


class VehicleStub:
    def __init__(self) -> None:
        self.state = {
            "pos": (1.0, 2.0, 3.0),
            "dir": (0.0, 1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "vel": (0.0, 5.0, 0.0),
        }

    def poll_sensors(self, *names: str) -> None:
        del names


def _armed_vision_worker(
    cameras: list[StreamingCameraStub],
) -> tuple[BeamNgWorker, list[VisionFrame]]:
    worker = BeamNgWorker()
    frames: list[VisionFrame] = []
    worker._sensor_mode = SENSOR_MODE_VISION
    worker._bng = object()  # type: ignore[assignment]
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._sensors = cameras  # type: ignore[assignment]
    worker._sensor_names = [f"cam_{index}" for index in range(len(cameras))]
    worker._geometry = _geometry()
    worker._palette = SemanticPalette.from_annotations({"STREET": (255, 0, 0)})
    worker._camera_rays = {
        name: build_camera_rays(_stub_mount(name, *camera.resolution))
        for name, camera in zip(worker._sensor_names, cameras)
    }
    worker._vision_eye_height_m = 1.3
    worker.vision_frame_ready.connect(frames.append)
    return worker, frames


def test_the_worker_defaults_to_lidar_mode() -> None:
    assert BeamNgWorker()._sensor_mode == SENSOR_MODE_LIDAR


def test_a_vision_tick_streams_every_camera_and_emits_both_frames() -> None:
    """
    Rung 0.5: a vision tick emits the camera grid's frame AND the BEV frame,
    because the unprojected cloud rejoins the common pipeline -- the same
    tick, the same frame object the LiDAR set produces.
    """
    cameras = [StreamingCameraStub() for _ in range(8)]
    worker, frames = _armed_vision_worker(cameras)
    bev_frames: list[BevFrame] = []
    worker.frame_ready.connect(bev_frames.append)

    worker._poll_once()

    assert [camera.stream_raw_calls for camera in cameras] == [1] * 8
    assert len(frames) == 1
    frame = frames[0]
    assert len(frame.images) == 8
    assert frame.images[0].rgba.shape == (3, 4, 4)
    assert frame.images[0].rgba.dtype == np.uint8
    assert frame.speed_mps == pytest.approx(5.0)
    assert len(bev_frames) == 1
    bev = bev_frames[0]
    assert bev.raw_point_count == 8 * 4 * 3
    assert bev.speed_mps == pytest.approx(5.0)


def test_the_unprojected_cloud_lands_where_the_depth_says_and_is_classified() -> None:
    """
    A wall 20 m down the optical axis of a camera 0.6 m ahead of the node,
    painted STREET in the annotation channel, comes out of the tick as ROAD
    points 20.6 m ahead -- the whole waist, end to end, through the real
    semantic pass.
    """
    camera = StreamingCameraStub(depth_m=20.0, annotation=(255, 0, 0))
    worker, _ = _armed_vision_worker([camera])
    bev_frames: list[BevFrame] = []
    worker.frame_ready.connect(bev_frames.append)
    snapshots: list[object] = []
    worker.perception_ready.connect(snapshots.append)

    worker._poll_once()

    bev = bev_frames[0]
    assert len(bev.road_points) == 12 and len(bev.obstacle_points) == 0
    assert np.allclose(bev.road_points[:, 1], 20.6, atol=0.01)
    assert len(snapshots) == 1
    world = snapshots[0].points_world
    # VehicleStub faces world +Y from (1, 2, 3): the wall is at y = 22.6.
    assert np.allclose(world[:, 1], 22.6, atol=0.01)


def test_a_camera_without_depth_still_reaches_the_grid_but_not_the_cloud() -> None:
    cameras = [StreamingCameraStub(with_depth=False), StreamingCameraStub()]
    worker, frames = _armed_vision_worker(cameras)
    bev_frames: list[BevFrame] = []
    worker.frame_ready.connect(bev_frames.append)

    worker._poll_once()

    assert len(frames[0].images) == 2
    assert bev_frames[0].raw_point_count == 12


def test_vision_mode_offers_driving_by_default_since_milestone_5() -> None:
    """
    Milestone 5's code change: VISION_DRIVING_ENABLED ships True, earned by
    the phase-2 measurements (ground band -1..-2 cm against the LiDAR floor
    out to 60 m, staging ~= 0, zero-mean detection jitter after the
    seen-time centring). The refusal machinery stays -- the closed-gate test
    below pins the other direction -- and TRUST is gated on the live phantom
    checklist, which is driving, not code.
    """
    worker, _ = _armed_vision_worker([StreamingCameraStub()])

    worker.set_aeb(True)
    worker.set_rear_aeb(True)
    worker.set_self_driving(True)
    worker.set_parking_scan(True)

    assert worker._aeb_enabled is True
    assert worker._rear_aeb_enabled is True
    assert worker._self_driving is True
    assert worker._parking_scan is True


def test_porosity_reasons_from_the_tallest_camera_in_vision_mode() -> None:
    """
    The see-through veto needs the eye height of the unit that can see OVER
    a short object. In LiDAR mode that is the roof unit; on the camera rig
    it is the windshield pair, and the LiDAR geometry's roof mount must not
    be consulted for a rig that does not have one.
    """
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    assert worker._porosity_sensor_height(_geometry()) == pytest.approx(1.3)
    worker._sensor_mode = SENSOR_MODE_LIDAR
    assert worker._porosity_sensor_height(_geometry()) == 0.0


def test_the_image_is_a_private_copy_not_the_shared_buffer() -> None:
    """stream_raw hands back a view of the LIVE shared-memory buffer; the
    frame must carry its own bytes or the grid tears as the sim writes."""
    worker, frames = _armed_vision_worker([StreamingCameraStub()])

    worker._poll_once()

    image = frames[0].images[0].rgba
    assert image.flags["OWNDATA"] or image.base.flags["OWNDATA"]
    image_value = int(image[0, 0, 0])
    assert image_value == 17


def test_rereads_of_an_unchanged_buffer_do_not_count_as_acquisition() -> None:
    """The tick re-reads shared memory faster than the cameras update, so the
    acquisition metric counts genuinely NEW frames -- the LiDAR rule. The
    digest is taken on the DEPTH lattice now, which the unprojection gathers
    anyway, so freshness costs nothing extra."""
    camera = StreamingCameraStub()
    worker, frames = _armed_vision_worker([camera])

    worker._poll_once()
    worker._poll_once()
    assert frames[1].acquisition_fps == pytest.approx(0.0)

    camera.repaint()
    worker._poll_once()
    assert frames[2].acquisition_fps > 0.0


def test_a_frames_age_is_counted_from_when_its_depth_last_changed() -> None:
    """
    The simulator stamps nothing, so the only measurable part of a frame's
    age is how long since its buffer changed. The worker keeps that per
    camera and places each camera's cloud from the pose the car had then.
    """
    camera = StreamingCameraStub()
    worker, _ = _armed_vision_worker([camera])

    worker._poll_once()
    first_seen = worker._camera_frame_seen["cam_0"]
    worker._poll_once()
    assert worker._camera_frame_seen["cam_0"] == first_seen

    camera.repaint()
    worker._poll_once()
    assert worker._camera_frame_seen["cam_0"] > first_seen


def test_a_fresh_frames_seen_time_is_centred_between_the_last_two_looks() -> None:
    """
    The digest notices a frame change only ON a tick, so the true change
    time is uniform over the tick that elapsed -- stamping `now` under-ages
    every frame by half a tick on average (~20 ms, 0.22 m of forward
    misplacement at the 40 km/h cap), which the 2026-08-24 fence-run
    regression measured live as +32 +/- 17 ms per unit speed. The midpoint
    of the last two looks zeroes the mean error and halves the worst case.
    """
    camera = StreamingCameraStub()
    worker, _ = _armed_vision_worker([camera])

    worker._poll_once()
    checked_before = worker._camera_frame_checked["cam_0"]
    camera.repaint()
    worker._poll_once()
    checked_after = worker._camera_frame_checked["cam_0"]
    seen = worker._camera_frame_seen["cam_0"]
    assert checked_before < checked_after
    assert seen == pytest.approx((checked_before + checked_after) / 2.0)


def test_the_closed_vision_gate_still_refuses_every_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate ships OPEN now, but it stays the shut-off: closed, all three
    slots must bounce through the WORKER's own refusal -- a guard living
    only in the window is one a queued signal walks straight past."""
    import beamng_lidar_bev.worker as worker_module

    monkeypatch.setattr(worker_module, "VISION_DRIVING_ENABLED", False)
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    answers: list[bool] = []
    worker.self_driving_changed.connect(answers.append)
    worker.aeb_changed.connect(answers.append)
    worker.rear_aeb_changed.connect(answers.append)

    worker.set_self_driving(True)
    worker.set_aeb(True)
    worker.set_rear_aeb(True)

    assert answers == [False, False, False]
    assert worker._self_driving is False
    assert worker._aeb_enabled is False
    assert worker._rear_aeb_enabled is False


def test_the_closed_gate_refuses_the_parking_scan_and_the_parking_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Parking sits behind the SAME gate as self-driving and both brakes -- one
    constant, four slots -- so with the gate shut it must bounce exactly as
    they do, through the WORKER's own refusal. (The original rationale, that
    the camera rig produced nothing to classify, died with rung 0.5: the
    annotation channel fills the marking store in both modes now.)
    """
    import beamng_lidar_bev.worker as worker_module

    monkeypatch.setattr(worker_module, "VISION_DRIVING_ENABLED", False)
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    answers: list[bool] = []
    worker.parking_changed.connect(answers.append)
    worker.parking_drive_changed.connect(answers.append)

    worker.set_parking_scan(True)
    worker.set_parking_drive(True)

    assert answers == [False, False]
    assert worker._parking_scan is False
    assert worker._parking_driving is False


def test_the_parking_refusal_names_the_missing_instrument_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `set_parking_drive` already refused transitively -- no bay can be found
    behind a closed gate, so no selection can match -- but it said "select a
    parking bay", which is unactionable when finding one is impossible.
    """
    import beamng_lidar_bev.worker as worker_module

    monkeypatch.setattr(worker_module, "VISION_DRIVING_ENABLED", False)
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    messages: list[str] = []
    worker.status_changed.connect(lambda _state, detail: messages.append(detail))

    worker.set_parking_scan(True)
    worker.set_parking_drive(True)

    assert len(messages) == 2
    assert all("Vision mode" in message for message in messages)


def test_leaving_vision_mode_lets_the_parking_scan_arm_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is the MODE's, not a latch: it must not outlive the mode."""
    import beamng_lidar_bev.worker as worker_module

    monkeypatch.setattr(worker_module, "VISION_DRIVING_ENABLED", False)
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    worker.set_parking_scan(True)
    assert worker._parking_scan is False

    worker._sensor_mode = SENSOR_MODE_LIDAR
    worker.set_parking_scan(True)

    assert worker._parking_scan is True


def test_switching_mode_mid_stream_reattaches_through_the_one_funnel() -> None:
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    worker._sensor_mode = SENSOR_MODE_LIDAR
    reattaches: list[bool] = []
    modes: list[str] = []
    worker.attach_to_player = lambda: reattaches.append(True)  # type: ignore
    worker.sensor_mode_changed.connect(modes.append)

    worker.set_sensor_mode(SENSOR_MODE_VISION)
    worker.set_sensor_mode(SENSOR_MODE_VISION)  # repeat is a no-op

    assert reattaches == [True]
    assert modes == [SENSOR_MODE_VISION]


def test_switching_to_hybrid_mid_stream_reattaches_once() -> None:
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    worker._sensor_mode = SENSOR_MODE_LIDAR
    calls: list[bool] = []
    modes: list[str] = []
    worker.attach_to_player = lambda: calls.append(True)  # type: ignore
    worker.sensor_mode_changed.connect(modes.append)

    worker.set_sensor_mode(SENSOR_MODE_HYBRID)
    worker.set_sensor_mode(SENSOR_MODE_HYBRID)

    assert calls == [True]
    assert modes == [SENSOR_MODE_HYBRID]


def test_a_mode_change_while_idle_only_records_the_choice() -> None:
    worker = BeamNgWorker()
    reattaches: list[bool] = []
    worker.attach_to_player = lambda: reattaches.append(True)  # type: ignore

    worker.set_sensor_mode(SENSOR_MODE_VISION)

    assert worker._sensor_mode == SENSOR_MODE_VISION
    assert reattaches == []


def test_cleanup_forgets_the_camera_digests() -> None:
    """A re-attach must start with a clean freshness ledger, or the first
    frames of the new session read as stale re-reads."""
    worker, _ = _armed_vision_worker([StreamingCameraStub()])
    worker._poll_once()
    assert worker._camera_digests
    assert worker._camera_rays

    worker._cleanup_sensors()

    assert worker._camera_digests == {}
    assert worker._camera_frame_seen == {}
    assert worker._camera_rays == {}


def test_shrinking_an_image_is_resampled_and_magnifying_it_is_not() -> None:
    """
    `drawImage` is bilinear whatever the scale, which is right going up and
    wrong coming down: shrinking 1280x960 into a 400x300 tile steps over ~3
    source pixels per output pixel, so asphalt and foliage alias into speckle.
    Reported live as camera "noise" -- and it appeared when the rig went from
    640x480 (upscaled, merely soft) to 1280x960 (downscaled).
    """
    assert wants_prescale(1280, 960, 400.0, 300.0), "a grid tile shrinks 3x"
    assert wants_prescale(1280, 960, 900.0, 675.0), "even the focused pane shrinks"
    assert not wants_prescale(640, 480, 900.0, 675.0), "magnifying needs no resample"
    assert not wants_prescale(1280, 960, 1280.0, 960.0), "1:1 needs no resample"


def test_a_single_shrunk_axis_still_resamples() -> None:
    """A pane can be wide and short; either axis shrinking causes the aliasing."""
    assert wants_prescale(1280, 960, 1400.0, 300.0)
    assert wants_prescale(1280, 960, 400.0, 1000.0)


def test_a_degenerate_size_never_resamples() -> None:
    """Zero-sized panes happen mid-layout; they must not reach QImage.scaled."""
    assert not wants_prescale(0, 0, 400.0, 300.0)
    assert not wants_prescale(1280, 960, 0.0, 300.0)
    assert not wants_prescale(1280, 960, 400.0, 0.0)


def test_the_cameras_fourth_byte_is_not_treated_as_opacity() -> None:
    """
    BeamNG's colour buffer carries something scene-dependent in its fourth
    byte, not opacity: measured on one 1280x960 frame it ran 40..255 with only
    50.75% of pixels at 255. Read as RGBA, Qt composites every pixel against
    the tile background and the view fills with black speckle -- the reported
    camera "noise". RGBX has the identical layout and ignores that byte.

    Measured by PAINTING a multi-pixel image, which is the only form that
    shows it: `pixelColor` reports the stored byte under either format, and a
    1x1 image blends under BOTH (a degenerate path that misled this test once).
    On the real frame: mean error against the true colour 26.08 as RGBA, 0.00
    as RGBX.
    """
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QPainter

    from beamng_lidar_bev.vision_view import _IMAGE_FORMAT

    size = 8
    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    alpha = np.full((size, size), 40, dtype=np.uint8)
    buffer = np.dstack([rgb, alpha]).copy()

    def painted(fmt: QImage.Format) -> float:
        source = QImage(buffer.data, size, size, size * 4, fmt)
        canvas = QImage(size, size, QImage.Format.Format_RGB32)
        canvas.fill(0xFF000000)
        painter = QPainter(canvas)
        painter.drawImage(QRectF(0, 0, size, size), source)
        painter.end()
        return float(canvas.pixelColor(size // 2, size // 2).red())

    assert painted(_IMAGE_FORMAT) == 128.0, (
        "the camera's own colour must survive painting untouched"
    )
    blended = painted(QImage.Format.Format_RGBA8888)
    assert blended < 60.0, (
        "guard: RGBA really does blend with that byte -- a mid-grey pixel goes "
        f"nearly black over a dark tile (measured {blended}), which is the "
        "speckle. Without this the test could pass against any format."
    )


def _bearing_deg(direction: tuple[float, float, float]) -> float:
    """Compass bearing in the vehicle frame: forward 0, LEFT +90, rear 180."""
    return math.degrees(math.atan2(direction[0], -direction[1]))


def _uncovered_bearings(rig, step_deg: float = 0.1) -> list[tuple[float, float]]:
    """Runs of bearing no camera sees, as (start, end) in degrees."""
    slots = int(round(360.0 / step_deg))
    covered = [False] * slots
    for mount in rig.values():
        centre = _bearing_deg(mount.direction_vehicle)
        half = mount.horizontal_fov_deg / 2.0
        first = int(round((centre - half) / step_deg))
        last = int(round((centre + half) / step_deg))
        for index in range(first, last + 1):
            covered[index % slots] = True
    runs: list[tuple[float, float]] = []
    index = 0
    while index < slots:
        if not covered[index]:
            start = index
            while index < slots and not covered[index]:
                index += 1
            runs.append((start * step_deg, (index - 1) * step_deg))
        else:
            index += 1
    return runs


def test_the_rig_leaves_no_gap_all_the_way_round() -> None:
    """
    The eight apertures have to TILE THE CIRCLE, and at the first FOVs they did
    not: 80-degree pillars and 60-degree repeaters aimed 30 off rearward left a
    **24.5-degree hole per side at bearings 95-120** -- over the driver's
    shoulder, the blind spot the repeaters exist for. Reported live as the side
    cameras feeling too narrow, which turned out to be measurable rather than a
    matter of taste.
    """
    geometry = _offset_geometry()

    assert _uncovered_bearings(derive_camera_rig(geometry)) == []


def test_the_side_gap_closes_with_margin_not_by_a_hair() -> None:
    """
    Closing it exactly would be one per-vehicle tweak away from reopening, and
    the mounts are metres apart so their coverage is not purely angular anyway.
    The pillar and repeater on each side must genuinely overlap.
    """
    rig = derive_camera_rig(_offset_geometry())
    pillar, repeater = rig["pillar_left"], rig["repeater_left"]

    pillar_outer = (
        _bearing_deg(pillar.direction_vehicle) + pillar.horizontal_fov_deg / 2.0
    )
    repeater_inner = (
        _bearing_deg(repeater.direction_vehicle) - repeater.horizontal_fov_deg / 2.0
    )

    assert pillar_outer - repeater_inner >= 5.0, (
        f"pillar reaches {pillar_outer:.1f} deg, repeater starts at "
        f"{repeater_inner:.1f} -- they must overlap, not merely touch"
    )


def test_the_outboard_cameras_are_symmetric_about_the_body_not_the_node() -> None:
    """
    A pair placed against its own side's surface is symmetric about the BODY
    even though the raw x values look lopsided, because the reference node is
    off centre (measured 0.16 m on the vivace).
    """
    geometry = _offset_geometry()
    rig = derive_camera_rig(geometry)
    centre_x = (geometry.left_m - geometry.right_m) / 2.0

    for left_name, right_name in (
        ("pillar_left", "pillar_right"),
        ("repeater_left", "repeater_right"),
    ):
        left, right = rig[left_name], rig[right_name]
        offsets = (
            left.position_vehicle[0] - centre_x,
            right.position_vehicle[0] - centre_x,
        )
        assert offsets[0] == pytest.approx(-offsets[1], abs=1e-9), (
            f"{left_name}/{right_name} sit at {offsets} from the body centreline"
        )
        assert left.position_vehicle[1] == pytest.approx(right.position_vehicle[1])
        assert left.position_vehicle[2] == pytest.approx(right.position_vehicle[2])


def test_a_centreline_camera_sits_on_the_body_centre_not_the_reference_node() -> None:
    """
    The node is not the body centre, so `x = 0` is 0.16 m off on the vivace --
    the same defect the WORLD ego model and the AEB corridor each had. A bumper
    camera looking straight ahead must do so from the middle of the car.
    """
    geometry = _offset_geometry()
    rig = derive_camera_rig(geometry)
    centre_x = (geometry.left_m - geometry.right_m) / 2.0
    assert centre_x != 0.0, "guard: this vehicle must be off-centre to test it"

    for name in ("front_bumper", "rear"):
        assert rig[name].position_vehicle[0] == pytest.approx(centre_x), (
            f"{name} is on the reference node, not the body centreline"
        )
    # The windshield pair straddles that centre rather than the node.
    straddle = (
        rig["front_main"].position_vehicle[0] + rig["front_wide"].position_vehicle[0]
    ) / 2.0
    assert straddle == pytest.approx(centre_x)
