from __future__ import annotations

import math

import numpy as np
import pytest

from beamng_lidar_bev.geometry import (
    CAMERA_NAMES,
    camera_vertical_fov_deg,
    derive_camera_rig,
)
from beamng_lidar_bev.models import VehicleGeometry, VisionFrame
from beamng_lidar_bev.vision_view import grid_dimensions
from beamng_lidar_bev.worker import (
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


# --- The rig ------------------------------------------------------------------


def test_the_rig_is_the_eight_hw4_cameras_in_a_stable_order() -> None:
    rig = derive_camera_rig(_geometry())
    assert tuple(rig) == CAMERA_NAMES
    assert len(rig) == 8


def test_forward_is_negative_y_never_the_intuitive_positive() -> None:
    """
    Vehicle space is +X left, +Y REARWARD: a camera built with the intuitive
    dir=(0, 1, 0) films the rear seats. Verified live in the feasibility work
    across all mounts, so the sign is pinned here.
    """
    rig = derive_camera_rig(_geometry())
    for name in ("front_main", "front_wide", "front_bumper"):
        assert rig[name].direction_vehicle == (0.0, -1.0, 0.0)
    assert rig["rear"].direction_vehicle == (0.0, 1.0, 0.0)


def test_side_cameras_sit_outside_the_body_shell() -> None:
    """There is no hide-ego flag: a mount inside the glasshouse films the
    cabin (measured 68% CAR on the first windshield attempt)."""
    geometry = _geometry()
    rig = derive_camera_rig(geometry)
    for name in ("pillar_left", "repeater_left"):
        assert rig[name].position_vehicle[0] > geometry.left_m
    for name in ("pillar_right", "repeater_right"):
        assert rig[name].position_vehicle[0] < -geometry.right_m
    assert rig["front_bumper"].position_vehicle[1] < -geometry.front_m
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


# --- The worker's vision path -------------------------------------------------


class StreamingCameraStub:
    """A Camera the way the worker drives one: stream_raw only, never poll."""

    def __init__(self, width: int = 4, height: int = 3) -> None:
        self.resolution = (width, height)
        self.stream_raw_calls = 0
        self._pixel = 17

    def repaint(self) -> None:
        """Change the buffer, as a new simulator frame would."""
        self._pixel = (self._pixel + 41) % 256

    def stream_raw(self) -> dict[str, bytes]:
        self.stream_raw_calls += 1
        width, height = self.resolution
        return {"colour": bytes([self._pixel]) * (width * height * 4)}

    def poll(self) -> None:
        raise AssertionError(
            "The display loop must not issue blocking camera polls"
        )


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
    worker.vision_frame_ready.connect(frames.append)
    return worker, frames


def test_the_worker_defaults_to_lidar_mode() -> None:
    assert BeamNgWorker()._sensor_mode == SENSOR_MODE_LIDAR


def test_a_vision_tick_streams_every_camera_and_emits_one_frame() -> None:
    cameras = [StreamingCameraStub() for _ in range(8)]
    worker, frames = _armed_vision_worker(cameras)
    bev_frames: list[object] = []
    worker.frame_ready.connect(bev_frames.append)

    worker._poll_once()

    assert [camera.stream_raw_calls for camera in cameras] == [1] * 8
    assert bev_frames == []
    assert len(frames) == 1
    frame = frames[0]
    assert len(frame.images) == 8
    assert frame.images[0].rgba.shape == (3, 4, 4)
    assert frame.images[0].rgba.dtype == np.uint8
    assert frame.speed_mps == pytest.approx(5.0)


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
    acquisition metric counts genuinely NEW frames -- the LiDAR rule."""
    camera = StreamingCameraStub()
    worker, frames = _armed_vision_worker([camera])

    worker._poll_once()
    worker._poll_once()
    assert frames[1].acquisition_fps == pytest.approx(0.0)

    camera.repaint()
    worker._poll_once()
    assert frames[2].acquisition_fps > 0.0


def test_vision_mode_refuses_self_driving_and_both_aebs() -> None:
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

    worker._cleanup_sensors()

    assert worker._camera_digests == {}
