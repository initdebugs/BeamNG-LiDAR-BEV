from __future__ import annotations

import math
import time
from hashlib import blake2b
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_lidar_bev import worker as worker_module
from beamng_lidar_bev.config import (
    CAMERA_NEAR_FAR_PLANES,
    HYBRID_CAMERA_BODY_CLEARANCE_M,
)
from beamng_lidar_bev.geometry import (
    HYBRID_CAMERA_NAMES,
    camera_vertical_fov_deg,
    derive_hybrid_camera_rig,
)
from beamng_lidar_bev.models import (
    BevFrame,
    CameraMount,
    SensorMount,
    VehicleGeometry,
    VisionFrame,
)
from beamng_lidar_bev.semantics import SemanticPalette
from beamng_lidar_bev.vision_view import (
    grid_dimensions,
    toggle_focus,
    wants_prescale,
)
from beamng_lidar_bev.worker import (
    SENSOR_MODE_HYBRID,
    SENSOR_MODE_LIDAR,
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


def _bearing_deg(direction: tuple[float, float, float]) -> float:
    """Compass bearing in the vehicle frame: forward 0, LEFT +90, rear 180."""
    return math.degrees(math.atan2(direction[0], -direction[1]))


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


def test_the_vertical_fov_is_derived_from_the_horizontal_and_aspect() -> None:
    # The constructor takes field_of_view_y; the rig is designed in horizontal
    # apertures, so the projection has to run backwards through the aspect.
    vfov = camera_vertical_fov_deg(90.0, (640, 480))
    assert vfov == pytest.approx(
        math.degrees(2.0 * math.atan(math.tan(math.radians(45.0)) * 0.75))
    )
    for mount in derive_hybrid_camera_rig(_geometry()).values():
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


def test_a_camera_pair_lays_out_left_and_right_on_any_landscape_pane() -> None:
    # Stacking a left/right pair puts the left camera ABOVE the right one,
    # which reads as nothing at all -- so on a pane with the width for it the
    # spatial reading outranks the area search.
    for size in ((1600.0, 700.0), (1000.0, 1000.0), (900.0, 640.0)):
        assert grid_dimensions(2, *size) == (1, 2)


def test_a_portrait_pane_stacks_the_pair_rather_than_shrinking_it() -> None:
    # Forced side by side on a tall pane the tiles lose 3.7x their pixels, and
    # nothing about the layout is worth that.
    assert grid_dimensions(2, 700.0, 1600.0) == (2, 1)


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


class ExplodingCameraStub(StreamingCameraStub):
    def stream_raw(self) -> dict[str, bytes]:
        self.stream_raw_calls += 1
        raise RuntimeError("camera gone")


class RemovableCameraStub(StreamingCameraStub):
    def __init__(self, raises_on_remove: bool = False) -> None:
        super().__init__()
        self.raises_on_remove = raises_on_remove
        self.remove_calls = 0

    def remove(self) -> None:
        self.remove_calls += 1
        if self.raises_on_remove:
            raise RuntimeError("camera removal failed")


class EpisodicCameraStub(StreamingCameraStub):
    """A camera that can fail in discrete episodes between good frames."""

    def __init__(self) -> None:
        super().__init__()
        self._failures_remaining = 0

    def fail_next(self, attempts: int) -> None:
        self._failures_remaining = attempts

    def stream_raw(self) -> dict[str, bytes]:
        if self._failures_remaining:
            self._failures_remaining -= 1
            raise RuntimeError("camera gone")
        return super().stream_raw()


class InvalidColourCameraStub(StreamingCameraStub):
    """A streaming camera whose colour channel is absent or not a pixel buffer."""

    def __init__(self, raw: dict[str, object]) -> None:
        super().__init__()
        self._raw = raw

    def stream_raw(self) -> dict[str, bytes]:
        self.stream_raw_calls += 1
        return self._raw  # type: ignore[return-value]


class ByteMutatingCameraStub(StreamingCameraStub):
    """A shared colour buffer whose caller can change one exact byte."""

    def __init__(self) -> None:
        super().__init__()
        width, height = self.resolution
        self._colour = bytearray([17]) * (width * height * 4)

    def mutate_byte(self, index: int, value: int) -> None:
        self._colour[index] = value

    def stream_raw(self) -> dict[str, bytes]:
        self.stream_raw_calls += 1
        return {"colour": self._colour}  # type: ignore[return-value]


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
    assert kwargs["pos"] == mount.position_vehicle
    assert kwargs["dir"] == mount.direction_vehicle
    assert kwargs["field_of_view_y"] == mount.vertical_fov_deg
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


class _AttachTimerStub:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _HybridAttachVehicleStub:
    def __init__(self) -> None:
        self.model = "test_vehicle"
        self.state = {
            "pos": (0.0, 0.0, 0.0),
            "dir": (0.0, 1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "vel": (0.0, 0.0, 0.0),
        }

    def connect(self, bng: object) -> None:
        del bng

    def poll_sensors(self, *names: str) -> None:
        del names

    def get_bbox(self) -> dict[str, tuple[float, float, float]]:
        return {}

    def is_connected(self) -> bool:
        return False


def _hybrid_attach_geometry() -> VehicleGeometry:
    lidar_names = ("front", "left", "right", "rear", "roof", "road")
    return VehicleGeometry(
        ground_z_vehicle=-0.3,
        left_m=0.9,
        right_m=0.9,
        front_m=2.2,
        rear_m=2.3,
        height_m=1.45,
        mounts={
            name: SensorMount(
                name=name,
                position_vehicle=(float(index), 0.0, 1.0),
                direction_vehicle=(0.0, -1.0, 0.0),
            )
            for index, name in enumerate(lidar_names)
        },
    )


def _hybrid_attach_worker(
    monkeypatch: pytest.MonkeyPatch, *, fail_second_camera: bool = False
) -> tuple[
    BeamNgWorker,
    _AttachTimerStub,
    list[tuple[str, str]],
    list[object],
    list[object],
]:
    """Offline BeamNG construction boundary for attach-to-player tests."""

    constructor_order: list[tuple[str, str]] = []
    lidars: list[object] = []
    cameras: list[object] = []

    class LidarStub:
        def __init__(
            self, name: str, bng: object, vehicle: object, **kwargs: object
        ) -> None:
            del bng, vehicle, kwargs
            self.name = name
            self.remove_calls = 0
            lidars.append(self)
            constructor_order.append(("Lidar", name))

        def remove(self) -> None:
            self.remove_calls += 1

    class CameraStub:
        def __init__(
            self, name: str, bng: object, vehicle: object, **kwargs: object
        ) -> None:
            del bng, vehicle, kwargs
            self.name = name
            self.remove_calls = 0
            # The sensor-origin probe is a throwaway camera at (0, 0, 0); it
            # is not part of the rig and must not appear in the build order.
            self.is_origin_probe = name.startswith("origin_")
            if self.is_origin_probe:
                return
            constructor_order.append(("Camera", name))
            if fail_second_camera and name.endswith("_a_pillar_right"):
                raise RuntimeError("second hybrid camera failed")
            cameras.append(self)

        def get_position(self) -> tuple[float, float, float]:
            # Reports the reference node, so the measured correction is zero
            # and the mounts keep the offline suite's numbers.
            return (0.0, 0.0, 0.0)

        def remove(self) -> None:
            self.remove_calls += 1

    vehicle = _HybridAttachVehicleStub()
    worker = BeamNgWorker()
    timer = _AttachTimerStub()
    worker._poll_timer = timer  # type: ignore[assignment]
    worker._sensor_mode = SENSOR_MODE_HYBRID
    worker._bng = SimpleNamespace(
        vehicles=SimpleNamespace(
            get_player_vehicle_id=lambda: {"vid": "ego", "id": 1},
            get_current=lambda include_config: {"ego": vehicle},
        )
    )  # type: ignore[assignment]
    monkeypatch.setattr("beamngpy.sensors.Lidar", LidarStub)
    monkeypatch.setattr("beamngpy.sensors.Camera", CameraStub)
    monkeypatch.setattr(
        worker_module,
        "derive_vehicle_geometry",
        lambda state, bbox, **kwargs: _hybrid_attach_geometry(),
    )
    monkeypatch.setattr(worker, "_attach_electrics", lambda vehicle: None)
    monkeypatch.setattr(worker, "_load_annotations", lambda: {"STREET": (1, 2, 3)})
    return worker, timer, constructor_order, lidars, cameras


def test_attach_to_player_hybrid_builds_lidars_then_cameras_and_starts_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, timer, constructor_order, lidars, cameras = _hybrid_attach_worker(
        monkeypatch
    )
    worker._vision_streaming_since = 0.0
    worker._logged_vision_check = True
    worker._logged_vision_silence = True
    statuses: list[tuple[str, str]] = []
    worker.status_changed.connect(
        lambda state, detail: statuses.append((state, detail))
    )

    worker.attach_to_player()

    expected_names = (
        "front",
        "left",
        "right",
        "rear",
        "roof",
        "road",
        "a_pillar_left",
        "a_pillar_right",
    )
    assert [kind for kind, _name in constructor_order] == ["Lidar"] * 6 + [
        "Camera",
        "Camera",
    ]
    assert all(
        name.endswith(f"_{expected}")
        for (_kind, name), expected in zip(constructor_order, expected_names)
    )
    assert worker._sensors == lidars
    assert worker._hybrid_cameras == cameras
    assert worker._vision_streaming_since is not None
    assert worker._vision_streaming_since > 0.0
    assert worker._logged_vision_check is False
    assert worker._logged_vision_silence is False
    assert (
        "STREAMING",
        "6 LiDAR sensors + 2 cameras active on ego",
    ) in statuses
    assert timer.start_calls == 1


def test_attach_to_player_hybrid_failure_removes_every_constructed_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, timer, constructor_order, lidars, cameras = _hybrid_attach_worker(
        monkeypatch, fail_second_camera=True
    )
    worker._hybrid_camera_digests = {"old": b"stale"}
    worker._hybrid_camera_failures = {"old"}
    worker._hybrid_camera_frames = {"old": np.zeros((2, 2, 4), dtype=np.uint8)}
    statuses: list[tuple[str, str]] = []
    worker.status_changed.connect(
        lambda state, detail: statuses.append((state, detail))
    )

    worker.attach_to_player()

    assert [kind for kind, _name in constructor_order] == ["Lidar"] * 6 + [
        "Camera",
        "Camera",
    ]
    assert len(cameras) == 1
    assert [sensor.remove_calls for sensor in lidars] == [1] * 6
    assert cameras[0].remove_calls == 1
    assert worker._sensors == []
    assert worker._sensor_names == []
    assert worker._hybrid_cameras == []
    assert worker._hybrid_camera_names == []
    assert worker._hybrid_camera_digests == {}
    assert worker._hybrid_camera_failures == set()
    assert worker._hybrid_camera_frames == {}
    assert worker._geometry is None
    assert worker._palette is None
    assert all(state != "STREAMING" for state, _detail in statuses)
    assert statuses[-1][0] == "READY"
    assert timer.start_calls == 0


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


def _armed_hybrid_camera_worker(
    cameras: list[StreamingCameraStub],
) -> BeamNgWorker:
    worker = BeamNgWorker()
    worker._sensor_mode = SENSOR_MODE_HYBRID
    worker._hybrid_cameras = cameras  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES[: len(cameras)])
    worker._vision_streaming_since = time.perf_counter()
    return worker


def test_hybrid_rgb_acquisition_preserves_order_and_owns_the_bytes() -> None:
    cameras = [StreamingCameraStub(), StreamingCameraStub()]
    worker = _armed_hybrid_camera_worker(cameras)

    images = worker._acquire_hybrid_camera_images(time.perf_counter())

    assert [image.name for image in images] == list(HYBRID_CAMERA_NAMES)
    assert [camera.stream_raw_calls for camera in cameras] == [1, 1]
    assert all(image.rgba.shape == (3, 4, 4) for image in images)
    assert all(image.rgba.dtype == np.uint8 for image in images)
    assert all(
        image.rgba.flags["OWNDATA"] or image.rgba.base.flags["OWNDATA"]
        for image in images
    )


def test_unchanged_hybrid_colour_is_not_re_copied_and_the_tile_survives() -> None:
    camera = StreamingCameraStub()
    worker = _armed_hybrid_camera_worker([camera])

    first = worker._acquire_hybrid_camera_images(time.perf_counter())
    held = first[0].rgba
    # A tick on which nothing changed re-shows the frame already held rather
    # than copying 4.9 MB again -- and the tile must not drop out of the grid.
    second = worker._acquire_hybrid_camera_images(time.perf_counter())
    assert [image.name for image in second] == [HYBRID_CAMERA_NAMES[0]]
    assert second[0].rgba is held

    camera.repaint()
    third = worker._acquire_hybrid_camera_images(time.perf_counter())
    assert third[0].rgba is not held


def test_a_buffer_inside_the_digest_budget_is_sampled_whole() -> None:
    """
    The freshness test reads a BYTE BUDGET, not the whole frame -- digesting
    4.9 MB twice a tick cost 9.8 ms of the 40 ms tick. A buffer smaller than
    the budget is still read whole, so a single changed byte is seen.
    """
    camera = ByteMutatingCameraStub()
    worker = _armed_hybrid_camera_worker([camera])

    first = worker._acquire_hybrid_camera_images(time.perf_counter())
    held = first[0].rgba
    assert worker._acquire_hybrid_camera_images(time.perf_counter())[0].rgba is held
    camera.mutate_byte(1, 99)
    assert (
        worker._acquire_hybrid_camera_images(time.perf_counter())[0].rgba
        is not held
    )


def test_the_digest_budget_bounds_what_a_full_size_frame_costs() -> None:
    # 1280x960x4 is 4.9 MB; the sample must stay near the budget however big
    # the frame is, or the cost this exists to remove comes straight back.
    step = max(1, (1280 * 960 * 4) // worker_module._CAMERA_DIGEST_BYTES)
    assert len(range(0, 1280 * 960 * 4, step)) <= 2 * worker_module._CAMERA_DIGEST_BYTES


def test_hybrid_malformed_colour_buffers_are_omitted_without_failing() -> None:
    cameras = [
        InvalidColourCameraStub({}),
        InvalidColourCameraStub({"colour": b"short"}),
    ]
    worker = _armed_hybrid_camera_worker(cameras)

    images = worker._acquire_hybrid_camera_images(time.perf_counter())

    assert images == []
    assert worker._poll_failures == 0
    # Recorded, not silently dropped: a camera whose buffer is permanently
    # unusable vanishes from the grid, and without this nothing in the log
    # ever mentions it again.
    assert worker._hybrid_camera_failures == set(HYBRID_CAMERA_NAMES)


def test_hybrid_non_buffer_colour_is_omitted_without_failing() -> None:
    worker = _armed_hybrid_camera_worker(
        [InvalidColourCameraStub({"colour": object()})]
    )

    images = worker._acquire_hybrid_camera_images(time.perf_counter())

    assert images == []
    assert worker._poll_failures == 0


def test_typed_hybrid_colour_with_the_wrong_byte_size_does_not_hide_a_peer() -> None:
    malformed = InvalidColourCameraStub(
        {"colour": np.zeros(48, dtype=np.uint16)}
    )
    worker = _armed_hybrid_camera_worker([malformed, StreamingCameraStub()])

    images = worker._acquire_hybrid_camera_images(time.perf_counter())

    assert [image.name for image in images] == ["a_pillar_right"]
    assert worker._poll_failures == 0
    assert worker._hybrid_camera_failures == {"a_pillar_left"}


def test_one_hybrid_camera_failure_does_not_hide_the_other() -> None:
    failed = EpisodicCameraStub()
    failed.fail_next(1)
    worker = _armed_hybrid_camera_worker([failed, StreamingCameraStub()])

    images = worker._acquire_hybrid_camera_images(time.perf_counter())

    assert [image.name for image in images] == ["a_pillar_right"]
    assert worker._hybrid_camera_failures == {"a_pillar_left"}
    assert worker._poll_failures == 0


def test_hybrid_camera_recovers_and_only_warns_once_per_failure_episode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    camera = EpisodicCameraStub()
    worker = _armed_hybrid_camera_worker([camera])
    caplog.set_level("WARNING", logger=worker_module.__name__)

    camera.fail_next(2)
    worker._acquire_hybrid_camera_images(time.perf_counter())
    worker._acquire_hybrid_camera_images(time.perf_counter())
    recovered = worker._acquire_hybrid_camera_images(time.perf_counter())
    camera.fail_next(1)
    worker._acquire_hybrid_camera_images(time.perf_counter())

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Hybrid camera a_pillar_left failed" in record.getMessage()
    ]
    assert [image.name for image in recovered] == ["a_pillar_left"]
    assert worker._hybrid_camera_failures == {"a_pillar_left"}
    assert len(warnings) == 2
    assert worker._poll_failures == 0


def test_hybrid_liveness_reports_the_colour_only_channel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = _armed_hybrid_camera_worker([StreamingCameraStub()])
    worker._vision_streaming_since = time.perf_counter() - 0.25
    caplog.set_level("INFO", logger=worker_module.__name__)

    worker._acquire_hybrid_camera_images(time.perf_counter())

    assert any(
        "Vision check: first fresh frames" in record.getMessage()
        and record.getMessage().endswith("| colour")
        for record in caplog.records
    )


def test_hybrid_liveness_warns_about_silent_colour_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = _armed_hybrid_camera_worker([InvalidColourCameraStub({})])
    worker._vision_streaming_since = 0.0
    caplog.set_level("WARNING", logger=worker_module.__name__)

    worker._acquire_hybrid_camera_images(
        worker_module._VISION_SILENCE_WARN_S + 0.1
    )

    assert any(
        "no camera has delivered a new colour frame" in record.getMessage()
        and "HYBRID_CAMERA_UPDATE_TIME_S" in record.getMessage()
        for record in caplog.records
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


class HybridLidarStub:
    def __init__(self, x: float) -> None:
        self.x = x
        self.stream_calls = 0

    def stream(self) -> dict[str, np.ndarray]:
        self.stream_calls += 1
        return {
            "pointCloud": np.asarray(((self.x, 12.0, 3.0),), dtype=np.float32),
            "colours": np.asarray(((255, 0, 0),), dtype=np.uint8),
        }


def _armed_hybrid_tick_worker(
    cameras: list[StreamingCameraStub],
) -> tuple[BeamNgWorker, list[HybridLidarStub]]:
    lidars = [HybridLidarStub(float(index)) for index in range(6)]
    worker = BeamNgWorker()
    worker._sensor_mode = SENSOR_MODE_HYBRID
    worker._bng = object()  # type: ignore[assignment]
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._sensors = lidars  # type: ignore[assignment]
    worker._sensor_names = [f"lidar_{index}" for index in range(6)]
    worker._hybrid_cameras = cameras  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES)
    worker._geometry = _geometry()
    worker._palette = SemanticPalette.from_annotations(
        {"STREET": (255, 0, 0), "CAR": (0, 255, 0)}
    )
    return worker, lidars


def _streaming_worker() -> BeamNgWorker:
    """A worker with a live sensor set, for the mode-switch funnel tests."""
    worker = BeamNgWorker()
    worker._bng = object()  # type: ignore[assignment]
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._sensors = [HybridLidarStub(0.0)]  # type: ignore[assignment]
    worker._sensor_names = ["front"]
    worker._geometry = _geometry()
    worker._palette = SemanticPalette.from_annotations({"STREET": (255, 0, 0)})
    return worker


def test_the_worker_defaults_to_lidar_mode() -> None:
    assert BeamNgWorker()._sensor_mode == SENSOR_MODE_LIDAR


def test_hybrid_tick_emits_camera_feed_but_only_lidar_perception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = [
        StreamingCameraStub(depth_m=1.0, annotation=(0, 255, 0)),
        StreamingCameraStub(depth_m=250.0, annotation=(0, 255, 0)),
    ]
    worker, lidars = _armed_hybrid_tick_worker(cameras)
    # The RGB buffers are deliberately unchanged: LiDAR freshness must still
    # advance HYBRID's acquisition clock.
    unchanged = blake2b(bytes([17]) * (4 * 3 * 4), digest_size=16).digest()
    worker._hybrid_camera_digests = {
        name: unchanged for name in HYBRID_CAMERA_NAMES
    }
    bev_frames: list[BevFrame] = []
    vision_frames: list[VisionFrame] = []
    snapshots: list[object] = []
    extraction_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    planning_inputs: list[np.ndarray] = []
    aeb_inputs: list[np.ndarray] = []
    parking_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    planner_band = np.asarray(((2.0, 9.0),), dtype=np.float32)
    aeb_band = np.asarray(((-2.0, 11.0),), dtype=np.float32)

    def capture_extraction(
        bev: np.ndarray,
        heights: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        extraction_inputs.append((bev.copy(), heights.copy()))
        return planner_band, aeb_band

    def capture_plan(
        state: dict[str, object],
        obstacles: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> None:
        del state, args, kwargs
        planning_inputs.append(obstacles)

    def capture_aeb(
        system: object,
        obstacles: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> None:
        del system, args, kwargs
        aeb_inputs.append(obstacles)

    def capture_parking(
        state: dict[str, object],
        bev: np.ndarray,
        heights: np.ndarray,
        materials: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> tuple[()]:
        del state, args, kwargs
        parking_inputs.append((bev.copy(), heights.copy(), materials.copy()))
        return ()

    monkeypatch.setattr(worker_module, "geometric_obstacle_sets", capture_extraction)
    worker._compute_plan = capture_plan  # type: ignore[method-assign]
    worker._compute_aeb = capture_aeb  # type: ignore[method-assign]
    worker._scan_for_parking = capture_parking  # type: ignore[method-assign]
    worker._self_driving = True
    worker._aeb_enabled = True
    worker._parking_scan = True
    worker.frame_ready.connect(bev_frames.append)
    worker.vision_frame_ready.connect(vision_frames.append)
    worker.perception_ready.connect(snapshots.append)

    worker._poll_once()

    expected_world = np.asarray(
        [(float(index), 12.0, 3.0) for index in range(6)], dtype=np.float32
    )
    expected_bev = np.column_stack(
        (np.arange(-1.0, 5.0, dtype=np.float32), np.full(6, 10.0, np.float32))
    )
    assert [lidar.stream_calls for lidar in lidars] == [1] * 6
    assert [camera.stream_raw_calls for camera in cameras] == [1, 1]
    assert len(worker._frame_times) == 1
    assert len(bev_frames) == len(vision_frames) == len(snapshots) == 1
    assert bev_frames[0].raw_point_count == 6
    assert np.array_equal(bev_frames[0].road_points, expected_bev)
    assert bev_frames[0].obstacle_points.shape == (0, 2)
    assert np.array_equal(snapshots[0].points_world, expected_world)
    assert np.array_equal(snapshots[0].semantic_groups, np.zeros(6, np.uint8))
    assert np.array_equal(snapshots[0].surface_materials, np.ones(6, np.uint8))
    assert [image.name for image in vision_frames[0].images] == list(
        HYBRID_CAMERA_NAMES
    )
    assert np.array_equal(extraction_inputs[0][0], expected_bev)
    assert np.array_equal(extraction_inputs[0][1], np.zeros(6, np.float32))
    assert planning_inputs == [planner_band]
    assert aeb_inputs == [aeb_band]
    assert np.array_equal(parking_inputs[0][0], expected_bev)
    assert np.array_equal(parking_inputs[0][1], np.zeros(6, np.float32))
    assert np.array_equal(parking_inputs[0][2], np.ones(6, np.uint8))


def test_hybrid_tick_contains_one_camera_failure_beside_lidar_frames() -> None:
    cameras = [ExplodingCameraStub(), StreamingCameraStub()]
    worker, lidars = _armed_hybrid_tick_worker(cameras)
    worker._first_failure_at = 0.0
    bev_frames: list[BevFrame] = []
    vision_frames: list[VisionFrame] = []
    snapshots: list[object] = []
    stopped: list[None] = []
    fatals: list[str] = []
    worker.frame_ready.connect(bev_frames.append)
    worker.vision_frame_ready.connect(vision_frames.append)
    worker.perception_ready.connect(snapshots.append)
    worker.sensors_stopped.connect(lambda: stopped.append(None))
    worker.fatal_error.connect(fatals.append)

    worker._poll_once()

    assert [lidar.stream_calls for lidar in lidars] == [1] * 6
    assert [camera.stream_raw_calls for camera in cameras] == [1, 1]
    assert len(bev_frames) == len(vision_frames) == len(snapshots) == 1
    assert bev_frames[0].raw_point_count == 6
    assert snapshots[0].points_world.shape == (6, 3)
    assert [image.name for image in vision_frames[0].images] == [
        "a_pillar_right"
    ]
    assert worker._poll_failures == 0
    assert worker._first_failure_at is None
    assert worker._bng is not None
    assert worker._sensors == lidars
    assert worker._hybrid_cameras == cameras
    assert stopped == []
    assert fatals == []


def test_switching_mode_mid_stream_reattaches_through_the_one_funnel() -> None:
    worker = _streaming_worker()
    worker._sensor_mode = SENSOR_MODE_LIDAR
    reattaches: list[bool] = []
    modes: list[str] = []
    worker.attach_to_player = lambda: reattaches.append(True)  # type: ignore
    worker.sensor_mode_changed.connect(modes.append)

    worker.set_sensor_mode(SENSOR_MODE_HYBRID)
    worker.set_sensor_mode(SENSOR_MODE_HYBRID)  # repeat is a no-op

    assert reattaches == [True]
    assert modes == [SENSOR_MODE_HYBRID]


def test_an_unknown_instrument_set_is_refused_rather_than_attached() -> None:
    worker = BeamNgWorker()
    modes: list[str] = []
    worker.sensor_mode_changed.connect(modes.append)

    # The removed VISION mode is exactly this case for anyone whose saved
    # setting still names it.
    worker.set_sensor_mode("VISION")

    assert worker._sensor_mode == SENSOR_MODE_LIDAR
    assert modes == []


def test_switching_to_hybrid_mid_stream_reattaches_once() -> None:
    worker = _streaming_worker()
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

    worker.set_sensor_mode(SENSOR_MODE_HYBRID)

    assert worker._sensor_mode == SENSOR_MODE_HYBRID
    assert reattaches == []


def test_cleanup_forgets_the_camera_digests_and_the_held_frames() -> None:
    """
    A re-attach must start with a clean freshness ledger AND no held frames:
    a stale digest reads the new session's first frames as re-reads, and a
    stale frame would be painted for a camera that no longer exists.
    """
    worker = _armed_hybrid_camera_worker([StreamingCameraStub()])
    worker._acquire_hybrid_camera_images(time.perf_counter())
    assert worker._hybrid_camera_digests
    assert worker._hybrid_camera_frames

    worker._cleanup_sensors()

    assert worker._hybrid_camera_digests == {}
    assert worker._hybrid_camera_frames == {}
    assert worker._hybrid_camera_failures == set()


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


# --- The A-pillar pair lands where it is asked --------------------------------


def _vivace() -> VehicleGeometry:
    """The live vivace: 2.02 m wide with the reference node 0.16 m off centre,
    and the sensor origin measured by tools/mount_origin_probe.py."""
    return VehicleGeometry(
        ground_z_vehicle=-0.243,
        left_m=1.17,
        right_m=0.85,
        front_m=1.858,
        rear_m=2.489,
        height_m=1.44,
        mounts={},
        body_floor_z=-0.243,
        sensor_origin_vehicle=(0.160, 0.362, -0.233),
    )


def test_the_pair_lands_the_same_distance_outside_each_body_side() -> None:
    """
    THE REPORTED DEFECT, in one assertion.

    Placed from the node the pair is symmetric on paper and lands 0.32 m
    asymmetric about the car: measured live, the left camera sat 0.28 m clear
    of the body with the car entirely out of frame while the right camera was
    0.04 m INSIDE the shell, filling 6.6% of its pixels with bodywork against
    the left camera's 0.65%.
    """
    geometry = _vivace()
    origin_x = geometry.sensor_origin_vehicle[0]
    rig = derive_hybrid_camera_rig(geometry)

    left_x = rig["a_pillar_left"].position_vehicle[0] + origin_x
    right_x = rig["a_pillar_right"].position_vehicle[0] + origin_x

    # Where each camera really ends up, measured from its OWN body face.
    assert left_x - geometry.left_m == pytest.approx(
        HYBRID_CAMERA_BODY_CLEARANCE_M
    )
    assert -geometry.right_m - right_x == pytest.approx(
        HYBRID_CAMERA_BODY_CLEARANCE_M
    )
    # And therefore mirror images about the body centre, not the node.
    body_centre = (geometry.left_m - geometry.right_m) / 2.0
    assert left_x - body_centre == pytest.approx(body_centre - right_x)


def test_neither_camera_ends_up_inside_the_bodywork() -> None:
    geometry = _vivace()
    origin_x = geometry.sensor_origin_vehicle[0]
    rig = derive_hybrid_camera_rig(geometry)

    assert rig["a_pillar_left"].position_vehicle[0] + origin_x > geometry.left_m
    assert (
        rig["a_pillar_right"].position_vehicle[0] + origin_x
        < -geometry.right_m
    )


def test_the_pair_sits_ahead_of_the_body_centre_at_the_a_pillar() -> None:
    """Placed from the node the 'A-pillar' station landed 0.10 m BEHIND the
    body centre -- the door-mirror plane, not the A-pillar."""
    geometry = _vivace()
    origin_y = geometry.sensor_origin_vehicle[1]
    for mount in derive_hybrid_camera_rig(geometry).values():
        landed_y = mount.position_vehicle[1] + origin_y
        body_centre_y = (geometry.rear_m - geometry.front_m) / 2.0
        assert landed_y < body_centre_y
        # ...and still well behind the nose.
        assert landed_y > -geometry.front_m


def test_an_unmeasured_origin_leaves_the_pair_exactly_as_it_was() -> None:
    plain = derive_hybrid_camera_rig(_geometry())
    explicit = derive_hybrid_camera_rig(
        VehicleGeometry(
            ground_z_vehicle=-0.3,
            left_m=0.9,
            right_m=0.9,
            front_m=2.2,
            rear_m=2.3,
            height_m=1.45,
            mounts={},
            sensor_origin_vehicle=(0.0, 0.0, 0.0),
        )
    )
    for name, mount in plain.items():
        assert mount.position_vehicle == pytest.approx(
            explicit[name].position_vehicle
        )


# --- Exposure -----------------------------------------------------------------


class _ExposureBngStub:
    def __init__(self, reply: str = "2") -> None:
        self.chunks: list[str] = []
        self._reply = reply
        self.control = SimpleNamespace(queue_lua_command=self._queue)

    def _queue(self, chunk: str, response: bool = False) -> str:
        del response
        self.chunks.append(chunk)
        return self._reply


def test_the_cameras_are_handed_to_the_engines_own_auto_exposure() -> None:
    """
    A tech Camera does NOT auto-expose: measured live it ships
    `useManualEV=true, manualEV=0.001`, a fixed exposure ~5x brighter than the
    view the player sees (mean 232-241 of 255, 36-64% of pixels clipped white).
    beamngpy has no exposure argument and BeamNG's own Lua wrappers are
    commented out, so the C++ binding is the only route.
    """
    worker = BeamNgWorker()
    bng = _ExposureBngStub()
    worker._bng = bng  # type: ignore[assignment]

    worker._apply_camera_exposure(["rig_a_pillar_left", "rig_a_pillar_right"])

    assert len(bng.chunks) == 1
    chunk = bng.chunks[0]
    assert "Research.Camera.clearManualEV(id)" in chunk
    assert '["rig_a_pillar_left"]=true' in chunk
    assert '["rig_a_pillar_right"]=true' in chunk
    # Undocumented API: a version that lacks it must leave the cameras working.
    assert "pcall" in chunk


def test_a_pinned_exposure_sends_the_linear_value_not_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sensor's manualEV is a linear MULTIPLIER, not stops -- measured,
    0.0001 reads mean 73 and anything at or above ~50 saturates the frame."""
    monkeypatch.setattr(worker_module, "HYBRID_CAMERA_AUTO_EXPOSURE", False)
    monkeypatch.setattr(worker_module, "HYBRID_CAMERA_MANUAL_EV", 0.0002)
    worker = BeamNgWorker()
    bng = _ExposureBngStub(reply="1")
    worker._bng = bng  # type: ignore[assignment]

    worker._apply_camera_exposure(["rig_a_pillar_left"])

    assert "Research.Camera.setManualEV(id, 0.0002)" in bng.chunks[0]


def test_an_exposure_failure_is_reported_and_never_stops_the_attach(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Exploding:
        control = SimpleNamespace(
            queue_lua_command=lambda chunk, response=False: (_ for _ in ()).throw(
                RuntimeError("no such API")
            )
        )

    worker = BeamNgWorker()
    worker._bng = Exploding()  # type: ignore[assignment]
    caplog.set_level("WARNING", logger=worker_module.__name__)

    worker._apply_camera_exposure(["rig_a_pillar_left"])

    assert any("Exposure check" in r.getMessage() for r in caplog.records)


def test_a_simulator_that_sets_fewer_cameras_than_asked_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = BeamNgWorker()
    worker._bng = _ExposureBngStub(reply="1")  # type: ignore[assignment]
    caplog.set_level("WARNING", logger=worker_module.__name__)

    worker._apply_camera_exposure(["left", "right"])

    assert any("Exposure check" in r.getMessage() for r in caplog.records)
