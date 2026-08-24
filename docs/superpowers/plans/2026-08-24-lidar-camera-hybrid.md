# LiDAR-First Two-Camera Hybrid Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable HYBRID mode that runs the existing six-LiDAR perception stack unchanged while streaming two literal A-pillar RGB cameras into a live two-up CAMERAS view.

**Architecture:** Keep the worker's existing `_sensors` collection as the authoritative perception set and add a separately owned two-camera auxiliary collection used only in HYBRID. Camera colour buffers travel through the existing `VisionFrame` contract; no camera depth or annotation is rendered or admitted to BEV, WORLD, parking, planning, or AEB.

**Tech Stack:** Python 3.11, NumPy, PyQt6 signals/widgets, BeamNGpy 1.36 shared-memory Camera/Lidar sensors, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-24-lidar-camera-hybrid-design.md`

## Global Constraints

- `HYBRID` contains all six existing LiDAR mounts and exactly two new A-pillar cameras.
- LiDAR is the only HYBRID source for WORLD, RAW BEV, parking occupancy, planning, automatic parking, forward AEB, and reverse AEB.
- Hybrid cameras render colour only at 1280x960 with a positive 0.10-second requested update period.
- Camera depth, annotation, and instance channels remain disabled in HYBRID.
- Runtime camera read failures never enter the LiDAR poll-failure budget.
- Existing `LIDAR` and eight-camera `VISION` behaviour stays unchanged.
- The offline test suite never launches BeamNG.tech or creates a `QApplication`.
- Production changes follow red-green-refactor: every new behaviour is first observed failing for the intended reason.

## File Structure

- `src/beamng_lidar_bev/config.py`: hybrid-camera optics, placement fractions, resolution, and rate.
- `src/beamng_lidar_bev/geometry.py`: stable hybrid camera names and pure vehicle-bbox-to-mount derivation.
- `src/beamng_lidar_bev/worker.py`: third mode, auxiliary Camera ownership, attach/cleanup, RGB acquisition, and hybrid poll integration.
- `src/beamng_lidar_bev/main_window.py`: persisted mode selection, third selector button, camera-view availability, status text.
- `src/beamng_lidar_bev/vision_view.py`: deterministic two-image, one-row layout.
- `tests/test_vision_mode.py`: geometry, Camera kwargs, auxiliary acquisition, isolation, cleanup, and poll integration.
- `tests/test_view_selection.py`: pure third-mode parsing, camera availability, controls, and view selection.
- `README.md`: user-facing description of the new mode and how to inspect the camera pair.

---

### Task 1: Derive the literal A-pillar camera pair

**Files:**

- Modify: `src/beamng_lidar_bev/config.py:278-319`
- Modify: `src/beamng_lidar_bev/geometry.py:249-430`
- Test: `tests/test_vision_mode.py:1-142`

**Interfaces:**

- Produces: `HYBRID_CAMERA_NAMES: tuple[str, str]`
- Produces: `derive_hybrid_camera_rig(geometry: VehicleGeometry) -> dict[str, CameraMount]`
- Consumes later: worker attachment and MainWindow readiness logging.

- [ ] **Step 1: Write failing geometry tests**

Add imports for `HYBRID_CAMERA_NAMES` and `derive_hybrid_camera_rig`, then add:

```python
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
```

- [ ] **Step 2: Run the geometry tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k "hybrid_rig or hybrid_cameras or hybrid_pair" -v
```

Expected: collection fails because `HYBRID_CAMERA_NAMES` and
`derive_hybrid_camera_rig` do not exist.

- [ ] **Step 3: Add the hybrid camera constants**

Add beside the existing camera configuration:

```python
HYBRID_CAMERA_RESOLUTION = (1280, 960)
HYBRID_CAMERA_UPDATE_TIME_S = 0.10
HYBRID_CAMERA_HFOV_DEG = 105.0
HYBRID_CAMERA_YAW_DEG = 37.0
HYBRID_CAMERA_PITCH_DEG = 7.0
HYBRID_CAMERA_HEIGHT_FRACTION = 0.88
HYBRID_CAMERA_FRONT_FRACTION = 0.25
HYBRID_CAMERA_BODY_CLEARANCE_M = 0.12
```

- [ ] **Step 4: Implement the pure hybrid rig derivation**

Import the constants into `geometry.py` and add:

```python
HYBRID_CAMERA_NAMES = ("a_pillar_left", "a_pillar_right")


def derive_hybrid_camera_rig(
    geometry: VehicleGeometry,
    resolution: tuple[int, int] = HYBRID_CAMERA_RESOLUTION,
) -> dict[str, CameraMount]:
    yaw = math.radians(HYBRID_CAMERA_YAW_DEG)
    pitch = math.radians(HYBRID_CAMERA_PITCH_DEG)
    horizontal = math.cos(pitch)
    y = -HYBRID_CAMERA_FRONT_FRACTION * geometry.front_m
    z = HYBRID_CAMERA_HEIGHT_FRACTION * geometry.height_m
    left = CameraMount(
        name="a_pillar_left",
        position_vehicle=(
            geometry.left_m + HYBRID_CAMERA_BODY_CLEARANCE_M,
            y,
            z,
        ),
        direction_vehicle=(
            math.sin(yaw) * horizontal,
            -math.cos(yaw) * horizontal,
            -math.sin(pitch),
        ),
        horizontal_fov_deg=HYBRID_CAMERA_HFOV_DEG,
        vertical_fov_deg=camera_vertical_fov_deg(
            HYBRID_CAMERA_HFOV_DEG, resolution
        ),
        resolution=resolution,
    )
    right = CameraMount(
        name="a_pillar_right",
        position_vehicle=(
            -(geometry.right_m + HYBRID_CAMERA_BODY_CLEARANCE_M),
            y,
            z,
        ),
        direction_vehicle=(
            -math.sin(yaw) * horizontal,
            -math.cos(yaw) * horizontal,
            -math.sin(pitch),
        ),
        horizontal_fov_deg=HYBRID_CAMERA_HFOV_DEG,
        vertical_fov_deg=camera_vertical_fov_deg(
            HYBRID_CAMERA_HFOV_DEG, resolution
        ),
        resolution=resolution,
    )
    return {left.name: left, right.name: right}
```

- [ ] **Step 5: Run geometry tests and the existing rig tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k "rig or camera or pillar or coverage" -v
```

Expected: all selected tests pass; the eight-camera rig remains unchanged.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/beamng_lidar_bev/config.py src/beamng_lidar_bev/geometry.py tests/test_vision_mode.py
git commit -m "feat: define two-camera A-pillar rig"
```

---

### Task 2: Add third-mode semantics without changing sensor acquisition

**Files:**

- Modify: `src/beamng_lidar_bev/worker.py:184-190,791-811`
- Modify: `src/beamng_lidar_bev/main_window.py:47-138`
- Test: `tests/test_view_selection.py:1-103`
- Test: `tests/test_vision_mode.py:541-565`

**Interfaces:**

- Produces: `SENSOR_MODE_HYBRID = "HYBRID"`
- Produces: `sensor_mode_has_cameras(mode: str) -> bool`
- Changes: `resolve_sensor_mode()` accepts all three exact names and defaults unknown values to LiDAR.
- Changes: `BeamNgWorker.set_sensor_mode()` accepts HYBRID and retains repeat/no-op/live-reattach semantics.

- [ ] **Step 1: Write failing pure mode tests**

Extend imports and add:

```python
def test_hybrid_is_a_persisted_instrument_set() -> None:
    assert resolve_sensor_mode("HYBRID") == SENSOR_MODE_HYBRID
    assert resolve_sensor_mode("hybrid") == SENSOR_MODE_HYBRID


def test_hybrid_has_camera_view_and_lidar_controls(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main_window, "VISION_DRIVING_ENABLED", False)
    assert sensor_mode_has_cameras(SENSOR_MODE_HYBRID) is True
    assert sensor_mode_has_cameras(SENSOR_MODE_VISION) is True
    assert sensor_mode_has_cameras(SENSOR_MODE_LIDAR) is False
    assert controls_offered(SENSOR_MODE_HYBRID) is True
    assert controls_offered(SENSOR_MODE_VISION) is False
```

Add a worker test:

```python
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
```

- [ ] **Step 2: Run mode tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_view_selection.py tests/test_vision_mode.py -k hybrid -v
```

Expected: import/collection failure because HYBRID symbols do not exist.

- [ ] **Step 3: Implement mode constants and pure UI helpers**

In `worker.py` add the constant and include it in validation:

```python
SENSOR_MODE_LIDAR = "LIDAR"
SENSOR_MODE_HYBRID = "HYBRID"
SENSOR_MODE_VISION = "VISION"

if mode not in (SENSOR_MODE_LIDAR, SENSOR_MODE_HYBRID, SENSOR_MODE_VISION):
    ...
```

In `main_window.py` add:

```python
def resolve_sensor_mode(requested: str | None) -> str:
    mode = str(requested or "").upper()
    return (
        mode
        if mode in (SENSOR_MODE_LIDAR, SENSOR_MODE_HYBRID, SENSOR_MODE_VISION)
        else SENSOR_MODE_LIDAR
    )


def sensor_mode_has_cameras(mode: str) -> bool:
    return mode in (SENSOR_MODE_HYBRID, SENSOR_MODE_VISION)
```

Leave `controls_offered()` Vision-specific so HYBRID remains enabled when the
Vision gate is closed.

- [ ] **Step 4: Run mode tests and all view-selection tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_view_selection.py tests/test_vision_mode.py -k "mode or hybrid or controls" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/beamng_lidar_bev/worker.py src/beamng_lidar_bev/main_window.py tests/test_view_selection.py tests/test_vision_mode.py
git commit -m "feat: add hybrid sensor mode semantics"
```

---

### Task 3: Own and attach the auxiliary RGB cameras

**Files:**

- Modify: `src/beamng_lidar_bev/worker.py:10-23,331-358,603-775,2158-2241,3800-3828`
- Test: `tests/test_vision_mode.py:172-278,566-582`

**Interfaces:**

- Produces: `BeamNgWorker.hybrid_camera_sensor_kwargs(mount: CameraMount) -> dict[str, Any]`
- Produces: `BeamNgWorker._attach_hybrid_camera_rig(vehicle, geometry, sensor_prefix) -> int`
- State: `_hybrid_cameras`, `_hybrid_camera_names`, `_hybrid_camera_digests`, `_hybrid_camera_failures`.
- Cleanup invariant: every hybrid Camera is removed even when a later attach step or another removal fails.

- [ ] **Step 1: Write failing Camera-kwargs and cleanup tests**

Add:

```python
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


class RemovableCameraStub(StreamingCameraStub):
    def __init__(self) -> None:
        super().__init__()
        self.remove_calls = 0

    def remove(self) -> None:
        self.remove_calls += 1


def test_cleanup_removes_and_forgets_every_hybrid_camera() -> None:
    worker = BeamNgWorker()
    cameras = [RemovableCameraStub(), RemovableCameraStub()]
    worker._hybrid_cameras = cameras  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES)
    worker._hybrid_camera_digests = {"a_pillar_left": b"old"}
    worker._hybrid_camera_failures = {"a_pillar_right"}

    worker._cleanup_sensors()

    assert [camera.remove_calls for camera in cameras] == [1, 1]
    assert worker._hybrid_cameras == []
    assert worker._hybrid_camera_names == []
    assert worker._hybrid_camera_digests == {}
    assert worker._hybrid_camera_failures == set()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k "hybrid_camera_constructor or cleanup_removes" -v
```

Expected: failures because hybrid Camera state and kwargs do not exist.

- [ ] **Step 3: Add auxiliary Camera state and pure kwargs**

Under `TYPE_CHECKING`, import `Camera` with `Lidar`. Initialise:

```python
self._hybrid_cameras: list[Camera] = []
self._hybrid_camera_names: list[str] = []
self._hybrid_camera_digests: dict[str, bytes] = {}
self._hybrid_camera_failures: set[str] = set()
```

Add a pure constructor helper mirroring `camera_sensor_kwargs` but using
`HYBRID_CAMERA_UPDATE_TIME_S` and disabling all non-colour channels.

- [ ] **Step 4: Add the dedicated attach method and attach branch**

Implement:

```python
def _attach_hybrid_camera_rig(
    self, vehicle: Vehicle, geometry: VehicleGeometry, sensor_prefix: str
) -> int:
    from beamngpy.sensors import Camera

    rig = derive_hybrid_camera_rig(geometry)
    for index, mount in enumerate(rig.values()):
        self.status_changed.emit(
            "ATTACHING",
            f"Attaching {mount.name} camera ({index + 1}/{len(rig)})",
        )
        camera = Camera(
            f"{sensor_prefix}_{mount.name}",
            self._bng,
            vehicle,
            **self.hybrid_camera_sensor_kwargs(mount),
        )
        self._hybrid_camera_names.append(mount.name)
        self._hybrid_cameras.append(camera)
    self._check_capture_settings()
    return len(rig)
```

In `attach_to_player()`, keep Vision's branch unchanged. For HYBRID, attach all
LiDAR mounts first and then call `_attach_hybrid_camera_rig`; initialise camera
liveness state and report `6 LiDAR sensors + 2 cameras active`.

- [ ] **Step 5: Extend the one teardown funnel**

Before removing `_sensors`, reverse-remove every `_hybrid_cameras` item with the
same per-sensor exception protection. Clear all hybrid collections and liveness
state. Do not change `_sensor_set_is_complete()`: HYBRID always has the primary
LiDAR list and attach atomicity prevents a partial set from persisting.

- [ ] **Step 6: Run focused and regression tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py tests/test_worker_state.py -k "constructor or cleanup or teardown or sensor" -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/beamng_lidar_bev/worker.py tests/test_vision_mode.py
git commit -m "feat: attach auxiliary hybrid cameras"
```

---

### Task 4: Stream RGB frames with isolated failures

**Files:**

- Modify: `src/beamng_lidar_bev/worker.py:1936-2067,2300-2341`
- Test: `tests/test_vision_mode.py:172-458`

**Interfaces:**

- Produces: `_acquire_hybrid_camera_images(now: float) -> tuple[list[CameraImage], bool]`
- Contract: returned arrays are private `(H, W, 4)` `uint8` copies in configured left/right order.
- Error contract: one Camera exception suppresses only that image and is logged once until recovery.

- [ ] **Step 1: Write failing RGB acquisition tests**

Add a helper:

```python
def _armed_hybrid_camera_worker(
    cameras: list[StreamingCameraStub],
) -> BeamNgWorker:
    worker = BeamNgWorker()
    worker._sensor_mode = SENSOR_MODE_HYBRID
    worker._hybrid_cameras = cameras  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES[: len(cameras)])
    worker._vision_streaming_since = time.perf_counter()
    return worker
```

Add tests:

```python
def test_hybrid_rgb_acquisition_preserves_order_and_owns_the_bytes() -> None:
    cameras = [StreamingCameraStub(), StreamingCameraStub()]
    worker = _armed_hybrid_camera_worker(cameras)
    images, fresh = worker._acquire_hybrid_camera_images(time.perf_counter())
    assert fresh is True
    assert [image.name for image in images] == list(HYBRID_CAMERA_NAMES)
    assert [camera.stream_raw_calls for camera in cameras] == [1, 1]
    assert all(image.rgba.shape == (3, 4, 4) for image in images)
    assert all(
        image.rgba.flags["OWNDATA"] or image.rgba.base.flags["OWNDATA"]
        for image in images
    )


def test_unchanged_hybrid_colour_is_not_counted_as_a_new_frame() -> None:
    camera = StreamingCameraStub()
    worker = _armed_hybrid_camera_worker([camera])
    _, first = worker._acquire_hybrid_camera_images(time.perf_counter())
    _, second = worker._acquire_hybrid_camera_images(time.perf_counter())
    camera.repaint()
    _, third = worker._acquire_hybrid_camera_images(time.perf_counter())
    assert (first, second, third) == (True, False, True)


class ExplodingCameraStub(StreamingCameraStub):
    def stream_raw(self) -> dict[str, bytes]:
        raise RuntimeError("camera gone")


def test_one_hybrid_camera_failure_does_not_hide_the_other() -> None:
    worker = _armed_hybrid_camera_worker(
        [ExplodingCameraStub(), StreamingCameraStub()]
    )
    images, fresh = worker._acquire_hybrid_camera_images(time.perf_counter())
    assert fresh is True
    assert [image.name for image in images] == ["a_pillar_right"]
    assert worker._hybrid_camera_failures == {"a_pillar_left"}
```

- [ ] **Step 2: Run the acquisition tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k "hybrid_rgb or hybrid_colour or hybrid_camera_failure" -v
```

Expected: failures because `_acquire_hybrid_camera_images` does not exist.

- [ ] **Step 3: Implement colour-only acquisition**

For each name/camera pair:

```python
try:
    raw = camera.stream_raw()
except Exception as exc:
    if name not in self._hybrid_camera_failures:
        LOGGER.warning("Hybrid camera %s failed: %s", name, exc)
    self._hybrid_camera_failures.add(name)
    continue

self._hybrid_camera_failures.discard(name)
width, height = camera.resolution
colour = raw.get("colour")
if colour is None or len(colour) != width * height * 4:
    continue
pixels = np.frombuffer(colour, dtype=np.uint8).copy()
digest = bytes(pixels[::_VISION_DIGEST_STRIDE])
if digest != self._hybrid_camera_digests.get(name):
    self._hybrid_camera_digests[name] = digest
    any_fresh = True
images.append(
    CameraImage(name=name, rgba=pixels.reshape((height, width, 4)))
)
```

Call the existing liveness watcher with a new optional channel label so Vision
continues to log `colour + depth + annotation` and HYBRID logs `colour`.

- [ ] **Step 4: Run acquisition and existing Vision-frame tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k "hybrid or image or freshness or rereads" -v
```

Expected: all selected tests pass; existing depth-based Vision freshness stays
unchanged.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/beamng_lidar_bev/worker.py tests/test_vision_mode.py
git commit -m "feat: stream isolated hybrid camera frames"
```

---

### Task 5: Integrate the camera feed while proving LiDAR-only perception

**Files:**

- Modify: `src/beamng_lidar_bev/worker.py:1372-1887`
- Test: `tests/test_vision_mode.py:278-377`

**Interfaces:**

- Consumes: `_acquire_lidar_cloud()` and `_acquire_hybrid_camera_images()`.
- Produces: one ordinary LiDAR-backed `BevFrame`, one LiDAR-backed `PerceptionSnapshot`, and one two-image `VisionFrame` per HYBRID tick.
- Safety invariant: Camera depth/annotation bytes cannot affect point counts, semantic groups, planning, parking, or AEB.

- [ ] **Step 1: Write a failing end-to-end hybrid tick test**

Define a minimal streaming LiDAR stub in `test_vision_mode.py` and add:

```python
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


def test_hybrid_tick_emits_camera_feed_but_only_lidar_perception() -> None:
    lidars = [HybridLidarStub(float(index)) for index in range(6)]
    cameras = [
        StreamingCameraStub(depth_m=1.0),
        StreamingCameraStub(depth_m=250.0),
    ]
    worker = BeamNgWorker()
    worker._sensor_mode = SENSOR_MODE_HYBRID
    worker._bng = object()  # type: ignore[assignment]
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._sensors = lidars  # type: ignore[assignment]
    worker._sensor_names = [f"lidar_{index}" for index in range(6)]
    worker._hybrid_cameras = cameras  # type: ignore[assignment]
    worker._hybrid_camera_names = list(HYBRID_CAMERA_NAMES)
    worker._geometry = _geometry()
    worker._palette = SemanticPalette.from_annotations({"STREET": (255, 0, 0)})
    bev_frames: list[BevFrame] = []
    vision_frames: list[VisionFrame] = []
    snapshots: list[object] = []
    worker.frame_ready.connect(bev_frames.append)
    worker.vision_frame_ready.connect(vision_frames.append)
    worker.perception_ready.connect(snapshots.append)

    worker._poll_once()

    assert [lidar.stream_calls for lidar in lidars] == [1] * 6
    assert [camera.stream_raw_calls for camera in cameras] == [1, 1]
    assert bev_frames[0].raw_point_count == 6
    assert snapshots[0].points_world.shape == (6, 3)
    assert len(vision_frames[0].images) == 2
```

Add a second test with one `ExplodingCameraStub` and assert a LiDAR frame and
snapshot still emit, `_poll_failures == 0`, and the surviving image remains.

- [ ] **Step 2: Run hybrid tick tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py -k hybrid_tick -v
```

Expected: no `VisionFrame` is emitted because `_poll_once()` knows only Vision
or LiDAR acquisition.

- [ ] **Step 3: Add the HYBRID branch to `_poll_once()`**

Introduce `hybrid = self._sensor_mode == SENSOR_MODE_HYBRID`. Keep Vision's
existing branch intact; in HYBRID:

```python
point_chunks, colour_chunks = self._acquire_lidar_cloud(state)
vision_images, _camera_fresh = self._acquire_hybrid_camera_images(started)
fresh = bool(point_chunks)
```

Do not append camera chunks. Change only the frame-emission condition:

```python
if vision or hybrid:
    self.vision_frame_ready.emit(VisionFrame(...))
```

Leave `_frame_times` driven by `fresh`, which is LiDAR freshness in HYBRID.
Leave `_vision_refuses_driving()` and `_porosity_sensor_height()` Vision-only.

- [ ] **Step 4: Run hybrid, Vision, LiDAR, AEB, and parking regression tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_vision_mode.py tests/test_worker_state.py tests/test_aeb.py tests/test_parking.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/beamng_lidar_bev/worker.py tests/test_vision_mode.py
git commit -m "feat: publish hybrid feed beside lidar perception"
```

---

### Task 6: Expose HYBRID and the deterministic two-up feed in the UI

**Files:**

- Modify: `src/beamng_lidar_bev/main_window.py:47-138,573-645,907-1065`
- Modify: `src/beamng_lidar_bev/vision_view.py:53-87`
- Test: `tests/test_view_selection.py:1-103`
- Test: `tests/test_vision_mode.py:142-171`

**Interfaces:**

- UI selector: `LIDAR | HYBRID | VISION`.
- Camera availability: `sensor_mode_has_cameras(_active_sensor_mode)`.
- Layout contract: `grid_dimensions(2, ...) == (1, 2)` for all positive pane dimensions.

- [ ] **Step 1: Write failing layout and selection tests**

Add:

```python
@pytest.mark.parametrize("size", ((1600.0, 700.0), (700.0, 1600.0)))
def test_two_camera_angle_check_is_always_left_right(
    size: tuple[float, float],
) -> None:
    assert grid_dimensions(2, *size) == (1, 2)
```

Extend view-selection tests so `resolve_visualization(VIEW_CAMERAS, ..., cameras_available=sensor_mode_has_cameras(SENSOR_MODE_HYBRID))` returns `VIEW_CAMERAS`, while LiDAR still falls back.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_view_selection.py tests/test_vision_mode.py -k "two_camera or hybrid" -v
```

Expected: the tall-pane layout returns `(2, 1)` and UI symbols/buttons are
missing.

- [ ] **Step 3: Pin the two-up layout**

At the top of `grid_dimensions()` after the empty check:

```python
if count == 2:
    return (1, 2)
```

The existing focus state and labels need no change.

- [ ] **Step 4: Add and wire the third selector button**

Import `SENSOR_MODE_HYBRID` and `HYBRID_CAMERA_NAMES`. Construct
`self.hybrid_mode_button = QPushButton("HYBRID")`, include it in the exclusive
group, connect it to `_select_sensor_mode(SENSOR_MODE_HYBRID)`, and give it a
tooltip that says LiDAR remains authoritative.

Update:

- `_sync_sensor_buttons()` for all three buttons;
- `_on_sensor_mode_changed()` status text;
- CAMERAS enable/fallback checks to use `sensor_mode_has_cameras()`;
- `_select_visualization()` camera availability to use the helper;
- `_on_sensors_ready()` to log all six LiDAR mounts plus the two camera names
  in HYBRID;
- attach-button and CAMERAS tooltips so they no longer describe only one set.

- [ ] **Step 5: Run UI-pure and full Vision tests**

Run:

```powershell
.venv39\Scripts\python -m pytest tests/test_view_selection.py tests/test_vision_mode.py -v
```

Expected: all tests pass without a `QApplication`.

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/beamng_lidar_bev/main_window.py src/beamng_lidar_bev/vision_view.py tests/test_view_selection.py tests/test_vision_mode.py
git commit -m "feat: expose hybrid two-camera live view"
```

---

### Task 7: Documentation, full verification, and live angle check

**Files:**

- Modify: `README.md:1-90`
- Verify: all `src/beamng_lidar_bev/*.py`, `tests/*.py`
- Runtime evidence: `logs/beamng_lidar_bev.log`

**Interfaces:**

- User workflow: select HYBRID, attach, then select CAMERAS.
- Acceptance: two correctly labelled feeds, unchanged LiDAR perception, no cabin-dominated view, useful centre overlap, stable 10 Hz request.

- [ ] **Step 1: Document the user-visible mode**

Update README's mode description and run instructions with:

```text
HYBRID keeps the six-LiDAR perception and safety stack active and adds two
colour-only A-pillar cameras. Select CAMERAS to compare their live aiming;
WORLD and RAW BEV remain LiDAR-backed.
```

State that cameras are requested at 10 Hz and require BeamNG visible with a
graphics preset above Lowest.

- [ ] **Step 2: Run formatting and static checks**

Run:

```powershell
.venv39\Scripts\python -m ruff check src tests
```

Expected: exit 0 with no diagnostics.

- [ ] **Step 3: Run the complete offline suite**

Run:

```powershell
.venv39\Scripts\python -m pytest
```

Expected: exit 0 with zero failures and zero unexpected xpasses.

- [ ] **Step 4: Review the complete diff against the approved spec**

Run:

```powershell
git diff 6ed3782 --check
git diff 6ed3782 --stat
git status --short
```

Verify every Global Constraint against code/tests and confirm no unrelated
files changed.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md
git commit -m "docs: explain lidar-first hybrid mode"
```

- [ ] **Step 6: Launch the app for the user-visible angle check**

Run `run_app.bat` visibly. In the application select HYBRID, attach to the
player vehicle, then select CAMERAS. Keep BeamNG.tech visible and graphics above
Lowest.

Check the two labelled images for:

- left/right identity;
- no dominant cabin/body obstruction;
- level horizon with approximately 7-degree downward pitch;
- continuous straight-ahead overlap;
- useful front-quarter ground and parking-line visibility;
- no missing/black buffers.

- [ ] **Step 7: Record live runtime evidence**

Inspect the new attach, liveness, reach, polling, and scene lines in
`logs/beamng_lidar_bev.log`. Record stationary and moving behaviour. If camera
angles need correction, change only the geometry constants through a new
failing geometry test, rerun the full suite, and repeat the live check.

- [ ] **Step 8: Request final code review**

Dispatch a reviewer with base `6ed3782`, current `HEAD`, the approved spec, and
this plan. Fix every Critical or Important finding, rerun Ruff and the full
pytest suite, then report the verified result and any live-only limitation.
