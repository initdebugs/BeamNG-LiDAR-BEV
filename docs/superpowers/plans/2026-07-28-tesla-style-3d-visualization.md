# Tesla-style 3D Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tesla-style, GPU-rendered reconstructed WORLD view using hybrid LiDAR/BeamNG data while retaining the existing RAW BEV behind a safe GUI toggle.

**Architecture:** `BeamNgWorker` remains the only BeamNG I/O owner and emits compact perception snapshots in addition to the existing `BevFrame`. A dedicated `SceneWorker` converts only the newest snapshot into bounded temporal surface/actor geometry. `WorldView` embeds Qt Quick 3D and consumes immutable `WorldFrame` data on the GUI thread; initialization errors fall back to the existing `BevWidget`.

**Tech Stack:** Python 3.11, NumPy 1.x, PyQt6/Qt 6.7.1, Qt Quick 3D/QML, pytest, ruff

## Global Constraints

- Keep BeamNG calls, sensor reads, and vehicle control on `BeamNgWorkerThread`.
- Keep normal pytest offline: no BeamNG.tech and no `QApplication`.
- Keep `BevFrame`, `BevWidget`, planner, controller, AEB behavior, and 40-ms sensor/control cadence intact.
- WORLD scene construction keeps only the newest pending snapshot and uses bounded TTL/radius state.
- WORLD initialization/build failures are recoverable visualization failures and must never stop sensors or controls.
- Use only generic primitive vehicle models; do not copy Tesla or BeamNG assets.
- This directory has no `.git` metadata, so commit steps are recorded as unavailable rather than executed.

---

### Task 1: Scene semantics and immutable models

**Files:**
- Modify: `src/beamng_lidar_bev/semantics.py`
- Modify: `src/beamng_lidar_bev/models.py`
- Modify: `src/beamng_lidar_bev/config.py`
- Modify: `tests/test_semantics.py`
- Create: `tests/test_world_scene.py`

**Interfaces:**
- Produces: `classify_scene_groups(colours, heights_vehicle, ground_z_vehicle, palette) -> np.ndarray`
- Produces: `ActorObservation`, `PerceptionSnapshot`, `WorldActor`, `WorldFrame`
- Produces constants: `SCENE_ROAD`, `SCENE_VEHICLE`, `SCENE_VULNERABLE`, `SCENE_BOUNDARY`, `SCENE_UNKNOWN`

- [ ] **Step 1: Write failing semantic-group and data-shape tests**

```python
def test_scene_groups_preserve_road_vehicle_and_unknown():
    palette = SemanticPalette.from_annotations({
        "STREET": (1, 2, 3), "CAR": (4, 5, 6), "GRASS": (7, 8, 9)
    })
    groups = classify_scene_groups(
        np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), np.uint8),
        np.zeros(4), 0.0, palette,
    )
    np.testing.assert_array_equal(
        groups, (SCENE_ROAD, SCENE_VEHICLE, SCENE_BOUNDARY, SCENE_UNKNOWN)
    )

def test_perception_snapshot_rejects_mismatched_groups():
    with pytest.raises(ValueError, match="point and semantic-group"):
        PerceptionSnapshot(
            points_world=np.zeros((2, 3), np.float32),
            semantic_groups=np.zeros(1, np.uint8),
            ego_pos_world=(0, 0, 0),
            ego_dir_world=(1, 0, 0),
            ego_up_world=(0, 0, 1),
            timestamp=0.0,
            speed_mps=0.0,
            vehicle_geometry=GEOMETRY,
        )
```

- [ ] **Step 2: Run tests and verify missing symbols fail**

Run: `py -3.11 -m pytest tests/test_semantics.py tests/test_world_scene.py -q`  
Expected: collection fails for the new imports.

- [ ] **Step 3: Add scene groups and palette code sets**

Extend `SemanticPalette` with `vehicle_codes`, `vulnerable_codes`, and
`boundary_codes`. Implement `classify_scene_groups` with semantic priority and
the same unknown/background ground fallback used by `classify_road_points`.
Recognize class-name families:

```python
VEHICLE_CLASSES = frozenset({"CAR", "TRUCK", "BUS", "MOTORCYCLE"})
VULNERABLE_CLASSES = frozenset({"PEDESTRIAN", "BICYCLE", "CYCLIST"})
```

Known non-road classes map to `SCENE_BOUNDARY`; unknown non-ground returns map
to `SCENE_UNKNOWN`.

- [ ] **Step 4: Add frozen scene dataclasses and exact shape validation**

```python
@dataclass(frozen=True)
class ActorObservation:
    actor_id: str
    kind: str
    pos_world: tuple[float, float, float]
    dir_world: tuple[float, float, float]
    velocity_world: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]  # width, height, length

@dataclass(frozen=True)
class PerceptionSnapshot:
    points_world: np.ndarray
    semantic_groups: np.ndarray
    ego_pos_world: tuple[float, float, float]
    ego_dir_world: tuple[float, float, float]
    ego_up_world: tuple[float, float, float]
    timestamp: float
    speed_mps: float
    vehicle_geometry: VehicleGeometry
    actors: tuple[ActorObservation, ...] = ()
    plan: DrivingPlan | None = None
    aeb: AebState | None = None
    rear_aeb: AebState | None = None

@dataclass(frozen=True)
class WorldActor:
    actor_id: str
    kind: str
    position: tuple[float, float, float]  # right, height, -forward
    yaw_deg: float
    scale: tuple[float, float, float]
    confidence: float

@dataclass(frozen=True)
class WorldFrame:
    road_vertices: np.ndarray
    road_indices: np.ndarray
    boundary_vertices: np.ndarray
    boundary_indices: np.ndarray
    path_vertices: np.ndarray
    path_indices: np.ndarray
    uncertain_points: np.ndarray
    actors: tuple[WorldActor, ...]
    ego_scale: tuple[float, float, float]
    speed_kph: float
    target_speed_kph: float
    autonomy_mode: str
    alert: str
    camera_position: tuple[float, float, float]
    camera_euler: tuple[float, float, float]
    timestamp: float
    perception_available: bool
```

- [ ] **Step 5: Add bounded scene constants**

Add `WORLD_*` constants for 0.5-m cells, 0.8-s fade, 1.2-s expiry, 80-m radius,
10-Hz actor state interval, 1-Hz registry interval, and 60-fps animation target.

- [ ] **Step 6: Run focused tests**

Run: `py -3.11 -m pytest tests/test_semantics.py tests/test_world_scene.py -q`  
Expected: PASS.

### Task 2: Pure scene assembly and geometry

**Files:**
- Create: `src/beamng_lidar_bev/world_scene.py`
- Modify: `tests/test_world_scene.py`

**Interfaces:**
- Consumes: `PerceptionSnapshot`, semantic group constants, planner `path_polyline`
- Produces: `world_to_render(points, snapshot) -> np.ndarray`
- Produces: `path_ribbon(points, half_width_m) -> tuple[np.ndarray, np.ndarray]`
- Produces: `WorldSceneAssembler.update(snapshot) -> WorldFrame`
- Produces: `WorldSceneAssembler.clear() -> None`

- [ ] **Step 1: Add failing transform, ribbon, TTL, and actor tests**

```python
def test_world_to_render_is_right_up_negative_forward():
    snapshot = make_snapshot(
        points=((0, 1, 0),), ego_pos=(0, 0, 0),
        ego_dir=(0, 1, 0), ego_up=(0, 0, 1),
    )
    np.testing.assert_allclose(
        world_to_render(snapshot.points_world, snapshot),
        ((0.0, 0.0, -1.0),),
    )

def test_path_ribbon_has_two_vertices_per_sample_and_triangles():
    vertices, indices = path_ribbon(np.array(((0, 0), (0, 5), (2, 10))), 1.0)
    assert vertices.shape == (6, 3)
    assert indices.shape == (12,)

def test_temporal_cells_expire_after_ttl():
    assembler = WorldSceneAssembler()
    assert assembler.update(make_road_snapshot(timestamp=0.0)).road_indices.size
    assert not assembler.update(make_empty_snapshot(timestamp=1.3)).road_indices.size

def test_ground_truth_actor_is_hidden_until_lidar_corroborates_it():
    assembler = WorldSceneAssembler()
    assert assembler.update(make_actor_snapshot(vehicle_hits=0)).actors == ()
    assert len(assembler.update(make_actor_snapshot(vehicle_hits=8, timestamp=.1)).actors) == 1
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `py -3.11 -m pytest tests/test_world_scene.py -q`  
Expected: import failure for `world_scene`.

- [ ] **Step 3: Implement coordinate transforms and ribbon triangulation**

Use an orthonormal right/forward basis matching `geometry.vehicle_axes`. Render
coordinates are `(right, gravity-relative height, -forward)`. Build the path
ribbon from segment tangents and normals, returning contiguous `float32`
vertices and `uint32` triangle indices.

- [ ] **Step 4: Implement bounded temporal road cells**

Quantize road world XY to 0.5-m cells, aggregate Z by `np.unique` plus
`np.bincount`, retain `last_seen`, expire cells past 1.2 s, and cull cells beyond
80 m before producing one flat quad per occupied cell. Reject pose jumps over
25 m by clearing state.

- [ ] **Step 5: Implement boundaries and uncertainty**

Render capped, decimated non-road/static samples as narrow camera-facing
prisms/triangles. Return at most `WORLD_MAX_BOUNDARY_MARKS` and
`WORLD_MAX_UNCERTAIN_POINTS` using deterministic stride selection.

- [ ] **Step 6: Implement actor corroboration**

Transform actor pose into ego coordinates. Count `SCENE_VEHICLE` points inside
each oriented generic footprint. Raise confidence on corroboration, coast for
0.35 s, fade to zero by 0.8 s, and omit never-confirmed actors. Compute QML yaw
from the actor forward vector relative to ego.

- [ ] **Step 7: Build `WorldFrame` including actual planner path and camera targets**

Sample `planner.path_polyline` for a plan, omit the ribbon when the plan is
`None`, derive target speed/mode, choose cruise/junction/reverse camera targets,
and produce AEB alert text without changing control behavior.

- [ ] **Step 8: Run pure tests**

Run: `py -3.11 -m pytest tests/test_world_scene.py -q`  
Expected: PASS.

### Task 3: BeamNG actor enrichment and perception snapshots

**Files:**
- Modify: `src/beamng_lidar_bev/worker.py`
- Modify: `tests/test_worker_state.py`

**Interfaces:**
- Produces signal: `perception_ready = pyqtSignal(object)`
- Produces helpers: `_refresh_actor_registry(now)`, `_poll_actor_observations(now)`
- Consumes: `classify_scene_groups`, `PerceptionSnapshot`, cached actor registry

- [ ] **Step 1: Write failing worker tests**

Add stubs for `bng.vehicles.get_current_info` and `get_states`. Pin:

```python
def test_actor_states_are_batched_and_player_is_excluded():
    observations = BeamNgWorker._poll_actor_observations(worker, now=10.0)
    assert vehicles_api.state_requests == [("traffic_1", "traffic_2")]
    assert {actor.actor_id for actor in observations} == {"traffic_1", "traffic_2"}

def test_actor_query_failure_returns_cached_empty_without_poll_failure():
    assert BeamNgWorker._poll_actor_observations(worker, now=10.0) == ()
```

Also assert the snapshot point/group lengths match after range and ego culling.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `py -3.11 -m pytest tests/test_worker_state.py -q`  
Expected: missing helpers/signal assertions fail.

- [ ] **Step 3: Add cached actor registry and batched polling**

Initialize registry/state timestamps on attach, refresh registry at 1 Hz with
`include_config=False`, exclude the current player VID, map model names to the
four generic kinds/dimensions, and call `get_states(tuple(ids))` once per
10-Hz interval. Catch/log failures locally and return the previous observations
only within the configured short stale interval.

- [ ] **Step 4: Preserve filtered world points and emit `PerceptionSnapshot`**

Apply every range/ego/finite mask to `points_world`, compute scene groups before
decimation, and emit the snapshot after `BevFrame`. Copy/contiguize arrays so
the scene thread owns stable data. Snapshot emission must remain inside the
successful poll path but outside planner/AEB failure handling.

- [ ] **Step 5: Clear actor caches on stop, fault, bridge loss, and attach**

Use one helper so every existing teardown funnel leaves no scenario actors.

- [ ] **Step 6: Run worker and full offline tests**

Run: `py -3.11 -m pytest tests/test_worker_state.py -q`  
Expected: PASS.  
Run: `py -3.11 -m pytest -q`  
Expected: PASS.

### Task 4: Dedicated latest-snapshot scene worker

**Files:**
- Create: `src/beamng_lidar_bev/scene_worker.py`
- Create: `tests/test_scene_worker.py`

**Interfaces:**
- Produces class: `SceneWorker(QObject)`
- Slots: `submit(snapshot: PerceptionSnapshot)`, `clear()`, `shutdown()`
- Signals: `world_frame_ready(object)`, `scene_error(str)`, `build_time_changed(float)`

- [ ] **Step 1: Write failing tests with a fake assembler**

Pin that `submit` processes a valid snapshot, emits build time/frame, clears
after exceptions, emits `scene_error`, and that a newer snapshot replaces an
older pending snapshot rather than appending to a queue.

- [ ] **Step 2: Run and verify failure**

Run: `py -3.11 -m pytest tests/test_scene_worker.py -q`  
Expected: import failure.

- [ ] **Step 3: Implement `SceneWorker`**

Use a zero-delay single-shot `QTimer` scheduled on its own thread. `submit`
stores `_pending = snapshot`; `_process_pending` swaps it to local, calls
`assembler.update`, emits results, and reschedules only if another snapshot
arrived during construction. No unbounded collection is created.

- [ ] **Step 4: Run tests**

Run: `py -3.11 -m pytest tests/test_scene_worker.py -q`  
Expected: PASS.

### Task 5: Qt Quick 3D bridge and QML scene

**Files:**
- Create: `src/beamng_lidar_bev/world_view.py`
- Create: `src/beamng_lidar_bev/qml/WorldScene.qml`
- Create: `src/beamng_lidar_bev/qml/qmldir`
- Create: `tests/test_world_view_buffers.py`

**Interfaces:**
- Produces: QML-compatible `qml_vectors(vertices)` and `qml_indices(indices)`
- Produces: `ActorListModel(QAbstractListModel).set_actors(actors)`
- Produces: `SceneBridge(QObject).set_frame(frame)`
- Produces: `WorldView(QWidget)` with signals `rendering_failed(str)` and method `set_frame(frame)`

- [ ] **Step 1: Write failing buffer/model tests without `QApplication`**

Test the pure QML conversion helpers:

```python
vectors = qml_vectors(np.array(((1, 2, 3),), np.float32))
indexes = qml_indices(np.array((0,), np.uint32))
assert (vectors[0].x(), vectors[0].y(), vectors[0].z()) == (1, 2, 3)
assert indexes == [0]
```

Test actor role names and stable-ID reset/update behavior using
`QAbstractListModel` only.

- [ ] **Step 2: Run and verify failure**

Run: `py -3.11 -m pytest tests/test_world_view_buffers.py -q`  
Expected: import failure.

- [ ] **Step 3: Implement QML-owned procedural meshes**

Expose vector and index lists through `SceneBridge`. Bind QML-owned
`ProceduralMesh` instances to those lists so Qt owns geometry lifetime and
buffer invalidation. Greedily merge adjacent road cells before the GUI handoff
to keep list conversion compact.

- [ ] **Step 4: Implement actor list model and scene bridge**

Expose roles for ID, kind, x/y/z, yaw, width/height/length, confidence. Expose
ego scale, HUD text, camera position/rotation, and perception state as Qt
properties with notify signals.

- [ ] **Step 5: Build QML scene**

Use `View3D`, `PerspectiveCamera`, `DirectionalLight`, `SceneEnvironment`,
models bound to bridge geometry, an ego primitive model, and `Repeater3D`
actors built from body/cabin cube primitives. Use grey materials, a blue path,
red alert material, and 180–250-ms property behaviors. Add a minimal 2D HUD.

- [ ] **Step 6: Implement `WorldView` loading/failure**

Embed with `QQuickWidget`, expose `sceneBridge` as a root-context property,
load the local QML via `QUrl.fromLocalFile`, collect `statusChanged/errors`,
and emit one recoverable `rendering_failed` message on QML error.

- [ ] **Step 7: Run tests and an offscreen load smoke**

Run: `py -3.11 -m pytest tests/test_world_view_buffers.py -q`  
Expected: PASS.

Run with `QT_QPA_PLATFORM=offscreen`:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH="$PWD\src"
py -3.11 -c "from PyQt6.QtWidgets import QApplication; from beamng_lidar_bev.world_view import WorldView; app=QApplication([]); view=WorldView(); assert view.is_ready"
```

Expected: exit code 0.

### Task 6: Main-window toggle, scene thread, and safe fallback

**Files:**
- Modify: `src/beamng_lidar_bev/main_window.py`
- Modify: `tests/test_launcher.py` or create `tests/test_view_selection.py`

**Interfaces:**
- Consumes: worker `perception_ready`, `SceneWorker.world_frame_ready`, `WorldView`
- Produces UI controls: `WORLD`, `RAW BEV`
- Produces behavior: `_select_visualization(name)`, `_on_world_rendering_failed(message)`

- [ ] **Step 1: Write view-selection state tests against a small pure helper**

Pin default `WORLD`, persisted setting values, forced `RAW BEV` when
`world_available=False`, and no worker/sensor side effects.

- [ ] **Step 2: Run and verify failure**

Run: `py -3.11 -m pytest tests/test_view_selection.py -q`  
Expected: missing helper failure.

- [ ] **Step 3: Add stacked views and segmented buttons**

Place `WorldView` and the existing `BevWidget` in a `QStackedWidget`. Add
checkable WORLD/RAW BEV buttons to the header, default WORLD, persist the
choice, and retain all existing metrics. In WORLD mode, hide the diagnostic
legend; RAW BEV restores it.

- [ ] **Step 4: Start and wire `SceneWorkerThread`**

Create the thread alongside `BeamNgWorkerThread`; connect
`worker.perception_ready -> scene_worker.submit`,
`scene_worker.world_frame_ready -> world_view.set_frame`, clear on sensor stop,
and shut down/quit/wait in `closeEvent` without making it block on BeamNG.

- [ ] **Step 5: Implement fallback**

On `rendering_failed`, log the exact message, disable WORLD buttons for the
session, select RAW BEV, and leave every sensor/control button and phase
unchanged.

- [ ] **Step 6: Run focused and full tests**

Run: `py -3.11 -m pytest tests/test_view_selection.py -q`  
Expected: PASS.  
Run: `py -3.11 -m pytest -q`  
Expected: PASS.

### Task 7: Documentation, quality gates, and validation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-07-28-tesla-style-3d-visualization-design.md` only if implementation details materially differ

**Interfaces:**
- Documents WORLD/RAW BEV behavior, hybrid honesty contract, fallback, live checks

- [ ] **Step 1: Update user and maintainer documentation**

Document the new default, view toggle, what is LiDAR-derived versus
ground-truth-enriched, actor fade semantics, QML fallback, scene thread, and
offscreen smoke command.

- [ ] **Step 2: Run placeholder and consistency scans**

Run:

```powershell
rg -n "TBD|TODO|FIXME|pass\\s*$|NotImplemented" src tests README.md CLAUDE.md docs/superpowers/specs/2026-07-28-tesla-style-3d-visualization-design.md
```

Expected: no new placeholders.

- [ ] **Step 3: Run formatting/static checks**

Run: `py -3.11 -m ruff check src tests`  
Expected: PASS.

- [ ] **Step 4: Run the complete offline suite**

Run: `py -3.11 -m pytest -q`  
Expected: PASS.

- [ ] **Step 5: Run offscreen QML smoke and render a synthetic frame**

Instantiate `WorldView`, feed a `WorldFrame` containing road mesh, one actor,
and a path, process events, and save a widget grab to a temporary PNG. Inspect
the PNG for camera, ego, road, actor, path, and HUD visibility.

- [ ] **Step 6: Record live-only checks**

Report that BeamNG actor polling, LiDAR corroboration, frame pacing,
map-load/teleport clearing, reverse camera, and AEB visuals require a live
simulator session and provide the exact checklist from the design spec.
