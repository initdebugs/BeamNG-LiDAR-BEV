# Tesla-style 3D autonomy visualization

Date: 2026-07-28  
Status: approved  
Decisions: personal-use Tesla FSD aesthetic, reconstructed 3D world, hybrid
LiDAR/BeamNG data, Qt Quick 3D renderer, existing raw BEV retained behind a
toggle

## Goal

Replace the default raw point-cloud presentation with a calm, readable
Tesla-style reconstruction of the driving world. The view should communicate,
at a glance:

- where the drivable surface and its boundaries are;
- which vehicles the sensor system currently corroborates;
- where the ego vehicle plans to travel;
- current and planned speed;
- whether self-driving or AEB is active.

The existing top-down point-cloud display remains available as `RAW BEV`. It is
the diagnostic truth view for raw semantic returns, candidate paths, range,
sensor mounts, and AEB geometry.

## Current limitations

The existing `BevFrame` reduces every scan to two `Nx2` arrays: road and
obstacle. This discards height, semantic class, sensor provenance, and temporal
continuity before the GUI receives the frame. `BevWidget` then draws a fixed
210-m-wide square with constant-size grey and red pixels. The representation is
fast and technically honest, but:

- the ego vehicle and near-field hazards are too small;
- all non-road objects look equally dangerous and anonymous;
- sparse distant returns are difficult to interpret;
- the planner fan, AEB corridor, grid, rings, points, and labels compete;
- the display has no scene memory, depth, recognizable actors, or contextual
  camera.

## Product behavior

### View selection

The main visualization header gains a two-way segmented control:

- `WORLD` — the new default Qt Quick 3D scene;
- `RAW BEV` — the current `BevWidget`, unchanged in behavior.

Switching views is a GUI-only operation. It does not restart sensors, reset
planning, alter self-driving/AEB state, or change BeamNG polling. If Qt Quick 3D
cannot initialize or its QML reports an error, the app selects `RAW BEV`, logs a
recoverable event, and continues streaming.

### WORLD visual hierarchy

Always visible:

- a neutral reconstructed road/free-space surface;
- curb or non-road boundary geometry;
- a generic low-poly ego vehicle;
- LiDAR-corroborated traffic vehicles;
- a broad blue ribbon sampled from the planner's actual composite path;
- ego speed, target speed, self-driving state, and an exceptional AEB state.

Hidden from the default view:

- raw point speckles;
- the full planner candidate fan;
- range rings and sensor mounts;
- acquisition, display, point-count, and polling diagnostics;
- detailed AEB corridor geometry while it is merely armed.

Those remain visible in `RAW BEV` and in the existing metric band.

Colour has one job per role:

- neutral greys: environment and ordinary actors;
- blue: planned motion and autonomy state;
- amber: uncertain or degraded information;
- red: an actively controlled collision threat or AEB braking.

No decorative neon grid, radar sweep, or unrelated sci-fi ornament is added.

### Camera

The camera is ego-relative and interpolates independently of the sensor rate:

- **Cruise:** a low trailing perspective with the ego in the lower third.
  Look-ahead increases smoothly with speed.
- **Junction/corner:** path curvature raises and widens the camera enough to
  show crossing context and the intended turn.
- **Reverse:** the camera eases 180 degrees to look in the direction of travel.
- **AEB:** orientation remains stable. The threat, path corridor, and stopping
  state change colour; there is no dramatic camera jump.

Camera targets are deterministic values carried in the scene frame. QML
`Behavior` animations interpolate between them at display refresh rate.

## Architecture

### Threading

The existing `BeamNgWorker` remains the sole owner of BeamNG and sensor calls.
Scene construction must not consume its already tight 40-ms control/poll
budget.

A new `SceneWorker` lives on its own `QThread`:

1. `BeamNgWorker` emits the newest perception snapshot after its existing
   planning work.
2. `SceneWorker` retains only the latest pending snapshot. If construction is
   still busy, superseded snapshots are dropped rather than queued.
3. It updates the bounded temporal surface model, corroborates actors, builds
   render geometry, and emits an immutable `WorldFrame`.
4. The GUI updates geometry/model data on the GUI thread.

The existing `BevFrame`/`frame_ready` path remains intact and continues driving
`BevWidget` and the metrics.

### Data sources

#### LiDAR-owned perception

The semantic LiDAR snapshot carries:

- `points_world`: `(N, 3)` in BeamNG world coordinates;
- a compact semantic group per point: road, vehicle, pedestrian/cyclist,
  boundary/static, or unknown;
- the current ego position, forward direction, and up direction needed for the
  world-to-render transform;
- a monotonic timestamp.

LiDAR owns visibility of road, boundaries, and dynamic actors. A perfect
simulator scene is never painted simply because it exists.

#### BeamNG actor enrichment

BeamNGpy's installed API supports:

- a slow registry query through `vehicles.get_current_info(include_config=False)`;
- a batched pose query through `vehicles.get_states(vehicle_ids)`.

The registry refreshes at approximately 1 Hz. Actor poses update in one batched
request at 10 Hz so actor enrichment cannot multiply blocking round trips by the
number of traffic vehicles. The player vehicle is excluded.

Actor type is mapped to a small generic visual vocabulary rather than loading
BeamNG meshes:

- car;
- SUV/pickup/van;
- truck/bus;
- unknown vehicle.

### Hybrid honesty contract

Ground truth supplies actor identity, model/type, pose, and velocity. Generic
dimensions come from the mapped visual type; the renderer does not open a
per-vehicle connection merely to fetch exact bounding boxes. LiDAR evidence
supplies visual confidence:

- semantic vehicle returns inside an actor's oriented footprint raise
  confidence;
- a confirmed actor is rendered as a solid generic model;
- short missed intervals coast using its last velocity and fade;
- after the bounded persistence interval it disappears;
- an actor that has never been corroborated is not rendered.

This produces stable recognizable actors without masking sensor occlusion or
failure.

## Components

### Pure data models

`models.py` gains:

- `PerceptionSnapshot` — compact sensor/pose/actor input for `SceneWorker`;
- `ActorObservation` — BeamNG identity, type, pose, velocity, and mapped generic
  dimensions;
- `WorldActor` — ego-relative render pose and confidence;
- `WorldFrame` — surface mesh, boundary ribbons, uncertain points, actors,
  planned-path ribbon, HUD values, camera targets, and timestamp.

Arrays are NumPy arrays with explicit shapes and `float32`/`uint32` render
dtypes. Models remain Qt-free.

### `world_scene.py`

This Qt-free module contains:

- ego/world coordinate transforms;
- semantic grouping;
- the bounded temporal voxel surface;
- road-cell meshing;
- boundary extraction;
- actor footprint corroboration and confidence decay;
- path-ribbon triangulation;
- camera-target formulas;
- the latest-snapshot scene assembler.

#### Temporal surface

Road points enter a world-space voxel grid with an initial 0.5-m horizontal
cell size. Each cell stores median/filtered height, semantic group, confidence,
and last-seen time. Cells:

- are limited to the display horizon around the current ego pose;
- remain stable through short missed scans and expire after a 1.2-second TTL;
- are meshed only when adjacent road cells have compatible heights;
- never bridge a large height discontinuity.

The first implementation greedily merges adjacent grid cells into flat
rectangles instead of using an expensive general triangulator. It preserves the
occupied grid footprint while greatly reducing the QML transfer and draw size.
It is deterministic, bounded, dependency-free, and sufficient for the
low-detail FSD aesthetic.

Boundary/static returns render as subdued vertical marks or short ribbons.
Unclassified points render only as sparse, low-opacity fragments so uncertainty
is visible without reverting to a full point-cloud view.

### `scene_worker.py`

`SceneWorker` owns `WorldSceneAssembler`. It exposes slots for new perception
snapshots and clearing state. It emits `world_frame_ready`, `scene_error`, and
lightweight build-time telemetry.

Construction errors clear only temporal visualization state and emit
`scene_error`; they never stop sensors, disengage controls, or enter the
BeamNG worker's polling-failure budget.

### `world_view.py`

`WorldView` embeds a `QQuickWidget`. A `SceneBridge` exposes:

- QML-compatible vector/index lists for road, boundary, and path meshes;
- a `QAbstractListModel` actor pool;
- ego/camera/HUD properties;
- rendering status and QML errors.

QML-owned `ProceduralMesh` objects build the native geometry buffers. This is
the supported Qt 6.6+ path for application-generated QML geometry and avoids
the ownership and invalidation ambiguity of Python-owned
`QQuick3DGeometry` objects in PyQt6. Actor delegates use stable IDs and
interpolate updates rather than teleporting between sensor ticks.

### QML scene

`qml/WorldScene.qml` contains:

- one perspective camera and environment light;
- road, boundary, uncertain, and path models;
- an ego model;
- a `Repeater3D` of low-poly actor delegates;
- the compact 2D HUD overlay;
- smooth property behaviors for camera and actor transforms.

Low-poly vehicles are composed from Qt Quick 3D primitives in the first
implementation, avoiding new binary assets and licensing concerns.

## Planner and AEB presentation

The selected path ribbon is generated from `planner.path_polyline`, the same
source used by the current BEV. Its half-width approximates the ego body plus a
small visual margin.

When self-driving is off, the blue ribbon is absent. When AEB is actively
braking:

- the relevant forward or reverse corridor becomes red;
- the blocking actor/geometry becomes red when it can be associated;
- the HUD shows `AEB FULL BRAKE` and time to collision when finite.

Armed-but-clear AEB remains a small status indicator. The detailed dashed
corridor and brake-now chord stay in `RAW BEV`.

## Performance and boundedness

- GUI rendering target: 60 fps on the existing machine.
- Sensor/control cadence: unchanged at 40 ms.
- Scene construction target: median below 25 ms, p95 below 40 ms.
- Scene input queue: latest snapshot only, maximum one pending.
- Road surface history: fixed TTL and radius; no unbounded accumulation.
- Actor registry: bounded to current scenario vehicles.
- Mesh update rate may be lower than actor/camera interpolation; visual motion
  remains smooth through QML interpolation.

If build time repeatedly exceeds budget, surface construction coarsens its cell
size before reducing the LiDAR/control cadence.

## Degraded and error states

- **No LiDAR returns:** environmental geometry ages out on its normal TTL; actors
  lose corroboration and fade. The HUD shows `PERCEPTION UNAVAILABLE`.
- **Actor query failure:** road and path continue; actors fade and the event log
  records `Actor enrichment unavailable`.
- **Scene-worker exception:** discard temporal scene state and continue from the
  next snapshot.
- **QML/OpenGL failure:** select `RAW BEV`, disable `WORLD` for the session, and
  log the QML error.
- **Map load or pose jump:** detect a discontinuity, clear temporal voxels and
  actor tracks, and rebuild without dragging old geometry into the new pose.
- **View toggle:** both pipelines remain warm; switching does not show a stale
  reconstruction.

## Testing

The normal pytest suite remains offline: no BeamNG.tech and no
`QApplication`.

Pure tests cover:

- world/ego transforms and coordinate handedness;
- semantic grouping;
- voxel aging, expiry, radius bounding, and pose-jump reset;
- road mesh topology and height-discontinuity rejection;
- deterministic path-ribbon vertices and triangle winding;
- actor oriented-footprint corroboration;
- confidence rise, coasting, fade, and removal;
- no rendering of never-corroborated ground-truth actors;
- speed/curvature/reverse camera targets;
- latest-snapshot-wins queue behavior;
- empty and malformed input degradation.

Qt-facing classes use small unit tests where possible without constructing an
application. A separate manual/offscreen smoke command validates QML loading,
geometry binding, toggling, and fallback behavior.

Live acceptance checks:

1. WORLD and RAW BEV show the same chosen path and obstacle situation.
2. Traffic vehicles become solid only when LiDAR returns corroborate them.
3. An occluded or missed vehicle fades instead of teleporting or persisting.
4. Road geometry remains stable through normal motion and clears across a map
   load/teleport.
5. Cruise, intersection, reverse, and AEB camera states transition smoothly.
6. QML failure falls back to RAW BEV without affecting sensors or controls.
7. Poll/control p95 does not regress because scene construction runs on its own
   thread.

## Non-goals

- reproducing Tesla assets, branding, fonts, or proprietary behavior;
- exact BeamNG vehicle meshes;
- a camera-derived neural occupancy network;
- map-perfect roads painted from simulator ground truth;
- changing planner, controller, AEB, sensors, or safety semantics;
- removing the existing diagnostic BEV.

## Delivery sequence

1. Add pure scene data models, geometry construction, and tests.
2. Add bounded actor discovery/state polling to `BeamNgWorker`.
3. Add `SceneWorker` and latest-snapshot handoff.
4. Add Qt Quick 3D geometry bridge and QML scene.
5. Add WORLD/RAW BEV toggle and fallback.
6. Add telemetry, offline tests, offscreen QML smoke validation, and live-check
   documentation.
