# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Windows-only PyQt6 desktop app that drives BeamNG.tech 0.37.6 over BeamNGpy: it launches the
simulator with the communication bridge enabled, attaches four semantic LiDAR sensors to the
player vehicle, and renders their merged point cloud as an EGO-fixed bird's-eye view.
Python 3.12 (`py -3.12`), src-layout package `src/beamng_lidar_bev`.

## Commands

```powershell
py -3.12 -m pip install -r requirements-dev.txt   # runtime deps + pytest + ruff
py -3.12 -m pytest                                # pyproject sets pythonpath=["src"]; no PYTHONPATH needed
py -3.12 -m pytest tests/test_geometry.py::test_transforms_world_points_to_ego_right_forward_frame
py -3.12 -m ruff check src tests                  # E, F, I; line-length 88
```

**The documented interpreter is not what is installed** (verified 2026-07-27). This machine has
only Python **3.11.0**, so `py -3.12` silently resolves to it. PyQt6 is pinned `<6.8` in
`requirements.txt` and **6.7.1 is installed**: 6.11.0 could not be imported on 3.11.0 (DLL
procedure error), which broke `run_app.bat` (it gates on that very import). A second, distinct
failure hit only pytest: the globally installed **pytest-qt** plugin probes for a Qt binding at
startup, and with PyQt5/PyQt6/PySide6 all present it loads a *second* Qt runtime into the
process, after which `import PyQt6.QtCore` in `worker.py` dies with `0xc0000139` during
collection — even on a PyQt6 that imports fine standalone. `pyproject.toml` therefore sets
`addopts = "-q -p no:pytest-qt"`; the whole suite collects and passes with that in place.
Installed `beamngpy` is **1.35.1** against a `requirements.txt` pin of `1.34.1`, and
the simulator at `config.BEAMNG_EXE` is BeamNG.tech **0.38.5.0**, not 0.37.6.

Run the app: `run_app.bat`, or `$env:PYTHONPATH = "$PWD\src"; py -3.12 -m beamng_lidar_bev`
(the `.bat` uses `pyw` so there is no console window). Runtime logs land in
`logs/beamng_lidar_bev.log`; `__main__._configure_logging` resolves that directory as
`Path(__file__).parents[2] / "logs"`, so it depends on the `src/<pkg>/` layout staying put.

The whole test suite is offline — no BeamNG.tech, no `QApplication`. Keep it that way: tests
duck-type the simulator (`StreamingLidarStub`, `EmptyLidarStub`, `VehicleStub` in
`tests/test_worker_state.py`) and drive `BeamNgWorker` by assigning its private attributes or
calling methods unbound against a `SimpleNamespace`. `beamngpy` is imported lazily inside worker
methods (and under `TYPE_CHECKING` at module scope) specifically so importing `worker` stays
cheap and side-effect free.

Note `QTimer.start()` is a silent no-op off a `QThread`, so `tests/test_bridge_monitor.py` drives
`BridgeMonitor._probe()` directly and swaps in a `_TimerSpy` to assert the rearm interval rather
than checking `isActive()`.

The offline suite can only pin *arithmetic*; it cannot prove what the simulator does with a
value. Claims about sensor placement, FOV or ray budget need a live check — attach and read the
`Mount check:` line the worker logs (one per mount now, since they no longer share a height),
and confirm VISIBLE POINTS is in the tens of thousands with grey reaching the outer rings.

The **roof unit has not been live-checked at all** (added 2026-07-29). Three things the offline
suite cannot reach. Whether `density` holds the ray *count* constant as the FOV narrows or scales
it with solid angle is still undocumented and unmeasured — the `Sensor reach:` line prints each
unit's own return count and furthest return, which settles it. Whether a fifth unit at
`LIDAR_ROOF_DENSITY = 12.5` (≈ +57% total ray budget, still less than halving `LIDAR_DENSITY`
globally) costs sim frames — if it does, raise it to 25.0 first, which halves the azimuth
sampling and leaves the radial win untouched. And whether the road visibly fills: the prediction
is that the arcs-with-bands look disappears inside ~29 m immediately, before any of the WORLD
accumulation work. **It also puts new returns into the AEB height band from above, so the
phantom-braking checklist below has to be re-run before trusting it.** Self-driving has its own live checklist: the `Drive check:` line
at engage, the slope allowance at the 40 km/h cap on a hilly map (see `SLOPE_ALLOWANCE_PER_M`),
the steering gain settling near a per-vehicle constant while cornering, and a corner approach
that visibly brakes to the bend's entry speed rather than toward a stop.

AEB has its own live checklist, and the phantom-braking half of it is the part the offline suite
cannot reach. Two of its cases now have dedicated geometry behind them and both need re-proving on
a real map: **a gradient** (drive a hill at 40–70 km/h and reverse up a driveway ramp — the vertical
extent test is what should keep it ARMED, where the old slope cone could not) and **roadside
foliage** (drive the scrub — the porosity test is what should keep it ARMED). If either still
fires, the `AEB evidence:` line says which: a large height spread means it really was solid, a small
one means the extent threshold wants raising. Then the rest, unchanged:
read the `AEB check:` line at arm, then drive **with self-driving off** and confirm
the AEB metric never leaves ARMED — flat and empty **at well over the 40 km/h self-driving cap**
(the first reported phantom only appeared above 64 km/h, which self-driving can never reach), over
crests and dips, through corners close to the kerb, and under hard manual braking (the brake dive
is the case `AEB_OBSTACLE_MIN_HEIGHT_M` exists for). Every `AEB: BRAKING` line names its own
threat distance and required deceleration, and `threat none` in that line means the brake fired
with nothing detected at all — which is a bug in the trigger, not a sensing problem. Then confirm
it does fire, on a wall or a stopped car — and repeat the whole checklist **reversing**, where the
system arms at 3 km/h and every parking manoeuvre is a near miss by design. Both deceleration
figures are measured from that vehicle (the tables are in `config.py`), so re-measure before
trusting either in anything heavier. The
`AEB check:` line prints the brake-now distance for this vehicle at 30/50/70/100 km/h -- compare what the car actually does against those four
numbers, because "it braked too late" and "it braked too early" are the two complaints that need
a number to settle. Expect one continuous full application, never a series of pulses.

### The plant is one car's, and three log lines now say so

**Every braking figure in `config` is a property of ONE vehicle, and nothing recorded which**, so
"it phantom-brakes on the pickup but not on this" had no baseline to be measured against.
`PLANT_REFERENCE_VEHICLE` now names it and three diagnostics report against it. **None of them
changes what AEB does** — there is no per-vehicle registry, no auto-calibration and no altered
default; they exist so the decision about those can be made from numbers.

- **`Vehicle check:`** at attach — `vehicle.model`, the bbox, both overhangs, `ground_z_vehicle`
  and every mount height, plus a warning when the model is not the one the plant came from. **Read
  the WIDTH first.** It is the full oriented bounding box and the corridor both AEB systems scan is
  that width plus `AEB_CLEARANCE_MARGIN_M`, so anything bolted on (large mirrors, wide arches) is
  inside it — a corridor well over the real body width sweeps a band no collision can happen in,
  and a kerb face in it easily supplies the `AEB_MIN_HITS` returns.
- **`Brake measure:`** after any full stop, whether AEB fired it or a human did. The manual half is
  the important one: it needs no AEB event, and it runs **whether or not either system is armed**,
  because switching AEB off is the first thing anyone does when it brakes for nothing. Reports
  achieved mean and peak deceleration beside the configured figure, and the ratio. **Validate the
  instrument on the reference car first** — a hard stop there should read ≈10 m/s² and ≈1.0x — then
  measure the new one at 30/50/70 km/h forward and 10/20/30 reversing. Those rows are what a
  per-vehicle plant would be built from. Body pitch is logged with each one, because a stop down a
  grade flatters the plant and one up it slanders it.
- **`AEB evidence:`** once per firing. `AEB: BRAKING` names a distance, and a distance cannot say
  whether what blocked the corridor was a wall or the road surface arriving in the height band —
  which is the whole difference between a correct firing and a phantom. This line reports the
  **vertical extent** of the returns around the threat, which is invariant to both grade and ride
  height, plus the measured ground rise and the value the slope cone clamped it to:

  | spread (read against the range span) | what fired the brake |
  |---|---|
  | < ~0.10 m over metres of range | a surface lying along the ground — flat road under a brake dive, or a hillside crossing the floor because `ground_rise` is clamped into a 1.5% cone. **Not a plant or width problem** |
  | ~0.10–0.20 m | a kerb or low lip: real, but not a crash — corridor too wide, or the floor too low |
  | tens of cm over almost no range | genuinely solid, and the brake was right |

  A wide gap between the measured rise and the clamped one means the estimator saw a grade the
  floor was not allowed to believe. On any real gradient that is the expected reading, and it is
  a different defect from anything the vehicle's size causes.

## Architecture

### Two-phase startup (deliberate, don't collapse it)

`launcher.start_beamng_process()` spawns the exe with `-tcom -tport …` and returns
**immediately** — it never waits for the TCP bridge. The BeamNGpy handshake happens much later,
in `BeamNgWorker.attach_to_player`, after the user has loaded a map and picked a vehicle.
`launcher.bridge_is_reachable()` is a bounded `socket.create_connection` probe used to fail fast
with a readable message, because `BeamNGpy.open()` blocks hard when the bridge isn't up yet.
`BeamNGpy(..., quit_on_close=False)` plus the disconnect-only `shutdown` slot means closing this
app leaves BeamNG.tech running.

Launch is **not** a prerequisite for Attach. `BridgeMonitor` polls
`bridge_is_reachable()` from its own `QThread` and emits `bridge_up`/`bridge_down` on
transitions only; `MainWindow._on_bridge_up` enables Attach for any running session, however it
was started. Because `quit_on_close=False` deliberately leaves BeamNG.tech running, the common
case is that a session outlives the app — so `launch_beamng` also checks `bridge_is_reachable()`
before spawning, or it would start a second instance fighting for port 64256.

Self-driving and the two AEB systems are independently toggleable, and all three are gated on
`STREAMING` because the worker needs a live vehicle and four sensors for any of them.
`MainWindow` mirrors the worker's `self_driving_changed`/`aeb_changed`/`rear_aeb_changed` rather
than trusting its own buttons — the worker owns the truth, arms both brakes at attach, and
auto-disengages everything on faults and teardown.

`MainWindow._phase` (`IDLE`/`LAUNCHING`/`BUSY`/`STREAMING`) gates the monitor's slots. Don't
substitute `_last_status` for it: `"READY"` is emitted both by a successful `stop_sensors` and by
a *failed* attach, so it can't distinguish them. The `_set_*_enabled` helpers are idempotent
because a 2 s tick would otherwise reset the focus visuals of whichever button has focus, and
because `QMessageBox` spins a nested event loop in which queued bridge slots run.

### Threading

`MainWindow` owns a `QThread` and moves a `BeamNgWorker` onto it (`_start_worker`). Every
simulator call, `sensor.stream()` read, and numpy transform happens on that thread; the GUI
thread only paints. Cross-thread traffic is Qt signals in both directions — `MainWindow`'s
`launch_requested`/`attach_requested`/`stop_requested` into worker slots, and the worker's
`status_changed`/`frame_ready`/`fatal_error` back out. The polling `QTimer` is parented to the
worker so it lives on and fires on the worker thread. `closeEvent` is the one exception and must
stay a `QMetaObject.invokeMethod(..., BlockingQueuedConnection)` so sensors are removed before
the thread quits. Never touch `_bng`, `_vehicle`, or `_sensors` from the GUI thread.

**The state poll is PREFETCHED, and its socket safety is by construction, not by locking.**
`poll_sensors("state")` is a ~33 ms blocking round-trip — most of the 40 ms tick, against 0.5 ms
for all the LiDAR streams — so each tick submits the next tick's poll to a one-thread pool
(`_prefetch_vehicle_state`) as its **last** statement, after `_actuate` and the actor poll, and the
next tick collects it (`_take_vehicle_state`) as its **first**. Nothing on the worker thread
touches a socket between those two points, so the connection is never used from two threads at
once; beamngpy has no internal locking, which is why this ordering is the whole of the safety
argument — do not move either call. The collected position is advanced by `vel · age` to restore
the pose-to-cloud alignment a synchronous poll had; a state older than
`_STATE_PREFETCH_MAX_AGE_S` (an app stall) is discarded and re-polled. `_cleanup_sensors` calls
`_drop_state_future` FIRST — a bounded wait, never `result()`, because the socket has no timeout —
so teardown traffic cannot interleave with an in-flight prefetch. Pinned by
`test_two_rate_pipeline.py`.

WORLD adds a second `WorldSceneWorkerThread`. `BeamNgWorker.perception_ready`
queues immutable `PerceptionSnapshot` objects to `SceneWorker`, whose one-slot
handoff retains only the latest pending snapshot. Surface meshing and actor
corroboration therefore cannot consume the already tight BeamNG/control tick
or create a stale-frame backlog. `SceneWorker` is Qt-aware only for signals and
timers; all scene arithmetic is in the Qt-free `world_scene.py`. Scene failures
clear visualization state and emit `scene_error`; they never enter the
BeamNG-worker poll-failure budget or disengage controls.

### Frame pipeline

`worker._poll_once` (every `DISPLAY_INTERVAL_MS`) → `sensor.stream()` on all four LiDARs →
(when self-driving or AEB is on, `planner.geometric_obstacle_sets` once for both floors, then the
plan and the AEB step, each in its own `try` so one fault never switches the other off) →
concat + drop non-finite → `geometry.world_points_to_bev` (gravity-referenced heights) → radius
filter → drop returns inside
the ego bounding box → `semantics.classify_road_points` → decimate by stride to
`MAX_ROAD/OBSTACLE_RENDER_POINTS` → emit a `BevFrame` → `BevWidget.set_frame` →
`raster.rasterize_points` builds an RGBA numpy array wrapped as a `QImage`, cached until the
frame or widget size changes.

The same successful poll emits a `PerceptionSnapshot` after actuation. Unlike
`BevFrame`, it retains filtered world XYZ and a compact semantic group per
point. It carries cached traffic observations from one batched
`vehicles.get_states()` request (10 Hz; registry refresh 1 Hz), the current
plan/AEB states, vehicle pose and geometry. Actor enrichment happens after
control so an occasional traffic-state round trip cannot delay the current
command.

Sensors are created with `is_streaming=True` + `is_using_shared_memory=True`, so BeamNG writes
each scan into shared memory and `stream()` returns the latest one without a round trip. The
display loop must never call the blocking `sensor.poll()` — `StreamingLidarStub.poll` raises to
guard that invariant. After `_POLL_FAILURE_GRACE_S` of *continuous* failure the worker tears down
sensors, drops the connection and emits both `fatal_error` and `sensors_stopped`. The budget is
time-based on purpose: the old three-strike count was a 99 ms grace period, far too eager to
survive a map load, and it never emitted `sensors_stopped`, so the dead frame stayed on screen.

**Nothing may gate on the sensor COUNT.** `_poll_once` and both `set_self_driving`/`_set_aeb`
guards each carried a literal `len(self._sensors) != 4`. Adding the roof mount made all three
reject at once, and the failure was completely silent: the display loop returned before its
first statement, so no frame was emitted, no poll ever failed, the `_POLL_FAILURE_GRACE_S`
budget was never touched and nothing logged — the badge read STREAMING over a frozen, empty
view, and AEB answered "attach to a vehicle" immediately after a successful attach.
`_sensor_set_is_complete` is a presence check (`bool(self._sensors)`) rather than a count,
because the count was never load-bearing: `attach_to_player` wraps the whole build loop and every
failure funnels through `_cleanup_sensors`, so the list is either empty or the full set.
`test_the_display_loop_does_not_depend_on_how_many_sensors_there_are` pins it over 1/3/4/5/6
sensors — verified to fail on every count except 4 against the old guard.

**`_poll_once` emits a frame on every tick, including when no sensor returns anything.** Empty
`(0, 2)` arrays flow through the whole render path safely, so the widget keeps drawing the grid,
rings and ego polygon while VISIBLE POINTS honestly reads 0 and EGO SPEED keeps updating. The
early `return` that used to skip the emit froze the display *and* every metric while the badge
still read STREAMING. `_frame_times` is flushed rather than appended to after a gap, so
ACQUISITION decays to 0.0 Hz instead of smearing a stale rate across the outage.

### Coordinate frames

Three frames are in play and mixing them up is the easiest bug to introduce here:

- **World** — what BeamNG returns for vehicle `pos` and every LiDAR point.
- **Vehicle** — BeamNG convention: **+X left, +Y rearward, +Z up**. Sensor mount `pos`/`dir`
  passed to the `Lidar` constructor are in this frame.
- **BEV display** — `(right, forward)`, produced by `world_points_to_bev` via the orthonormal
  basis from `geometry.vehicle_axes` (which re-orthogonalizes forward against up).

`VehicleGeometry` bridges them by storing *positive* distances: `left_m = max_x`,
`right_m = -min_x`, `front_m = -min_y`, `rear_m = max_y`. That is why `BevWidget._draw_ego`
negates mount coordinates (`screen(-local_x, -local_y)`) when drawing the sensor markers —
it is converting vehicle space to right/forward, not correcting a sign error.

**Sensor `pos.z` and `ground_z_vehicle` are different data and must not be mixed.** The
simulator already references a vehicle-space sensor `pos` to the vehicle's ground plane, so
`derive_vehicle_geometry` passes `SENSOR_HEIGHT_ABOVE_GROUND_M` through **verbatim**. Adding
the bbox bottom on top (as it once did) buried the sensors ~3 cm underground, which silently
killed every downward ray, collapsed the 179° horizontal sweep to 37°, capped the BEV at 11.6 m
and classified 0% of returns as road. `ground_z_vehicle` is a different quantity — the bbox
bottom — and is the reference for heights. `worker._verify_mount_height` logs the measured
offset at attach time and warns past 0.15 m, which turns that whole class of regression into one
log line; `test_mount_height_ignores_the_bbox_bottom` pins it offline.

**Heights are referenced to gravity, not to the body's up axis, and this is load-bearing.**
`world_points_to_bev` returns `offsets[:, 2]` (BeamNG's world Z is up) and `ground_z_vehicle` is
measured along world Z to match; only the `(right, forward)` projection stays body-referenced,
where a 3° error costs 0.04 m over 30 m. Measured against `state["up"]` instead, every degree of
pitch tips the whole cloud: at 1° nose-down the flat road 15 m ahead reads 0.26 m high against a
0.20 m obstacle floor, and at 3° the road at 25 m reads 1.31 m. A road car pitches 1–3° under
ordinary braking, so the planner saw a wall across the road, braked, pitched further and saw
more wall — a latch that presents as "it brakes for no reason", and as "it gets stuck" once the
phantom crossed `STOP_MARGIN_M`. `test_heights_are_referenced_to_gravity_not_the_body_axis`
pins it.

### Semantic classification

The annotation palette is read from the **running simulator** (`bng.camera.get_annotations()`),
falling back to `<BEAMNG_HOME>/tech/annotations.json`, so class colours track whatever map is
loaded. `semantics` packs RGB triples into `uint32` (`pack_rgb_rows`) so classification is a
vectorized `np.isin` over the whole cloud. BeamNG 0.37's `annotations.json` contains a channel
value of `256`; `_normalise_rgb` applies `% 256` to match the renderer's uint8 wraparound —
without it those classes never match.

`classify_road_points` gives semantics precedence and applies the height band **only** to
unknown or `BACKGROUND` labels. This is what keeps unannotated community maps usable without
turning known `GRASS`/sidewalk returns grey; `test_semantics.py` pins that precedence.

**The palette is matched ONCE per tick.** `classify_scene_groups` shares one `pack_rgb_rows`
pass and one `_road_mask` with `classify_road_points` instead of calling it, and the worker takes
its BEV split from `groups == SCENE_ROAD` rather than classifying a second time. That was three
packs and nine `np.isin` sweeps over the whole cloud where one pack and six do; every one is
O(cloud) and the fifth sensor made all of them dearer. The equivalence holds because the four
code sets are disjoint by construction — `boundary_codes` is defined as everything not road,
fallback, vehicle or vulnerable — so nothing the road rule accepts can be overwritten by the
group assignments that follow. `test_scene_road_group_is_exactly_the_road_mask` pins it across
the height band.

`classify_scene_groups` preserves a second, compact vocabulary for WORLD:
road, vehicle, vulnerable road user, boundary/static, and unknown. WORLD's
hybrid honesty rule is load-bearing: BeamNG supplies a traffic actor's stable
identity/type/pose/velocity, but the generic model is not rendered until
semantic vehicle returns overlap its oriented footprint. Confirmed actors
coast briefly through a missed scan, fade by `WORLD_ACTOR_FADE_S`, and then
disappear. Never render every simulator actor unconditionally; that would turn
the display into ground truth rather than a perception visualization.

**`classify_surface_materials` is a THIRD, orthogonal question, and keeping it separate is what
makes unannotated maps work.** The groups say what kind of thing a return belongs to; the
materials (`SURFACE_PAVED`/`SIDEWALK`/`VEGETATION`/`BARE`/`WATER`/`UNKNOWN`) say what the ground is
made of, and nothing else. **Shape decides what IS a surface; semantics only decide what colour the
surface it found should be** — so a map with no annotations at all still gets a floor, in the
unidentified-ground colour, and `NATURE` appearing in `VEGETATION_CLASSES` does not turn a tree
canopy into grass, because the canopy is a tall run that never reaches the surface mesh.

It is one `searchsorted` over a sorted `(colour → material)` table rather than an `isin` sweep per
material: the table is 29 entries and the cloud is 50–110k, and this tick already runs six O(cloud)
sweeps. `ROAD_CLASSES` still answers its own separate question — *may the car drive here* — and
anything the road rule accepted but no material named is forced to `SURFACE_PAVED`, so the
geometric fallback band keeps looking exactly as it always has.

**A car's VISIBILITY must never depend on that actor path, and it once did.** Vehicle returns
were excluded from the scene geometry on the understanding that traffic would be drawn as
corroborated actor models instead — and `_poll_actor_observations` gets its poses from
`bng.vehicles.get_states()`, which BeamNG.tech **rejects in free-roam**, exactly as
`worker._get_vehicle_state` already documents for the ego. In the normal workflow (load a map in
free-roam, pick a car) no actor is ever confirmed, so traffic was drawn by neither route: a car
was completely invisible, not even a solid. `SCENE_VEHICLE` now feeds the voxel store like any
other solid and is meshed as its own class in the actor blue, so a car is drawn from LiDAR alone;
the ground-truth model is enrichment on top. Traffic runs on `WORLD_VEHICLE_TTL_S` rather than
`WORLD_COLUMN_MEMORY_M`, because a car MOVES and the scenery window would draw it as a streak of
itself — and, as below, the two are not merely different thresholds but different **clocks**.

### WORLD scene assembly

**Static geometry is forgotten by the METRE and traffic by the SECOND, and that is two clocks, not
two thresholds.** What the display shows is ego motion sweeping the LiDAR's ground rings and
azimuth stripes through the world; a wall-clock TTL was only ever a proxy for how much of that
sweep sat in the window. Stop the car and the sweep stops while the clock does not, so the store
drained and the view collapsed to what a single stationary frame resolves — concentric arcs with
empty bands between them, which is the whole complaint.

`WorldSceneAssembler` therefore keeps its own odometer, summed in `_track_ego_motion` from
successive `snapshot.ego_pos_world` values, and road cells and static voxels are stamped with it
(`WORLD_CELL_MEMORY_M`, `WORLD_COLUMN_MEMORY_M`). Three things make that safe here that would not
be safe on a real vehicle:

- **Ego pose is ground truth.** There is no odometry drift, which is the reason real mapping
  systems must decay old observations at all. Static geometry seen a minute ago is exactly as valid
  as geometry from this frame.
- **Anything that moves is class-separated already** and stays on the wall clock. A car crossing in
  front of a *stopped* ego must still fade in `WORLD_VEHICLE_TTL_S`, and it cannot if it shares an
  odometer that is not advancing — so `_expire_boundary_columns` selects per class between two
  parallel stamp arrays (`_voxel_seen`, `_voxel_travel`). Both are non-decreasing, so
  `maximum.reduceat` still means "most recently seen" under either.
- **"Newest wins" never compared the stamps.** `_update_road_cells` relies on a stable sort with
  this frame's cells appended last, so a stamp that *stalls* while parked — rather than a clock that
  always advances — changes nothing there.

**The road store had no bound of its own, and distance-stamping is what forced it to grow one.**
The radius test used to live in `_road_mesh` and decided only what was *drawn*; the 1.2 s TTL was
the only thing bounding the store itself. Parked, a distance-stamped store would have kept every
cell the sensors ever reached. The cull now sits in `_expire_road_cells` with a
`WORLD_MAX_ROAD_CELLS` cap beside it, mirroring `_expire_boundary_columns` exactly, and `update`
expires before it meshes so the drawn surface is unchanged.

Two known costs, both intended: geometry that *changes* while you sit still persists until it is
re-observed (only free-space carving fixes that), and the stores are larger on a long drive — watch
SCENE BUILD, which already logs over-budget builds.

**Both accumulators are parallel numpy arrays, and every stage is vectorized.** `WorldSceneAssembler`
holds road cells and boundary voxels as `(N, 3)` int keys plus value/timestamp arrays, never as
a dict of dataclasses, and `merge_cell_runs` finds X-runs with numpy so its Python loop iterates
*runs* rather than *cells*. Measured on a 51k cloud over an open 40 m radius (17.6k road cells),
the scene build went **109 ms → 25.7 ms** against a 40 ms tick — it was running WORLD at 9 Hz.
Three separate things were wrong and all three were per-cell Python:

- `_road_mesh` filtered by radius with a `np.linalg.norm` call **per cell** in a list
  comprehension (66 ms), and the merge then did a dict lookup and tuple build per cell. (That
  filter has since moved into `_expire_road_cells` — see above — but it is the same one pass.)
- `_update_road_cells` built a `_RoadCell` per unique cell in a Python loop (33 ms).
- `np.unique(..., axis=0)` sorts a void view of each row and is dramatically slower than sorting
  plain integers — 41 ms against 6 ms for the same cells. `pack_cell_keys` packs three 21-bit
  fields into an int64 so every `unique` stays 1-D. That covers ±524 km of cell index.

`SceneWorker.build_time_changed` now drives the SCENE BUILD metric, and an over-budget build is
logged (throttled) because the band is hidden in WORLD — which is exactly when it matters. The
signal existed but was connected to nothing, so none of the above was visible.

**`merge_cell_runs` picks its scan axis per call, and on walls that is worth an order of
magnitude.** Its Python loop costs one iteration per run, so scanning across a long straight wall
is the worst possible orientation: a 200 m wall parallel to Y is 800 single-cell X-runs, one per
row, against a handful of Y-runs. Fixing the axis at X put **7.4 ms a tick** into merging four
such walls once the view reached 150 m. Counting both ways first is two vectorised passes and
picks the cheaper on exactly the geometry a street scene is made of. Measured on a 140k-point
worst case, the whole build went 39.4 ms → 33.4 ms from that change alone, and 32.5 ms once
`pack_cell_keys` stopped being recomputed on the reordered keys it had just packed.

**Sort once. Sorting IS the cost of both accumulators**, and the quarter-metre grid multiplied the
cell count by four, so passes that were affordable at 0.5 m stopped being so. `_update_road_cells`
ran `np.unique`'s internal sort, then a second `np.unique`, then a `lexsort` — three sorts where
one stable `argsort` on the packed key does the job, because a stable sort with this frame's cells
appended last makes "newest wins" simply "take the last of each group", with no timestamp
comparison. `_update_boundary_columns` likewise ran `np.unique` and then re-sorted its own
inverse. Removing the redundant passes took the build from **24.0 ms → 20.9 ms** on a 73k cloud.

**A run too short to be structure is dropped in `_column_runs`, BEFORE bridging, and that is worth
half the build.** `_slab_mesh` used to bridge every run and then discard the ones under
`WORLD_MIN_SLAB_HEIGHT_M`. On open ground — a dirt yard, or anything whose surface is not
road-classified and so arrives as boundary returns — that meant bridging tens of thousands of flat
ground runs across two axes and throwing nearly all of them away: `bridge_gaps` alone was 41 ms of
an 85 ms build, against a 40 ms tick. Measured on a synthetic street with striped facades, kerbs and
open ground, **69.1 ms → 26.7 ms with every structure over a metre tall preserved exactly** (41
boxes either way). What disappears is short ground fragments bridging was inventing, so it is also
the more honest rule: interpolating between a flat ground column and a kerb manufactures a ramp
nobody observed. Note the tidier-looking alternative — pre-culling whole *layers* that cannot
survive — is provably output-identical but saves only 5 ms, because `WORLD_MIN_SLAB_HEIGHT_M` (0.10)
sits inside the first `WORLD_SLAB_HEIGHT_BUCKET_M` (0.5) bucket, which is exactly where the ground
lives.

**The road is bridged at MESH time and the store is never told.** `bridge_gaps` is shared by both
accumulators because both suffer the same sampling asymmetry, just on different axes — see the
road-mesh section below. Bridging the *store* instead would accumulate an inference as though it
were an observation, and it would then outlive its own memory window.

**Boundary returns are accumulated VOXELS extruded into slabs, not billboards and not columns.**
They were once one 0.16 × 0.32 m card per point, rebuilt from the current snapshot only and
capped at 4,000 marks: 70–80% of the wall evidence was discarded before rendering, the survivors
were re-chosen every frame from a re-ordered array so they shimmered, and a wall drawn as
confetti has gaps between the confetti by construction. Now world-anchored `(x, y, height-bin)`
voxels keep min/max height over a `WORLD_COLUMN_MEMORY_M` window, collapse into vertical runs, merge
into slabs and extrude. A dense 10 m wall costs **24 vertices against 2,560**.

**The third key field is what lets a tree be a tree, and it is not a refinement.** The store used
to hold one `(base, top)` span per XY column. Grass and terrain are boundary returns — the group
is defined as everything that is not road, vehicle or vulnerable — so the column under a canopy
holds returns at ankle height *and* at 3–7 m, and a single span smears the two together. That
rendered a tree as a solid block from the treetop down into the grass, which is what it looked
like in the app. The same defect drew a bridge deck, an overhead gantry and a tunnel roof as
columns to the ground. Voxels keep the void, `_column_runs` finds runs of occupied height with
`WORLD_COLUMN_VERTICAL_BRIDGE_BINS` of slack for sampling noise, and the gap survives to the
screen.

Five things there are load-bearing:

- **The gap is azimuth, and the roof unit does not touch it.** A wall is sampled at `r·Δθ`
  vertically and `r·Δazimuth` horizontally — 0.04 m against 1.24 m at 20 m, thirty to one — so it
  arrives as vertical stripes over a metre apart and extruding stripes just gives striped
  buildings. `WORLD_COLUMN_BRIDGE_CELLS` interpolates across single-frame stripe gaps; it is
  bounded so it can never close an opening the car could drive through, which
  `test_azimuth_stripe_gaps_are_bridged_but_a_real_opening_is_not` pins from both sides.
- **Slabs merge only within an altitude AND height bucket** (`WORLD_SLAB_HEIGHT_BUCKET_M`), or a
  6 m facade and the 0.15 m kerb in front of it average into one waist-high wall, and a balcony
  merges with the wall under it. The layer key packs both buckets into one integer;
  `merge_cell_runs` only ever merges within one layer value, so any injective pairing works.
- **A voxel keeps the tallest look it ever got.** The vertical FOV means a wall's observed top
  *falls* as you approach it, so taking the current frame's maximum makes buildings shrink as you
  drive at them.
- **A run that you can pass under is culled, and the test is on where it STARTS, never where it
  ends** (`WORLD_COLLISION_CEILING_M`). Clipping tall things to roof height instead would make a
  garden wall and an office block the same object, which is the distinction the view exists to
  draw. The ground reference is the lowest boundary return in that same column, so it follows
  terrain; where a structure hid its own footing (under its own canopy, or over a road, which is
  never a boundary return) the ego's ground plane stands in. The known cost is that a structure
  perched more than the ceiling above the ego plane — the top of a steep embankment — is culled
  until its footing comes into view. That is the intended direction of error.
- **The store is left SORTED by packed `(x, y, bin)` and `_column_runs` depends on it.** That key
  order groups by column with the bins ascending inside each one, which is exactly what finding
  vertical runs needs, so the run pass sorts nothing. Expiry only ever drops rows, so it
  preserves the order; anything that reorders the voxel arrays must re-sort or the runs silently
  fragment.

### The chase camera

**Every quantity is damped toward a target, and nothing was.** `_camera` used to be a stateless
pure function, so continuity came entirely from speed being continuous — which is exactly why the
reverse flip teleported, and why any state keyed on a threshold would have snapped. It is now an
instance method holding a `CameraPose`, with the target still computed by a pure
`camera_target(snapshot, parked, alerting)` and the step by a pure `damp(current, target, dt, tau)`.
Four things about it are load-bearing:

- **The position is DERIVED from an orbit angle**, not chosen per branch. `(d·sin θ, h, d·cos θ)`
  reproduces the old `(0, h, ±d)` at θ = 0 and 180 exactly, and damping that one number sweeps the
  reverse swing round the *side* of the car instead of cutting through it. The yaw is kept as a
  plain scalar rather than wrapped: the only targets are ~0 and ~180 plus a bounded corner offset,
  so it never leaves [-30, 210] and there is no shortest-path ambiguity to resolve.
- **The first frame snaps.** A pose of `None` means "no pose yet", so a fresh assembler lands on its
  target rather than easing in from a guess — which is what keeps every camera test that builds a
  fresh assembler measuring the scene instead of the initial condition.
- **There is ONE framing, and standing still is not a special case of it.** A near-vertical
  standstill tilt was built — with a hysteresis *and* a dwell, because parking manoeuvres live at
  0.3–1 m/s and a single threshold nods the view every time the car creeps — and then removed. The
  gating was never the real problem: every threshold that can switch framings sits inside the band
  ordinary driving spends real time in (junctions, queues, traffic, parking), so no amount of
  hysteresis stops the view changing shape while the situation has not. The speed terms already
  close the view in as the car slows, so the second framing bought little, and distance is cued by
  depth tint and by a *stable* frame. Don't reintroduce it as "just a small tilt": the failure is
  the mode switch, not its magnitude. `WORLD_CAM_PITCH_LIMIT_DEG` (−80°) survives as a guard —
  at exactly −90° the euler yaw is degenerate and the view spins on its own — but nothing
  approaches it now, so it constrains whatever pitch term comes next rather than anything today.
  Pinned by `test_stopping_does_not_change_the_framing`.
- **The AEB framing move is gated on `_alert`'s own string**, which `update()` already computes and
  which is non-empty only while a pedal is down. Reusing it is what guarantees the camera and the
  overlay agree about what an event is; a view that moved for the armed state would be a nuisance
  rather than an alarm.

Curvature for the corner lean comes from the plan when self-driving, and otherwise from
`AebState.curvature` — which is derived from *measured* yaw and runs under a human driver, which is
when the camera most needs it. Verified on the real D3D11 backend by rendering a synthetic drive
through `WorldView.grab()`, per the pixel-questions-get-measured rule below.

**The mouse orbit is an OFFSET on the chase pose, not a second camera.** Right-drag rotates,
the wheel zooms, a right double-click resets, and there is deliberately no panning. `WorldView`
holds the offsets and `world_scene.apply_view_orbit` (pure, pinned by `test_view_controls.py`)
lays them over whatever pose the damped chase camera produced — so the view keeps following the
car under a user orbit, the one-framing rule is untouched (nothing moves without the user's
hand), and identity inputs return the pose bit-for-bit, leaving the default view byte-identical.
`SceneBridge` keeps the raw chase pose beside the displayed one so a mouse move re-aims between
frames, and the elevation clamp (3°–85°) stops the orbit at the same euler degeneracy
`WORLD_CAM_PITCH_LIMIT_DEG` guards.

### WORLD renderer and RAW BEV toggle

`WorldView` embeds `qml/WorldScene.qml` in a `QQuickWidget`. `SceneBridge` owns four
`SceneGeometry` objects — `QQuick3DGeometry` subclasses that take the numpy buffer **verbatim**
via `setVertexData` — and QML binds `Model.geometry` to them once. An `ActorListModel` drives
low-poly vehicle delegates. The scene uses render coordinates
`(right, gravity-relative height, -forward)`. The ego and actors are generic Qt Quick 3D
primitives — no Tesla or BeamNG mesh assets.

**The raw-buffer bridge is what sets the detail of the whole view, and it replaced
`ProceduralMesh`.** `ProceduralMesh` takes `list<vector3d>`, so feeding it meant building one
`QVector3D` per vertex in a Python loop **on the GUI thread, every frame**. The grid sizes were
therefore set by what that loop could carry rather than by what the sensors resolve — which is
what "it looks like very big pixels" was. `setVertexData` is O(1) Python in vertex count: the
same scene went from 188 road vertices to 20,472 while the GUI-thread cost *fell* from 1.2 ms to
0.3 ms. Two mechanics matter:

- **Attributes are declared once, in the constructor.** `addAttribute` appends, so re-declaring
  per frame grows the list without bound. Per-frame work is `setVertexData`/`setIndexData`/
  `setBounds`/`update()` only.
- **`QQuick3DGeometry` will not take a plain `QObject` parent** (it wants a `QQuick3DObject`), so
  the bridge holds Python references and pins `CppOwnership` explicitly. Without that, ownership
  of an object still bound to a live `Model` is the engine's guess.

The interleaved layout — xyz then linear rgba, 28 bytes — is a silent contract with
`addAttribute`: a mismatch renders as garbage geometry rather than raising, so
`test_world_view_buffers.py` pins it.

`MainWindow.visual_stack` keeps WORLD and the existing `BevWidget` alive at the
same time. The header toggle is GUI-only: it does not touch sensor or control
state. WORLD hides the diagnostic metric band and point legend; RAW BEV
restores them. A QML load error disables WORLD for the session, selects RAW
BEV, and logs the exact error.

The header also carries five per-unit coverage toggles (RAW BEV only, hidden in WORLD so they
never promise something that view won't draw). Each draws its unit's wedge from
`geometry.sensor_coverage` over the same `SensorMount` the sensor was built from, so the overlay
is provably the requested aperture. The roof unit draws as its ground annulus
(`LIDAR_ROOF_NEAR_M`–`FAR_M`) rather than its slant range, because every one of its rays points
below the horizon and the annulus is the road it was fitted to. The BEV also wheel-zooms
(1x–8x): the zoom re-rasterizes at the new radius rather than scaling the cached image, and
every overlay takes the same `radius_m` so they cannot drift apart.

**Two luminance steps, then hue — and a light air is what forces that.** Air is `#d7dadc`
(relative luminance 0.698), so air-to-black is 14.96:1 in *total* and supports only two steps at
the 3:1 floor for a graphical object, since 3³ = 27 > 15. Giving road, boundary, actors and the
ego a rung each is not possible: lifting the road to separate it from buildings pushes the road
into the air. The ladder is therefore

```
air #d7dadc -> road #6a7176      3.53:1    the drivable surface against the void
road        -> obstacle band     3.46:1    everything solid, in ONE dark band
```

with `boundary #171c20`, `actor #1b3c5c` and `ego #1c2126` all inside that dark band, separated
by **hue and lighting** rather than luminance: boundary is flat unlit near-black neutral, actors
are lit blue so they shade and carry a glass panel, the ego is neutral and always centre screen.
`path #4ea8f2` deliberately breaks the ramp — it is a guidance overlay rather than something
perceived, so chroma is its channel (1.94:1 vs road). `uncertain #545c62` is deliberately the
weakest mark at 1.37:1, because it is the least confident thing drawn.

The old palette had this inverted: empty air was the *brightest* surface, so a building
silhouetted against it was **1.35:1**.

**The surface materials all live on the ROAD's rung and separate by HUE, and that is forced, not
chosen.** With only two steps in the whole range, a material that separated itself by lightness has
to leave the rung and collide with the air or with the obstacle band — the first attempt at a light
concrete sidewalk (`#848a90`) came out at **2.48:1 against air**. The usable band is arithmetic:
relative luminance 0.1335–0.1992 for 3:1 both ways. Every `WORLD_SURFACE_*_RGB` was therefore solved
in CIELAB at a chosen hue and lightness and converted back, not picked by eye; measured worst case
is 3.04:1 against the obstacle band, 3.08:1 against air.

**Contrast is the wrong instrument for telling two of them apart** — two colours of equal lightness
are 1.0:1 apart however different they look — so `test_world_palette.py` measures pairwise **CIELAB
distance** instead, and requires ΔE ≥ 6. `SURFACE_PAVED` *is* `WORLD_ROAD_RGB`, aliased rather than
copied, so every existing contrast fact about the road still describes what is on screen.
`surface_unknown #6d6569` is deliberately the closest pair to it (ΔE 7.1) for the same reason
`uncertain` is the weakest mark: it is ground the sensors resolved but nothing identified. Rock was
tried as its own grey and abandoned at **ΔE 6.7 from paved** — paved, sidewalk and rock are three
greys and a band this narrow will not hold them, so hard unpaved ground is one material.

**The road is one continuous surface with SHARED corners, not a quilt of merged rectangles.** It
used to be `merge_cell_runs` output drawn flat at each rectangle's mean height; neighbouring
rectangles share no vertices, so every difference in mean height was a hard step and a sloping
road terraced. Cells now share their lattice corners, which removes the failure rather than
reducing it — a corner holds one height, so two cells meeting there cannot disagree. There is no
Python loop left in `_ground_mesh` at all, which is why dropping the merge is *cheaper* than
keeping it despite emitting far more geometry. A corner holds one *material* and one *fade radius*
for the same reason, so both average over the same cells the height does.

**The ground mesh has TWO sources and only paved ground comes from the road store.** Everything the
annotations did not call road arrives as a `SCENE_BOUNDARY` return, and the flat ones used to be
dropped in `_column_runs` for being too short to be structure — so grass, dirt, a gravel yard, and
the whole of any unannotated map rendered as **nothing at all**, a hole where the ground is. Those
runs are now promoted to the ground surface, which is where their shape says they belong. Four
things about it are load-bearing:

- **The threshold is `WORLD_MIN_SLAB_HEIGHT_M` itself, not a second constant**, so the promotion
  takes *exactly* what was being discarded and nothing that draws as a slab today changes. A 0.12 m
  kerb is still structure the planner steers around, not floor.
- **A promoted run must be the LOWEST in its column**, never merely a short one. A flat roof, the
  top of a wall and the underside of a canopy are all short runs sitting above something, and a
  floor drawn at roof height is a floor through the middle of the building.
- **There is deliberately no ceiling test on it.** A hillside 10 m above the ego plane is still a
  surface; `WORLD_COLLISION_CEILING_M` asks whether something can be driven *into*, which is a
  different question from whether it can be stood on.
- **Where both sources hold a cell the ROAD wins** — "the car may drive here" is the more specific
  claim, and it is the one the view exists to make. It falls out of the same stable-sort
  "last one wins" mechanic `_update_road_cells` uses, with the road appended last.

`_column_runs` is called ONCE in `update` and its answer handed to both consumers, because the
ground surface and the slabs are the two halves of one reduceat over the voxel store: what is flat
enough to stand on, and what is not.

**Going to 0.25 m cells outruns the sampling at range, and the road must be bridged or it breaks
into a checkerboard.** Ground returns thin as `r²` radially and as `r` in azimuth, so past
roughly 20 m they stop reaching every quarter-metre cell; without `WORLD_ROAD_BRIDGE_CELLS` the
far road renders as a lattice of disconnected quads, which reads as a far *worse* "big pixels"
than the coarse grid it replaced. This is the same `bridge_gaps` the boundary columns use, for
the same underlying reason, and it is applied to the mesh and never to the store.

**`tests/test_world_palette.py` recomputes every one of these from the colours the QML actually
ships, and it exists because the first version of this section was wrong.** The ratios were
written from an estimate rather than a calculation, and the estimate hid a real defect — actors
shipped at `#394046` against a `#3c4348` boundary, i.e. **1.05:1**, so a traffic car and a wall
were the same colour. A contrast number in a comment is not evidence.

**The palette is not in the QML any more, and the four unlit materials ship pure white.** Every
unlit surface is vertex-coloured, so `config.WORLD_*_RGB` holds the colours and the vertex buffer
carries them. Two things have to vary *within* one mesh and a material constant can express
neither: range, and face orientation. `test_world_palette.py` reads config for the unlit surfaces
and the QML for the lit ones (ego, actors), and pins that `clearColor` still equals
`WORLD_AIR_RGB` — the depth tint mixes toward that value, so a drift between them grows a visible
band at the horizon instead of a dissolve.

Three GPU-measured facts underpin all of it (Qt 6.7.1, D3D11 — probes rendered and pixel-sampled,
not reasoned about):

- **A `NoLighting` `DefaultMaterial` multiplies base by vertex colour in LINEAR space.** A white
  base plus `linear_rgb(target)` reproduces `target` exactly, checked against all five palette
  entries. Writing sRGB straight into the buffer would darken everything — `#6a7176` would land
  on `#3f4448`.
- **A vertex colour above 1.0 clamps to white**, so brightening past the base is not available;
  everything written stays in range.
- **`SceneEnvironment` `Fog` is a NO-OP on `NoLighting` materials.** CLAUDE.md flagged this as
  undocumented and it resolves the pessimistic way: a red fog from 20 m to 60 m over geometry
  spanning 5–75 m did not move one red channel. Every large surface here is `NoLighting`, so the
  `Fog` block was decorative and is gone. **Distance is cued by `world_scene.depth_tint`**, baked
  per vertex, which no shader can ignore.

**The AEB overlay is drawn from `AebState`'s own curvature and half-width** through
`aeb.predicted_corridor`, so what is on screen is provably the corridor that was scanned — the
same reason `bev_widget` imports the function rather than reimplementing the arc. Every element
is a strip between two matched corridor edges (`corridor_edges` + `_strip`), so none of them can
overstate where the system is looking. Violet while armed, red once the pedal is down, so the
colour change *is* the event. The rear system is un-rotated on the way out, exactly as
`bev_widget._aeb_to_screen` does — it reasons in a 180°-rotated frame, which is what lets it
share every arc helper in `planner`.

Five elements, each carrying something the others cannot:

- **Rails** run the full `horizon_m`, not to the threat. "I checked this far and found nothing"
  is a distinct statement from "this stretch is dangerous", and without it a clear corridor and a
  blind one look identical. They fade exponentially with range so a 100 m scan at motorway speed
  does not end in two hard lines at the horizon.
- **Wash** appears ONLY when there is a threat, and stops at it. Filling the scanned length is
  the same error as scoring an empty road as an obstacle parked at the horizon — and it rendered
  as a solid white band down the whole road, drowning the scene it sits on.
- **Brake-now bar** at `brake_now_m` — the last point at which braking still works, and the
  trigger the entire system turns on. The gap between it and the threat *is* the margin left,
  which nothing else on screen shows.
- **Threat marker**: a panel fading upward, framed on three sides so it reads as a reticle rather
  than a floating card, with a pool of light where it meets the road to anchor it to a place on
  the surface.
- **Urgency** scales the wash between watching and firing, so the corridor builds as a threat
  closes instead of snapping on. It rides on alpha alone — hue stays reserved for STATE, because
  a colour change is what reads as an event.

**Transparency is per-vertex, and that was measured, not assumed.** Vertex alpha multiplies the
material's own opacity exactly: a black quad at vertex alpha 1.0/0.5/0.0 over white under a 0.8
material rendered 51/153/255 on the real GPU. Both AEB materials are therefore left fully opaque
and the buffer carries the whole ramp, which is what lets one mesh hold a 0.04 wash and a 0.80
rail at once. **Blending is linear**, so a bright hue at a low alpha *lightens* the grey road
rather than saturating it — the braking red at a watching alpha reads as a pink glow, and in a
palette where dark means important that reads as less urgent, not more. `WORLD_AEB_BRAKING_BOOST`
is what recovers it: the same red at 0.71 alpha lands on (194, 74, 65) instead of (163, 92, 85).
Reach for alpha before reaching for a different colour here.

The marker material is the one animated thing in the scene, pulsing only while the pedal is down.
A full-authority brake is the single event worth taking the driver's eye, and motion does that
where another static shape would not; it rides on the material rather than the geometry, so the
pulse costs nothing per frame on the scene thread.

Depth tint is aerial perspective and it is doing real work, not decoration. Sampling thins with
range whatever the sensors do — ground rings as `r²`, azimuth as `r` — so the far field is always
the sparsest part of the scene, and fading it reads as "too far to tell" rather than as holes in
a surface. It is also **the only thing that separates two surfaces of the same orientation**: a
wall at 12 m and the building behind it at 20 m are 1.77:1 apart, where face shading alone cannot
tell them apart at all. Nothing inside `WORLD_DEPTH_NEAR_M` is tinted, so the band the planner and
AEB work in keeps the full contrast ladder — buying the depth cue by washing out near obstacles
would be a bad trade. The path ribbon is the one surface with no tint: it is guidance, not
percept, and hazing its far end fades out exactly the part that says where the car is going.

**Slab faces are shaded, and a box costs 24 vertices because of it.** A shared corner belongs to
three faces and can hold only one colour. Unlit geometry of a single flat colour has no edges
whatsoever, so abutting slabs merge into one silhouette — the other half of "a wall in front of a
building looks like one blob". The light is horizontal-only in RENDER space with the top and
bottom faces pinned separately (`WORLD_SLAB_LIGHT_DIR`, `WORLD_SLAB_TOP_SHADE`): render space so
the shading does not rotate with the car, and split because one 3D light cannot separate the
three faces a trailing camera sees — from above it bunches all four sides together, laterally it
flattens the top into them. Expect ratios of only 1.1–1.3:1 between faces; a dark obstacle band
under a light air has almost no luminance room, so face shading is a *crease* cue and depth is
the contrast cue.

**Slabs are ORIENTED to the surface they describe, not to the world lattice.** The voxel grid is
world-aligned, so every box used to be world-aligned too: a wall at 30° came out as a staircase of
cubes and a car parked at an angle as a heap of them. `orientation_frames` measures the direction
from the FOOTPRINT — the accumulated store, which is the evidence that has already been expired and
bounded correctly — and `_slab_mesh` then merges and extrudes once per orientation bucket, each in
its own rotated frame. Measured on a 20 m wall: **one to three boxes instead of dozens, drawn within
3.75° of the truth and within 0.04 m of the true line.** Six things carry it:

- **The direction cannot be measured inside one column.** A 0.25 m cell holds less evidence than the
  azimuth stripe spacing the returns arrive with (1.24 m at 20 m), so it comes from a
  `WORLD_ORIENT_CELL_M` neighbourhood — and from a **sliding 3×3** of those, because a fixed tile is
  a world-aligned box whose corner a surface can clip, leaving too few cells to fit anything to.
  That showed up as stray untilted cubes along an otherwise clean wall. Every statistic is a plain
  sum, so widening the window is just adding neighbours' sums: nine `searchsorted` lookups over the
  tile keys, which are far fewer than the cells.
- **The angle is a key field**, exactly like the altitude and height buckets — runs only merge with
  runs that agree on a frame. Unlike those it cannot simply be folded into `layers`, because the
  cells have to be *re-gridded* in the bucket's frame before `merge_cell_runs` can find runs along
  it. Buckets fold into [0, 90) because the grid is square.
- **The merge KEY and the drawn POSITION are deliberately different numbers**, and this is what
  removes the visible defect. Rotating world cell centres scatters them up to half a cell diagonal
  (0.18 m) about the true line, which a 0.25 m bin cannot hold without straddling: a wall at 15° or
  60° split across two rotated rows and came back as a **0.50 m step down its whole length in 9–23
  fragments**. Thickness now comes from the key and the centre from the neighbourhood's measured
  mean, so a surface half a bucket off its frame still fragments — that is unavoidable — but the
  fragments stay coplanar and the seams are invisible.
- **Bucket 0 IS the world-aligned frame**, so the fallback and the old behaviour are one code path
  rather than two that have to be kept in step. For an unoriented cell the "measured mean" is its
  own centre, and the mean of cell centres across a full rectangle is that rectangle's centre, so
  the position rule collapses to the old one exactly.
- **Nothing is oriented unless the footprint supports it** (`WORLD_ORIENT_MIN_CELLS`,
  `WORLD_ORIENT_MIN_ANISOTROPY`). A bush is a blob with no direction to find, and so is the inside
  corner of an L-shaped building, where two walls average to a 45° answer fitting neither. Both fall
  back. Inventing an orientation would be the same class of error as rendering every simulator actor
  unconditionally — a claim the perception did not make.
- **Face shading is per bucket, and it reuses the world-normal path rather than replacing it.**
  Rotating a box turns its face normals the same way, and `n · right` for a turned normal equals the
  untouched normal against a basis turned the other way — so `face_shades` is handed a
  counter-rotated basis. Every box in a bucket shares a frame, so this is one call per bucket, not
  per box.

Finer buckets are **cheaper as well as better**, which is not the obvious direction: a frame that
fits merges into fewer, longer boxes. Measured on a street scene, 12 buckets gave 22 slabs and
31.3 ms against 33 slabs and 33.2 ms at 6. The reason not to keep going is the fixed cost of a
group holding almost nothing, not the geometry.

The ego's contact shadow is faked with stacked translucent discs because **nothing in the scene
can receive a real one**: the ego casts, and both `DirectionalLight`s are configured for it, but
every large surface is a `NoLighting` material and those skip the lighting path entirely — so
`receivesShadows: true` on the road mesh is a no-op.

**There are THREE radii, because structure, road and open ground are not observable to the same
distance.** `WORLD_RADIUS_M` is 150 m and covers structure, traffic and actors: the front
unit reaches 200 m, a wall is a big vertical target, and azimuth spacing (which grows only as
`r`) still puts several returns on a building at 150 m. `WORLD_ROAD_RADIUS_M` is 70 m and covers
the ground, because ground rings go as `r²` — 0.24 m apart at 20 m but ~1.5 m at 50 m and ~6 m at
100 m — so past about 70 m there is no surface to reconstruct, only isolated rings metres apart,
and meshing those gives a corrugated road rather than distance.

`WORLD_SURFACE_RADIUS_M` is 40 m and covers **unpaved** ground. Ring spacing is `(r²/h)·Δθ` =
`5.9e-4·r²` for the roof unit — 0.72 m at 35 m, 1.19 m at 45 m — against the 0.75 m
`WORLD_ROAD_BRIDGE_CELLS` can close, so past roughly 36 m a single frame yields disconnected rings
rather than a surface. **The road reaches further because it is driven ALONG**: accumulation over
`WORLD_CELL_MEMORY_M` sweeps the rings down its length and fills it in, which never happens for the
terrain out to one side. It is also the cost bound — see the scene-build note below.

Each half of the ground carries its own fade radius per cell, averaged to the corners exactly as
height and colour are, so the seam between them is a gradient and each dissolves where it really
ends. Without the fade a surface ends on a hard rim which reads as a cliff — a drawn boundary where
there is only the end of what was measured.

**Surfacing the whole ground is the change most likely to blow the scene budget, and it did once.**
The road is a ribbon; the ground is a disc. A 0.25 m lattice over the full 70 m radius meshed to
110k vertices and took the build to **77 ms against a 40 ms tick**; bounded to 40 m it is 40k
vertices and 37 ms, pinned by `test_covering_the_ground_stays_inside_the_scene_budget`. Area goes
as `r²`, so `WORLD_SURFACE_RADIUS_M` is the constant to move in either direction if SCENE BUILD
starts logging. Note this runs on `SceneWorker`'s thread, so an overrun costs WORLD frames rather
than control latency.

**Corner averaging is a 2×2 BOX SUM over a dense lattice, and needs no sort at all.** Both sorted
shapes were tried first and both lose: `reduceat` over a `(4N, 5)` repeat gathered into sort order
(43.4 ms against 37.4 on a 40k-cell disc — the accumulators win with that idiom because there the
sort is what is being avoided), and `bincount` per channel over `np.unique`d corner keys. The
insight that beats both is that the answer is knowable by *arithmetic*: the cells occupy a dense
rectangle of lattice, so scattering them into it and adding four shifted slices IS the grouping.
Measured on 86k ground cells, **20.8 ms → 8.5**. Three things carry it:

- **The cells must be unique in `(x, y)`**, or a plain scatter keeps whichever wrote last instead of
  averaging. That is exactly the stacked case the layer field exists for — a bridge deck over a road
  — so `_corner_means` falls back to `_keyed_corner_means`, which carries the layer and separates
  them. `np.count_nonzero` on the occupancy plane counts the distinct `(x, y)` for free, since a
  duplicate overwrites rather than accumulates, so the precondition is *checked* rather than assumed.
- **The box filter also HEALS a defect the keyed path has.** Putting the layer in the corner identity
  means two adjacent cells either side of a `_GROUND_LAYER_M` (0.75 m) contour got two coincident
  corners rather than one, so the surface was topologically torn along every contour on any slope —
  measured on a 5% ramp, 123 duplicated corners disagreeing by 15 mm. Small enough never to have been
  noticed, but a seam is precisely what sharing corners exists to make impossible.
- **`float32` end to end.** The vertex buffer is float32 anyway and a world Z of a few thousand
  metres still resolves to well under a tenth of a millimetre, so promoting to float64 in between
  just doubles the traffic: 15.8 ms against 8.5 for the same box filter. The same reasoning runs
  back through `bridge_gaps` (which now preserves float32 rather than promoting) and forward through
  `depth_tint` (4.1 ms → 2.8, worst-case channel error 1.2e-7 against an 8-bit output).

Three smaller measured wins sit alongside it, all output-identical:

- **`_scan_order` replaces `np.lexsort`** in `bridge_gaps` and `merge_cell_runs`. A lexsort is one
  stable argsort per key — three passes — where packing the three fields into a single int64 in the
  same precedence needs one: 2.98 ms → 2.11 on 86k cells.
- **Corner keys are derived by ADDING to the packed cell key** rather than building a `(N, 4, 3)`
  key block and packing it: `pack_cell_keys` puts x in the high bits and y next, so the corner at
  `(dx, dy)` is the cell's key plus two constants. 9.17 ms → 0.83.
- **`_newest` uses `argpartition`, not `argsort`.** Once a store sits at its cap it is culled on
  every tick, so this ran a full sort of the whole store every frame to find the top-K; the order of
  the survivors is never read, because both callers re-sort the indices they select. 5.0 ms → 1.2 on
  90k voxels.

Together these took the steady-state build from **~100 ms to ~60 ms** on an accumulated street drive
(67k-point cloud, 15k road cells, the voxel store at its 90k cap, 88k ground vertices). The
remaining cost is roughly half `_ground_mesh` (bridging 10 ms, the box filter 8.5, the ego-relative
tail 5) and half the voxel store (`_update_boundary_columns` 16 ms, expiry 6).

**That lever has now been taken: the build runs at TWO RATES on one thread.** Everything before the
ego-relative tail is **world-anchored** — it depends on the stores, not on the pose — so
`WorldSceneAssembler.update(refresh_stores=False)` re-presents the cached `WorldMesh`es (world
vertices, untinted linear colour, indices, fade radii) into the current snapshot's ego frame:
`world_to_render` + `depth_tint` + the per-snapshot elements (AEB overlay, path, actors, camera), a
few milliseconds against the ~60. `SceneWorker` refreshes the stores on
`WORLD_STORE_REFRESH_INTERVAL_S` and composes every snapshot in between, so the view tracks the car
at the display rate however slow the store work gets. Three things are load-bearing: a compose
tick's cloud is **not ingested** (the named freshness trade — no different in kind from the
snapshots the one-slot mailbox already dropped); `_track_ego_motion` still runs on every tick so a
**teleport during a compose still clears** — clear() drops the mesh cache, which forces the rebuild;
and SCENE BUILD keeps meaning "the store refresh", so the over-budget warning still watches the
right number. Face shading is baked into the cached colour, so the crease cue re-aims at the store
rate — bounded staleness on a 1.1–1.3:1 cue. Pinned by `test_two_rate_pipeline.py`.

Qt's `offscreen` platform plugin is not QRhi/3D capable, so an offscreen smoke
can validate QML loading, property binding, lifecycle and fallback but cannot
validate 3D pixels. Component compilation *does* happen before rendering, so an
offscreen `QQuickView.setSource` is enough to prove every type and property
resolves — that is how `Repeater3D` and `PrincipledMaterial.alphaMode`
were verified against the installed Qt 6.7.1. Always call
`WorldView.shutdown()` before destroying the QML context; otherwise QML
bindings evaluate once against a null `sceneBridge` during teardown.

**Pixel questions ARE answerable on this machine, and guessing at them has been wrong twice.**
On the normal Windows platform a `QQuickWidget` renders on the real D3D11 backend; `show()`, a
`QTimer.singleShot` of a second or so, then `grabFramebuffer()` (or `QWidget.grab()`) yields a
`QImage` that can be pixel-sampled. That is how the fog no-op, the linear-space vertex-colour
multiply and the >1.0 clamp were settled, and how the whole scene can be reviewed without
launching BeamNG: feed a synthetic `PerceptionSnapshot` through the real `WorldSceneAssembler`
into a real `WorldView`. Any claim about what the renderer does should be measured this way
rather than reasoned about — the `Fog` block sat in the QML for a long time doing nothing, under
a comment saying it might.

### Self-driving

`planner` (pure geometry) → `controller` (state machine) → `worker` actuation, with
`navigation` supplying an optional turn hint. All three are Qt-free and BeamNGpy-free.

The planner is **geometric, not semantic** — a deliberate choice, not an oversight. Drivable
means "no return in the obstacle height band", so flat grass and car parks read as drivable and
the car will explore them; on a kerbed road the 0.20 m mount sees the kerb face and that is what
keeps it on the tarmac. `classify_road_points` is display-only. If this needs tightening, add a
road-coverage *bonus* to the cost function — the semantic mask is already computed for the
display — rather than swapping the input.

Two filters stand between the raw cloud and that height band, and both were added because the
arc scan takes the **nearest** blocking point per arc, so anything spurious ends an arc on its
own:

- `despeckle` drops returns whose 3×3 cell neighbourhood holds fewer than
  `OBSTACLE_MIN_SUPPORT` points. One stray return at 10 m was measured to take a clear road's
  free distance from 33.2 m to 10.0 m, and the speed law brakes below about 25 m — so a single
  speck was a full brake application. Real structures are surfaces and put dozens of returns on
  a kerb face even at the horizon, so a support of 2 rejects only genuinely isolated points.
- `ground_rise` estimates the local ground per range ring (a low percentile, interpolated
  between ring centres) and the `SLOPE_ALLOWANCE_PER_M` cone becomes a **bound** on that
  estimate rather than the estimate itself. The cone alone put the obstacle floor at 0.27 m by
  20 m and 0.50 m by 35 m, so **no kerb was an obstacle beyond about 12 m** — 1.1 s of road-edge
  information at the speed cap. The car could not see a bend coming, ran wide into the outside
  of it and blocked. The clamp is two-sided and that is what makes it safe: never below the
  ego's own plane, so a ditch beside the road cannot drag the floor under the tarmac, and never
  above the cone, so terrain behaves exactly as it did before.

An arc of curvature `k` is the circle of radius `1/|k|` centred at `(-1/k, 0)` in BEV
`(right, forward)`. That one expression lets a whole 41-arc fan be scanned against the whole
obstacle cloud as a single `(obstacles × arcs)` numpy matrix — no per-arc loop. **Positive
curvature is left**, which is the OPPOSITE of BeamNG's steering input.

Candidates are **two-segment**: hold the curvature currently being driven for one of
`TRANSITION_DISTANCES_M` (0/6/12/18 m), then bend to one of the 41 targets. The zero-transition
family is the classic immediate fan (and is what the widget draws as the fan); the deferred
families are scanned by rotating the cloud into segment A's endpoint frame, so the same circle
trick applies. Four things about the cost stack were found the hard way:

- **The fan's spacing IS the lateral resolution**, because endpoint offset goes as `k·L²/2`. The
  candidates are quadratically spaced (`|k| = K_MAX·u²`, dense around straight ahead) after a
  uniform fan put 3.75 m between adjacent offsets at a 30 m lookahead: "correct by a metre" was
  not a candidate, so the planner used a deferred family as a finer-offset workaround every
  tick, the immediate command never corrected, and the car wove kerb to kerb until it blocked.
  `test_driving_loop.py` pins lane-keeping closed-loop against exactly that regression.

- Smoothness scores the *eventual* curvature change identically in every family. Discounting a
  deferred turn makes "later" always cheaper than "now", and under per-tick re-planning later
  never arrives — the car holds straight forever. Deferral must win on geometry (the immediate
  turn collides, the deferred one does not), never on comfort, plus a small `COST_TRANSITION`
  tie-break toward acting now. `test_a_deferred_gap_plan_commits_to_the_turn_in_time` pins the
  convergence.
- The scan's transcendentals are the budget. Swept-angle `arctan2` over the full
  `(4000 × 41) × 5` stack costs ~34 ms — most of the 40 ms tick. It runs sparsely on blocking
  pairs only (free distance stays exact), and the clearance window uses the exact chord
  equivalence `progress <= w  <=>  y > 0 and |P| <= 2R·sin(w/2R)` instead; deferred families
  also scan a half-decimated cloud. Measured 6.7 ms mean (p95 8.8) on a worst-case synthetic
  cloud.
- The winner is refined by parabolic interpolation between fan steps (so steering is continuous,
  not 41-stepped); its free distance takes the minimum with the one neighbour the interpolation
  moved toward, which is rigorous because adjacent collision corridors overlap through the
  horizon.

`ArcPlan.curvature` is the *immediate* command (the current curvature when the winner defers);
`next_curvature`/`transition_distance_m` describe the bend, and the controller brakes **to the
corner-entry speed** `sqrt(a_lat/|k_next| + 2·a_decel·d1)` instead of toward a stop — that is
the whole point of the deferred families. The lookahead passed by the worker scales with speed
(`LOOKAHEAD_TIME_S`, clamped 16–30 m) to keep the keep-right/nav character constant.

**BeamNG steering is positive-RIGHT**, verified in `lua/vehicle/input.lua`: `kbdSteer` sends
`kbdSteerRight - kbdSteerLeft`, so the steer-right binding produces `+1`. BeamNGpy's
`Vehicle.control` docstring claims the reverse ("negative = right") and is **wrong** — trusting it
made the car steer the mirror image of the arc drawn in the BEV. `STEERING_SIGN = -1.0` is the
single place the two conventions are reconciled, and `test_controller.py` pins it in both
directions. Treat the beamngpy docstrings as hints, not as ground truth; the game's Lua is the
authority and is readable on disk.

Cost terms are each normalised into roughly `[0, 1]` so the weights are comparable, and all three
*steering* terms (nav heading, keep-right, smoothness) are expressed in the same units — nav is
scored as the composite path's **heading at the lookahead**, which reduces algebraically to the
old curvature-error form for the immediate family. Two things here are load-bearing and were
found the hard way:

- Free distance is scored against `REQUIRED_FREE_DISTANCE_M` (the braking envelope), **not**
  against the horizon. Scoring against the horizon makes 20 m of clear road "worse" than 35 m,
  and a planner that believes that will never turn.
- The lookahead matters because the curvature needed to reach a lateral target falls as
  `1/L²`. At a short lookahead the polite keep-right nudge becomes a swerve, which the collision
  test then rejects — so keep-right silently stops working. That is why the worker scales it as
  ~2.8 s of travel (16–30 m) instead of fixing it: 20 m was ~3 s at the old 25 km/h cap, and at
  40 km/h a fixed 20 m would quietly halve the time horizon.

Clearance is scored only over `min(free_distance, lookahead)`, the stretch actually driven before
the next re-plan. Scored to the full free distance, every arc is punished for the pinch point
that ends it, and keep-right (which by definition asks the car to sit nearer one edge) can never
win.

The pedals work in **acceleration units**: desired accel `SPEED_KV·error` clipped into
`[-HARD_DECEL, COMFORT_ACCEL]`, throttle and brake mapped through nominal full-pedal
accelerations with a **coast band** between them (a small overspeed is covered by engine drag,
like a human lifting off — also what stops throttle/brake chatter at the boundary). One foot
works both pedals: engaging one instantly zeroes the other, and each moves under its own slew
limits, which is the jerk limiting that makes inputs look human.

Three things about the longitudinal path are load-bearing and were found by tracing a corner:

- **The coast band and the drag credit are different numbers.** `COAST_DECEL_MPS2` decides
  whether to touch the brake at all; `ENGINE_DRAG_MPS2` is what gets subtracted from the demand
  once you do. Crediting the full coast figure assumed the engine was already delivering it, so
  every brake application came out short by whatever drag was really missing — the car entering
  a 35 m corner sat 10 km/h over target on 0.16 of brake, the steering saturated trying to make
  up the difference, and it ran wide until it blocked. Brake engagement is also hysteretic
  (`BRAKE_RELEASE_FRACTION`), or the pedal chatters across the boundary.
- **The speed target is low-passed** (`TARGET_SPEED_TAU_S`), because free distance is a
  nearest-point measurement over a cloud rebuilt every 40 ms and its single-tick dips are noise.
  A collapse past `TARGET_SPEED_BYPASS_MPS` below the current speed skips the filter, so real
  braking is never delayed by it.
- **The immediate curvature gets no entry allowance.** Crediting the distance the steering takes
  to wind on ("hold speed, I am not really in the corner yet") reads as smoothness and is
  circular: the credit keeps the target high, the high target commands no braking, and the
  wind-on completes with the car still at speed — measured at 6.7 m/s² through a 17 m bend
  against a 3.5 limit. It is the same trap as discounting a deferred turn in the planner.
  Anticipation has to come from the path (the deferred families, whose transition distance is a
  geometric fact that shrinks as the car advances), never from the controller. The throttle trim integrator
learns the *vehicle* (mass, drag), so it survives mode changes and only ever winds while the
throttle path is active. Blocked braking is full and slew-free above `HOLD_TAPER_SPEED_MPS`
(an emergency outranks smoothness) and tapers to `BRAKE_HOLD_FRACTION` once stopped.

Steering slews in **curvature, scheduled by speed** (`LAT_JERK_MAX / v²`, capped at
`K_RATE_CEIL` so parking-speed manoeuvring stays as brisk as the old steering-value slew) and is
clamped to `MAX_LATERAL_ACCEL / v²` — a driver cannot turn in harder than grip allows and
neither can this. **`MAX_LATERAL_ACCEL_MPS2` (grip, 6.0) and `CORNERING_ACCEL_MPS2` (comfort,
2.8) are different quantities and collapsing them is what stopped the car getting round a tight
corner at all.** The clamp is the grip figure; the speed law plans corner speeds with the
comfort figure. When both were 3.5 the planned speed needed exactly the ceiling curvature to
make the corner, so the few km/h of steady-state error a proportional speed loop always carries
left the steering saturated with nothing in reserve: the car tracked wider every tick until the
free distance collapsed. The gap between the two IS the tracking authority.

On top of that, an adaptive gain closes the loop the open-loop `k/k_max` map never had: measured
curvature
(wrap-aware filtered yaw rate over speed, from successive `state["dir"]` headings) is compared
against the command and a clamped gain trims slowly — only in DRIVING, above walking pace, at
meaningful curvature, and only when the car turns the way it was asked (a sign mismatch is a
slide or kerb strike, not data). `test_steering_gain_adapts_to_an_understeering_plant` pins the
closed loop against a 0.7-gain plant.

**`_poll_once` computes the plan before the emit and actuates after it.** `Vehicle.control()` is a
blocking ack and the display must not wait on it; `BevFrame.control_ms` therefore reports the
*previous* tick's figure. The self-driving step has its own `try/except` nested inside the
existing one, because a planner bug caught by the outer handler would masquerade as a lost bridge
and tear the connection down after `_POLL_FAILURE_GRACE_S`.

Two safety invariants, both pinned by `tests/test_worker_state.py`:

- **Every teardown path zeroes the controls.** `_cleanup_sensors` is the single funnel
  (`stop_sensors`, `handle_bridge_lost`, `shutdown`, the poll-failure branch, and re-attach) and
  calls `_disengage_aeb` then `_disengage_self_driving` first, while the vehicle handle is still
  live. Controls are *released*, not braked — the human takes over a coasting car. AEB's release
  is sent only if it was the one holding the pedal, or it would stamp on a human's own braking;
  the same one-shot release runs when an AEB event simply ends, and it deliberately bypasses the
  `CONTROL_INTERVAL_MS` gate, because one swallowed message leaves the car braking indefinitely.
- **No returns means blocked, not clear.** An empty cloud is indistinguishable from an open road,
  so `_BLIND_ARC` reports zero free distance and the controller brakes instead of accelerating
  through a map load.

The controller's `_blocked_by_stall` flag is not redundant: a kerbed car sees a clear path
forever, so a stall cleared by "the path looks clear" flip-flops `BLOCKED`→`DRIVING` and never
reaches the reverse recovery.

**Gear indices are gearbox-family dependent, and getting this wrong parks the car.**
`Vehicle.control(gear=…)` reaches `drivetrain.shiftToGear` → `mainController.shiftToGearIndex`.
Automatic-family boxes (`automaticGearbox`, `dctGearbox`, `cvtGearbox`, `cvtGearbox2`,
`electricMotor`) resolve it through
`hShifterModeLookup = {[-1] = "R", [0] = "N", "P", "D", "S", "2", "1", "M1"}` — Lua arrays start
at 1, so **index 1 is PARK and index 2 is Drive**. Manual and sequential boxes have no such
lookup: there index 1 really is first gear. `gear=1` therefore drove manuals and silently parked
every automatic, which presented as "reports `Clear for X m`, applies full throttle, never moves,
then reverses" — reverse worked because `-1` is `"R"` for both. `controller.forward_gear_index`
picks the index from what `electrics.gear` reports: automatics return a mode **string**
(`"P"`/`"D"`/`"S3"`/`"M2"`), manuals a **number**, which is how BeamNG itself discriminates
(`type(gearName) == "string"`, `vehicleController.lua`). An unreadable gearbox defaults to the
automatic index because the failure is asymmetric — index 2 on a manual is second gear and still
crawls away, index 1 on an automatic cannot move at all.

Note `set_shift_mode("realistic_automatic")` does **not** convert a manual gearbox;
`drivetrain.setShifterMode` only picks `"realistic"` over `"arcade"`. The family comes from the
vehicle's jbeam, so it must be detected, never assumed.

Two related actuation rules: **the parking brake is sent explicitly on every message** (beamngpy
drops `None` arguments and `submitInput` no-ops on absent keys, so one you never mention is one
you never release — BeamNG's own vehicle spawner clears it by hand for the same reason), and
**`gear` is sent only when it needs to change**, because `shiftToGearIndex` has side effects and
the display loop would otherwise re-enter the clutch state machine 25 times a second.

### Emergency braking (AEB, forward and reverse)

Two independent toggles (`worker.set_aeb` / `set_rear_aeb`) over **one** implementation: same
corridor scan, same phantom filters, same state machine. The rear system is `EmergencyBraking`
run on a **180-degree-rotated cloud** — `aeb.mirror_points` negates the BEV points and
`aeb.mirrored` swaps front/rear and left/right on the geometry. That is a rotation rather than a
reflection, so handedness survives and the curvature convention holds unchanged, which is what
lets the rear scan reuse every arc helper in `planner`. Path curvature is `yaw / |speed|` in the
frame of travel either way, so the heading is fed in **unflipped**; only the sign of the speed
changes. `AebState.rearward` tells the overlay which way to draw, and `_aeb_to_screen` un-rotates.

The two differ only in the measured **plant**, held in an `aeb.BrakingProfile`: deceleration,
standoff and arming speed. Reversing throws the load onto the rear axle, which carries the
smaller brakes — measured 0.70–0.79 g backwards against 1.02–1.07 forward, so the car stops ~30%
worse and sharing the forward figure would fire far too late. Reverse also arms at 0.8 m/s rather
than 2.0, because backing into a wall at a crawl is exactly what it is for, and stops 0.35 m from
the bumper rather than 0.6 because reverse parking is deliberately close work. Closed-loop
against the measured reverse plant: fires at 3.3 / 5.6 / 9.3 / 20.0 m for 10 / 20 / 30 / 50 km/h,
stopping 0.5 / 0.8 / 1.8 / 3.7 m clear.

**Both arm themselves at attach.** They are protective rather than behavioural — neither steers,
and neither touches a pedal until a collision is otherwise unavoidable — so defaulting them off
would mean the safety net is missing exactly when nobody thought about it. Self-driving stays
opt-in, because that one changes what the car does.

One obstacle band serves both: the height band is a radial cull and the two corridor scans differ
only in direction, so `_poll_once` sizes a single AEB band to whichever system reaches further at
the current speed.

**Each system needs its own tick clock, and `_last_aeb_at` is a dict for that reason.** They both
step inside the same tick, so a single timestamp gave whichever ran second a `dt` of microseconds
(clamped to the 1 ms floor). Two things broke, and only the second was visible: `_seen_for`
advanced 1 ms per tick instead of 40, so the rear brake needed **4.8 s of continuous threat**
before it could fire and never did; and `_observe_yaw` divided a full tick's heading change by
1 ms, inflating the yaw rate ~40x and curling the corridor to full lock the moment the car turned
— which is every reversing manoeuvre. Reported as "reverse AEB says standby and never brakes";
pinned by `test_each_aeb_system_gets_its_own_tick_clock`.

`aeb` is Qt-free and BeamNGpy-free like the planner, and it is deliberately **not** built on the
controller:

- It runs with self-driving **off**, under a human driver, so the path comes from the yaw the car
  is *measured* to be turning at (`state["dir"]` differences, the same wrap-aware filter the
  controller uses, duplicated because there is no controller to ask).
- It touches **the pedals only**. With self-driving off the control message is
  `{throttle: 0, brake: x}` and nothing else — no steering, no gear, no parking brake, because a
  human is holding all three. With self-driving on, `_actuate` zeroes the throttle and takes
  `max(controller_brake, aeb_brake)`, so AEB can never soften the controller and vice versa.
- Obstacles are treated as **static**. A single-frame cloud carries no velocity and nothing
  tracks returns between frames, so there is no honest relative speed; the error is toward
  braking early behind a moving leader, which is the safe direction.
- **The trigger is the LAST POINT TO BRAKE, and the pedal is always full.** `stopping_distance(v)
  = v·AEB_LATENCY_S + v²/(2·AEB_BRAKING_DECEL_MPS2)`; it fires when
  `available <= AEB_TRIGGER_MARGIN · needed`, where `available = threat_m − (front_m +
  AEB_STANDOFF_M)`. The grading lives in the *timing*, not the pedal.

  The earlier design scored `a_req = v²/(2·available)` against a threshold and served a pedal
  proportional to it. That brakes **early and gently**, which is a driver-assist rather than an
  emergency brake: measured, at 50 km/h it fired 22.9 m out on 0.52 of brake and stopped with
  7.2 m to spare. It also silently dropped the latency term, which at 30 km/h is a third of the
  whole stopping distance.
- **Every term in `stopping_distance` is physics; `AEB_TRIGGER_MARGIN` is the only judgement
  call**, and it is the dial to turn if AEB feels early or late. Four stacked conservatisms
  originally left 7.0 m of unnecessary approach at 50 km/h — a 0.25 s latency (which
  double-counted `AEB_CONFIRM_S`; the confirmation window elapses while the threat is still far
  away and adds no delay at fire time), a 1.15 margin, a 1.2 m standoff and a *guessed* 8.0 m/s²
  deceleration. At 0.15 / 1.10 / 0.6 / 10.0 the same stop leaves 2.1 m.
- **The deceleration figures are MEASURED, not assumed**, and they are properties of the car
  rather than of this code. Full-pedal braking distances for the vehicle in use are recorded in
  `config.py` and pinned by `test_the_model_never_under_predicts_the_real_stopping_distance`:
  0.40 m at 11 km/h through 160 m at 200, implying a near-constant 10.0–10.5 m/s² that only
  falls off (to 9.65) at 200 km/h. A least-squares fit of `d = A·v + B·v²` puts `A` at −0.14 s,
  i.e. zero — the measurements carry no reaction time, so `AEB_LATENCY_S` stays a separate term
  instead of being double-counted.

  The constant must sit at or **below** the worst deceleration the car actually achieves, or the
  model under-predicts the real stop and the trigger fires after the last point it would have
  worked. 10.0 keeps 0.2–11 m of slack at every measured speed once the margin is applied.
  **Re-measure for a different or heavier vehicle**: a van, or any wet surface, will not hold
  1.02 g, and there the brake would fire too late.
- Closed-loop against the *measured* braking curve rather than an idealised one, with the
  actuation delay modelled: fires at 5 / 11 / 20 / 33 / 49 / 68 / 117 m for
  20 / 40 / 60 / 80 / 100 / 120 / 160 km/h, and the bumper stops 1.2 / 2.1 / 3.1 / 4.9 / 6.9 /
  8.8 / 14.1 m clear — one continuous full application every time.
- **Release is latched against the gap the event started with**, never on the ratio alone.
  `needed` falls as `v²` while the distance to the obstacle falls only as `v`, so partway through
  a stop the ratio recovers *even though the car is closer than when it fired* — traced at
  50 km/h, the brake let go at 11.6 m and again at 6.4 m, delivering an emergency stop as three
  pulses. It releases when the threat has been gone for `AEB_CONFIRM_S`, when `available` exceeds
  both `AEB_RELEASE_MARGIN · needed` and the fire-time gap, or **the moment the car reaches rest**.
- **There is no post-stop hold.** Reaching a standstill *is* the objective, so once the car is
  stopped the event is over and the pedal goes straight back to the driver — the same rule every
  teardown path in `worker` follows, which hands back a coasting car rather than a braked one. A
  timed `AEB_STOPPED_HOLD_S` existed and was removed. Note what this means and does not: the car is
  **not** held against a gradient afterwards, and it never was — the hold expired regardless.
  `AEB_STOPPED_SPEED_MPS` (0.05) is what "at rest" means and is deliberately far tighter than
  `STALL_SPEED_MPS` (0.3, "is this car moving", for the stall and hold checks): releasing at 0.3
  hands back a car still rolling at over 1 km/h toward the thing it just braked for, and under a
  full pedal that is one tick before it is genuinely stopped, so the looser figure buys nothing.
  `AEB_MIN_ENGAGED_S` is a separate rule and still applies — it bars a single-tick blip, and is not
  a hold.
- **The scan horizon is latched the same way, for the same reason.** It scales with `needed`, so
  braking pulls it in behind the very obstacle being braked for: at 50 km/h the wall sat at
  10.2 m while the horizon had come down to 9.2 m, the threat read as *gone*, and the brake
  released and re-fired from much closer.
- **`AebState.threat_m` is nullable, and that is load-bearing.** A clear corridor reports `None`
  and `a_req = 0`, not "free distance equals the horizon". Conflating the two scored an empty
  road as though an obstacle sat at the horizon, giving `a_req = v²/(2·(horizon − standoff))`.
  Below 40 km/h the horizon grows with `v²` so that lands on a harmless constant (2.0), but past
  the speed where it clamps at `PLANNER_HORIZON_M` the ratio climbs with `v²` and crosses the
  5.0 trigger on its own: **on a flat, empty gridmap the car braked itself from 64 km/h down to
  45, released, accelerated, and did it again.** No obstacle was ever involved. Parametrised over
  speed by `test_an_empty_map_never_brakes_at_any_speed`.
- **AEB has its own horizon and its own sensor.** It scans `standoff + AEB_HORIZON_MARGIN ·
  needed`, capped at `AEB_MAX_HORIZON_M = 150 m` against the planner's 35. Sharing the planner's
  horizon meant that at 100 km/h — which needs 48 m to stop — the brake could not come on until
  13 m *after* the last point it would have worked. A 70 m ceiling then made 125 km/h look
  correct for the wrong reason: it fired *at* the horizon, which happened to land near the right
  distance, rather than because the geometry said so. The planner's 35 m is about where a chosen
  *path* stays trustworthy; AEB only has to answer "is there something solid dead ahead", which
  survives much further out.
- **The front LiDAR is a different instrument from the other three** — 200 m instead of 120, and
  a 50° sweep at a quarter of the sparsity divisor instead of 170°. The narrowing is the part
  that makes long range work, and it is not a cost saving. The shared ray budget puts ~48 azimuth
  samples across 170°, i.e. 3.5° apart, which is **9.3 m between ray columns at 150 m** against a
  2 m-wide car; vertical spacing over the same cloud is 0.118°, thirty times finer, so azimuth is
  the entire bottleneck. Coverage is unaffected: the left and right units sweep 170° each and
  already reach to within 5° of dead ahead, so the front wedge fills the gap between them.
  `SensorMount` carries the per-unit range, FOV and density; `LIDAR_RANGE_M` (the cull) is sized
  for the front unit, and the other three simply stop at their own shorter reach.
- **The obstacle extraction is culled to the furthest band before anything else runs**, and AEB's
  band is built to `EmergencyBraking.horizon_for(speed)` rather than to the full 150 m. Both
  matter: everything upstream of the corridor scan is O(cloud), and on a 110k cloud the naive
  full-radius version cost **24.5 ms of the 40 ms tick**. Measured after: 2.9 ms at 50 km/h,
  4.5 at 70, 9.8 at 100, 15.3 at 125 — the range is only paid for when the speed uses it.
- **Above ~170 km/h it is a mitigation system, not an avoidance one**, which is now a sensor
  limit rather than a tuning choice.

**Five filters stand between flat, empty road and the pedal**, because a false positive here is
a full-authority brake rather than a cost term:

- **Height.** AEB has its own obstacle floor, `AEB_OBSTACLE_MIN_HEIGHT_M = 0.30`, against the
  planner's 0.12. The planner steers around kerbs; AEB brakes for crashes, and nobody wants an
  emergency stop for a kerb they were driving over deliberately.
- **Vertical extent**, and it is the one that makes AEB grade-proof — the height test above cannot
  be, at any value. See below.
- **Porosity**: a bush is see-through and a wall is not. Also below.
- **Support.** The blocking distance is the `AEB_MIN_HITS`-th nearest return in the corridor, not
  the nearest. The corridor scan is a nearest-return measurement, so one speck that survived
  `despeckle` would end it on its own — with a full brake on the end of it.
- **Persistence.** A threat must have been *in the corridor* for `AEB_CONFIRM_S`; a clear tick
  resets it. Note it counts how long the threat has been **seen**, not how long it has been
  urgent — counting urgency would spend the window after the situation was already critical and
  so delay every real brake by `AEB_CONFIRM_S`. The horizon reaches well past the distance the
  trigger fires from, which is what `AEB_HORIZON_MARGIN` buys. Time-based rather than a tick
  count for the same reason `_POLL_FAILURE_GRACE_S` is.

**Full braking is a last resort — enforced by the trigger, not by the pedal.** AEB does nothing
at all until a full-authority stop is the last thing that still works, and then applies all of
it. No slew limits either, the same reason `controller._hold_brake` bypasses them. Grading the
pedal instead was the mistake: it made "firing" cheap and every firing gentle.

No returns cannot **arm** AEB (an empty cloud looks exactly like a clear road, so firing on it
would be a brake application at every map load) and cannot **disarm** it either (mid-event the
last measured distance is held). This is the opposite of the planner's `_BLIND_ARC`, and
deliberately so: the planner is the primary system and must fail safe by stopping, while AEB is
a supplementary layer whose false positives are what get it switched off.

### AEB decides obstacle-ness by SHAPE, and that is what fixed hills and bushes

Two filters answer "what shape is the thing standing here" instead of "how high above an estimated
ground plane is this return". They are the answer to two live complaints — braking on any gradient,
and braking for every roadside bush — and they run on **AEB's band only**: the planner should keep
steering around kerbs and bushes, and `ObstacleBand` defaults both off so its band is untouched.
Both can only ever *remove* candidates, so neither can invent an obstacle; that is what makes them
safe under a full-authority brake, and why the live checklist only has to re-prove that it fires.

**Vertical extent, measured per 0.4 m cell over EVERY return in it.** The height floor cannot be
made grade-proof by tuning, because on a climb the road itself rises through any fixed value:
`planner.ground_rise` is clamped into a 1.5% cone to protect the planner's kerb detection, so at 5%
and 25 m the estimator measured **1.20 m of rise against 0.225 m allowed** and the whole hillside
entered the band as a dense, persistent surface. Extent is immune — 0.08 m of spread on a 20% slope
against metres for a wall — and being a *difference* it is equally immune to brake dive and to
suspension heave, which is the near field the cone could never reach and the entire operating range
of the reverse system. Three things about it are load-bearing:

- **Measured over the whole cell, floor-rejected returns included.** A 0.35 m rock puts 0.05 m above
  the 0.30 m floor; measuring the survivors would call it 0.05 m tall and delete every kerbstone,
  bollard and low post there is.
- **The cell BASE is then the ground reference for the floor *and* the ceiling.** Referencing the
  ceiling to the clamped estimate instead made AEB go blind to real walls on steep hills — at 20%
  and 30 m the surface is 6 m up, so a wall standing on it starts above `OBSTACLE_MAX_HEIGHT_M` and
  was discarded as overhead. Caught by `test_a_wall_on_that_same_hill_is_still_an_obstacle`.
- It also makes AEB independent of `SLOPE_ALLOWANCE_PER_M` in both directions, so the planner's cone
  can be tuned without touching the brake.

**Porosity: the window is the shadow a SOLID twin would cast.** An object of height `a` at range `r`
seen from a sensor at height `h` hides the ground behind it for `r·a/(h − a)`; ground returns inside
that shadow mean the rays got through, which is a bush. At 20 m a 0.6 m bush would have hidden
12.2 m and the ground comes back at 21–32 m; a 0.6 m post genuinely hides it and stays an obstacle.

- **`a ≥ h` makes the shadow infinite and the window EMPTY, so anything as tall as the roof unit can
  never be vetoed.** That is the safety property and it is *derived*, not imposed: no parallax
  between the five mount positions can talk AEB out of braking for a wall, a car or a person. It is
  written as an explicit negative shadow rather than left to the arithmetic, so it cannot depend on
  the cloud happening to contain nothing beyond a wall.
- **It must be the ROOF unit's height.** From the 0.20 m mounts a 0.6 m bush hides the ground behind
  it completely, so from down there a bush and a wall are indistinguishable by construction. No roof
  mount ⇒ height 0 ⇒ nothing is ever vetoed, the conservative direction.
- Decided per **cell**, not per return, and the shadow length comes from the cell's extent — the
  whole object casts it. Using a return's own height up the object would shorten every window and,
  worse, make a wall's lower half testable when the wall as a whole must never be.
- **The window's WIDTH is derived too, and treating it as a constant is what made AEB brake late.**
  Evidence only counts if the candidate would have blocked it, so the azimuth window is the cell's
  own angular width, `OBSTACLE_CELL_M / r`. `AEB_POROSITY_AZIMUTH_DEG` was 2.0 and *was* the window:
  a fixed angle against objects of fixed width, so it outgrew them with range — a 1.8 m car spans a
  2° bin only inside ~52 m and covers one outright only inside ~26 m, past which the road **beside**
  the car, which no part of it ever stood in front of, vetoed it. It is now the grid **resolution**
  (0.5°), and only bins lying entirely inside a candidate's wedge are consulted; none fully inside
  means no evidence, which reads as solid.
- **The (bin, range) key must be clamped to its own bin.** A shadow longer than the key stride ran
  into the *next* bin and counted the road in **front** of the object as proof of seeing through it.
  A car's shadow is over 3× its range, so this fired for every car past ~15 m.
  Together the two defects made AEB blind to a stopped car beyond about 10 m — measured on a
  ring-sampled scene at 15/20/30/40/50/60 m and three lateral offsets, all eighteen blind, which is
  the whole 20–49 m band the trigger fires in at 60–100 km/h. Pinned by
  `test_a_stopped_car_survives_both_shape_tests_at_every_firing_range`, which fails 18/18 against
  the old window while the bush and post cases pass either way — a filter that only ever *removes*
  candidates still has to be proved not to remove the target.
- Two honest limits: a candidate near the scan horizon has nothing beyond it to see, and neither
  does one on a crest. Both fail toward braking.

`planner.geometric_obstacle_sets` builds both floors from **one** ground estimate. Measured on a
60k worst case, the shared ranges + `ground_rise` are 2.6 ms of 3.3 while a second floor's mask
and despeckle add 0.7 — calling the whole function twice took the both-features-on tick from 4.0
to 7.8 ms. `geometric_obstacles` is now a single-floor wrapper over it.

The shape tests are the expensive part now, and both are O(cloud). Measured on a synthetic street
sampled the way the sensors actually lay it down (ground *rings*, not uniform noise): **8.0 ms at a
50 km/h horizon, 9.7 at 100 km/h, 11.4 at 125** for both bands together, against 1.2–1.8 ms for the
planner's band alone. Three things keep it there, and all three matter:

- `_cell_profile` **sorts rather than argsorts** — the height is packed into the low bits of the
  same integer key, so the group minimum and maximum are the first and last of each run and the
  permutation is never needed. That alone was 5.9 ms of a 10.6 ms profile on a 110k cloud.
- Porosity is asked **only of cells shorter than the sensor**, since taller ones are immune anyway.
  In a street scene that skips building the evidence set entirely.
- Keys are integers end to end (range quantised to the centimetre), because numpy sorts integers by
  radix and floats by comparison.

Benchmark on *uniform noise* instead and it is roughly twice that, because one return per cell is
the worst possible case for a per-cell test and nothing like a scene. `test_the_shape_tests_stay_
inside_the_tick` uses ring-sampled ground for that reason.

The BEV overlay draws the corridor from `AebState`'s own curvature and half-width (so it is
provably what was scanned), filled **only as far as the blockage** — filling to the horizon reads
as "all 34 m is dangerous" when the point that matters is the one about to be hit. Violet is used
while armed because it is the one hue nothing else in the view uses; a neutral grey was tried
first and vanished into the road points.

`navigation` reads `core_groundMarkers.routePlanner.path` through
`bng.control.queue_lua_command(chunk, response=True)`. Two gotchas: `techCore` replies with
`tostring(<first return value>)`, so the chunk **must** `return jsonEncode(...)` or a table
arrives as the literal `"table: 0x..."`; and `response=True` is required or only an ACK comes
back. It is polled at `NAV_POLL_INTERVAL_MS`, never per tick — it is a blocking round-trip and
the route only changes when the player sets a destination. Every failure degrades to "no hint".

## Configuration gotchas

`config.BEAMNG_EXE` is a **hardcoded absolute path** to the local BeamNG.tech install. Anyone
with a different install location must edit `config.py` — there is no settings UI or env var.
`MainWindow` disables Launch and shows an ERROR badge when that path doesn't resolve.

Every LiDAR constant in `config.py` was set by measuring against a live simulator, and the
comments record the numbers. Before "cleaning up" any of them, read the comment:

- `SENSOR_HEIGHT_ABOVE_GROUND_M = 0.20` is a **requirement, not a default** — a sensor this low
  grazes the road so kerbs break the height profile and cast an occlusion shadow. Do not raise it
  to buy range; raise `LIDAR_VERTICAL_RESOLUTION` instead.
- `LIDAR_VERTICAL_RESOLUTION = 256` is the far-field lever and is nearly free: measured live, it
  does **not** change the total point count. `LIDAR_DENSITY` sets the ray budget; the channel
  count decides how those rays spread vertically. 128 → 256 took returns beyond 50 m from 535 to
  747 at an unchanged ~49.8k total.
- `LIDAR_DENSITY = 50.0` is a **sparsity divisor** (1 = dense, 100 = sparse), so *lowering* it
  densifies. Count scales as 1/density: 49,708 at 50, 99,736 at 25.
- `LIDAR_HORIZONTAL_FOV_DEG = 170.0`, not 179. Both are accepted — the `[1, 179]` bound is a
  beamngpy docstring and editor clamp, not engine validation — but 179 sits on the depth
  pre-pass's rectilinear `tan()` cliff: 7,389 returns of which only 3,053 unique, against
  7,826 / 5,613 at 170.
- `LIDAR_MAX_DISTANCE_M` (slant, per sensor) and `LIDAR_RANGE_M` (horizontal cull, from the
  reference node) are deliberately different; a sensor sits up to ~2 m off that node, so merging
  them truncates the outer annulus. `test_config.py` pins the invariant.
- **The roof unit's aperture is DERIVED, not configured**, and every optical property now lives
  on `SensorMount` rather than in a global. It is fitted to a ground *annulus* in metres
  (`LIDAR_ROOF_NEAR_M`/`FAR_M`) because that is what has to be right: a fixed 13°/7.5° starts at
  6.0 m on a 1.45 m saloon but 8.5 m on a 2.0 m van, outside the ~7 m the low mounts resolve, so
  the two sets leave a **blind ring** around the car. `test_geometry.py` pins the annulus across
  1.28–2.9 m of vehicle.

  A channel at depression θ meets the ground at `r = h/tanθ`, so consecutive ground *rings* land
  `Δr = (r²/h)·Δθ` apart. That is the whole reason the road looked gappy: at the 0.20 m mounts it
  is 0.50 m at 7 m, **4.11 m at 20 m and 25.7 m at 50 m** against a 0.5 m `WORLD_CELL_SIZE_M`.
  The road arrives as concentric arcs with empty bands between them, and **no ray count closes
  them** — extra rays go to azimuth. Measured with the roof unit: 0.24 m at 20 m, 1.48 m at 50 m,
  so single-frame sub-cell coverage goes from **7.0 m to 29.1 m**.

  Note what the arithmetic actually says, because it is not the obvious thing: with the annulus
  pinned, `Δθ` scales with `h` and `Δr = (r²/h)·Δθ` cancels it — ring spacing is a property of
  the annulus and the channel count and is near enough **independent of mount height** (measured
  0.24 / 0.24 / 0.23 / 0.22 m at 20 m across a 1.28 m sports car to a 2.9 m van). Height is not
  buying the sampling. It buys **occlusion** (shadow behind an occluder of height `a` is
  `d·a/(h−a)`, which blows up as `h` approaches `a`: a 0.15 m kerb 10 m out shadows 30 m of road
  at 0.20 m against 1.2 m at 1.6 m, and the road's own 0.10 m crown at 20 m shadows 20 m against
  1.3 m) and **pitch robustness** (the same annulus at 0.20 m needs a 1.8° aperture, and a road
  car pitches 1–3° under ordinary braking, so the ring set would swing off the ground every time
  it slowed down). It deliberately does *not* see building faces — every ray points below the
  horizon — and it does not out-look a car, since `d·a/(h−a)` is still unbounded when `a ≥ h`.
- `DISPLAY_INTERVAL_MS = 40`, not 33, because `poll_sensors("state")` is a blocking round-trip
  measured at 32.7 ms (p95 35.3) while all four `stream()` calls cost 0.54 ms combined. That poll
  is now prefetched (see the threading section), so the round-trip overlaps the idle gap between
  ticks and the tick's own busy time is the numpy work plus whatever remains of the join. 33 ms is
  therefore plausibly reachable now, but lower it only against a live measurement of the joined
  wait, not from this comment.

The driving constants have their own trap: several of them look like one quantity and are two.
Before merging any pair, read the comment — each split below was made because collapsing them
had a measured failure:

- `MAX_LATERAL_ACCEL_MPS2` (6.0, **grip**, clamps the steering) vs `CORNERING_ACCEL_MPS2`
  (2.8, **comfort**, sets planned corner speeds). The gap is the tracking authority.
- `COAST_DECEL_MPS2` (1.2, the band: *whether* to brake) vs `ENGINE_DRAG_MPS2` (0.3, the
  credit: *how hard* once braking). Over-crediting silently under-brakes.
- `SLOPE_ALLOWANCE_PER_M` is now a **bound** on `planner.ground_rise`, not the estimate.
  Raising it back into a standalone cone re-blinds the car to kerbs past ~12 m.
- `LAT_JERK_MAX_MPS3 = 4.0`, up from 2.5: at the speed cap 2.5 allowed 0.020 /s, so winding on
  the 0.04 curvature of an ordinary 25 m bend took 2.0 s and 22 m — most of the 35 m horizon.

`geometric_obstacle_sets` costs ~4.4 ms of the 40 ms tick for the planner's floor alone and
~5.1 ms for both floors, measured on a 60k-point worst case; AEB's corridor scan adds 0.05 ms.

The WORLD constants have a trap of their own: **several of them are sized by the SENSOR and one
is sized by the RENDERER, and it is no longer the renderer that binds.**

- `WORLD_RADIUS_M` (150) vs `WORLD_ROAD_RADIUS_M` (70) vs `WORLD_SURFACE_RADIUS_M` (40) are three
  different questions — how far STRUCTURE is observable, how far the ROAD is, and how far OPEN
  GROUND is. Collapsing any pair either shreds a surface into rings or throws away the reach. The
  road outreaches the terrain because it is driven along and accumulation fills it in; the terrain
  beside it is never swept that way. The third one is also the scene-build cost bound, and it is
  the constant to move if SCENE BUILD starts logging. See the renderer section.
- `ROAD_CLASSES` (may the car drive here) vs the `WORLD_SURFACE_*` class sets (what is the ground
  made of) are likewise different questions over the same palette, and a class can be in both. Do
  not fold the material sets into the road set to "simplify": road feeds the road store and the
  BEV split, materials feed colour only, and `NATURE` covers both grass and tree canopy.
- `WORLD_CELL_SIZE_M` / `WORLD_COLUMN_SIZE_M` are 0.25, down from 0.5. They were 0.5 because of
  the `ProceduralMesh` Python loop, not because of the data; with the raw-buffer bridge the grid
  follows the sensors instead. 0.25 is what the roof unit supports — ring spacing is 0.24 m at
  20 m — and going finer would resolve less than the returns carry.
- `WORLD_ROAD_BRIDGE_CELLS` (3) and `WORLD_COLUMN_BRIDGE_CELLS` (4) are the same idea on
  different surfaces and both scale with the cell size: halving the cell must double them or the
  same physical gap stops being closed. The column figure went 2 → 4 for exactly that reason.
- `WORLD_COLUMN_VERTICAL_BRIDGE_BINS` (1 bin, 0.25 m) is the *opposite* axis and must stay small.
  It has to exceed the vertical sampling gap on a wall (0.10 m at 50 m) so walls stay solid, and
  stay far under the clear air beneath a canopy or a bridge deck so those still split. Raising it
  to swallow noise would re-merge the tree with the grass under it.
- `WORLD_ORIENT_CELL_M` (1.0) is a STRIDE, not the window: the window is a sliding 3x3 of tiles,
  so it is 3 m. Long enough to hold a clear line through a wall, short enough that a curved one is
  followed rather than averaged into a chord. `WORLD_ORIENT_BUCKETS` (12) is the angular
  resolution — see the slab section for why finer is cheaper — and the two guards
  (`WORLD_ORIENT_MIN_CELLS`, `WORLD_ORIENT_MIN_ANISOTROPY`) are what stop a bush being handed a
  direction it does not have. Relax those and the failure is silent and everywhere: every clump of
  foliage acquires a confident, wrong angle.
- `WORLD_COLLISION_CEILING_M` (2.6) is a question about the VEHICLE, not the scene. Raise it for
  anything tall enough to hit a branch a car passes under.
- `WORLD_CELL_MEMORY_M` (25) and `WORLD_COLUMN_MEMORY_M` (90) are **metres travelled, not
  seconds**, and `WORLD_VEHICLE_TTL_S` (0.15) deliberately still is. Do not "tidy" the three into
  one unit: static geometry and traffic are on different clocks by design, and the second one is
  what lets a car cross in front of a stopped ego and still fade. The two distances are sized to
  reproduce the old windows at cruising speed (1.2 s was 17 m at 50 km/h; 4 s was 56 m), so raising
  them costs store size and scene-build time rather than correctness.
- `WORLD_MAX_ROAD_CELLS` (120k) is the bound the road store never needed while a TTL emptied it,
  and needs now. Its companion is the radius cull in `_expire_road_cells`; the cap alone would let
  a parked car in open terrain accumulate whatever fits inside it.
- `WORLD_DEPTH_*` is **exponential extinction over `WORLD_DEPTH_SCALE_M`, not a ramp to a far
  distance**, and the shape had to change when the view went from a 45 m rim to a 150 m one. A
  ramp has to spend its gradient somewhere: normalised to 150 m, a wall at 12 m and a building at
  20 m came back to **1.17:1** — the exact complaint the tint exists to fix — and no exponent
  rescued it, because lowering it to recover the mid-field flattens it from the other side. At
  scale 34 m a wall at 12 m and one at 20 m are 1.81:1 apart while boundary still holds 1.70:1
  over road at 20 m. The table is worth re-running rather than guessing.
- The camera constants (`WORLD_CAM_*`) exist so the view covers roughly the distance about to be
  travelled rather than a fixed patch of road. Reversing is read from the SIGNED forward speed,
  never from `plan.command.mode`: `plan` is None whenever self-driving is off, which is exactly
  when a human is doing the reversing, so the old check only ever swung the camera round for the
  autonomous reverse recovery. `PerceptionSnapshot.forward_speed_mps` carries the sign because
  `speed_mps` is `norm(vel)` and cannot. **The camera is stateful and damped, and there is exactly
  one framing — see its own section above before adding a second.**
- `WORLD_COLUMN_VERTICAL_BRIDGE_BINS` went 1 → 2 when the reach went to 150 m, and it had to.
  Vertical sampling on a wall is `r·Δθ` at 0.118° — 0.10 m at 50 m but 0.31 m at 150 m — so at one
  0.25 m bin every return past ~110 m became its own sub-minimum run and was dropped, and the far
  field fell apart into floating fragments exactly where the extra reach was meant to buy
  something. 0.5 m is still far under the metres of clear air beneath a canopy, so trees still
  split.

The AEB constants have the same "looks like one quantity, is two" trap as the driving ones:

- `AEB_OBSTACLE_MIN_HEIGHT_M` (0.30) vs `OBSTACLE_MIN_HEIGHT_M` (0.12). Different questions —
  steer around it, or brake for it. Lowering AEB's to match is what puts the road surface into
  its obstacle set under a brake dive.
- `AEB_MIN_VERTICAL_EXTENT_M` (0.25) is not a third floor: it is a question about the OBJECT
  (how tall is it) rather than about a return (how high is it). Raising it toward a bush's height
  to swallow foliage would also delete kerb-height solids; foliage is porosity's job.
- `AEB_POROSITY_GAP_M` keeps an object's own returns out of its own window and
  `AEB_POROSITY_MIN_HITS` means one stray return cannot clear a real obstacle. Neither sets *how
  far* the test looks — that is the shadow length, which is geometry and has no constant. Nor does
  `AEB_POROSITY_AZIMUTH_DEG` set how *wide*: that is the candidate's own wedge, equally derived.
  It is the grid **resolution** (0.5°, two of the front unit's 0.26° columns) and wants to stay fine
  enough that a candidate still covers a whole bin at the ranges the test is for — a 0.4 m cell does
  out to ~46 m. Coarsening it back toward the object's own angular size re-creates the late-braking
  regression above.
- `AEB_TRIGGER_MARGIN` (1.15, *when* to fire) vs `AEB_RELEASE_MARGIN` (2.0, when to let go).
  Both multiply the same `stopping_distance(v)`; equal values chatter on the boundary.
- `AEB_MAX_HORIZON_M` (70, how far AEB looks) vs `PLANNER_HORIZON_M` (35, how far a *path* is
  trustworthy). Sharing the planner's number is what made AEB brake 10 m before a wall at
  100 km/h.
- `AEB_CLEARANCE_MARGIN_M` (0.15) vs the planner's `CLEARANCE_MARGIN_M` (0.35). The planner's
  margin buys comfortable paths; every extra centimetre on AEB's is another kerb that fires.

## Conventions

Every module starts with `from __future__ import annotations`. Modules are single-responsibility
and one-directional — `config` → `models` → {`geometry`, `semantics`, `raster`, `launcher`,
`planner`, `controller`, `navigation`} → `aeb` → {`worker`, `bridge_monitor`} →
{`bev_widget`, `main_window`}; keep the pure/testable layers free
of Qt and BeamNGpy imports (`geometry`, `semantics`, `raster`, `launcher`, `planner`,
`controller`, `navigation` and `aeb` currently import
neither — `bridge_monitor` exists as a separate module precisely so `launcher` can stay
Qt-free; `navigation` takes its Lua runner as an injected callable for the same reason).
`aeb` sits just below `worker` rather than beside `planner` because it depends on
`planner.corridor_blocking_distances`; that is the only sibling edge in the pure layer.
`bev_widget` imports `planner.arc_polyline` and `aeb.predicted_corridor` so the drawn arc and
corridor are provably the ones that were planned and scanned.
Point-cloud
work stays vectorized in numpy — no per-point Python loops in the hot path. Qt styling is a
single stylesheet string in `MainWindow._apply_styles`, with state-driven appearance done
through dynamic properties (`badge.setProperty("state", …)` followed by unpolish/polish) rather
than per-widget stylesheets.
