# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Windows-only PyQt6 desktop app that drives BeamNG.tech 0.37.6 over BeamNGpy: it launches the
simulator with the communication bridge enabled, attaches four semantic LiDAR sensors to the
player vehicle, and renders their merged point cloud as an EGO-fixed bird's-eye view.
Python 3.12 (`py -3.12`), src-layout package `src/beamng_lidar_bev`.

## Commands

```powershell
install_dependencies.bat                          # creates .venv39 and fills it
.venv39\Scripts\python -m pytest                  # pyproject sets pythonpath=["src"]; no PYTHONPATH needed
.venv39\Scripts\python -m pytest tests/test_geometry.py::test_transforms_world_points_to_ego_right_forward_frame
.venv39\Scripts\python -m ruff check src tests    # E, F, I; line-length 88
```

**There is a virtualenv now (`.venv39`, 2026-08-23) and it is per SIMULATOR
VERSION**, because beamngpy's pin and `config.BEAMNG_EXE` move together. Both
failures below were caused by one global site-packages shared with another
project, and both are gone inside it — pytest-qt is simply not installed there,
so `addopts = "-p no:pytest-qt"` is now belt-and-braces for anyone still on the
global interpreter rather than the thing holding the suite up. `run_app.bat` and
`install_dependencies.bat` prefer `.venv39` and fall back to `py -3.12` with a
message. The global interpreter still works and still has both hazards.

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

**2026-08-22: the pin is now `beamngpy==1.36` and `BEAMNG_EXE` points at
0.39.4.0, and the two move TOGETHER** — 1.36 speaks bridge protocol v1.27
(BeamNG 0.39.x) and `hello()` refuses v1.26 (0.38.5), while 1.35.1 refuses
0.39.4 the same way. Run `pip install -r requirements-dev.txt` once to pick the
new pin up (ideally into a venv — the PyQt6 6.11 failure above was two projects
sharing one global site-packages). Two 1.36 behaviours the code now accounts
for: `vehicles.get_current()` SILENTLY drops vehicles whose ids fail
object-name validation (reserved `"vehicle"`, leading digit, `/`, leading `%`)
— `attach_to_player` distinguishes that from "not present" via
`get_current_info` and says which; and the streaming `Camera` sensor's buffers
stay zero-filled forever at `requested_update_time=0.0`, which is why
`CAMERA_UPDATE_TIME_S` must be positive and the worker logs a `Vision check:`
warning if no fresh frame arrives.

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

The **roof and road-scan units have not been live-checked at all** (roof added 2026-07-29;
2026-08-10 the ground work split in two: the roof unit narrowed to a 6–55 m annulus at density
25, and a sixth **road-scan unit** — an 80° forward wedge fitted to a 20–100 m annulus at
density 12.5, 512 channels — took the far road, because an equal-angle aperture starves far
rings quadratically: over 6–100 m, 74% of the roof unit's channels landed inside 20 m). Four
things the offline suite cannot reach.
Whether `density` holds the ray *count* constant as the FOV narrows or scales it with solid angle
is still undocumented and unmeasured — the `Sensor reach:` line prints each unit's own return
count and furthest return, which settles it. Whether the fifth unit (≈ +28% total ray budget at
density 25) costs sim frames. Whether **512 channels over a ~14° aperture** (0.027°/channel,
finer than anything measured on this engine) actually delivers the halved ring spacing the
100 m road radius is built on. And whether the road visibly fills: the prediction is that the
arcs-with-bands look disappears inside ~29 m immediately, before any of the WORLD
accumulation work. **It also puts new returns into the AEB height band from above, so the
phantom-braking checklist below has to be re-run before trusting it.** Self-driving has its own live checklist: the `Drive check:` line
at engage, the slope allowance at the 40 km/h cap on a hilly map (see `SLOPE_ALLOWANCE_PER_M`),
the steering gain settling near a per-vehicle constant while cornering, and a corner approach
that visibly brakes to the bend's entry speed rather than toward a stop.

The **route-following upgrade (2026-08-10) has not been live-checked at all**, and its checklist
is: the `Route check:` line on setting a destination (whether `radius`/`linkCount` actually
arrive on this BeamNG version — the chunk defaults them rather than failing, so only this line
can tell); the car keeping the commanded branch at a junction and holding right-of-centre; a 90°
bend approached at the cap braking EARLY and smoothly to entry speed (the preview's doing, not
the free-distance collapse); a turning junction slowing to ~25 km/h while a straight-through
crossroads does not brake; arrival stopping ~5 m short of the marker and HOLDING without creep
(`Route check: arrived` logs; a new destination resumes); behaviour with no destination set being
indistinguishable from before; the `Memory check:` line staying bounded and free distance staying
steady past an occluding parked car, with nothing braking for the old map after a teleport; a
nose-in pocket escaping along an arc (`Reverse check:` line) with rear AEB still firing on a wall
mid-reverse; and one spot check that the AEB metric never leaves ARMED on the standard phantom
drive with everything engaged — its inputs are untouched by construction (memory never reaches
it), so this is a confirmation, not a checklist re-run.

The **2026-08-11 driving and overlay work has not been live-checked at all**, and its checklist is
below. Everything in it was measured offline against synthetic clouds, which can pin arithmetic
and cannot pin what the simulator actually returns.

- **Hills, and this is the one to do first.** Drive a 3–10% grade with self-driving on. Before,
  the road surface itself entered the planner's obstacle band above ~2% and free distance
  collapsed to `STOP_MARGIN_M`, so the car braked, blocked and reversed on any real incline. It
  should now hold the speed cap up a hill with `Drive:` reporting a free distance near the
  horizon. **The one thing that could make this worse rather than better** is within-cell height
  dispersion on genuinely flat tarmac: the floor is now measured from each 0.4 m cell's own base,
  so if the real cloud carries more than a couple of centimetres of scatter inside one cell the
  planner will start inventing obstacles on the flat. Watch the free distance on a flat empty
  stretch before trusting the hill.
- **Bridges, gantries and tunnel mouths.** Drive under one. The coarse ceiling
  (`OBSTACLE_COARSE_CELL_M`) is what stops a soffit reading as a wall now that the floor is
  cell-referenced, and it is the fix with the least offline evidence behind it because it depends
  on the real azimuth stripe spacing at range. A wall must still stop the car, on a hill as well
  as on the flat.
- **Reversing.** The counter that ends the recovery was unreachable, so confirm that a genuinely
  stuck car now reaches STUCK and hands back instead of cycling for ever — and, much more
  importantly, that ordinary driving reverses far LESS often than it did, because the gradient
  phantom that caused most of it is gone.
- **Going round something.** Park a car in the lane and drive at it, with and without a
  destination set. With a route set it previously would not leave the lane at any distance. On a
  wide road it should now pass; on an ordinary 7 m road it is still expected to stop, which is a
  known open defect in the candidate model, not a regression.
- **Verges, grass and scrub.** Drive close to roadside vegetation. The planner now applies the
  porosity veto, so a see-through bush should be driven past without a flinch while a post, a
  kerb and a wall still block. The thing to watch for is the opposite failure: something solid
  that the sensors happen to see past being ignored. Nothing as tall as the roof unit can be
  vetoed at all, so the risk is confined to objects under ~1.6 m.
- **The overlays.** On a hilly map with self-driving and both AEB systems engaged, confirm the
  AEB rails, the brake-now bar, the threat reticle, the plan ribbon and the dashed route all lie
  ON the road over a climb, a crest and a dip and through a corner's camber; that the reticle
  stands upright and full height on a slope; that the rear corridor drapes on the ground BEHIND
  while reversing up a ramp; and that the route dashes follow the terrain past the end of the
  surfaced ground — that last one is the navgraph-Z fallback, the only part with no offline proof
  that the node Z really is road-surface height on this BeamNG version.
- **AEB is touched in exactly one way** — the coarse ceiling, which only ever removes candidates
  — so its checklist below does not need re-running, but the confirmation that it still fires on a
  wall and a stopped car does.

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

**The spawn must pass stdout and stderr EXPLICITLY, and 0.39 is where that
started mattering** (2026-08-23). `run_app.bat` starts the GUI with `pyw`,
which has no console: `sys.stdout` is None and the process's std handles are
invalid. Inheriting those, the **0.39.4 launcher aborts with 0xC0000409**
(`STATUS_STACK_BUFFER_OVERRUN`) 0.75 s in, leaves a **zero-byte
`beamng-launcher.log`**, and never spawns the engine — 0.38.5 tolerated it,
which is why Launch worked on 15 Aug and stopped after the 0.39 upgrade.
Measured from a windowless parent: inherited → 0xC0000409; `DEVNULL` → boots;
`CREATE_NEW_CONSOLE` → **still** 0xC0000409, because Python hands the parent's
invalid handles down regardless. Giving the child a console is not the fix;
naming the handles is.

It was invisible for the same reason every regression here is: `Popen` returns
a pid, so the launch *looked* successful, and **nothing ever asked whether the
process was still alive**. The window waits `_BRIDGE_WAIT_GRACE_S` (300 s) for
a slow boot with Launch disabled, so a simulator gone in under a second
presented as five silent minutes of "BeamNG.tech is starting". `worker.
_watch_launch` now polls the spawned process once a second until the bridge
opens or it exits, and reports the exit code — the code IS the diagnosis, and
no other signal in the app distinguished a death from a slow boot. A stub
exiting 0 is a hand-off to the engine, not a failure.

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
argument — do not move either call. The collected position is advanced by `vel · age` — and, since
2026-08-23, the HEADING by a measured yaw rate · age (`_observe_state_yaw`): a stale heading is
invisible in RAW BEV (the same state transforms the cloud both ways) but rotates every cloud
WORLD's stores accumulate during a turn, which was the live "the whole world turns with me"
report — to restore the pose-to-cloud alignment a synchronous poll had; a state older than
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

### HYBRID mode: six LiDARs, plus two A-pillar cameras as a VIEW (2026-08-24)

The header carries TWO toggles: the VIEW (WORLD / RAW BEV / CAMERAS, GUI-only)
and the INSTRUMENT SET (LIDAR / HYBRID, worker-owned). **HYBRID is the LiDAR set
plus two colour cameras at the A-pillars, and the cameras reach nothing but the
screen.** They render colour only — `is_render_depth=False`,
`is_render_annotations=False` — so there is nothing to turn into a point even by
accident, and the planner, both AEB bands and parking see exactly the six-unit
cloud they see in LIDAR mode. That is why the driving controls are offered
unconditionally now: there is no second cloud to gate.

This replaced a VISION-only mode that swapped the whole LiDAR set for an
eight-camera rig and unprojected its engine depth into the same cloud
(`unprojection.py`, `derive_camera_rig`, `camera_basis`, the oracle and the
staging/ghosting probes). All of it is deleted; what it measured is in the git
history and in docs/VISION_ROADMAP.md. Three findings from it are worth keeping
in mind because they are about the SENSOR rather than that rung: computed stereo
resolved a 0.11 m kerb at 15 m and nowhere beyond it (matching fails on
low-texture asphalt, so engine depth was the only viable source); a fully covered
simulator window throttles the renderer to ~2 Hz, which `Capture check:` still
warns about; and the camera buffer's fourth byte is NOT opacity, which is why
`vision_view` declares `Format_RGBX8888`.

The CAMERAS view is enabled only by the worker's `sensor_mode_changed(HYBRID)`,
because it draws those images and the LiDAR set has none. Four things are
load-bearing:

- **The worker owns the mode** exactly as it owns the driving toggles:
  `set_sensor_mode` no-ops on a repeat, records the choice when idle, and
  re-attaches THROUGH `attach_to_player` when sensors are live, so the one
  teardown funnel (`_cleanup_sensors`) is the only path between rigs and a
  half-swapped set cannot exist. It refuses any name it does not know, which is
  also what a persisted `"VISION"` setting now lands on.
- **`stream_raw()` returns a memoryview of the LIVE shared buffer** — the
  simulator keeps writing into it — so a frame is copied before anything holds
  it. The copy is made **only when the frame changed**: freshness is a strided
  read of the live buffer against `_CAMERA_DIGEST_BYTES` (64 KB), and a camera
  that has not refreshed re-shows the frame already held rather than dropping
  its tile out of the grid. Digesting the whole buffer instead cost a measured
  **9.8 ms of the 40 ms tick** for the pair, ahead of `_actuate`, which delayed
  every control and AEB command in the mode advertised as having unchanged
  perception.
- **`HYBRID_CAMERA_UPDATE_TIME_S` must be positive**: at 0.0 every streaming
  buffer stays zero-filled while the loop spins — a working rig of black frames,
  measured live. The one-shot `Vision check:` line reports first fresh frames or
  warns after 5 s of silence, naming that trap and the graphics-preset one; an
  all-zero buffer is deliberately NOT counted as fresh, or the first tick latches
  the check over a rig of black tiles and permanently disarms the warning.
- **A camera whose buffer is unusable is RECORDED, not silently skipped.** The
  failure set is what the once-per-episode warning hangs on; without it a
  permanently dead camera vanished from the grid and nothing in the log ever
  mentioned it again.

Failure handling shares the LiDAR path's time-based budget via
`_note_poll_failure`.

### A mount does not land where you ask unless you ask in the SIMULATOR's frame

**`derive_vehicle_geometry` measures every extent from the REFERENCE NODE, and
the simulator does not read a sensor `pos` from there.** Measured live with
`tools/mount_origin_probe.py` on the vivace: a Camera AND a Lidar both asked for
`(0, 0, 0)` land at **(+0.160, +0.362, −0.233)** in the node frame — the body
centre laterally and longitudinally, the ground plane vertically. It is a pure
translation (identical at four probe positions) and stable to 0.1 mm across
repeats. The vivace's own jbeam corroborates it: its reference node is `f2r` at
model x = −0.160, and `vivace_body.jbeam` carries `cameraChase offset x: 0.16`
as BeamNG's own correction for the same thing. The engine side is C++
(`Research.SensorMatrixManager.attachSensor`), so measurement is the only
authority here — there is no Lua to read.

So every mount built from a body FACE was displaced by that whole vector:

| mount | asked for | landed | |
|---|---|---|---|
| right LiDAR | 0.05 m outboard | **0.11 m INSIDE the shell** | self-occluded |
| front LiDAR | 0.05 m ahead of the nose | 0.31 m behind it | inside the bonnet |
| rear LiDAR | 0.05 m behind the tail | 0.41 m behind it | floating |
| a_pillar_left | 0.12 m outboard | 0.28 m outboard | car out of frame |
| a_pillar_right | 0.12 m outboard | **0.04 m INSIDE the shell** | filled with wing |

That last row IS the live "one cam shows more bodywork than the other": measured
from the annotation channel, **6.64% of the right camera's pixels were ego
bodywork against the left camera's 0.65%**.

`derive_vehicle_geometry` therefore takes a `sensor_origin` and expresses the
face-derived mounts in that frame; `worker._measure_sensor_origin` measures it
once per attach with a throwaway 16×16 camera at `(0, 0, 0)` (11 ms: create 3,
read 4, remove 5) and logs it as `Origin check:`. After the fix, measured on the
same car: **2.87% and 2.86%** — the two frames are mirror images. Four things:

- **Z is deliberately NOT corrected.** `pos.z` is already measured from the
  vehicle ground plane, which is the sensor frame's own z origin (the bbox
  bottom measures −0.010 m in it). That is the one axis this project had checked,
  and it was right.
- **The centreline mounts stay at x = 0**, because 0 in the sensor frame IS the
  body centreline. Note what this means for the deleted `derive_camera_rig`: its
  `centre_x` "correction" was reasoning, not measurement, and it moved those
  cameras 0.16 m further OFF centre rather than onto it.
- **The fallback is the old behaviour.** A failed measurement logs a warning and
  returns `(0, 0, 0)`, which is exactly what the app built before — a diagnostic
  must not be able to stop the app working — and it is what every offline test
  uses, so the suite's numbers are unchanged.
- **`Mount check:` now checks all three axes.** It measured the HEIGHT ONLY, and
  that is precisely how this survived: z is the one axis where the two frames
  agree. It reports the residual against `pos + sensor_origin` and warns past
  `_MOUNT_PLACEMENT_TOLERANCE_M`.

**The LiDAR half is NOT live-checked**, and its checklist is: the `Origin check:`
and `Mount check:` lines on attach (every residual should read ±0.00x); that
VISIBLE POINTS and the per-unit `Sensor reach:` counts do not fall — the right
unit coming out of the bodywork and the front unit out of the bonnet should if
anything raise them; and that nothing in the AEB or planner behaviour changes
character, which is expected because LiDAR returns are WORLD-space, so moving a
unit changes what it can SEE (occlusion, parallax) and never biases a measured
range. The camera half is measured, above.

### The cameras do not auto-expose, and that is why they look blown out

**A tech Camera sensor ships with `useManualEV=true, manualEV=0.001`** — a FIXED
linear exposure multiplier — measured live on 0.39.4 by reading the untouched
state of a freshly created camera. The game's own view meanwhile runs its eye
adaptation, which had settled near 2^−12.4 = 0.00019 on the same map, so the
sensor renders roughly **5× brighter** than what the player sees, and being fixed
it cannot adapt to a scene at all. Measured on the A-pillar pair: mean luminance
232 and 241 of 255 with **36% and 64% of pixels hard-clipped at white**.

BeamNG ships the controls and hides them: the four Lua wrappers are COMMENTED
OUT in `lua/ge/extensions/tech/sensors.lua:438-441`, and beamngpy's Camera has no
exposure argument at all. The C++ bindings under them are live, so
`worker._apply_camera_exposure` calls `Research.Camera.clearManualEV` directly
through `queue_lua_command`, once per attach, inside a `pcall` — an undocumented
API that a future version may drop must leave the cameras working, merely bright.
Measured after: **mean 133 and 153 with 0.00% and 0.03% clipped**.

- **The value is a LINEAR multiplier, not stops**, and that is the trap. A sweep
  in stops measures nothing but white: anything at or above ~50 saturates the
  frame outright. Measured, 0.0001 reads mean 73, the shipped 0.001 reads 90 with
  clipping already starting, 0.01 reads 154, 1.0 reads 238.
  `HYBRID_CAMERA_MANUAL_EV` is in those units.
- **Auto is the default** (`HYBRID_CAMERA_AUTO_EXPOSURE`) because it is what makes
  the tiles behave like the stock camera; set it False to pin a fixed exposure if
  the tiles must not change brightness between frames.
- **Exposure is per render view and independent per camera** — measured: adding a
  sky-facing camera moved a level camera's mean by 12 of 255, i.e. drift, not
  coupling. So two cameras looking at different content will still disagree, which
  is what a real camera does. It is also why the centring fix HELPS here: a frame
  with a slab of dark car in it adapts brighter, and that was most of the
  brightness difference between the two tiles.
- `GraphicEVCompensation` is reachable through `bng.settings` and is the WRONG
  tool: it is global (it would darken the player's own window), clamped to ±3
  stops, and biases the main view's tone mapper rather than the sensor.
- The re-test on a future BeamNG is one Lua probe: read
  `Research.Camera.getUseManualEV(id)` and `getManualEV(id)` on a freshly created
  sensor. `tools/camera_exposure_probe.py` does exactly that and then sweeps.

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

**The BEV origin is the REFERENCE NODE, and the node is not the body centre — anything centred
on the origin is centred on the wrong thing.** The extents above are asymmetric on a real
vehicle, and two consumers ignored that: the WORLD ego model was drawn centred on the render
origin (so the car stood beside walls the scene drew correctly — `WorldFrame.ego_centre` now
carries the body centre and the QML ego node binds to it), and **the AEB corridor was centred on
the node's path** (`half_width` either side of x = 0), so it swept a band partly beside the body —
measured on the D-Series backing into a garage doorway it was centred in, one corridor edge
reached the wall and fired the reverse brake at 0.7–0.9 m. `aeb.step` now shifts its scan cloud
by `(right_m − left_m)/2` into the body-centred frame, reports the shift as
`AebState.lateral_offset_m`, and both overlays (`_aeb_to_screen`, `_aeb_bev_to_render`) apply it
before the rear un-rotation — so the drawn corridor stays provably the scanned one. The mirrored
geometry swaps left/right, which is exactly the sign the 180°-rotated cloud needs; the `Vehicle
check:` line logs the offset at attach. The PLANNER's corridor scans still reason about the node's
path — its 0.35 m margin absorbs typical offsets, and re-centring it changes steering, which
wants live validation first. `test_body_centring.py` pins all of it, in both directions.

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

**Road markings are the case where a class sits in BOTH vocabularies on purpose.**
`SOLID_LINE`/`DASHED_LINE`/`ZEBRA_CROSSING` are in `ROAD_CLASSES` (paint is drivable tarmac, and
the road store is what draws the surface it lies on) AND in `MARKING_CLASSES` →
`SURFACE_MARKING`, which is listed LAST in the material table because last match wins the colour.
**`DRIVING_INSTRUCTIONS` and `SPEED_BUMP` are deliberately road-but-not-paint**: markings are
decal QUADS and the annotation labels the whole quad, transparent texels included — a line's quad
is barely wider than its paint, but junction furniture ships as big rectangular decals whose full
footprint came back as marking, and entire roundabout approaches rendered as one sheet of paint.
The `Marking check:` line prints per-class counts (excluded classes included) as the evidence for
ever revisiting that.

**The visual-paint experiment, RETIRED with a measured verdict** (2026-08-10): the hope was to
read paint by BRIGHTNESS with `is_annotated=False`, the way reflectance lidars do. Probed live on
a marked road: the unannotated colour channel is BeamNG's own **range-coded rainbow
visualization** (corr(range, R) −0.90, corr(range, B) +0.88; the rendered probe image is
concentric bands with no trace of the lane lines the road visibly had) — no albedo, no
intensity. **Paint cannot be read from this sensor except through annotation, on this engine
version.** The flag and the one-shot `Colour check:` probe (plus the `road_colour_probe.npz`
dump) remain so the conclusion is re-testable on future BeamNG versions; do not re-run the
experiment without them, the summary statistics alone were ambiguous. The remaining alternatives
if annotation quality ever becomes intolerable: a camera sensor with classic bright-pixel lane
extraction (high fidelity, real cost), or blob-filtering the annotation (demote marking regions
wider than paint can be). The marking colour (`#857b49`, a muted paint-yellow) was solved like the
rest — top of the road's luminance rung, ΔE ≥ 17 from every other surface. The whole feature
hangs on one undocumented engine fact: whether the LiDAR's annotation pass labels road *decals*
with the marking class or with the `STREET` beneath them — there is no intensity channel to fall
back on, so `worker._watch_for_markings` logs a one-shot `Marking check:` line either way, and a
zero on a marked road means paint is simply invisible to this sensor. (Confirmed live 2026-08-10:
decals DO annotate through the LiDAR; paint draws.)

Three things were then fixed from the first live look, all pinned in `test_world_scene.py`:
**paint is near-white (`#c6c8c1`) and the SECOND deliberate ladder break** after the path ribbon —
paint is a graphic ON the road, its contrast partner is the tarmac (3.4:1), not the air, and the
in-band paint-yellow read as a stain; **marking cells are re-drawn as crisp full-colour quads 2 cm
above the blended surface** (`_append_paint_quads`), because shared-corner averaging renders a
one-cell line as a soft two-cell tent — the quads are observed cells only, deliberately unbridged,
since a gap in paint (a dash gap) is data in a way a gap in tarmac is not; and **the road store's
material takes the group MAXIMUM rather than newest-wins**, because a 0.25 m cell over a 0.12 m
line holds street returns beside the paint and flickered — marking is the highest code and the
road store only ever holds paved or marking, so the maximum means "paint was ever seen here".

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

### A car is FITTED, not accumulated, and that is forced by the standstill (2026-08-12)

Feeding `SCENE_VEHICLE` to the voxel store made a car visible. It did not make it look like a car,
and the reason is arithmetic. Everything else in the store is forgotten by the METRE, so a stopped
ego keeps every look it ever got and a wall fills in; traffic is on the wall clock at
`WORLD_VEHICLE_TTL_S` (0.15 s) because it MOVES, and `refresh` ingests one snapshot per
`WORLD_STORE_REFRESH_INTERVAL_S` — so the traffic store holds **one snapshot, sometimes two**.
One snapshot of a car at 15 m is four or five azimuth stripes over a metre apart, and
`_column_runs` meshes exactly what it is handed. Reported from the app as a parked car being
"barely shown ... while we're both at standstill", and standstill is the worst case rather than
an unlucky one: **a stationary ego re-samples the same rays, so waiting adds nothing.** Two more
thinnings stack on top — `_limit_pair` decimates traffic and the whole city's boundary returns on
one stride against `WORLD_MAX_BOUNDARY_POINTS`, and the slab mesher wants dense evidence.

`vehicle_fit.fit_vehicle_boxes` (config + numpy; Qt-free and BeamNGpy-free, beside `parking` in
the pure layer) fits an oriented box instead, and `_fitted_actors` renders it through the actor
delegate that already existed. **Accumulation is how you build a surface whose shape you do not
know; a car is an object whose shape you do.** Five stripes cannot mesh a surface and are ample to
fit a footprint. It runs in `compose`, per snapshot, so it also has none of the store's refresh
lag — which is the same defect seen from the other side, since only a MOVING thing reveals that
the slab mesh is a cadence tick plus a build old.

**It is ADDITIVE, and that is the safety argument.** Nothing is removed from the voxel store; a
cluster the fitter declines to claim still draws as solids exactly as before. So the failure mode
is the old picture, never a car that vanished because a fit was wrong — and `refresh` needs no
knowledge of it, which is what keeps the two-rate confinement contract untouched.

Six things were measured wrong first, each fixed by a rendered scene or a test:

- **Clustering is in the SENSOR's lattice (azimuth × range), not in world XY.** Stripes spread as
  `r`, so no world-space distance threshold works: wide enough to hold one car together at 30 m
  welds two parked cars together at 8 m. In polar coordinates the spacing is one cell everywhere.
  The range link is deliberately **three times** the azimuth link — a flank seen obliquely is
  crossed by very few stripes and each lands much further down its length, measured at five range
  cells between one car's own end face and its flank, so a symmetric link split every car in two.
- **PCA is the wrong fit and MINIMUM-AREA is degenerate on exactly this shape.** A car seen from
  behind and to one side is an L; an L's principal axis runs diagonally across it. The textbook fix
  is the minimum-area rectangle — and the convex hull of a clean L is a **triangle**, for which
  every rectangle on a hull edge has area `2 × area`, identical. Measured on an L with 1.2 m and
  2.7 m arms it tied and `argmin` took 24°, drawing a car 24° off the kerb. The **closeness**
  criterion (sum of `1/distance-to-nearest-edge`) has no tie: both real cases put every return ON
  an edge, and the diagonal frame puts them through the middle.
- **Stripe sampling under-measures every horizontal extent**, and the correction must be MEASURED
  from the cluster rather than assumed from the range. A 1.9 m car at 12.75 m spans 0.87 m raw —
  thin enough to be rejected as not a vehicle. But applying the nominal spacing unconditionally
  over-reads anything densely sampled, and a 1.7 m face came back 2.6 m wide. The largest hole
  between consecutive azimuths IS the sampling interval when the cluster is striped and is near
  zero when it is dense, so it self-calibrates; capped at the nominal spacing so an occlusion hole
  inside one object cannot inflate it. The short span is corrected only when it already spans a
  stripe, or the correction would MANUFACTURE depth and report an assumed length as a measured one.
- **The only question is "was a FLANK seen".** A length is observable exactly when something longer
  than any vehicle is wide came back. Believing a short measurement instead drew a **2.4 × 2.4 m
  car**: a parked car alongside the ego shows a badly foreshortened flank, so its L has two short
  arms and taking them at face value asserts a square vehicle.
- **In a corner observation, which arm is the width is settled by GEOMETRY, not by which is
  longer.** Both arms are under a flank's length by the test above, so length says nothing —
  and assuming the longer one is the length drew a car 30 m ahead **lying across the road**. The
  face turned toward the ego is the one across the line of sight; the other arm is the
  foreshortened stub of a flank. One rule covers the bare end face and the corner, and it agrees
  with the end face exactly.
- **The length-to-width ratio of road vehicles is usable evidence about the dimension the returns
  resolve worst.** Small errors in the frame land almost entirely on the short axis, and the
  parked cars came back 2.4 and 2.8 m wide against a true 1.85. `VEHICLE_MIN_ASPECT` only ever
  narrows an over-read; a van at 5.1 × 2.1 and an artic at 9 × 2.6 are untouched.

Three smaller rules: every inference goes **behind** the evidence (the box is pushed away from the
ego by exactly what was added, so the drawn near face stays on measured returns);
`confidence` carries how much was resolved and rides the delegate's opacity, so an assumed
dimension is visibly less certain — but **not too faint**, because a parked car almost always has
one, so `VEHICLE_FIT_ONE_FACE_CONFIDENCE` is the normal case and at 0.72 the near car showed the
far one through itself and read as a rendering fault; and a cluster too long to be a vehicle is
split at its largest hole **only when both halves fit whole vehicles**, because at range one car's
own end face and flank are separated by exactly the gap two parked cars leave between them.

Ids are carried frame to frame by nearest match (`VEHICLE_FIT_TRACK_MATCH_M`) purely so the
delegate's damping works: `ActorListModel.set_actors` avoids a model reset exactly when the id
tuple is unchanged, and a reset rebuilds the delegate and discards its animation state.

The QML delegate gained wheels, a shoulder and a set-back greenhouse — four extra primitives, no
mesh assets, still generic. Its node now stands at the actor's **ground contact** and the model is
built upward; the reference-node correction that used to be a bare `y: model.y - 0.45` moved into
`_render_actor` as `WORLD_ACTOR_GROUND_DROP_M`, because a fitted box reports a true base and must
not get it.

**Verified by rendering, not by reasoning**, per the pixel-questions rule below: a synthetic
street (two cars parked at a kerb, one ahead, sampled with BOTH instruments' real azimuth spacing
— 0.062 rad for the 170° units and 0.26° for the front wedge) fed through the real assembler into
a real `WorldView`, grabbed and looked at. Before: thin blue slivers. After: three recognisable
cars, all within 0.15 m of their true centres and pointing along the road.

**Not live-checked**, and its checklist is: that a real car reads as a car at a standstill, which
is the reported case; that a car crossing in front is drawn along its own direction of travel and
not across it; that a row of parked cars comes back as separate cars rather than one long box (the
known limit — two cars closer together than the stripe spacing at their range are not separable,
and the split rule is what catches the rest); that nothing which is NOT a vehicle acquires a box,
which would show as a car-shaped model over a hedge or a bin; that the box does not visibly lag a
moving car, since it is composed per snapshot while the solids under it are not; and that a
distant car with one stripe on it still shows as solids rather than disappearing, which is the
additive property doing its job.

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

### The ground is ONE height per cell, CONNECTED to the car (2026-08-23)

The promotion rule hands the ground mesh the lowest short run of every
0.125 m column, and four columns land in one 0.25 m cell — so where one
column saw the ground under a parked car and its neighbour saw only the
roof, the cell held BOTH as stacked `(x, y, layer)` layers. Measured on the
first live camera capture: **155k ground cells over 65k distinct (x, y)**,
the stacks a median 3.3 m apart — car tops, wall tops, hedge tops, building
roofs — every one drawn as a floating patch of floor, and every refresh
pushed onto `_keyed_corner_means`, the slow fallback the box-sum fast path
exists to avoid (58 ms of a 280 ms refresh). The camera rig, which sees the
top of everything from 1.3 m, tripled what the LiDAR produced (31k stacked);
the defect itself predates vision mode and the fix applies to both rigs.

Two rules now, in `_ground_cells` + `connected_ground` (scipy.ndimage.label
— scipy ships as beamngpy's own dependency):

- **Per (x, y) the candidate NEAREST THE EGO'S OWN GROUND PLANE wins**, not
  the lowest — a bridge deck the car is driving on beats the road seen
  beneath it. The layer field is then zeroed: left in place it stopped
  `bridge_gaps` joining cells straddling a 0.75 m contour, and the x and y
  passes each invented a fill at the same (x, y) in different layers, which
  was a stack again.
- **Only ground REACHABLE from the car by gentle steps is ground**, decided
  AFTER bridging (before it, ring-sampled ground is rows with holes and
  nothing connects): a raster pass cuts edges steeper than
  `WORLD_GROUND_STEP_M` (0.5 m) between 4-neighbours, labels what remains,
  and keeps components holding a seed near the car; cut rims rejoin where
  they adjoin a kept cell gently (the kerb line), and fragments past the
  bridge's reach rejoin in `WORLD_GROUND_REACH_HOPS` bounded distance-
  transform hops with a slope allowance — a level road ring 2 m past the
  last kept cell returns, a roof 2 m up beside it never does. No seed at all
  (the car over a hole in the store) filters nothing. Measured on the live
  capture: what is dropped is almost entirely ≥1 m above the ego plane —
  the street around a sunken car park — with 46 near-level road cells lost
  inside 20 m. The known cost is the slab ceiling's, in the same direction:
  terrain behind a genuine cliff is not drawn until a gentler way onto it
  is seen. Pinned by four tests in `test_world_scene.py` ("a roof is not
  ground but a kerbed pavement is", the hillside, the no-seed fallback, the
  readmitted fragment).

With the stacks gone the ground half of the live refresh went 105 ms → 24
(`_corner_means` always takes the box path), and `merge_cell_runs` lost its
last Python loop the same day: the cross-row merge is a `lexsort` chain now —
runs sorted by `(layer, x0, x1, y)` merge exactly where y stays consecutive —
measured 64 ms → 12 on the same scene, and every one of those Python
iterations used to hold the GIL. Live capture, both rigs, steady state at
full stores: **~280 ms per refresh → ~170** (the budget is the 120 ms
cadence; the duty stretch above absorbs the remainder on worst-case scenes).

### The blocks are finer than the ground, and that asymmetry is measured (2026-08-12)

`WORLD_COLUMN_SIZE_M` is **0.125 m** against `WORLD_CELL_SIZE_M`'s 0.25, where the two were equal
before. The complaint was that the view looks blocky, and the two halves answer it very
differently.

**Only the slabs are blocky.** `_ground_mesh` shares lattice corners, so the ground is one
continuous surface with no steps in it by construction — refining it cannot remove a defect it
does not have. A slab's drawn thickness, on the other hand, tracks the cell size *exactly*:
measured 0.50 / 0.25 / 0.15 / 0.12 m of drawn wall at those cell sizes, so a kerb was a full
quarter-metre thick and so was every wall.

**The slabs are also the cheaper half and the better-sampled one.** Measured on an accumulated
street drive (99.5k-point cloud, 70 m of travel, the stores at steady state) against the 120 ms
`WORLD_STORE_REFRESH_INTERVAL_S` cadence:

| ground | slabs | build | |
|---|---|---|---|
| 0.25 | 0.25 | 44 ms | what this replaces |
| 0.25 | **0.125** | **73 ms** | **this** |
| 0.125 | 0.25 | 129 ms | |
| 0.125 | 0.125 | 182 ms | over the cadence |

The ground is ~2.5× the cost because it is a **disc** and area goes as `r²`, where a facade is a
surface. It is also where refining buys least, because the ground's radial sampling is what
collapses with range. Fraction of carriageway a return actually hits:

| band | 0.25 | 0.125 |
|---|---|---|
| 0–15 m | 99% | 92% |
| 15–30 m | 64% | 56% |
| 30–50 m | 46% | 31% |
| 50–80 m | 27% | 11% |
| 80–100 m | 8% | 3% |

Inside 15 m the sampling genuinely supports a finer ground cell; past 30 m halving it nearly
halves the hit rate, because the same returns are divided among 4× the cells, and cells invented
by `bridge_gaps` rather than observed rise from **5.7% to 13.9%** of the surface. A wall has no
such problem: it is sampled thirty times finer vertically than in azimuth (`r·Δθ` 0.04 m against
`r·Δazimuth` 1.24 m at 20 m), so that detail was there and was being quantised away.

Three companions move with it, and two of them are the ways to make the change a **regression**:

- **`WORLD_MAX_COLUMNS` 90k → 200k.** The store fills with ~4× the voxels for the same geometry
  (measured 142k against 58k), and at the old cap the cull binds and drops the excess
  **oldest-first** — which is precisely the accumulated stripe sweep that fills a striped facade
  in, so walls behind the car would start dissolving. It would have discarded 37% of evidence it
  had already paid to collect.
- **`WORLD_COLUMN_BRIDGE_CELLS` 6 → 12**, so the bridge still spans the same physical 1.5 m.
- **`WORLD_COLUMN_HEIGHT_M` stays at 0.25**, now deliberately coarser than the horizontal size.
  Halving it too was measured at 87 ms against 86 and 164k voxels against 142k — it buys nothing
  visible and costs a fifth of the store.

`WORLD_ORIENT_MIN_CELLS` deliberately does **not** move; see its own entry under the WORLD
constants, where the reasoning that it should is worked through and refuted with a measurement.

The slab half of the build had no budget guard at all — the existing
`test_covering_the_ground_stays_inside_the_scene_budget` uses a disc of ground returns and barely
touches the voxel store — so `test_the_slab_half_of_the_build_stays_inside_the_scene_budget` is
its companion, and it fails against the old cap rather than merely passing against the new one.

**Not live-checked.** The measurements are offline against synthetic clouds sampled from the
documented ring and stripe geometry, which pins arithmetic and cost but cannot pin what the
simulator returns. Its checklist: that walls, kerbs and parked cars read visibly crisper rather
than merely thinner (a 0.125 m slab has less visual mass, and the obstacle band has very little
luminance room — face shading is a 1.1–1.3:1 crease cue); that SCENE BUILD does not start
logging over budget on a dense city map, which is the scene most likely to exceed the numbers
above; and that walls BEHIND the car still fill in over a drive rather than dissolving, which is
the `WORLD_MAX_COLUMNS` failure and the one to watch for first.

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
  azimuth stripe spacing the returns arrive with (a metre and more at range), so it comes from a
  `WORLD_ORIENT_CELL_M` neighbourhood — a **sliding 7×7** of those tiles, because a fixed tile is
  a world-aligned box whose corner a surface can clip, and because a small window starves on
  exactly the sparse sampling walls at range get: with a 3 m window a 30° wall sampled every
  0.8 m drew as **38 world-aligned boxes wandering 0.28 m off its line** — the live "staircases
  for straight walls" complaint — where 7 m draws it as one box dead on it. The curve cost is the
  chord sagitta 49/(8R) (0.10 m at R = 60 m); tighter curves fail the anisotropy guard and stay
  world-aligned as before. Every statistic is a plain sum, so widening the window is just adding
  neighbours' sums over the tile keys, which are far fewer than the cells. Pinned by
  `test_a_sparsely_sampled_wall_still_lies_along_its_line`.
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
distance.** `WORLD_RADIUS_M` is **190 m** — the merged cloud's own cull, `LIDAR_RANGE_M`
(2026-08-24) — and covers structure, traffic and actors: the front unit reaches 200 m, a wall is
a big vertical target, and azimuth spacing (which grows only as `r`) still puts several returns
on a building out there. It was 150 m, which threw away the outer 40 m of the only unit that
reaches that far; nothing beyond the cull exists downstream, so matching it is the ceiling. Only
the front wedge sees past `LIDAR_MAX_DISTANCE_M` (120 m) at all, so the gain is forward
structure specifically, and the QML camera's `clipFar` (420) must keep clearing this plus the
chase distance. `WORLD_ROAD_RADIUS_M` is 100 m and covers
the ground, and since 2026-08-10 the ground is TWO instruments: the roof unit owns the near bowl
and the terrain (6–55 m annulus, all around), and the **road-scan unit** owns the far road (20–
100 m annulus through an 80° forward wedge — rings 0.20 m at 50 m and 0.78 m at 100, single-frame
sub-bridge to the full radius, with `WORLD_ROAD_BRIDGE_CELLS` at 6 sized to its ~1.55 m azimuth
stripes at 100 m). The split exists because an equal-angle aperture spends channels quadratically
close-in — one 6–100 m annulus put 74% of its 512 channels inside 20 m and ~33 across 50–100 m,
which is why the far road stayed thin no matter the channel count. Behind and beside, the road
still fills by accumulation while driving.

`WORLD_SURFACE_RADIUS_M` is 40 m and covers **unpaved** ground. **The road reaches further because
it is driven ALONG**: accumulation over `WORLD_CELL_MEMORY_M` sweeps the rings down its length and
fills it in, which never happens for the terrain out to one side. Since the 512-channel unit the
sampling would carry ~58 m single-frame, so the binding constraint here is now the scene-build
COST (area goes as `r²`), not the sampling — see the scene-build note below.

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

**That lever has now been taken twice: the build runs at TWO RATES on TWO THREADS.** Everything
before the ego-relative tail is **world-anchored** — it depends on the stores, not on the pose —
so the assembler is split into `refresh(snapshot)` (odometer, stores, and the cached
`WorldMesh`es: world vertices, untinted linear colour, indices, fade radii) and
`compose(snapshot)` (`world_to_render` + `depth_tint` over the cache plus the per-snapshot
elements — AEB overlay, path, actors, camera — a few milliseconds against the ~60).
`SceneWorker` composes every snapshot on its own thread and runs `refresh` on a one-thread pool
at `WORLD_STORE_REFRESH_INTERVAL_S`, so **composes never wait on a refresh** — run inline, the
refresh stalled the view for its whole duration on every cadence tick, a rhythmic hitch. Safety
is **confinement, not locking**: the stores and odometer belong to the refresh thread
(single-flight), the view state (actor tracks, camera pose) to the compose thread, and the only
shared value is the immutable `_MeshCache`, committed in one attribute assignment. Five things
are load-bearing: a compose tick's cloud is **not ingested** (the named freshness trade — no
different in kind from the snapshots the one-slot mailbox already dropped); the **teleport guard
is split** — `compose` drops a cache anchored further than `WORLD_POSE_JUMP_RESET_M` from the
current pose (presentational, so the old map is never drawn in the new one) while the refresh's
own `_track_ego_motion` clears the stores, and it clears **stores only**, because full `clear()`
would touch compose-thread state from the refresh thread; the **first build after construction or
clear runs inline** so the first frame carries a world; SCENE BUILD still means "the store
refresh", but its budget is now the **cadence** (120 ms), not the display tick — both the
over-budget log and `test_covering_the_ground_stays_inside_the_scene_budget` measure against it,
and since 2026-08-23 a build that overruns the cadence stretches its own interval to
`last_build / WORLD_STORE_REFRESH_DUTY` so the refresh thread never runs back-to-back — it
shares the process's GIL with the worker tick and the compose thread, and saturating it taxed
both;
and a refresh error clears from the compose thread only after the future is reaped, when nothing
is in flight. Face shading is baked into the cached colour, so the crease cue re-aims at the
refresh rate — bounded staleness on a 1.1–1.3:1 cue. Pinned by `test_two_rate_pipeline.py` and
`test_scene_worker.py` (including a hung-refresh test proving composes keep flowing).

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

### The planner's band is CELL-REFERENCED, and that is most of the hill fix (2026-08-11)

**The slope cone bounds the ground estimate at 1.5%/m, so on any road steeper than 1.5% the road
surface itself climbed into the planner's obstacle band.** `SLOPE_ALLOWANCE_PER_M` is 0.015, and
`geometric_obstacle_sets` clamps the measured `ground_rise` into that cone — so the floor is only
ever allowed to follow a 1.5% grade, and everything steeper reads as a wall standing on the road.
Measured on ring-sampled ground, the planner's own band, free distance on an empty graded road:

| grade | obstacle returns | free distance |
|---|---|---|
| 0% | 0 | 35.0 m (clear) |
| **2%** | 4000 (the cap) | **6.0 m** |
| **3%** | 4000 | **4.0 m — which IS `STOP_MARGIN_M`** |
| 5–30% | 4000 | 2.7 m |

At 3% the controller blocks, holds, and reverses. That is "it brakes for hills/inclines" AND a
large share of "it constantly reverses", from one clamp — and `planning_map` then remembers the
phantom, whose odometer cannot advance while the car sits braked in front of it.

The planner band now sets `ObstacleBand.cell_referenced`, the same machinery that already made
AEB grade-proof: heights are measured from each `OBSTACLE_CELL_M` cell's OWN base. The same
scenes then produce **zero** obstacle returns to a 30% grade, and kerb detection *improves*
(1104 kerb returns against 659 at 5%, 1880 against 563 at 12%) because the cone had begun
swallowing the kerb along with the hill. `cell_referenced` is a separate field from
`min_vertical_extent_m` because they answer to different consumers — the planner wants the
grade-proofing and NOT the extent filter, since it should still steer around a kerb and a bush.
Note the extent test is provably **inert** at any threshold at or below `min_height_m`: a return
that high above its own cell base puts at least that much spread in the cell.

**The ceiling needs a COARSER reference than the floor, and finding that out cost a regression.**
Referencing the ceiling to the fine cell assumes the cell contains the ground under it, and at
range it does not — the ground units' azimuth stripes are 0.8–1.2 m apart at 19 m against a 0.4 m
cell. A cell can therefore hold a dense overhead surface and no floor at all, its base becomes the
soffit, and a bridge reads as a 0.6 m wall across the road: measured, a soffit at 2.4–3.0 m with
its face at 19 m took free distance from 35.0 m to **19.0 m**. `_coarse_base` reduces the fine
profile to the lowest return in each `OBSTACLE_COARSE_CELL_M` (2.0 m) neighbourhood and the
ceiling is measured from that. All four properties then hold at once, on flat ground and on a
grade: drives under bridges, blocks for walls, ignores empty hills, keeps kerbs. **AEB gets this
too** (its band is cell-referenced), which fixes the same latent false positive there — it only
ever REMOVES candidates, so it cannot make AEB miss anything.

### The planner is bounded by CELLS, not by points (2026-08-11)

`PLANNER_MAX_OBSTACLE_POINTS` bounded a quantity the cloud does not measure in. A realistic
street puts tens of thousands of band returns into a couple of thousand distinct
`OBSTACLE_CELL_M` cells — dozens of returns per cell all saying the same thing — and
`np.linspace` over the POINTS kept 4,000 of them landing on a fraction of the cells. Measured on
a synthetic street, the cap covered **127 of 218 occupied cells**: it was not thinning a dense
cloud, it was deleting 42% of the places the planner knew something was standing.

It failed in the **unsafe** direction. Where a corridor edge cuts a densely populated cell only
some of that cell's returns are inside the corridor; keeping one in sixteen by index misses them
and `_scan_arcs` reports the arc clear to the next blocker.

`ObstacleBand.reduce_to_cells` collapses each occupied cell to its MEAN inside `despeckle`,
reusing the grid that function already builds. It is complete by construction, and it is
*cheaper*: 512 points instead of 4,000 on the same scene took `plan_arc` from **10.3 ms to
1.17 ms**. That headroom is what let the deferred families stop scanning `scan_points[::2]` —
against a cell-reduced cloud that second stride would delete half the occupied cells, and do it
to precisely the candidates that decide "the turn comes later". Off by default so **AEB's array
stays byte-identical**: AEB counts the `AEB_MIN_HITS`-th nearest return, so reducing its cloud
changes what its trigger counts and that belongs to its own live checklist.

### Guidance yields to sensing when its own line is blocked (2026-08-11)

CLAUDE.md promised that "guidance never becomes authority ... a blocked arc is never chosen just
because the route asked for it — LiDAR always wins". **It was false, and measurably so.** Both
lateral terms SATURATE — the route cross-track clips at `ROUTE_XTRACK_SCALE_M` (2.0 m) and
keep-right at `KEEP_RIGHT_SCALE_M` (2.0 m) — while passing a 1.8 m car needs 2.3–3.0 m of offset.
So every way round sat in the clipped region and paid the full weight as a flat toll, against
which the free-distance term can only ever offer the difference between two clipped values.
Measured on a kerbed road with a stopped car in the lane and the other lane empty: with no
destination set the planner went round at every gap from 10 to 20 m; **with a route set it went
round at no gap at all** — it drove at the car until free distance crossed `STOP_MARGIN_M`, then
blocked, then reversed.

Two changes, and they are different repairs to the same defect:

- **`_guidance_authority`** scales `COST_ROUTE_XTRACK`, `COST_ROUTE_HEADING` and
  `COST_KEEP_RIGHT` by how blocked the REFERENCE LINE is — the candidate that best obeys the
  guidance — so the question asked is exactly "is the line you are being told to hold actually
  open?". Multiplicative and exactly 1.0 while that line is clear, so **a clear road is scored
  bit-for-bit as before** (verified by A/B: identical curvature and free distance). Measured, it
  is what makes the car commit to the pass at a 20 m and a 15 m gap instead of driving into the
  car.
- **`priced_offset`** replaces `clip((e/s)², 0, 1)` with `e²/(e² + s²)`. Same bound, and for
  e << s it IS the old form to first order so ordinary lane discipline is unchanged — but it
  approaches its bound asymptotically, so the gradient survives at every offset. The clipped form
  had none past one scale: traced in the closed loop, the keep-right target correctly moved to the
  open side of a parked car and the car, 3.6 m away from it, felt a constant force and never
  converged.

**The free-distance denominator now follows the SPEED** (`required_free_distance`), rather than
being the 40 km/h cap's envelope unconditionally. At 17 km/h the car needs 8.4 m and scoring an
ample 8 m of junction box as 0.72 "blocked" tipped three decisions at once — a 90° turn at a
crossroads (whose corridor runs into the far kerb of the receiving arm, so its free distance is
8–9 m by construction), a detour, and any confined manoeuvre. Identical at the cap.
`REQUIRED_FREE_FLOOR_M` exists because at a standstill the envelope collapses to `STOP_MARGIN_M`
and would leave the argmin with no free-distance signal in exactly the situation that needs one.

**A BUSH IS A THING TO IGNORE, NOT A THING TO STEER AROUND** (2026-08-11). The planner's band was
built without the porosity veto on the reasoning that "the planner should steer around a bush even
though AEB should not brake for one". That is wrong in practice: it makes the car flinch at every
verge and refuse gaps that are open. Porosity is now ON for the planner band as well, and it is
the same geometric test — an object of height `a` at range `r` hides the ground behind it for
`r·a/(h − a)`, so ground returns inside that shadow mean the rays went through. Measured on
stripe-sampled ground: a see-through clump at 15 m goes from 12 occupied cells to **0**, while a
SOLID post of the same height keeps all 4, and a wall and a kerb are untouched. The safety
property carries over unchanged and is derived rather than imposed: `a ≥ h` makes the shadow
infinite and the evidence window empty, so nothing as tall as the roof unit can ever be vetoed.
The extent test stays off — it is provably inert at or below `OBSTACLE_MIN_HEIGHT_M`, and any
value above that deletes kerbs.

**What is still NOT fixed, and it is the candidate model rather than any weight.** Every candidate
is hold-then-bend with the bend held to the horizon, so a lane change is not in the set at all. On
a 7 m road the only line that clears a parked car is scored against the kerb it then runs into:
measured at a 24 m gap, the in-lane arc was blocked by the car at 24.3 m and the clearing arc by
the opposite kerb at 24.8 m, so nothing rewarded going round. The pass therefore works today only
where the road is wide enough that the swerve does not reach the far kerb inside the horizon
(verified closed-loop on an 11 m road: 305 m driven, 1.74 m clearance, never leaves DRIVING) and
`test_a_parked_car_on_an_ordinary_road_is_passed` is a **strict xfail** holding the narrow case.

**An S-curve family was built, measured, and NOT landed, and the reason is worth keeping.** A
straight third segment is not enough — a constant-curvature swerve arrives ~32° off, so
straightening there just aims the car at the kerb — so it takes an S: bend out for `L`, bend back
for `L`, which returns the heading exactly and leaves `k·L²` of offset. Implemented (a
per-candidate-pose arc scan, since the return bend starts wherever the outbound bend finished) it
**does fix the 7 m case**, driving past the parked car with 1.43 m of clearance. It also **breaks
the pocket escape, from 92.55 m clear to −4.61 m**, and that is not a tuning problem:

- The sustained free distance quietly carries real information — *a hard turn held to the horizon
  does leave the road* — and in a confined space that shortfall is what stops the planner choosing
  one. The S erases it: every curvature reads the same free distance, the free term stops
  separating them, and the remaining terms pick a bend. Measured, the car left a recovery on a
  26 m-radius arc and drove in a circle for the rest of the run.
- Capping the S at `required_free` removes the circle **and the benefit with it**, because the free
  term clips at that value anyway, so the cap changes no cost at all.
- The root cause is the deferred-family trap in a new place: under per-tick re-planning the car
  drives the OUTBOUND half and re-plans, and if nothing in the scene changed it re-picks the same
  outbound half for ever — the promised return leg never arrives, exactly as a deferral discount
  makes "turn later" never arrive. Landing it safely needs the planner to COMMIT to a manoeuvre it
  has begun, which is trajectory state it does not have.

### The reverse recovery does not ROTATE the car, and that is the real "it turns the wrong way"

**The steering sign is correct.** It was doubted from the driver's seat and checked four
independent ways — the bicycle model from scratch, the mirror-frame algebra, re-deriving both
existing sign tests rather than trusting them, and a closed-loop sim on a plant that reads
`command.steering` as BeamNG's raw "+1 = right" and never touches `STEERING_SIGN`. In every case
where the recovery commands a turn, the tail swings toward the open space. The chain
`previous_curvature = −current_curvature` → `plan_arc(mirror_points(...))` → `−k_m` →
`STEERING_SIGN` is right end to end. Do not "fix" it.

What is wrong is the OBJECTIVE. The reverse planner maximises room BEHIND and has no term for
where the car ends up POINTING, so measured over 108 nose-in geometries with clear space behind
it backed **dead straight in 108 of 108 — zero degrees of heading change** — and where the room
behind was asymmetric it steered productively in only 2 of 6. The car therefore re-approaches the
obstacle at the same angle it failed at, which from the driver's seat is indistinguishable from
steering the wrong way. `MAX_RECOVERY_ATTEMPTS` now bounds that loop (see below) so it ends in
STUCK instead of running for ever, but the loop itself is untouched.

**The mechanism is that the reverse planner has nothing to decide WITH.** With open space behind,
all 41 candidates report `free_distance = 35.0` (the horizon) and full clearance, so the free and
clearance terms are identical across the entire fan and the total cost spread is *exactly*
`REVERSE_COST_SMOOTHNESS` = 0.300. Only the smoothness tie-break separates them, and it always
wins at k = 0. That is why it backs straight: not a bad choice among alternatives, but no
alternatives at all.

**The productive reverse is toward the side OPPOSITE the forward gap**, which is worth stating
because it is counter-intuitive and is what makes a side-preference heuristic hard to get right.
Reversing D metres at travel curvature k_m rotates the car by +D*k_m (measured: k_m = +0.10 over
6 m gives +34.5 deg, nose LEFT, tail RIGHT). So to point the nose at a gap on the LEFT the car
must reverse toward the behind-RIGHT. The planner instead picks whichever side has more room
behind, which is uncorrelated: measured over (forward gap side) x (open-behind side), it was
productive in 2 of 6.

**The cheap fix does not work, and was measured.** Biasing the reverse plan's nav heading toward
whichever side has more FORWARD room took the pocket escape from 92.55 m clear to 0.05 m and
STUCK -- in a pocket the free distances behind are NOT tied, so a preference term worth 0.39
overrides real geometry. The fix has to be the computation the heuristic was standing in for:
score each candidate reverse arc by the forward freedom available from its PREDICTED END POSE
(re-plan forward from each and take the best). Note also that even a productive reverse is partly
handed back -- three recoveries gained +39.8 / +26.8 / +21.5 degrees and the forward re-plan
immediately returned -28.8 / -23.2 / -10.2 -- so the end-pose score has to be what CHOOSES the
arc, not a nudge applied to a choice made on other grounds.

**Two things make correct behaviour look wrong from the driver's seat, and neither is a bug.**
Nose and tail rotate OPPOSITE ways: at k_m = +0.08 the wheels go RIGHT, the tail moves +1.24 m
right and the nose moves 0.02 m LEFT, so a driver reading "which way did it turn" off the nose
sees the opposite of the wheel. And `camera_target` sets `yaw_deg = 180` while reversing, swinging
the chase camera to the front of the car looking back along the direction of travel -- in that
view the car's real right projects onto the screen's LEFT, so every reverse manoeuvre is
left-right mirrored on screen. Combined with the 2-in-6 above, "it always turns the wrong way" is
an entirely expected report from a system whose sign is correct.

### Self-driving

`planner` (pure geometry) → `controller` (state machine) → `worker` actuation, with
`navigation` + `route_model` supplying the reference path (see "Route following" below) and
`planning_map` supplying the planner-only obstacle memory. All of them are Qt-free and
BeamNGpy-free.

The planner is **geometric, not semantic** — a deliberate choice, not an oversight. Drivable
means "no return in the obstacle height band", so flat grass and car parks read as drivable and
the car will explore them; on a kerbed road the 0.20 m mount sees the kerb face and that is what
keeps it on the tarmac. The road-coverage *bonus* this section always named as the upgrade path
now exists (`COST_ROAD_BONUS`, see the planning-memory section below) and is exactly that — a
bonus over unchanged geometric authority, never a swapped input, and it vanishes on unannotated
maps.

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
the route only changes when the player sets a destination. Every failure degrades gracefully —
but transport failure and "no target" degrade DIFFERENTLY now: a parseable no-target clears the
cache immediately (the player cancelling is data), an exception keeps the last good route for
`ROUTE_STALE_GRACE_S`, because one dropped reply used to wipe a perfectly good route for a full
poll interval. The chunk also reads `distToTarget`, `linkCount` and the navgraph `radius` per
node, each total-function with -1/0 defaults so a game version lacking a field degrades that
field rather than the route — the `Route check:` line reports which actually arrived, which is
the live-only fact the offline suite cannot prove.

### Route following: the route became a reference path (2026-08-10)

**The route is now a path to follow, and guidance still never becomes authority.** `route_model`
(imports config/models/geometry/numpy — navigation's exact footprint; the planner never imports
it) turns the cached route into a `models.RoutePath` per tick: projected into the ego frame,
RESAMPLED at `ROUTE_SAMPLE_STEP_M` before any curvature is measured (navgraph spacing is tens of
metres on straights — curvature from raw chords is meaningless), smoothed over
`ROUTE_CURVATURE_SMOOTH_M` (sized against a 90° junction encoded as ONE vertex: smearing π/2
over 9 m reads k≈0.17 → a ~4 m/s creep, conservative because over-reading curvature under-reads
speed), with junction flags (`linkCount > 2` AND the heading actually turns — degree alone brakes
at every straight-through crossroads, curvature alone misses single-vertex turns) and a backward
speed pass at `COMFORT_DECEL_MPS2` folding corners, junctions and the destination into ONE
value.

Five things there are load-bearing:

- **When a route is present it REPLACES nav-heading and keep-right entirely** (the gate in
  `plan_arc`). Two lateral targets fighting — kerb band versus route centreline — is the failure
  the gate avoids, and the kerb band measures a STRAIGHT slice that degrades on exactly the
  bends the route tangent is strongest on. With no route (or fewer than two usable nodes ahead)
  the legacy terms run byte-identically — pinned.
- **Conformance is the MEAN priced deviation over samples along each candidate's composite path**
  (`ROUTE_MATCH_SAMPLES`), each matched to its NEAREST route sample — never at one fixed arc
  distance. One endpoint cannot measure a path through a bend: a shallow arc that cuts a 90°
  corner lands its endpoint ON the ribbon at the apex and priced there it read as perfect —
  measured 7.4 m inside the bend in the closed loop. The tangent term matches each EXACT
  endpoint to its nearest sample and rides `COST_NAV_HEADING`'s normalisation so the tuned rank
  carries over. Both terms are one rule for every family: no deferral discount enters here.
- **The speed preview reaches the controller as `ArcPlan.route_speed_limit_mps` and the
  controller only ever takes `min()` with it** — the sanctioned anticipation channel, same
  category as `next_curvature`. The value is the backward pass sampled `ROUTE_PREVIEW_LEAD_S` of
  travel DOWN the path, not at the ego: the speed loop is a proportional law behind a low-pass
  and carries a standing error of decel·(1/KV + τ) ≈ 3.9 m/s against a ramping target — measured
  arriving at a R=15 corner at 7.9 against 6.5 m/s and overshooting the destination by 8 m. The
  lead cancels exactly that; on a flat stretch of the pass it changes nothing.
- **Arrival is an explicit branch, then a latch.** At a zero target the throttle path serves the
  trim integrator (wound to 0.35 on the drive there) and the coast band hands the last metres to
  engine drag — measured 6.7 m of idle-creep past the marker. So below `HOLD_TAPER_SPEED_MPS`
  with the limit at ~0, `_drive` brakes gently to rest (`ROUTE_ARRIVAL_DECEL_MPS2`, bypassing
  the coast band — against a permanently-zero target there is no chatter to prevent) and holds at
  rest, mode still DRIVING. And because groundMarkers CLEARS the route at the marker (and a
  consumed polyline is too short to build a path from), the worker latches inside
  `ROUTE_ARRIVAL_LATCH_M`: the hold survives the route disappearing, and only a route with more
  than that left — a new destination — releases it. Without the latch the car got its full speed
  cap back right at the destination.
- **The 40 km/h cap is a target, not a promise**: the preview modulates below it for corners,
  junctions and arrival; nothing was changed about the cap itself.

### Planning memory: the planner remembers, AEB never does (2026-08-10)

`planning_map.PlanningMemory` (imports config+numpy only, worker-thread confined — the
controller's confinement argument, not the scene stores' pool-thread contract) accumulates the
planner band's obstacle output as world-anchored 0.4 m cells: per-cell newest per-tick MEAN
position (the cell centre quantises by up to 0.28 m, which eats the 0.35 m clearance margin),
cumulative support, odometer stamp, vehicle wall-clock stamp. Static cells are forgotten by the
METRE (`MEMORY_DISTANCE_M`) and vehicle-classified cells by the WALL CLOCK
(`MEMORY_VEHICLE_TTL_S`) — the WORLD stores' two-clocks design, for the same reasons — with the
same 25 m teleport guard. The query re-projects surviving cells into the current BEV frame and
withholds cells under `MEMORY_MIN_SUPPORT`.

Three rules are load-bearing and pinned:

- **AEB never reads it.** The merged cloud feeds `plan_arc` and `rear_free_distance` only;
  `aeb_obstacles` production and both `_compute_aeb` calls are byte-untouched. A full-authority
  brake on a remembered ghost is the unacceptable failure, so its checklists did not need
  re-running — `test_memory_never_reaches_the_aeb_band` pins it by IDENTITY.
- **Memory can never unblind a blind tick.** The merge sits inside `had_returns`; a sensor
  outage still plans `_BLIND_ARC` and brakes.
- **Merged points, not a soft cost term**: a real remembered kerb must be able to block, and a
  weight that could be outbid would let it be driven through. The accepted consequence is that a
  static ghost can hard-block for up to 20 m of travel — and while parked the odometer stalls, so
  its designed escape is the reverse recovery, which moves the car and expires it.

A fourth rule was added from adversarial review: **seeing the GROUND somewhere outranks
remembering an obstacle there.** The vehicle TTL hangs on the semantic mark, and unannotated
maps never produce one — a crossing car's returns entered as static scenery and painted a
believed phantom wall across the lane, which a then-parked ego could never expire. Any
remembered cell that this tick's road-classified returns cover with `MEMORY_MIN_SUPPORT` hits
while contributing no obstacle return is evicted; the geometric road fallback classifies
vacated tarmac as road on any map, and it keeps re-sampling while parked. An occluded kerb gets
neither kind of return and persists; a visible kerb face gets obstacle returns and is
protected. The same review also gave marked cells WORLD's wall-clock semantics — ANY
observation refreshes the stamp, so a once-marked cell still being observed never expires
mid-observation.

The same store keeps strided road-mask cells for the **road-coverage BONUS** — the upgrade path
the geometric-planner note always named. `worker.build_road_grid` scatters road returns (+ the
remembered cells) into a coarse `models.RoadGrid`; `plan_arc` subtracts
`COST_ROAD_BONUS · coverage` sampled along each composite path over min(free, lookahead). A
bonus, never authority: the free term alone reaches 0.35 at 18.6 m so a pinched arc always
outranks full coverage, and under `ROAD_BONUS_MIN_CELLS` occupied cells (an unannotated map) the
grid is not built and the term vanishes exactly as nav/keep-right do.

### MAX_RECOVERY_ATTEMPTS was unreachable, so the car reversed for ever (2026-08-11)

`_enter` cleared `self._attempts` on every entry to DRIVING, and **every reverse recovery buys at
least one tick of DRIVING on the way back** — backing `REVERSE_DISTANCE_M` recovers free distance
well past `STOP_MARGIN_M + RESUME_HYSTERESIS_M` — so the counter was cleared before it could ever
reach 3. `STUCK` was therefore only reachable when the path never cleared at all. Since the
recovery is open-loop (a fixed distance back, then the same approach from the same offset at the
same heading), what that produced was an exact limit cycle with zero net progress: reverse, creep
forward, block, reverse, for ever. Measured on the pocket scene, forward progress per cycle was
7.29 / 6.81 / 7.42 m with the car returning to where it started each time.

The counter is now cleared by `RECOVERY_PROGRESS_M` (15 m) of forward travel in DRIVING, which is
further than one recovery can hand back, so it cannot be satisfied by the recovery itself — only
by actually getting somewhere.

**A confirmation window on the BLOCKED entry was built and removed, and the reason is worth
keeping.** Requiring N ticks of `free <= STOP_MARGIN_M` before engaging the recovery machine is
the obvious guard against a single-tick phantom, and it does not change the pedal at all (the
window commands the same full `_hold_brake`). It changes the PHASE of the recovery — and that
alone turned the `test_a_cornered_car_escapes_with_a_steered_reverse` pocket from an escape (92 m
clear) into an exact limit cycle (0.8 m, then STUCK), because the recovery's geometry depends on
where the car happens to be when the hold expires. Delaying it is not the neutral change it looks
like. The phantom it was meant to catch has been attacked at the source instead: the gradient
phantom (which was PERSISTENT, so a window would not have caught it anyway) is gone with the
cell-referenced band, and the sampling phantoms are gone with the cell reduction.

### The steered reverse (2026-08-10)

The recovery no longer backs up dead straight. When BLOCKED-and-stopped or REVERSING, the worker
runs `plan_arc` on `mirror_points(merged)` with the mirrored geometry — exactly as rear AEB
mirrors its corridor; rotation preserves handedness so every helper applies unchanged. **The
mirror is a curvature-domain negation**: for front-frame curvature k_f, reversing yaw is
v_signed·k_f with v_signed < 0, so travel-frame curvature k_m = −k_f — `previous_curvature`
enters negated and `_reverse` steers `−reverse_arc.curvature`, before `STEERING_SIGN`, so the
one-place-reconciles rule survives. Pinned by a hand-derived sign test (obstacle behind-left →
tail swings right → positive BeamNG steering, because a fixed-steering car traces the same circle
forward and backward).

**Reversing is a different REGIME and two forward weights were provably wrong for it**
(`REVERSE_REQUIRED_FREE_M`, `REVERSE_COST_SMOOTHNESS`): free distance scored against the 40 km/h
braking envelope made 5 m and 25 m of reverse room nearly indistinguishable, and smoothness at
1.5 made the recovery choose a blocked straight over an open diagonal every time — "steer least"
at 2 m/s is a tie-break, not passenger comfort. Entry and abort both read the ARC's own free
distance: the arc is what will actually be driven, and gating on the straight-back corridor
would refuse a recovery whose whole point is steering around what is straight behind. Rear AEB
stays armed and unchanged underneath, on its own un-memoried band. keep-right, route and the
road grid are all off in reverse; the winner is plan_arc's own argmin — the forward re-plan
after recovery re-orients the car with the full cost stack, so there is deliberately no bespoke
bi-level objective.

### The overlays are DRAPED on the road, not laid in one flat plane (2026-08-11)

Every overlay — both AEB corridors, the planned path, the navigation route — was drawn at ONE
scalar height: `vehicle_geometry.ground_z_vehicle`, the ego's own ground plane, extended flat
across the whole scene. Render Y is gravity-relative height measured from the ego's world Z (see
`world_to_render`, which takes `offsets[:, 2]`), so on any gradient, crest, dip or camber the
drawn road rises straight through the guidance and hides it. In the very test that pinned the
route ribbon, the drawn road sat at render y = 0 and the ribbon at −0.475: **half a metre under
the surface it describes.** That test was pinning the defect.

`GroundField` is a world-anchored height raster of the surface actually being drawn, built in
`refresh` beside the ground mesh from the same bridged cells, and carried **inside `_MeshCache`**
— which is what keeps the two-rate confinement contract intact: `compose` reads one immutable
object committed in a single attribute assignment and never touches a store. It rides in the
cache rather than in a second attribute for a reason: two assignments are two things a compose
tick can catch half-done, and a height field from before a teleport draped over a mesh from after
it would be worse than either alone.

`drape` then lifts each overlay. Six things are load-bearing:

- **The vertices carry their LIFT above the ground in Y and leave carrying a height.** That split
  is what lets one pass drape a flat band and a standing panel together: the threat marker's base
  and top share an XY, so they take the same surface height and the reticle stays upright and
  full-size. Draping "the mesh" rather than "the lift" would shear it with the gradient.
- **It happens AFTER `_aeb_bev_to_render`.** The rear system reasons in a 180°-rotated frame, so
  sampling the terrain in that frame would drape the rear corridor with the geometry in FRONT of
  the car — on a hill, by metres.
- **`confidence` is separate from `heights` and both matter.** Heights are defined everywhere
  (`_pyramid_fill`); confidence says where that is a measurement rather than an inference, and
  decays smoothly so a vertex leaving the observed region FADES back to the ego plane instead of
  stepping off a rim. Where no field exists at all the behaviour is exactly the old flat
  placement, which is the right fallback rather than a missing overlay.
- **The raster is PADDED with never-observed cells.** `sample` clamps out-of-range positions to
  the border, so without the margin a vertex beyond the surfaced ground — the route ribbon runs to
  `ROUTE_PREVIEW_M`, well past `WORLD_ROAD_RADIUS_M` — would take the edge cell's height at full
  confidence, which is the one way a drape can be badly wrong rather than merely unknown.
- **The field resolves stacked ground layers toward the EGO's own plane.** `_ground_cells` keys on
  `(x, y, layer)`, so a bridge deck and the road beneath it are two legitimate rows at one XY and
  the mesh correctly draws both; a height field cannot, and averaging them puts the overlay in
  mid-air for the length of the bridge. Resolved toward the ego rather than toward the lowest,
  because both directions really happen — driving under a bridge the road below is right, driving
  over it the deck is. This is the one place the field deliberately diverges from the mesh.
- **The push is SMOOTHED.** A pull-push fill hands every cell of a hole the same coarse block
  mean, so a ribbon crossing a 6 m occlusion shadow on a gradient would step onto a flat shelf and
  off again. One 3×3 mean per pyramid level makes it a ramp between the hole's own rims, and it is
  EDGE-padded — these are world Z values of hundreds of metres, so a zero-padded border would drag
  the outermost cells toward sea level and the drape with them.

**The route ribbon has a second, better height source and was throwing it away.** Every navgraph
node carries a real `pos.z` (see `navigation.LUA_ROUTE_CHUNK`), so `snapshot.route_world` arrives
with a true road-surface elevation per node; `_route_ribbon` projected it, kept only the plan view,
and flattened the lot. The node Z is now the per-vertex FALLBACK, used wherever the sensors have
not surfaced the ground — which also carries the route out past the end of the store, where the
old code drew a level line into a hill.

Cost: the field is 2.1 ms of a 43.4 ms build on the realistic worst case, inside the 60 ms budget.
The index arithmetic is integer floor-division rather than metres-and-floor, which is worth 14 ms
of a 15 ms build on half a million cells — the two grids are commensurate, and
`floor((c + 0.5)/k) == c // k`.

### Route overlays

The route reaches both views as data, never via a second computation: `BevFrame.route_points`
carries the plan-tick `RoutePath.points` (dashed polyline under the plan arc; the nav spoke now
prefers the route TANGENT over the legacy bearing-to-a-node), and
`PerceptionSnapshot.route_world` carries the world polyline clipped to the preview — the
`surface_materials` optional-field precedent, so WORLD's compose thread gets it through the
frozen snapshot only and the two-rate confinement contract is untouched. WORLD draws it as a
DASHED thin ribbon (`_route_ribbon`): deliberately unbridged dashes (a gap in guidance is a gap
on purpose), subordinate to the plan path by CHROMA and ALPHA, never luminance
(`WORLD_ROUTE_RGB` solved against `test_world_palette`'s CIELAB matrix, `WORLD_ROUTE_ALPHA`
under the path's 1.0), 5 mm under the path's height so the plan wins where they overlap, and not
depth-tinted for the path's own reason. Both overlays exist only while self-driving follows a
route — the overlay disappearing when disengaged is the honest reading.

### Parking bays are found from PAINT, and that is forced by the empty case (2026-08-12)

`parking.py` (config + models + numpy; Qt-free and BeamNGpy-free like `planner` and `aeb`) finds
candidate bays, and `BevFrame.parking_slots` draws them for the user to click. **Detection and
selection only — nothing here steers, brakes, or reaches the planner or either AEB band**, which
is what makes a speculative bay an acceptable thing to draw at all:
`test_the_parking_scan_never_reaches_the_planner_or_either_aeb_band` pins it by IDENTITY, the
same way `test_memory_never_reaches_the_aeb_band` pins the planning memory.

**The obvious detector is gap-based and it cannot do the job asked for.** Measuring the hole
between two parked cars is what production parking assists do, works on unannotated maps and
needs no new sensing — and an EMPTY lot is one enormous gap with nothing in it to find. An empty
bay is *defined* by its dividers, so paint is the only signal that survives the empty case.
(Confirmed live: bay dividers on this map ship as annotated decals and read through the LiDAR,
the same as the lane paint `Marking check:` confirmed. A lot whose bays are baked into the ground
texture returns nothing, and no amount of accumulation recovers it — there is no intensity
channel, per the retired visual-paint experiment.) Gap-based remains the right FALLBACK for
unpainted lots and is not built.

Three pipeline steps, and each was measured wrong at least once first:

- **The stripe angle is SWEPT, not fitted.** A global PCA over a bay row returns the frontage,
  because eight 5 m dividers spread over 17.5 m have their greatest variance across the row — the
  wrong answer by exactly 90°, on the one geometry this exists for. The sweep instead scores each
  angle by how sharply the PERPENDICULAR projection concentrates the cells.
- **And it is swept MORE THAN ONCE, because a lot is rarely one row.** The sweep returns ONE
  angle, so a single pass keeps whichever row carries the most paint and silently drops every row
  at a different orientation. Reported from the app as "the scan only uses the front sensor" —
  and it is not a sensing problem at all: **nothing in the detector filters by bearing**, and a
  row directly behind the car is found in full when it is the only one there (measured: 4 of 4).
  What it is, is that in a real lot the row you happen to be facing wins the sweep, which is
  indistinguishable from a forward-only scan. Each pass now consumes the cells its stripes
  claimed and re-sweeps the remainder (`PARKING_MAX_ROWS`), so two rows at 90° both survive
  (measured: 4 + 4 alone, 8 together, and 11 across three rows spanning bearings 7°–292°). A pass
  consumes its stripes whether or not they yielded a bay, or the next pass re-finds the same angle
  for ever and only the loop bound ends it.
- **That score needs both a width cap and a per-run division, and each fixes a different
  failure.** Sum-of-squared bin counts — the standard concentration measure — lets one long head
  line across the bay heads outscore eight genuine dividers (7,744 against 5,408). Restricting to
  narrow runs fixes that but then PLATEAUS: a 6 m divider stays inside the 0.7 m cap for ±7°, so
  mass was flat across a 14° band and `argmax` took its first element — measured, 83° for
  dividers lying at exactly 90°, after which the stripes merged and no bay survived. Dividing
  each run's mass by its width keeps the score climbing as the projection tightens (62.0 at 90°
  against 20.7 at 83°).
- **`PARKING_OFFSET_BIN_M` must never be finer than `PARKING_MARKING_CELL_M`, and the failure is
  total rather than gradual.** Store cells sit on a 0.2 m lattice, so at a 0.1 m bin a divider
  seen ALONG its length lands in alternating occupied and empty bins — it reads as a row of
  one-bin "stripes" and ties the score of the same divider seen end-on, so the angle 90° out
  scores equally and nothing is found at all. At or above the cell pitch it fills consecutive
  bins, becomes one wide run, and is correctly rejected. It costs no precision: a stripe's offset
  is the MEAN of its cells' own positions, never its bin centre.
- **`PARKING_MIN_BIN_CELLS` is what makes a head line survivable rather than merely outvoted.**
  Seen from the dividers' own angle, a line across their heads does not pile into one bin — it
  smears along the whole row and BRIDGES every divider's bin into a single run too wide to be a
  stripe. Measured, that took a scene from seven bays to none, which is worse than being
  outscored. A divider seen end-on stacks its whole length into one bin; anything crossing the
  projection leaves a sample or two, so a floor of 3 removes only the smear.

Two further rules the geometry carries:

- **Depth is the SHORTER DIVIDER'S LENGTH, and the overlap is only an ADJACENCY test.** Measuring
  depth as the overlap is the obvious reading and it silently deletes every ANGLED lot. In a
  herringbone layout the dividers start along a common aisle edge and run off at an angle, so
  neighbours are staggered along their own direction by `width / tan(angle)` — and subtracting
  that stagger from the depth measures something the bay's real depth does not depend on.
  Measured on a proper 2.5 × 5.0 m lot, the overlap runs 5.00 / 4.33 / **3.56** / 2.50 / 0.67 m
  at 90 / 75 / 60 / 45 / 30°, so a **60° bay — the commonest angled layout — was rejected by
  4 cm** against the 3.6 m floor, and everything below it too. Perpendicular bays were unaffected,
  so a lot whose row curves gets some bays and not others. The aisle test survives the change
  because it works on the LENGTH directly: two 25 m edge lines are 25 m deep either way.
  `PARKING_STRIPE_MIN_OVERLAP_M` (0.5) is what still separates two facing rows, whose overlap is
  metres NEGATIVE, and it is 0.5 rather than 1.0 because a 30° lot leaves only 0.67 m.
- **An offset run is not yet a divider, and TWO FACING ROWS are what prove it.** Facing rows
  across an aisle put their dividers at the SAME perpendicular offset, so each pair merged into
  one stripe spanning both rows and every bay between them then failed `PARKING_BAY_MAX_DEPTH_M`.
  Measured: 5 bays + 5 bays came back as **0**. Each offset run is therefore split into segments
  along its own length wherever it gaps by more than `PARKING_STRIPE_GAP_M`.
- **Pairing runs ONCE over every divider the scan found, in 2D, across all sweeps** — and getting
  there took three tries, each fixing a real live failure. Sort-adjacency on the offset within a
  pass was the first: once runs are split the offsets *repeat*, so it saw only zero-width and
  cross-row pairs, and a stray fragment (a kerb line, a worn patch) between two dividers broke
  the chain and took out the bay on **both** sides. Nearest-overlapping-partner fixed that but
  was still **per pass**, which is the defect a real lot actually hit: a row following a curved
  wall has non-parallel dividers, so one swept angle fits part of it and later sweeps claim the
  rest — and neighbours found on different passes could never meet. Measured live: **18 dividers
  → 12 bays with 6 unpaired**, and on a single sweep **10 dividers → 5 bays with 5 unpaired**.
  `_Stripe` therefore carries centre/direction/half-length rather than the sweep's own
  offset/lo/hi, so pairing is frame-free.
- **Each divider takes its nearest valid partner on EACH SIDE**, not one nearest overall. A
  divider in the middle of a row bounds the bay to its left *and* the one to its right, and
  nominating once loses whichever it did not pick — measured, two bays of ten on a pair of facing
  rows. Both members of an adjacent pair then nominate each other, which makes the dedupe exact.
- **`PARKING_STRIPE_MAX_WIDTH_M` is also the tolerance on the SWEEP ANGLE**, which is not obvious
  from its name. A divider `t` off the swept angle spreads its length over `L·sin(t)`, so at 5 m
  long a 0.7 m cap silently dropped anything past 8° — on a 150 m-radius curved row that cost a
  divider and with it a bay. At 1.0 m it tolerates ~11°, and stays under half the narrowest bay
  so two adjacent dividers still cannot merge into one run.
- **The scan radius must be comparable to how far paint is DRAWN.** WORLD renders road markings
  to `WORLD_ROAD_RADIUS_M` (100 m) while the scan culled at 35, so a lot showed a full row of
  painted bays with outlines on only the near few — measured, detection stopped dead at 32.9 m.
  `PARKING_SCAN_RADIUS_M` is now 60, sized to `LIDAR_ROOF_FAR_M` (55 m, the all-round ground
  reach) rather than to a round number, since past that only the forward road-scan wedge reaches
  and bays beside and behind the car would stop being found anyway. The sweep's cost is driven by
  cell count and the angle count, not the radius: 9.1 ms at 35 m against 9.3 at 60 on the same
  2,786-cell lot.

**`ScanReport` exists because a screenshot cannot answer "why is there no outline on that painted
bay".** Detection is a chain of geometric filters and the screen shows only the survivors, which
is exactly how the two defects above were reported. The report counts what each filter consumed
and the `Parking check:` line prints it — one-shot, then only when the bay count CHANGES or a
minute passes, because a per-scan line at 2 Hz would bury the log. A row that is present but
unpaired reads as `N dividers -> 0 bays ... N unpaired`; a lot whose paint never annotated reads
as `0 marking cells`; the cap reads as `over the cap`.
- **Occupancy never adds or removes a bay**, it only marks one. A bay whose paint is there is a
  bay; "that one has a car in it" is a different claim from "that one does not exist". The
  rectangle is shrunk by `PARKING_OCCUPANCY_MARGIN_M` first, because the dividers themselves, the
  kerb at the head and a neighbour's wing mirror all sit on the boundary.

**Three different rates, on purpose.** Marking cells are folded into `MarkingMemory` EVERY tick,
because the continuous dividers the detector needs exist only by accumulation — one frame lays
ground rings ACROSS a divider, not along it, so a stationary car at the lot entrance sees arcs
cutting the lines rather than the lines. The bay SET is rebuilt every `PARKING_SCAN_INTERVAL_S`
(the sweep is the expensive part and a lot does not change shape). The PROJECTION into the BEV
frame runs every tick regardless, so the drawn rectangles stay glued to the ground between scans
instead of lagging the car.

`MarkingMemory` is its own store rather than a third array on `PlanningMemory`: that one is
documented planner-only and is gated on `self._self_driving`, whereas scanning for a bay is
something you do while driving the car yourself. It forgets by the METRE like every other store
(`PARKING_MARKING_MEMORY_M` at 80 m, four times the planner's, because paint does not move, the
ego pose is ground truth, and a lot is crossed at a crawl) with the same 25 m teleport guard.

**Bays are drawn and picked in WORLD, and RAW BEV deliberately does not draw them.** They are a
patch of GROUND, so the view that draws the ground draped over real terrain is where they belong;
in the plan view they were a rectangle at 10 px on a 35 m radius. `PerceptionSnapshot.parking_slots`
carries them to the compose thread — the `route_world` precedent, frozen snapshot only, so the
two-rate confinement contract is untouched — already projected into the BEV frame by the worker,
because BEV to render is a fixed relabelling (`right, lift, -forward`) and re-deriving it would
introduce a second pose to disagree with the first.

**Picking is the ENGINE's raycast, not arithmetic here.** `View3D.pick` is the only thing that
knows this camera's projection, so QML answers *where in the scene* the click landed and
`SceneBridge.parkingPicked` turns that point into *which bay* with the same containment test the
worker uses. Reproducing the projection in Python would mean pinning down Qt Quick 3D's euler
convention by hand — the class of guess this project has measured its way out of twice. Verified
by clicking: a viewport sweep picks each drawn bay and gets its correct world centre back.

**Bays a scan MISSES survive for a short distance** (`remember_bays`, `PARKING_BAY_MEMORY_M`).
Each bay sits near several thresholds at once, so a scan can drop it and find it again a moment
later — reported live as the bays flashing and, far worse, as the SELECTION going away with them,
because a selection is re-matched against the offered set every scan. Remembered by the METRE
like every other store here, and a freshly found bay always replaces its remembered twin, so
memory only ever fills gaps and never overrides what the sensors say now. The identity key is a
coarse 1 m grid: the same bay is re-measured every scan and its centre wanders a few centimetres,
so an exact key would make every scan a different bay and remember nothing.

**A selection is held as a WORLD POSE, never as an index.** The bay set is rebuilt on its own
cadence and a subscript means a different bay afterwards, so the bridge reports the clicked bay's
world centre and the worker re-matches it within `PARKING_SELECT_MATCH_M`. `match_selection`
returns None rather than the nearest bay regardless — a selection whose paint has aged out must
go quiet, not silently become its neighbour. A click that hits no bay emits
`parking_selection_cleared` rather than a coordinate chosen to miss, because "deselect" is a
different message from "select the bay here" and a fabricated coordinate would depend on the
match radius to stay a miss.

### Translucent vertex colour DOES NOT BLEND in this scene, and a bay is an outline because of it

The bays were built as a translucent wash first. Measured on the real GPU over the road, one flat
`#c6c8c1` quad renders:

| vertex alpha | sampled | |
|---|---|---|
| 1.0 | (198,200,193) | correct — exactly `#c6c8c1` |
| 0.999 | (198,200,193) | correct |
| 0.9 | (220,222,215) | **brighter than opaque** |
| 0.5 | (255,255,255) | saturated white |
| 0.2 | (255,255,255) | saturated white |

Premultiplying the colour changes nothing, and swapping in the AEB corridor's own material
changes nothing, so it is not the material declaration. Vertex **RGB** binds perfectly (alpha 1.0
reproduces the configured colour exactly; a red-forced buffer renders red), so it is the alpha
channel specifically. A 0.20 wash therefore rendered every bay as one solid white slab.

So the parking overlay is **entirely opaque** and gets its hierarchy from HUE and from GEOMETRY:
a candidate bay is an OUTLINE (which is how a real bay is painted anyway, and it leaves the road
and the bay's own dividers visible through the middle), an occupied bay adds a cross, and only
the SELECTED bay is filled — exactly one ever is, so an opaque fill is affordable there and is
what makes the choice unmistakable. `test_world_scene.py` pins that every parking vertex ships at
alpha 1.0.

**This puts a question over the AEB overlay**, which carries a 0.04 wash and a 0.80 rail in one
buffer on the same assumption. CLAUDE.md's supporting measurement used a **black** quad, and
black is the one colour that cannot tell a correct blend from this failure — both give
`background · (1 − a)`. The AEB corridor's bright violet and red at low alpha are exactly the
case that would blow out. Not investigated here; it is a separate overlay with its own checklist.

**Not live-checked**, and its checklist is: bays appearing at all on a real lot (the
`Parking check:` line reports count, evidence and the nearest bay's size, and a marked lot
producing nothing means that map's bays are texture-baked rather than decals); bays appearing
BEHIND and BESIDE the car as well as ahead, which is what the multi-row sweep is for; the row
filling in as you drive past rather than only under the car; whether real bay paint holds
`PARKING_MIN_BIN_CELLS` per bin at range or wants lowering; whether a real lot's head line, arrows
and hatching (which are `DRIVING_INSTRUCTIONS`, deliberately not in `MARKING_CLASSES`) leave the
sweep alone; that a bay with a car in it reads occupied from the car's own returns; and that a
click lands on the bay under the cursor at a range and a camera orbit, since the raycast is the
one part with no offline proof.

### Driving into the bay: a separate controller, because the planner cannot (2026-08-12)

`parking_drive.py` (config + models + numpy; Qt-free and BeamNGpy-free) drives the manoeuvre.
It is deliberately **not** the arc planner, and the reason is already in this file: the S-curve
family was built, measured to fix the narrow-road pass, and **not landed** because under per-tick
re-planning the car drove the outbound half and re-picked it for ever — "landing it safely needs
the planner to COMMIT to a manoeuvre it has begun". Parking is that problem in its purest form.

**What is committed is the GOAL, not the path.** The bay is world-anchored and the worker
re-projects it into the BEV frame every tick, so the path is re-derived each tick to an unmoving
target. That is a feedback law, not a re-choice — there is one target and one path family, and
the planner's failure mode (re-picking a different candidate each tick) cannot occur when there
is nothing to pick between. It produces a `ControlCommand` inside a `DrivingPlan`, so
`worker._actuate` sends it unchanged, gear handling and the AEB override included.

**Scope: forward, nose-in only.** A bay needing a pull-forward-and-back shuffle is reported
UNREACHABLE rather than half-attempted, which would end with the car across the lines. The phase
machine is shaped so a reverse segment could be added; it is not implemented.

**There is a HARD geometric envelope, and it is the feature's main limitation.** One arc of
radius R changes heading by 90° and displaces the car *exactly* R sideways and R forwards, so a
square-on bay nearer than `MIN_TURN_RADIUS_M` (6 m) ahead, or nearer than that to the side,
**cannot be entered nose-first at all**. Reported live as "it starts and then stops right away"
— and it was not a bug: the bay clicked was inside the envelope. Three things follow, and the
first two are dead ends worth recording so they are not re-tried:

- **A smaller displacement needs a smaller radius, which the car does not have.**
- **An S-turn does not help.** A single arc is already the MINIMUM lateral displacement for a
  given heading change; any S displaces more.
- **A straight staging run up the aisle does not help either**, and was built and removed. It
  moves the bay CLOSER, which is the wrong direction for precisely the bays that fail — measured,
  the reachable envelope was identical with and without it.

What reaches a near bay is reversing in or repositioning first. `reachability()` names the reason
in words the driver can act on ("only 4.0 m ahead and turning into it needs about 6.0 m — back
up, or pick one further ahead") instead of the bare "no single forward move fits", which is true
and useless. `test_the_reachable_envelope_is_the_turning_circle_not_a_tuning_choice` pins it so
nobody "fixes" it by loosening a constant.

### The parking planner is a SEARCH now: Hybrid A* over Reeds-Shepp (2026-08-12)

`reeds_shepp.py` + `hybrid_astar.py` replaced the hand-written manoeuvre families, because those
could not be patched into covering a real lot and the reason was structural rather than a matter
of tuning. Each family was ONE straight-arc-straight, which needs `R(1 - cos t)` of lateral room
and `R sin t` of longitudinal room to turn through `t`: a bay 1.9 m to the side needs 56 degrees,
so 2.6 m across and 5.0 m along **for the arc alone**, and the setup pose it would have to reach
first is itself beside the car. Widening every search range changed the reachable envelope by
nothing, which is what proved it was the path family and not the parameters.

- **Reeds-Shepp is a STEERING FUNCTION, not a planner.** Shortest path between two poses for a
  car with a minimum radius that can reverse, in EMPTY space. It knows nothing about obstacles.
  Implemented as three word formulas plus three symmetries (timeflip, reflect, backwards) = 24 of
  the 48 words; the omitted CCSC/CCSCC families cost a slightly longer path, never a wrong one.
  Leaving out the BACKWARDS symmetry alone left 37% of random poses with no word at all.
- **The LRL branch is derived, not copied.** Published sources disagree on it and the wrong one
  is silently plausible -- it yields a path of the right SHAPE ending somewhere else. `t` is
  `theta + u/2`, not `theta + pi/2 + acos(rho/4)`, and the only check that catches it is driving
  the path and measuring where it stops. Every returned path lands on its goal to 1e-8.
- **Hybrid A\* is the planner.** It searches the car's STATE (position and heading) by expanding
  short arcs it could really steer, so every node is drivable by construction. Reeds-Shepp appears
  twice: as the heuristic (straight-line distance is near zero for a goal beside the car facing
  the wrong way, which is the case needing the most manoeuvring) and as an analytic shortcut to
  finish exactly. Measured: every previously-refused bay solves, 16-141 ms, ending on the goal.
- **`Occupancy` distinguishes FREE, BLOCKED and UNKNOWN**, and that third state is new to this
  codebase. The stores record where returns CAME FROM, so absence of a return has always read as
  drivable -- and a planner allowed to route through never-observed space plans through walls it
  has not looked at. Road returns are positively free; everything unseen is traversable at a
  COST rather than forbidden, because forbidding it strands the car in a lot it has half seen.

Four things the executor needed before it could drive a searched path, each caught by a test:

- **A searched leg carries its own PATH**, not just an endpoint. The canned legs were each a
  single arc, so a path could be re-derived from the endpoint; re-deriving a searched leg that
  way produced a **148 m path for a leg 6 m away**.
- **Pure pursuit must locate the car ON the path first.** A re-derived path starts at the car, so
  index 0 is right; a searched leg is a fixed path over the ground and the car is somewhere along
  it. Assuming index 0 sent the tracker chasing the far end and drove the car **71 m away**.
- **Re-plan at every cusp.** Leg two was planned from where leg one was MEANT to finish, and over
  a few metres of reversing there is no distance in which to soak up the difference -- it parked
  **1.20 m off the centreline**. Planning afresh from the real pose resets the error, and a cusp
  happens once or twice a manoeuvre rather than once a tick.
- **The gear is committed only once STOPPED.** `self._gear` is what the shift holds while the car
  is still rolling, so assigning it the gear about to be requested makes the hold a no-op and
  sends the shift at speed -- caught at 1.13 m/s.

**Cost, and it is real.** A search is 16-141 ms and re-planning at each cusp means several per
manoeuvre. That is affordable once per manoeuvre on the worker thread but it is NOT a per-tick
budget: if the engage moment ever feels like a hitch, moving the search to the scene worker's
pool is the fix, not making it cheaper.

### The multi-leg manoeuvre: position, then reverse in (2026-08-12, superseded)

`plan_manoeuvre` plans the legs that put the car in a bay the nose-in envelope cannot reach —
**position forwards, then reverse in** — and `ParkingDriver` walks them. Measured closed-loop
against a kinematic bicycle: bays 2 m ahead and 6 m to either side, level with the car, and 3 m
BEHIND it all park square and facing out, which is what reversing in means.

- **A leg is held in the BAY's frame, not as a path**, and that is the design. The bay is
  world-anchored and re-projected every tick, so a leg stays put while the path to it is
  re-derived from wherever the car actually is — the single move's feedback argument, extended to
  a sequence. Storing a path would freeze a plan the car then drifts off; re-deriving the whole
  SEQUENCE every tick would reintroduce the re-choice problem this controller exists to avoid,
  flipping between manoeuvres and finishing none. **Commit to the leg, re-derive the path.**
- **Reversing is the forward solver in a frame rotated 180°** (`_reverse_reach`): solve to the
  negated target and negate the answer. Same trick as `aeb.mirror_points` and the steered
  reverse — a rotation preserves handedness, so every helper applies unchanged.
- **Backing in ends the car facing OUT**, so it is the TAIL that clears the head of the bay and
  the stop pose is measured from `rear_m`, not `front_m`.
- **Every leg is checked from where the PREVIOUS one finishes.** A sequence is offered only when
  all of it solves; starting one whose second half is impossible leaves the car across the aisle.

Measured coverage for 90° bays (rows metres ahead, columns metres to the side): the 5–8 m band
now solves at **every** distance including 1 m ahead, where nose-in needs 6 m. **Still unreached:
bays under ~5 m to the side**, which need a genuine three-point turn (forward, reverse, forward)
rather than two legs.

Five things the executor got wrong first, each caught by a test written for it:

- **The gear must not be SENT until the car is at rest.** A leg ends while the car is still
  rolling, so commanding the new direction as the leg ends sent a reverse shift at **1.22 m/s**.
  The old gear is held until the car has actually stopped, and only then is the new one asked
  for; `shiftToGearIndex` has side effects and the box only engages at rest.
- **A finished park must keep asking for the gear it ARRIVED in.** The hold branch hard-coded
  forward, so a park that ended by reversing was sent back to drive at **1.01 m/s** while still
  rolling to a stop.
- **A SETUP leg is not made until the car is SQUARE to it, not merely at its position.** Position
  alone declared it done with the car still turning; the reverse then would not solve from where
  the car actually was, so it re-planned, arrived at the same setup, and cycled — stuck in
  SHIFTING for the whole run. The FINAL leg is exempt: nothing follows it, and its heading is
  what the tracker spends the last metre correcting.
- **Re-planning is BOUNDED** (`PARKING_MAX_REPLANS`). Re-planning when the committed sequence has
  become undriveable is right — the goal never changes, so it is not per-tick re-choosing — but
  unbounded it cycles.
- **The setup's offset side and its FACING side are searched independently.** Tying them together
  made the manoeuvre work on one side of the aisle and not the other: a bay 7 m to the RIGHT
  failed while its mirror 6 m to the left parked. Which way the car should point at the setup
  depends on where it is coming FROM, not on which side of the bay it waits.

**Rear AEB is left armed and allowed to win.** Unlike the forward brake it arms at 0.5 m/s, so it
really can fire while backing in — which is exactly when it should. If it fires the manoeuvre
hands back rather than fighting a system that has decided the car is about to hit something.

**Still not reached: some poses beside the bay** — measured, a bay 4 m ahead and 7 m to the right
plans a setup the car then cannot reverse out of, exhausts the re-plan budget and hands back. It
stops safely rather than cycling, but it does not park. A three-point turn (forward, reverse,
forward) is what covers those, and it is not built.

Five things were measured wrong first, and each is a constant or a construction now:

- **A cubic Bézier is the wrong path family and cannot be told about a minimum radius**, only
  measured afterwards. On the commonest case in a lot — a bay square-on to the aisle — the
  flattest Bézier reaching the pose still bent to **0.20 1/m against the car's 0.167 limit**, so
  every square-on bay came back unreachable. `_straight_arc_straight` hits the limit exactly and
  is what a driver does: straighten, one steady turn, straighten.
- **The approach distance must be SEARCHED, and the intuition is backwards.** The entry point
  sits that far back *along the bay axis*, which for a square-on bay is back across the aisle
  toward the car — so a long approach leaves only a couple of metres of lateral offset while
  turning 90° displaces the car by a whole radius. They solve at about a metre: turn in AT the
  mouth, not five metres before it.
- **The TIGHTEST feasible radius is preferred, not the widest**, and this is also the opposite of
  the obvious choice. On a square-on bay the arc must deliver a fixed lateral offset, so a tighter
  arc finishes sooner and leaves a **longer straight run-in**: measured, the widest feasible
  radius left 0.6 m of run-in and the car was still **7.2° off square** as it crossed the stop
  plane; the tightest leaves 1.8 m and it arrives straight. A tight turn costs nothing at parking
  speed — 0.33 m/s² at the 6 m minimum.
- **Arrival is judged on the STOP POSE, before any path is planned.** The car tracks with a
  lookahead so it can roll centimetres past the stop point, and once the pose is behind it *no
  forward path reaches it* — so judging arrival by "the path got short" reported UNREACHABLE at
  the exact moment of arriving, measured **0.41 m past the mark**. Passing the stop plane IS
  arriving. Two smaller versions of the same trap: `PARKING_PATH_SLACK_M` (tracking lag put
  `run_in` at −0.016 m and the exact construction refused it) and the zero-approach endgame
  candidate.
- **Speed is capped while there is HEADING left to lose**, not only by distance-to-go. Distance
  alone let the car cross the stop plane at cruising speed with the wheel still wound on —
  centred in the bay but visibly crooked. `PARKING_TURN_SLOW_DEG` is what gives the tracker time
  to straighten.

**Parking creeps below `AEB_MIN_SPEED_MPS` on purpose** (1.4 against 2.0 m/s), so the forward
emergency brake stays in STANDBY and cannot fire at the kerbs, walls and neighbours a park drives
close to by design. That is a deliberate property, not a coincidence to be tidied: the manoeuvre
therefore does **its own** corridor check (`blocking_distance`) over the swept body, and
`test_the_manoeuvre_stays_below_the_speed_the_brake_arms_at` pins the relationship.

Self-driving is disengaged when parking engages — one thing steers the car at a time — and the
manoeuvre's ribbon **replaces** the plan ribbon in WORLD while it runs, because the arc planner
is not running and two ribbons would claim two opinions about where the car is going. `ARRIVED`
stays engaged and HOLDS the brake: this is the one place the hand-back-a-coasting-car rule is
wrong, since releasing at the stop line lets the car roll out of the bay.

`test_parking_drive.py` runs the real controller closed-loop against a kinematic bicycle,
re-projecting the world-anchored bay every tick exactly as the worker does, and asserts where the
car **stops** — within 0.30 m of the bay centreline and 6° of square, for straight-in, square-on
left and right, 45°, and an offset start.

**Not live-checked.** Its checklist: that a real bay is entered squarely rather than clipping a
line; that `Park check:` reports UNREACHABLE rather than attempting a bay the car cannot make in
one move; that the corridor check stops the car for a neighbour parked over the line (AEB will
NOT help — it is in STANDBY by design); that the car holds in the bay instead of creeping; that
disengaging hands back a coasting car; and that the gear stays in DRIVE throughout, since a
single forward move is all this commits to.

### Reliable parking execution update (2026-08-12)

The active bay and trajectory are now both committed. Engagement copies the complete selected
`ParkingBay` into an immutable `ParkingJob`; rescans may change the overlay but cannot move the
active goal. Every leg carries a bay-relative path and the tracker advances monotonically along
it. Replanning is event-driven and bounded: blockage, excessive cross-track error, lack of
progress, a direction cusp, or a stopped terminal correction.

Parking independently activates obstacle extraction and maintains a bounded world-cell
`ParkingMap`. Hybrid A* searches pose plus gear, checks the swept oriented body between samples,
and returns costed cusp-safe forward/reverse legs. The live corridor check uses the same body
geometry from current progress. An occupied selected bay is refused before actuation.

Crossing the path endpoint enters `SECURING`, not `ARRIVED`. Success requires a stopped dwell,
pose and heading tolerances, and the measured body inside the bay envelope. The final control
applies the parking brake, marks the job `SUCCEEDED`, and ends automatic parking without running
the normal release-controls funnel. Blockage remains latched until the car is stopped and clear;
an unreadable gearbox uses limited signed-direction confirmation and opposite motion fails
stopped rather than being guessed correct.

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
  them** — extra rays go to azimuth. Only the CHANNEL count does: the roof unit runs
  `LIDAR_ROOF_VERTICAL_RESOLUTION = 512` over its 6–100 m annulus, computing to 0.12 m spacing at
  20 m and 0.75 m at 50 m — the halved-`Δθ` figures the 100 m road radius is built on, still
  awaiting their live check. (Measured at 256 over 6–80: 0.24 m at 20 m, 1.48 m at 50 m.)

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
- `SLOPE_ALLOWANCE_PER_M` is a **bound** on `planner.ground_rise`, not the estimate — and since
  2026-08-11 **neither band uses it**: both are cell-referenced, so the cone survives only as the
  fallback path for a band that asks for neither. Do not reach for it to fix a gradient problem;
  it is the thing that CAUSED the gradient problem.
- `AEB_POROSITY_*` are now shared by BOTH bands, not just AEB's. The planner uses the same
  see-through veto, so loosening them to stop AEB braking for foliage also makes the PLANNER
  ignore more, and vice versa. They were tuned for the brake; check both if you move them.
- `OBSTACLE_CELL_M` (0.4, the FLOOR's reference) vs `OBSTACLE_COARSE_CELL_M` (2.0, the CEILING's).
  Two different questions — how high is this above the ground under it, and is this thing
  overhead — and they cannot share a cell. See the cell-referenced section.
- `REQUIRED_FREE_DISTANCE_M` is FIXED at the 40 km/h envelope on purpose. A speed-scaled version
  was built and removed: approaching an obstacle the speed law holds `free ~= v^2/(2a) + margin`
  while the scaled requirement is `v^2/(2a) + STOP_MARGIN`, so the two track each other by
  construction, the free term goes dead during the whole approach, and the planner concludes
  "this way ends but I can stop, so it is fine" -- measured, it removed the entire reward for
  going round a parked car.
- `PLANNER_MAX_OBSTACLE_POINTS` is now a **backstop, not the primary bound** — the primary bound
  is one point per occupied cell. Raising it does nothing; it never binds on a real band.
- `LAT_JERK_MAX_MPS3 = 4.0`, up from 2.5: at the speed cap 2.5 allowed 0.020 /s, so winding on
  the 0.04 curvature of an ordinary 25 m bend took 2.0 s and 22 m — most of the 35 m horizon.
- `RECOVERY_PROGRESS_M` (15) is what makes `MAX_RECOVERY_ATTEMPTS` reachable. It must stay
  comfortably above `REVERSE_DISTANCE_M` or one recovery satisfies it and the limit cycle
  returns.
- `ROUTE_XTRACK_SCALE_M` / `KEEP_RIGHT_SCALE_M` are now the HALF-COST point of
  `planner.priced_offset`, not the saturation point: `e²/(e² + s²)` reaches 0.5 at one scale and
  approaches 1 asymptotically. The near-field shape is unchanged, so these keep their tuned
  meaning for ordinary lane discipline; what changed is that there is a gradient past them.
- `WORLD_GROUND_FIELD_CELL_M` (1.0) is deliberately four times `WORLD_CELL_SIZE_M`. This is a
  height LOOKUP, not geometry: a road varies smoothly at the metre scale and the overlay only has
  to clear the surface, so the extra resolution buys nothing and costs 16× the raster.
  `WORLD_GROUND_FIELD_FILL_CELLS` is how far the field is trusted beyond what was observed, and
  `WORLD_GROUND_FIELD_MAX_SPAN_CELLS` is a guard against a store that failed to expire, not a
  tuning knob.

`geometric_obstacle_sets` costs ~4.4 ms of the 40 ms tick for the planner's floor alone and
~5.1 ms for both floors, measured on a 60k-point worst case; AEB's corridor scan adds 0.05 ms.

The parking constants have the same "looks like one quantity, is two" trap:

- `PARKING_MARKING_CELL_M` (0.2, the store's grid) vs `PARKING_OFFSET_BIN_M` (0.25, the sweep's
  grouping). The second must stay **at or above** the first — see the parking section; below it
  the detector silently finds nothing at all rather than finding less.
- `PARKING_MIN_BIN_CELLS` (3, is this bin a divider seen end-on) vs `PARKING_MIN_STRIPE_CELLS`
  (8, is this run a believed divider). The first removes a crossing line's smear so the runs stay
  separate; the second decides whether a separated run is real. Raising the first toward the
  second starts deleting genuine dividers at range.
- `PARKING_STRIPE_MAX_WIDTH_M` (0.7, how wide a stripe may be) is what makes both the width cap
  and the sharpness division mean anything; it is not a paint width, it is the tolerance on
  seeing a divider end-on.
- `PARKING_MARKING_RADIUS_M` (70, how far paint is STORED) vs `PARKING_SCAN_RADIUS_M` (60, how
  far bays are OFFERED). The store keeps what has been seen so the picture stays filled in; the
  scan offers what is close enough to be worth driving to. **The store must stay the larger of
  the two**, or it rather than the scan bounds how far bays are found — and silently, since the
  scan would simply receive fewer cells. Raise both together.
- `PARKING_STRIPE_GAP_M` (3.0) is bounded on BOTH sides and neither bound is slack: above the
  observation gaps along one divider (ground returns thin with range, so metre-scale holes are
  normal) and below an aisle (6 m and up). Too small and one divider becomes several fragments,
  each too short to bound a bay; too large and two facing rows merge back into one stripe and the
  whole lot disappears.
- `PARKING_MARKING_MEMORY_M` (80) is deliberately four times `MEMORY_DISTANCE_M` (20). That one
  bounds the lifetime of a remembered ghost that can hard-block the planner; this one bounds how
  long a bay you can see is offered, and its worst failure is a stale suggestion.
- `PARKING_MAX_ROWS` (3) is how many row ORIENTATIONS are looked for, and it is the constant that
  makes bays appear anywhere other than the row you are facing. It is not a bay count —
  `PARKING_MAX_SLOTS` is. Two facing rows either side of an aisle plus a perpendicular row along
  an end wall is what 3 covers.
- Every `WORLD_PARKING_*` colour is OPAQUE and there are no alpha constants, deliberately: see
  the vertex-alpha section above. Reach for a different hue or for outline-vs-fill, never for a
  wash.

The WORLD constants have a trap of their own: **several of them are sized by the SENSOR and one
is sized by the RENDERER, and it is no longer the renderer that binds.**

- `WORLD_RADIUS_M` (190) vs `WORLD_ROAD_RADIUS_M` (100) vs `WORLD_SURFACE_RADIUS_M` (55) are three
  different questions — how far STRUCTURE is observable, how far the ROAD is, and how far OPEN
  GROUND is. Collapsing any pair either shreds a surface into rings or throws away the reach. The
  road outreaches the terrain because it is driven along and accumulation fills it in; the terrain
  beside it is never swept that way. The third one is also the scene-build cost bound, and it is
  the constant to move if SCENE BUILD starts logging. See the renderer section.
- `ROAD_CLASSES` (may the car drive here) vs the `WORLD_SURFACE_*` class sets (what is the ground
  made of) are likewise different questions over the same palette, and a class can be in both. Do
  not fold the material sets into the road set to "simplify": road feeds the road store and the
  BEV split, materials feed colour only, and `NATURE` covers both grass and tree canopy.
- `WORLD_CELL_SIZE_M` (0.25, the ground) and `WORLD_COLUMN_SIZE_M` (0.125, the slabs) are no
  longer equal, and **the asymmetry is the point** — see "The blocks are finer than the ground"
  below. Both were 0.5 originally because of the `ProceduralMesh` Python loop rather than because
  of the data; with the raw-buffer bridge the grid follows the sensors instead.
- `WORLD_ROAD_BRIDGE_CELLS` (6) and `WORLD_COLUMN_BRIDGE_CELLS` (12) are the same idea on
  different surfaces and both scale with **their own** cell size — which is now a different
  number for each: halving a cell must double its bridge or the same physical gap stops being
  closed. Both span ~1.5 m today. The column figure went 6 → 12 with `WORLD_COLUMN_SIZE_M`, and
  the road figure 3 → 4 → 6 as `LIDAR_ROOF_DENSITY` went 12.5 → 25 and the azimuth stripes
  widened. Getting this wrong is how a finer grid renders *worse* than a coarse one — the far
  field breaks into disconnected fragments, which reads as harder blockiness than the cell size
  ever did. `test_azimuth_stripe_gaps_are_bridged_but_a_real_opening_is_not` is the guard and it
  is written in METRES (1.5 m stripes bridged, a 4 m opening not) precisely so it catches a cell
  size that moves without its bridge.
- `WORLD_MAX_COLUMNS` (200k) must move with `WORLD_COLUMN_SIZE_M`, and forgetting it is a
  correctness failure rather than a memory one: the cull drops the excess **oldest-first**, and
  the oldest voxels are the accumulated stripe sweep that fills a striped facade in, so walls
  behind the car dissolve. Pinned by
  `test_the_slab_half_of_the_build_stays_inside_the_scene_budget`.
- `WORLD_COLUMN_VERTICAL_BRIDGE_BINS` (2 bins, 0.5 m) is the *opposite* axis and must stay small.
  It counts `WORLD_COLUMN_HEIGHT_M` bins, not `WORLD_COLUMN_SIZE_M` cells, so the slabs going to
  0.125 m left it alone.
  It has to exceed the vertical sampling gap on a wall (0.10 m at 50 m) so walls stay solid, and
  stay far under the clear air beneath a canopy or a bridge deck so those still split. Raising it
  to swallow noise would re-merge the tree with the grass under it.
- `WORLD_ORIENT_CELL_M` (1.0) is a STRIDE, not the window: the window is a sliding 7x7 of tiles,
  so it is 7 m — sized to hold enough azimuth-stripe samples of a wall at range to fit a line to
  (see the slab section; 3 m starved on sparse walls and they shattered into world-aligned
  confetti). `WORLD_ORIENT_BUCKETS` (24) is the angular resolution — finer is cheaper as well as
  better — and the two guards (`WORLD_ORIENT_MIN_CELLS` 4, `WORLD_ORIENT_MIN_ANISOTROPY`) are
  what stop a bush being handed a direction it does not have. The count guard dropped 6 → 4 with
  the bigger window because four collinear stripe samples ARE a direction; the anisotropy guard
  is what rejects four clumped ones, and relaxing THAT one is the silent-failure dial: every
  clump of foliage acquires a confident, wrong angle.

  **`WORLD_ORIENT_MIN_CELLS` does NOT scale with `WORLD_COLUMN_SIZE_M`, and the obvious
  reasoning that it must is wrong.** The window is 7 m of world, so halving the cell size looks
  like it has to quadruple the cells inside it and quietly relax the guard by 4×. Measured, it
  does not: a cell count is bounded by the RETURNS, not by the lattice. A wall sampled at 1.5 m
  azimuth stripes holds **6 cells in the densest window at 0.25 m and 6 at 0.125 m** — each
  stripe lands in one cell either way. Raising it to 16 to compensate (which is what this
  document recommended before the measurement) would have refused every sparse wall and brought
  back the staircases the orientation pass exists to remove. Only a densely-sampled BLOB scales
  (a bush went 88 cells → 298), and the blob was never this guard's to reject — measured at both
  sizes it comes back oriented 0 of 88 and 0 of 298, because anisotropy is a **scale-free ratio**.
  The two guards asking different questions is exactly what makes the finer grid safe here.
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

The vehicle-fit constants have their own version of it:

- `VEHICLE_FIT_STRIPE_RAD` (0.062) is the azimuth spacing of the **170° units**, which is the
  worst case; the front wedge is 14× finer. It is used for two different things — the clustering
  lattice, where coarse is harmless because cells just hold more points, and the CAP on the
  measured extent correction. Do not re-derive it from the front unit.
- `VEHICLE_FIT_LINK_AZIMUTH_CELLS` (2) vs `VEHICLE_FIT_LINK_RANGE_CELLS` (6). The asymmetry is
  measured, not tidy — see the vehicle-fit section. Equalising them splits every car whose flank
  is oblique.
- `VEHICLE_FIT_SIDE_LENGTH_M` (3.0) is the threshold on BELIEVING a length, and
  `VEHICLE_MIN_LENGTH_M` (3.0) is the validation floor under it. They are equal today and are not
  the same question: the first decides whether the returns measured a flank, the second rejects a
  fit whose length nothing supports.
- `VEHICLE_FIT_SPLIT_LENGTH_M` (6.0, ask whether this is two vehicles) vs `VEHICLE_MAX_LENGTH_M`
  (14.0, nothing this long drives). Between them a cluster is interrogated and kept; above the
  second it is claimed only if it splits into two real vehicles.
- `VEHICLE_FIT_ONE_FACE_CONFIDENCE` is not a marginal case — a parked car almost always has an
  inferred dimension — so it sets the opacity of the COMMON car, and it was chosen by rendering.
- `VEHICLE_MIN_ASPECT` (2.2) bounds an over-read WIDTH using the length. It only ever narrows.

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
`planner`, `controller`, `navigation`, `parking`, `parking_drive`, `vehicle_fit`} → `aeb` →
{`worker`, `bridge_monitor`} →
{`bev_widget`, `main_window`}; keep the pure/testable layers free
of Qt and BeamNGpy imports (`geometry`, `semantics`, `raster`, `launcher`, `planner`,
`controller`, `navigation`, `parking`, `parking_drive`, `vehicle_fit` and `aeb` currently
import neither — `bridge_monitor` exists as a separate module precisely so `launcher` can stay
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
