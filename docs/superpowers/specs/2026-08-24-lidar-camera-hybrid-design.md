# LiDAR-First Two-Camera Hybrid Mode Design

## Goal

Add a third `HYBRID` instrument mode that keeps the existing six-LiDAR rig
unchanged and adds exactly two live RGB cameras, one just outside each
A-pillar. The existing WORLD, RAW BEV, planning, parking occupancy, automatic
parking, and both AEB systems continue to use only the LiDAR cloud. The
CAMERAS view shows the two new feeds so their physical placement and aim can be
validated live.

The same two RGB feeds become the input contract for a later vision-only
parking-bay experiment. That detector is designed here but is not part of this
implementation.

## Scope

This change includes:

- a persisted `HYBRID` sensor mode beside `LIDAR` and `VISION`;
- a dedicated two-camera A-pillar rig derived from the live vehicle bounding
  box;
- colour-only shared-memory Camera sensors at 10 Hz;
- camera-frame transport through the existing `VisionFrame` signal;
- the existing CAMERAS view enabled for both `VISION` and `HYBRID`;
- a deterministic left/right two-up feed, with the existing click-to-focus
  behaviour retained;
- lifecycle, liveness, and failure handling that cannot disable LiDAR safety
  merely because an auxiliary camera fails;
- offline tests and a live angle/performance acceptance check.

This change does not:

- add camera depth or annotation to the hybrid runtime;
- add camera points to any LiDAR cloud;
- change planner, AEB, parking occupancy, or automatic-parking inputs;
- implement the vision-only parking-bay detector;
- alter the existing eight-camera `VISION` rig.

## Approaches Considered

### 1. LiDAR-first hybrid with separate RGB cameras — selected

The six existing LiDARs remain the primary sensor list. Two separately owned
Camera sensors produce display images only. This preserves the measured LiDAR
range and safety behaviour, avoids camera-depth ghosting, and pays for only two
colour render submissions.

### 2. Merge two camera depth clouds into LiDAR

This could add dense near-field paint points, but it would also reintroduce the
camera mode's registration jitter, overlap ghosts, depth bandwidth, and
semantic-render pass. It would make safety behaviour depend on two differently
timed geometries without being needed for the requested live feed. It is
rejected for the first hybrid mode.

### 3. Reuse the existing `pillar_left/right` cameras

This is the smallest code change, but those mounts are B-pillar cameras at
approximately +/-55 degrees. Selecting them would not implement the requested
A-pillar arrangement and would provide a different near-field view. It is
rejected; the hybrid rig receives new `a_pillar_left/right` mounts.

## Camera Geometry

The rig has stable order:

```text
HYBRID_CAMERA_NAMES = ("a_pillar_left", "a_pillar_right")
```

For a `VehicleGeometry` with left, right, front, and height extents, the mounts
are:

```text
left  = (left_m + 0.12,  -0.25 * front_m, 0.88 * height_m)
right = (-(right_m + 0.12), -0.25 * front_m, 0.88 * height_m)
```

Both cameras pitch down 7 degrees. The left camera yaws 37 degrees left of
straight ahead and the right camera mirrors it. Vehicle forward remains
negative Y. With a 105-degree horizontal FOV, their far-field coverage is:

```text
left:   -15.5 to +89.5 degrees
right:  -89.5 to +15.5 degrees
union:  -89.5 to +89.5 degrees
overlap: 31 degrees around straight ahead
```

The cameras use 1280x960 images. Two cameras make this quality affordable, and
the 4:3 frame retains useful road coverage when pitched down. The requested
update period is 0.10 seconds (10 Hz) for the first live benchmark. The camera
constructor enables shared memory, streaming, and colour only; depth,
annotation, and instance buffers remain disabled.

The positions deliberately sit outside the shell because BeamNG has no
hide-ego option and an inboard windshield experiment returned mostly cabin.
The fractions are generic stations rather than exact mesh landmarks, so the
live camera feed is the acceptance instrument for each vehicle body.

## Runtime Architecture

`SENSOR_MODE_HYBRID = "HYBRID"` is a worker-owned mode with the same request,
confirmation, persistence, and reattach lifecycle as the two existing modes.

The current `_sensors` and `_sensor_names` remain the primary perception set:

- `LIDAR`: six LiDARs;
- `VISION`: eight Cameras, preserving the existing depth-unprojection path;
- `HYBRID`: six LiDARs.

HYBRID adds separate `_hybrid_cameras` and `_hybrid_camera_names` collections.
This deliberately avoids a broad refactor of the mature LiDAR and Vision
paths. It also prevents `_acquire_lidar_cloud()` from calling `.stream()` on a
Camera or `_acquire_vision_cloud()` from treating a LiDAR as a Camera.

Attachment is atomic through the existing `attach_to_player()` try/cleanup
funnel:

1. attach all six LiDARs exactly as LiDAR mode does;
2. derive and attach both A-pillar Cameras;
3. publish `STREAMING` and `sensors_ready` only after all eight sensors exist;
4. on any constructor failure, remove every already-created sensor.

During each HYBRID tick:

1. acquire the six LiDAR streams;
2. run the existing shared perception, WORLD, control, AEB, and parking path
   from those LiDAR chunks only;
3. independently read each Camera with `stream_raw()`;
4. validate and privately copy each `(height, width, 4)` colour buffer;
5. emit a `VisionFrame` containing the images in left/right order.

The `BevFrame.acquisition_fps` remains the LiDAR acquisition rate in HYBRID.
Camera freshness is tracked separately from a sparse colour-buffer digest for
liveness logging; rereading one shared frame does not count as a new frame.

## Failure and Teardown Behaviour

Construction failure is fatal to the attach because a mode advertised as
six-plus-two must not silently become six-plus-one. Runtime camera read failure
is auxiliary: it is caught per camera, reported once until that camera
recovers, and never enters the LiDAR poll-failure budget. LiDAR frames, WORLD,
planning, parking, and AEB continue.

Malformed, missing, or zero-length colour buffers produce no `CameraImage` for
that camera. A five-second no-fresh-camera warning names the known causes:
zero update period, Lowest graphics, or a covered/minimized BeamNG window.
Recovery clears the one-shot failure state and resumes the feed.

The single cleanup funnel removes hybrid cameras and then primary sensors in
reverse construction order, clears camera digests/liveness state, and clears
the GUI through the existing `sensors_stopped` path. Switching among all three
modes reattaches through this funnel; no partial mixed set survives a switch.

The Vision driving gate applies only to `VISION`. `HYBRID` is control-enabled
because its perception is exactly the LiDAR path. Porosity and obstacle-height
logic continue to use the LiDAR/roof geometry rather than an A-pillar eye
height.

## User Interface

The instrument selector becomes `LIDAR | HYBRID | VISION`. HYBRID's tooltip
states that six LiDARs remain authoritative and the two cameras are live-view
auxiliaries.

The CAMERAS visualization is available when the confirmed worker mode is
either HYBRID or VISION and falls back to WORLD only when entering LIDAR. In
HYBRID it receives exactly `A PILLAR LEFT` and `A PILLAR RIGHT`. A two-image
frame lays out as one row and two columns so the aiming overlap can be judged
directly; clicking either tile retains the existing full-frame focus action.

Selecting HYBRID does not forcibly change the current visualization. This
preserves the user's WORLD/RAW BEV choice; CAMERAS remains one click away and
will be selected during the live acceptance run.

## Test Design

All automated tests remain offline and avoid constructing a `QApplication`.
The implementation follows red-green-refactor slices:

1. Geometry tests pin the two names, mirrored A-pillar stations, yaw/pitch,
   105-degree FOV, 31-degree overlap, and 1280x960 resolution.
2. Pure constructor-kwargs tests pin positive 0.10-second updates, shared
   memory, streaming colour, and disabled depth/annotation.
3. Mode tests pin parsing, persistence, repeat no-op, live reattach, and
   controls offered for HYBRID even when the Vision gate is closed.
4. Pipeline tests arm LiDAR and Camera stubs together and prove that one tick
   reads each through the correct API, emits six LiDAR returns into `BevFrame`
   and `PerceptionSnapshot`, and emits exactly two camera images.
5. Safety-isolation tests make camera streams fail or carry contradictory
   depth and prove LiDAR output and control availability are unchanged.
6. Ownership tests mutate a source buffer after the tick and prove the
   `CameraImage` owns a private copy.
7. View tests pin camera availability in HYBRID, fallback in LIDAR, and a
   deterministic left/right two-up layout.
8. Cleanup tests prove both sensor collections are removed and all camera
   freshness state is cleared.

## Live Acceptance

The offline suite cannot prove BeamNG camera placement or renderer cadence.
After tests pass:

1. launch the app and select HYBRID;
2. attach to the player vehicle with BeamNG visible and graphics above Lowest;
3. open CAMERAS and confirm left/right identity, horizon level, body/cabin
   occlusion, centre overlap, near-ground visibility, and approximately
   180-degree frontal union;
4. record camera delivery rate plus LiDAR return counts, poll p50/p95, WORLD
   scene warnings, and simulator FPS for 60 seconds stationary and 60 seconds
   driving;
5. verify AEB arms, LiDAR reach is unchanged, and camera loss does not stop the
   LiDAR feed;
6. tune only mount station, yaw, or pitch from what the two images show. A
   request-rate increase to 15 or 20 Hz comes only after the 10 Hz benchmark
   has headroom.

## Vision-Only Parking-Bay Detector Brainstorm

Here, "vision-only" means the deployable detector receives the two RGB images
plus fixed camera calibration and ordinary ego pose; it receives no LiDAR,
engine depth, or BeamNG semantic annotation at inference time. Its output is
initially an overlay-only set of bay suggestions, isolated from planner, AEB,
selection-to-drive, and automatic parking.

### Approach A: BeamNG oracle and label factory

For offline data collection only, record RGB together with hidden annotation,
depth, and pose. Project marking-labelled pixels onto the ground and run the
existing divider/bay fitter. This establishes the best result the two
viewpoints can support and creates pseudo-labels. It is not a vision-only
runtime because it consumes simulator truth.

This must be the first experiment: if the oracle cannot recover enough of a
bay from the two views, improving RGB recognition cannot fix the geometry.
BeamNG's marking labels also cover generic paint rather than a dedicated
parking-bay class, and some decals label their transparent quad, so a manually
reviewed truth set remains necessary.

### Approach B: classical RGB markings — recommended first runtime

For each camera:

1. normalize brightness and colour;
2. segment likely white/yellow paint using colour, local contrast, and edges;
3. reject texture fragments using line length, width, parallelism, and temporal
   persistence;
4. project accepted pixels into a calibrated near-field ground mosaic;
5. accumulate only fresh observations while the vehicle moves;
6. feed the resulting marking points into an isolated copy/adapter of the
   existing stripe-pair bay fitter.

This is inspectable, fast, and likely adequate for clean nearby bays. Its
weaknesses are worn paint, wet glare, shadows, night lighting, coloured
surfaces, slopes, and parked cars hiding divider ends.

### Approach C: learned segmentation and endpoints

If the classical baseline plateaus, train a small model that predicts three
pixel products from each image: divider paint, head/other markings, and divider
endpoints. Fuse the two cameras after projection into the common ground mosaic,
then retain the same geometric stripe pairing and physical width/depth checks.
This is preferable to an opaque end-to-end bay-box model because the final
geometry remains explainable and testable.

Training data uses the oracle output as a starting label, manually corrected
on a held-out corpus. Splits are by map and scenario rather than random frames
so adjacent video images cannot leak into validation.

### Dataset and evaluation

Capture 0-15 m bays on both front quarters: perpendicular, parallel, angled,
double-sided, partial, empty, and occupied rows across multiple maps,
vehicles, surfaces, slopes, weather, time of day, and approach angles. Include
hard negatives such as lane lines, crosswalks, arrows, kerbs, road seams,
shadows, and broad decals.

Score divider boundary F1 and bay precision/recall, plus centre, yaw, width,
and depth error. Also record false offers per 100 m, distance/time to first
stable detection, slot-count flicker, left/right contribution, inference time,
pair freshness, and suppression time. Initial promotion gates are 95% bay
precision, 80% recall within 12-15 m, 95th-percentile centre error at or below
0.25 m, yaw error at or below 5 degrees, and no control integration until a
substantial manually reviewed corpus meets those gates.

The largest viewpoint limitation is fundamental: forward-outboard A-pillar
cameras cannot see bays wholly beside or behind the car. Temporal accumulation
while approaching or turning is required, and detections are deliberately
suppressed when the pair is stale, vehicle pitch/roll cannot support the
ground projection, or too little divider evidence is visible.
