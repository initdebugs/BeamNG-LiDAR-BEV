# Vision Mode — Project Specification

Status: rung 0 landed 2026-08-22 (camera rig + live grid). Everything below
rung 0 is implemented; everything above it is specified and unbuilt.

This document is the working spec for replacing the LiDAR stack with cameras,
one honest step at a time. It consolidates two research passes: the live
feasibility study of 20 Aug 2026 (measured on BeamNG.tech 0.39.4.0, beamngpy
1.36, RTX 5070 Ti / Ryzen 7 9700X) and the literature/benchmark sweep of
21 Aug 2026 (real-time perception networks, fast BEV/occupancy, sim-trained
networks, classical stereo). Figures below are tagged **[measured]** (taken
live on the target machine) or **[published]** (external benchmark on named
hardware, scaled conservatively).

---

## 1. Goal and philosophy

Build a genuinely camera-driven perception mode alongside the LiDAR mode —
a **ladder**, not a switch:

| Rung | Name | Depth source | Semantics source | Status |
|------|------|--------------|------------------|--------|
| 0 | Camera rig + live grid | none (display only) | none | **DONE** |
| 0.5 | Engine unprojection | engine Z-buffer | engine annotation | specified |
| 1 | Stereo | **computed from image pairs** | engine annotation | specified |
| 2 | Sim-trained networks | stereo | **learned from pixels** | specified |
| 3 | Learned BEV / occupancy | one network, pixels → world | ditto | sketched |
| 4 | End-to-end imitation | pixels → controls | — | wild card |

Each rung swaps ONE component and keeps the rung below as its regression
oracle: the engine's depth/annotation channels can always be rendered in
parallel and diffed, per pixel, per frame. That oracle loop is the project's
structural advantage over any real-world stack and every rung must preserve
it.

Two principles carried over from the LiDAR stack:

- **The perception waist stays `points_world (N,3) + colours (N,3) + state`.**
  Anything that emits that tuple keeps the planner, controller, AEB, BEV and
  WORLD views working. Rungs 0.5+ exist to refill exactly that contract.
- **Safety systems get geometry they can prove, not networks they must
  trust.** AEB range comes from the Z-buffer (rung 0.5) or stereo (rung 1),
  never from monocular depth inference (see §10).

## 2. The camera rig (rung 0 — implemented)

Tesla HW4 layout, eight cameras, derived per vehicle from the live bounding
box by `geometry.derive_camera_rig` exactly as the LiDAR mounts are:

| Camera | Station | HFOV | Looks |
|--------|---------|------|-------|
| `front_main` | windshield header, +8 cm | 50° | ahead (long range) |
| `front_wide` | windshield header, −8 cm | 100° | ahead (context) |
| `front_bumper` | front bumper | 110° | ahead (near field) |
| `pillar_left/right` | B-pillar, just outside shell | 90° | forward-outboard ±55° |
| `repeater_left/right` | front fender, just outside shell | 90° | rear-outboard ±45° |
| `rear` | rear bumper line | 110° | behind |

Facts the implementation is built on, all hit live during the feasibility
work:

- **Vehicle frame forward is `(0, −1, 0)`.** The intuitive `(0, 1, 0)`
  renders the rear seats. Pinned by test.
- **The eight apertures must TILE THE CIRCLE, and the first set did not**
  [measured 2026-08-23]. At Tesla's own 90°/60° with the repeaters aimed 30°
  off rearward — and worse at the 80° pillars originally shipped — the union of
  all eight left a **24.5° hole per side at bearings 95–120°**: over the
  driver's shoulder, precisely the blind spot the repeaters exist for. Reported
  live as the side cameras feeling too narrow, and it is arithmetic rather than
  taste. Widening the pillar alone leaves 20°; both to 90° leaves 5°; 90/90
  with the repeaters re-aimed to **45° off rearward** closes it with a 10°
  overlap either side, at 7.1 px/deg for both at 640 wide. Rearward is
  untouched — the 110° rear camera spans 125–235°. `test_the_rig_leaves_no_gap_
  all_the_way_round` pins the union and fails against the old values (49.8°
  uncovered); a companion requires genuine OVERLAP rather than a hair's-breadth
  join, because the mounts are metres apart and their coverage is not purely
  angular at close range.
- **A centreline camera goes on the BODY centre, never on `x = 0`** [fixed
  2026-08-23]. The reference node is not the body centre — measured 0.16 m off
  on the vivace — so `front_bumper` and `rear` were sitting 0.16 m to one side
  of the car they look straight out of, and the windshield pair straddled the
  wrong axis. This is the third time this exact defect has appeared (the WORLD
  ego model and the AEB corridor were the first two), and it survived because
  the rig's only test vehicle was SYMMETRIC (`left_m == right_m`), which cannot
  distinguish the node from the centre. The outboard pairs never had it: each
  is placed against its own side's surface, so they are symmetric about the
  body already — measured 0.0000 m lateral asymmetry, identical y and z, which
  is the answer to "are the repeaters in the same position" (they are; the raw
  x values look lopsided only because the node is).
- **`pos.z` is referenced to the vehicle ground plane**, same as the LiDAR
  mounts — never add the bbox bottom.
- **There is no hide-ego flag**; a mount inside the glasshouse films the
  cabin (first windshield attempt: 68 % CAR). All mounts sit on or outside
  the shell. Bodywork in frame is correct (a real bumper camera sees bonnet).
- **`field_of_view_y` is VERTICAL**; the horizontal aperture is what the rig
  is designed around, so `camera_vertical_fov_deg` runs the rectilinear
  projection backwards through the aspect ratio.
- **Streaming shared memory is the only viable transport** [measured]:
  poll path 204 ms per 8-camera frame; ad-hoc async ~51 ms/camera; streaming
  ~1.3 ms to read all eight at 640×480.
- **`requested_update_time` must be > 0** [measured]: at 0.0 every streaming
  buffer stays zero-filled while the read loop spins at 88 kHz — a "working"
  rig of black frames. `CAMERA_UPDATE_TIME_S = 0.05`.
- **`stream_raw()` returns a view of the LIVE shared buffer.** The worker
  copies once per camera per tick before anything reads twice.
- **Colour only at this rung** [measured]: annotation is a second full
  geometry pass (sim 42 → 33 Hz); depth doubles the copied bytes. Neither is
  consumed yet, so neither is rendered yet.
- Measured rates on the target machine, 8 cams 640×480 colour: **18.6 Hz per
  camera, sim at 42 Hz**. Resolution is nearly free (1280×960 → 16.2 Hz);
  the cost is per-camera draw submission.
- **The Camera sensor has NO auto-exposure, and the game's own view does**
  [measured 2026-08-23]. Reported live as "washed out". With BeamNG's camera
  placed at the *identical* vehicle-frame pose as the sensor mount, the game
  rendered the scene at mean luma 124.8 with **0.3 %** of pixels at ≥250, while
  the sensor rendered mean 206.7 with **53 %** — so it is the sensor pass, not
  the scene, not the exposure of the world, and not the Qt display path (which
  already paints with `SmoothPixmapTransform` and wraps the buffer as
  `Format_RGBA8888`). The clipping is **hard and per-channel** (front_main: 82 %
  of RED pegged at exactly 255 against 6 % of blue), so no software curve
  recovers it — the detail is gone before the buffer is read.

  **The 53% figure is overstated and the severity claim is withdrawn.** That
  measurement was taken through the wrong vehicle -- `next(iter(get_current()))`
  handed back a `simple_traffic` van facing a sunlit white wall in a car park
  (see the fourth-byte entry above) -- and the game-view half used
  `set_relative`, which was later found not to move the camera reliably. On the
  PLAYER's car, re-measured 2026-08-23 on a normal street: front_main **5.3%**
  blown, front_wide 2.3%, pillar_left 0.2%, repeater_left 5.9%, rear 13.4%,
  means 134-159. The missing auto-exposure is real and the direction of the
  effect holds; the magnitude does not. Re-measure against the game view at a
  verified matching pose before planning around it.

  It is **entirely scene-brightness driven**, which is what makes it workable.
  Same camera, same pose, sweeping time of day:

  | time of day | mean luma | pixels ≥250 |
  |---|---|---|
  | 0.00 (high sun) | 199.7 | 25.3 % |
  | 0.05 | 179.2 | 18.6 % |
  | 0.10 | 122.0 | 14.1 % |
  | 0.15 | 89.5 | 5.2 % |
  | **0.18–0.35** | 65–92 | **0.0 %** |
  | 0.40 | 0.0 | night |

  **Capture below a high sun.** This is not cosmetic: a saturated region has no
  texture, so it is exactly where stereo matching has nothing to correlate —
  the phase-1 kerb experiment and any phase-4 training capture must run at a
  lower sun or they measure the clipper rather than the scene.
- **THE CAMERA BUFFER'S FOURTH BYTE IS NOT OPACITY, and reading it as one is
  what the reported "noise" actually was** [measured 2026-08-23]. On one
  1280x960 frame alpha ran **40..255 with only 50.75% of pixels at 255**, and
  what is in there tracks the scene's materials. `vision_view` declared
  `Format_RGBA8888`, so Qt composited every pixel against the dark tile
  background: wherever that byte dipped the tile showed through, which renders
  as black speckle on textured surfaces, dark outlines around buildings, and a
  clean sky (alpha 255 there). Painted against the true colour, mean error
  **26.08 as RGBA and 0.00 as `Format_RGBX8888`** -- identical byte layout,
  fourth byte ignored.

  **It survived a long investigation for two reasons, both worth remembering.**
  Every probe saved `rgba[..., :3]` and threw the channel away, so captured
  PNGs were clean while the app was not -- exposure, anti-aliasing, SSAO and
  tile downscaling were each measured, and none of them was this. And the
  capture tool picked its vehicle with `next(iter(get_current()))`, which
  returns an arbitrary actor: it photographed `clone2`, a `simple_traffic` van
  parked in a car park, while the player sat on the street being complained
  about. **A probe must select the player with `get_player_vehicle_id`, exactly
  as the worker does, or it is not looking at the same thing the app is.**

  A 1x1 image blends under BOTH formats and `pixelColor` reports the stored
  byte under both, so neither shows the difference; only painting a multi-pixel
  image does. `test_the_cameras_fourth_byte_is_not_treated_as_opacity` pins it
  that way, with a guard asserting RGBA still blends.
- **The sensor pass gets no anti-aliasing, and the main view does**
  [measured 2026-08-23]. `GraphicAntialias` is 4 for the game; the Camera
  sensor has no equivalent and no temporal accumulation. Against a 2x
  supersampled render of the same view, the native frame differs by >8 levels
  on **5.6 %** of pixels — edges: poles, railings, kerb lines, car outlines.
  Rendering 2x and area-averaging down fixes it visibly, and is **not
  affordable**: 8 cameras at 2560x1920 cost 18.7 ms a tick to copy and 305 ms
  to box-average in numpy, against a 40 ms tick. If it is ever wanted, it has
  to happen on the GPU or for one camera at a time, not per-rig on the CPU.

  Reported live as "so much noise in the camera compared to BeamNG itself".
  Three contributors, separated by measurement — and the ranking matters
  because only one of them is worth acting on:

  | contributor | evidence | lever |
  |---|---|---|
  | **exposure** (dominant) | the same asphalt is smooth at a low sun and harsh speckle at a high one; high-freq energy 5.86 → 7.85 | sun angle (above) |
  | no AA | 5.6 % of pixels differ from a 2x render | none affordable |
  | tile downscaling | grid tiles only, see below | fixed |

  **SSAO is NOT a contributor**, and it took two attempts to establish that.
  It is the obvious suspect — it ships at `quality: 0`, the fewest samples —
  and the first test toggled it through `VariableRegistry.set` and showed
  nothing, which was worthless because that write is INERT (see below). Re-run
  through the mechanism the game's own options handler uses, with the toggle
  verified by reading `isEnabled()` back as `false`: grain 4.462 → 4.316 with
  SSAO fully off, and 4.492 at `setSamples(64)`. It is genuinely not the cause.

- **`VariableRegistry.set` DOES NOT APPLY, and this invalidated three
  experiments before it was noticed** [measured 2026-08-23]. The game's own
  settings handler (`lua/ge/extensions/core/settings/graphic.lua`) writes the
  registry *and then* acts on the scene object — and it is the second half that
  takes effect:

  ```lua
  VariableRegistry.set('$SSAOPostFx::Enable', value)   -- decorative
  scenetree.findObject("SSAOPostFx"):disable()         -- this is the one
  ```

  For `$pref::Video::defaultAnisotropy` the write does not even stick: set 16,
  set 0, read back 8 every time. **Any postfx or graphics experiment must
  verify its own lever before believing a null result** — read the value back,
  and prefer the scenetree object's own methods. The HDR sweep above is
  suspect for exactly this reason and would need re-running through
  `scenetree.findObject("HDRPostFx")` before its null result means anything.

  Note the asphalt aggregate is REAL texture, not noise — a 2x render resolves
  *more* of it, so a high-frequency-energy metric goes UP when the image gets
  better. Judge this one by eye against a supersampled reference; an energy
  number will rank it backwards.
- **Grid tiles must be resampled before they are drawn, and the focused view
  need not be** [measured]. `QPainter.drawImage` under `SmoothPixmapTransform`
  is bilinear at any scale, which undersamples on minification. Against an
  area-averaged reference: at the focused pane's 0.7x it is already correct
  (error 1.25 either way, **0 %** gained), but at a grid tile's 3x it is 10x
  wrong (4.33 against 0.40, **91 %** better prescaled). `vision_view.
  wants_prescale` therefore resamples only when shrinking, cached per camera
  and target size because `QImage.scaled` is ~1 ms a tile and eight of them
  every paint is not affordable.
- **The HDR post-processing block does not reach either renderer, so exposure
  cannot be fixed from there** [measured]. `$HDRPostFX::enableToneMapping`,
  `keyValue` and `whiteCutoff` were set through `VariableRegistry` and read back
  changed, with `extensions.client_postFx.reloadPostEffects()` after each — and
  neither the sensor nor the game view moved by more than noise. Note BeamNG's
  own default preset ships `enableToneMapping: false`, so the main view's
  correct exposure comes from somewhere else. Do not repeat this sweep without
  first proving the toggle moves the GAME view: an earlier pass concluded "the
  sensor ignores post-processing" from toggles that turned out to be inert on
  both renderers, which proves nothing.

### GUI (implemented)

A third header view, **VISION**, beside WORLD and RAW BEV. Unlike that pair
it is not GUI-only: it selects the instrument set, so switching while
streaming re-attaches through the worker's single teardown funnel. The view
is a labelled live grid (`vision_view.VisionView`); the grid arithmetic
(`grid_dimensions`) picks rows × columns to maximise cell area at the
cameras' aspect. Clicking a tile focuses that camera full-frame and any
click returns to the grid (`toggle_focus`, pure and pinned); focus is
tracked by camera name so a rig change dissolves it harmlessly. ACQUISITION counts genuinely new frames (per-camera buffer
digest), DISPLAY the paint rate; VISIBLE POINTS reads "—" honestly.
Self-driving and both AEBs are refused by the worker and not offered by the
GUI in this mode (they consume the LiDAR cloud; rung 0.5 restores them).

### Rung-0 live checklist (offline suite cannot reach these)

- The `Vision check:` line reports first fresh frames within ~1 s of attach;
  the 5 s silence warning never fires on a healthy setup.
- All eight tiles show plausible views (no cabin interiors, no sky-only
  tiles); some bonnet in `front_bumper`/`front_wide` is expected.
- Sim rate while streaming stays near the measured 42 Hz; ACQUISITION near
  16–18 Hz.
- The tiles are exposed, not blown white. Under a high sun a quarter to a half
  of every frame clips (see the exposure fact above); lower the sun before
  judging image quality or capturing anything.
- Switching VISION ↔ WORLD/RAW BEV mid-stream re-attaches cleanly both ways,
  and self-driving/AEB re-arm on return to LiDAR.
- Graphics preset above "Lowest" (empty camera buffers otherwise) and
  `PostFXMotionBlurEnabled=false` for capture-quality frames.

## 3. Rung 0.5 — engine unprojection (the previous report's "recommended")

Turn on the depth (+ annotation) channels and rebuild the perception waist:
per camera, stride-subsample the depth image, unproject through a precomputed
per-pixel ray LUT, sample annotation at the same pixels, concatenate all
eight into `points_world` + `colours`. The whole downstream stack — planner,
AEB, semantics, BEV, WORLD — resumes working, now fed by cameras.

Key facts and decisions:

- **Depth decodes as `raw_float32 × far_plane`, linear metres** [measured:
  10 m read 9.65, 25 m → 24.17, 50 m → 49.51]. It is planar Z, not radial
  range — the cosine divide per pixel is mandatory or the ground bows.
- Construct cameras with `integer_depth=False` (default True quantises to
  0–255 — silently) and `postprocess_depth=False` (a 256-iteration Python
  loop per frame); read via `stream_raw` and cast.
- **Subsample the image, not the cloud**: 8 × 214×160 ≈ 274 k points ≈
  today's budget; naive full-res is 2.46 M points and ~30 MB per snapshot.
- Annotation returns the same palette `semantics` already matches — but
  lane-marking classes are map-dependent (east_coast_usa bakes paint into
  the road material [measured]); the `Marking check:` line already reports
  this per map.
- The per-camera frames are staged "a frame or two" apart with no
  timestamps; at 16 Hz and 30 m/s that is ~1.9 m of ego motion between
  cameras. Mitigation: per-camera ego-motion compensation by estimated age,
  and tolerance in the accumulation stores; measure before trusting AEB.
- Acceptance: BEV/WORLD indistinguishable in character from LiDAR mode;
  `Sensor reach:`-equivalent line per camera; AEB fires on a wall (its live
  checklist re-runs entirely — new sampling distribution).

## 4. Rung 1 — stereo: vision-only geometry, no networks

Re-pair the rig into stereo pairs with exact baselines (free in sim, denied
to real cars) and compute depth from image pairs. Engine depth demotes to a
hidden diff oracle.

- **Matcher**: CUDA semi-global matching (libSGM class): **1.5–2 ms per pair
  at ~VGA with subpixel on an RTX 3080** [published]. Four pairs at 16 Hz ≈
  13 % of the GPU. Learned stereo (CGI-Stereo 29 ms, LightStereo 17 ms on a
  3090 [published]) does NOT scale to 4 pairs — optional front-pair-only
  upgrade at reduced rate.
- **Baseline arithmetic** (error = δ·z²/(b·f)): at 1280×960 / 60° HFOV /
  δ=0.25 px, a **0.56 m baseline holds error < 1 m at 50 m** — the AEB
  spec, met with classical matching. At 640×480 the required baselines are
  roof-rack-sized; the front pair therefore runs 1280×960 narrow-FOV.
- **Kerbs**: with a good front pair, per-point height noise at 25 m ≈
  1.7 cm — a 10 cm kerb is ~6σ per point — but published stereo curb work
  rarely exceeds 20 m, matching windows blur 3–5 px kerb faces, and
  per-cell temporal accumulation (the existing cell-referenced machinery) is
  mandatory, not optional. **The kerb experiment (§9) gates this rung.**
- **Hills**: v-disparity road-profile fitting / multi-layer stixels
  [published: >400 FPS stixel computation on a Titan X; Daimler drove 100 km
  on stereo in 2013] replace any flat-ground assumption. IPM is disqualified
  (a 3 % grade maps 30 m to 75 m).
- **New phantom class**: stereo smears depth at object silhouettes into
  coherent radial streaks that SURVIVE despeckle/support (they are dense and
  correlated, not speckle). Standard fix: cull points within N px of a depth
  discontinuity, plus a per-point range-variance channel feeding the cell
  stores. The AEB phantom checklist re-runs after this rung, always.
- Safety re-derivations flagged by the port review: `_porous` (ray-shadow
  model keyed on roof-LiDAR height) and `ground_rise` (fits by range ring)
  assume LiDAR sampling and must be re-derived for camera-frustum sampling
  (density falls as 1/r², no stripes). `vehicle_fit` clusters in the LiDAR's
  polar lattice and needs rewriting.
- Acceptance: car drives with engine depth OFF; oracle diff (stereo vs
  Z-buffer) within budgeted error bands at 10/25/50 m; AEB fires on a wall
  and a stopped car; phantom checklist clean on hills, brake dive, bushes,
  reverse.

## 5. Rung 2 — small networks trained on BeamNG's own labels

Replace the annotation channel with learned semantics; keep stereo geometry.
"Geometry from stereo, meaning from networks" is how most non-Tesla stacks
actually work.

- **Data is free and near-instant**: lockstep capture (`pause() → step(N) →
  poll all 8 → write`) yields perfectly aligned colour + depth + annotation.
  At 80 labelled frames/s, a VKITTI-scale dataset (~21 k images) is ~5
  minutes of driving [published/measured]. Watch storage: 8×640×480×3
  channels at 10 fps ≈ 295 MB/s.
- **In-domain synthetic segmentation is demonstrably easy** [published]:
  ~84 % mIoU with a vanilla net on SHIFT's clear-day split; one map + one
  renderer is narrower still. Semi-supervised results reach ~76–80 % mIoU
  from 100 labels — and here labels are unlimited.
- **Architectures**: PIDNet-S / DDRNet-23-slim / EfficientViT-B0 class
  (0.7–7.6 M params). Batched across 8 cameras ≈ **2.5–5 ms** on the target
  GPU [published, scaled]. Add YOLO-nano-class 2D detection (~2.5–4 ms ×8
  batched) and UFLDv2 lanes on front cameras (~1 ms) — the full stack fits
  ~6–9 ms.
- **Batching is mandatory**: one batch-8 TensorRT engine per task per tick;
  8 separate launches waste milliseconds under WDDM scheduling.
- **Windows/WDDM constraint**: no CUDA MPS; render + inference are
  time-sliced, so frame time is render **plus** inference. Budget ≤ 10–15 ms
  total and measure sim-rate impact first.
- Lane paint becomes visible on EVERY map (read from colour, not from the
  per-map annotation lottery) — a strict capability gain over LiDAR.
- Training budget: hours on the 5070 Ti per net [published: Monodepth2-class
  8–15 h on a 2017 Titan Xp]. TensorRT FP16 export path required.
- Acceptance: A/B against the annotation channel live (mIoU dashboard tile);
  driving character unchanged with semantics swapped; unannotated-map
  behaviour no worse than the geometric fallback today.

## 6. Rung 3 — learned BEV / occupancy (sketch)

The fast end of the Tesla-style family fits the budget, revising the first
report's blanket "impossible":

- **FlashOcc M0**: full 6-camera occupancy pipeline **5.1 ms FP16 / 2.5 ms
  INT8 on an RTX 3090** [published]; ~40 M params; mIoU ~30–32.
- **BEVDet-R50 TensorRT**: 7.2 ms on a 3090; INT8 13.8→7.8 ms on an A4000
  [published]. Backbone ≈ 65 % of cost, scales ×1.33 for 8 cameras.
- **CVT** (cross-view transformer BEV segmentation): simplest to train from
  scratch, 28.5 ms PyTorch on a 2080 Ti → plausibly ~10 ms TensorRT here;
  calibration-aware, which suits per-vehicle rigs.
- Training data: thousands of frames suffice in-domain [published: CARLA BEV
  datasets of 4–7 k samples; nuCarla proves the recipe at scale]. 3D voxel
  labels can be synthesised from the accumulated LiDAR voxel store — the
  LiDAR stack becomes the label factory.
- Still ruled out: SparseOcc (~50 ms), FastOcc (80 ms), GSD-Occ (50 ms),
  SurroundOcc/TPVFormer/UniAD (0.3–0.6 s).
- Decision point deferred until rungs 1–2 exist: by then the capture
  pipeline, deployment tooling and contention measurements make this a
  choice made with numbers.

## 7. Rung 4 — the wild card (recorded, not planned)

End-to-end driving is compute-cheap (TCP: one camera + ResNet-34 topped the
CARLA leaderboard; trivially < 10 ms here [published]) but data-hungry and
un-debuggable, and it outputs steering rather than an inspectable world. The
interesting variant: **imitation-learn from the existing LiDAR planner as
privileged teacher** (record camera frames beside its controls). A side
experiment at most; it must never replace the legible pipeline.

## 8. Architecture (as implemented at rung 0, extended per rung)

- **Sensor mode** is worker-owned state (`SENSOR_MODE_LIDAR` /
  `SENSOR_MODE_VISION`): GUI requests via `sensor_mode_requested`, worker
  confirms via `sensor_mode_changed`, and a mid-stream switch re-attaches
  through `attach_to_player` — the single funnel, so a half-swapped rig
  cannot exist.
- **Module layout** keeps the one-directional rule: `config → models →
  geometry → worker → vision_view/main_window`. `vision_view` is Qt-only and
  BeamNGpy-free; the grid arithmetic is a pure function. Rung 0.5 adds a
  pure `unprojection.py` (config/models/numpy only); rung 1 a `stereo.py`
  wrapping the CUDA matcher behind the same points-out contract.
- **Data contracts**: `CameraMount` (per-camera optics, vehicle frame),
  `CameraImage`/`VisionFrame` (display), and — from rung 0.5 — the existing
  `points_world + colours + state` waist unchanged.
- **Timing contracts preserved**: prefetched state poll (socket safety by
  construction), time-based poll-failure budget shared via
  `_note_poll_failure`, a frame emitted every tick, acquisition counting
  genuinely new data only.

## 9. The kerb experiment — run before building rung 1

The single highest-leverage measurement, with veto power over the stereo
design: one wide-baseline front pair (1280×960, ~60° HFOV, b = 0.5–1.0 m),
one 0.10–0.15 m kerb, ranges 15–30 m, accumulate into the existing 0.4 m
cell machinery, and answer: does the kerb face separate from the road at
25 m? If yes, rung 1 proceeds as specified. If no: hybrid fallback — stereo
for obstacles/AEB, engine depth for the ground band — decided then, not
after months of porting.

## 10. Known dead ends (do not revisit without new evidence)

- **Monocular depth networks for AEB range**: best AbsRel ≈ 0.07–0.15 ⇒
  ±2–7.5 m at 30–50 m against a ~1–2 m requirement, with per-scene
  systematic bias [published]. Latency also fails ×8.
- **Motion parallax as primary forward range**: unobservable at the focus of
  expansion — dead ahead, where AEB threats live — and co-moving objects
  reconstruct at infinity.
- **Time-to-contact from optical flow as the AEB metric**: no distance, and
  pitch-sensitive exactly like the phantom classes this project fought.
- **Flat-ground IPM as a metric representation**: +45 m longitudinal error
  at 30 m on a 3 % grade.
- **Tesla's literal 36 Hz**: transport caps at ~16–18 Hz per camera
  [measured]; all budgets assume 16 Hz.
- **Visual lane paint from the LiDAR sensor**: retired 2026-08-10 (the
  unannotated channel is a range rainbow). Cameras make it moot.

## 11. Cross-cutting risks

1. **GPU contention is the unmeasured number.** Every published latency
   assumed an otherwise-idle GPU; here BeamNG renders, eight camera passes
   render, and inference shares under WDDM. Arithmetic says ~2× headroom at
   rungs 1–2; only a live measurement says it truly fits.
2. **The licence clock** (tech.key timestamp ~2026-09-22, unconfirmed as an
   expiry) predates all of this and blocks everything if real.
3. **Per-camera frame skew** (~1.9 m at speed) has no timestamps to correct
   from; rung 0.5 must characterise it before AEB trusts unprojected clouds.
4. **Constant re-derivation is the real cost** of every rung: code that is
   textually reused but whose constants were fitted to LiDAR sampling is
   re-derived, not reused. Budget accordingly (previous estimate: 3–4 months
   for the base port; rung 1 +3–6 weeks; rung 2 +4–8 weeks; rung 3 +2–3
   months).

## 12. Test strategy

- **Offline** (this suite, no sim, no QApplication): rig geometry and FOV
  arithmetic, grid layout, worker mode machinery, copy-not-view invariant,
  freshness accounting, mode refusals — all pinned in `test_vision_mode.py`
  and `test_view_selection.py`. Rung 0.5+: unprojection arithmetic against
  synthetic depth images (planar-Z cosine divide, ray LUT), stereo geometry
  against synthetic disparity, oracle-diff harnesses.
- **Live checklists** (per rung, in this file and CLAUDE.md): rung 0's list
  in §2; every later rung re-runs the AEB phantom checklist because each
  changes the sampling distribution that feeds it.
