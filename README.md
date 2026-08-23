# BeamNG LiDAR BEV

A Windows desktop app for BeamNG.tech 0.39.x (via beamngpy 1.36) that launches
the simulator with the BeamNGpy communication bridge enabled, connects after
the map and vehicle are loaded, attaches a semantic LiDAR set — or, in Vision
mode, a Tesla-style eight-camera rig — to the current player vehicle, and
renders the output as a reconstructed autonomous-driving world, an EGO-fixed
diagnostic bird's-eye view, or a live camera array.

beamngpy 1.36 speaks bridge protocol v1.27, which is BeamNG.tech 0.39.x; it
cannot connect to 0.38.x or earlier. The install path lives in
`config.BEAMNG_EXE`.

## Start

1. Run `install_dependencies.bat` once.
2. Run `run_app.bat`.
3. If BeamNG.tech is not already running, select **Launch BeamNG.tech**.
4. Load a map, spawn/select the intended EGO vehicle, then select
   **Attach to Player Vehicle**.

The app probes for a running BeamNG.tech bridge every two seconds, so a session
you started yourself — from Steam, a shortcut, or an earlier run of this app —
is picked up automatically and **Attach** becomes available without pressing
Launch first. Launch is disabled while a session is already detected, so it
cannot start a second instance fighting for the same port.

The Launch action starts the configured executable immediately with the required
`-tcom` and `-tport` arguments. The Attach action performs the BeamNGpy
connection later, after the map and vehicle are ready. If BeamNG.tech
disappears while streaming, the app releases its sensors and returns to
OFFLINE by itself. Closing the BEV app removes its sensors and disconnects
BeamNGpy; BeamNG.tech stays open.

The window automatically switches to a compact toolbar below 980 px wide and
hides secondary metrics below 720 px. It can be reduced to 640 x 560 for a
laptop split-screen layout, and its last size and screen position are restored
the next time it opens.

The visualization header switches between:

- **WORLD** — the default Tesla-style 3D reconstruction with a smooth road
  surface, generic traffic vehicles, the selected path, speed, and autonomy
  state.
- **RAW BEV** — the original top-down semantic point cloud with range rings,
  planner candidates, sensor mounts, AEB corridors, and performance metrics.
- **VISION** — the eight-camera rig (Tesla HW4 layout: windshield main + wide,
  front bumper, two B-pillars, two fender repeaters, rear) streamed live as a
  labelled grid; click a camera to view it full-frame, click again to return.
  This is rung 0 of the vision-only ladder; see `docs/VISION_MODE_SPEC.md`
  for the spec and `docs/VISION_ROADMAP.md` for the development roadmap.

Switching between WORLD and RAW BEV is instantaneous and does not restart
sensors or change driving state. Switching to or from VISION swaps the
instrument set on the car, so it re-attaches the sensors; self-driving and
both AEB systems need the LiDAR point cloud and are unavailable in Vision
mode at this rung. If Qt Quick 3D cannot initialize, the app falls back to
RAW BEV while the sensor and control loops continue normally; VISION has no
3D-renderer dependency and keeps working.

## Sensor Configuration

| Setting | Value |
| --- | --- |
| Placement | Front, left, right, and rear, computed from the live vehicle bounding box |
| Height | 0.20 m above the bounding-box tire/ground plane |
| Body clearance | 0.08 m beyond the applicable bumper or side |
| RAW BEV display radius | 105 m (sensors request 120 m of slant range) |
| Horizontal view | 170 degrees per sensor, static |
| Vertical view | 30 degrees, 256 channels |
| Requested update rate | 30 Hz |
| Transport | Local shared memory |
| Output | Semantic point cloud |

The 0.20 m mount height is deliberate. A sensor this low grazes the road
surface, so a 0.10–0.15 m kerb both breaks the height profile and casts a long
occlusion shadow. A roof-height mount looks down onto kerbs, never sees their
face, and casts no shadow.

Reach is recovered through channel count rather than by raising the sensor.
Ground coverage from a low mount is set by the channel closest to horizontal:
32 channels across 30 degrees leaves the nearest channel at 0.48 degrees, whose
ground intersection is only 24 m away. 256 channels put it at 0.06 degrees,
which carries past 100 m while the bottom channel still lands at 0.75 m.
Measured live, raising the channel count does not increase the total point
count — `density` sets the ray budget and the channel count decides how those
rays are spread vertically.

Each sensor uses 170 rather than 179 degrees. Both are accepted, but 179 sits
on the depth pre-pass's rectilinear `tan()` cliff: measured 7,389 returns of
which only 3,053 were unique, against 7,826 / 5,613 at 170 degrees. Four
170-degree wedges 90 degrees apart still cover a full circle with overlap.

`max_distance` is a slant range from each sensor, while the display cull is a
horizontal distance from the vehicle reference node. A sensor sits up to ~2 m
from that node, so the two constants are deliberately different.

## BEV Classification

BeamNG's semantic annotation palette is read from the running simulator.
`STREET`, `RESTRICTED_STREET`, `ASPHALT`, `COBBLESTONE`, lane markings, and
crossings are remapped to grey. Cars, buildings, poles, walls, sidewalks,
guardrails, vegetation, and all other known non-road classes are remapped to
red.

Community maps are not always fully annotated. For unknown or `BACKGROUND`
labels only, a narrow ground-height fallback keeps low road-surface returns grey
and raised geometry red. Known grass, sidewalk, and other non-road annotations
remain red regardless of height.

## WORLD Reconstruction

WORLD deliberately uses a hybrid perception contract. LiDAR determines the
visible road, boundaries, uncertain structure, and whether a traffic actor has
actually been observed. BeamNG supplies stable identity, pose, velocity, and a
generic visual type for nearby traffic vehicles. A simulator vehicle is not
drawn until semantic LiDAR returns corroborate its footprint; brief missed
returns coast and fade rather than teleporting or persisting forever.

A bounded 0.5 m world-space surface grid retains just over one second of road
returns. This removes per-scan flicker while still clearing quickly after
occlusion, a map load, or a teleport. The selected blue path is sampled from
the same planner geometry used by RAW BEV, so the two views cannot disagree
about the intended turn.

## Performance

The sensor manager is configured for 30 Hz and the view is scheduled at 40 ms.
All simulator I/O and point-cloud transforms run on a background Qt thread.
WORLD surface construction runs on a second worker thread that keeps only the
latest pending snapshot; it cannot build a backlog or delay vehicle control.
Qt Quick 3D interpolates camera and actor motion on the GUI thread.
The app displays the measured acquisition and display rates; these can fall
below 30 Hz if BeamNG.tech cannot render four annotated LiDAR passes fast enough
on the current map and graphics settings.

The frame budget is dominated by the per-frame vehicle state fetch, measured at
32.7 ms (p95 35.3 ms) against 0.54 ms for all four shared-memory reads
combined. That is why the tick is 40 ms rather than 33 ms.

If the LiDAR load is too heavy for your machine, lower `LIDAR_DENSITY` in
`config.py` — it is a sparsity divisor, so raising it thins the cloud. Measured
across all four sensors: 49,708 points at 50, 99,736 at 25.

A frame is emitted on every tick even when no sensor returns anything, so the
view and the metrics stay live instead of freezing on the last good frame.

## Development

```powershell
py -3.12 -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
py -3.12 -m pytest
py -3.12 -m ruff check src tests
```

Runtime logs are written to `logs/beamng_lidar_bev.log`.
