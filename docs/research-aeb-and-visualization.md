# Research: AEB & visualization improvements

Research and brainstorming only — no code was changed. Each section states what the code
actually does today (with file references), why the reported symptom happens, what the
options are, and where the difficulties lie. Anything that would need a live simulator
measurement is flagged, following the house rule that pixel/sensor questions get measured,
not guessed.

---

## 1. Ego vehicle size — Vivace fine, D-Series phantom-brakes

### What the code does today

The claim "it does not take the size into account" is only *half* true, and the half that
is true is not the half you would guess.

**What already scales with the vehicle** (derived live from the bounding box at attach,
`geometry.derive_vehicle_geometry`):

- Corridor half-width: `geometry.width_m / 2 + AEB_CLEARANCE_MARGIN_M` (`aeb.py:273`)
- Longitudinal standoff: `geometry.front_m + profile.standoff_m` (`aeb.py:272`), mirrored
  for the rear system
- The roof sensor's aperture and every mount position

**What does NOT scale — four fixed quantities, all measured on one car** (almost
certainly the Vivace, given that is the car that behaves):

1. **`AEB_BRAKING_DECEL_MPS2 = 10.0` and `AEB_REVERSE_BRAKING_DECEL_MPS2 = 6.8`.**
   The braking tables in `config.py` (11–200 km/h) imply 1.02–1.07 g — sports-hatch
   figures. A D-Series is a heavy body-on-frame pickup; it will not hold 1 g. The config
   comment itself warns: *"Re-measure for a heavy or low-grip vehicle... there the brake
   would fire too late."* Note the direction: a wrong (too high) decel makes AEB fire
   **late**, not early — so this is not the phantom-braking cause, but it is a real
   safety gap on the D-Series, and it also mislocates the on-screen brake-now bar and the
   "X m before it acts" numbers, which corrupts the driver's read of the system.

2. **`AEB_OBSTACLE_MIN_HEIGHT_M = 0.30` is fixed.** This is the most likely cause of
   "thinks I'm going to hit stuff while I'm not." The 0.30 m floor answers "what would be
   a crash *for the measured car*" — and it was explicitly sized to clear the measured
   car's brake-dive (~5 cm) plus camber. Two things go wrong on a D-Series:
   - *Ground clearance.* A pickup straddles or rolls over things around 0.30 m — tall
     kerbs, berms, ruts, off-road terrain steps — that would total the Vivace. The truck
     driver correctly believes they will clear them; AEB counts them as crashes.
   - *Dive/heave headroom.* Heights are gravity-referenced, so pitch itself does not tip
     the cloud — but `ground_z_vehicle` is a body-fixed constant captured **once at
     attach** at static ride height (`geometry.py:163`). Suspension heave moves *every*
     height by the travel amount. A soft, long-travel truck (especially unladen at the
     rear, or loaded in the bed) heaves much more than 5 cm, eating most of the margin
     that 0.30 m was sized to provide. Combined with camber and a rough surface, road
     returns can cross the floor — and `AEB_CONFIRM_S = 0.12` is short enough for a
     washboard section to satisfy.

3. **The bbox width includes everything bolted to the truck.** Corridor half-width comes
   from the full OOBB width. If the D-Series bbox includes its large mirrors (or wide
   arches), the corridor sweeps a band wider than anything that would actually be a
   crash, so kerb faces and parked cars beside the real path fall inside it. A kerb face
   easily supplies the `AEB_MIN_HITS = 4` returns.
   **Live check available today:** the `AEB check:` log line at arm already prints
   `geometry.width_m` — compare it against the D-Series' real body width (~1.9 m). If the
   log says 2.1–2.3 m, mirrors are in the corridor.

4. **The swept shape is a rigid ribbon around the reference-node path.** A long vehicle
   off-tracks: the rear cuts inside the arc, the front overhang sweeps outside it. At the
   small yaw-derived curvatures AEB sees at speed this is centimetres; at parking-speed
   reverse arcs (where the rear system lives) it is tens of centimetres. Minor, but on
   the list.

### Possibilities

- **Per-vehicle braking profiles.** A small registry keyed on the vehicle model name
  (the worker knows what it attached to), holding measured `braking_decel`,
  standoffs, and an obstacle floor. Default entry = today's constants. Cheapest
  correct-by-construction fix; the cost is measuring each car once (the repo already has
  a measuring methodology — the tables and their least-squares fit).
- **Auto-calibration.** The controller already has a trim integrator that "learns the
  vehicle"; the same philosophy applies here: whenever a full brake is applied (AEB event
  or hard manual stop), measure achieved deceleration from the speed trace and update the
  plant estimate (down only, or slowly in both directions, clamped). Handles wet roads
  and loaded beds for free. Difficulties: needs grade compensation (braking downhill
  under-measures — usable via the pitch/`state["up"]` signal), needs a conservative
  starting default before the first hard stop, and a conservative default *is* early
  firing — the exact complaint — so the default matters.
- **Scale the obstacle floor with the vehicle.** The honest input would be ground
  clearance / approach geometry, which the bbox does not carry (its bottom is tire
  contact). Options: per-vehicle profile entry (simplest, pairs with the registry
  above); a proxy from bbox height (crude: floor = 0.30 for cars, ~0.40–0.45 for
  anything over ~1.8 m tall); or probing wheel/suspension node positions via BeamNGpy if
  available (needs research on what `get_node_positions`-style APIs return per vehicle).
- **Mirror allowance.** Either subtract a fixed allowance from the bbox width for the
  corridor (with a floor at track width), or put true body width in the per-vehicle
  profile. Needs the live width check first to confirm the bbox actually includes
  mirrors.
- **Track heave live.** Re-derive the ground reference continuously from the returns
  directly around the ego (the r≈0 end of the ground estimate the planner already
  computes) instead of the attach-time constant. This removes the heave term for every
  vehicle at once, shrinking how much of the 0.30 m budget is spent on suspension travel.
  Care needed: this reference feeds everything (planner, AEB, semantics), and a bad
  estimate under the car (e.g. straddling a kerb) moves the whole world — it wants heavy
  filtering and a clamp against the attach-time value.

### Difficulties

- Any per-vehicle table is stale the moment a new vehicle/mod is driven; auto-calibration
  or conservative derivation has to back it up.
- Lowering sensitivity for the truck must not lower it for the hatchback — everything
  should key off measured/derived vehicle properties, never a global constant tweak.
- The phantom-brake checklist in CLAUDE.md has to be re-run per vehicle class after any
  of this; the offline suite cannot reach it.

---

## 2. Point-cloud memory: time-based today, location-based instead?

### What the code does today

Diagnosis confirmed — the observation is exactly right. `WorldSceneAssembler` expires by
wall-clock age against `snapshot.timestamp` (`time.monotonic()` from the worker):

- Road cells: `WORLD_CELL_TTL_S = 1.2` (`world_scene.py:1053`)
- Static boundary voxels: `WORLD_COLUMN_TTL_S = 4.0`
- Vehicle-classified voxels: `WORLD_VEHICLE_TTL_S = 0.15` (`world_scene.py:1241`)

The detail you see while moving is the ego motion sweeping the LiDAR ground rings and
azimuth stripes through the world; the store accumulates a 1.2 s / 4 s window of that
sweep. Stop the car and the sweep stops, the window drains, and after 1.2 s (road) / 4 s
(walls) the display collapses to what a single stationary frame resolves — concentric
arcs with gaps. The TTL was really a proxy for "how much sweep is in the window," which
is why it behaves badly at zero speed.

**Key insight that makes the fix safe:** ego pose comes from the simulator and is ground
truth. There is no odometry drift, which is the reason real-world mapping systems *must*
decay old data. Here, static geometry accumulated a minute ago is exactly as valid as
geometry from this frame — provided moving things are kept out of it, and they already
are: vehicle returns are class-separated with their own short TTL.

### Possibilities

1. **Motion-gated clock (smallest change).** Advance a "scene clock" only while the ego
   moves (or scale `dt` by speed). Stopped ⇒ nothing static expires. A few lines: expiry
   compares against an accumulated travel-time instead of `snapshot.timestamp`. Traffic
   voxels must stay on the wall clock — a car crossing in front of a *stopped* ego must
   still fade in 0.15 s, so the per-class TTL split becomes a per-class *clock* split.
2. **Distance-stamped expiry (the actual "based on location" ask, recommended).** Stamp
   every cell/voxel with the ego's odometer reading at write; expire when
   `odometer_now − odometer_seen > D` (road ~20 m, static ~70–100 m). At cruising speed
   this reproduces today's behaviour almost exactly (1.2 s at 50 km/h ≈ 17 m); at
   standstill nothing expires. It is the principled version of option 1 and costs the
   same: one float per entry, an odometer accumulated in the worker (it already computes
   speed each tick), and the comparison swapped. Radius culling and
   `WORLD_POSE_JUMP_RESET_M` stay as-is and keep memory bounded.
3. **Expiry by contradiction (free-space carving) — the long-term correct model.** Keep
   static geometry indefinitely; delete a voxel only when a newer ray *passes through*
   it without a return. This is classic occupancy-grid negative evidence. It buys
   minutes-long persistence, removes ghosts of things that moved (a parked car that
   departs), and is the same machinery occlusion reasoning needs (point 3). Difficulty:
   honest ray-through-voxel traversal for ~100k rays × 5 sensors in numpy is the
   expensive part; a vectorizable approximation is a per-frame polar free-space map
   (azimuth bin → nearest-return range per height band, one `minimum.reduceat` pass)
   tested against voxel centres. Risks: carving through sparse/glassy surfaces, and
   BeamNG only reports hits — "no return in a direction" is implicit and must not carve.
4. **Hybrid (pragmatic recommendation):** distance-stamped expiry for static geometry
   now; keep wall-clock TTL for vehicle-class voxels; consider carving later as the
   upgrade path that also serves prediction/occlusion work.

### Difficulties

- Anything static that *changes* while you sit still (a gate opens, a wall is knocked
  down by traffic) persists wrongly until contradicted — only option 3 fixes that; with
  options 1–2 the stale patch is refreshed the moment its cells are re-observed, since
  "newest observation wins" already governs re-writes.
- Scene-build cost scales with store size. Distance-based expiry grows the store on long
  straight drives compared to 4 s of TTL (~70–100 m of wall history vs today's ~50 m at
  speed) — same order of magnitude, and `WORLD_MAX_COLUMNS` already caps it (drops
  oldest first), but SCENE BUILD should be watched; the metric and over-budget logging
  exist.
- Parked cars are vehicle-class and never accumulate (0.15 s TTL), so they stay
  thin-looking even while you are stopped next to them. That is a *separate* choice worth
  revisiting: a car measured stationary for several seconds could be promoted to a
  longer window and demoted the instant it moves (needs the tracking from point 3/6 to
  do honestly).

---

## 3. Prediction: behind the car beside me, and the road around the corner

Three tiers, from honest-and-cheap to research-grade. The app's ethos (CLAUDE.md's
"hybrid honesty rule": never render inference as ground truth) is worth keeping — every
predicted surface should ship in its own visual style. Usefully, the palette already
reserves an *uncertainty* channel: `WORLD_UNCERTAIN_RGB` is documented as "deliberately
the weakest mark," and `WORLD_MAX_UNCERTAIN_POINTS` exists. The hooks are there.

### Tier 1 — Remember (mostly falls out of point 2)

Most of what is "behind the car next to me" *was seen* before the geometry got occluded —
you drove past it, or it was visible before the car pulled alongside. Location-based
persistence keeps it on screen. This is the highest-value, lowest-risk 80% of the ask and
requires no new inference machinery at all.

### Tier 2 — Reason about visibility (occlusion honesty)

Today, "no data" renders as empty air, so occluded ground and genuinely empty ground look
identical. Computing the per-frame occlusion shadow is cheap and vectorizable: bin the
cloud by azimuth around each (or one virtual) sensor origin, take the nearest blocking
return per bin (a `minimum.reduceat`), and everything beyond it is *unknown*, not
*empty*. Render never-observed occluded road/ground as a faint uncertain wash (or simply
don't fade persisted geometry there). This one change makes the display stop claiming
knowledge it doesn't have — the same honesty Tesla's visualization signals with its
ghosting. Difficulty: doing it per-sensor is more correct (five origins) but five polar
maps are still cheap; the scene thread has maybe 10 ms of headroom, so it must stay
vectorized.

### Tier 3 — Predict

In increasing order of difficulty:

1. **Road-ahead from the nav route (cheap, already plumbed).** `RouteHint.path_world`
   already carries world-space route polyline nodes when the player sets a destination
   (`models.py:55`). Draw an uncertain-styled ribbon along it beyond the sensed road —
   that *is* "what the road does around the corner," from the game's own routing. Only
   works with a destination set.
2. **The map's road graph (the big one).** BeamNG exposes the AI road network in game
   Lua — `map.getMap()` returns nodes (position + radius) and links for every drivable
   road on the map. Fetched once per map load through the existing
   `queue_lua_command(..., response=True)` channel (same gotchas as `navigation.py`:
   must `return jsonEncode(...)`), it gives centrelines *and widths* for every road,
   including around corners and over crests. Rendered in the uncertain style, clipped to
   some radius, this is a full answer to "predict how the road looks around the corner"
   without inventing anything — it is map data, visually distinct from perception.
   Difficulties: payload size on big maps (fetch once, cache; possibly chunk the query
   by distance), community maps with sparse/wrong AI graphs, and keeping the visual
   distinction unmistakable so it never reads as measurement.
3. **Geometric road extrapolation.** Fit the observed road-edge curvature (from the road
   cell store or boundary voxels) and extend it a handful of metres past the last
   observation, clothoid-style. Works without a destination or map data; goes wrong at
   junctions and driveways; modest value next to option 2.
4. **Object permanence for traffic.** Track clusters of `SCENE_VEHICLE` returns
   frame-to-frame (centroid + constant-velocity Kalman), and when a tracked car enters
   occlusion, coast a ghost with growing uncertainty for 1–2 s. The actor pipeline
   already has coast/fade concepts (`WORLD_ACTOR_COAST_S`, `WORLD_ACTOR_FADE_S`) but is
   starved in free-roam because `vehicles.get_states()` is rejected there; a LiDAR-native
   tracker fixes that *and* produces relative velocities, which AEB wants anyway
   (point 6). Medium effort: clustering (the voxel store already effectively clusters),
   association, and track lifecycle — all standard, all vectorizable at these object
   counts.
5. **Learned occupancy completion / neural scene prediction.** State of the art
   (occupancy networks, point-cloud completion, diffusion) can hallucinate the far side
   of the occluding car. Realistically out of scope: training data, GPU inference inside
   a 40 ms Python tick, and it directly contradicts the honesty rule — the far side of
   that car would be *pure invention*. Predicting unobserved static structure is
   guessing; predicting road via the map (option 2) is looking it up. Recommendation:
   don't go here; spend the effort on tiers 1–2 and options 1, 2, 4.

---

## 4. Dynamic camera

### What the code does today

`world_scene.SceneWorker._camera` (`world_scene.py:1714`) is a stateless pure function:
height = base + speed·slope + |curvature|·corner-lift, distance = base + speed·slope,
pitch fixed at −21°, and an **instant** 180° yaw flip when the signed forward speed says
reversing. There is no smoothing anywhere — continuity comes entirely from speed being
continuous, which is exactly why the reverse flip teleports.

### Options

- **Standstill top-down tilt (the specific ask) — very feasible.** Blend pitch from −21°
  toward near-vertical as smoothed speed approaches zero, while pulling distance in and
  height up; reverse the blend on pull-away. Two implementation routes:
  - *Python-side smoothing:* make `_camera` stateful (the assembler already carries
    per-frame state) and critically damp a target pose — full control, testable offline,
    frame-rate independent if time-constant-based. Recommended.
  - *QML-side:* `Behavior on` NumberAnimations over the bridge properties — least code,
    but position and angles animate independently, so the look-at point swings
    mid-transition; doing it properly wants a rig (yaw pivot → pitch pivot → dolly
    node) in the QML, which is also the cleaner structure for every other move below.
  Two traps: at exactly −90° pitch the euler yaw becomes degenerate (gimbal) — stop at
  ~−80°, which still reads as top-down; and threshold chatter — parking manoeuvres live
  at 0.3–1 m/s, so the tilt needs hysteresis plus a dwell (e.g. tilt down only after
  ~1 s below 0.5 m/s; tilt back immediately above ~2 m/s), or it will nod at every
  give-way line.
- **Smooth the reverse swing.** Animate the yaw through 180° over ~0.5 s instead of
  teleporting, choosing the swing side from the current steering/curvature so the camera
  orbits over the side you're steering toward.
- **AEB event framing.** On BRAKING (or urgency above a threshold), lift/pull the camera
  so ego *and* threat marker are framed together. The colour change is the event today;
  a camera move makes it unmissable. Must be a single smooth move with no oscillation —
  and it should never fire for the armed/watching state or it becomes a nuisance.
- **Corner look-through.** The corner lift exists; the other half is yawing a few degrees
  into `plan.arc.next_curvature` (or measured yaw when self-driving is off) and offsetting
  laterally, so the inside of the bend isn't hidden behind the ego.
- **Parking mode.** Below walking pace with reverse or small gear movements, a closer,
  steeper three-quarter view showing both bumpers and both AEB standoffs — pairs
  naturally with the rear AEB visualization.
- **Speed-based FOV** (mild dolly-zoom) instead of pure distance, keeping the ego's
  screen size constant while the visible road length grows.
- **Free orbit** with mouse drag on the QQuickWidget, auto-returning to the chase pose
  after a few idle seconds.

### Difficulties

- All camera state lives on the scene-worker/QML side, driven per snapshot (~25 Hz);
  smoothing there is fine, but every animated quantity must be slew/critically-damped —
  the palette and overlays were designed around a stable view, and a bouncy camera
  destroys the depth-tint distance cues.
- The transition must not break AEB overlay readability mid-event (top-down actually
  *improves* corridor readability, so the standstill tilt is safe; the event framing
  needs care).
- Depth tint is computed from the render origin, not the camera, so tilting is safe on
  that front (verified in `world_scene.depth_tint`'s design).

---

## 5. AEB slope sensitivity ("slightly uphill ⇒ it brakes")

### Mechanism — found, and it is structural, not a tuning slip

Heights are gravity-referenced (deliberate, load-bearing — see `geometry.py:195`). The
local ground estimate that adjusts the obstacle floor, `planner.ground_rise`, is clamped
**into the slope cone**: `rise = clip(measured, 0, 0.015 · max(r − 10, 0))`
(`planner.py:246`). That clamp exists to protect the *planner's* kerb detection (the
comment says why: unbounded allowance made every kerb past ~12 m invisible). But it
means the system refuses to believe any grade over **1.5%**, and *any* grade at all
inside 10 m. Real roads run 3–10%.

On an uphill of grade g, road surface at range r sits ≈ g·r above the ego plane, while
the allowed floor is `0.30 + 0.015·(r−10)`. At 5% grade and 25 m: road at 1.25 m vs
0.53 m allowed — the entire hillside enters AEB's obstacle band. It is a dense, real,
persistent surface, so it sails through all three phantom filters (height ✓, dozens of
hits ≫ `AEB_MIN_HITS`, persists ≫ `AEB_CONFIRM_S`) and the corridor scan reports a
"wall" at whatever range the road crosses the floor. At 40–70 km/h the AEB horizon is
30–60 m, so even a mild hill *far ahead* fires it. The release latch then holds the
brake, so one phantom is a full stop, not a blip.

**The rear system has it strictly worse:** it operates at 2–8 m, entirely inside
`SLOPE_ALLOWANCE_START_M = 10`, where the allowance is exactly zero. Reversing up any
driveway ramp or parking-garage grade ≥ ~8% puts the ramp surface at 0.30 m within a
few metres — phantom by construction.

Both AEB directions and the planner share one clipped `rise` array
(`geometric_obstacle_sets`, `planner.py:249`) — so today the clamp cannot be loosened
for AEB without loosening it for kerb detection. That coupling is the thing to break.

### Possibilities

1. **Per-band clamp (cheap, targeted).** Give the AEB band its own, much looser bound on
   the *same* measured `ground_rise` — e.g. clip to a 20–25% grade cone instead of 1.5%
   — while the planner keeps the tight cone. One extra clip and per-band selection in
   `geometric_obstacle_sets`. Why it is safe-ish: AEB does not care about kerbs (its
   floor is 0.30 m), and `ground_rise` is a 20th-percentile-per-ring estimate, so a wall
   standing on flat ground does not inflate it much — the percentile lands near the
   wall's base, and the wall still stands ≥ 0.30 m above that. Residual risk: terrain
   the car genuinely *cannot* climb (a steep embankment dead ahead) reads as "ground
   rising" and AEB goes quiet — hence bounding at a drivable-grade cone (~25%) rather
   than unclamping entirely, plus a cap on ring-to-ring estimate jumps (a vertical face
   moves the percentile by metres between adjacent rings; real terrain cannot).
2. **Model the grade, score the residual (the proper estimator fix).** Fit a low-order
   model (line or quadratic in r) to the trusted near rings' ground percentiles with
   outlier rejection, extrapolate it as the expected ground, and measure obstacle height
   against *that*. A constant grade is then exactly absorbed at every range; a crest is
   absorbed by the quadratic term; a wall is a huge residual. This replaces the cone as
   the primary mechanism (the cone can remain as an outer sanity bound). Medium effort,
   entirely offline-testable — synthetic graded clouds with kerbs and walls pin it.
3. **Verticality test (grade-invariant, strongest single idea).** Decide obstacle-ness
   from *vertical extent within a small footprint* rather than absolute height above an
   estimated floor: bin the AEB-band candidates into ~0.4 m XY cells, compute per-cell
   max−min height, and count only cells whose spread exceeds ~0.3 m (wall-like) — a 10%
   slope puts ~4 cm of spread in such a cell; a wall, car face, post or person puts
   dozens of centimetres. Immune to grade *and* to heave/dive (spread is differential),
   which also helps point 1's truck-dive problem. Risks: sparse far-field cells with one
   return have no spread (fall back to the height test past the density limit), and it
   is a philosophical change to the AEB band worth pinning with a thorough offline
   matrix (slopes × kerbs × walls × crests).
4. **Seed the near field with body pitch.** `state["up"]`/`dir` give the body's grade on
   a steady climb; a low-passed pitch (grade changes slowly; dive is transient) could
   tilt the expected ground plane inside 10 m, where the cone offers nothing — the main
   fix for the *reverse* system on ramps. The danger is re-importing the dive latch that
   gravity-referencing killed; the filter must reject transients aggressively (fuse with
   longitudinal acceleration, which predicts dive), and this option should only ever
   *raise* the expected ground, never lower the obstacle floor below its flat-ground
   value.
5. **Reuse the accumulated road-height map.** The WORLD store already holds a terrain
   height field (`_road_keys`/`_road_height`) built from the roof unit's dense sampling —
   the honest local ground truth this problem wants. It lives on the scene thread, so
   using it for control means either duplicating a light ego-frame accumulator in the
   worker or a careful cross-thread snapshot. Architecturally the heaviest option; the
   payoff is one shared terrain model for planner, AEB and display.

Pragmatic combination: (1) now as relief, (2) or (3) as the real fix, (4) only for the
reverse/near-field case, with the offline test matrix pinning all of it.

### Difficulties

- Every loosening trades a phantom for a potential miss on genuinely unclimbable
  terrain; the design has to keep walls unmistakable (verticality or residual size does
  this; a raw unclamp does not).
- The phantom-braking live checklist (CLAUDE.md) explicitly includes crests and dips —
  any change here re-runs it, uphill and downhill, forward and reverse.
- `ground_rise` quality at range depends on ground returns existing there; on crests the
  far side is occluded and the estimate extrapolates flat (np.interp holds end values) —
  the conservative direction, worth keeping.

---

## 6. Broader improvements and new features

### AEB

- **Relative velocity / moving-obstacle awareness.** Obstacles are static by design
  today, which brakes early behind a slowing leader and cannot distinguish "closing at
  15 m/s" from "opening at 5 m/s". A LiDAR-native tracker (point 3, tier 3.4) yields
  per-object closing speeds; the trigger generalizes from distance to true TTC. Biggest
  functional upgrade available to AEB; medium effort; conservative fallback (treat
  untracked as static) preserves today's behaviour.
- **Steering-informed corridor.** The corridor curvature comes from measured yaw, which
  *lags* steering input by a good fraction of a second. Reading the steering wheel from
  `electrics` and blending (steer-predicted curvature for the near corridor, yaw for
  confirmation) aims the scan where the car is *about* to go — fewer corner-entry false
  positives on outside-of-bend scenery.
- **Two-stage warning.** Everything needed for a forward-collision *warning* (visual +
  audible chime at, say, 1.4× the brake-now distance) is already computed per tick. Real
  cars warn first; today the first sign is the full brake. Cheap; the overlay already has
  urgency plumbing.
- **Per-vehicle plant + auto-calibration** — covered in point 1; also mid-event honesty:
  compare achieved decel against the model during an AEB stop and log the shortfall
  (wet/loaded), feeding the learned profile.
- **Vulnerable-road-user band.** `SCENE_VULNERABLE` already exists in the semantic
  vocabulary. A pedestrian-class return could use a lower floor and earlier trigger
  without any change to how kerbs/road are filtered. Note this breaks AEB's
  "geometry-only" purity — a deliberate, documented trade if taken.
- **AEB blackbox.** Ring-buffer the last ~5 s of `AebState` + decimated obstacle cloud;
  dump to disk on every firing. Every "it braked for no reason" report becomes a
  replayable file instead of a checklist drive. Disproportionately valuable given points
  1 and 5 are exactly this kind of report.
- **Cross-traffic alert (rear).** While reversing, the corridors only look along the
  path; a car crossing behind never triggers until it is *in* the corridor. With tracked
  objects, a simple "will its path cross mine within TTC" test gives rear cross-traffic
  alert — a marquee parking feature.

### Visualization

- **Occlusion/unknown shading** — the tier-2 item from point 3; the single most
  informative addition for trust in the display.
- **Lane markings, almost free.** `ROAD_CLASSES` already distinguishes `DASHED_LINE`,
  `SOLID_LINE`, `ZEBRA_CROSSING` — and then merges them into "road". Carrying the paint
  classes through to the road-cell store (a per-cell class byte) and tinting those cells
  slightly lighter draws real lane lines on the WORLD road surface with data the app
  already computes every tick. Likely the biggest visual-fidelity win per unit effort.
- **Slope shading on the road mesh.** Corner heights exist per vertex; a subtle
  slope-based shade (or contour tint) would make crests, dips and camber visible —
  which also lets the driver *see* exactly the terrain that point 5's AEB is reasoning
  about.
- **Traffic velocity arrows / short trails** once tracking exists; ghost boxes through
  occlusion.
- **AEB margin HUD**: a live bar of `available` vs `needed` (the two numbers the whole
  trigger reduces to). Explains every firing and every non-firing at a glance.
- **Kerb/low-obstacle accent**: cells between the planner floor (0.12) and AEB floor
  (0.30) drawn as a low lip in their own shade — visualizes what the planner steers
  around but AEB ignores, which is exactly the boundary points 1 and 5 argue about.
- **Snapshot record/replay.** `PerceptionSnapshot` is already immutable and
  Qt-free; serializing the stream to disk and replaying it through the real
  `WorldSceneAssembler`/`WorldView` offline turns visual review (and every camera/
  palette experiment above) into a desk exercise. The synthetic-snapshot review path in
  CLAUDE.md is most of the way there; this formalizes it.
- **Reverse inset / picture-in-picture** while the rear system is active, instead of (or
  in addition to) the camera flip.

### App/system

- **Config file or UI for `BEAMNG_EXE`** (documented gotcha; hardcoded absolute path).
- **Per-vehicle profile store** (JSON keyed on model name): measured braking plants,
  body width vs bbox width, obstacle floor, standoffs. Points 1 and 5 both land here.
- **Scripted live-check harness.** The AEB/self-driving live checklists are manual.
  BeamNGpy can spawn scenarios (wall ahead, fixed approach speeds) and assert fire
  distances against the `_brake_now_table` — turning the checklist into a repeatable
  half-automated test run per vehicle. This is also how the per-vehicle plants get
  measured without hand-driving.

---

## Suggested priority

1. **Point 5 relief** (per-band slope clamp now; verticality/residual estimator as the
   real fix) — it is a structural false-positive affecting everyone, worst in reverse.
2. **Point 1** per-vehicle profiles + the two live checks (logged width vs body width;
   measured D-Series braking table) — a correctness *and* safety gap.
3. **Point 2** distance-stamped expiry — small, self-contained, big perceived-quality
   win, and it feeds point 3's tier 1.
4. **Occlusion shading + lane paint** — the two cheapest large visualization wins.
5. **Camera standstill tilt + smoothed reverse swing** — contained in the scene worker,
   offline-testable.
6. **LiDAR-native tracking** — the enabling investment for prediction ghosts, TTC-based
   AEB, and cross-traffic alert.
