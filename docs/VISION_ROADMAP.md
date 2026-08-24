> **RETIRED, 2026-08-24.** The vision-ONLY ladder this roadmap describes is
> gone: the eight-camera rig, `unprojection.py`, the oracle and the staging and
> ghosting probes were all removed when HYBRID replaced VISION. HYBRID keeps the
> six LiDARs authoritative and adds two colour-only A-pillar cameras as a live
> view, so nothing below is being worked on. It is kept because the
> MEASUREMENTS in it are about the simulator rather than about that rung, and
> they are the reason the ladder was abandoned:
>
> * computed stereo resolved a 0.11 m kerb at 15 m and nowhere beyond it —
>   matching fails on low-texture asphalt, so engine depth was the only viable
>   depth source;
> * the camera ground band agreed with the LiDAR floor to −1…−2 cm out to 60 m
>   but starved past 20 m (~175 returns per 4 m ring against the road-scan
>   unit's ~1300);
> * camera frame staging measured ≈ 0 by two independent instruments;
> * a fully covered simulator window throttles the renderer to ~2 Hz;
> * the camera buffer's fourth byte is not opacity.
>
> Two later findings supersede parts of it outright, and both are in CLAUDE.md:
> a sensor `pos` is resolved from the body centre rather than the reference
> node, and a tech Camera does not auto-expose.

# Vision Project — Development Roadmap

The phased plan for the whole vision-only effort. The technical detail behind
every phase lives in `docs/VISION_MODE_SPEC.md` (what each rung is, the
measured numbers, the contracts); this file is the ORDER of work, what gates
what, and how to tell each phase is done. The research report the ladder came
from is checked in beside it as `docs/vision_only_ladder.html` (open it in a
browser).

Effort figures are working estimates for one person, on top of whatever the
simulator demands on the day. Every phase ends with the offline suite green
and a live checklist run — the suite pins arithmetic, never what the
simulator does with it.

---

## Overview

| Phase | What | Effort | Gate it answers | Status |
|-------|------|--------|-----------------|--------|
| 0 | Foundations: beamngpy 1.36, 0.39, Vision mode rung 0 | — | Can cameras stream at all? | **DONE 2026-08-22** |
| 0v | Rung-0 live verification | ~30 min | Do the mounts and rates hold on the real car? | **DONE 2026-08-23** |
| 1 | The kerb experiment | 1 afternoon | Is stereo good enough for the planner's kerb requirement? | **DONE 2026-08-23 — FAILED, see below** |
| 2 | Rung 0.5: engine-depth unprojection | 3–4 weeks | The car drives in Vision mode | **IN PROGRESS** — milestones 1–3 landed 2026-08-23 |
| 3 | Rung 1: stereo depth | 3–6 weeks | The car drives on vision geometry for OBSTACLES; the ground band stays on engine depth | re-shaped by phase 1 |
| 4 | Rung 2: sim-trained semantics | 4–8 weeks | Meaning comes from pixels, on every map | after 3 |
| 5 | Rung 3 decision: learned BEV / occupancy | 2–3 months | Is the true Tesla architecture worth the ML project? | decide after 4 |
| 6 | Wild card: imitation from the LiDAR teacher | open | Side experiment only | optional |

Standing item, outside every phase: **the licence question**. The tech.key
timestamp is 2026-09-22 if it is an expiry, which is unconfirmed and would
interrupt everything mid-phase. **Deliberately deferred 2026-08-23** — the
owner's call, not an oversight. Nothing in phase 2 depends on resolving it
first; the risk is that it lands mid-phase rather than that it blocks a start.

---

## Phase 0 — Foundations (DONE)

Landed 2026-08-22:

- [x] beamngpy pinned to 1.36; `BEAMNG_EXE` on 0.39.4.0 (the two move
      together — protocol v1.27).
- [x] 1.36 behaviour differences handled (`get_current` name-validation
      dropout; positive `requested_update_time`).
- [x] Vision sensor mode in the worker, eight-camera HW4 rig derived from the
      live bbox, live grid view with click-to-focus, offline tests.

Still open from the original foundations list (do alongside 0v):

- [x] A virtualenv per simulator version (`.venv39` with 1.36), ending the
      shared-global-site-packages situation that broke PyQt6 once already —
      done 2026-08-23. Suite green in it (770 passed) and pytest-qt is absent
      from it entirely. Both `.bat` files prefer it and fall back with a
      message.
- [x] Renderer settings for capture — done 2026-08-23, but as a **CHECK, not a
      pin**: `bng.settings.change` + `apply_graphics` was measured on 0.39.4
      and moved neither the sensor nor the game view, so setting them would
      have been a line that quietly did nothing. `launcher.
      capture_setting_warnings` is a pure rule over the values the worker reads
      via `settings.getValue`, logged once per vision attach as
      `Capture check:` and warned about only when actually wrong (the
      black-frame presets, and motion blur). Current machine: clean.
- [~] Start the licence conversation — **deferred by decision 2026-08-23**,
      see the standing item above.

## Phase 0v — Rung-0 live verification (DONE 2026-08-23)

The checklist in VISION_MODE_SPEC.md §2, in short. Everything measurable in the
app is confirmed; the one item needing a number from BeamNG's own HUD was
dropped by decision rather than met.

Four defects were found and fixed in the course of it, none of them predicted by
the checklist: the 0.39 launcher aborting on a windowless parent (Launch had
been dead since the upgrade), the camera buffer's fourth byte being read as
opacity (the reported "noise"), a 24.5-degree blind gap either side of the rig,
and the centreline cameras sitting on the reference node instead of the body
centre. That ratio is the argument for live verification existing at all.

- [x] `Vision check:` line within ~1 s of attach; the 5 s silence warning
      never fires on a healthy setup — confirmed 2026-08-23, **0.1 s**, eight
      cameras delivering, on three attaches (two at 1280×960). No warning, and
      no WARN/ERROR anywhere in the day's log.
- [x] All eight tiles plausible: no cabin interiors, no sky-only tiles, some
      bonnet in the wide/bumper views is correct — confirmed 2026-08-23, all
      eight stream. (First live look caught the bumper camera inside the shell
      — fixed with its own standoff.)
- [x] Image quality — RESOLVED 2026-08-23. "Pixelated" was the 640×480 source
      (now 1280×960). "Grainy" was **ours**: the camera buffer's fourth byte is
      not opacity, and `vision_view` read it as `Format_RGBA8888`, so Qt
      composited every pixel against the dark tile — mean error 26.08 against
      the true colour, 0.00 once read as `Format_RGBX8888`. "Washed out" is a
      real but much milder effect than first reported (the sensor has no
      auto-exposure; 2-13% of pixels clip on a normal street, not the 53% first
      measured through the wrong vehicle). Spec §2 carries all three.
- [x] ACQUISITION ~16–18 Hz — confirmed 2026-08-23 at **17.5 Hz**, at
      1280×960, i.e. the resolution rise cost ~1 Hz as predicted.
- [~] Sim rate near the measured 42 Hz while streaming — **dropped by decision
      2026-08-23**, and 0v is called done without it. It is the one item with no
      in-app source: `Engine.getFPS` and `Engine.Render.getFPS` do not exist on
      0.39.4, so it has to be read from the `Performance Timers` UI app
      (`ui/modules/apps/SimplePerfTimers`, Telemetry category) by hand. The
      evidence that the rig keeps up is indirect but real — ACQUISITION held
      17.5 Hz at 1280x960 against a measured 16-18 Hz expectation, and no poll
      failure has been logged in any session.
- [x] VISION ↔ WORLD/RAW BEV switching mid-stream re-attaches cleanly both
      ways; self-driving/AEB re-arm on return to LiDAR — confirmed live
      2026-08-23, all three parts: the round trip, both AEBs re-arming, and the
      six-unit LiDAR rig attaching. This was the 0v item with real machinery
      behind it (a re-attach through the single `_cleanup_sensors` funnel and
      the worker's refusal list handing back), and it was the first LiDAR
      attach since the 0.39 upgrade.
- [x] Click-to-focus works on every tile and returns on the second click —
      confirmed live 2026-08-23, both halves.

## Phase 1 — The kerb experiment (1 afternoon; VETO POWER over phase 3)

The cheapest measurement with the most influence over the design. One
wide-baseline front stereo pair (1280×960, ~60° HFOV, baseline 0.5–1.0 m),
one 0.10–0.15 m kerb, ranges 15–30 m:

- [ ] Capture rectified pairs at several ranges (lockstep: pause → step →
      poll both). **Watch the exposure** — the Camera sensor has no
      auto-exposure (spec §2), and a saturated region is precisely the one
      stereo cannot correlate, so a clipped capture would measure the clipper
      rather than the kerb. On a normal street it clips 2–13% of pixels, which
      is tolerable; under a high sun it is much worse. Check before trusting a
      run, rather than assuming either way.
- [ ] Compute disparity (CPU SGBM is fine offline; the live rung uses CUDA
      SGM), unproject, accumulate into the existing 0.4 m cell machinery.
- [ ] Verdict table per range: does the kerb face separate from the road
      surface at 15 / 20 / 25 / 30 m, and at how many sigma?

### VERDICT 2026-08-23: **FAIL** — the stereo rung pivots to a hybrid

Measured with `tools/kerb_experiment.py` on a straight two-kerb street
(west_coast_usa), one lockstep pair per configuration, the engine depth channel
as the oracle. Separation is the stereo kerb step divided by the stereo road
noise at that range; 3σ is the bar.

| baseline | width | 15 m | 20 m | 25 m | 30 m |
|---|---|---|---|---|---|
| 0.6 m | 1280 | 3.6σ | no data | −3.9σ | −4.1σ |
| 1.0 m | 1280 | 4.2σ | no data | no data | −3.1σ |
| 1.6 m | 1280 | −1.7σ | no data | no data | no data |
| 1.0 m | 1920 | 4.7σ | no data | no data | no data |
| **oracle control** | 1280 | **9.4σ** | 13.8σ | 15.0σ | 16.4σ |

**Stereo resolves the kerb at 15 m and nowhere beyond it.** Past 20 m it either
produces no depth at the kerb line at all, or a physically impossible NEGATIVE
step — the pavement reading lower than the road, which the bias table explains:
stereo over-ranges by 0.44–0.66 m there, and a too-far point on a grazing
surface reads as a lower one. Even at 15 m it under-reads the step, 0.077 m
against a true 0.113.

Three things this rules out as the cause:

- **Not the SGBM parameters.** Swept blockSize 3–11 against two P2/uniqueness
  settings: road depth sigma at 30 m stayed 0.87–1.53 m throughout.
- **Not the baseline.** 1.6 m was the WORST (32% valid pixels — a wide pair
  loses overlap and occludes more than the extra disparity buys).
- **Not the resolution.** 1920 halved the depth noise (0.96 → 0.56 m at 30 m)
  and still produced nothing at the kerb past 15 m. The failure at range is
  MATCHING on low-texture asphalt, not precision.

The oracle control passing at 9.4–16.4σ through the identical pipeline is what
makes this a statement about stereo rather than about the measurement.

**Consequence, as the plan already anticipated:** stereo for obstacles and AEB
range (big vertical targets, plenty of texture), engine depth for the ground
band. Phase 3 is re-shaped, not cancelled, and this cost an afternoon instead of
a 3–6 week port.

Caveats worth keeping: one scene, one map, one lighting condition, one kerb
geometry. A re-run on a differently-textured road surface would strengthen it,
and nothing here tests a kerb against a WET or high-contrast surface where
matching would be easier.

## Phase 2 — Rung 0.5: engine-depth unprojection (3–4 weeks) — **IN PROGRESS**

Phase 1's verdict promotes this from scaffolding to the permanent source of the
ground band: engine depth is no longer a stepping stone that stereo replaces,
it is what kerbs and road surface will keep coming from. Build it accordingly —
the oracle harness below is now load-bearing rather than a diagnostic.

Turn on depth (+ annotation) and rebuild the perception waist
(`points_world + colours + state`) from the cameras. Everything downstream —
planner, controller, AEB, BEV, WORLD — resumes working, fed by the rig.
Spec §3 has the technical detail (planar-Z cosine divide, `integer_depth`
traps, subsampling budget, frame-skew compensation).

Milestones, in order:

- [x] Pure `unprojection.py`: per-camera ray LUT, depth decode, cosine
      divide, strided subsample — pinned offline against synthetic depth
      images. **Landed 2026-08-23**, 21 tests (`tests/test_unprojection.py`):
      a flat floor comes back flat to 0.5 mm across a 100° frame (the
      planar-Z proof), handedness, the pitched rear camera, sky/bodywork
      culls, frame-age rewind, the rig's sample budget. 5.5 ms per tick for
      the whole rig at 960×720 (12.6 before float32 end-to-end and the
      one-word annotation gather).
- [x] Worker: vision ticks emit `BevFrame`/`PerceptionSnapshot`; BEV and
      WORLD light up in Vision mode. **Landed 2026-08-23.** `_poll_once` is
      one tick for either instrument set — acquisition is the only split;
      everything from `points_world + colours` on is the LiDAR's code. The
      header gained a separate LIDAR/VISION instrument toggle beside the
      WORLD/RAW BEV/CAMERAS view toggle, so the cloud views work on cameras.
      The driving controls stay refused behind `VISION_DRIVING_ENABLED`
      (milestone 5). Offline-proven only; the app has not been run on the
      rig since.
- [x] Oracle harness: `tools/unprojection_oracle.py` — both rigs on the
      player's car at once, one lockstep capture, both clouds through the
      real bands. **Landed and run once live 2026-08-23** on the west_coast_usa
      car park the car happened to be parked in:
      - **Handedness settled**: planner-band IoU direct 0.151 vs mirrored
        0.015 — image right is vehicle right, as `camera_basis` assumes.
      - **Ground band**: road floor per 4 m ring agrees with the LiDAR to
        +3…+9 mm out to 24 m. The far road was occluded on that scene, so
        the reach past 25 m is still unmeasured — re-run on a street.
      - Two honest sampling differences for milestone 5: a parked car
        behind lands one 0.4 m cell nearer from the 0.9 m rear camera (rear
        glass) than from the 0.2 m rear LiDAR (bumper); and a tree 18 m
        behind-left enters the planner band from `repeater_left` (554
        returns at 1.8–6.6 m) where the LiDAR's 61 canopy returns were
        dropped — the cameras see low branches the rings miss.
- [x] Per-camera ego-motion compensation measured and applied (frames are
      staged ~1–2 frames apart; ~1.9 m at speed). The measurable half — the
      time since each camera's depth lattice last changed — is applied per
      camera, position AND heading (`pose_from_state(state, age, yaw_rate)`;
      the stale-heading half was the live "world turns with me" report, see
      below). The fixed staging part, `CAMERA_FRAME_STAGING_S`, is zero
      until measured; `tools/camera_staging_probe.py` measures it (swing a
      probe camera, count stepped frames until the buffer follows) and
      REFUSES while the simulator window is covered — in that state the
      renderer throttles to ~2 Hz and every latency reads as throttle. The
      probe now also refuses to recommend a global constant unless four
      stepped trials and four real-time trials are stable and agree within
      one camera frame; contradictory timing evidence leaves the constant
      untouched.

      **Run 2026-08-24, and the refusal rule fired correctly.** Stepped:
      3, 4, 4, 3 sim steps — which is the camera's own 0.05 s update
      period expressed in steps, so the stepped instrument measures
      period + staging and bounds staging at ≤ ~1 step, it cannot isolate
      it. Real time: 5, 8, 106, 100 ms — bimodal, which is the swing
      landing at a random phase of the 50 ms camera cycle, not noise; the
      5–8 ms responses are the strong half of the evidence, because a
      buffer staged 1–2 frames behind can never follow a swing within
      8 ms. Conclusion: staging on 0.39.4 is bounded well under one
      camera period; `CAMERA_FRAME_STAGING_S` stays 0.0 for now, and the
      DEFINITIVE instrument is the ghosting probe's moving case below —
      a systematic per-camera along-travel offset at speed IS
      staging × speed, measured on the thing the constant exists to fix.

      **And it was, the same day (the fence run under the ghosting
      milestone): +32 ± 17 ms of total speed-scaled age error, of which
      the probe's own detection latency predicts ~17–20 — staging ≈ 0,
      two independent instruments agreeing. `CAMERA_FRAME_STAGING_S`
      stays 0.0 as a MEASURED value, closing this milestone.**
- [ ] WORLD ghosting from MOTION registration error — reported live
      2026-08-24 as one pole drawn several times near the same spot, and as
      the vision scene reading cluttered and less crisp than the LiDAR's.
      LiDAR points arrive from the engine already world-registered, so the
      90 m scenery memory only ever sharpens them; camera points are
      reconstructed against an ESTIMATED pose, and every pose error paints
      a displaced copy that the store then faithfully keeps. Three
      mechanisms, sized at the 40 km/h cap:
      - the unmeasured `CAMERA_FRAME_STAGING_S` (1–2 frames ≈ 60–120 ms)
        misplaces every moving capture by 0.7–1.3 m along the direction of
        travel — the staging milestone above removes this one, which is
        why it comes first;
      - the freshness digest sees a frame change only on the 40 ms tick,
        and the beat between that tick and the ~57 ms camera period sweeps
        the detection latency over 0–40 ms: up to 0.44 m of frame-to-frame
        jitter that no constant can remove;
      - a depth read overlapping the simulator's write mixes two frames
        ~57 ms apart along one row boundary (depth is deliberately not
        copied), splitting one object by ~0.6 m inside a single frame.
      Candidate mitigations — ingesting into the WORLD stores only on
      fresh-frame ticks, copying or tear-detecting depth, a shorter
      scenery memory in vision mode — are chosen from the RESIDUAL measured
      after the staging constant lands, never from appearance.
      Discriminators that cost nothing: RAW BEV draws one pole (no
      accumulation), and a pole observed from a standstill stays single —
      every mechanism above is motion × time.
- [ ] Cross-camera spatial fusion and WORLD ghosting validation. The eight
      overlapping camera clouds are concatenated before the shared semantic
      and WORLD pipeline, so observations of one narrow static object can
      occupy adjacent 0.125 m columns and the 90 m scenery memory can preserve
      offset copies. Measure disagreement before choosing a fusion radius —
      nearby real objects must never be merged just to make the view cleaner.
      Acceptance has two live cases on the same isolated pole or bollard:
      **stationary**, every camera overlap resolves to one connected structure;
      **moving turn**, repeated observations leave no persistent parallel copy
      after the pole has crossed an overlap seam. Capture per-camera point
      provenance and report lateral/radial spread for both cases so any later
      deduplication threshold is derived from evidence rather than appearance.

      `tools/ghosting_probe.py` is the instrument (auto-picks the nearest
      narrow tall isolated structure, or `--target X Y`; culls each
      camera's cloud to a cylinder around it with per-camera/per-tick
      provenance; reports along/cross centroid offsets, per-tick jitter,
      and the ghost-column ratio — distinct 0.125 m columns painted over
      the capture against one tick's worth, the store's-eye view of the
      smear). **The STATIONARY half is measured (2026-08-24**, a 3.5 m
      pole 18.8 m away, 186 ticks): three cameras on the structure agree
      to **±1.5 cm** — a tenth of a voxel column — tick jitter ≤ 19 mm,
      ghost ratio **1.0x**. A parked rig resolves ONE structure, so the
      multi-pole clutter is motion registration, not rig geometry. The
      report separates the structure from the ground inside the cylinder,
      because a camera whose frame misses the object still lands road
      returns there (front_main's whole contribution was a 3 cm slice of
      tarmac that read as a 1.26 m offset until split out).

      **The first MOVING run (2026-08-24, a drive-by at ~40 km/h) caught
      the instrument, not the answer, and taught it three things.** The
      only returns in the cylinder were a planter kerb ~1 m from the pole,
      whose VISIBLE WINDOW slides with the car (per-frame offsets marched
      −0.43 → +1.20 m in lockstep with the ego — 1.66 m of frame spread
      against a 0.25 m physical jitter bound of v × one loop tick), and
      the pole itself gave zero moving returns — either the coarse column
      strides miss a thin post beyond ~10 m (repeaters sample every 0.56°)
      or its copies were displaced outside the 1.5 m cylinder, which the
      capture culled and so could not distinguish. The probe now measures
      offsets from the target's REST position (its own parked first frame;
      pooled-centre offsets subtract away exactly the common bias when one
      camera contributes), groups by camera FRAME rather than tick, warns
      when the tracked returns exceed the jitter bound (the slide
      signature), and defaults the moving cylinder to 3.5 m. What the run
      DID establish: the stale-frame pose rewind is sound (within-frame
      drift 0.05 m over 66 ms at 7 m/s), and with both rigs attached the
      cameras delivered at ~10 Hz, not 17.5 — ages reach ~100 ms, so
      age-dependent error is roughly twice the single-rig arithmetic.
      **The second MOVING run (2026-08-24, `--ahead` at a fence) closed
      the staging question, after one more instrument confound.** The
      tracked returns were again a LINE — a second structure ~2.3 m from
      the picked face — and dividing its offset by the speed
      manufactured +125…+475 ms of fake age error, because a static
      offset does not scale with speed and a registration error does.
      The report now separates them: it decomposes on the structure's
      own NORMAL (the window slide lives entirely on the tangent —
      nothing can slide a straight line through itself) and REGRESSES
      per-frame normal offset against crossing speed. Result over 43
      crossing frames: **static −2.33 m + 32 ± 17 ms × speed, residual
      0.43 m rms**. The probe loop's own detection latency predicts a
      ~17–20 ms mean (half its 34 ms tick), so staging ≈ 14 ± 17 ms —
      zero, corroborating the swing probe. The ghost clutter therefore
      has NO large constant behind it: what remains is the one-tick
      detection jitter (the digest sees a frame change only on the
      40 ms tick — up to ±0.2 m per frame at the 40 km/h cap) and the
      torn depth reads. The evidence-backed first mitigation — SEEN-TIME
      CENTRING, stamping a fresh frame at the MIDPOINT of the last two
      looks (the true change time is uniform over the interval between
      them, so the midpoint zeroes the mean error and halves the worst
      case), worth ~20 of the ~32 ms — **LANDED the same day**: worker
      (`_camera_frame_checked`, pinned by `test_a_fresh_frames_seen_
      time_is_centred_between_the_last_two_looks`) and mirrored in the
      probe, so future captures measure the residual the app actually
      carries. Still open: the acceptance re-drive — one pole laid down
      ONCE at speed in WORLD — and only if its residual still smears,
      the torn-read question.
- [~] The vision ground band's REACH is bounded by ROW sampling, and the
      arithmetic puts the single-frame edge near 30 m — reported live
      2026-08-24 as "less range than the LiDAR world". Rows are the range
      axis and rings land `(r²/h)·Δθ` apart: from the 1.30 m eye at
      front_main's row stride that is ~0.6 m at 20 m, ~1.3 m at 30 m and
      ~2.3 m at 40 m — against the road-scan unit's 0.20 m at 50 m — and
      `WORLD_ROAD_BRIDGE_CELLS` closes only 1.5 m, so the single-frame
      camera road fragments past ~30 m (sooner on the coarser side and
      rear strides), where the LiDAR rig carries a unit fitted to 20–100 m.
      Accumulation fills it while DRIVING, exactly as the pre-road-scan
      LiDAR road did. Measure on the street oracle capture first (the open
      item above), then decide: a horizon-weighted row stride (finer rows
      only where they map to 20 m+, paid for by coarsening the near
      field), a ninth narrow far-road camera, or accepting accumulation.
      Any stride change is bounded by the 150–320k sample-budget test.

      **Measured 2026-08-24 (street capture,
      `tools/oracle_data/street.npz`): the band is ACCURATE to −1…−2 cm
      against the LiDAR floor on every ring out to 60 m — density, not
      accuracy, is the binder — with ~175 returns per 4 m ring at
      20–24 m against the road-scan unit's ~1300, thinning to ~30 by
      50 m.** The decision fell to the horizon-weighted stride and it
      LANDED the same day (`CAMERA_FAR_ROAD_BAND_M`,
      `CameraMount.far_road_band_m`, `unprojection._far_road_rows`): all
      level ground from 20 to 100 m lives in a ~54-row strip just under
      front_main's horizon (planar geometry, image y = h/r; the builder
      raises on a pitched camera), so that strip is sampled at stride
      1 — ~7k samples on the 283k lattice — halving the far ring spacing
      and moving the single-frame road edge from ~30 m to ~45 m, where
      stride-1 rings outrun the 1.5 m bridge. Beyond ~45 m accumulation
      while driving fills the road; a ninth camera or a taller front_main
      resolution remain the levers if that is ever not enough. Pinned by
      `test_the_far_road_band_rows_are_sampled_at_full_density`. Still
      open: the live look — the vision WORLD's far road visibly filling
      to ~45 m single-frame on a street.
- [~] Re-enable self-driving + AEB in Vision mode behind the full live
      checklist re-run — the sampling distribution is new, so the whole AEB
      phantom checklist applies (hills, brake dive, bushes, kerbs, reverse,
      and the tree case above). The code change is flipping
      `VISION_DRIVING_ENABLED`; the rest is driving.

      **Flipped 2026-08-24** — earned by the measurements: ground band
      −1…−2 cm to 60 m, staging ≈ 0, jitter zero-mean after the centring,
      stationary rig coherent to ±1.5 cm, one perception waist. All four
      slots (self-driving, both AEBs, parking) open together; the gate
      machinery stays and one constant shuts them all again, pinned both
      directions by `test_vision_mode_offers_driving_by_default_since_
      milestone_5` and the closed-gate tests. **The LIVE CHECKLIST is now
      the whole remaining milestone.** BeamNG window visible; both AEBs
      arm at attach on their own; drive with self-driving OFF first:
      - flat empty road well over the 40 km/h cap, crests, dips, corners
        near the kerb, hard manual braking (the brake-dive case) — the
        AEB metric never leaves ARMED;
      - a hill at 40–70 km/h, and reversing up a ramp — the
        cell-referenced extent test is what should hold it ARMED, and
        `AEB evidence:` says so if not (small height spread = the ground
        arrived in the band, a sampling phantom; large = genuinely solid);
      - roadside bushes and scrub — porosity now reasons from the 1.3 m
        camera eye, not the roof unit;
      - the two measured sampling differences: low canopy over the road
        (the repeaters see branches the LiDAR rings missed — watch for a
        flinch or a brake under trees), and reversing toward a car (its
        rear glass reads one 0.4 m cell nearer than its bumper — the
        SAFE direction, but confirm the rear brake is not early);
      - then that it still FIRES: a wall and a stopped car, forward and
        reversing;
      - then self-driving ON: lane keeping on a street, free distance
        near the horizon on an empty straight, a corner braked for
        early and smoothly, and the junction/route behaviour unchanged
        in character from LiDAR mode.
      One flip of `VISION_DRIVING_ENABLED` closes everything if any of
      it misbehaves — say what fired and the `AEB evidence:` line says
      why.

      **First checklist drive (2026-08-24): it drives, it brakes for
      inclines "every now and then", and the evidence lines classified
      every firing. Two fixes landed the same day.**
      - **Near-field phantoms on crests and brake dives** (threats at
        2–4 m, 0.3–0.7 m of within-cell height spread, required decel
        60–∞ m/s²): STALE-FRAME PITCH. Frames were rewound in position
        and yaw but pitch was "left as it is", so eight cameras with
        different ages disagreed by r × Δpitch of height exactly during
        the 5–15°/s of a dive or grade transition — and AEB's own full
        application pitched the car further, sustaining the phantom it
        fired on. FIXED: the worker measures a pitch rate beside the yaw
        rate (`_observe_state_rates`) and `pose_from_state` rewinds it
        about the vehicle's right axis. Re-drive the hill and brake-dive
        cases to confirm.
      - **Canopy-as-wall at range** (returns starting 8–14 m above the
        plane at 39/68/95 m, with `ground rise` readings of 5–11 m — the
        estimator was following the canopy): the coarse-base ceiling has
        no floor context where the camera lays no ground, and the
        level-fitted far-road band left the road on any grade or pitch —
        the SAME defect that made the drawn road pop between ~10 and
        ~40 m. Partially fixed by the band's new ±2° grade/pitch margin
        (`CAMERA_FAR_ROAD_PITCH_MARGIN_DEG`, ~17k samples). STILL OPEN:
        beyond ~50 m no camera ground context exists at all, so at
        100+ km/h AEB scans a region where a soffit cannot be told from
        a wall. The honest options: a vision-mode AEB horizon cap
        (~60 m — above ~100 km/h it becomes mitigation-only, the same
        sensor-limit statement the LiDAR makes at 170), or accepting it
        and keeping the checklist under 100 km/h. Decide after the
        re-drive.
      - The 10.5 m firing (min 1.74 m up, 12 m tall) was a tree branch
        at windscreen height — arguably a TRUE positive; low canopy over
        the road is the case the oracle predicted.
      - The REAR firings at 56–67 km/h (canopy 8–11 m up, behind) need
        one answer from the driver: was the car genuinely reversing at
        that speed? If not, the rear arming has a sign bug to find.
- [ ] **Milestone: the car drives in Vision mode** (on engine depth).

Also landed alongside, by request: the rear camera is 130° and pitched 15°
down (`CAMERA_REAR_PITCH_DEG`) — it is the reversing camera, and its frame now
reaches the ground ~0.3 m behind the lens.

### The first live drive (2026-08-23 evening) and what it found

The mode worked — cameras attached, cloud produced, WORLD lit up — but slow
("like 10 FPS") and the WORLD view "remembered": on a turn the whole scene
turned with the car and took a few seconds of driving to correct. Diagnosed
from the log and live probes; all fixes landed, none vision-specific:

- The "turning world" was the prefetched state's STALE HEADING rotating
  every cloud stamped into the world-anchored stores during a turn (RAW BEV
  cannot show it — the same state transforms the cloud both ways). Fixed by
  a measured yaw rate advancing the prefetched heading and rewinding each
  camera frame.
- The worker thread spent ~0.5 s of every second on REFUSED actor
  enrichment round trips (get_states 39 ms at 10 Hz + get_current_info
  120 ms at 1 Hz, each logging a traceback). One refusal now rests both for
  15 s (`WORLD_ACTOR_RETRY_S`).
- The scene refresh ran 200–460 ms against its 120 ms cadence,
  back-to-back, taxing every thread through the GIL. The ground store held
  90k stacked "ground" cells — car roofs, wall tops, building roofs
  promoted as floor — now collapsed to one height per cell and filtered by
  connectivity to the car (`connected_ground`); the slab merge lost its
  Python loop. ~280 ms → ~170 on the live capture, and an overrunning build
  now stretches its own cadence (`WORLD_STORE_REFRESH_DUTY`).
- **A fully covered simulator window throttles the renderer to ~2 Hz** —
  LiDAR included; measured live, and `poll_sensors` stays normal so it
  looks like app lag. `Capture check:` warns now. Keep BeamNG visible while
  streaming.

Still open from that session: the staging measurement itself (the probe
needs the sim window focused). The 2026-08-24 re-drive reported neither of
the old complaints; what it reported instead — ghost copies of narrow
objects, and shorter ground reach than the LiDAR view — is diagnosed and
queued as the two milestones above (motion registration error, row-sampling
reach).

## Phase 3 — Rung 1: stereo (3–6 weeks; RE-SHAPED by phase 1's verdict)

**Not** a wholesale swap any more. Phase 1 measured stereo failing to resolve a
kerb beyond 15 m in every configuration tried, so the split is by WHAT IS BEING
SENSED rather than by rung:

- **Stereo takes obstacles and AEB range** — big, textured, mostly vertical
  targets, which is the case it handles well.
- **Engine depth keeps the ground band** — kerbs, road surface, the drivable
  floor. This is permanent unless a later measurement overturns phase 1.

The milestone below therefore changes: "engine depth OFF" is no longer the goal,
and a vision-only-geometry claim cannot be made honestly on this evidence. Spec
§4 still describes the stereo machinery; ignore its framing as a full swap.

- [ ] Rig re-paired (wide-baseline 1280×960 front pair; VGA side/rear pairs).
- [ ] CUDA SGM integrated (~2 ms/pair measured elsewhere; measure HERE under
      WDDM contention first).
- [ ] Edge-culling filter for silhouette streaks (the new phantom class) +
      per-point range-variance channel into the cell stores.
- [ ] Re-derive the LiDAR-fitted machinery the port review flagged:
      `_porous`, `ground_rise`, `vehicle_fit` — safety re-derivations, not
      tuning.
- [ ] Constants retuned against stereo noise; oracle diff within budget at
      10/25/50 m.
- [ ] Full phantom-braking checklist, again.
- [ ] **Milestone: the car drives with stereo supplying obstacles and AEB
      range**, engine depth supplying the ground band. Still a complete result
      with zero neural networks — but NOT a vision-only-geometry claim, which
      phase 1's measurement does not support.

## Phase 4 — Rung 2: sim-trained semantics (4–8 weeks)

Replace the annotation channel with a segmentation network trained on
BeamNG's own free labels; add lanes and 2D detection if wanted. Spec §5.

- [ ] Lockstep capture tool (pause → step → poll ×8 → write); storage budget
      ~295 MB/s at 10 fps — capture in bursts.
- [ ] Train a small net (PIDNet/DDRNet/EfficientViT class) overnight on the
      5070 Ti; TensorRT FP16 batch-8 export.
- [ ] Live A/B tile: net output vs annotation channel, mIoU on screen.
- [ ] Swap semantics source; unannotated-map behaviour must be no worse than
      today's geometric fallback.
- [ ] **Milestone: meaning from pixels — lane paint on every map.**

## Phase 5 — Rung 3 decision point (2–3 months if taken)

Only decide once phases 3–4 exist: by then the capture pipeline, deployment
tooling and the real contention numbers make it a choice made with data.
Candidates and figures in spec §6 (FlashOcc ≈ 2.5–5 ms, BEVDet-TensorRT,
CVT). The 3D-label factory is the accumulated LiDAR voxel store.

## Phase 6 — Wild card (optional, never the main line)

Imitation-learn a small driving network from the LiDAR planner as privileged
teacher (spec §7). Compute-cheap, data-cheap in sim — but it replaces the
legible pipeline with a black box, so it stays a side experiment.

---

## Instruments that must stay true through every phase

- The **oracle loop**: every rung diffable against the rung below, live.
- The **check lines** (`Vision check:`, `Mount check:`, `Sensor reach:`,
  `AEB evidence:` …): each phase adds its own one-shot live diagnostics
  rather than trusting offline arithmetic.
- The **perception waist**: `points_world + colours + state`. Anything that
  breaks that contract breaks every consumer at once.
- The **AEB re-run rule**: any change to the sampling distribution that
  feeds the bands re-runs the phantom checklist. No exceptions; this is
  where every LiDAR-era regression hid.
