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
| 2 | Rung 0.5: engine-depth unprojection | 3–4 weeks | The car drives in Vision mode | **NEXT** |
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

## Phase 2 — Rung 0.5: engine-depth unprojection (3–4 weeks) — **NEXT**

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

- [ ] Pure `unprojection.py`: per-camera ray LUT, depth decode, cosine
      divide, strided subsample — pinned offline against synthetic depth
      images.
- [ ] Worker: vision ticks emit `BevFrame`/`PerceptionSnapshot`; BEV and
      WORLD light up in Vision mode.
- [ ] Oracle harness: LiDAR-mode and Vision-mode clouds diffed on the same
      scene (drive the same stretch in both modes; compare band outputs).
- [ ] Per-camera ego-motion compensation measured and applied (frames are
      staged ~1–2 frames apart; ~1.9 m at speed).
- [ ] Re-enable self-driving + AEB in Vision mode behind the full live
      checklist re-run — the sampling distribution is new, so the whole AEB
      phantom checklist applies (hills, brake dive, bushes, kerbs, reverse).
- [ ] **Milestone: the car drives in Vision mode** (on engine depth).

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
