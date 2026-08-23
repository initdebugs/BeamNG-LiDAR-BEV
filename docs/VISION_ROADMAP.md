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
| 0v | Rung-0 live verification | ~30 min | Do the mounts and rates hold on the real car? | in progress |
| 1 | The kerb experiment | 1 afternoon | Is stereo good enough for the planner's kerb requirement? | **NEXT — has veto power** |
| 2 | Rung 0.5: engine-depth unprojection | 3–4 weeks | The car drives in Vision mode | not started |
| 3 | Rung 1: stereo depth | 3–6 weeks | The car drives on genuine vision-only geometry | blocked on phase 1 |
| 4 | Rung 2: sim-trained semantics | 4–8 weeks | Meaning comes from pixels, on every map | after 3 |
| 5 | Rung 3 decision: learned BEV / occupancy | 2–3 months | Is the true Tesla architecture worth the ML project? | decide after 4 |
| 6 | Wild card: imitation from the LiDAR teacher | open | Side experiment only | optional |

Standing item, outside every phase: **the licence question**. The tech.key
timestamp (2026-09-22 if it is an expiry — unconfirmed) interrupts everything
mid-phase. Chase it with whoever supplied the key before committing to phases
2+.

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

- [ ] A virtualenv per simulator version (`.venv39` with 1.36), ending the
      shared-global-site-packages situation that broke PyQt6 once already.
- [ ] Pin renderer settings for capture in `launcher.py`
      (`PostFXMotionBlurEnabled=false`; graphics preset above "Lowest").
- [ ] Start the licence conversation.

## Phase 0v — Rung-0 live verification (~30 minutes, needs the sim)

The checklist in VISION_MODE_SPEC.md §2, in short:

- [ ] `Vision check:` line within ~1 s of attach; the 5 s silence warning
      never fires on a healthy setup.
- [ ] All eight tiles plausible: no cabin interiors, no sky-only tiles, some
      bonnet in the wide/bumper views is correct. (First live look already
      caught the bumper camera inside the shell — fixed with its own
      standoff; re-check it.)
- [ ] Sim rate near the measured 42 Hz while streaming; ACQUISITION ~16–18 Hz.
- [ ] VISION ↔ WORLD/RAW BEV switching mid-stream re-attaches cleanly both
      ways; self-driving/AEB re-arm on return to LiDAR.
- [ ] Click-to-focus works on every tile and returns on the second click.

## Phase 1 — The kerb experiment (1 afternoon; VETO POWER over phase 3)

The cheapest measurement with the most influence over the design. One
wide-baseline front stereo pair (1280×960, ~60° HFOV, baseline 0.5–1.0 m),
one 0.10–0.15 m kerb, ranges 15–30 m:

- [ ] Capture rectified pairs at several ranges (lockstep: pause → step →
      poll both).
- [ ] Compute disparity (CPU SGBM is fine offline; the live rung uses CUDA
      SGM), unproject, accumulate into the existing 0.4 m cell machinery.
- [ ] Verdict table per range: does the kerb face separate from the road
      surface at 15 / 20 / 25 / 30 m, and at how many sigma?

Outcomes:

- **Pass** → phase 3 proceeds as specced (stereo is the depth source).
- **Fail** → the stereo rung pivots to a hybrid (stereo for obstacles and
  AEB range, engine depth for the ground band) — decided now, for the cost
  of an afternoon, instead of after the port.

## Phase 2 — Rung 0.5: engine-depth unprojection (3–4 weeks)

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

## Phase 3 — Rung 1: stereo (3–6 weeks; shape set by phase 1's verdict)

Swap engine depth for computed stereo depth, pair by pair, with engine depth
demoted to a hidden diff oracle. Spec §4.

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
- [ ] **Milestone: engine depth OFF — the car drives on vision-only
      geometry.** A complete, publishable result with zero neural networks.

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
