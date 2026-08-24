# Task 7 Offline Closure Report

Status: complete for the offline closure. The live BeamNG angle check and
runtime-log evidence are deliberately not included here.

## Files and commits

This task changed:

- `README.md` — operator workflow and LiDAR-first HYBRID explanation.
- `src/beamng_lidar_bev/geometry.py` — one mechanical hybrid-import ordering
  correction required by Ruff; no geometry behaviour changed.

Commits created by this task:

- `4eaf666 style: sort hybrid geometry imports`
- `8299a59 docs: explain lidar-first hybrid mode`
- `docs: record hybrid offline verification` — this closure report

## Verification

```powershell
& 'C:\Users\initd\Documents\Projects\BeamNG.Tech Mods\BeamNG-LiDAR-BEV\.venv39\Scripts\python.exe' -m ruff check src tests
```

Result: exit 0, `All checks passed!`. The first run found only `I001` at the
`geometry.py` import block; the follow-up changed only the ordering of
`HYBRID_CAMERA_HEIGHT_FRACTION` and `HYBRID_CAMERA_HFOV_DEG`.

```powershell
& 'C:\Users\initd\Documents\Projects\BeamNG.Tech Mods\BeamNG-LiDAR-BEV\.venv39\Scripts\python.exe' -m pytest
```

Result: exit 0, `850 passed, 1 xfailed in 43.08s`. Failures: 0. Unexpected
xpasses: 0. The one xfail is expected.

```powershell
git diff 6ed3782 --check
```

Result: exit 0, no diff whitespace diagnostics.

```powershell
git diff 6ed3782 --stat
```

Result at final HEAD: 12 changed files, 2,308 insertions, and 63 deletions.

```powershell
git status --short
```

Result at final HEAD: clean (no status entries).

## Static constraint audit

- HYBRID has the six existing LiDAR mounts plus exactly the ordered
  `a_pillar_left` / `a_pillar_right` camera pair. The rig derivation,
  attach-order, and constructor tests cover this.
- HYBRID's poll path obtains point chunks only from `_acquire_lidar_cloud()`;
  RGB images travel separately in `VisionFrame`. Tests pin six LiDAR returns
  in both `BevFrame` and `PerceptionSnapshot`, including when one camera fails.
  WORLD, RAW BEV, planning, parking, automatic parking, and both AEB paths
  therefore remain LiDAR-backed.
- Camera settings are statically configured as 1280x960, a positive 0.10 s
  requested update, shared memory, streaming, and colour only. Depth,
  annotation, and instance rendering are disabled and have dedicated tests.
- A camera read error is caught per camera, warning once until recovery; it is
  not allowed into the LiDAR poll-failure path. Malformed buffers also omit
  only the affected image.
- LIDAR and eight-camera VISION retain separate acquisition/attachment
  branches. The complete existing offline suite passed.
- The CAMERAS view is enabled only for confirmed HYBRID/VISION modes, falls
  back in LIDAR, and fixes two feeds to a one-row left/right layout. HYBRID
  controls remain enabled while VISION stays gated.
- The suite was run offline in this closure; no BeamNG app or Qt application
  was launched or controlled.
- The branch contains targeted geometry, lifecycle, acquisition, isolation,
  UI-selection, and layout tests. This closure did not perform the separate
  final code-review step.

## Scope audit

The production, test, and README files in the branch match the approved hybrid
scope. Three files outside the direct runtime/test/documentation surface are
already committed coordination metadata from earlier work: `.gitignore`,
`docs/superpowers/plans/2026-08-24-lidar-camera-hybrid.md`, and
`.superpowers/sdd/2026-08-24-lidar-camera-hybrid/task-2-report.md`. They are
not modified by this Task 7 closure; `.gitignore` excludes local agent
workspaces, while the other two record the implementation plan and Task 2.

## Live handoff

The remaining acceptance work is live-only: choose HYBRID, attach, then select
CAMERAS with BeamNG visible and graphics above Lowest. Confirm labels, cabin
occlusion, horizon/pitch, centre overlap, ground/parking-line coverage, and
non-black buffers; then inspect stationary and moving runtime logs for camera
liveness, LiDAR reach, polling, scene warnings, AEB arm state, and camera-loss
isolation.

Concerns: live camera placement, renderer cadence, and runtime-log acceptance
remain outstanding for the root agent; no runtime evidence was fabricated or
log file edited.
