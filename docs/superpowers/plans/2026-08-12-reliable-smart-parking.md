# Reliable Smart Parking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make automatic parking consume live occupancy, execute a stable committed trajectory, and complete only after a verified stopped pose with the parking brake applied.

**Architecture:** The worker latches a world-space parking job and maintains a parking-only occupancy memory. Corrected Hybrid A* returns cusp-safe, costed trajectories that `ParkingDriver` tracks with real progress and error feedback. Planning and execution share swept-footprint collision geometry.

**Tech Stack:** Python 3.12, NumPy, pytest, PyQt6 worker signals, BeamNGpy vehicle controls.

## Global Constraints

- Preserve the existing uncommitted parking feature and unrelated user edits.
- Keep planning modules Qt-free and BeamNGpy-free.
- Use test-first red/green cycles for every behavior change.
- Parking must fail closed when perception or transmission direction is unknown.
- A successful job applies the parking brake and ends automatic parking.

---

### Task 1: Hybrid A* state and cusp correctness

**Files:**
- Modify: `src/beamng_lidar_bev/hybrid_astar.py`
- Modify: `src/beamng_lidar_bev/reeds_shepp.py`
- Create: `tests/test_hybrid_astar.py`
- Create: `tests/test_reeds_shepp.py`

**Interfaces:**
- Produces: `PlannedPath(poses, expansions, cost)` with `legs()` returning continuous cusp-safe pieces.
- Produces: `plan(..., start_gear: int = 1)` using gear in state identity.

- [ ] Write a failing test proving identical poses reached in opposite gears are distinct search states.
- [ ] Run the focused test and confirm the old three-element key fails it.
- [ ] Add gear to the state key and carry explicit `g` values in frontier entries; skip stale entries.
- [ ] Run the focused test to green.
- [ ] Write failing tests proving every adjacent leg shares the cusp pose and adjacent legs alternate direction.
- [ ] Run them and confirm the current one-sample filtering/analytic-tail gear behavior fails.
- [ ] Give `integrate()` the first segment's gear, reconstruct without a conflicting duplicate tail pose, and split paths while preserving cusp endpoints.
- [ ] Run cusp and endpoint tests to green.
- [ ] Write a failing test proving goal choice compares total penalized cost rather than geometric length.
- [ ] Expose final search cost on `PlannedPath` and consume it in `parking_drive._search_manoeuvre`.
- [ ] Run all Hybrid A*/Reeds-Shepp tests.

### Task 2: Shared swept-footprint collision model

**Files:**
- Modify: `src/beamng_lidar_bev/hybrid_astar.py`
- Modify: `src/beamng_lidar_bev/parking_drive.py`
- Test: `tests/test_hybrid_astar.py`
- Test: `tests/test_parking_drive.py`

**Interfaces:**
- Produces: `Occupancy.motion_cost(start, end, half_width, front, rear)`.
- Produces: live blocking distance measured ahead of a supplied progress index.

- [ ] Write a failing test with a thin obstacle halfway between two 0.7 m primitive endpoints.
- [ ] Confirm endpoint-only collision checking accepts the invalid move.
- [ ] Interpolate primitives at no more than half a cell and check oriented body samples at each pose.
- [ ] Run the test to green.
- [ ] Write failing live-check tests for an obstacle behind current progress and one ahead on a curve.
- [ ] Replace centreline-only `blocking_distance` with the shared oriented swept-footprint calculation starting at current progress.
- [ ] Run collision-focused tests to green.

### Task 3: Committed trajectory tracking

**Files:**
- Modify: `src/beamng_lidar_bev/parking_drive.py`
- Modify: `src/beamng_lidar_bev/config.py`
- Test: `tests/test_parking_drive.py`

**Interfaces:**
- Produces: every `ParkingLeg.path_bay` as a committed trajectory.
- Produces: `ParkingDriveState.cross_track_m` and monotonic path progress.

- [ ] Write failing tests proving canned and searched legs both retain their full path.
- [ ] Convert canned paths to bay-relative committed paths during planning.
- [ ] Run the tests to green.
- [ ] Write a failing test that offsets the vehicle from the path and expects non-zero cross-track error plus corrective steering.
- [ ] Project ego onto the committed path, maintain monotonic progress, and blend curvature feed-forward with cross-track/heading feedback.
- [ ] Run tracking tests to green.
- [ ] Write failing tests for excessive deviation and no-progress replanning.
- [ ] Add bounded event-driven replan triggers without changing the latched goal.
- [ ] Run the complete closed-loop parking controller tests.

### Task 4: Parking occupancy memory and production wiring

**Files:**
- Create: `src/beamng_lidar_bev/parking_map.py`
- Modify: `src/beamng_lidar_bev/worker.py`
- Modify: `src/beamng_lidar_bev/parking_drive.py`
- Test: `tests/test_parking_map.py`
- Test: `tests/test_worker_state.py`

**Interfaces:**
- Produces: `ParkingMap.update(...)`, `ParkingMap.occupancy_bev(...)`, and `ParkingMap.clear()`.
- `ParkingDriver.step(..., occupancy: Occupancy | None)` consumes the snapshot for initial and later plans.

- [ ] Write failing pure tests for world-anchored free/blocked cells surviving ego motion and clearing on teleport/reset.
- [ ] Implement the bounded world-cell store and projection.
- [ ] Run map tests to green.
- [ ] Write a failing worker test proving parking alone activates obstacle extraction and supplies both free and blocked cells to the driver.
- [ ] Include `_parking_driving` in the drive/perception block, update `ParkingMap` from road and planner-band returns, and pass its snapshot into every driver step.
- [ ] Run worker wiring tests to green.

### Task 5: Stable ParkingJob and occupied-bay refusal

**Files:**
- Modify: `src/beamng_lidar_bev/models.py`
- Modify: `src/beamng_lidar_bev/worker.py`
- Modify: `src/beamng_lidar_bev/parking.py`
- Test: `tests/test_worker_state.py`

**Interfaces:**
- Produces: immutable `ParkingJob` containing the latched world-space bay and status.

- [ ] Write a failing test proving a rescan may move the overlay bay but cannot move the active job target.
- [ ] Latch the complete matched `ParkingBay` at engagement and project that bay for driving independently of scan matching.
- [ ] Run the test to green.
- [ ] Write a failing test proving an occupied selected bay cannot engage parking.
- [ ] Reject occupied engagement with a clear status and no actuation.
- [ ] Run job-selection tests to green.

### Task 6: Verified terminal state and parking brake

**Files:**
- Modify: `src/beamng_lidar_bev/models.py`
- Modify: `src/beamng_lidar_bev/parking_drive.py`
- Modify: `src/beamng_lidar_bev/worker.py`
- Test: `tests/test_parking_drive.py`
- Test: `tests/test_worker_state.py`

**Interfaces:**
- Extends `ControlCommand` with `parking_brake: float = 0.0`.
- Produces `PARK_SECURING` before `PARK_ARRIVED`.

- [ ] Write a failing test proving crossing the endpoint at 0.68 m/s is not `ARRIVED`.
- [ ] Add pose/heading/speed validation and a stopped dwell in `PARK_SECURING`.
- [ ] Run the test to green.
- [ ] Write a failing worker test proving success sends parking brake 1.0, emits completion, disables active parking, and does not immediately send the normal release command.
- [ ] Route `parking_brake` through `_actuate` and add a completion path separate from disengagement.
- [ ] Run terminal-state tests to green.

### Task 7: Stable blockage and signed transmission validation

**Files:**
- Modify: `src/beamng_lidar_bev/parking_drive.py`
- Modify: `src/beamng_lidar_bev/worker.py`
- Modify: `src/beamng_lidar_bev/config.py`
- Test: `tests/test_parking_drive.py`
- Test: `tests/test_worker_state.py`

**Interfaces:**
- `ParkingDriver.step` consumes signed forward speed.
- Produces latched `PARK_BLOCKED`/waiting behavior followed by clear-dwell resume or bounded replan.

- [ ] Write a failing test proving an AEB stop cannot alternate `BLOCKED/BACKING` before a stopped clear dwell.
- [ ] Latch blockage, brake to zero, and require clear dwell before resuming or replanning.
- [ ] Run the test to green.
- [ ] Write failing tests for unreadable gear with unexpected motion direction and for confirmed requested-direction motion.
- [ ] Remove timeout-as-confirmation, consume signed speed, and fail stopped on direction disagreement.
- [ ] Run gear and blockage tests to green.

### Task 8: Full verification and documentation alignment

**Files:**
- Modify: `CLAUDE.md`
- Verify: `src/beamng_lidar_bev/**/*.py`
- Verify: `tests/**/*.py`

- [ ] Run all focused parking, map, Hybrid A*, Reeds-Shepp, and worker-state tests.
- [ ] Run the complete pytest suite.
- [ ] Run Ruff across `src` and `tests`.
- [ ] Review `git diff --check` and the scoped diff for accidental changes.
- [ ] Update the parking architecture notes and stale “forward nose-in only” log text to match verified behavior.
