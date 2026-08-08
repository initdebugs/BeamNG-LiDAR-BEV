# Smarter Self-Driving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two-segment predictive path planning, acceleration-domain throttle/brake, curvature-domain steering with yaw-rate feedback, and a 40 km/h cap — per `docs/superpowers/specs/2026-07-27-smarter-self-driving-design.md`.

**Architecture:** All changes live in the existing pure layers (`config` → `models` → `planner`/`controller`) plus thin wiring in `worker` and drawing in `bev_widget`. New `ArcPlan` fields carry defaults so every construction site stays valid between tasks. Module boundaries, teardown funnels, gear logic and recovery states are untouched.

**Tech Stack:** Python 3.11 (on this machine; `py -3.12` resolves to it), numpy, pytest, ruff. No Qt/BeamNGpy in planner/controller.

## Global Constraints

- **No git repository here** — every "commit" becomes a verify step (pytest + ruff). Do not `git init`.
- Test command: `py -3.12 -m pytest tests/<file> -v` (pythonpath comes from pyproject). Lint: `py -3.12 -m ruff check src tests`.
- Qt-importing test modules (`test_worker_state.py`, etc.) cannot collect until PyQt6<6.8 is installed (Task 7). Until then run the pure modules' tests explicitly.
- Planner sign convention: **positive curvature = LEFT**; `STEERING_SIGN = -1.0` reconciles BeamNG's positive-RIGHT input and must not change.
- BEV heading rotation: a left turn by θ is the standard CCW matrix `[[cosθ, −sinθ],[sinθ, cosθ]]` in `(right, forward)` coordinates. Arc endpoint for curvature k after length s: `x = −(1−cos(ks))/k, y = sin(ks)/k` (straight when |k| < 1e−4). Heading after s: `θ = k·s`.
- Point-cloud work stays vectorized — no per-point Python loops in the hot path.
- Every module keeps `from __future__ import annotations`; ruff line length 88.

---

### Task 1: Config constants + config tests

**Files:**
- Modify: `src/beamng_lidar_bev/config.py` (self-driving section)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces (all imported by later tasks): `MAX_SPEED_MPS = 40.0/3.6`, `COMFORT_DECEL_MPS2 = 2.5`, `MAX_LATERAL_ACCEL_MPS2 = 3.0`, `COMFORT_ACCEL_MPS2 = 2.0`, `HARD_DECEL_MPS2 = 4.5`, `COAST_DECEL_MPS2 = 0.7`, `SPEED_KV = 0.9`, `THROTTLE_GAIN_MPS2 = 3.5`, `BRAKE_GAIN_MPS2 = 6.0`, `TRIM_RATE_PER_S = 0.02`, `TRIM_MAX = 0.35`, `THROTTLE_SLEW_UP_PER_S = 1.2`, `THROTTLE_SLEW_DOWN_PER_S = 4.0`, `BRAKE_SLEW_UP_PER_S = 2.0`, `BRAKE_SLEW_DOWN_PER_S = 3.5`, `BRAKE_HOLD_FRACTION = 0.35`, `HOLD_TAPER_SPEED_MPS = 2.0`, `LAT_JERK_MAX_MPS3 = 2.5`, `K_RATE_CEIL_PER_S = 0.42`, `STEER_GAIN_MIN = 0.6`, `STEER_GAIN_MAX = 1.8`, `STEER_GAIN_ADAPT_RATE = 0.15`, `STEER_GAIN_MIN_SPEED_MPS = 3.0`, `STEER_GAIN_MIN_CURVATURE = 0.02`, `YAW_FILTER_ALPHA = 0.3`, `TRANSITION_DISTANCES_M = (0.0, 5.0, 10.0, 15.0, 20.0)`, `LOOKAHEAD_TIME_S = 2.8`, `LOOKAHEAD_MIN_M = 16.0`, `LOOKAHEAD_MAX_M = 30.0`.
- `SPEED_KP`, `SPEED_KI`, `STEER_RATE_PER_S` stay **until Tasks 3/4 remove their last consumers**, then are deleted there.

- [ ] **Step 1: Update the failing config tests first.** In `tests/test_config.py`: rename/replace `test_the_speed_cap_is_the_requested_25_kph` with

```python
def test_the_speed_cap_is_the_requested_40_kph() -> None:
    assert MAX_SPEED_MPS * 3.6 == 40.0


def test_the_lookahead_window_is_ordered_and_visible() -> None:
    assert LOOKAHEAD_MIN_M < LOOKAHEAD_MAX_M < PLANNER_HORIZON_M
    # PLANNER_LOOKAHEAD_M remains the static default inside the window.
    assert LOOKAHEAD_MIN_M <= PLANNER_LOOKAHEAD_M <= LOOKAHEAD_MAX_M


def test_transition_distances_cover_the_braking_envelope() -> None:
    distances = TRANSITION_DISTANCES_M
    assert distances[0] == 0.0  # today's fan is always a candidate family
    assert list(distances) == sorted(distances)
    # A turn deferred further than the car can brake toward is a turn the
    # speed law cannot prepare for.
    assert distances[-1] <= REQUIRED_FREE_DISTANCE_M
```

- [ ] **Step 2: Run to verify failure.** `py -3.12 -m pytest tests/test_config.py -v` — expect ImportError/AssertionError on the new names.
- [ ] **Step 3: Implement config changes.** Set the three changed values, add the new constants in the self-driving section with house-style comments (why each number, not what it is). Key comments: 40 km/h envelope = 11.11²/(2·2.5)+4 ≈ 28.7 m inside the 35 m horizon; `COAST_DECEL` is the engine-drag band a human covers by lifting off; `K_RATE_CEIL` ≈ old 2.5/s steering slew × k_max; `LAT_JERK_MAX/v²` is the curvature slew at speed; transition distances are the "keep going, then turn" deferral options.
- [ ] **Step 4: Verify.** `py -3.12 -m pytest tests/test_config.py -v` → all pass. `py -3.12 -m ruff check src tests` → clean. (`test_controller.py`/`test_planner.py` still pass — nothing they import changed values they pin by formula.)

---

### Task 2: Two-segment planner + models

**Files:**
- Modify: `src/beamng_lidar_bev/models.py` (`ArcPlan`), `src/beamng_lidar_bev/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Produces: `ArcPlan` gains `next_curvature: float = 0.0`, `transition_distance_m: float = 0.0`, `lookahead_m: float = 20.0` (defaults keep worker/tests valid until Task 5). `curvature` stays the *immediate* command.
- Produces: `planner.path_polyline(curvature0, transition_m, curvature1, length_m, samples=32) -> np.ndarray` (BEV (N,2) float32, starts at origin).
- `plan_arc` signature unchanged; `previous_curvature` is now documented as "the curvature currently being driven (controller post-slew)" and is segment A's curvature for the deferred families.

- [ ] **Step 1: Write the failing planner tests** (append to `tests/test_planner.py`):

```python
def _corner_corridor() -> np.ndarray:
    """A corridor that ends in a left exit: hold straight, then turn."""
    return np.concatenate(
        (
            _rail(-2.5, 1.0, 13.0),   # left kerb, ends where the exit opens
            _rail(2.5, 1.0, 30.0),    # right kerb, continuous
            _wall(-12.0, 2.5, 22.0),  # the road ahead is closed
        )
    )


def test_a_corner_ahead_is_planned_as_hold_then_turn() -> None:
    plan = plan_arc(_corner_corridor(), GEOMETRY, previous_curvature=0.0)

    assert plan.transition_distance_m > 0.0
    assert plan.next_curvature > 0.0            # the exit is to the left
    assert plan.curvature == pytest.approx(0.0, abs=1e-6)  # hold course now
    assert plan.free_distance_m > 22.0          # past the wall via the exit


def test_the_chosen_path_clears_every_obstacle() -> None:
    rng = np.random.default_rng(7)
    obstacles = np.column_stack(
        (rng.uniform(-10.0, 10.0, 60), rng.uniform(2.0, 30.0, 60))
    ).astype(np.float32)

    plan = plan_arc(obstacles, GEOMETRY, previous_curvature=0.05)

    driven = max(plan.free_distance_m - 0.5, 0.1)
    path = path_polyline(
        plan.curvature, plan.transition_distance_m, plan.next_curvature, driven
    )
    gaps = np.hypot(
        obstacles[:, 0:1] - path[:, 0][None, :],
        obstacles[:, 1:2] - path[:, 1][None, :],
    ).min(axis=1)
    half_width = GEOMETRY.width_m / 2.0 + CLEARANCE_MARGIN_M
    # Sampling the path at 32 points leaves small chord gaps; 0.25 m covers it.
    assert gaps.min() >= half_width - 0.25


def test_path_polyline_reduces_to_a_single_arc() -> None:
    np.testing.assert_allclose(
        path_polyline(0.1, 0.0, 0.1, 15.0, samples=12),
        arc_polyline(0.1, 15.0, samples=12),
        atol=1e-5,
    )


def test_path_polyline_straight_then_turn_offsets_the_turn() -> None:
    path = path_polyline(0.0, 10.0, 0.1, 20.0, samples=64)

    early = path[path[:, 1] <= 9.9]
    np.testing.assert_allclose(early[:, 0], 0.0, atol=1e-6)  # straight prefix
    assert path[-1, 0] < -1.0  # the turn bends left after the prefix
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert float(segments.sum()) == pytest.approx(20.0, rel=5e-3)


def test_curvature_interpolates_between_fan_steps() -> None:
    # A slightly offset obstacle should move the answer by less than one
    # fan step, which is only possible with sub-step interpolation.
    base = np.asarray(((1.6, 12.0),), dtype=np.float32)
    nudged = np.asarray(((1.75, 12.0),), dtype=np.float32)

    step = (2.0 / MIN_TURN_RADIUS_M) / (plan_arc(
        base, GEOMETRY).candidate_curvatures.size - 1)
    delta = abs(
        plan_arc(base, GEOMETRY).next_curvature
        - plan_arc(nudged, GEOMETRY).next_curvature
    )
    assert 0.0 < delta < step
```

  Also update the two imports at the top (`path_polyline`, `CLEARANCE_MARGIN_M`).

- [ ] **Step 2: Run to verify failure.** `py -3.12 -m pytest tests/test_planner.py -v` — new tests fail on import.
- [ ] **Step 3: Implement.**

  `models.ArcPlan` — add after `curvature`:

```python
    next_curvature: float = 0.0
    """Segment-B curvature: what the path bends to after the transition."""
    transition_distance_m: float = 0.0
    """Metres of `curvature` driven before bending to `next_curvature`."""
    lookahead_m: float = 20.0
    """Where keep-right/nav were evaluated; the worker scales it with speed."""
```

  (dataclass field ordering: the three defaults must sit after the existing
  non-default fields — append at the end of the class instead if ordering
  fights the existing defaults.)

  `planner.py` — implementation outline (all vectorized):

```python
def _arc_endpoint(curvature: float, length: float) -> tuple[float, float, float]:
    """(x, y, heading) after driving `length` along one arc."""
    k = float(curvature)
    if abs(k) < _MIN_CURVATURE:
        return 0.0, float(length), 0.0
    return -(1.0 - np.cos(k * length)) / k, np.sin(k * length) / k, k * length


def path_polyline(curvature0, transition_m, curvature1, length_m, samples=32):
    d1 = float(min(max(transition_m, 0.0), length_m))
    if d1 <= 1e-6 or abs(curvature0 - curvature1) < 1e-9:
        return arc_polyline(curvature1, length_m, samples)
    n_a = max(2, int(round(samples * d1 / length_m)))
    part_a = arc_polyline(curvature0, d1, n_a)
    x1, y1, theta1 = _arc_endpoint(curvature0, d1)
    part_b = arc_polyline(curvature1, length_m - d1, max(2, samples - n_a))
    cos_t, sin_t = np.cos(theta1), np.sin(theta1)
    rotated = np.column_stack((
        cos_t * part_b[:, 0] - sin_t * part_b[:, 1],
        sin_t * part_b[:, 0] + cos_t * part_b[:, 1],
    ))
    return np.concatenate(
        (part_a, (rotated + (x1, y1))[1:])
    ).astype(np.float32)
```

  `plan_arc` restructure — factor the existing scan into a helper so segment A
  (one arc) and every segment-B family (41 arcs against transformed points)
  reuse it:

```python
def _scan_arcs(points, curvatures, half_width, horizon):
    """(free, deviation, progress) for each arc against `points`."""
    # exactly the existing radius/centre/swept maths, returning
    # free: (A,), deviation/progress: (N, A) for the clearance windows
```

  Per family `d1 > 0` with `k0 = previous_curvature`:
  1. `free_a, dev_a, prog_a = _scan_arcs(points, [k0], ...)` (compute once).
  2. `x1, y1, theta1 = _arc_endpoint(k0, d1)`; transform
     `q = (points - (x1, y1)) @ [[cos, sin], [-sin, cos]]`
     (i.e. rotate by −θ1: `qx = c*dx + s*dy`, `qy = -s*dx + c*dy`).
  3. `free_b, dev_b, prog_b = _scan_arcs(q, CANDIDATE_CURVATURES, ...)`.
  4. `free = where(free_a < d1, free_a, minimum(d1 + free_b, horizon))`.
  5. Clearance window `w = minimum(free, lookahead)`: segment A part uses
     `prog_a <= minimum(w, d1)`, segment B part uses `prog_b <= w - d1`;
     composite clearance = elementwise min, `MAX_CORRIDOR_HALF_WIDTH_M`
     when nothing lands in the window.
  6. Cost terms:
     - free-distance and clearance: identical formulas to today.
     - smoothness: immediate command is `k0`, so the term is 0 for the whole
       family (documented: deferring a turn is exactly as smooth as holding
       course); for the `d1 = 0` family it stays `((k - prev)/k_max)^2`.
     - keep-right: lateral offset at the lookahead along the composite path —
       `L <= d1`: `_lateral_offsets([k0], L)`; else
       `x1 + cosθ1·bx(k, L-d1) - sinθ1·by(k, L-d1)` with `bx, by` the arc
       endpoint formulas, vectorized over k.
     - nav heading: candidate heading `theta_L = k0*min(d1, L) + k*max(L-d1, 0)`
       against `nav_heading_rad`, normalised by `MAX_CURVATURE * L`:
       `COST_NAV_HEADING * ((theta_L - nav) / (MAX_CURVATURE * L))**2`.
       The `d1 = 0` family uses the same heading form (`k*L` vs nav), which
       equals the old curvature form after dividing by L.
  7. Stack cost `(len(TRANSITION_DISTANCES_M), 41)`, flat argmin → family, i.
  8. Parabolic interpolation inside the winning family over `i-1, i, i+1`
     (skip at edges / non-finite): `delta = 0.5*(c0 - c2)/(c0 - 2*c1 + c2)`
     clamped to ±0.5; `k_star = k_i + delta*step`;
     `free_star = min(free[i-1:i+2])` (conservative).
  9. `ArcPlan(curvature = k0 if family_d1 > 0 else k_star,
              next_curvature = k_star, transition_distance_m = family_d1,
              lookahead_m = lookahead_m, ...)`; candidate arrays stay the
      `d1 = 0` family's rows so the displayed fan keeps its meaning.

  Guard: when `abs(previous_curvature) >= MAX_CURVATURE` clamp k0 into range.
  Empty cloud: all families identical → argmin lands in `d1=0` family (order
  the stack so `d1=0` is row 0; np.argmin takes the first minimum), preserving
  today's straight-ahead tie-break.

- [ ] **Step 4: Verify.** `py -3.12 -m pytest tests/test_planner.py tests/test_config.py -v` → all pass, including every pre-existing pin (gap steering, nav, keep-right, smoothness, caps).
- [ ] **Step 5: Sanity-time the scan.** Scratchpad script: 4000 random obstacles, 100 calls to `plan_arc`, report mean ms. Budget: < 8 ms/call on this machine (5 families ≈ 5× the old sub-millisecond scan). Record the number for the CLAUDE.md update in Task 7.

---

### Task 3: Acceleration-domain longitudinal control

**Files:**
- Modify: `src/beamng_lidar_bev/controller.py`, `src/beamng_lidar_bev/config.py` (delete `SPEED_KP`, `SPEED_KI`)
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: Task 1 constants; `ArcPlan.next_curvature` / `.transition_distance_m` from Task 2.
- Produces: same `ControlCommand`; `DrivingController._longitudinal(target, speed, dt)` replaced; `_target_speed(plan)` corner-entry aware. No signature changes yet (Task 4 adds `heading_rad`).

- [ ] **Step 1: Write the failing tests** (append/replace in `tests/test_controller.py`):

```python
def test_target_speed_brakes_to_corner_entry_not_to_zero() -> None:
    plan = ArcPlan(
        curvature=0.0, free_distance_m=30.0, clearance_m=3.0,
        keep_right_target_m=None, nav_heading_rad=None,
        candidate_curvatures=_EMPTY, candidate_costs=_EMPTY,
        candidate_free_distances=_EMPTY,
        next_curvature=1.0 / 6.0, transition_distance_m=10.0,
    )
    command = DrivingController().step(plan, 8.0, DT)

    entry = np.sqrt(
        MAX_LATERAL_ACCEL_MPS2 * 6.0 + 2.0 * COMFORT_DECEL_MPS2 * 10.0
    )
    assert command.target_speed_mps == pytest.approx(entry, rel=1e-6)
    assert command.target_speed_mps > np.sqrt(MAX_LATERAL_ACCEL_MPS2 * 6.0)


def test_slightly_over_target_coasts_instead_of_braking() -> None:
    controller = DrivingController()
    # 0.5 m/s over: inside the lift-off band, so neither pedal moves.
    command = _run(controller, _plan(), MAX_SPEED_MPS + 0.5, 10)

    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_throttle_ramps_at_the_slew_limit_from_rest() -> None:
    controller = DrivingController()
    first = controller.step(_plan(), 0.0, DT)
    second = controller.step(_plan(), 0.0, DT)

    assert first.throttle == pytest.approx(THROTTLE_SLEW_UP_PER_S * DT)
    assert second.throttle == pytest.approx(2 * THROTTLE_SLEW_UP_PER_S * DT)


def test_an_emergency_stop_brakes_immediately_and_fully() -> None:
    command = DrivingController().step(_plan(free_distance_m=1.0), 8.0, DT)

    assert command.mode == "BLOCKED"
    assert command.brake == 1.0


def test_the_brake_relaxes_to_a_hold_once_stopped() -> None:
    controller = DrivingController()
    command = _run(controller, _plan(free_distance_m=1.0), 0.0, 5)

    assert command.brake == pytest.approx(BRAKE_HOLD_FRACTION)
```

  Update imports; `test_it_brakes_when_above_the_target_speed` must use a
  speed well beyond the coast band (`MAX_SPEED_MPS + 3.0` already is).

- [ ] **Step 2: Run to verify failure.** `py -3.12 -m pytest tests/test_controller.py -v`.
- [ ] **Step 3: Implement.**

```python
def _slew(previous, target, up_rate, down_rate, dt):
    if target >= previous:
        return min(target, previous + up_rate * dt)
    return max(target, previous - down_rate * dt)


# state: self._throttle = 0.0, self._brake = 0.0, self._trim = 0.0 (reset())

def _longitudinal(self, target, speed, dt):
    error = target - speed
    a_des = max(-HARD_DECEL_MPS2, min(COMFORT_ACCEL_MPS2, SPEED_KV * error))
    if a_des >= 0.0:
        raw = a_des / THROTTLE_GAIN_MPS2 + self._trim
        if raw < 0.95:  # anti-windup: stop trimming into saturation
            self._trim = max(
                0.0, min(TRIM_MAX, self._trim + TRIM_RATE_PER_S * error * dt)
            )
        throttle_target, brake_target = min(1.0, raw), 0.0
    elif a_des >= -COAST_DECEL_MPS2:
        throttle_target, brake_target = 0.0, 0.0   # human lift-off
    else:
        throttle_target = 0.0
        brake_target = min(1.0, (-a_des - COAST_DECEL_MPS2) / BRAKE_GAIN_MPS2)
    self._throttle = _slew(self._throttle, throttle_target,
                           THROTTLE_SLEW_UP_PER_S, THROTTLE_SLEW_DOWN_PER_S, dt)
    self._brake = _slew(self._brake, brake_target,
                        BRAKE_SLEW_UP_PER_S, BRAKE_SLEW_DOWN_PER_S, dt)
    return self._throttle, self._brake


@staticmethod
def _target_speed(plan):
    target = MAX_SPEED_MPS
    if abs(plan.curvature) > 1e-6:
        target = min(target, math.sqrt(MAX_LATERAL_ACCEL_MPS2 / abs(plan.curvature)))
    if abs(plan.next_curvature) > 1e-6:
        corner = MAX_LATERAL_ACCEL_MPS2 / abs(plan.next_curvature)
        target = min(target, math.sqrt(
            corner + 2.0 * COMFORT_DECEL_MPS2 * max(plan.transition_distance_m, 0.0)
        ))
    headroom = max(0.0, plan.free_distance_m - STOP_MARGIN_M)
    return min(target, math.sqrt(2.0 * COMFORT_DECEL_MPS2 * headroom))
```

  `_blocked`/`_enter`'s hold branch: replace `brake=1.0` with

```python
def _hold_brake(self, speed, dt):
    if abs(speed) > HOLD_TAPER_SPEED_MPS:
        self._brake = 1.0          # emergency: bypass the slews entirely
    else:
        taper = BRAKE_HOLD_FRACTION + (1.0 - BRAKE_HOLD_FRACTION) * (
            abs(speed) / HOLD_TAPER_SPEED_MPS
        )
        self._brake = _slew(self._brake, taper,
                            BRAKE_SLEW_UP_PER_S, BRAKE_SLEW_DOWN_PER_S, dt)
    self._throttle = 0.0
    return self._brake
```

  `_reverse` keeps calling `_longitudinal(REVERSE_SPEED_MPS, abs(speed), dt)`.
  Delete `_MAX_INTEGRAL`, `self._integral` (including the `_enter` reset —
  `_trim` deliberately survives mode changes; it encodes the vehicle, not the
  situation. Pedal slew states also persist for continuity). Remove
  `SPEED_KP`/`SPEED_KI` from config and all imports.

- [ ] **Step 4: Verify.** `py -3.12 -m pytest tests/test_controller.py tests/test_config.py -v` → all pass (recovery, gear and sign pins untouched).

---

### Task 4: Curvature-domain steering with yaw feedback

**Files:**
- Modify: `src/beamng_lidar_bev/controller.py`, `src/beamng_lidar_bev/config.py` (delete `STEER_RATE_PER_S`)
- Test: `tests/test_controller.py`

**Interfaces:**
- Produces: `step(..., heading_rad: float | None = None)` keyword; properties `current_curvature: float`, `steering_gain: float`, `measured_curvature: float | None` (worker log + tests).

- [ ] **Step 1: Write the failing tests:**

```python
def test_curvature_slew_is_speed_scheduled() -> None:
    fast, slow = DrivingController(), DrivingController()
    fast.step(_plan(curvature=1.0 / 6.0), 10.0, DT)
    slow.step(_plan(curvature=1.0 / 6.0), 0.0, DT)

    assert fast.current_curvature == pytest.approx(
        (LAT_JERK_MAX_MPS3 / 100.0) * DT
    )
    assert slow.current_curvature == pytest.approx(K_RATE_CEIL_PER_S * DT)


def test_steering_gain_adapts_to_an_understeering_plant() -> None:
    controller = DrivingController()
    heading, speed, plant_gain = 0.0, 6.0, 0.7
    for _ in range(3000):  # 120 simulated seconds
        command = controller.step(
            _plan(curvature=0.08), speed, DT, heading_rad=heading
        )
        real_k = plant_gain * (command.steering / STEERING_SIGN) * (
            1.0 / MIN_TURN_RADIUS_M
        )
        heading += speed * real_k * DT

    assert 1.2 < controller.steering_gain < 1.6   # ~1/0.7, inside the clamps


def test_the_gain_never_leaves_its_clamps() -> None:
    controller = DrivingController()
    heading = 0.0
    for _ in range(2000):
        command = controller.step(
            _plan(curvature=0.1), 6.0, DT, heading_rad=heading
        )
        # A wildly oversteering plant: it should pin at the clamp, not run away.
        heading += 6.0 * 5.0 * (command.steering / STEERING_SIGN) * DT

    assert controller.steering_gain >= STEER_GAIN_MIN


def test_heading_wrap_does_not_spike_the_yaw_estimate() -> None:
    controller = DrivingController()
    controller.step(_plan(), 5.0, DT, heading_rad=3.10)
    controller.step(_plan(), 5.0, DT, heading_rad=-3.10)  # crossed +/- pi

    measured = controller.measured_curvature
    assert measured is not None
    # Unwrapped: |dh| = 2*pi - 6.2 = 0.083 rad -> |k| < 1; the naive diff
    # would have read 6.2 rad in 40 ms -> |k| = 31.
    assert abs(measured) < 1.0


def test_without_a_heading_the_gain_stays_nominal() -> None:
    controller = DrivingController()
    _run(controller, _plan(curvature=0.1), 6.0, 100)

    assert controller.steering_gain == pytest.approx(1.0)
```

  Rewrite `test_steering_is_rate_limited` for the new law (speed 0 → ceiling):

```python
def test_steering_is_rate_limited() -> None:
    controller = DrivingController()
    command = controller.step(_plan(curvature=1.0 / 6.0), 0.0, DT)

    assert abs(command.steering) == pytest.approx(
        (K_RATE_CEIL_PER_S * DT) / (1.0 / MIN_TURN_RADIUS_M)
    )
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**

```python
# state (reset()): self._curvature = 0.0, self._gain = 1.0,
#                  self._last_heading = None, self._yaw_filtered = None

def step(self, plan, forward_speed_mps, dt, rear_free_distance_m=...,
         reported_gear=None, heading_rad=None):
    ...
    self._observe_yaw(heading_rad, float(forward_speed_mps), dt)
    # dispatch unchanged

def _observe_yaw(self, heading_rad, speed, dt):
    if heading_rad is None:
        return
    if self._last_heading is not None:
        delta = math.remainder(heading_rad - self._last_heading, math.tau)
        yaw_rate = delta / dt
        self._yaw_filtered = (
            yaw_rate if self._yaw_filtered is None
            else YAW_FILTER_ALPHA * yaw_rate
            + (1.0 - YAW_FILTER_ALPHA) * self._yaw_filtered
        )
    self._last_heading = heading_rad

@property
def measured_curvature(self):
    if self._yaw_filtered is None:
        return None
    return self._yaw_filtered  # divided by speed at the use site

def _adapt_gain(self, speed, dt):
    # Only while genuinely cornering, moving, and turning the way we asked.
    if self._yaw_filtered is None or self._mode != DRIVING:
        return
    if speed < STEER_GAIN_MIN_SPEED_MPS:
        return
    if abs(self._curvature) < STEER_GAIN_MIN_CURVATURE:
        return
    measured = self._yaw_filtered / max(speed, 1.0)
    if measured * self._curvature <= 0.0:
        return
    ratio = min(3.0, measured / self._curvature)
    self._gain = max(STEER_GAIN_MIN, min(
        STEER_GAIN_MAX,
        self._gain + STEER_GAIN_ADAPT_RATE * (1.0 - ratio) * dt,
    ))

def _steer(self, curvature, dt, speed=0.0):
    rate = min(K_RATE_CEIL_PER_S,
               LAT_JERK_MAX_MPS3 / max(speed, 1.0) ** 2)
    target = max(-_MAX_CURVATURE, min(_MAX_CURVATURE, curvature))
    self._curvature = max(self._curvature - rate * dt,
                          min(self._curvature + rate * dt, target))
    self._adapt_gain(speed, dt)
    steering = STEERING_SIGN * self._gain * self._curvature / _MAX_CURVATURE
    return max(-1.0, min(1.0, steering))
```

  Note `measured_curvature` property should return yaw/`max(speed,1)` — store
  the last speed in `step` (`self._last_speed_abs = abs(forward_speed_mps)`)
  and divide there so the test reads curvature, not yaw. Call sites pass
  `speed`: `_drive` → `self._steer(plan.curvature, dt, abs(speed))`; blocked/
  reverse paths steer toward 0.0 at their (low) speeds. `current_curvature`
  and `steering_gain` are plain read-only properties. Delete
  `STEER_RATE_PER_S` from config and imports. `_MIN_CURVATURE`-style epsilon
  guards: `math.remainder(x, math.tau)` is the wrap; python ≥3.8 stdlib.

- [ ] **Step 4: Verify.** `py -3.12 -m pytest tests/test_controller.py tests/test_planner.py tests/test_config.py -v` and ruff → clean.

---

### Task 5: Worker wiring

**Files:**
- Modify: `src/beamng_lidar_bev/worker.py`
- Test: `tests/test_worker_state.py`

**Interfaces:**
- Consumes: `plan_arc(..., lookahead_m=...)`, `controller.current_curvature`, `controller.step(..., heading_rad=...)`, new `ArcPlan` fields.
- Produces: `_BLIND_ARC` with explicit `next_curvature=0.0, transition_distance_m=0.0`; `_compute_plan` passes heading + dynamic lookahead.

- [ ] **Step 1: Write the failing tests** (match the file's existing unbound-method + SimpleNamespace style):

```python
def test_the_lookahead_scales_with_speed(worker_ns) -> None:
    # exact shape depends on the file's fixtures; pin:
    #   at 3 m/s  -> plan.arc.lookahead_m == LOOKAHEAD_MIN_M
    #   at 11 m/s -> plan.arc.lookahead_m == pytest.approx(30.0)  (clamped max)


def test_the_controller_receives_the_vehicle_heading(worker_ns) -> None:
    # stub state dir=(0, 1, 0): heading = atan2(1, 0) = pi/2; assert the
    # controller stub captured heading_rad == pytest.approx(math.pi / 2)
```

  (Write them concretely against the real fixtures in the file; drive
  `_compute_plan` directly like the neighbouring tests do.)

- [ ] **Step 2: Run to verify failure** — note these two plus the whole module need PyQt6; if collection fails, run Task 7's PyQt6 pin first, then return here.
- [ ] **Step 3: Implement.** In `_compute_plan`:

```python
_, forward, _ = vehicle_axes(state)
forward_speed = float(vec3(state.get("vel", (0.0, 0.0, 0.0))) @ forward)
heading = float(np.arctan2(forward[1], forward[0]))
lookahead = min(LOOKAHEAD_MAX_M,
                max(LOOKAHEAD_MIN_M, LOOKAHEAD_TIME_S * abs(forward_speed)))
previous = (
    self._controller.current_curvature if self._controller is not None else 0.0
)
# plan_arc(..., previous_curvature=previous, lookahead_m=lookahead)
# controller.step(..., heading_rad=heading)
```

  Drop `self._last_curvature` (its three touch points) — the controller now
  owns the driven curvature. `_BLIND_ARC` gains the explicit new fields.
  Extend the Drive-check log line with: speed cap ×3.6, `len(TRANSITION_DISTANCES_M)`
  plan families, lookahead window, "yaw-gain adaptation on".

- [ ] **Step 4: Verify.** Full suite: `py -3.12 -m pytest -v` (post-PyQt6 pin) — every safety pin (blind→brake, teardown zeroes, gear rules, parking brake) must stay green. Ruff clean.

---

### Task 6: BEV widget draws the composite path

**Files:**
- Modify: `src/beamng_lidar_bev/bev_widget.py`

**Interfaces:**
- Consumes: `planner.path_polyline`, `ArcPlan.next_curvature/transition_distance_m/lookahead_m`.

- [ ] **Step 1: Implement (no offline test can see pixels; keep it minimal).** In `_draw_plan`: the chosen-path polyline becomes

```python
points = path_polyline(
    arc.curvature, arc.transition_distance_m, arc.next_curvature,
    max(arc.free_distance_m, 4.0), samples=40,
)
```

  while the candidate fan keeps using `arc_polyline` over the `d1=0`
  candidate arrays (unchanged meaning: "what could I do right now").
  Anywhere `PLANNER_LOOKAHEAD_M` positioned the keep-right/nav markers, use
  `arc.lookahead_m`; drop the config import if now unused.
- [ ] **Step 2: Verify.** `py -3.12 -m pytest -v` (widget module imports in Qt tests), ruff clean.

---

### Task 7: Environment fix, full verification, docs

**Files:**
- Modify: `requirements.txt` (PyQt6 pin), `CLAUDE.md`

- [ ] **Step 1: Pin PyQt6 so the Qt test modules can collect on this machine** (CLAUDE.md-documented remedy: 6.11.0 cannot load on the installed Python 3.11.0):
  `py -3.12 -m pip install "PyQt6<6.8"` and change the requirements pin to `PyQt6<6.8`.
- [ ] **Step 2: Full verification.** `py -3.12 -m pytest -v` → whole suite green. `py -3.12 -m ruff check src tests` → clean.
- [ ] **Step 3: Update CLAUDE.md's self-driving + configuration sections in place:** two-segment candidate families and why deferral is not procrastination; the corner-entry speed law; acceleration-domain pedals (coast band, slews, hold taper, emergency bypass); curvature-domain steering with the yaw-gain loop and its enable conditions; 40 km/h envelope numbers (28.7 m < 35 m); planner timing measured in Task 2; the new live-check list (slope allowance at 40 on hills, gain convergence in the log, corner approach brakes to entry speed).
- [ ] **Step 4: Re-read the spec, confirm every section maps to landed code; fix anything missed.**
