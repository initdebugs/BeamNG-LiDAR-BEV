# Smarter self-driving: predictive planning and natural control

Date: 2026-07-27
Status: approved (design questions answered: 40 km/h cap, two-segment plans,
balanced character, recovery unchanged)

## Goal

Make the self-driving mechanism plan and drive like a competent human at up to
40 km/h: brake **to corner-entry speed** instead of toward a stop, turn in at
the right moment instead of immediately, steer continuously instead of in arc-fan
steps, track the commanded curvature accurately on any vehicle, and modulate
throttle/brake smoothly with no chatter. Recovery (BLOCKED/REVERSING/STUCK) is
deliberately untouched.

## Non-goals

- Semantic planning (the planner stays geometric; documented invariant).
- Route following beyond the existing heading hint.
- Steered reversing (possible follow-up).
- Any change to sensors, frame pipeline, teardown funnels, or gear logic.

## 1. Two-segment predictive planner (`planner.py`)

Today every candidate is a single constant-curvature arc: the planner cannot
represent "continue, then turn", so an upcoming corner appears only as a wall
that shortens free distance, and the controller brakes toward a stop, turns at
crawl speed, then re-accelerates.

**Candidate set** becomes `TRANSITION_DISTANCES_M = (0, 6, 12, 18)` families
x 41 target curvatures (four families, not the five first sketched: the count
was set by measuring the scan against the 40 ms tick, and per-tick re-planning
continuously refines the coarse deferral grid anyway). The curvatures are
**quadratically spaced** (`|k| = K_MAX * u^2`), not uniform — a live-driving
finding: endpoint offset goes as `k * L^2 / 2`, so uniform spacing left 3.75 m
between adjacent lateral offsets at a 30 m lookahead, gentle lane-keeping
corrections were inexpressible, and the planner deferred every correction into
a kerb-to-kerb weave (pinned by `test_driving_loop.py`):

- Segment A: the *current* curvature `k0` (the controller's post-slew curvature,
  passed in as `previous_curvature`), held for `d1` metres.
- Segment B: target curvature `k`, from segment A's endpoint to the horizon.
- The `d1 = 0` family is exactly today's fan and stays the displayed fan.

**Collision scan** stays the vectorized circle-deviation trick, run per family:
segment A is one arc scanned once; for segment B the (decimated, <=4000 point)
obstacle cloud is transformed into segment A's endpoint frame (one 2D
rotate+translate per family) and scanned against all 41 arcs. Family free
distance = `free_A` if `free_A < d1`, else `d1 + free_B` capped at the horizon.
Clearance = min across the driven window `min(free, lookahead)`, split across
segments as today.

**Cost terms** keep their meaning and tuned weights:

- free distance vs `REQUIRED_FREE_DISTANCE_M` — unchanged.
- clearance — unchanged.
- smoothness — scored on the *eventual* curvature change `(k - k0)` identically
  in every family. (Implementation finding: scoring only the immediate command
  gives deferral a smoothness discount, and under per-tick re-planning "later"
  never arrives — the car holds straight forever. Uniform smoothness plus a
  small `COST_TRANSITION` tie-break toward acting now makes deferral win only
  on geometry, and a closed-loop planner test pins the turn-in convergence.)
- keep-right — lateral offset at the lookahead evaluated analytically along the
  composite path.
- nav heading — compared against the candidate's *heading* at the lookahead
  `theta(L) = k0*min(d1, L) + k*max(0, L - d1)`, normalised by `k_max * L`; same
  units, now correct for deferred turns.

**Continuous curvature**: parabolic interpolation of the cost over the winning
family's three candidates around the argmin yields a sub-step curvature; free
distance for the interpolated pick takes the minimum with the one neighbour
the interpolation moved toward — rigorous, because adjacent collision
corridors overlap through the horizon. This removes the 0.05-steering-step
hops the rate limiter currently has to sand down.

**Speed-scaled lookahead**: the worker passes
`clip(LOOKAHEAD_TIME_S * |v|, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)` (2.8 s,
16–30 m) instead of a fixed 20 m, preserving the ~3 s character the 1/L²
keep-right note documents at every speed. `LOOKAHEAD_MAX_M < PLANNER_HORIZON_M`
stays pinned.

**`ArcPlan` gains** `next_curvature`, `transition_distance_m`, `lookahead_m`
(defaults 0.0/0.0/20.0 keep every existing construction site valid;
`curvature` remains the immediate command). `planner.path_polyline(k0, d1, k,
length)` renders the composite path; `bev_widget` draws the chosen path with it
(still provably the planned one) and keeps drawing the fan from the `d1=0`
candidate arrays.

## 2. Natural longitudinal control (`controller.py`)

The PI-to-throttle/brake law is replaced with an **acceleration-domain** law;
gears, modes and every teardown invariant stay put.

```
error   = target - speed
a_des   = clip(SPEED_KV * error, -HARD_DECEL_MPS2, COMFORT_ACCEL_MPS2)
a_des >= 0                      -> throttle = a_des / THROTTLE_GAIN_MPS2 + trim
-COAST_DECEL_MPS2 <= a_des < 0  -> coast: no throttle, no brake (human lift-off)
a_des < -COAST_DECEL_MPS2       -> brake = (-a_des - COAST_DECEL_MPS2) / BRAKE_GAIN_MPS2
```

- `trim` is a slow integrator active only while throttling (anti-windup,
  clamped), replacing `SPEED_KI` as the per-vehicle adaptive feedforward.
- **Slew limits** on both pedals (`THROTTLE_SLEW_*`, `BRAKE_SLEW_*`) supply the
  jerk limiting that makes inputs look human; brake releases faster than it
  applies.
- **Emergency bypass**: entering BLOCKED above ~2 m/s commands full brake
  immediately — safety outranks smoothness. Once stopped, brake tapers to
  `BRAKE_HOLD_FRACTION` instead of a permanent 1.0 stand-on-the-pedal.

**Target speed** becomes corner-aware using the two-segment plan:

```
target = min( MAX_SPEED_MPS,
              sqrt(a_lat / |k_now|),                       # current arc
              sqrt(a_lat / |k_next| + 2*a_comfort*d1),     # brake TO corner entry
              sqrt(2*a_comfort*(free - STOP_MARGIN)) )     # stop within free
```

(the third term is `sqrt(v_corner^2 + 2*a*d1)` — full speed 20 m before a
corner, arriving at corner speed exactly at turn-in).

## 3. Accurate, natural steering (`controller.py`)

- **Curvature-domain slew** replaces the fixed steering-value slew: the
  commanded curvature moves toward the plan at
  `clip(LAT_JERK_MAX_MPS3 / max(v, 1)^2, ..., K_RATE_CEIL_PER_S)` — quick when
  manoeuvring at parking speeds (ceiling matches the old 2.5/s feel), gentle at
  40 km/h where the same curvature step would be a lurch.
- **Closed-loop curvature trim**: the worker passes the vehicle's world heading
  (`atan2` of `vehicle_axes` forward) into `step()`. The controller derives a
  wrap-aware, EMA-filtered yaw rate, measures `k_meas = yaw_rate / v`, and
  adapts a steering gain `g` (clamped ~[0.6, 1.8], slow rate) whenever driving
  a meaningful curvature above ~3 m/s with consistent sign. Output:
  `steering = STEERING_SIGN * clip(g * k_cmd / k_max, -1, 1)`. This closes the
  loop the open-loop `k/k_max` mapping never had: per-vehicle steering ratio and
  understeer stop producing systematic path error. `heading_rad=None` disables
  the whole path (offline tests, missing state).
- `STEERING_SIGN` and its both-direction regression tests are untouched.
- Reversing still steers straight (pinned).

## 4. Speed cap 40 km/h (`config.py`)

- `MAX_SPEED_MPS = 40/3.6`, `COMFORT_DECEL_MPS2 = 2.5`,
  `MAX_LATERAL_ACCEL_MPS2 = 3.0` (balanced character).
- `REQUIRED_FREE_DISTANCE_M` = 11.11²/5 + 4 ≈ 28.7 m — still inside the 35 m
  horizon; the pinned envelope test keeps holding by formula.
- `test_the_speed_cap_is_the_requested_25_kph` is updated to pin 40 — a
  deliberate product change, not a drive-by.

New constants (all commented in config, per house style): `TRANSITION_DISTANCES_M`,
`LOOKAHEAD_TIME_S/MIN/MAX`, `SPEED_KV`, `COMFORT_ACCEL_MPS2`, `HARD_DECEL_MPS2`,
`COAST_DECEL_MPS2`, `THROTTLE_GAIN_MPS2`, `BRAKE_GAIN_MPS2`,
`THROTTLE_SLEW_UP/DOWN_PER_S`, `BRAKE_SLEW_UP/DOWN_PER_S`,
`BRAKE_HOLD_FRACTION`, `HOLD_TAPER_SPEED_MPS`, `LAT_JERK_MAX_MPS3`,
`K_RATE_CEIL_PER_S`, `STEER_GAIN_MIN/MAX`, `STEER_GAIN_ADAPT_RATE`,
`YAW_FILTER_ALPHA`. `SPEED_KP/SPEED_KI/STEER_RATE_PER_S` are retired with the
laws that used them.

## 5. Worker wiring (`worker.py`)

- `_compute_plan` passes: speed-scaled lookahead, the controller's post-slew
  `current_curvature` as `previous_curvature` (more truthful than the last
  plan's output), and `heading_rad` from `vehicle_axes`.
- `_BLIND_ARC` gains the new fields (still zero free distance — blind means
  blocked, pinned).
- The engage-time "Drive check" log line grows the new facts: cap, planner
  families, steering-gain adaptation state.

## 6. Testing

Offline suite pins the arithmetic, as documented:

- planner: deferred-turn corner scenario (winner has `d1 > 0`, immediate
  curvature holds course, `next_curvature` turns); composite free distance
  continuity; segment-B transform consistency against `path_polyline` sampling;
  interpolated curvature bounded by neighbours; all existing gap/keep-right/nav
  pins stay green.
- controller: corner-entry target-speed formula; coast band (no chatter);
  pedal slew bounds per tick; emergency-brake bypass; brake taper at standstill;
  curvature slew respects the speed-scheduled limit; gain adaptation converges
  in a closed-loop kinematic sim against a plant with gain != 1 and never
  exceeds its clamps; heading wrap across ±pi; every existing gear/recovery/
  sign test stays green (only the steering-rate pin is rewritten for the new
  law).
- config: 40 km/h pin, lookahead bounds, transition distances ordered and
  inside the horizon; envelope-fits-horizon unchanged.
- worker: existing safety pins (blind→brake, teardown zeroes controls, gear
  rules) stay green; heading/lookahead plumbing pinned.

Live checks (logged, per house convention): slope allowance at 40 km/h on a
hilly map; steering-gain convergence value in the log; corner approach visibly
brakes to entry speed rather than to a stop.

## 7. Docs

CLAUDE.md's self-driving section is updated in place: two-segment candidates,
acceleration-domain longitudinal law, curvature-domain steering with yaw
feedback, 40 km/h envelope numbers, and the new live-check list.

Note: this directory is not a git repository, so the spec cannot be committed;
it is saved alongside the code instead.
