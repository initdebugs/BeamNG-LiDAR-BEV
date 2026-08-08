"""
Longitudinal and lateral control, plus the blocked/reverse recovery.

Pure Python: no Qt, no BeamNGpy, no wall clock. Elapsed time arrives as ``dt`` so
the state machine is fully deterministic under test.

The controller owns the only mutable driving state in the app. Everything it
produces is a `ControlCommand`, which the worker hands straight to
``Vehicle.control``.
"""

from __future__ import annotations

import math

from .config import (
    BLOCKED_HOLD_S,
    BRAKE_GAIN_MPS2,
    BRAKE_HOLD_FRACTION,
    BRAKE_RELEASE_FRACTION,
    BRAKE_SLEW_DOWN_PER_S,
    BRAKE_SLEW_UP_PER_S,
    COAST_DECEL_MPS2,
    COMFORT_ACCEL_MPS2,
    COMFORT_DECEL_MPS2,
    CORNERING_ACCEL_MPS2,
    ENGINE_DRAG_MPS2,
    HARD_DECEL_MPS2,
    HOLD_TAPER_SPEED_MPS,
    K_RATE_CEIL_PER_S,
    LAT_JERK_MAX_MPS3,
    MAX_LATERAL_ACCEL_MPS2,
    MAX_RECOVERY_ATTEMPTS,
    MAX_SPEED_MPS,
    MIN_TURN_RADIUS_M,
    RESUME_HYSTERESIS_M,
    REVERSE_DISTANCE_M,
    REVERSE_MIN_CLEARANCE_M,
    REVERSE_SPEED_MPS,
    SPEED_KV,
    STALL_SPEED_MPS,
    STEER_GAIN_ADAPT_RATE,
    STEER_GAIN_MAX,
    STEER_GAIN_MIN,
    STEER_GAIN_MIN_CURVATURE,
    STEER_GAIN_MIN_SPEED_MPS,
    STEERING_SIGN,
    STOP_MARGIN_M,
    STUCK_TIMEOUT_S,
    TARGET_SPEED_BYPASS_MPS,
    TARGET_SPEED_TAU_S,
    THROTTLE_GAIN_MPS2,
    THROTTLE_SLEW_DOWN_PER_S,
    THROTTLE_SLEW_UP_PER_S,
    TRIM_MAX,
    TRIM_RATE_PER_S,
    YAW_FILTER_ALPHA,
)
from .models import ArcPlan, ControlCommand

DRIVING = "DRIVING"
BLOCKED = "BLOCKED"
REVERSING = "REVERSING"
STUCK = "STUCK"

# Gear indices, as BeamNG's shift logic interprets them.
#
# Automatic-family boxes (automaticGearbox, dctGearbox, cvtGearbox, cvtGearbox2,
# electricMotor) resolve the index through
#     hShifterModeLookup = {[-1] = "R", [0] = "N", "P", "D", "S", "2", "1", "M1"}
# and Lua arrays start at 1, so index 1 is **PARK** and index 2 is Drive.
# Manual and sequential boxes have no such lookup -- there, index 1 is first
# gear. There is no single value that is correct for both, hence the detection
# in forward_gear_index below. Reverse is -1 for either.
AUTOMATIC_FORWARD_GEAR = 2
MANUAL_FORWARD_GEAR = 1
REVERSE_GEAR = -1
_FORWARD_MODE_PREFIXES = ("D", "S", "M")

_MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M


def _slew(previous: float, target: float, up: float, down: float, dt: float) -> float:
    """Move a pedal toward its target no faster than its slew rates allow."""
    if target >= previous:
        return min(target, previous + up * dt)
    return max(target, previous - down * dt)


def forward_gear_index(reported_gear: object) -> int:
    """
    The gear index that means "go forward" for the gearbox reporting this.

    `reported_gear` is BeamNG's `electrics.values.gear`, which is
    `controlLogicModule.getGearName()`: automatic-family shift logic returns a
    mode **string** ("P", "R", "N", "D", "S3", "M2"), while manual and
    sequential return `gearbox.gearIndex`, a **number**. BeamNG itself
    discriminates exactly this way (`type(gearName) == "string"` in
    vehicleController.lua), so this is the sanctioned signal, not a guess.

    An unreadable gearbox defaults to the automatic index, because the two
    failure modes are not symmetric: index 2 on a manual is second gear and the
    car still crawls away, whereas index 1 on an automatic is PARK and the car
    cannot move at all.
    """
    if isinstance(reported_gear, str):
        return AUTOMATIC_FORWARD_GEAR
    if isinstance(reported_gear, (int, float)) and not isinstance(
        reported_gear, bool
    ):
        return MANUAL_FORWARD_GEAR
    return AUTOMATIC_FORWARD_GEAR


def is_forward_gear(reported_gear: object) -> bool:
    """Whether the gearbox is already in a mode that will drive the car forward."""
    if isinstance(reported_gear, str):
        mode = reported_gear.strip().upper()
        return bool(mode) and mode.startswith(_FORWARD_MODE_PREFIXES)
    if isinstance(reported_gear, (int, float)) and not isinstance(
        reported_gear, bool
    ):
        return reported_gear > 0
    return False


def is_reverse_gear(reported_gear: object) -> bool:
    """Whether the gearbox is already in reverse."""
    if isinstance(reported_gear, str):
        return reported_gear.strip().upper().startswith("R")
    if isinstance(reported_gear, (int, float)) and not isinstance(
        reported_gear, bool
    ):
        return reported_gear < 0
    return False


def gear_is_engaged(commanded_gear: int, reported_gear: object) -> bool:
    """
    Whether the box is already in a gear that satisfies the commanded index.

    Used to keep `gear` out of the control message when nothing needs to change:
    `shiftToGearIndex` has side effects -- on a manual it re-enters the clutch
    state machine -- and the display loop would otherwise call it 25 times a
    second.
    """
    if commanded_gear == REVERSE_GEAR:
        return is_reverse_gear(reported_gear)
    return is_forward_gear(reported_gear)


class DrivingController:
    """Turns a sequence of `ArcPlan`s into throttle, brake, steering and gear."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mode = DRIVING
        # Pedal positions (for the slew limits) and the throttle trim. The
        # trim learns the vehicle -- mass, drag, grade -- not the situation,
        # so unlike the pedals it also survives mode changes.
        self._throttle = 0.0
        self._brake = 0.0
        self._trim = 0.0
        # Whether the brake is currently engaged, for the release hysteresis,
        # and the low-passed speed target. None until the first step, so the
        # filter starts settled instead of ramping up from a standstill it was
        # never in.
        self._braking = False
        self._target: float | None = None
        # Steering state: the curvature actually being commanded (slewed), the
        # adaptive gain closing the loop on measured yaw, and the yaw pipeline.
        self._curvature = 0.0
        self._gain = 1.0
        self._last_heading: float | None = None
        self._yaw_filtered: float | None = None
        self._speed_abs = 0.0
        self._mode_elapsed = 0.0
        self._stalled_for = 0.0
        self._reversed_m = 0.0
        self._attempts = 0
        self._forward_gear = AUTOMATIC_FORWARD_GEAR
        # A stall is not cleared by a clear-looking path, because the path
        # always looks clear when the car is kerbed or wedged. Without this the
        # controller flip-flops BLOCKED -> DRIVING forever and never recovers.
        self._blocked_by_stall = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def current_curvature(self) -> float:
        """The slewed curvature actually being commanded right now."""
        return self._curvature

    @property
    def steering_gain(self) -> float:
        return self._gain

    @property
    def measured_curvature(self) -> float | None:
        """Curvature the car is measured to drive, or None without headings."""
        if self._yaw_filtered is None:
            return None
        return self._yaw_filtered / max(self._speed_abs, 1.0)

    def step(
        self,
        plan: ArcPlan,
        forward_speed_mps: float,
        dt: float,
        rear_free_distance_m: float = REVERSE_MIN_CLEARANCE_M,
        reported_gear: object = None,
        heading_rad: float | None = None,
    ) -> ControlCommand:
        dt = max(float(dt), 1e-3)
        speed = float(forward_speed_mps)
        self._forward_gear = forward_gear_index(reported_gear)
        self._speed_abs = abs(speed)
        self._observe_yaw(heading_rad, dt)
        self._mode_elapsed += dt

        if self._mode == REVERSING:
            self._reversed_m += max(0.0, -speed) * dt
            return self._reverse(plan, speed, dt, float(rear_free_distance_m))
        if self._mode in (BLOCKED, STUCK):
            return self._blocked(plan, speed, dt, float(rear_free_distance_m))
        return self._drive(plan, speed, dt)

    # --- modes ---------------------------------------------------------------

    def _drive(
        self, plan: ArcPlan, speed: float, dt: float
    ) -> ControlCommand:
        if plan.free_distance_m <= STOP_MARGIN_M:
            return self._enter(
                BLOCKED, plan, speed, dt, "Path blocked ahead"
            )

        target = self._target_speed(plan, speed, dt)
        throttle, brake = self._longitudinal(target, speed, dt)

        # Commanding throttle and going nowhere means kerbed, wedged or nose-in
        # against something the sweep never saw. The arc looks fine, so only the
        # motion check can catch it.
        if throttle > 0.05 and abs(speed) < STALL_SPEED_MPS:
            self._stalled_for += dt
            if self._stalled_for >= STUCK_TIMEOUT_S:
                self._blocked_by_stall = True
                return self._enter(
                    BLOCKED, plan, speed, dt, "Throttle applied but not moving"
                )
        else:
            self._stalled_for = 0.0

        steering = self._steer(plan.curvature, dt, abs(speed))
        return ControlCommand(
            steering=steering,
            throttle=throttle,
            brake=brake,
            gear=self._forward_gear,
            mode=DRIVING,
            target_speed_mps=target,
            reason=f"Clear for {plan.free_distance_m:.0f} m",
        )

    def _blocked(
        self, plan: ArcPlan, speed: float, dt: float, rear_free_m: float
    ) -> ControlCommand:
        if (
            self._mode != STUCK
            and not self._blocked_by_stall
            and plan.free_distance_m > (STOP_MARGIN_M + RESUME_HYSTERESIS_M)
        ):
            return self._enter(DRIVING, plan, speed, dt, "Path clear")

        stopped = abs(speed) < STALL_SPEED_MPS
        if (
            self._mode == BLOCKED
            and stopped
            and self._mode_elapsed >= BLOCKED_HOLD_S
        ):
            self._attempts += 1
            if self._attempts > MAX_RECOVERY_ATTEMPTS:
                return self._enter(
                    STUCK, plan, speed, dt, "Recovery exhausted; take over"
                )
            if rear_free_m >= REVERSE_MIN_CLEARANCE_M:
                return self._enter(
                    REVERSING, plan, speed, dt, "Backing up to re-plan"
                )
            # No room behind either. Hold, and let the attempt counter run the
            # car into STUCK rather than reversing into whatever is back there.
            self._mode_elapsed = 0.0

        return ControlCommand(
            steering=self._steer(0.0, dt, abs(speed)),
            throttle=0.0,
            brake=self._hold_brake(speed, dt),
            gear=self._forward_gear,
            mode=self._mode,
            target_speed_mps=0.0,
            reason=(
                "Recovery exhausted; take over"
                if self._mode == STUCK
                else "Blocked; holding"
            ),
        )

    def _reverse(
        self, plan: ArcPlan, speed: float, dt: float, rear_free_m: float
    ) -> ControlCommand:
        done = self._reversed_m >= REVERSE_DISTANCE_M
        if done or rear_free_m < REVERSE_MIN_CLEARANCE_M:
            return self._enter(
                BLOCKED,
                plan,
                speed,
                dt,
                "Reverse finished" if done else "Blocked behind",
            )

        # Speed is signed, so reversing at 2 m/s reads as -2.0. Compare
        # magnitudes and let the same PI law do the work.
        throttle, brake = self._longitudinal(REVERSE_SPEED_MPS, abs(speed), dt)
        return ControlCommand(
            steering=self._steer(0.0, dt, abs(speed)),
            throttle=throttle,
            brake=brake,
            gear=REVERSE_GEAR,
            mode=REVERSING,
            target_speed_mps=REVERSE_SPEED_MPS,
            reason=f"Reversing {self._reversed_m:.1f}/{REVERSE_DISTANCE_M:.0f} m",
        )

    # --- helpers -------------------------------------------------------------

    def _enter(
        self, mode: str, plan: ArcPlan, speed: float, dt: float, reason: str
    ) -> ControlCommand:
        self._mode = mode
        self._mode_elapsed = 0.0
        self._stalled_for = 0.0
        if mode == REVERSING:
            self._reversed_m = 0.0
        if mode in (DRIVING, REVERSING):
            self._blocked_by_stall = False
        if mode == DRIVING:
            self._attempts = 0
            return self._drive(plan, speed, dt)
        if mode == REVERSING:
            return self._reverse(plan, speed, dt, REVERSE_MIN_CLEARANCE_M)
        return ControlCommand(
            steering=self._steer(0.0, dt, abs(speed)),
            throttle=0.0,
            brake=self._hold_brake(speed, dt),
            gear=self._forward_gear,
            mode=mode,
            target_speed_mps=0.0,
            reason=reason,
        )

    @staticmethod
    def _corner_speed(curvature: float, distance_m: float) -> float:
        """Speed a corner of this curvature allows, ``distance_m`` before it."""
        return math.sqrt(
            CORNERING_ACCEL_MPS2 / abs(curvature)
            + 2.0 * COMFORT_DECEL_MPS2 * max(distance_m, 0.0)
        )

    def _raw_target_speed(self, plan: ArcPlan) -> float:
        target = MAX_SPEED_MPS
        # The immediate command gets NO entry allowance, and that is deliberate.
        # Crediting the distance the steering takes to wind on ("hold speed, I
        # will slow before I am really in the corner") is self-referential: the
        # credit keeps the target high, the high target commands no braking,
        # and the wind-on completes with the car still at speed. Measured on a
        # 25 m bend it peaked at 6.7 m/s^2 against the 3.5 limit of the time.
        # It is the same
        # trap as discounting a deferred turn in the planner -- under per-tick
        # re-planning, "later" never arrives.
        #
        # Anticipation has to come from the PATH, not from the controller: the
        # deferred families below describe a real straight before a real bend,
        # so their allowance is a geometric fact that shrinks as the car
        # advances. Smoothness here comes from the plan not jittering
        # (despeckling, gravity-referenced heights) and from the target filter.
        curvature = abs(plan.curvature)
        if curvature > 1e-6:
            target = min(target, math.sqrt(CORNERING_ACCEL_MPS2 / curvature))
        # Brake TO the corner-entry speed of a deferred turn, not toward a
        # stop: full speed 20 m before a bend, arriving at cornering speed
        # exactly at the turn-in point the planner picked.
        if abs(plan.next_curvature) > 1e-6:
            target = min(
                target,
                self._corner_speed(
                    plan.next_curvature, plan.transition_distance_m
                ),
            )
        headroom = max(0.0, plan.free_distance_m - STOP_MARGIN_M)
        return min(target, math.sqrt(2.0 * COMFORT_DECEL_MPS2 * headroom))

    def _target_speed(self, plan: ArcPlan, speed: float, dt: float) -> float:
        """
        The raw speed law, low-passed so noise cannot reach the brake pedal.

        Free distance is a nearest-point measurement over a cloud that is
        rebuilt every 40 ms, so its single-tick dips are measurement noise
        rather than corners -- and the brake band is only 1.3 m/s wide, so
        unfiltered they were felt directly. A genuine collapse bypasses the
        filter, so real braking is never delayed by it.
        """
        raw = self._raw_target_speed(plan)
        if self._target is None or raw < speed - TARGET_SPEED_BYPASS_MPS:
            self._target = raw
        else:
            alpha = min(1.0, dt / max(TARGET_SPEED_TAU_S, 1e-3))
            self._target += alpha * (raw - self._target)
        return self._target

    def _longitudinal(
        self, target: float, speed: float, dt: float
    ) -> tuple[float, float]:
        """
        Acceleration-domain pedals: desired accel from the speed error, mapped
        through nominal full-pedal accelerations, with a coasting band between
        them. One foot works both pedals -- engaging one instantly releases
        the other -- and each pedal moves under its own slew limits.
        """
        error = target - speed
        a_des = max(-HARD_DECEL_MPS2, min(COMFORT_ACCEL_MPS2, SPEED_KV * error))
        # Hysteresis around the coast threshold: once on the brake, stay on it
        # until the demand has genuinely eased. Sharing one threshold for both
        # edges makes the pedal chatter across the boundary, which is felt as a
        # shudder rather than as braking.
        engage = COAST_DECEL_MPS2 * (
            BRAKE_RELEASE_FRACTION if self._braking else 1.0
        )
        self._braking = a_des < -engage
        if self._braking:
            self._throttle = 0.0
            self._brake = _slew(
                self._brake,
                # Credit only the drag that is really there, not the whole
                # coast band -- the band decided whether to brake, this decides
                # how hard. Clamped at zero because inside the hysteresis band
                # the pedal should simply be coming off.
                max(0.0, min(1.0, (-a_des - ENGINE_DRAG_MPS2) / BRAKE_GAIN_MPS2)),
                BRAKE_SLEW_UP_PER_S,
                BRAKE_SLEW_DOWN_PER_S,
                dt,
            )
        elif a_des < 0.0:
            # The lift-off band: engine drag covers a small overspeed, the way
            # a human eases off rather than riding the brake. Also what keeps
            # the pedals from chattering across the throttle/brake boundary.
            self._throttle = 0.0
            self._brake = 0.0
        else:
            raw = a_des / THROTTLE_GAIN_MPS2 + self._trim
            if raw < 0.95:  # anti-windup: stop trimming into saturation
                self._trim = max(
                    0.0,
                    min(TRIM_MAX, self._trim + TRIM_RATE_PER_S * error * dt),
                )
            self._brake = 0.0
            self._throttle = _slew(
                self._throttle,
                min(1.0, raw),
                THROTTLE_SLEW_UP_PER_S,
                THROTTLE_SLEW_DOWN_PER_S,
                dt,
            )
        return self._throttle, self._brake

    def _hold_brake(self, speed: float, dt: float) -> float:
        """
        Brake for the blocked/stuck hold: full and immediate while genuinely
        moving (an emergency outranks the slew limits), tapering to a firm
        hold once stopped instead of standing on the pedal forever.
        """
        self._throttle = 0.0
        self._braking = True
        if abs(speed) > HOLD_TAPER_SPEED_MPS:
            self._brake = 1.0
        else:
            taper = BRAKE_HOLD_FRACTION + (1.0 - BRAKE_HOLD_FRACTION) * (
                abs(speed) / HOLD_TAPER_SPEED_MPS
            )
            self._brake = _slew(
                self._brake,
                taper,
                BRAKE_SLEW_UP_PER_S,
                BRAKE_SLEW_DOWN_PER_S,
                dt,
            )
        return self._brake

    def _observe_yaw(self, heading_rad: float | None, dt: float) -> None:
        """Wrap-aware yaw rate from successive world headings, low-passed."""
        if heading_rad is None:
            return
        heading = float(heading_rad)
        if self._last_heading is not None:
            yaw_rate = math.remainder(heading - self._last_heading, math.tau) / dt
            self._yaw_filtered = (
                yaw_rate
                if self._yaw_filtered is None
                else YAW_FILTER_ALPHA * yaw_rate
                + (1.0 - YAW_FILTER_ALPHA) * self._yaw_filtered
            )
        self._last_heading = heading

    def _adapt_gain(self, speed: float, dt: float) -> None:
        """
        Close the loop the open-loop curvature-to-steering map never had:
        compare the curvature the car measurably drives against the command
        and trim a clamped gain, slowly. Only while genuinely cornering above
        walking pace in DRIVING, and only when the car turns the way it was
        asked -- a sign mismatch is noise (slides, kerb strikes), not data.
        """
        if (
            self._mode != DRIVING
            or self._yaw_filtered is None
            or speed < STEER_GAIN_MIN_SPEED_MPS
            or abs(self._curvature) < STEER_GAIN_MIN_CURVATURE
        ):
            return
        measured = self._yaw_filtered / max(speed, 1.0)
        if measured * self._curvature <= 0.0:
            return
        ratio = min(3.0, measured / self._curvature)
        self._gain = max(
            STEER_GAIN_MIN,
            min(
                STEER_GAIN_MAX,
                self._gain + STEER_GAIN_ADAPT_RATE * (1.0 - ratio) * dt,
            ),
        )

    @staticmethod
    def _curvature_rate(speed: float) -> float:
        """
        How fast the steering may wind on, in curvature per second.

        Lateral jerk is v^2 times the curvature rate, so the limit falls with
        speed, with a ceiling that keeps parking-speed manoeuvring as brisk as
        the old steering slew. This is comfort only -- the hard limit on how
        much curvature may be asked for at all is `_curvature_ceiling`.
        """
        return min(K_RATE_CEIL_PER_S, LAT_JERK_MAX_MPS3 / max(speed, 1.0) ** 2)

    @staticmethod
    def _curvature_ceiling(speed: float) -> float:
        """
        Tightest curvature the lateral limit allows at this speed.

        A driver cannot turn in harder than grip allows and neither can this:
        commanding it anyway just means understeering off the chosen line. The
        limit belongs here, as a hard constraint on the steering, rather than
        as something the speed law has to chase -- the speed law only asks for
        a target and the pedals take seconds to deliver it, so the wheel was
        fully wound on long before the car had slowed down, measured at 5.3
        m/s^2 through a 25 m bend against a 3.5 limit. As the car slows the
        ceiling opens and the turn completes.

        Below about 16 km/h the ceiling is wider than full lock, so parking
        manoeuvres are untouched.
        """
        return MAX_LATERAL_ACCEL_MPS2 / max(abs(speed), 1e-3) ** 2

    def _steer(self, curvature: float, dt: float, speed: float = 0.0) -> float:
        rate = self._curvature_rate(speed)
        ceiling = min(_MAX_CURVATURE, self._curvature_ceiling(speed))
        target = max(-ceiling, min(ceiling, float(curvature)))
        self._curvature = max(
            self._curvature - rate * dt,
            min(self._curvature + rate * dt, target),
        )
        self._adapt_gain(speed, dt)
        steering = STEERING_SIGN * self._gain * self._curvature / _MAX_CURVATURE
        return max(-1.0, min(1.0, steering))
