"""
Autonomous emergency braking over the car's own predicted path.

Pure Python/numpy: no Qt, no BeamNGpy, no wall clock -- elapsed time arrives as
``dt``, so the state machine is fully deterministic under test.

Deliberately independent of `controller` and of self-driving. AEB is a safety
layer that has to work while a **human** is driving, so:

- the path comes from the yaw the car is MEASURED to be turning at, never from a
  command. There is no controller to ask when self-driving is off, and even when
  it is on, what the car is actually doing is the honest prediction;
- it touches nothing but the pedals. Steering, gear and the parking brake stay
  with whoever is driving;
- obstacles are treated as STATIC, because a single-frame cloud carries no
  velocity and nothing here tracks returns between frames. That errs toward
  braking early behind a moving leader, which is the safe direction, and the
  standoff and trigger threshold are sized to keep it out of ordinary following.

Curvature sign follows `planner`: **positive turns left**, the opposite of
BeamNG's steering input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import (
    AEB_BRAKING_DECEL_MPS2,
    AEB_CLEARANCE_MARGIN_M,
    AEB_CONFIRM_S,
    AEB_HOLD_BRAKE,
    AEB_HORIZON_MARGIN,
    AEB_LATENCY_S,
    AEB_MAX_HORIZON_M,
    AEB_MIN_ENGAGED_S,
    AEB_MIN_HITS,
    AEB_MIN_HORIZON_M,
    AEB_MIN_SPEED_MPS,
    AEB_RELEASE_MARGIN,
    AEB_REVERSE_BRAKING_DECEL_MPS2,
    AEB_REVERSE_MIN_SPEED_MPS,
    AEB_REVERSE_STANDOFF_M,
    AEB_STANDOFF_M,
    AEB_STOPPED_SPEED_MPS,
    AEB_TRIGGER_MARGIN,
    AEB_YAW_FILTER_ALPHA,
    MIN_TURN_RADIUS_M,
    STALL_SPEED_MPS,
)
from .models import ARMED, BRAKING, STANDBY, AebState, VehicleGeometry
from .planner import corridor_blocking_distances

MAX_CURVATURE = 1.0 / MIN_TURN_RADIUS_M
# Matches planner._MIN_CURVATURE: below this an arc is straight for every
# practical purpose, and substituting it keeps the geometry free of a
# divide-by-zero special case.
_MIN_CURVATURE = 1e-4


@dataclass(frozen=True)
class BrakingProfile:
    """
    What differs between braking forwards and backwards.

    The geometry, the corridor scan, the phantom filters and the state machine
    are all direction-agnostic -- `EmergencyBraking` runs the rear system on a
    180-degree-rotated cloud -- so this is the whole of the difference, and all
    three numbers are properties of the CAR rather than of the algorithm.
    """

    label: str
    braking_decel_mps2: float
    standoff_m: float
    min_speed_mps: float


# Measured; see the tables in config.py. The car stops ~30% worse backwards
# because braking in reverse loads the rear axle, which has the smaller brakes.
FORWARD = BrakingProfile(
    label="AEB",
    braking_decel_mps2=AEB_BRAKING_DECEL_MPS2,
    standoff_m=AEB_STANDOFF_M,
    min_speed_mps=AEB_MIN_SPEED_MPS,
)
REVERSE = BrakingProfile(
    label="REAR AEB",
    braking_decel_mps2=AEB_REVERSE_BRAKING_DECEL_MPS2,
    standoff_m=AEB_REVERSE_STANDOFF_M,
    min_speed_mps=AEB_REVERSE_MIN_SPEED_MPS,
)


@dataclass(frozen=True)
class BrakeMeasurement:
    """What one full-authority stop actually achieved, as opposed to modelled."""

    cause: str
    """
    Who braked, and which way the car was going: AEB, REAR AEB, MANUAL,
    MANUAL REVERSE. The caller names it, because only the caller knows.
    """
    from_speed_mps: float
    to_speed_mps: float
    duration_s: float
    distance_m: float
    mean_decel_mps2: float
    """The headline. Directly comparable to `profile.braking_decel_mps2`, and
    the number a new row of the config tables is made of."""
    peak_decel_mps2: float
    modelled_decel_mps2: float
    """What the configured plant claimed, so the two sit side by side."""
    modelled_distance_m: float
    """
    What the configured plant says a stop from this speed takes.

    The PURE braking distance, ``v^2 / (2 a)``, deliberately without
    `stopping_distance`'s latency term: the tables in config are pure braking
    distances with no reaction time folded in, and this number exists to be
    compared against them. AEB's own trigger uses the full `stopping_distance`
    and is unaffected by anything here.
    """
    pitch_deg: float
    """Body pitch at the start, positive nose-up. A stop down a grade flatters
    the plant and a stop up one slanders it, so the number is logged rather than
    corrected for."""

    @property
    def optimism(self) -> float:
        """
        How much further the car really took than the model promised.

        Above 1.0 the model UNDER-predicts the real stop, which is the dangerous
        direction: AEB fires at the last point the model says works, so a plant
        measured on a lighter car would have it fire after the last point that
        actually does. Expect a little over 1.0 on an AEB stop even with a
        correct plant, because that trace starts when the pedal is COMMANDED and
        so carries the actuation delay the modelled figure leaves out.
        """
        if self.modelled_distance_m <= 0.0:
            return 0.0
        return self.distance_m / self.modelled_distance_m


class BrakeEvent:
    """
    Records a stop and reports the deceleration the car actually delivered.

    Pure measurement: nothing here feeds back into the trigger, and it is
    deliberately not wired into `EmergencyBraking`. Every braking figure in
    `config` is a property of ONE vehicle that nothing in the repo recorded, so
    this is the instrument that says what the attached car does -- validate it
    against the car the published tables came from before trusting it on
    another.

    Wall-clock-free like the rest of the module: ``dt`` arrives from the caller.
    """

    def __init__(self, profile: BrakingProfile = FORWARD) -> None:
        self.profile = profile
        self._active = False
        self._cause = ""
        self._from_speed = 0.0
        self._last_speed = 0.0
        self._pitch_deg = 0.0
        self._duration = 0.0
        self._distance = 0.0
        self._peak_decel = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def start(self, speed_mps: float, pitch_deg: float, cause: str) -> None:
        speed = abs(float(speed_mps))
        self._active = True
        self._cause = cause
        self._from_speed = speed
        self._last_speed = speed
        self._pitch_deg = float(pitch_deg)
        self._duration = 0.0
        self._distance = 0.0
        self._peak_decel = 0.0

    def sample(self, speed_mps: float, dt: float) -> None:
        """One tick of the trace. Speed is unsigned here: reverse stops too."""
        if not self._active:
            return
        step = max(float(dt), 1e-3)
        speed = abs(float(speed_mps))
        # Trapezoidal, because a tick is 40 ms and the speed is falling fast
        # enough through it that either endpoint alone biases the distance.
        self._distance += 0.5 * (self._last_speed + speed) * step
        self._duration += step
        self._peak_decel = max(
            self._peak_decel, (self._last_speed - speed) / step
        )
        self._last_speed = speed

    def finish(self) -> BrakeMeasurement | None:
        """
        Close the event and report, or None if there is nothing to report.

        A single-tick event has no trace to measure and a stop that never lost
        any speed is not a stop, so both come back as None rather than as a
        deceleration divided by nothing.
        """
        if not self._active:
            return None
        self._active = False
        lost = self._from_speed - self._last_speed
        if self._duration <= 0.0 or lost <= 0.0:
            return None
        return BrakeMeasurement(
            cause=self._cause,
            from_speed_mps=self._from_speed,
            to_speed_mps=self._last_speed,
            duration_s=self._duration,
            distance_m=self._distance,
            mean_decel_mps2=lost / self._duration,
            peak_decel_mps2=self._peak_decel,
            modelled_decel_mps2=self.profile.braking_decel_mps2,
            modelled_distance_m=(
                self._from_speed
                * self._from_speed
                / (2.0 * self.profile.braking_decel_mps2)
            ),
            pitch_deg=self._pitch_deg,
        )


def stopping_distance(
    speed_mps: float, profile: BrakingProfile = FORWARD
) -> float:
    """
    Distance a full-authority emergency stop needs from this speed.

    The latency term is not padding: it is the display tick, the blocking
    ``Vehicle.control`` ack, and pad bite plus weight transfer. It matters most
    at LOW speed, where it is a large share of a short stopping distance. Note
    AEB_CONFIRM_S is deliberately not in it -- see the constant.
    """
    speed = max(0.0, float(speed_mps))
    return (
        speed * AEB_LATENCY_S
        + speed * speed / (2.0 * profile.braking_decel_mps2)
    )


def mirrored(geometry: VehicleGeometry) -> VehicleGeometry:
    """
    The vehicle as seen by a rear-facing system: a 180-degree rotation.

    Swapping front/rear and left/right is exactly what rotating the BEV frame
    by pi does, so the rear system can reuse every helper that assumes "ahead"
    means +forward. Handedness is preserved -- a rotation, not a reflection --
    so the curvature sign convention survives it, which is what lets the rear
    scan share `planner`'s arc geometry unchanged.
    """
    return replace(
        geometry,
        front_m=geometry.rear_m,
        rear_m=geometry.front_m,
        left_m=geometry.right_m,
        right_m=geometry.left_m,
    )


def mirror_points(bev: np.ndarray) -> np.ndarray:
    """Rotate a BEV cloud by 180 degrees, putting what was behind in front."""
    points = np.asarray(bev, dtype=np.float32).reshape((-1, 2))
    return -points if len(points) else points


def scan_horizon(
    speed_mps: float,
    geometry: VehicleGeometry,
    profile: BrakingProfile = FORWARD,
) -> float:
    """
    How far ahead the corridor is scanned at this speed.

    Public because the worker has to build AEB's obstacle band to the same
    reach: everything upstream of the scan is O(cloud), and always extracting
    to AEB_MAX_HORIZON_M would pay the 150 m price at 30 km/h, where the scan
    only wants 12. See AEB_HORIZON_MARGIN for why it reaches past the trigger.
    """
    standoff = geometry.front_m + profile.standoff_m
    return min(
        AEB_MAX_HORIZON_M,
        standoff
        + max(
            AEB_MIN_HORIZON_M,
            AEB_HORIZON_MARGIN * stopping_distance(speed_mps, profile),
        ),
    )


def predicted_corridor(
    curvature: float,
    length_m: float,
    half_width_m: float,
    samples: int = 28,
) -> np.ndarray:
    """
    The swept corridor as a closed BEV ``(right, forward)`` polygon.

    Right-hand edge outbound, left-hand edge back, so it drops straight into a
    filled polygon in the overlay. The normal pointing right of an arc of
    curvature ``k`` at arc length ``s`` is ``(cos(ks), sin(ks))`` -- at ``s = 0``
    that is ``(1, 0)``, which is the +right axis, as it should be.
    """
    count = max(2, int(samples))
    progress = np.linspace(0.0, float(length_m), count)
    k = float(curvature)
    half_width = float(half_width_m)
    if abs(k) < _MIN_CURVATURE:
        centre = np.column_stack((np.zeros_like(progress), progress))
        normal = np.column_stack(
            (np.ones_like(progress), np.zeros_like(progress))
        )
    else:
        theta = k * progress
        centre = np.column_stack(
            ((np.cos(theta) - 1.0) / k, np.sin(theta) / k)
        )
        normal = np.column_stack((np.cos(theta), np.sin(theta)))
    right = centre + half_width * normal
    left = centre - half_width * normal
    return np.concatenate((right, left[::-1])).astype(np.float32)


def corridor_cross_section(
    curvature: float, distance_m: float, half_width_m: float
) -> np.ndarray:
    """The ``(2, 2)`` chord across the corridor ``distance_m`` along the path."""
    return predicted_corridor(curvature, distance_m, half_width_m, samples=2)[
        1:3
    ]


class EmergencyBraking:
    """Turns a sequence of obstacle clouds into a brake pedal, and nothing else."""

    def __init__(self, profile: BrakingProfile = FORWARD) -> None:
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self._engaged = False
        self._brake = 0.0
        self._engaged_for = 0.0
        # How long a threat has been continuously in the corridor, and how
        # long the corridor has been continuously clear. A phantom lasts a
        # frame; a wall does not, and neither does a gap in the returns. See
        # AEB_CONFIRM_S.
        self._seen_for = 0.0
        self._clear_for = 0.0
        # How long the car has been at rest under an AEB stop, and the room
        # there was when the event fired -- the release is latched against the
        # latter so that slowing down cannot satisfy it.
        self._fired_available = 0.0
        self._fired_horizon = 0.0
        # Same wrap-aware yaw pipeline as DrivingController._observe_yaw, and
        # duplicated rather than shared on purpose: AEB has to run with
        # self-driving off, when no controller exists to ask.
        self._last_heading: float | None = None
        self._yaw_filtered: float | None = None
        # Distance to the last measured blockage, or None for a clear corridor.
        # Held across a gap in the returns.
        self._threat_m: float | None = None

    def horizon_for(
        self, speed_mps: float, geometry: VehicleGeometry
    ) -> float:
        """
        The horizon the next `step` will use, latch included.

        The worker calls this to size AEB's obstacle band, so the band and the
        scan can never disagree about how far "ahead" reaches.
        """
        horizon = scan_horizon(speed_mps, geometry, self.profile)
        return max(horizon, self._fired_horizon) if self._engaged else horizon

    def step(
        self,
        obstacles: np.ndarray,
        geometry: VehicleGeometry,
        forward_speed_mps: float,
        dt: float,
        heading_rad: float | None = None,
        has_returns: bool = True,
    ) -> AebState:
        """
        One tick, and the whole public surface: everything a caller needs comes
        back on the `AebState`, so this object keeps no queryable state of its
        own.

        ``obstacles`` is `planner.geometric_obstacles` output at AEB's own floor,
        in BEV metres; ``forward_speed_mps`` is signed, so reversing is negative.
        """
        dt = max(float(dt), 1e-3)
        speed = float(forward_speed_mps)
        self._observe_yaw(heading_rad, dt)

        standoff = geometry.front_m + self.profile.standoff_m
        half_width = geometry.width_m / 2.0 + AEB_CLEARANCE_MARGIN_M
        # The corridor is centred on the BODY, not on the reference node the
        # BEV frame is built around. The node sits off-centre in the bounding
        # box, so a node-centred corridor sweeps a band partly beside the car:
        # backing the D-Series into a garage doorway it was CENTRED in, one
        # corridor edge reached into the wall and fired the brake. For the rear
        # system the mirrored geometry swaps left and right, which is exactly
        # the sign the 180-degree-rotated cloud needs.
        centre_m = (geometry.right_m - geometry.left_m) / 2.0
        # The distance a full-authority stop actually needs from here. Every
        # threshold below is a multiple of it, which is what makes the feature
        # behave the same way at 30 km/h as at 100 -- and what lets one
        # implementation serve both directions off different measured plants.
        needed = stopping_distance(speed, self.profile)
        horizon = self.horizon_for(speed, geometry)
        if self._engaged:
            # ...and never shrinks mid-event. The horizon scales with `needed`,
            # so braking pulls it in behind the very obstacle being braked for:
            # traced at 50 km/h, the wall sat at 10.2 m while the horizon had
            # come down to 9.2 m, the threat read as GONE, and the brake
            # released and then re-fired from much closer. Latched at the value
            # the event started with, the obstacle cannot escape out of the
            # back of the scan.
            horizon = max(horizon, self._fired_horizon)

        # Reversing or crawling: the corridor is ahead of the car and the yaw
        # estimate is meaningless down here, so AEB simply stands by. Note this
        # gate blocks ARMING only -- an event already under way brakes the car
        # all the way to rest.
        if speed < self.profile.min_speed_mps and not self._engaged:
            self._brake = 0.0
            self._threat_m = None
            self._seen_for = 0.0
            self._clear_for = 0.0
            return AebState(
                status=STANDBY,
                brake=0.0,
                curvature=0.0,
                rearward=self.profile is REVERSE,
                threat_m=None,
                horizon_m=horizon,
                standoff_m=standoff,
                brake_now_m=standoff + AEB_TRIGGER_MARGIN * needed,
                corridor_half_width_m=half_width,
                required_decel_mps2=0.0,
                time_to_collision_s=math.inf,
                reason=(
                    f"Standing by below "
                    f"{self.profile.min_speed_mps * 3.6:.0f} km/h"
                ),
                lateral_offset_m=centre_m,
            )

        curvature = self._curvature_at(
            max(speed, self.profile.min_speed_mps)
        )
        # No returns cannot ARM the brake -- an empty cloud looks exactly like a
        # clear road, and firing on it would mean a full brake application at
        # every map load. It cannot DISARM it either: mid-event the last
        # measured threat is held, so an outage never releases the pedal into
        # whatever was in front of the car.
        if has_returns:
            # Shift the cloud into the body-centred frame rather than bending
            # the arc: the offset is centimetres against corridor radii of
            # many metres, so the parallel-path error this ignores is far
            # below the clearance margin.
            scan = obstacles
            if centre_m != 0.0 and len(scan):
                scan = scan - np.asarray(
                    (centre_m, 0.0), dtype=scan.dtype
                )
            blocking = corridor_blocking_distances(
                scan, curvature, half_width, horizon
            )
            # The Nth nearest, not the nearest: a thing worth braking for is a
            # surface and puts many returns in a corridor this narrow, while a
            # speck that survived despeckling puts one. See AEB_MIN_HITS.
            self._threat_m = (
                float(blocking[AEB_MIN_HITS - 1])
                if len(blocking) >= AEB_MIN_HITS
                else None
            )
        elif not self._engaged:
            self._threat_m = None
        threat = self._threat_m

        # A clear corridor is NOT an obstacle parked at the horizon. Treating it
        # as one meant the required deceleration on an empty road was
        # v^2 / (2 * (horizon - standoff)), which past the point where the
        # horizon clamps at PLANNER_HORIZON_M exceeds the trigger on its own --
        # measured, an empty gridmap braked itself from 64 km/h down to 45,
        # released, and did it again. There is nothing to hit, so there is
        # nothing to compute.
        if threat is None:
            available = horizon - standoff
            required = 0.0
            ttc = math.inf
        else:
            available = threat - standoff
            if available <= 0.0:
                required = math.inf
                ttc = 0.0
            else:
                required = speed * speed / (2.0 * available) if speed > 0.0 else 0.0
                ttc = available / speed if speed > 0.0 else math.inf
        # Where the car has to be braking by. Reported so the overlay can draw
        # it and the log can explain a firing after the fact.
        brake_now = standoff + AEB_TRIGGER_MARGIN * needed

        # The confirmation window counts how long a threat has been SEEN, not
        # how long it has been urgent. A phantom lasts a frame either way, but
        # counting urgency would spend the window after the situation had
        # already become critical and so delay every real brake by
        # AEB_CONFIRM_S. The horizon reaches well past the distance the trigger
        # fires from (that is what AEB_HORIZON_MARGIN buys), so a real obstacle
        # is confirmed long before it matters.
        if threat is None:
            self._seen_for = 0.0
            self._clear_for += dt
        else:
            self._seen_for += dt
            self._clear_for = 0.0
        # The last point to brake, and nothing sooner. Note this compares
        # DISTANCES rather than decelerations: the deceleration form silently
        # dropped the latency term, which is most of the answer at low speed.
        triggered = (
            speed >= self.profile.min_speed_mps
            and threat is not None
            and available <= AEB_TRIGGER_MARGIN * needed
        )

        if self._engaged:
            self._engaged_for += dt
            if self._engaged_for < AEB_MIN_ENGAGED_S:
                cleared = False
            else:
                cleared = (
                    # It went away, confirmed the same way it was detected.
                    self._clear_for >= AEB_CONFIRM_S
                    # ...or the gap genuinely opened up. Latched against the
                    # room the event STARTED with, because the ratio alone
                    # recovers as the car slows -- braking would otherwise
                    # release the brake. See AEB_RELEASE_MARGIN.
                    or available
                    > max(AEB_RELEASE_MARGIN * needed, self._fired_available)
                    # ...or the car has come to REST, at which point the event
                    # is over and the pedal goes straight back to the driver.
                    # The stop is the whole objective, so there is nothing left
                    # for the brake to achieve. Note the car is not held against
                    # a gradient afterwards: releasing means releasing, exactly
                    # as every teardown path in `worker` hands back a coasting
                    # car rather than a braked one.
                    #
                    # AEB_STOPPED_SPEED_MPS, not STALL_SPEED_MPS. "Counts as
                    # stationary" is 0.3 m/s, and letting go there hands back a
                    # car still rolling at 1 km/h toward the thing it just
                    # braked for -- which under a full pedal is one tick before
                    # it is actually stopped, so it buys nothing either.
                    or abs(speed) <= AEB_STOPPED_SPEED_MPS
                )
            if cleared:
                self._engaged = False
                self._brake = 0.0
            else:
                self._brake = self._brake_for(speed)
        elif triggered and self._seen_for >= AEB_CONFIRM_S:
            self._engaged = True
            self._engaged_for = 0.0
            self._fired_available = available
            self._fired_horizon = horizon
            self._brake = self._brake_for(speed)
        else:
            self._brake = 0.0

        return AebState(
            status=BRAKING if self._engaged else ARMED,
            brake=self._brake,
            curvature=curvature,
            rearward=self.profile is REVERSE,
            threat_m=threat,
            horizon_m=horizon,
            standoff_m=standoff,
            brake_now_m=brake_now,
            corridor_half_width_m=half_width,
            required_decel_mps2=required,
            time_to_collision_s=ttc,
            reason=self._reason(threat, horizon, brake_now, available, ttc, speed),
            lateral_offset_m=centre_m,
        )

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _brake_for(speed: float) -> float:
        """
        Full pedal, because by the time this is called nothing less works.

        The grading lives in the TRIGGER, not here: AEB does nothing at all
        until the car reaches the last point a full-authority stop still
        succeeds from, and at that point a partial application is simply a
        collision. Grading the pedal instead was what made it fire 22.9 m out
        at 50 km/h on 0.52 and stop with 7.2 m to spare -- early and soft, which
        is a driver-assist rather than an emergency brake.
        """
        if abs(speed) < STALL_SPEED_MPS:
            # Stopped against the obstacle. Hold it there instead of standing on
            # the pedal indefinitely.
            return AEB_HOLD_BRAKE
        return 1.0

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
                else AEB_YAW_FILTER_ALPHA * yaw_rate
                + (1.0 - AEB_YAW_FILTER_ALPHA) * self._yaw_filtered
            )
        self._last_heading = heading

    def _curvature_at(self, speed: float) -> float:
        if self._yaw_filtered is None:
            return 0.0
        curvature = self._yaw_filtered / max(speed, self.profile.min_speed_mps)
        return max(-MAX_CURVATURE, min(MAX_CURVATURE, curvature))

    def _reason(
        self,
        threat: float | None,
        horizon: float,
        brake_now: float,
        available: float,
        ttc: float,
        speed: float,
    ) -> str:
        if self._engaged:
            if abs(speed) < STALL_SPEED_MPS:
                return f"Stopped {max(available, 0.0):.1f} m short; holding"
            if math.isfinite(ttc):
                return f"Impact in {ttc:.1f} s; full brake"
            return "Full brake"
        if threat is None:
            return f"Clear to {horizon:.0f} m"
        # The distance still in hand beyond the last point to brake is what
        # says whether this is about to become an emergency or is just scenery.
        return (
            f"Watching an obstacle at {threat:.1f} m "
            f"({max(threat - brake_now, 0.0):.0f} m before it acts)"
        )
