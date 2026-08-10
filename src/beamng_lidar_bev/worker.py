from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

if TYPE_CHECKING:
    from beamngpy import BeamNGpy, Vehicle
    from beamngpy.sensors import Lidar

from .aeb import (
    FORWARD,
    REVERSE,
    BrakeEvent,
    BrakeMeasurement,
    BrakingProfile,
    EmergencyBraking,
    mirror_points,
    mirrored,
    stopping_distance,
)
from .config import (
    AEB_CONFIRM_S,
    AEB_MAX_HORIZON_M,
    AEB_MIN_HITS,
    AEB_MIN_VERTICAL_EXTENT_M,
    AEB_OBSTACLE_MIN_HEIGHT_M,
    AEB_TRIGGER_MARGIN,
    BEAMNG_EXE,
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    CONTROL_INTERVAL_MS,
    DISPLAY_INTERVAL_MS,
    LIDAR_RANGE_M,
    LIDAR_ROAD_VISUAL_COLOUR,
    LIDAR_UPDATE_HZ,
    LIDAR_UPDATE_TIME_S,
    LOOKAHEAD_MAX_M,
    LOOKAHEAD_MIN_M,
    LOOKAHEAD_TIME_S,
    MARKING_CLASSES,
    MAX_OBSTACLE_RENDER_POINTS,
    MAX_ROAD_RENDER_POINTS,
    MAX_SPEED_MPS,
    NAV_POLL_INTERVAL_MS,
    OBSTACLE_MIN_HEIGHT_M,
    PLANNER_HORIZON_M,
    PLANT_REFERENCE_VEHICLE,
    STEERING_SIGN,
    TRANSITION_DISTANCES_M,
    WORLD_ACTOR_FADE_S,
    WORLD_ACTOR_REGISTRY_INTERVAL_S,
    WORLD_ACTOR_STATE_INTERVAL_S,
)
from .controller import (
    REVERSE_GEAR,
    DrivingController,
    forward_gear_index,
    gear_is_engaged,
)
from .geometry import (
    derive_vehicle_geometry,
    vec3,
    vehicle_axes,
    world_points_to_bev,
)
from .launcher import (
    bridge_is_reachable,
    build_launch_command,
    start_beamng_process,
)
from .models import (
    BRAKING,
    ActorObservation,
    AebState,
    ArcPlan,
    BevFrame,
    ControlCommand,
    DrivingPlan,
    PerceptionSnapshot,
    SensorMount,
    VehicleGeometry,
)
from .navigation import fetch_route, route_heading
from .planner import (
    ObstacleBand,
    corridor_return_profile,
    geometric_obstacle_sets,
    plan_arc,
    rear_free_distance,
)
from .semantics import (
    SCENE_ROAD,
    SURFACE_MARKING,
    SemanticPalette,
    classify_scene_groups,
    classify_surface_materials,
    pack_rgb,
    pack_rgb_rows,
)

LOGGER = logging.getLogger(__name__)
UNKNOWN_SEMANTIC_RGB = np.asarray((1, 2, 3), dtype=np.uint8)
_EMPTY_BEV = np.empty((0, 2), dtype=np.float32)
_EMPTY_WORLD = np.empty((0, 3), dtype=np.float32)
_EMPTY_HEIGHTS = np.empty(0, dtype=np.float32)
_EMPTY_GROUPS = np.empty(0, dtype=np.uint8)
_NO_CANDIDATES = np.empty(0, dtype=np.float32)
# Stands in for a real plan when not a single sensor returned anything. An empty
# cloud is indistinguishable from a perfectly clear road, so it is reported as
# fully blocked and the controller brakes instead of accelerating into a map
# load. See test_it_brakes_rather_than_driving_blind_when_no_returns_arrive.
_BLIND_ARC = ArcPlan(
    curvature=0.0,
    free_distance_m=0.0,
    clearance_m=0.0,
    keep_right_target_m=None,
    nav_heading_rad=None,
    candidate_curvatures=_NO_CANDIDATES,
    candidate_costs=_NO_CANDIDATES,
    candidate_free_distances=_NO_CANDIDATES,
    next_curvature=0.0,
    transition_distance_m=0.0,
)
# Acquisition FPS decays to 0 when no real returns arrive inside this window, so
# ACQUISITION honestly reads 0.0 Hz during an outage while DISPLAY keeps ticking.
_ACQUISITION_STALE_S = 0.5
# Continuous failure budget before tearing the connection down. Time-based, not
# a strike count: three ticks is under a tenth of a second, which is far too
# eager to survive a map load.
_POLL_FAILURE_GRACE_S = 2.0
# A prefetched vehicle state older than this is re-polled instead of used: the
# position compensation below is linear in the age, so a state from before an
# app stall would be extrapolated across the whole stall.
_STATE_PREFETCH_MAX_AGE_S = 0.3
# Scans without a single marking return before the Marking check line reports
# the silence -- about ten seconds, enough driving to have crossed paint on any
# marked road.
_MARKING_SILENCE_SCANS = 250
# Cadence of the driving telemetry line. Well below the display tick: it exists
# to explain a run after the fact, not to trace every frame.
_TELEMETRY_INTERVAL_S = 1.0

# --- What counts as a hard stop worth measuring -------------------------------
#
# Purely a diagnostic trigger: nothing downstream reads these, and no braking
# behaviour depends on them. The point is to catch a human standing on the brake
# so the vehicle's real plant can be read off a normal drive, which is how a car
# other than the one in the config tables gets measured at all.
#
# 6.0 m/s^2 is far past engine braking, a lift-off or a trailing-throttle
# corner, and comfortably under the ~10 the reference car achieves -- so a
# vehicle that brakes considerably worse still registers. The release threshold
# is lower so a stop that eases off near rest is one event rather than several.
_MANUAL_BRAKE_DECEL_MPS2 = 6.0
_MANUAL_BRAKE_RELEASE_MPS2 = 2.0
# ...and it is only reported if it lasted and actually took speed off, so a dab
# at the pedal is not filed as a braking measurement.
_MANUAL_BRAKE_MIN_S = 0.3
_MANUAL_BRAKE_MIN_DROP_MPS = 3.0


class BeamNgWorker(QObject):
    status_changed = pyqtSignal(str, str)
    launch_ready = pyqtSignal()
    sensors_ready = pyqtSignal(str, object)
    frame_ready = pyqtSignal(object)
    perception_ready = pyqtSignal(object)
    sensors_stopped = pyqtSignal()
    self_driving_changed = pyqtSignal(bool)
    aeb_changed = pyqtSignal(bool)
    rear_aeb_changed = pyqtSignal(bool)
    fatal_error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._bng: BeamNGpy | None = None
        self._beamng_process: subprocess.Popen[bytes] | None = None
        self._vehicle: Vehicle | None = None
        self._sensors: list[Lidar] = []
        # Parallel to _sensors, so the per-unit reach diagnostic can name them.
        self._sensor_names: list[str] = []
        self._geometry: VehicleGeometry | None = None
        self._palette: SemanticPalette | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._poll_failures = 0
        self._first_failure_at: float | None = None
        self._last_speed = 0.0
        self._last_forward_speed = 0.0
        self._player_vid = ""
        self._actor_registry: dict[
            str, tuple[str, tuple[float, float, float]]
        ] = {}
        self._actor_observations: tuple[ActorObservation, ...] = ()
        self._last_actor_registry_at = -float("inf")
        self._last_actor_state_at = -float("inf")
        self._last_actor_success_at = -float("inf")

        self._self_driving = False
        self._controller: DrivingController | None = None
        self._route = None
        self._last_nav_poll_at = 0.0
        self._last_plan_at = 0.0
        self._last_control_at = 0.0
        self._last_control_ms = 0.0
        self._last_telemetry_at = 0.0
        self._last_logged_mode: str | None = None

        # AEB is a separate toggle and a separate object: it also runs with
        # self-driving off, under a human driver.
        self._aeb_enabled = False
        self._aeb = EmergencyBraking(FORWARD)
        # The rear system is the same machine on a 180-degree-rotated cloud,
        # with its own measured plant. See aeb.BrakingProfile.
        self._rear_aeb_enabled = False
        self._rear_aeb = EmergencyBraking(REVERSE)
        self._mirrored_geometry: VehicleGeometry | None = None
        # Per system, NOT shared. Both step in the same tick, so one timestamp
        # gave whichever ran second a dt of microseconds: its confirmation
        # window advanced 1 ms per tick instead of 40, so the rear brake needed
        # 4.8 s of continuous threat before it could fire and never did, and
        # its yaw rate came out 40x inflated the moment the car turned.
        self._last_aeb_at: dict[str, float] = {}
        # Whether the last control message we sent was an AEB-only brake, so the
        # release is sent exactly once when the event ends.
        self._aeb_brake_sent = False
        self._last_logged_aeb: dict[str, str] = {}
        # One-shot per-sensor reach diagnostic, emitted from the first tick that
        # actually carries returns.
        self._logged_reach = False
        # One-shot road-marking diagnostics: the palette has marking classes,
        # but whether the LiDAR's annotation pass labels road DECALS with them
        # is only knowable live. Two independent one-shots, because paint can
        # first appear long after a silence verdict was reasonable.
        self._logged_markings = False
        self._logged_marking_silence = False
        self._marking_free_scans = 0
        # Packed palette colour -> class name for every paint-ish class, built
        # at attach so the Marking check line can attribute counts per class.
        self._marking_names: dict[int, str] = {}
        # One-shot for the visual-paint experiment: what the road unit's
        # colour channel actually carries with annotation off.
        self._logged_colour_probe = False

        # --- Plant diagnostics, which change nothing ------------------------
        #
        # Every braking figure in config is a property of ONE vehicle and the
        # repo did not record which, so a report of braking too early or too
        # late had no baseline to be measured against. These three watch; they
        # never feed back into a trigger or a threshold.
        self._vehicle_model = ""
        # One recorder per AEB system, plus one for stops a human made. The
        # manual one is what measures a new vehicle's plant WITHOUT having to
        # provoke an AEB event first -- and it runs whether or not either system
        # is armed, because switching AEB off is exactly what someone does when
        # it brakes for nothing.
        self._aeb_events: dict[str, BrakeEvent] = {}
        self._manual_event: BrakeEvent | None = None
        self._manual_prev_speed = 0.0
        self._last_tick_at: float | None = None
        self._last_pitch_deg = 0.0
        # ARMED -> BRAKING transitions seen this tick, awaiting the evidence
        # line. Collected rather than logged in place because the cloud it
        # describes lives in `_poll_once`.
        self._pending_evidence: list[AebState] = []

        # The state poll is a ~33 ms blocking round-trip -- measured, and most
        # of the 40 ms tick -- while everything else the tick does is a few
        # milliseconds of numpy. It is therefore PREFETCHED: each tick submits
        # the next tick's poll to this one-thread pool on its way out, so the
        # round-trip runs while the worker thread is idle between timer fires,
        # and the next tick starts by collecting a result that is usually
        # already there. Socket safety is by CONSTRUCTION, not by locking:
        # the prefetch is submitted after the last vehicle-socket use of the
        # tick (`_actuate`) and collected before the first of the next, so at
        # any moment exactly one thread is using the connection.
        self._state_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="beamng-state"
        )
        self._state_future: Future[tuple[dict[str, Any], float]] | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._poll_timer.setInterval(DISPLAY_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_once)

    @pyqtSlot()
    def launch_beamng(self) -> None:
        # bridge_is_reachable() is checked last so the cheap handle tests
        # short-circuit. Without it, a BeamNG.tech started externally -- or by a
        # previous run of this app, which deliberately leaves it running -- is
        # invisible here and Launch spawns a second instance fighting for the
        # same port. It only proves *something* listens on the port, which is
        # still far better than the double spawn.
        if (
            self._bng is not None
            or (
                self._beamng_process is not None
                and self._beamng_process.poll() is None
            )
            or bridge_is_reachable()
        ):
            self.status_changed.emit("LAUNCHED", "BeamNG.tech is already running")
            self.launch_ready.emit()
            return
        if not BEAMNG_EXE.is_file():
            self._emit_fatal(f"BeamNG.tech executable was not found:\n{BEAMNG_EXE}")
            return

        command = build_launch_command()
        self.status_changed.emit("STARTING", "Starting BeamNG.tech process")
        LOGGER.info("Launching BeamNG.tech: %s", subprocess.list2cmdline(command))
        try:
            self._beamng_process = start_beamng_process()
        except Exception as exc:
            LOGGER.exception("Could not start BeamNG.tech")
            self._beamng_process = None
            self._emit_fatal(f"Could not launch BeamNG.tech: {exc}")
            return

        LOGGER.info("BeamNG.tech process created with PID %s", self._beamng_process.pid)
        self.status_changed.emit(
            "LAUNCHED", "BeamNG.tech launched; load a map and player vehicle"
        )
        self.launch_ready.emit()

    @pyqtSlot()
    def attach_to_player(self) -> None:
        if self._bng is None:
            if not bridge_is_reachable():
                self.status_changed.emit("LAUNCHED", "BeamNG.tech is still starting")
                self.fatal_error.emit(
                    "The BeamNG.tech communication bridge is not ready yet.\n\n"
                    "Wait until the map and player vehicle are fully loaded, "
                    "then retry."
                )
                return

            self.status_changed.emit("CONNECTING", "Connecting to BeamNG.tech")
            try:
                from beamngpy import BeamNGpy

                bng = BeamNGpy(
                    BEAMNG_HOST,
                    BEAMNG_PORT,
                    home=str(BEAMNG_HOME),
                    binary=BEAMNG_EXE.name,
                    quit_on_close=False,
                )
                self._bng = bng.open(launch=False)
                LOGGER.info(
                    "Connected to the BeamNG.tech bridge on port %d", BEAMNG_PORT
                )
            except Exception as exc:
                LOGGER.exception("Could not connect to the BeamNG.tech bridge")
                if self._bng is not None:
                    try:
                        self._bng.disconnect()
                    except Exception:
                        pass
                self._bng = None
                self.status_changed.emit("LAUNCHED", "BeamNG.tech bridge is not ready")
                self.fatal_error.emit(
                    f"Could not connect to BeamNG.tech: {exc}\n\n"
                    "Wait until the map and player vehicle are fully loaded, "
                    "then retry."
                )
                return

        from beamngpy.sensors import Lidar

        self._poll_timer.stop()
        self._cleanup_sensors()
        self.status_changed.emit("ATTACHING", "Finding the current player vehicle")

        try:
            player = self._bng.vehicles.get_player_vehicle_id()
            player_vid = str(player.get("vid", "")).strip()
            if not player_vid or int(player.get("id", -1)) < 0:
                raise RuntimeError("No player vehicle is active")
            self._player_vid = player_vid

            vehicles = self._bng.vehicles.get_current(include_config=False)
            vehicle = vehicles.get(player_vid)
            if vehicle is None:
                raise RuntimeError(
                    f"Player vehicle '{player_vid}' is not available through BeamNGpy"
                )

            vehicle.connect(self._bng)
            self._vehicle = vehicle
            self._vehicle_model = str(getattr(vehicle, "model", "") or "")
            self._attach_electrics(vehicle)
            state = self._get_vehicle_state()
            geometry = derive_vehicle_geometry(state, vehicle.get_bbox())
            self._log_vehicle_check(geometry)
            annotations = self._load_annotations()
            palette = SemanticPalette.from_annotations(annotations)
            # Every paint-ish class by its palette colour, INCLUDING the ones
            # deliberately excluded from MARKING_CLASSES: the per-class counts
            # in the Marking check line are the evidence for revisiting that
            # exclusion, so the excluded classes have to be counted too.
            self._marking_names = {
                pack_rgb(rgb): name.upper()
                for name, rgb in annotations.items()
                if name.upper()
                in (MARKING_CLASSES | {"DRIVING_INSTRUCTIONS", "SPEED_BUMP"})
            }

            sensor_prefix = f"bev_{os.getpid()}_{int(time.monotonic() * 1000)}"
            self._sensors = []
            self._sensor_names = []
            mount_count = len(geometry.mounts)
            bbox_z = self._bbox_bottom(vehicle)
            for index, mount in enumerate(geometry.mounts.values()):
                self.status_changed.emit(
                    "ATTACHING",
                    f"Attaching {mount.name} LiDAR ({index + 1}/{mount_count})",
                )
                sensor = Lidar(
                    f"{sensor_prefix}_{mount.name}",
                    self._bng,
                    vehicle,
                    requested_update_time=LIDAR_UPDATE_TIME_S,
                    update_priority=0.0,
                    pos=mount.position_vehicle,
                    dir=mount.direction_vehicle,
                    up=(0.0, 0.0, 1.0),
                    frequency=LIDAR_UPDATE_HZ,
                    # Every optic is per-mount, not global: the front unit
                    # reaches much further on a narrow, dense sweep, and the
                    # roof unit trades vertical aperture for ground-ring
                    # spacing. See SensorMount.
                    vertical_resolution=mount.vertical_resolution,
                    vertical_angle=mount.vertical_fov_deg,
                    horizontal_angle=mount.horizontal_fov_deg,
                    max_distance=mount.max_distance_m,
                    density=mount.density,
                    is_rotate_mode=False,
                    is_360_mode=False,
                    is_using_shared_memory=True,
                    is_visualised=False,
                    # BeamNG writes each latest 30 Hz scan directly into shared
                    # memory so the display loop never waits on four requests.
                    is_streaming=True,
                    # The road-scan unit runs unannotated while the visual-
                    # paint experiment is on: its colour channel then carries
                    # whatever the engine renders instead of class colours,
                    # and the one-shot `Colour check:` line reports what that
                    # actually is. See LIDAR_ROAD_VISUAL_COLOUR.
                    is_annotated=not (
                        LIDAR_ROAD_VISUAL_COLOUR and mount.name == "road"
                    ),
                    is_static=False,
                    is_snapping_desired=False,
                    is_force_inside_triangle=False,
                    is_dir_world_space=False,
                )
                self._sensor_names.append(mount.name)
                self._sensors.append(sensor)
                self._verify_mount_height(sensor, bbox_z, mount)

            self._geometry = geometry
            self._palette = palette
            self._frame_times.clear()
            self._poll_failures = 0
            self._first_failure_at = None
            self._last_speed = 0.0
            self._last_forward_speed = 0.0
        except Exception as exc:
            LOGGER.exception("Could not attach LiDAR sensors")
            self._cleanup_sensors()
            self.status_changed.emit("READY", f"Sensor attach failed: {exc}")
            self.fatal_error.emit(
                f"Could not attach to the player vehicle: {exc}\n\n"
                "Make sure a map is loaded and the intended EGO vehicle is selected."
            )
            return

        self.status_changed.emit(
            "STREAMING", f"{mount_count} LiDAR sensors active on {player_vid}"
        )
        self.sensors_ready.emit(player_vid, geometry)
        self._poll_timer.start()
        # Both emergency brakes arm themselves. They are protective rather than
        # behavioural -- neither steers, and neither touches a pedal until a
        # collision is otherwise unavoidable -- so defaulting them off would
        # mean the safety net is missing exactly when nobody thought about it.
        # Self-driving stays opt-in, because that one changes what the car does.
        self._set_aeb(True, rearward=False)
        self._set_aeb(True, rearward=True)

    @pyqtSlot(bool)
    def set_self_driving(self, enabled: bool) -> None:
        if not enabled:
            self._disengage_self_driving("Self-driving disengaged")
            return
        if (
            self._vehicle is None
            or self._geometry is None
            or not self._sensor_set_is_complete()
        ):
            self.self_driving_changed.emit(False)
            self.status_changed.emit(
                "READY", "Attach to a vehicle before engaging self-driving"
            )
            return

        self._controller = DrivingController()
        self._route = None
        self._last_nav_poll_at = 0.0
        self._last_plan_at = 0.0
        self._last_control_at = 0.0
        self._last_control_ms = 0.0
        self._last_telemetry_at = 0.0
        self._last_logged_mode = None
        try:
            # Only picks "realistic" over "arcade" -- drivetrain.setShifterMode
            # does NOT convert a manual gearbox into an automatic one. The
            # gearbox family comes from the vehicle's jbeam, which is why the
            # forward gear has to be detected rather than assumed.
            self._vehicle.set_shift_mode("realistic_automatic")
        except Exception:
            LOGGER.warning("Could not set the shifter mode", exc_info=True)

        self._self_driving = True
        # The offline suite can pin the arithmetic but never what the simulator
        # does with a value, so the conventions go in the log the way the mount
        # height does. If the car steers into obstacles, STEERING_SIGN is wrong;
        # if it never pulls away, compare the reported gear against the index.
        reported_gear = self._reported_gear()
        LOGGER.info(
            "Drive check: engaged, cap %.1f km/h, gearbox reports %r (%s family), "
            "forward gear %d, reverse gear %d, steering sign %+.0f "
            "(BeamNG steering is positive-right), %d plan families, "
            "lookahead %.0f-%.0f m, yaw-gain adaptation on",
            MAX_SPEED_MPS * 3.6,
            reported_gear,
            "manual" if isinstance(reported_gear, (int, float)) else "automatic",
            forward_gear_index(reported_gear),
            REVERSE_GEAR,
            STEERING_SIGN,
            len(TRANSITION_DISTANCES_M),
            LOOKAHEAD_MIN_M,
            LOOKAHEAD_MAX_M,
        )
        self.self_driving_changed.emit(True)
        self.status_changed.emit("STREAMING", "Self-driving engaged")

    def _disengage_self_driving(self, reason: str, announce: bool = True) -> None:
        """
        Stop driving and release the controls.

        Safe to call at any time, including when never engaged, and deliberately
        called from every teardown path *before* the vehicle handle is dropped.
        """
        was_engaged = self._self_driving
        self._self_driving = False
        self._controller = None
        self._route = None
        if was_engaged and self._vehicle is not None:
            try:
                # Released rather than braked: the human takes over a coasting
                # car, not one that suddenly stands on the brakes. The parking
                # brake goes back to 0 too, so nobody inherits a car we left
                # holding itself.
                self._vehicle.control(
                    steering=0.0, throttle=0.0, brake=0.0, parkingbrake=0.0
                )
            except Exception:
                LOGGER.debug("Could not zero the vehicle controls", exc_info=True)
        if was_engaged:
            LOGGER.info("Self-driving disengaged: %s", reason)
            self.self_driving_changed.emit(False)
            if announce:
                self.status_changed.emit("STREAMING", reason)

    @pyqtSlot(bool)
    def set_aeb(self, enabled: bool) -> None:
        self._set_aeb(enabled, rearward=False)

    @pyqtSlot(bool)
    def set_rear_aeb(self, enabled: bool) -> None:
        self._set_aeb(enabled, rearward=True)

    def _set_aeb(self, enabled: bool, rearward: bool) -> None:
        """
        Arm or disarm one of the two systems.

        They are separate toggles over one implementation: same corridor scan,
        same phantom filters, same state machine, different measured plant and
        opposite direction. See aeb.BrakingProfile.
        """
        changed = self.rear_aeb_changed if rearward else self.aeb_changed
        label = "Reverse AEB" if rearward else "Emergency braking"
        if not enabled:
            self._disengage_aeb(f"{label} disabled", rearward=rearward)
            return
        if (
            self._vehicle is None
            or self._geometry is None
            or not self._sensor_set_is_complete()
        ):
            changed.emit(False)
            self.status_changed.emit(
                "READY", f"Attach to a vehicle before arming {label.lower()}"
            )
            return

        system = self._rear_aeb if rearward else self._aeb
        system.reset()
        if rearward:
            self._rear_aeb_enabled = True
        else:
            self._aeb_enabled = True
        self._aeb_brake_sent = False
        self._last_aeb_at.pop(system.profile.label, None)
        self._last_logged_aeb.pop(system.profile.label, None)
        LOGGER.info(
            "%s check: on %r, armed above %.1f km/h, stopping %.2f m clear of "
            "the %s, fires at the last point a %.1f m/s^2 stop still works and "
            "then brakes FULL, corridor %.2f m wide, obstacle floor %.2f m "
            "with %d hits over %.2f s required, path predicted from measured "
            "yaw (brake only). Brake-now distances: %s",
            system.profile.label,
            self._vehicle_model or "unknown",
            system.profile.min_speed_mps * 3.6,
            system.profile.standoff_m,
            "rear bumper" if rearward else "front bumper",
            system.profile.braking_decel_mps2,
            self._geometry.width_m,
            AEB_OBSTACLE_MIN_HEIGHT_M,
            AEB_MIN_HITS,
            AEB_CONFIRM_S,
            self._brake_now_table(system.profile, rearward=rearward),
        )
        changed.emit(True)
        self.status_changed.emit("STREAMING", f"{label} armed")

    @staticmethod
    def _porosity_sensor_height(geometry: VehicleGeometry) -> float:
        """
        The eye height AEB's porosity test reasons from: the ROOF unit.

        It has to be the roof unit and not one of the 0.20 m mounts, because
        those sit BELOW anything worth testing -- a 0.6 m bush hides the ground
        behind it completely from 0.20 m, so from down there a bush and a wall
        are indistinguishable by construction. Only the unit above the bodywork
        can see over a short object at all.

        Zero when there is no roof mount, which disables the veto entirely --
        the conservative direction, since the test can only ever remove
        obstacles.
        """
        roof = geometry.mounts.get("roof")
        return float(roof.position_vehicle[2]) if roof is not None else 0.0

    def _log_vehicle_check(self, geometry: VehicleGeometry) -> None:
        """
        Which car this is, and every dimension the two features derive from it.

        Diagnostics, and the baseline the plant figures never had. The braking
        tables, AEB_OBSTACLE_MIN_HEIGHT_M and the standoffs were all measured on
        one vehicle, and until this line existed nothing recorded which -- so
        "it brakes for nothing on the pickup" had no way of being compared
        against the car that behaves.

        The WIDTH is the number to read first. It is the full oriented bounding
        box, so anything bolted to the bodywork is inside it, and the corridor
        both AEB systems scan is that width plus AEB_CLEARANCE_MARGIN_M. If it
        reads well over the real body width, the corridor is sweeping a band no
        collision could happen in and kerbs beside the path fall inside it.
        """
        mount_heights = ", ".join(
            f"{name} {mount.position_vehicle[2]:.2f}"
            for name, mount in geometry.mounts.items()
        )
        LOGGER.info(
            "Vehicle check: model %r (plant measured on %r) | bbox %.2f wide x "
            "%.2f long x %.2f tall | overhang front %.2f rear %.2f | body "
            "centre %+.2f m right of the reference node (the AEB corridor and "
            "the WORLD ego are shifted by this) | ground plane %+.3f m from "
            "the reference node | mount z: %s",
            self._vehicle_model or "unknown",
            PLANT_REFERENCE_VEHICLE,
            geometry.width_m,
            geometry.length_m,
            geometry.height_m,
            geometry.front_m,
            geometry.rear_m,
            (geometry.right_m - geometry.left_m) / 2.0,
            geometry.ground_z_vehicle,
            mount_heights,
        )
        if (
            self._vehicle_model
            and self._vehicle_model != PLANT_REFERENCE_VEHICLE
        ):
            # Not a warning about safety-critical behaviour -- nothing here
            # changes what AEB does -- but the one line that says the numbers
            # being applied were measured on a different car.
            LOGGER.warning(
                "Vehicle check: %r is not the vehicle the braking plant was "
                "measured on (%r). AEB is using %.1f m/s^2 forward and "
                "%.1f m/s^2 reverse regardless; drive a full stop and compare "
                "the 'Brake measure:' line before trusting the fire distances.",
                self._vehicle_model,
                PLANT_REFERENCE_VEHICLE,
                FORWARD.braking_decel_mps2,
                REVERSE.braking_decel_mps2,
            )

    def _brake_now_table(
        self, profile: BrakingProfile, rearward: bool = False
    ) -> str:
        """
        Where this system will actually fire, per speed, for this vehicle.

        The offline suite can pin the arithmetic but never what the car does
        with it, and "it braked too late" versus "it braked too early" is the
        one AEB complaint that needs a number to settle.
        """
        assert self._geometry is not None
        overhang = (
            self._geometry.rear_m if rearward else self._geometry.front_m
        )
        standoff = overhang + profile.standoff_m
        speeds = (10, 20, 30, 50) if rearward else (30, 50, 70, 100)
        return ", ".join(
            f"{kph} km/h -> "
            f"{self._fire_distance(standoff, profile, kph):.1f} m"
            for kph in speeds
        )

    @staticmethod
    def _fire_distance(
        standoff: float, profile: BrakingProfile, kph: int
    ) -> float:
        return standoff + AEB_TRIGGER_MARGIN * stopping_distance(
            kph / 3.6, profile
        )

    def _disengage_aeb(
        self, reason: str, announce: bool = True, rearward: bool | None = None
    ) -> None:
        """
        Switch a system off and hand the brake back.

        ``rearward=None`` means both, which is what every teardown path wants.
        Like `_disengage_self_driving` this is safe at any time and runs while
        the vehicle handle is still live. The release message is only sent if
        one of them was actually holding the pedal -- otherwise it would stamp
        on a human's own braking.
        """
        was_braking = self._aeb_brake_sent
        for rear in (False, True):
            if rearward is not None and rear != rearward:
                continue
            enabled = self._rear_aeb_enabled if rear else self._aeb_enabled
            if rear:
                self._rear_aeb_enabled = False
                self._rear_aeb.reset()
            else:
                self._aeb_enabled = False
                self._aeb.reset()
            if enabled:
                LOGGER.info(
                    "%s disabled: %s", "REAR AEB" if rear else "AEB", reason
                )
                (self.rear_aeb_changed if rear else self.aeb_changed).emit(False)
                if announce:
                    self.status_changed.emit("STREAMING", reason)
        if not (self._aeb_enabled or self._rear_aeb_enabled):
            self._aeb_brake_sent = False
            if was_braking and self._vehicle is not None:
                try:
                    self._vehicle.control(throttle=0.0, brake=0.0)
                except Exception:
                    LOGGER.debug(
                        "Could not release the AEB brake", exc_info=True
                    )

    @pyqtSlot()
    def stop_sensors(self) -> None:
        self._poll_timer.stop()
        self._cleanup_sensors()
        if self._bng is not None:
            self.status_changed.emit("READY", "Sensors stopped")
        else:
            self.status_changed.emit("OFFLINE", "Disconnected")
        self.sensors_stopped.emit()

    @pyqtSlot()
    def handle_bridge_lost(self) -> None:
        """
        Tear down after BeamNG.tech disappears.

        Deliberately not stop_sensors(): that leaves _bng set and would report
        READY, inviting the user to re-attach to a simulator that is gone.
        """
        self._poll_timer.stop()
        self._cleanup_sensors()
        if self._bng is not None:
            try:
                self._bng.disconnect()
            except Exception:
                LOGGER.debug("Disconnect after bridge loss failed", exc_info=True)
            self._bng = None
        self._beamng_process = None
        self._first_failure_at = None
        self.status_changed.emit("OFFLINE", "BeamNG.tech is no longer running")
        self.sensors_stopped.emit()

    @pyqtSlot()
    def shutdown(self) -> None:
        self._poll_timer.stop()
        self._cleanup_sensors()
        self._state_pool.shutdown(wait=False)
        if self._bng is not None:
            try:
                # quit_on_close=False: closing this app leaves BeamNG.tech open.
                self._bng.disconnect()
            except Exception:
                LOGGER.debug(
                    "BeamNG.tech disconnect failed during shutdown", exc_info=True
                )
            self._bng = None

    @pyqtSlot()
    def _poll_once(self) -> None:
        if (
            self._bng is None
            or self._vehicle is None
            or self._geometry is None
            or self._palette is None
            or not self._sensor_set_is_complete()
        ):
            return

        started = time.perf_counter()
        geometry = self._geometry
        road_points = _EMPTY_BEV
        obstacle_points = _EMPTY_BEV
        scene_points_world = _EMPTY_WORLD
        scene_groups = _EMPTY_GROUPS
        scene_materials = _EMPTY_GROUPS
        # Kept in scope for the planner, which needs the undecimated cloud and
        # the heights the semantic split throws away.
        bev = _EMPTY_BEV
        heights = _EMPTY_HEIGHTS
        raw_point_count = 0
        had_returns = False
        try:
            state = self._take_vehicle_state()
            velocity = vec3(state.get("vel", (0.0, 0.0, 0.0)))
            self._last_speed = float(np.linalg.norm(velocity))
            # Signed, and computed here rather than in the control block below,
            # because the scene needs it on every tick: the block that already
            # derives `forward_speed` only runs when self-driving or AEB is on,
            # and reversing under a human driver is precisely when neither is.
            forward_axis = vehicle_axes(state)[1]
            self._last_forward_speed = float(velocity @ forward_axis)
            # Body pitch, positive nose-up: a stop down a grade flatters the
            # plant and a stop up one slanders it, so every brake measurement
            # carries the number rather than pretending to correct for it.
            self._last_pitch_deg = float(
                np.degrees(np.arcsin(np.clip(forward_axis[2], -1.0, 1.0)))
            )
            self._watch_manual_braking(started)
            point_chunks: list[np.ndarray] = []
            colour_chunks: list[np.ndarray] = []

            reach: list[tuple[str, int, float]] = []
            for index, sensor in enumerate(self._sensors):
                reading = sensor.stream()
                points = self._coerce_points(reading.get("pointCloud"))
                if not self._logged_reach and len(points):
                    reach.append(
                        self._sensor_reach(index, points, vec3(state["pos"]))
                    )
                if not len(points):
                    continue
                colours = self._coerce_colours(reading.get("colours"), len(points))
                # The names list is parallel to _sensors from attach; offline
                # stubs may arm sensors without it, and they have no road unit.
                if index < len(self._sensor_names):
                    self._watch_visual_colours(
                        self._sensor_names[index], colours
                    )
                finite = np.isfinite(points).all(axis=1)
                if not finite.all():
                    points = points[finite]
                    colours = colours[finite]
                if not len(points):
                    continue
                point_chunks.append(points)
                colour_chunks.append(colours)

            if reach and not self._logged_reach:
                self._logged_reach = True
                # The one number the long-range front unit has to be judged on,
                # and the offline suite cannot reach it: whether narrowing the
                # sweep bought the azimuth density that reaching 150 m needs.
                LOGGER.info(
                    "Sensor reach: %s | AEB acts out to %.0f m",
                    " | ".join(
                        f"{name} {count} returns, furthest {far:.0f} m"
                        for name, count, far in reach
                    ),
                    AEB_MAX_HORIZON_M,
                )

            if point_chunks:
                had_returns = True
                points_world = np.concatenate(point_chunks, axis=0)
                colours = np.concatenate(colour_chunks, axis=0)
                bev, heights = world_points_to_bev(points_world, state)

                in_range = np.einsum("ij,ij->i", bev, bev) <= LIDAR_RANGE_M**2
                bev = bev[in_range]
                heights = heights[in_range]
                colours = colours[in_range]
                points_world = points_world[in_range]

                ego_margin = 0.18
                inside_ego = (
                    (bev[:, 0] >= -geometry.left_m - ego_margin)
                    & (bev[:, 0] <= geometry.right_m + ego_margin)
                    & (bev[:, 1] >= -geometry.rear_m - ego_margin)
                    & (bev[:, 1] <= geometry.front_m + ego_margin)
                )
                if inside_ego.any():
                    keep = ~inside_ego
                    bev = bev[keep]
                    heights = heights[keep]
                    colours = colours[keep]
                    points_world = points_world[keep]

                # One semantic pass, not two. The BEV split and the WORLD
                # vocabulary used to be classified independently, which packed
                # the cloud three times and ran the road rule twice -- all
                # O(cloud), and the fifth sensor made every one of them dearer.
                # `groups == SCENE_ROAD` IS the road mask; see
                # classify_scene_groups.
                scene_groups = classify_scene_groups(
                    colours,
                    heights,
                    geometry.ground_z_vehicle,
                    self._palette,
                )
                # What each surface is MADE OF, which is a separate question
                # from which group a return belongs to and is answered from the
                # colour alone -- one searchsorted, no height and no ground
                # plane. WORLD decides what IS ground from shape; this only
                # says what colour the ground it finds should be.
                scene_materials = classify_surface_materials(
                    colours, self._palette
                )
                self._watch_for_markings(scene_materials, colours)
                road_mask = scene_groups == SCENE_ROAD
                road_points = self._limit_points(
                    bev[road_mask], MAX_ROAD_RENDER_POINTS
                )
                obstacle_points = self._limit_points(
                    bev[~road_mask], MAX_OBSTACLE_RENDER_POINTS
                )
                scene_points_world = np.ascontiguousarray(
                    points_world, dtype=np.float32
                )
                raw_point_count = len(bev)

            finished = time.perf_counter()
            # ACQUISITION must keep meaning "LiDAR rate", not "tick rate", so the
            # window is flushed rather than smeared across a gap in the returns.
            stale = (
                self._frame_times
                and finished - self._frame_times[-1] > _ACQUISITION_STALE_S
            )
            if stale:
                self._frame_times.clear()
            if had_returns:
                self._frame_times.append(finished)

            plan: DrivingPlan | None = None
            aeb: AebState | None = None
            rear_aeb: AebState | None = None
            if self._self_driving or self._aeb_enabled or self._rear_aeb_enabled:
                # Every driving step below is isolated from the poll-failure
                # budget on purpose: a planner, AEB or controller bug must not
                # masquerade as a lost bridge and tear the whole connection down
                # two seconds later. The three steps are isolated from each
                # other too, so a fault in one never switches the other off.
                obstacles = _EMPTY_BEV
                aeb_obstacles = _EMPTY_BEV
                measured = had_returns

                _, forward, _ = vehicle_axes(state)
                forward_speed = float(
                    vec3(state.get("vel", (0.0, 0.0, 0.0))) @ forward
                )
                # World yaw of the vehicle's forward axis; both the controller
                # and AEB derive their measured curvature from successive values.
                heading = float(np.arctan2(forward[1], forward[0]))
                if self._mirrored_geometry is None:
                    self._mirrored_geometry = mirrored(geometry)
                mirror_geometry = self._mirrored_geometry

                try:
                    if had_returns:
                        # Two bands off one ground estimate. The planner steers
                        # around kerbs inside its own horizon; AEB brakes only
                        # for crashes (so the flat road must never reach its
                        # set, see AEB_OBSTACLE_MIN_HEIGHT_M) but has to see far
                        # enough to stop from motorway speed. Both are built
                        # even when only one feature is on: the second band
                        # costs 0.7 ms against the 2.6 ms the shared ground
                        # estimate does.
                        #
                        # AEB's reach is asked of the AEB object rather than
                        # fixed at AEB_MAX_HORIZON_M, so the band and the scan
                        # agree AND the extraction only pays for the range this
                        # speed actually uses -- the whole 150 m costs 24.5 ms
                        # of the 40 ms tick, against 5.5 ms at the ~30 m a
                        # 50 km/h scan wants.
                        # ONE band serves both AEB directions: the height band
                        # is a radial cull, and the corridor scans differ only
                        # in which way they point. Sized to whichever system
                        # reaches further at this speed.
                        obstacles, aeb_obstacles = geometric_obstacle_sets(
                            bev,
                            heights,
                            geometry.ground_z_vehicle,
                            (
                                # The planner's band is unchanged, and the two
                                # shape tests are deliberately off for it: it
                                # SHOULD steer around a kerb and around a bush.
                                ObstacleBand(
                                    OBSTACLE_MIN_HEIGHT_M, PLANNER_HORIZON_M
                                ),
                                ObstacleBand(
                                    AEB_OBSTACLE_MIN_HEIGHT_M,
                                    max(
                                        self._aeb.horizon_for(
                                            forward_speed, geometry
                                        ),
                                        self._rear_aeb.horizon_for(
                                            -forward_speed, mirror_geometry
                                        ),
                                    ),
                                    min_vertical_extent_m=(
                                        AEB_MIN_VERTICAL_EXTENT_M
                                    ),
                                    porosity=True,
                                ),
                            ),
                            sensor_height_m=self._porosity_sensor_height(
                                geometry
                            ),
                        )
                except Exception:
                    LOGGER.exception("Obstacle extraction failed")
                    # Degrade to blind rather than to clear: self-driving brakes
                    # and AEB holds whatever it already had.
                    obstacles = _EMPTY_BEV
                    aeb_obstacles = _EMPTY_BEV
                    measured = False

                if self._self_driving:
                    try:
                        plan = self._compute_plan(
                            state,
                            obstacles,
                            geometry,
                            forward_speed,
                            heading,
                            measured,
                        )
                    except Exception as exc:
                        LOGGER.exception("Self-driving planning failed")
                        self._disengage_self_driving(
                            f"Self-driving stopped: {exc}"
                        )
                if self._aeb_enabled:
                    try:
                        aeb = self._compute_aeb(
                            self._aeb,
                            aeb_obstacles,
                            geometry,
                            forward_speed,
                            heading,
                            measured,
                        )
                    except Exception as exc:
                        LOGGER.exception("Emergency braking failed")
                        self._disengage_aeb(
                            f"Emergency braking stopped: {exc}", rearward=False
                        )
                if self._rear_aeb_enabled:
                    try:
                        # The same machine on a 180-degree-rotated cloud, with
                        # the sign of the speed flipped so "ahead" means
                        # behind. The heading is NOT flipped: path curvature is
                        # yaw / |speed| in the frame of travel either way, and
                        # a 180-degree rotation preserves handedness -- see
                        # aeb.mirrored.
                        rear_aeb = self._compute_aeb(
                            self._rear_aeb,
                            mirror_points(aeb_obstacles),
                            mirror_geometry,
                            -forward_speed,
                            heading,
                            measured,
                        )
                    except Exception as exc:
                        LOGGER.exception("Reverse emergency braking failed")
                        self._disengage_aeb(
                            f"Reverse AEB stopped: {exc}", rearward=True
                        )

                if self._pending_evidence:
                    # Its own handler: a diagnostic must never disarm a brake,
                    # and must never reach the outer handler either, where it
                    # would be counted against the poll-failure budget and read
                    # as a lost bridge. It costs a `ground_rise` pass over the
                    # cloud and runs only on the tick a system fires.
                    try:
                        self._log_aeb_evidence(bev, heights, geometry)
                    except Exception:
                        LOGGER.exception("AEB evidence logging failed")
                        self._pending_evidence = []

            frame = BevFrame(
                road_points=road_points,
                obstacle_points=obstacle_points,
                raw_point_count=raw_point_count,
                acquisition_fps=self._calculate_fps(),
                poll_ms=(finished - started) * 1000.0,
                speed_mps=self._last_speed,
                vehicle_geometry=geometry,
                plan=plan,
                # The previous tick's figure: actuation deliberately happens
                # after the emit, so this tick's cost is not known yet.
                control_ms=self._last_control_ms,
                aeb=aeb,
                rear_aeb=rear_aeb,
            )
            self._poll_failures = 0
            self._first_failure_at = None
            self.frame_ready.emit(frame)

            # Actuate only after the frame is out. Vehicle.control() is a
            # blocking ack, and the display must not wait on it.
            self._actuate(plan, aeb, rear_aeb)
            snapshot_time = time.perf_counter()
            actors = self._poll_actor_observations(snapshot_time)
            self.perception_ready.emit(
                PerceptionSnapshot(
                    points_world=scene_points_world,
                    semantic_groups=scene_groups,
                    surface_materials=scene_materials,
                    ego_pos_world=tuple(
                        float(value) for value in vec3(state["pos"])
                    ),
                    ego_dir_world=tuple(
                        float(value) for value in vec3(state["dir"])
                    ),
                    ego_up_world=tuple(
                        float(value) for value in vec3(state["up"])
                    ),
                    timestamp=snapshot_time,
                    speed_mps=self._last_speed,
                    forward_speed_mps=self._last_forward_speed,
                    vehicle_geometry=geometry,
                    actors=actors,
                    plan=plan,
                    aeb=aeb,
                    rear_aeb=rear_aeb,
                )
            )
            # The last statement of a successful tick, deliberately after
            # `_actuate` and the actor poll: nothing on the worker thread will
            # touch a socket again until the next tick joins this future.
            self._prefetch_vehicle_state()
        except Exception as exc:
            now = time.perf_counter()
            if self._first_failure_at is None:
                self._first_failure_at = now
            self._poll_failures += 1
            failing_for = now - self._first_failure_at
            LOGGER.warning(
                "LiDAR polling failed (%d attempts over %.1fs): %s",
                self._poll_failures,
                failing_for,
                exc,
            )
            if failing_for >= _POLL_FAILURE_GRACE_S:
                self._poll_timer.stop()
                self._cleanup_sensors()
                if self._bng is not None:
                    try:
                        self._bng.disconnect()
                    except Exception:
                        LOGGER.debug("Disconnect after poll failure", exc_info=True)
                self._bng = None
                self._first_failure_at = None
                self.status_changed.emit("ERROR", "BeamNG.tech connection was lost")
                # Without this the stale frame stays on screen behind the dialog.
                self.sensors_stopped.emit()
                self.fatal_error.emit(
                    f"LiDAR streaming stopped after repeated connection errors: {exc}"
                )

    def _compute_plan(
        self,
        state: dict[str, Any],
        obstacles: np.ndarray,
        geometry: VehicleGeometry,
        forward_speed: float,
        heading: float,
        had_returns: bool,
    ) -> DrivingPlan:
        assert self._controller is not None

        if had_returns:
            # ~LOOKAHEAD_TIME_S seconds of travel, so keep-right and the nav
            # hint keep their tuned character at any speed.
            lookahead = min(
                LOOKAHEAD_MAX_M,
                max(LOOKAHEAD_MIN_M, LOOKAHEAD_TIME_S * abs(forward_speed)),
            )
            arc = plan_arc(
                obstacles,
                geometry,
                nav_heading_rad=self._nav_heading(state),
                # The curvature actually being driven right now, which is also
                # segment A of every deferred candidate.
                previous_curvature=self._controller.current_curvature,
                lookahead_m=lookahead,
            )
            rear_free_m = rear_free_distance(obstacles, geometry)
            obstacle_count = len(obstacles)
        else:
            arc = _BLIND_ARC
            rear_free_m = 0.0
            obstacle_count = 0

        now = time.perf_counter()
        dt = (
            now - self._last_plan_at
            if self._last_plan_at
            else DISPLAY_INTERVAL_MS / 1000.0
        )
        self._last_plan_at = now

        reported_gear = self._reported_gear()
        command = self._controller.step(
            arc,
            forward_speed,
            dt,
            rear_free_distance_m=rear_free_m,
            reported_gear=reported_gear,
            heading_rad=heading,
        )
        self._log_driving_telemetry(arc, command, forward_speed, obstacle_count)
        return DrivingPlan(
            arc=arc,
            command=command,
            forward_speed_mps=forward_speed,
            reported_gear=reported_gear,
        )

    def _log_driving_telemetry(
        self,
        arc: ArcPlan,
        command: ControlCommand,
        speed: float,
        obstacle_count: int,
    ) -> None:
        """
        One line a second of why the car did what it did.

        Without this the only evidence a live run leaves behind is "it braked
        and I do not know why": free distance, the speed target and the
        commanded-versus-measured curvature are the three signals that separate
        a real obstacle from a phantom one, and none of them are visible
        anywhere else. Rate-limited well below the display tick, and a mode
        change always prints so the blocked/reverse sequence is never elided.
        """
        assert self._controller is not None
        now = time.monotonic()
        changed = command.mode != self._last_logged_mode
        if not changed and now - self._last_telemetry_at < _TELEMETRY_INTERVAL_S:
            return
        self._last_telemetry_at = now
        self._last_logged_mode = command.mode
        measured = self._controller.measured_curvature
        LOGGER.info(
            "Drive: %s %.1f/%.1f km/h thr %.2f brk %.2f | free %.1f m clear "
            "%.2f m obstacles %d | k cmd %+.4f driven %+.4f measured %s "
            "gain %.2f | defer %.0f m -> %+.4f | %s",
            command.mode,
            speed * 3.6,
            command.target_speed_mps * 3.6,
            command.throttle,
            command.brake,
            arc.free_distance_m,
            arc.clearance_m,
            obstacle_count,
            arc.curvature,
            self._controller.current_curvature,
            "n/a" if measured is None else f"{measured:+.4f}",
            self._controller.steering_gain,
            arc.transition_distance_m,
            arc.next_curvature,
            command.reason,
        )

    def _compute_aeb(
        self,
        system: EmergencyBraking,
        obstacles: np.ndarray,
        geometry: VehicleGeometry,
        forward_speed: float,
        heading: float,
        had_returns: bool,
    ) -> AebState:
        label = system.profile.label
        now = time.perf_counter()
        previous = self._last_aeb_at.get(label)
        dt = now - previous if previous else DISPLAY_INTERVAL_MS / 1000.0
        self._last_aeb_at[label] = now
        state = system.step(
            obstacles,
            geometry,
            forward_speed,
            dt,
            heading_rad=heading,
            has_returns=had_returns,
        )
        if state.status == BRAKING and self._last_logged_aeb.get(label) != BRAKING:
            # Before `_log_aeb`, which is what owns that dict.
            self._pending_evidence.append(state)
        self._watch_aeb_braking(system, state, forward_speed, dt)
        self._log_aeb(label, state, forward_speed)
        return state

    # --- Plant diagnostics ---------------------------------------------------
    #
    # None of the four methods below influences anything. They measure the car
    # that is actually attached, because every braking figure in `config` is a
    # property of one particular vehicle and applying them to another is a
    # question nobody could previously answer with a number.

    def _watch_aeb_braking(
        self,
        system: EmergencyBraking,
        state: AebState,
        forward_speed: float,
        dt: float,
    ) -> None:
        """Record what a firing actually achieved, against what it assumed."""
        label = system.profile.label
        event = self._aeb_events.get(label)
        if event is None:
            event = BrakeEvent(system.profile)
            self._aeb_events[label] = event
        if state.engaged:
            if not event.active:
                # AEB owns the pedal now, so whatever the human was doing stops
                # being a measurement of the human.
                self._manual_event = None
                event.start(forward_speed, self._last_pitch_deg, label)
            else:
                event.sample(forward_speed, dt)
        elif event.active:
            event.sample(forward_speed, dt)
            self._log_brake_measurement(event.finish())

    def _watch_visual_colours(self, name: str, colours: np.ndarray) -> None:
        """
        Stage one of the visual-paint experiment, as one log line.

        With LIDAR_ROAD_VISUAL_COLOUR on, the road unit runs unannotated and
        this reports what its colour channel actually carries -- the
        undocumented fact the whole experiment turns on. Reading the verdict:
        near-100% black or a handful of unique values means the channel is
        dead in this mode and the experiment ends here; thousands of unique
        colours with a bright luminance tail on a marked road means it is the
        rendered scene, and paint is readable by brightness -- stage two.
        """
        if (
            self._logged_colour_probe
            or not LIDAR_ROAD_VISUAL_COLOUR
            or name != "road"
            or not len(colours)
        ):
            return
        self._logged_colour_probe = True
        luminance = colours.astype(np.float64) @ (0.2126, 0.7152, 0.0722)
        black = 100.0 * float(np.mean(np.all(colours == 0, axis=1)))
        unique = int(len(np.unique(pack_rgb_rows(colours))))
        spread = ", ".join(
            f"p{p} {v:.0f}"
            for p, v in zip(
                (5, 25, 50, 75, 95),
                np.percentile(luminance, (5, 25, 50, 75, 95)),
            )
        )
        bright = 100.0 * float(np.mean(luminance > 160.0))
        LOGGER.info(
            "Colour check: road unit visual channel over %d returns -- "
            "%.1f%% pure black, %d unique colours, luminance %s, %.2f%% "
            "above 160. Dead channel: near-100%% black or single-digit "
            "unique colours. Rendered scene: thousands of colours and a "
            "bright tail wherever there is paint.",
            len(colours),
            black,
            unique,
            spread,
            bright,
        )

    def _watch_for_markings(
        self, materials: np.ndarray, colours: np.ndarray
    ) -> None:
        """
        The live check the marking feature depends on, as one log line.

        Markings are DECALS, and the annotation labels a decal's whole QUAD,
        transparent texels included -- which is why the wide-area classes
        (DRIVING_INSTRUCTIONS, SPEED_BUMP) are excluded from MARKING_CLASSES:
        their footprints flooded entire junctions with paint. The per-class
        breakdown here is the evidence that exclusion rests on, counted for
        the excluded classes too so it can be revisited from numbers.
        """
        if self._logged_markings:
            return
        count = int(np.count_nonzero(materials == SURFACE_MARKING))
        if count:
            self._logged_markings = True
            LOGGER.info(
                "Marking check: %d road-marking returns in a %d-return scan "
                "(%s). Lane paint will draw; classes not in MARKING_CLASSES "
                "render as tarmac however many returns they post.",
                count,
                len(materials),
                self._marking_breakdown(colours),
            )
            return
        self._marking_free_scans += 1
        if (
            self._marking_free_scans == _MARKING_SILENCE_SCANS
            and not self._logged_marking_silence
        ):
            self._logged_marking_silence = True
            LOGGER.info(
                "Marking check: no road-marking returns in %d scans (%s). "
                "Either this stretch is unmarked, or the LiDAR annotation "
                "labels decals as the road beneath them -- drive over lane "
                "lines to settle it. This line will follow up if paint ever "
                "appears.",
                _MARKING_SILENCE_SCANS,
                self._marking_breakdown(colours),
            )

    def _marking_breakdown(self, colours: np.ndarray) -> str:
        """Per-class return counts for every paint-ish class, one-shot cost."""
        if not self._marking_names or not len(colours):
            return "no paint classes in this map's palette"
        packed = pack_rgb_rows(colours)
        parts = [
            f"{name} {hits}"
            for code, name in sorted(self._marking_names.items())
            if (hits := int(np.count_nonzero(packed == code)))
        ]
        return ", ".join(parts) if parts else "no paint-class returns"

    def _watch_manual_braking(self, now: float) -> None:
        """
        Catch a human standing on the brake, and measure that stop.

        This is how a vehicle other than the one in the config tables gets
        measured: drive it, brake hard, read the line. It runs on every tick
        regardless of which features are armed, because switching AEB off is the
        first thing anyone does when it brakes for nothing -- and that is
        precisely when the plant needs measuring.

        Deceleration comes from the speed trace alone. Reading the pedal would
        mean polling `electrics` on every tick in the plain viewer, which is a
        round trip this loop deliberately does not make.
        """
        previous = self._manual_prev_speed
        speed = abs(self._last_forward_speed)
        last_tick = self._last_tick_at
        self._last_tick_at = now
        self._manual_prev_speed = speed
        if last_tick is None:
            return
        dt = max(now - last_tick, 1e-3)
        if dt > _ACQUISITION_STALE_S:
            # A gap in the ticks is not a deceleration.
            self._manual_event = None
            return
        if any(event.active for event in self._aeb_events.values()):
            # AEB is holding the pedal, and its own recorder has this stop. The
            # check has to be here rather than only where an AEB event STARTS:
            # the deceleration does not cross this threshold until a tick or two
            # after the brake goes on, by which point that branch has been and
            # gone -- and the same stop was then filed twice, once each way.
            self._manual_event = None
            return
        decel = (previous - speed) / dt
        event = self._manual_event
        if event is None:
            if decel < _MANUAL_BRAKE_DECEL_MPS2:
                return
            reversing = self._last_forward_speed < 0.0
            event = BrakeEvent(REVERSE if reversing else FORWARD)
            # From the speed BEFORE this tick's drop: that is where the stop
            # started, and the trace has to include the metres already covered.
            event.start(
                previous,
                self._last_pitch_deg,
                "MANUAL REVERSE" if reversing else "MANUAL",
            )
            self._manual_event = event
        event.sample(speed, dt)
        if decel >= _MANUAL_BRAKE_RELEASE_MPS2 and speed > 0.0:
            return
        self._manual_event = None
        self._log_brake_measurement(event.finish())

    def _log_brake_measurement(
        self, measurement: BrakeMeasurement | None
    ) -> None:
        if measurement is None:
            return
        if (
            measurement.duration_s < _MANUAL_BRAKE_MIN_S
            or measurement.from_speed_mps - measurement.to_speed_mps
            < _MANUAL_BRAKE_MIN_DROP_MPS
        ):
            # Too short or too gentle to say anything about the plant.
            return
        LOGGER.info(
            "Brake measure: %s stop on %r | %.1f -> %.1f km/h in %.2f s over "
            "%.1f m | achieved %.2f m/s^2 mean, %.2f peak | model says %.1f m "
            "at %.1f m/s^2 (%.2fx) | pitch %+.1f deg",
            measurement.cause,
            self._vehicle_model or "unknown",
            measurement.from_speed_mps * 3.6,
            measurement.to_speed_mps * 3.6,
            measurement.duration_s,
            measurement.distance_m,
            measurement.mean_decel_mps2,
            measurement.peak_decel_mps2,
            measurement.modelled_distance_m,
            measurement.modelled_decel_mps2,
            measurement.optimism,
            measurement.pitch_deg,
        )

    def _log_aeb_evidence(
        self,
        bev: np.ndarray,
        heights: np.ndarray,
        geometry: VehicleGeometry,
    ) -> None:
        """
        Say WHAT the brake fired at, once per firing.

        `AEB: BRAKING` names a distance, and a distance cannot distinguish a
        wall from the road surface arriving in the height band -- which is the
        entire difference between a correct firing and a phantom. The vertical
        extent can, so it is measured and reported. See
        `planner.corridor_return_profile` for how to read it.
        """
        pending, self._pending_evidence = self._pending_evidence, []
        if not len(bev):
            return
        for state in pending:
            if state.threat_m is None:
                # The trigger fired with nothing detected at all, which is a bug
                # in the trigger rather than anything the cloud can explain.
                LOGGER.warning(
                    "AEB evidence: %s fired with no threat recorded",
                    "REAR AEB" if state.rearward else "AEB",
                )
                continue
            # The rear system reasons in a 180-degree-rotated frame, so the
            # cloud has to be handed over the same way it was scanned.
            points = mirror_points(bev) if state.rearward else bev
            profile = corridor_return_profile(
                points,
                heights,
                geometry.ground_z_vehicle,
                state.curvature,
                state.corridor_half_width_m,
                state.threat_m,
            )
            LOGGER.info(
                "AEB evidence: %s at %.1f m | %d returns spanning %.2f m of "
                "height over %.2f m of range (min %.2f, median %.2f, max %.2f "
                "above the ego plane, floor %.2f) | measured ground rise "
                "%.2f m, clamped to %.2f | corridor %.2f m half-width",
                "REAR AEB" if state.rearward else "AEB",
                state.threat_m,
                int(profile["count"]),
                profile["spread_m"],
                profile["range_span_m"],
                profile["height_min_m"],
                profile["height_median_m"],
                profile["height_max_m"],
                AEB_OBSTACLE_MIN_HEIGHT_M,
                profile["ground_rise_m"],
                # The clamp geometric_obstacle_sets applies: the cone bounds the
                # measured rise rather than replacing it, so a wide gap between
                # these two numbers means the estimate saw a grade the floor was
                # not allowed to believe.
                max(0.0, min(profile["ground_rise_m"], profile["cone_bound_m"])),
                state.corridor_half_width_m,
            )

    def _log_aeb(self, label: str, state: AebState, speed: float) -> None:
        """
        One line per AEB transition, and nothing in between.

        Unlike the driving telemetry this is not rate-limited-but-periodic: an
        armed AEB that never fires has nothing to say, and when it does fire the
        two lines either side of the event are the whole story.
        """
        if state.status == self._last_logged_aeb.get(label):
            return
        self._last_logged_aeb[label] = state.status
        LOGGER.info(
            "%s: %s at %.1f km/h | threat %s (brake-now %.1f, standoff %.1f, "
            "horizon %.1f) | required %.1f m/s^2 ttc %s | k %+.4f brake %.2f "
            "| %s",
            label,
            state.status,
            speed * 3.6,
            "none" if state.threat_m is None else f"{state.threat_m:.1f} m",
            state.brake_now_m,
            state.standoff_m,
            state.horizon_m,
            state.required_decel_mps2,
            "inf"
            if not np.isfinite(state.time_to_collision_s)
            else f"{state.time_to_collision_s:.2f}s",
            state.curvature,
            state.brake,
            state.reason,
        )

    def _nav_heading(self, state: dict[str, Any]) -> float | None:
        """
        Turn hint from the in-game destination, refreshed on its own cadence.

        Reading the route is a blocking Lua round-trip, and the route only
        changes when the player sets a new destination, so it must not ride the
        display loop.
        """
        now = time.monotonic()
        if now - self._last_nav_poll_at >= NAV_POLL_INTERVAL_MS / 1000.0:
            self._last_nav_poll_at = now
            self._route = fetch_route(self._run_lua)
        return route_heading(self._route, state)

    def _run_lua(self, chunk: str) -> Any:
        assert self._bng is not None
        return self._bng.control.queue_lua_command(chunk, response=True)

    def _actuate(
        self,
        plan: DrivingPlan | None,
        aeb: AebState | None,
        rear_aeb: AebState | None = None,
    ) -> None:
        """
        The single place the two features reach the car.

        Either AEB outranks the driving controller unconditionally: they zero
        the throttle and take the hardest brake demand of the three. With
        self-driving off it sends the pedals and NOTHING else -- no steering, no
        gear, no parking brake -- because a human is holding all three and this
        is a brake assist, not a takeover.

        The two AEB systems cannot both be firing: one arms only above a
        forward speed and the other only above a reverse one. Combining by max
        anyway costs nothing and means no ordering assumption to get wrong.
        """
        if self._vehicle is None:
            return
        firing = [
            state.brake
            for state in (aeb, rear_aeb)
            if state is not None and state.engaged
        ]
        aeb_brake = max(firing) if firing else 0.0
        rate_limited = True
        if plan is not None:
            command: ControlCommand = plan.command
            controls: dict[str, float] = {
                "steering": command.steering,
                "throttle": 0.0 if aeb_brake > 0.0 else command.throttle,
                "brake": max(command.brake, aeb_brake),
                # Sent explicitly on every message. beamngpy omits None
                # arguments and BeamNG's submitInput no-ops on absent keys, so a
                # parking brake we never mention is a parking brake we never
                # release -- and the game's own vehicle spawner clears it by
                # hand for this reason.
                "parkingbrake": 0.0,
            }
            # Only when it actually needs to change: shiftToGearIndex has side
            # effects, and this loop runs 25 times a second.
            if not gear_is_engaged(command.gear, plan.reported_gear):
                controls["gear"] = command.gear
        elif aeb_brake > 0.0:
            controls = {"throttle": 0.0, "brake": aeb_brake}
        elif self._aeb_brake_sent:
            # The event ended. This release must never be dropped by the rate
            # gate -- one swallowed message leaves the car braking indefinitely.
            controls = {"throttle": 0.0, "brake": 0.0}
            rate_limited = False
        else:
            return

        now = time.perf_counter()
        if (
            rate_limited
            and (now - self._last_control_at) * 1000.0 < CONTROL_INTERVAL_MS - 1.0
        ):
            return
        self._last_control_at = now
        self._aeb_brake_sent = aeb_brake > 0.0
        try:
            self._vehicle.control(**controls)
            self._last_control_ms = (time.perf_counter() - now) * 1000.0
        except Exception as exc:
            # A genuine bridge loss also fails the next state poll, and the
            # existing grace budget handles that. Here, just stop driving.
            LOGGER.exception("Actuation failed")
            self._disengage_self_driving(f"Self-driving stopped: {exc}")
            self._disengage_aeb(f"Emergency braking stopped: {exc}")

    def _sensor_reach(
        self, index: int, points: np.ndarray, origin: np.ndarray
    ) -> tuple[str, int, float]:
        """``(name, return count, furthest horizontal range)`` for one unit."""
        name = (
            self._sensor_names[index]
            if index < len(self._sensor_names)
            else str(index)
        )
        offsets = points[:, :2] - origin[:2].astype(np.float32)
        ranges = np.hypot(offsets[:, 0], offsets[:, 1])
        finite = ranges[np.isfinite(ranges)]
        return name, len(points), float(finite.max()) if len(finite) else 0.0

    def _sensor_set_is_complete(self) -> bool:
        """
        Whether a usable sensor set is attached.

        Deliberately a presence check and NOT a count. This gates the display
        loop and both control systems, and as a literal `!= 4` it silently
        blanked all three the moment a fifth mount was added -- `_poll_once`
        returned before its first statement, so no frame was ever emitted, no
        poll ever failed, and nothing logged. The count was never load-bearing
        either: `attach_to_player` wraps the whole build loop, and every failure
        funnels through `_cleanup_sensors`, so this list is either empty or the
        full set. A partial set cannot persist for it to catch.
        """
        return bool(self._sensors)

    @staticmethod
    def _bbox_bottom(vehicle: Vehicle) -> float | None:
        """The vehicle ground plane, which every mount height is measured from."""
        try:
            return min(float(vec3(p)[2]) for p in vehicle.get_bbox().values())
        except Exception:
            LOGGER.debug("Could not read the vehicle bounding box", exc_info=True)
            return None

    @staticmethod
    def _verify_mount_height(
        sensor: Lidar, bbox_z: float | None, mount: SensorMount
    ) -> None:
        """
        Confirm the simulator placed each sensor where we asked.

        Sensor `pos` is referenced to the vehicle ground plane; adding the bbox
        bottom on top used to bury the sensors underground, which silently
        killed every downward ray and collapsed the horizontal sweep. This turns
        that whole class of regression into one log line.

        Checked per mount rather than once, because the mounts no longer share a
        height: the roof unit is derived from the bounding box, so it is exactly
        the one a bad bbox would misplace, and its whole value is the height.
        """
        if bbox_z is None:
            return
        wanted = float(mount.position_vehicle[2])
        try:
            delta = float(sensor.get_position()[2]) - bbox_z
        except Exception:
            LOGGER.debug("Could not verify sensor mount height", exc_info=True)
            return
        LOGGER.info(
            "Mount check: %s sits %.3f m above the bbox bottom (want %.2f)",
            mount.name,
            delta,
            wanted,
        )
        if abs(delta - wanted) > 0.15:
            LOGGER.warning(
                "%s mount height is off by %.3f m. Below ground means no "
                "downward rays and a collapsed horizontal sweep; a low roof "
                "unit loses the ground-ring spacing it exists for.",
                mount.name,
                delta - wanted,
            )

    def _get_vehicle_state(self) -> dict[str, Any]:
        assert self._vehicle is not None
        # The per-vehicle state sensor works in both free-roam and scenarios.
        # VehiclesApi.get_states() uses ScenarioUpdate and is rejected by
        # BeamNG.tech when the user loaded a map/car through free-roam.
        #
        # Electrics rides along only while driving: Sensors.poll batches every
        # named sensor into one request, so the gearbox reading is free, but
        # there is no reason to ask for it in the plain viewer.
        if self._self_driving:
            self._vehicle.poll_sensors("state", "electrics")
        else:
            self._vehicle.poll_sensors("state")
        state = dict(self._vehicle.state)
        if not state:
            raise RuntimeError("The player vehicle has no active simulation state")
        return state

    def _take_vehicle_state(self) -> dict[str, Any]:
        """
        Collect the state the previous tick prefetched, or poll one fresh.

        The prefetched state finished its round-trip up to a tick ago, so the
        position is advanced by `vel * age` -- a linear correction that restores
        the pose-to-cloud alignment a synchronous poll had, leaving only the
        acceleration term (about a centimetre at full braking). The heading is
        left alone: over these ages it moves under a degree even at full lock.
        """
        future, self._state_future = self._state_future, None
        if future is None:
            return self._get_vehicle_state()
        try:
            state, done_at = future.result()
        except Exception:
            LOGGER.debug("Prefetched state poll failed; re-polling", exc_info=True)
            return self._get_vehicle_state()
        age = time.perf_counter() - done_at
        if age > _STATE_PREFETCH_MAX_AGE_S:
            # From before an app stall; extrapolating across it would be worse
            # than the round-trip it saves.
            return self._get_vehicle_state()
        if age > 0.0 and "pos" in state:
            position = vec3(state["pos"]) + vec3(
                state.get("vel", (0.0, 0.0, 0.0))
            ) * age
            state["pos"] = tuple(float(value) for value in position)
        return state

    def _prefetch_vehicle_state(self) -> None:
        """
        Start the NEXT tick's state poll, after this tick's last socket use.

        Runs on `_state_pool`'s single thread while the worker thread is idle
        between timer fires. The worker touches the vehicle socket again only
        after `_take_vehicle_state` has joined this future, so the connection
        is never used from two threads at once.
        """
        if self._vehicle is None or self._state_future is not None:
            return

        def request() -> tuple[dict[str, Any], float]:
            state = self._get_vehicle_state()
            return state, time.perf_counter()

        try:
            self._state_future = self._state_pool.submit(request)
        except RuntimeError:
            # The pool is shut down; the next tick simply polls synchronously.
            self._state_future = None

    def _drop_state_future(self) -> None:
        """
        Abandon any in-flight prefetch before teardown touches the socket.

        A brief bounded wait, never `result()`: the socket has no timeout, so a
        simulator stuck mid-request would otherwise hang every teardown path.
        If it does not finish in time the teardown proceeds anyway -- at that
        point the bridge is almost certainly gone and the disconnect below
        will error the request out of its recv.
        """
        future, self._state_future = self._state_future, None
        if future is not None and not future.cancel():
            futures_wait((future,), timeout=1.0)

    @staticmethod
    def _attach_electrics(vehicle: Vehicle) -> None:
        """
        Attach the Electrics sensor, which is how the gearbox is read.

        Best effort: without it `_reported_gear` returns None and the controller
        falls back to the automatic forward index, which still drives. Re-attach
        raises BNGValueError on a duplicate name, and that is fine too.
        """
        try:
            from beamngpy.sensors import Electrics

            vehicle.attach_sensor("electrics", Electrics())
        except Exception:
            LOGGER.debug("Could not attach the electrics sensor", exc_info=True)

    def _reported_gear(self) -> Any:
        """
        What the gearbox says it is in, or None when that cannot be read.

        Automatic-family boxes report a mode string ("P"/"R"/"N"/"D"/"S3"/"M2"),
        manual and sequential boxes a numeric gear index. `forward_gear_index`
        turns that into the right index to command; None is survivable and
        falls back to the automatic index.
        """
        if self._vehicle is None:
            return None
        try:
            return self._vehicle.sensors["electrics"]["gear"]
        except Exception:
            LOGGER.debug("Could not read the gearbox state", exc_info=True)
            return None

    def _load_annotations(self) -> dict[str, Any]:
        assert self._bng is not None
        try:
            return self._bng.camera.get_annotations()
        except Exception:
            annotation_file = Path(BEAMNG_HOME, "tech", "annotations.json")
            LOGGER.warning(
                "Could not query annotations; loading %s",
                annotation_file,
                exc_info=True,
            )
        try:
            with annotation_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise RuntimeError(
                "Could not read BeamNG's semantic annotation palette, from the "
                f"simulator or from {annotation_file}: {exc}"
            ) from exc

    @staticmethod
    def _coerce_points(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=np.float32)
        points = np.asarray(value, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return points.reshape((-1, 3))

    @staticmethod
    def _coerce_colours(value: Any, point_count: int) -> np.ndarray:
        colours = (
            np.asarray(value, dtype=np.uint8) if value is not None else np.empty(0)
        )
        available = int(colours.size) // 3
        if available >= point_count:
            return colours.reshape(-1)[: point_count * 3].reshape((-1, 3))
        # A short colour buffer used to discard the whole sensor's semantics and
        # paint every one of its points as unknown. Keep what did arrive.
        if available > 0:
            LOGGER.warning(
                "Semantic colours short by %d points; padding the remainder",
                point_count - available,
            )
            padded = np.tile(UNKNOWN_SEMANTIC_RGB, (point_count, 1))
            padded[:available] = colours.reshape(-1)[: available * 3].reshape((-1, 3))
            return padded
        return np.tile(UNKNOWN_SEMANTIC_RGB, (point_count, 1))

    @staticmethod
    def _limit_points(points: np.ndarray, limit: int) -> np.ndarray:
        if len(points) <= limit:
            return points.astype(np.float32, copy=False)
        # Evenly spaced indices rather than points[::ceil(n/limit)], whose
        # integer stride halves the output the moment n exceeds the limit by one.
        index = np.linspace(0, len(points) - 1, limit).astype(np.intp)
        return points[index].astype(np.float32, copy=False)

    def _calculate_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    def _refresh_actor_registry(self, now: float) -> None:
        if (
            self._bng is None
            or now - self._last_actor_registry_at
            < WORLD_ACTOR_REGISTRY_INTERVAL_S
        ):
            return
        self._last_actor_registry_at = now
        try:
            info = self._bng.vehicles.get_current_info(include_config=False)
            self._actor_registry = {
                str(actor_id): self._actor_visual_type(actor_info)
                for actor_id, actor_info in info.items()
                if str(actor_id) != self._player_vid
            }
        except Exception:
            LOGGER.warning("Actor registry unavailable", exc_info=True)
            self._actor_registry = {}

    def _poll_actor_observations(
        self, now: float
    ) -> tuple[ActorObservation, ...]:
        if self._bng is None:
            return ()
        self._refresh_actor_registry(now)
        if (
            now - self._last_actor_state_at < WORLD_ACTOR_STATE_INTERVAL_S
        ):
            return self._actor_observations
        self._last_actor_state_at = now
        if not self._actor_registry:
            self._actor_observations = ()
            return ()

        actor_ids = tuple(self._actor_registry)
        try:
            states = self._bng.vehicles.get_states(actor_ids)
            observations: list[ActorObservation] = []
            for actor_id in actor_ids:
                state = states.get(actor_id)
                if state is None:
                    continue
                kind, dimensions = self._actor_registry[actor_id]
                observations.append(
                    ActorObservation(
                        actor_id=actor_id,
                        kind=kind,
                        pos_world=tuple(
                            float(value) for value in vec3(state["pos"])
                        ),
                        dir_world=tuple(
                            float(value) for value in vec3(state["dir"])
                        ),
                        velocity_world=tuple(
                            float(value)
                            for value in vec3(
                                state.get("vel", (0.0, 0.0, 0.0))
                            )
                        ),
                        dimensions_m=dimensions,
                    )
                )
            self._actor_observations = tuple(observations)
            self._last_actor_success_at = now
        except Exception:
            LOGGER.warning("Actor pose enrichment unavailable", exc_info=True)
            if now - self._last_actor_success_at > WORLD_ACTOR_FADE_S:
                self._actor_observations = ()
        return self._actor_observations

    @staticmethod
    def _actor_visual_type(
        info: dict[str, Any],
    ) -> tuple[str, tuple[float, float, float]]:
        description = " ".join(
            str(info.get(key, "")) for key in ("model", "name", "config")
        ).lower()
        if any(label in description for label in ("bus", "semi", "truck")):
            return "truck", (2.6, 3.2, 9.0)
        if any(
            label in description
            for label in ("pickup", "roamer", "suv", "van")
        ):
            return "utility", (2.1, 1.9, 5.1)
        if description:
            return "car", (1.9, 1.5, 4.5)
        return "unknown", (2.0, 1.7, 4.7)

    def _clear_actor_cache(self) -> None:
        self._player_vid = ""
        self._actor_registry.clear()
        self._actor_observations = ()
        self._last_actor_registry_at = -float("inf")
        self._last_actor_state_at = -float("inf")
        self._last_actor_success_at = -float("inf")

    def _cleanup_sensors(self) -> None:
        # The single funnel every teardown path goes through -- stop_sensors,
        # handle_bridge_lost, shutdown, the poll-failure branch and re-attach --
        # and it still holds a live vehicle handle here, so this is the one
        # place that guarantees the car is never left driving itself, nor left
        # standing on a brake this app applied.
        #
        # The prefetch is dropped FIRST: everything below this line talks on
        # the same sockets the prefetch thread may still be using.
        self._drop_state_future()
        self._disengage_aeb("Sensors stopped", announce=False)
        self._disengage_self_driving("Sensors stopped", announce=False)
        for sensor in reversed(self._sensors):
            try:
                sensor.remove()
            except Exception:
                LOGGER.debug("Could not remove LiDAR sensor", exc_info=True)
        self._sensors.clear()
        self._sensor_names.clear()
        self._geometry = None
        self._mirrored_geometry = None
        self._logged_reach = False
        self._logged_markings = False
        self._logged_marking_silence = False
        self._marking_free_scans = 0
        self._marking_names = {}
        self._logged_colour_probe = False
        self._palette = None
        # A stop in progress when the sensors go is not a stop that was
        # measured, and the next attach may well be a different car.
        self._aeb_events.clear()
        self._manual_event = None
        self._manual_prev_speed = 0.0
        self._last_tick_at = None
        self._pending_evidence = []
        self._vehicle_model = ""
        self._clear_actor_cache()

        if self._vehicle is not None:
            try:
                if self._vehicle.is_connected():
                    self._vehicle.disconnect()
            except Exception:
                LOGGER.debug("Could not disconnect player vehicle", exc_info=True)
            self._vehicle = None

    def _emit_fatal(self, message: str) -> None:
        self.status_changed.emit("ERROR", message.splitlines()[0])
        self.fatal_error.emit(message)
