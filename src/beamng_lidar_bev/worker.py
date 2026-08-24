from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import replace
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
    AEB_MIN_SPEED_MPS,
    AEB_MIN_VERTICAL_EXTENT_M,
    AEB_OBSTACLE_MIN_HEIGHT_M,
    AEB_TRIGGER_MARGIN,
    BEAMNG_EXE,
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    CAMERA_FRAME_STAGING_S,
    CAMERA_NEAR_FAR_PLANES,
    CAMERA_UPDATE_TIME_S,
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
    MEMORY_ROAD_STRIDE,
    NAV_POLL_INTERVAL_MS,
    OBSTACLE_MIN_HEIGHT_M,
    PARKING_BAY_MAX_DEPTH_M,
    PARKING_BAY_MIN_DEPTH_M,
    PARKING_BAY_WIDTH_MAX_M,
    PARKING_BAY_WIDTH_MIN_M,
    PARKING_DRIVE_SPEED_MPS,
    PARKING_OCCUPANCY_MIN_HEIGHT_M,
    PARKING_SCAN_INTERVAL_S,
    PARKING_SCAN_RADIUS_M,
    PLANNER_HORIZON_M,
    PLANT_REFERENCE_VEHICLE,
    REVERSE_COST_SMOOTHNESS,
    REVERSE_DISTANCE_M,
    REVERSE_REQUIRED_FREE_M,
    ROAD_BONUS_CELL_M,
    ROAD_BONUS_HALF_WIDTH_M,
    ROAD_BONUS_MIN_CELLS,
    ROAD_BONUS_REACH_M,
    ROUTE_ARRIVAL_LATCH_M,
    ROUTE_PREVIEW_M,
    ROUTE_STALE_GRACE_S,
    STALL_SPEED_MPS,
    STEERING_SIGN,
    TRANSITION_DISTANCES_M,
    VISION_DRIVING_ENABLED,
    WORLD_ACTOR_FADE_S,
    WORLD_ACTOR_REGISTRY_INTERVAL_S,
    WORLD_ACTOR_RETRY_S,
    WORLD_ACTOR_STATE_INTERVAL_S,
)
from .controller import (
    BLOCKED,
    REVERSE_GEAR,
    REVERSING,
    DrivingController,
    forward_gear_index,
    gear_is_engaged,
)
from .geometry import (
    derive_camera_rig,
    derive_vehicle_geometry,
    outside_ego_body,
    rotate_about_up,
    vec3,
    vehicle_axes,
    world_points_to_bev,
)
from .hybrid_astar import Occupancy
from .launcher import (
    bridge_is_reachable,
    build_launch_command,
    capture_setting_warnings,
    start_beamng_process,
)
from .models import (
    BRAKING,
    ActorObservation,
    AebState,
    ArcPlan,
    BevFrame,
    CameraImage,
    CameraMount,
    ControlCommand,
    DrivingPlan,
    ParkingJob,
    ParkingSlot,
    PerceptionSnapshot,
    RoadGrid,
    SensorMount,
    VehicleGeometry,
    VisionFrame,
)
from .navigation import fetch_route_reply, parse_route, route_heading
from .parking import (
    MarkingMemory,
    ParkingBay,
    ScanReport,
    find_bays,
    match_selection,
    project_bays,
    remember_bays,
)
from .parking_drive import (
    PARK_APPROACH,
    PARK_ARRIVED,
    PARK_BACKING,
    PARK_BLOCKED,
    PARK_SECURING,
    PARK_SHIFTING,
    ParkingDriver,
    ParkingDriveState,
)
from .parking_map import ParkingMap
from .planner import (
    ObstacleBand,
    corridor_return_profile,
    geometric_obstacle_sets,
    plan_arc,
    rear_free_distance,
)
from .planning_map import PlanningMemory
from .route_model import build_route_path
from .semantics import (
    SCENE_ROAD,
    SCENE_VEHICLE,
    SURFACE_MARKING,
    SemanticPalette,
    classify_scene_groups,
    classify_surface_materials,
    pack_rgb,
    pack_rgb_rows,
)
from .unprojection import (
    CameraRays,
    build_rig_rays,
    pose_from_state,
    unproject_frame,
)

LOGGER = logging.getLogger(__name__)

# Which instrument set attach_to_player builds. The worker owns this the way it
# owns the driving toggles: the GUI requests, the worker confirms via
# sensor_mode_changed, and the mode only takes physical effect at attach (a
# switch mid-stream re-attaches through the same single funnel).
SENSOR_MODE_LIDAR = "LIDAR"
SENSOR_MODE_VISION = "VISION"

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
# Time constant of the yaw-rate estimate the prefetched heading and the
# camera frames are advanced with; short, because the thing it guards against
# is a tick that stretched, and a long filter would lag exactly then.
_YAW_RATE_TAU_S = 0.08
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
# Cadence of the Memory check: line -- store sizes change slowly, and the line
# exists to catch a runaway store or a map full of ghosts, not to trace it.
_MEMORY_LOG_INTERVAL_S = 5.0
# How long Vision mode waits for a first genuinely new camera frame before
# warning. The known silent failure it exists for: requested_update_time=0.0
# leaves every streaming buffer zero-filled while the read loop spins happily
# -- a working rig producing black frames, which cost a full benchmark round
# to diagnose live.
_VISION_SILENCE_WARN_S = 5.0
# Stride for the per-camera freshness digest, over the LATTICE indices the
# unprojection gathers anyway. A prime, so the samples drift across the frame
# instead of landing on one pixel column. It went 4093 -> 61 with rung 0.5:
# at 4093 a small camera's digest was SEVEN depth samples, few enough to miss
# real frame changes for seconds at a stretch -- and a missed change grows the
# frame's measured age, which the pose rewind then multiplies by velocity, so
# the whole camera's cloud slid backwards down the road. ~1.4k samples for the
# largest camera now, a 6 KB gather against the 40 ms tick.
_VISION_DIGEST_STRIDE = 61
# How often a spawned-but-not-yet-connected simulator is checked for still
# being alive. One second: this runs only between Launch and the bridge
# opening, and each tick is a `poll()` plus, at most, one bounded socket probe.
_LAUNCH_WATCH_INTERVAL_MS = 1000


def build_road_grid(
    road_bev: np.ndarray, remembered_road: np.ndarray
) -> RoadGrid | None:
    """
    The road-coverage bonus's occupancy grid, or None when there is not
    enough road to say anything.

    None below ROAD_BONUS_MIN_CELLS is the unannotated-map path: the term
    vanishes from the cost exactly as nav and keep-right do when their inputs
    are absent -- dropped, never guessed.
    """
    parts = [points for points in (road_bev, remembered_road) if len(points)]
    if not parts:
        return None
    points = parts[0] if len(parts) == 1 else np.concatenate(parts)
    width = int(round(2.0 * ROAD_BONUS_HALF_WIDTH_M / ROAD_BONUS_CELL_M))
    height = int(round(ROAD_BONUS_REACH_M / ROAD_BONUS_CELL_M))
    cols = np.floor(
        (points[:, 0] + ROAD_BONUS_HALF_WIDTH_M) / ROAD_BONUS_CELL_M
    ).astype(np.intp)
    rows = np.floor(points[:, 1] / ROAD_BONUS_CELL_M).astype(np.intp)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not inside.any():
        return None
    counts = np.bincount(
        rows[inside] * width + cols[inside], minlength=width * height
    )
    occupancy = (counts > 0).astype(np.uint8).reshape(height, width)
    if int(occupancy.sum()) < ROAD_BONUS_MIN_CELLS:
        return None
    return RoadGrid(
        occupancy=occupancy,
        cell_m=ROAD_BONUS_CELL_M,
        origin_right_m=-ROAD_BONUS_HALF_WIDTH_M,
        origin_forward_m=0.0,
    )

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
    vision_frame_ready = pyqtSignal(object)
    sensor_mode_changed = pyqtSignal(str)
    self_driving_changed = pyqtSignal(bool)
    aeb_changed = pyqtSignal(bool)
    rear_aeb_changed = pyqtSignal(bool)
    parking_changed = pyqtSignal(bool)
    parking_drive_changed = pyqtSignal(bool)
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
        # Vision mode. The digests detect genuinely NEW camera frames: the
        # display tick re-reads the shared buffers faster than the cameras
        # update, and counting re-reads would report the tick rate as the
        # acquisition rate.
        self._sensor_mode = SENSOR_MODE_LIDAR
        self._camera_digests: dict[str, bytes] = {}
        self._vision_streaming_since: float | None = None
        self._logged_vision_check = False
        self._logged_vision_silence = False
        # Rung 0.5: the per-camera ray tables, built once at attach, and when
        # each camera's depth buffer was last seen to CHANGE -- the only part
        # of a frame's age the worker can measure, since the simulator stamps
        # nothing. The eye height is the tallest camera's, for porosity.
        self._camera_rays: dict[str, CameraRays] = {}
        self._camera_frame_seen: dict[str, float] = {}
        self._camera_frame_checked: dict[str, float] = {}
        self._vision_eye_height_m = 0.0
        self._logged_unprojection = False
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
        self._actor_refused_at = -float("inf")
        self._logged_actor_refusal = False
        # Yaw rate measured from successive polled states, for advancing the
        # prefetched heading and rewinding camera frames.
        self._yaw_observation: tuple[float, float] | None = None
        self._yaw_rate_rps = 0.0

        self._self_driving = False
        self._controller: DrivingController | None = None
        self._route = None
        self._route_fresh_at = 0.0
        self._last_nav_poll_at = 0.0
        self._last_nav_rtt_ms = 0.0
        self._route_check_logged = False
        self._arrival_logged = False
        self._arrived_hold = False
        self._reverse_check_logged = False
        self._last_route_path = None
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
        # Planner-only obstacle/road memory, worker-thread confined. AEB
        # never reads it -- see planning_map's module docstring.
        self._memory = PlanningMemory()
        self._last_memory_log_at = 0.0

        # --- Parking bay scan, display and selection only -------------------
        #
        # Independent of self-driving on purpose: finding a bay is something
        # you do while driving the car yourself. Nothing here reaches the
        # planner or either AEB band, and the bays are held in WORLD so they
        # stay put between the scans that rebuild them.
        self._parking_scan = False
        self._marking_memory = MarkingMemory()
        self._parking_bays: tuple[ParkingBay, ...] = ()
        self._parking_selected: tuple[float, float] | None = None
        self._parking_job: ParkingJob | None = None
        self._last_parking_scan_at = 0.0
        self._logged_parking_check = False
        self._parking_seen_at: dict[tuple[int, int], float] = {}
        self._last_parking_report: ScanReport | None = None
        self._last_parking_log_at = 0.0
        self._last_parking_bay_count = -1
        # The MANOEUVRE, separate from the scan: you arm the scan to look, and
        # engage this to go. Mutually exclusive with self-driving, because one
        # thing steers the car at a time -- and the arc planner would fight a
        # committed manoeuvre for exactly the reasons parking_drive exists.
        self._parking_driving = False
        self._parking_driver = ParkingDriver()
        self._parking_map = ParkingMap()
        self._last_park_state: ParkingDriveState | None = None
        self._logged_park_phase = ""
        self._last_shift_log_at = 0.0
        self._last_drive_block_ms = 0.0
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

        # A spawned simulator that DIED is indistinguishable, from here, from
        # one merely slow to open its bridge -- and the window waits
        # _BRIDGE_WAIT_GRACE_S (300 s) for the latter, with Launch disabled
        # throughout. That is how a launcher aborting 0.75 s in presented as
        # five silent minutes of "BeamNG.tech is starting": `Popen` had already
        # returned a pid, so nothing ever asked whether the process was still
        # there. This asks, once a second, until the bridge opens or it is gone.
        self._launch_watch = QTimer(self)
        self._launch_watch.setInterval(_LAUNCH_WATCH_INTERVAL_MS)
        self._launch_watch.timeout.connect(self._watch_launch)

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
        self._launch_watch.start()
        self.status_changed.emit(
            "LAUNCHED", "BeamNG.tech launched; load a map and player vehicle"
        )
        self.launch_ready.emit()

    @pyqtSlot()
    def _watch_launch(self) -> None:
        """
        Report a spawned simulator that dies before its bridge ever opens.

        The exit code is the whole value of this: 0xC0000409 named the invalid
        std handles that `start_beamng_process` now passes explicitly, and no
        other signal in the app distinguished that from a slow boot.
        """
        process = self._beamng_process
        if process is None:
            self._launch_watch.stop()
            return

        code = process.poll()
        if code is None:
            # Still booting. The bridge opening is what ends the watch, not the
            # process merely surviving -- the engine outlives the launcher stub.
            if bridge_is_reachable():
                self._launch_watch.stop()
            return

        self._launch_watch.stop()
        self._beamng_process = None
        if code == 0 or bridge_is_reachable():
            # A launcher stub that hands off to the engine (or to a session
            # already running) exits 0 with everything working. Not a failure.
            LOGGER.info("BeamNG.tech launcher exited with code %s", code)
            return

        LOGGER.error(
            "BeamNG.tech exited with code %s (0x%08X) before opening its bridge",
            code,
            code & 0xFFFFFFFF,
        )
        self._emit_fatal(
            "BeamNG.tech closed before opening its bridge "
            f"(exit code {code}, 0x{code & 0xFFFFFFFF:08X}).\n"
            "Its own log is beamng-launcher.log in the BeamNG user folder; a "
            "zero-byte one means the launcher died before writing a line."
        )

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
                # beamngpy 1.36 SILENTLY drops vehicles whose ids fail its
                # object-name validation (the reserved id "vehicle", a leading
                # digit, a '/', a leading '%'), so a perfectly real player car
                # can be missing from get_current. Ask the raw registry to
                # tell "dropped by validation" apart from "not there at all",
                # because the two need opposite advice.
                try:
                    known_ids = self._bng.vehicles.get_current_info(
                        include_config=False
                    )
                except Exception:
                    known_ids = {}
                if player_vid in known_ids:
                    raise RuntimeError(
                        f"Player vehicle id {player_vid!r} is rejected by "
                        "beamngpy's object-name validation (reserved name, "
                        "leading digit, '/' or leading '%'). Rename the "
                        "vehicle in the simulator and retry."
                    )
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

            sensor_prefix = f"bev_{os.getpid()}_{int(time.monotonic() * 1000)}"
            self._sensors = []
            self._sensor_names = []

            # Both instrument sets produce the same annotated cloud now: the
            # camera rig renders the engine's annotation channel and the
            # palette matches it exactly as it matches the LiDAR's colours.
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

            if self._sensor_mode == SENSOR_MODE_VISION:
                # Rung 0.5 of the vision ladder: every camera renders depth
                # and annotation beside colour, and the tick unprojects them
                # into the same cloud the LiDAR set produces.
                mount_count = self._attach_camera_rig(
                    vehicle, geometry, sensor_prefix
                )
                self._camera_digests = {}
                self._camera_frame_seen = {}
                self._vision_streaming_since = time.perf_counter()
                self._logged_vision_check = False
                self._logged_vision_silence = False
                self._logged_unprojection = False
            else:
                mount_count = len(geometry.mounts)
            bbox_z = self._bbox_bottom(vehicle)
            lidar_mounts = (
                ()
                if self._sensor_mode == SENSOR_MODE_VISION
                else tuple(geometry.mounts.values())
            )
            for index, mount in enumerate(lidar_mounts):
                self.status_changed.emit(
                    "ATTACHING",
                    f"Attaching {mount.name} LiDAR ({index + 1}/{mount_count})",
                )
                sensor = Lidar(
                    f"{sensor_prefix}_{mount.name}",
                    self._bng,
                    vehicle,
                    **self.lidar_sensor_kwargs(mount),
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
            LOGGER.exception("Could not attach sensors")
            self._cleanup_sensors()
            self.status_changed.emit("READY", f"Sensor attach failed: {exc}")
            self.fatal_error.emit(
                f"Could not attach to the player vehicle: {exc}\n\n"
                "Make sure a map is loaded and the intended EGO vehicle is selected."
            )
            return

        vision = self._sensor_mode == SENSOR_MODE_VISION
        self.status_changed.emit(
            "STREAMING",
            f"{mount_count} {'cameras' if vision else 'LiDAR sensors'} active "
            f"on {player_vid}",
        )
        # Before sensors_ready, so the GUI knows which instrument set it is
        # enabling controls for when that signal lands.
        self.sensor_mode_changed.emit(self._sensor_mode)
        self.sensors_ready.emit(player_vid, geometry)
        self._poll_timer.start()
        # Both emergency brakes arm themselves. They are protective rather than
        # behavioural -- neither steers, and neither touches a pedal until a
        # collision is otherwise unavoidable -- so defaulting them off would
        # mean the safety net is missing exactly when nobody thought about it.
        # Self-driving stays opt-in, because that one changes what the car does.
        #
        # In Vision mode the same two calls run and `_set_aeb` refuses them
        # until VISION_DRIVING_ENABLED: the camera lattice is a new sampling
        # distribution and the phantom-braking checklist has not been re-run
        # on it. One arming path, one refusal, rather than a second branch
        # here that could drift from the slot's own rule.
        self._set_aeb(True, rearward=False)
        self._set_aeb(True, rearward=True)

    @pyqtSlot(str)
    def set_sensor_mode(self, mode: str) -> None:
        """
        Choose which instrument set the next attach builds.

        Owned by the worker like the driving toggles: the GUI requests, this
        confirms via sensor_mode_changed. A switch while sensors are live
        re-attaches immediately through the one funnel `attach_to_player`
        already is -- it stops the timer and pushes the old set through
        `_cleanup_sensors`, so a half-swapped rig cannot exist.
        """
        mode = str(mode).upper()
        if mode not in (SENSOR_MODE_LIDAR, SENSOR_MODE_VISION):
            LOGGER.warning("Ignoring unknown sensor mode %r", mode)
            return
        if mode == self._sensor_mode:
            return
        self._sensor_mode = mode
        LOGGER.info("Sensor mode set to %s", mode)
        self.sensor_mode_changed.emit(mode)
        if self._sensors:
            self.attach_to_player()

    @pyqtSlot(bool)
    def set_self_driving(self, enabled: bool) -> None:
        if not enabled:
            self._disengage_self_driving("Self-driving disengaged")
            return
        if self._vision_refuses_driving():
            # The camera cloud exists now (rung 0.5), but the planner's band
            # was fitted to LiDAR sampling and has not been re-proved on it
            # live. VISION_DRIVING_ENABLED is the gate.
            self.self_driving_changed.emit(False)
            self.status_changed.emit(
                "STREAMING" if self._sensors else "READY",
                "Self-driving needs the LiDAR set; switch out of Vision mode",
            )
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
        self._route_fresh_at = 0.0
        self._last_nav_poll_at = 0.0
        self._route_check_logged = False
        self._arrival_logged = False
        self._arrived_hold = False
        self._reverse_check_logged = False
        self._last_route_path = None
        self._memory.clear()
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
        self._route_fresh_at = 0.0
        self._arrived_hold = False
        self._last_route_path = None
        self._memory.clear()
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
    def set_parking_scan(self, enabled: bool) -> None:
        """
        Arm or disarm the parking bay scan. Display and selection only.

        Deliberately NOT gated on self-driving: scanning is what you do while
        driving the car yourself, looking for somewhere to put it. Disarming
        drops the accumulated paint as well as the bays, so re-arming starts
        from what the sensors can see now rather than from a stale lot.
        """
        if enabled and self._vision_refuses_driving():
            # The bay scan reads the SEMANTIC marking store. The camera rig
            # fills it now (the annotation channel labels decals exactly as
            # the LiDAR's does), but the parking manoeuvre drives on the
            # planner's band, which is behind the same live gate.
            self.parking_changed.emit(False)
            self.status_changed.emit(
                "STREAMING" if self._sensors else "READY",
                "Parking needs the LiDAR set; switch out of Vision mode",
            )
            return
        if enabled and not self._sensor_set_is_complete():
            self.parking_changed.emit(False)
            self.status_changed.emit(
                "READY", "Attach to a vehicle before scanning for parking"
            )
            return
        self._parking_scan = enabled
        if not enabled:
            self._marking_memory.clear()
            self._parking_bays = ()
            self._parking_selected = None
            self._logged_parking_check = False
        self.parking_changed.emit(enabled)
        if enabled:
            LOGGER.info(
                "Parking check: scanning for bays from road paint within "
                "%.0f m, %.1f-%.1f m wide and %.1f-%.1f m deep. Detection "
                "and selection only -- nothing here steers, brakes or "
                "reaches the planner.",
                PARKING_SCAN_RADIUS_M,
                PARKING_BAY_WIDTH_MIN_M,
                PARKING_BAY_WIDTH_MAX_M,
                PARKING_BAY_MIN_DEPTH_M,
                PARKING_BAY_MAX_DEPTH_M,
            )

    @pyqtSlot(bool)
    def set_parking_drive(self, enabled: bool) -> None:
        """
        Engage or release the manoeuvre that drives into the selected bay.

        Refused without a selection, because the goal IS the selection: there
        is nothing to commit to otherwise. Self-driving is disengaged first --
        one thing steers the car at a time, and the arc planner would fight a
        committed manoeuvre for exactly the reasons parking_drive exists.
        """
        if not enabled:
            was = self._parking_driving
            self._parking_driving = False
            if self._parking_job is not None:
                self._parking_job = replace(
                    self._parking_job, status="CANCELLED"
                )
            self._parking_driver.reset()
            self._last_park_state = None
            self._logged_park_phase = ""
            if was:
                self.parking_drive_changed.emit(False)
            return
        if self._vision_refuses_driving():
            # Reached only if a bay somehow survived the mode switch; the scan
            # itself is already refused above. Say which system is missing
            # rather than "select a bay", which would be unactionable here.
            self.parking_drive_changed.emit(False)
            self.status_changed.emit(
                "STREAMING" if self._sensors else "READY",
                "Parking needs the LiDAR set; switch out of Vision mode",
            )
            return
        matched = (
            None
            if self._parking_selected is None
            else match_selection(self._parking_bays, self._parking_selected)
        )
        if matched is None or not self._sensor_set_is_complete():
            self.parking_drive_changed.emit(False)
            self.status_changed.emit(
                "READY", "Select a parking bay before parking into it"
            )
            return
        if matched.occupied:
            self._parking_job = None
            self.parking_drive_changed.emit(False)
            self.status_changed.emit(
                "READY", "The selected parking bay is occupied"
            )
            return
        if self._self_driving:
            self._disengage_self_driving(
                "Self-driving off: parking takes the wheel", announce=False
            )
        self._parking_driver.reset()
        self._parking_map.clear()
        self._parking_job = ParkingJob(matched)
        self._parking_driving = True
        self._logged_park_phase = ""
        self.parking_drive_changed.emit(True)
        LOGGER.info(
            "Park check: engaged with a latched world-space bay, creeping at "
            "%.1f km/h (below the %.1f km/h AEB arms at, so the forward "
            "brake stays in STANDBY and the manoeuvre does its own corridor "
            "check). A bay needing a shuffle is refused, not attempted.",
            PARKING_DRIVE_SPEED_MPS * 3.6,
            AEB_MIN_SPEED_MPS * 3.6,
        )

    def _disengage_parking_drive(self, reason: str) -> None:
        """Single funnel, mirroring _disengage_self_driving."""
        if not self._parking_driving:
            return
        self._parking_driving = False
        if self._parking_job is not None:
            self._parking_job = replace(self._parking_job, status="FAILED")
        self._parking_driver.reset()
        self._last_park_state = None
        if self._vehicle is not None:
            try:
                self._vehicle.control(
                    throttle=0.0, brake=0.0, steering=0.0, parkingbrake=0.0
                )
            except Exception:
                LOGGER.debug("Could not zero the controls", exc_info=True)
        LOGGER.info("Park check: disengaged -- %s", reason)
        self.parking_drive_changed.emit(False)
        self.status_changed.emit("STREAMING", f"Parking stopped: {reason}")

    def _complete_parking_drive(self) -> None:
        """End automation after success without releasing its parking brake."""
        if not self._parking_driving:
            return
        self._parking_driving = False
        if self._parking_job is not None:
            self._parking_job = replace(self._parking_job, status="SUCCEEDED")
        LOGGER.info("Park check: complete -- vehicle secured")
        self.parking_drive_changed.emit(False)
        self.status_changed.emit("STREAMING", "Parking complete; vehicle secured")

    @pyqtSlot()
    def clear_parking_selection(self) -> None:
        """Drop the held bay. Clicking off the bays is how this arrives."""
        self._parking_selected = None

    @pyqtSlot(float, float)
    def select_parking_slot(self, world_x: float, world_y: float) -> None:
        """
        Hold, or clear, the bay the user clicked.

        The WORLD centre is the identity rather than an index into the last
        scan: the set is rebuilt every PARKING_SCAN_INTERVAL_S and a
        subscript means something different afterwards. A click that matches
        no bay clears the selection, which is how clicking empty space
        deselects.
        """
        if not self._parking_scan:
            return
        target = (float(world_x), float(world_y))
        matched = match_selection(self._parking_bays, target)
        self._parking_selected = None if matched is None else matched.centre
        if matched is not None:
            LOGGER.info(
                "Parking check: selected a %.2f m x %.2f m bay, %s, %d "
                "marking cells of evidence",
                matched.width_m,
                matched.depth_m,
                "occupied" if matched.occupied else "clear",
                matched.stripe_cells,
            )

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
        if self._vision_refuses_driving():
            changed.emit(False)
            self.status_changed.emit(
                "STREAMING" if self._sensors else "READY",
                f"{label} needs the LiDAR set; switch out of Vision mode",
            )
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

    def _vision_refuses_driving(self) -> bool:
        """
        Whether the control systems are refused because the cloud is the
        camera rig's. One rule for all four slots (self-driving, both AEBs,
        parking) so they cannot drift: Vision mode AND the live gate still
        closed. See VISION_DRIVING_ENABLED.
        """
        return (
            self._sensor_mode == SENSOR_MODE_VISION and not VISION_DRIVING_ENABLED
        )

    def _porosity_sensor_height(self, geometry: VehicleGeometry) -> float:
        """
        The eye height AEB's porosity test reasons from: the ROOF unit, or in
        Vision mode the tallest camera (the windshield pair).

        It has to be the roof unit and not one of the 0.20 m mounts, because
        those sit BELOW anything worth testing -- a 0.6 m bush hides the ground
        behind it completely from 0.20 m, so from down there a bush and a wall
        are indistinguishable by construction. Only the unit above the bodywork
        can see over a short object at all.

        Zero when there is no roof mount, which disables the veto entirely --
        the conservative direction, since the test can only ever remove
        obstacles.
        """
        if self._sensor_mode == SENSOR_MODE_VISION:
            return float(self._vision_eye_height_m)
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
        self._launch_watch.stop()
        self._beamng_process = None
        self._first_failure_at = None
        self.status_changed.emit("OFFLINE", "BeamNG.tech is no longer running")
        self.sensors_stopped.emit()

    @pyqtSlot()
    def shutdown(self) -> None:
        self._poll_timer.stop()
        self._launch_watch.stop()
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
        """
        One display tick, for EITHER instrument set.

        The two sets differ only in acquisition -- the LiDAR units stream a
        cloud, the cameras stream depth and annotation images that
        `unprojection` turns into one -- and everything from the concatenated
        `points_world + colours` on is shared: the BEV split, the semantic
        pass, both bands, the plan, AEB, the frame and the snapshot. That is
        the perception waist the vision ladder is built to refill, and
        keeping it one code path is what makes "the car drives in Vision
        mode" the same car, not a second implementation of it.
        """
        if (
            self._bng is None
            or self._vehicle is None
            or self._geometry is None
            or self._palette is None
            or not self._sensor_set_is_complete()
        ):
            return

        vision = self._sensor_mode == SENSOR_MODE_VISION
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
            vision_images: list[CameraImage] = []
            if vision:
                point_chunks, colour_chunks, vision_images, fresh = (
                    self._acquire_vision_cloud(state, started)
                )
            else:
                point_chunks, colour_chunks = self._acquire_lidar_cloud(state)
                fresh = bool(point_chunks)

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

                keep = outside_ego_body(bev, geometry)
                if not keep.all():
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
            # `fresh` rather than `had_returns`: the camera buffers persist
            # between simulator frames, so a vision tick always has returns
            # and only a tick that read a genuinely NEW frame counts toward
            # ACQUISITION -- the same honesty rule the LiDAR metric follows.
            if fresh:
                self._frame_times.append(finished)

            plan: DrivingPlan | None = None
            aeb: AebState | None = None
            rear_aeb: AebState | None = None
            parking_occupancy: Occupancy | None = None
            if (
                self._self_driving
                or self._aeb_enabled
                or self._rear_aeb_enabled
                or self._parking_driving
            ):
                # Every driving step below is isolated from the poll-failure
                # budget on purpose: a planner, AEB or controller bug must not
                # masquerade as a lost bridge and tear the whole connection down
                # two seconds later. The three steps are isolated from each
                # other too, so a fault in one never switches the other off.
                #
                # Timed as a whole because the on-screen POLL TIME deliberately
                # excludes it (`finished` is captured above); the Drive: line
                # reports the previous tick's figure, the same convention as
                # control_ms.
                drive_block_started = time.perf_counter()
                obstacles = _EMPTY_BEV
                aeb_obstacles = _EMPTY_BEV
                measured = had_returns

                right_axis, forward, _ = vehicle_axes(state)
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
                                # The planner is CELL-REFERENCED like AEB, and
                                # for the same reason: the slope cone bounds
                                # the ground estimate at 1.5%/m, so on any road
                                # steeper than that the surface itself climbed
                                # into the band -- measured, a 2% grade took
                                # free distance from 35 m to 6 m and a 3% grade
                                # to exactly STOP_MARGIN_M, which is the blocked
                                # entry and then the reverse recovery. That is
                                # most of "it brakes for hills" AND most of "it
                                # keeps reversing", from one clamp.
                                #
                                # POROSITY IS ON for the planner too, which
                                # reverses a deliberate old choice. The
                                # reasoning was "the planner should steer
                                # around a bush even though AEB should not
                                # brake for one" -- but a bush is not a thing
                                # to steer around, it is a thing to ignore, and
                                # treating every roadside shrub and grass tuft
                                # as a wall is what makes the car flinch at
                                # verges and refuse gaps that are actually
                                # open. The test is the same one AEB uses and
                                # it is geometric, not semantic: an object of
                                # height a at range r hides the ground behind
                                # it for r*a/(h - a), so ground returns inside
                                # that shadow mean the rays went THROUGH.
                                #
                                # Its safety property carries over unchanged
                                # and is derived rather than imposed: a >= h
                                # makes the shadow infinite and the evidence
                                # window empty, so nothing as tall as the roof
                                # unit can ever be vetoed. Kerbs, walls, cars
                                # and people are untouched; only see-through
                                # things are dropped. It can only ever REMOVE
                                # candidates, so it cannot invent an obstacle.
                                #
                                # The extent test stays OFF -- it is provably
                                # inert at or below OBSTACLE_MIN_HEIGHT_M, and
                                # any value above that deletes kerbs, which are
                                # what keep the car on the tarmac.
                                ObstacleBand(
                                    OBSTACLE_MIN_HEIGHT_M,
                                    PLANNER_HORIZON_M,
                                    cell_referenced=True,
                                    reduce_to_cells=True,
                                    porosity=True,
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

                if self._parking_driving:
                    ego_pos = vec3(state["pos"])
                    if measured:
                        self._parking_map.update(
                            ego_pos,
                            right_axis,
                            forward,
                            obstacles,
                            bev[scene_groups == SCENE_ROAD],
                        )
                    parking_occupancy = self._parking_map.occupancy_bev(
                        ego_pos, right_axis, forward
                    )

                if self._self_driving:
                    try:
                        # The planner's cloud gains the memory; AEB's band
                        # below never does. Gated on a tick that HAS returns,
                        # so memory can never unblind the planner: an empty
                        # cloud still plans _BLIND_ARC and brakes.
                        merged = obstacles
                        road_grid = None
                        if measured:
                            ego_pos = vec3(state["pos"])
                            now_mono = time.monotonic()
                            road_bev = bev[scene_groups == SCENE_ROAD]
                            self._memory.update(
                                ego_pos,
                                right_axis,
                                forward,
                                obstacles,
                                bev[scene_groups == SCENE_VEHICLE],
                                road_bev[::MEMORY_ROAD_STRIDE],
                                now_mono,
                            )
                            remembered = self._memory.obstacles_bev(
                                ego_pos, right_axis, forward
                            )
                            if len(remembered):
                                merged = np.concatenate(
                                    (obstacles, remembered)
                                )
                            road_grid = build_road_grid(
                                road_bev,
                                self._memory.road_bev(
                                    ego_pos, right_axis, forward
                                ),
                            )
                            self._log_memory(now_mono, len(merged))
                        plan = self._compute_plan(
                            state,
                            merged,
                            geometry,
                            forward_speed,
                            heading,
                            measured,
                            road_grid=road_grid,
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

                self._last_drive_block_ms = (
                    time.perf_counter() - drive_block_started
                ) * 1000.0

            parking_slots: tuple[ParkingSlot, ...] = ()
            # The planner's band if a driving feature built one this tick,
            # otherwise nothing -- parking must never invent obstacles, and a
            # missing band means the corridor check simply finds it clear,
            # which is why the manoeuvre also creeps.
            obstacles_for_parking = obstacles if self._parking_driving else None
            if self._parking_scan and had_returns:
                # Its own handler, for the reason every driving step has one:
                # a fault here must not masquerade as a lost bridge and tear
                # the connection down. It is also the weakest claim on the
                # tick -- nothing actuates from it -- so it fails to an empty
                # overlay rather than to anything else changing.
                try:
                    parking_slots = self._scan_for_parking(
                        state, bev, heights, scene_materials, geometry
                    )
                except Exception:
                    LOGGER.exception("Parking bay scan failed")
                    self._parking_bays = ()

            # The overlay may be rebuilt or disappear; the active target may
            # not. Project the immutable job bay independently of scan state.
            if self._parking_driving and self._parking_job is not None:
                parking_slots = project_bays(
                    (self._parking_job.bay,),
                    vec3(state["pos"]),
                    right_axis,
                    forward,
                    selected_world=self._parking_job.bay.centre,
                )

            if self._parking_driving:
                # Its own handler, like every other driving step: a fault here
                # must not masquerade as a lost bridge. It produces a full
                # DrivingPlan, so `_actuate` sends it exactly as it sends the
                # road controller's -- gear handling and the AEB override
                # included, which is what keeps one actuation path.
                try:
                    plan = self._drive_into_bay(
                        state,
                        parking_slots,
                        geometry,
                        obstacles_for_parking,
                        parking_occupancy,
                        rear_aeb,
                    )
                except Exception as exc:
                    LOGGER.exception("Parking manoeuvre failed")
                    self._disengage_parking_drive(str(exc))

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
                route_points=(
                    None
                    if plan is None or self._last_route_path is None
                    else self._last_route_path.points
                ),
            )
            self._poll_failures = 0
            self._first_failure_at = None
            self.frame_ready.emit(frame)
            if vision:
                # The camera grid keeps its own frame beside the BEV one: the
                # images are what the CAMERAS view draws, and the metrics it
                # used to carry now ride on the BevFrame like everything else.
                self.vision_frame_ready.emit(
                    VisionFrame(
                        images=tuple(vision_images),
                        acquisition_fps=frame.acquisition_fps,
                        poll_ms=frame.poll_ms,
                        speed_mps=frame.speed_mps,
                    )
                )

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
                    route_world=(
                        None if plan is None else self._route_world_preview()
                    ),
                    parking_slots=parking_slots,
                    parking_path=(
                        None
                        if self._last_park_state is None
                        or self._last_park_state.path is None
                        else self._last_park_state.path.points.astype(
                            np.float32
                        )
                    ),
                )
            )
            # The last statement of a successful tick, deliberately after
            # `_actuate` and the actor poll: nothing on the worker thread will
            # touch a socket again until the next tick joins this future.
            self._prefetch_vehicle_state()
        except Exception as exc:
            self._note_poll_failure(exc, "Camera" if vision else "LiDAR")

    def _acquire_lidar_cloud(
        self, state: dict[str, Any]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Stream every LiDAR unit: parallel lists of world points and colours."""
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
                    self._sensor_names[index], colours, points
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
        return point_chunks, colour_chunks

    def _acquire_vision_cloud(
        self, state: dict[str, Any], now: float
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[CameraImage], bool]:
        """
        Stream every camera: the unprojected cloud, the colour images for the
        grid, and whether any camera delivered a genuinely new frame.

        Three channels, three different treatments. COLOUR is copied whole,
        once, because the grid paints all of it and stream_raw hands back a
        view of the live buffer the simulator keeps writing into. DEPTH and
        ANNOTATION are never copied: the ray table's lattice is gathered
        straight from the live view, one vectorised read per channel, which
        is ~1/24th of the bytes at the default strides. A torn depth read --
        the simulator landing a frame mid-gather -- mixes two frames a
        sixtieth of a second apart along one row boundary, which the
        accumulation stores absorb; copying 2.7 MB per camera per tick to
        prevent it would not fit the tick.

        Each frame's AGE is measured from when its depth lattice last changed
        (the digest), so the cloud is placed from the pose the car had then
        rather than now -- see CAMERA_FRAME_STAGING_S for the part that
        cannot be measured from here.
        """
        assert self._geometry is not None
        geometry = self._geometry
        point_chunks: list[np.ndarray] = []
        colour_chunks: list[np.ndarray] = []
        images: list[CameraImage] = []
        any_fresh = False
        reach: list[tuple[str, int, float, float]] = []
        origin = vec3(state["pos"])

        for index, camera in enumerate(self._sensors):
            name = (
                self._sensor_names[index]
                if index < len(self._sensor_names)
                else str(index)
            )
            raw = camera.stream_raw()
            width, height = camera.resolution

            colour = raw.get("colour")
            if colour is not None and len(colour) == width * height * 4:
                # Copied exactly once, here, before anything reads it twice.
                pixels = np.frombuffer(colour, dtype=np.uint8).copy()
                images.append(
                    CameraImage(
                        name=name, rgba=pixels.reshape((height, width, 4))
                    )
                )

            rays = self._camera_rays.get(name)
            if rays is None:
                # A camera with no ray table is a display-only camera (or an
                # offline stub); its colour still reaches the grid above.
                continue
            depth_raw = raw.get("depth")
            if depth_raw is None or len(depth_raw) != width * height * 4:
                continue
            # Fresh-frame detection on the depth lattice itself: a strided
            # gather the unprojection needs anyway, so the digest is free.
            depth_digest = bytes(
                np.frombuffer(depth_raw, dtype=np.float32)[
                    rays.pixel_index[::_VISION_DIGEST_STRIDE]
                ]
            )
            checked = self._camera_frame_checked.get(name)
            if depth_digest != self._camera_digests.get(name):
                self._camera_digests[name] = depth_digest
                # The buffer changed somewhere between the LAST look and this
                # one, so the change time's best estimate is the MIDPOINT of
                # the two. Stamping `now` instead under-ages every frame by
                # half a tick on average (~20 ms -- 0.22 m of forward
                # misplacement at the 40 km/h cap), which the 2026-08-24
                # fence-run regression measured live as +32 +/- 17 ms per
                # unit speed against a ~17-20 ms detection-latency
                # prediction. Centring zeroes the mean and halves the worst
                # case; the residual half-tick jitter is the ghosting
                # milestone's remaining business.
                self._camera_frame_seen[name] = (
                    (now + checked) / 2.0 if checked is not None else now
                )
                any_fresh = True
            self._camera_frame_checked[name] = now
            seen = self._camera_frame_seen.get(name)
            age = (
                CAMERA_FRAME_STAGING_S
                if seen is None
                else CAMERA_FRAME_STAGING_S + max(0.0, now - seen)
            )

            result = unproject_frame(
                rays,
                depth_raw,
                raw.get("annotation"),
                pose_from_state(state, age, self._yaw_rate_rps),
                geometry,
                UNKNOWN_SEMANTIC_RGB,
            )
            if result is None:
                continue
            points, colours, sampled = result
            if not self._logged_unprojection:
                reach.append(
                    (name, sampled, self._furthest_m(points, origin), age)
                )
            if not len(points):
                continue
            point_chunks.append(points)
            colour_chunks.append(colours)

        self._watch_vision_liveness(now, any_fresh, images)
        if reach and any_fresh and not self._logged_unprojection:
            self._logged_unprojection = True
            # The `Sensor reach:` equivalent: per-camera counts and reach are
            # what decide the strides, and the frame age is the one number
            # the ego-motion milestone has to be judged against.
            LOGGER.info(
                "Unprojection check: %s | total %d points | mean frame age "
                "%.0f ms (staging %.0f ms assumed) | eye height %.2f m",
                " | ".join(
                    f"{name} {count} returns, furthest {far:.0f} m"
                    for name, count, far, _ in reach
                ),
                sum(count for _, count, _, _ in reach),
                1000.0 * float(np.mean([age for *_, age in reach])),
                1000.0 * CAMERA_FRAME_STAGING_S,
                self._vision_eye_height_m,
            )
        return point_chunks, colour_chunks, images, any_fresh

    @staticmethod
    def _furthest_m(points: np.ndarray, origin: np.ndarray) -> float:
        if not len(points):
            return 0.0
        offsets = points[:, :2] - origin[:2].astype(np.float32)
        ranges = np.hypot(offsets[:, 0], offsets[:, 1])
        finite = ranges[np.isfinite(ranges)]
        return float(finite.max()) if len(finite) else 0.0

    def _note_poll_failure(self, exc: Exception, what: str) -> None:
        """
        One failed tick against the continuous-failure budget.

        Shared by both sensor modes: the budget's semantics -- time-based, so a
        map load survives, and a teardown that emits sensors_stopped so no dead
        frame stays on screen -- predate Vision mode and must not fork.
        """
        now = time.perf_counter()
        if self._first_failure_at is None:
            self._first_failure_at = now
        self._poll_failures += 1
        failing_for = now - self._first_failure_at
        LOGGER.warning(
            "%s polling failed (%d attempts over %.1fs): %s",
            what,
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
                f"{what} streaming stopped after repeated connection errors: {exc}"
            )

    @staticmethod
    def lidar_sensor_kwargs(mount: SensorMount) -> dict[str, Any]:
        """
        The `Lidar` constructor arguments for one mount, after name, bng and
        vehicle. Public and pure so tools/unprojection_oracle.py builds the
        SAME unit the app does -- an oracle that differed from the app in any
        constructor argument would be measuring the difference.
        """
        return dict(
            requested_update_time=LIDAR_UPDATE_TIME_S,
            update_priority=0.0,
            pos=mount.position_vehicle,
            dir=mount.direction_vehicle,
            up=(0.0, 0.0, 1.0),
            frequency=LIDAR_UPDATE_HZ,
            # Every optic is per-mount, not global: the front unit reaches
            # much further on a narrow, dense sweep, and the roof unit trades
            # vertical aperture for ground-ring spacing. See SensorMount.
            vertical_resolution=mount.vertical_resolution,
            vertical_angle=mount.vertical_fov_deg,
            horizontal_angle=mount.horizontal_fov_deg,
            max_distance=mount.max_distance_m,
            density=mount.density,
            is_rotate_mode=False,
            is_360_mode=False,
            is_using_shared_memory=True,
            is_visualised=False,
            # BeamNG writes each latest 30 Hz scan directly into shared memory
            # so the display loop never waits on four requests.
            is_streaming=True,
            # The road-scan unit runs unannotated while the visual-paint
            # experiment is on: its colour channel then carries whatever the
            # engine renders instead of class colours, and the one-shot
            # `Colour check:` line reports what that actually is. See
            # LIDAR_ROAD_VISUAL_COLOUR.
            is_annotated=not (LIDAR_ROAD_VISUAL_COLOUR and mount.name == "road"),
            is_static=False,
            is_snapping_desired=False,
            is_force_inside_triangle=False,
            is_dir_world_space=False,
        )

    @staticmethod
    def camera_sensor_kwargs(mount: CameraMount) -> dict[str, Any]:
        """
        The `Camera` constructor arguments for one mount, after name, bng and
        vehicle -- the rung-0.5 rig: colour for the grid, depth and
        annotation for the unprojection. Public and pure for the same reason
        `lidar_sensor_kwargs` is.

        `integer_depth=False` and `postprocess_depth=False` are both TRAPS the
        spec names: the default quantises depth to 0-255 silently, and the
        postprocess is a 256-iteration Python loop per frame.
        """
        return dict(
            requested_update_time=CAMERA_UPDATE_TIME_S,
            update_priority=0.0,
            pos=mount.position_vehicle,
            dir=mount.direction_vehicle,
            up=(0.0, 0.0, 1.0),
            resolution=mount.resolution,
            field_of_view_y=mount.vertical_fov_deg,
            near_far_planes=CAMERA_NEAR_FAR_PLANES,
            is_using_shared_memory=True,
            is_render_colours=True,
            is_render_annotations=True,
            is_render_instance=False,
            is_render_depth=True,
            integer_depth=False,
            postprocess_depth=False,
            is_visualised=False,
            is_streaming=True,
            is_static=False,
            is_snapping_desired=False,
            is_force_inside_triangle=False,
            is_dir_world_space=False,
        )

    def _attach_camera_rig(
        self, vehicle: Vehicle, geometry: VehicleGeometry, sensor_prefix: str
    ) -> int:
        """
        Build the eight-camera Vision rig on the connected vehicle.

        All three channels from rung 0.5 on: depth and annotation are what the
        unprojection consumes, colour is what the grid draws. Annotation is a
        second full geometry pass (measured 42 -> 33 Hz sim rate for eight
        cameras) and depth doubles the bytes the simulator writes -- both are
        paid for now because both are read. Streaming shared memory is the
        only viable transport -- the poll path was measured at 204 ms per
        8-camera frame, ~150x slower.
        """
        from beamngpy.sensors import Camera

        rig = derive_camera_rig(geometry)
        self._camera_rays = build_rig_rays(rig)
        self._vision_eye_height_m = max(
            float(mount.position_vehicle[2]) for mount in rig.values()
        )
        for index, mount in enumerate(rig.values()):
            self.status_changed.emit(
                "ATTACHING",
                f"Attaching {mount.name} camera ({index + 1}/{len(rig)})",
            )
            sensor = Camera(
                f"{sensor_prefix}_{mount.name}",
                self._bng,
                vehicle,
                **self.camera_sensor_kwargs(mount),
            )
            self._sensor_names.append(mount.name)
            self._sensors.append(sensor)
        LOGGER.info(
            "Vision rig: %d cameras at %dx%d, update %.0f ms, colour + depth + "
            "annotation, %d lattice samples | %s",
            len(rig),
            rig[next(iter(rig))].resolution[0],
            rig[next(iter(rig))].resolution[1],
            CAMERA_UPDATE_TIME_S * 1000.0,
            sum(rays.sample_count for rays in self._camera_rays.values()),
            ", ".join(
                f"{m.name} hfov {m.horizontal_fov_deg:.0f} z "
                f"{m.position_vehicle[2]:.2f} stride "
                f"{m.sample_stride[0]}x{m.sample_stride[1]}"
                for m in rig.values()
            ),
        )
        self._check_capture_settings()
        return len(rig)

    _CAPTURE_SETTINGS_CHUNK = (
        "local ok, res = pcall(function() return {"
        "overall = settings.getValue('GraphicOverallQuality'),"
        "shader = settings.getValue('GraphicShaderQuality'),"
        "texture = settings.getValue('GraphicTextureQuality'),"
        "lighting = settings.getValue('GraphicLightingQuality'),"
        "aa = settings.getValue('GraphicAntialias'),"
        "motion_blur = settings.getValue('PostFXMotionBlurEnabled'),"
        "focused = Engine.isProgramFocused and Engine.isProgramFocused() or false"
        "} end) "
        "if ok then return jsonEncode(res) else return jsonEncode({}) end"
    )

    def _check_capture_settings(self) -> None:
        """
        Report the renderer settings the camera rig depends on, once per attach.

        This CHECKS rather than SETS. Writing them was the original plan and it
        was measured not to work: `bng.settings.change` plus `apply_graphics`
        on 0.39.4 moved neither the sensor nor the game view, so a "pin" would
        have been a line that quietly did nothing while reading as a guarantee.

        Everything here is best-effort -- an unreadable setting is logged as
        unknown and never warned about, and any failure at all leaves streaming
        untouched. A game version that does not expose a key is not evidence of
        a bad setting.
        """
        if self._bng is None:
            return
        try:
            reply = self._bng.control.queue_lua_command(
                self._CAPTURE_SETTINGS_CHUNK, response=True
            )
            values = json.loads(reply) if reply else {}
        except Exception:
            LOGGER.debug("Capture settings were not readable", exc_info=True)
            return
        if not isinstance(values, dict) or not values:
            LOGGER.debug("Capture settings came back empty")
            return

        LOGGER.info(
            "Capture check: quality overall=%s shader=%s texture=%s lighting=%s"
            " | antialias=%s | motion blur=%s | window focused=%s",
            values.get("overall", "?"),
            values.get("shader", "?"),
            values.get("texture", "?"),
            values.get("lighting", "?"),
            values.get("aa", "?"),
            values.get("motion_blur", "?"),
            values.get("focused", "?"),
        )
        for warning in capture_setting_warnings(values):
            LOGGER.warning("Capture check: %s", warning)

    def _watch_vision_liveness(
        self, now: float, any_fresh: bool, images: list[CameraImage]
    ) -> None:
        """
        One line when the rig comes alive, one warning if it never does.

        The silent failure this exists for was measured live: with
        requested_update_time=0.0 every streaming buffer stays zero-filled
        while the read loop spins -- a "working" rig of black frames. The
        offline suite cannot reach it, so it is a check line, like the mount
        heights.
        """
        if any_fresh and not self._logged_vision_check:
            self._logged_vision_check = True
            since = (
                now - self._vision_streaming_since
                if self._vision_streaming_since is not None
                else 0.0
            )
            LOGGER.info(
                "Vision check: first fresh frames %.1f s after attach | "
                "%d cameras delivering | colour + depth + annotation",
                since,
                len(images),
            )
            return
        if (
            not any_fresh
            and not self._logged_vision_check
            and not self._logged_vision_silence
            and self._vision_streaming_since is not None
            and now - self._vision_streaming_since > _VISION_SILENCE_WARN_S
        ):
            self._logged_vision_silence = True
            LOGGER.warning(
                "Vision check: no camera has delivered a new frame in %.0f s. "
                "Known trap: streaming buffers stay zero-filled when "
                "requested_update_time is 0.0 (CAMERA_UPDATE_TIME_S must be "
                "positive); also check the graphics preset is above 'Lowest', "
                "which returns empty camera buffers.",
                _VISION_SILENCE_WARN_S,
            )

    def _scan_for_parking(
        self,
        state: dict[str, Any],
        bev: np.ndarray,
        heights: np.ndarray,
        materials: np.ndarray,
        geometry: VehicleGeometry,
    ) -> tuple[ParkingSlot, ...]:
        """
        Accumulate paint every tick, re-detect bays occasionally, project both.

        The three rates are deliberate and different. Marking cells are folded
        in EVERY tick, because the continuous dividers the detector needs only
        exist by accumulation -- a single frame lays ground rings ACROSS a
        divider, not along it. The bay SET is rebuilt every
        PARKING_SCAN_INTERVAL_S, because the sweep is the expensive part and
        the lot is not changing shape. And the projection into the BEV frame
        happens every tick regardless, so the drawn rectangles stay glued to
        the ground between scans instead of lagging the car.
        """
        ego_pos = vec3(state["pos"])
        right_axis, forward, _ = vehicle_axes(state)
        self._marking_memory.update(
            ego_pos, right_axis, forward, bev[materials == SURFACE_MARKING]
        )

        now = time.monotonic()
        if now - self._last_parking_scan_at >= PARKING_SCAN_INTERVAL_S:
            self._last_parking_scan_at = now
            # Occupancy evidence comes from this tick's cloud rather than from
            # PlanningMemory: that store is planner-only and is gated on
            # self-driving, and a bay is worth scanning for with self-driving
            # off. Anything standing above the ego's ground plane counts --
            # the height floor keeps the surface itself and its paint out, and
            # the bay rectangle is shrunk before anything inside it is
            # believed.
            standing = bev[
                heights - geometry.ground_z_vehicle
                >= PARKING_OCCUPANCY_MIN_HEIGHT_M
            ]
            ego_xy = np.asarray(ego_pos, dtype=np.float64)[:2]
            r_xy = np.asarray(right_axis, dtype=np.float64)[:2]
            f_xy = np.asarray(forward, dtype=np.float64)[:2]
            r_xy = r_xy / max(float(np.hypot(*r_xy)), 1e-9)
            f_xy = f_xy / max(float(np.hypot(*f_xy)), 1e-9)
            standing_world = (
                ego_xy + standing[:, [0]] * r_xy + standing[:, [1]] * f_xy
                if len(standing)
                else np.empty((0, 2), dtype=np.float64)
            )
            report = ScanReport()
            found = find_bays(
                self._marking_memory.cells_world(),
                standing_world,
                ego_xy,
                report,
            )
            # Bays a scan happened to miss survive for a short distance. Each
            # sits near several thresholds at once, so one can drop out and
            # return a moment later -- which made the bays flash and, worse,
            # took the SELECTION with them, since a selection is re-matched
            # against the offered set every scan.
            self._parking_bays = remember_bays(
                self._parking_bays,
                found,
                self._marking_memory.travelled_m,
                self._parking_seen_at,
                ego_xy,
            )
            self._last_parking_report = report
            # A selection that no longer matches any bay is dropped here, so
            # the held pose can never outlive the evidence for it.
            if self._parking_selected is not None:
                matched = match_selection(
                    self._parking_bays, self._parking_selected
                )
                self._parking_selected = (
                    None if matched is None else matched.centre
                )
            self._log_parking_check(ego_xy)

        return project_bays(
            self._parking_bays,
            ego_pos,
            right_axis,
            forward,
            selected_world=self._parking_selected,
        )

    def _drive_into_bay(
        self,
        state: dict[str, Any],
        slots: tuple[ParkingSlot, ...],
        geometry: VehicleGeometry,
        obstacles: np.ndarray | None,
        occupancy: Occupancy | None = None,
        rear_aeb: AebState | None = None,
    ) -> DrivingPlan:
        """
        One tick of the manoeuvre, as a DrivingPlan so `_actuate` is unchanged.

        The bay is found by IDENTITY against the held world pose rather than
        by index: the scan rebuilds its list on its own cadence, and the goal
        this manoeuvre committed to is a place in the world, not a subscript.
        """
        selected = next(
            (slot for slot in slots if slot.selected), None
        )
        now = time.perf_counter()
        dt = (
            now - self._last_plan_at
            if self._last_plan_at
            else DISPLAY_INTERVAL_MS / 1000.0
        )
        self._last_plan_at = now
        forward = vehicle_axes(state)[1]
        forward_speed = float(
            vec3(state.get("vel", (0.0, 0.0, 0.0))) @ forward
        )
        reported_gear = self._reported_gear()
        command, park = self._parking_driver.step(
            selected,
            geometry,
            forward_speed,
            dt,
            obstacles=obstacles,
            occupancy=occupancy,
            reported_gear=reported_gear,
            forward_gear=forward_gear_index(reported_gear),
            # The REAR brake arms at parking speed and really can fire while
            # backing in. It is left armed and allowed to win: the manoeuvre
            # hands back rather than fighting a system that has decided the
            # car is about to hit something.
            rear_aeb_braking=bool(rear_aeb is not None and rear_aeb.engaged),
        )
        self._last_park_state = park
        if self._parking_job is not None:
            status = {
                PARK_BLOCKED: "WAITING",
                PARK_SECURING: "SECURING",
                PARK_ARRIVED: "SUCCEEDED",
            }.get(park.phase, "EXECUTING")
            self._parking_job = replace(self._parking_job, status=status)
        if park.phase != self._logged_park_phase:
            self._logged_park_phase = park.phase
            LOGGER.info(
                "Park check: %s -- %s (%.1f m to go) | gear commanded %+d, "
                "box reports %r, speed %.2f m/s, legs %s",
                park.phase,
                park.reason,
                park.remaining_m,
                command.gear,
                reported_gear,
                forward_speed,
                "none"
                if self._parking_driver.legs is None
                else " then ".join(
                    "reverse" if leg.reverse else "forward"
                    for leg in self._parking_driver.legs
                ),
            )
        elif park.phase == PARK_SHIFTING:
            # A shift that does not complete is invisible otherwise: the
            # phase never changes, so the one-shot line above never fires
            # again and the car simply sits there. Throttled, because the
            # normal case is two or three ticks.
            now_shift = time.monotonic()
            if now_shift - self._last_shift_log_at > 2.0:
                self._last_shift_log_at = now_shift
                LOGGER.info(
                    "Park check: still %s -- commanded %+d, box reports %r, "
                    "speed %.2f m/s",
                    park.phase,
                    command.gear,
                    reported_gear,
                    forward_speed,
                )
        if park.phase == PARK_ARRIVED:
            self._complete_parking_drive()
        elif park.phase not in (
            PARK_APPROACH,
            PARK_BACKING,
            PARK_SHIFTING,
            PARK_BLOCKED,
            PARK_SECURING,
        ):
            self._disengage_parking_drive(park.reason)
        return DrivingPlan(
            arc=_BLIND_ARC,
            command=command,
            forward_speed_mps=forward_speed,
            reported_gear=reported_gear,
        )

    def _log_parking_check(self, ego_xy: np.ndarray) -> None:
        """
        The scan's own account of itself, throttled.

        Detection is a chain of geometric filters and the screen shows only
        the survivors, so "there is paint there but no bay on it" is not
        answerable from the picture -- which is exactly how it was reported.
        `ScanReport` names the filter that consumed each candidate, so the
        question becomes one log line.

        It also carries the live-only fact the offline suite cannot reach:
        whether this map's bay dividers annotate through the LiDAR at all.
        Lane paint was confirmed to; bay paint is a separate question, and a
        lot whose bays are baked into the ground texture rather than shipped
        as decals returns nothing here however many cells accumulate.
        """
        report = self._last_parking_report
        if report is None:
            return
        now = time.monotonic()
        first = not self._logged_parking_check
        # After the one-shot, only when the answer CHANGES or a minute has
        # passed -- a per-scan line at 2 Hz would bury the log.
        changed = report.bays_found != self._last_parking_bay_count
        if not first and not changed and now - self._last_parking_log_at < 60.0:
            return
        self._logged_parking_check = True
        self._last_parking_log_at = now
        self._last_parking_bay_count = report.bays_found

        if not self._parking_bays:
            LOGGER.info("Parking check: no bays -- %s", report.summary())
            return
        nearest = self._parking_bays[0]
        clear = sum(1 for bay in self._parking_bays if not bay.occupied)
        LOGGER.info(
            "Parking check: %d bays (%d clear, %d occupied); nearest %.1f m "
            "away, %.2f m wide x %.2f m deep. %s",
            len(self._parking_bays),
            clear,
            len(self._parking_bays) - clear,
            float(np.hypot(*(np.asarray(nearest.centre) - ego_xy))),
            nearest.width_m,
            nearest.depth_m,
            report.summary(),
        )

    def _compute_plan(
        self,
        state: dict[str, Any],
        obstacles: np.ndarray,
        geometry: VehicleGeometry,
        forward_speed: float,
        heading: float,
        had_returns: bool,
        road_grid: RoadGrid | None = None,
    ) -> DrivingPlan:
        assert self._controller is not None

        if had_returns:
            # ~LOOKAHEAD_TIME_S seconds of travel, so the lateral guidance
            # terms keep their tuned character at any speed.
            lookahead = min(
                LOOKAHEAD_MAX_M,
                max(LOOKAHEAD_MIN_M, LOOKAHEAD_TIME_S * abs(forward_speed)),
            )
            # The reference path is rebuilt every tick (the ego frame moves
            # every tick) from the cached route; the legacy bearing hint runs
            # only when no path could be built, so the planner never sees two
            # lateral guidance sources at once.
            route_path = self._route_context(state)
            # Stashed for the display frames built later this tick, on this
            # same thread; cleared on disengage so the overlay only ever shows
            # a route the car is actually following.
            self._last_route_path = route_path
            route_remaining = (
                None if route_path is None else route_path.remaining_m
            )
            if route_path is not None and not self._route_check_logged:
                self._route_check_logged = True
                self._log_route_check(route_path)
            arc = plan_arc(
                obstacles,
                geometry,
                nav_heading_rad=(
                    None
                    if route_path is not None
                    else route_heading(self._route, state)
                ),
                # The curvature actually being driven right now, which is also
                # segment A of every deferred candidate.
                previous_curvature=self._controller.current_curvature,
                lookahead_m=lookahead,
                route=route_path,
                road_grid=road_grid,
            )
            if route_remaining is not None:
                self._arrived_hold = route_remaining <= ROUTE_ARRIVAL_LATCH_M
            elif self._arrived_hold:
                # Arriving CLEARS the in-game route (groundMarkers drops its
                # target at the marker, and the leftover polyline is too short
                # to build a path from), so the hold would die exactly when it
                # is needed and the car would pull away at the destination.
                # The latch survives the route disappearing; only a route with
                # meaningful distance left -- a new destination -- releases it.
                arc = replace(arc, route_speed_limit_mps=0.0)
            rear_free_m = rear_free_distance(obstacles, geometry)
            obstacle_count = len(obstacles)

            mode = self._controller.mode
            if mode == REVERSING or (
                mode == BLOCKED and abs(forward_speed) < STALL_SPEED_MPS
            ):
                # The steered reverse: plan_arc on the 180-degree-rotated
                # cloud, exactly as rear AEB mirrors its corridor -- rotation
                # preserves handedness, so every helper applies unchanged,
                # and previous_curvature maps into the travel frame as -k
                # (controller._reverse holds the derivation). keep_right off:
                # reversing wants clearance and free distance only, and the
                # winner is plan_arc's own argmin -- back toward the most
                # open region while steering least. The forward re-plan after
                # recovery re-orients the car with the full cost stack, so
                # there is deliberately no bespoke "maximise forward options"
                # objective here. The memory-merged cloud matters: what is
                # straight behind left the sensors' view long ago.
                reverse_arc = plan_arc(
                    mirror_points(obstacles),
                    self._mirrored_geometry
                    if self._mirrored_geometry is not None
                    else mirrored(geometry),
                    previous_curvature=-self._controller.current_curvature,
                    lookahead_m=REVERSE_DISTANCE_M + 4.0,
                    keep_right=False,
                    required_free_m=REVERSE_REQUIRED_FREE_M,
                    smoothness_weight=REVERSE_COST_SMOOTHNESS,
                )
                # Entry and abort both read the ARC's own free distance: the
                # arc is what will actually be driven, and gating on the
                # straight-back corridor would refuse a recovery whose whole
                # point is steering around what is straight behind.
                straight_back_m = rear_free_m
                rear_free_m = float(reverse_arc.free_distance_m)
                if mode == REVERSING and not self._reverse_check_logged:
                    self._reverse_check_logged = True
                    LOGGER.info(
                        "Reverse check: steered reverse -- travel-frame k "
                        "%+.4f, arc free %.1f m against %.1f m straight back",
                        reverse_arc.curvature,
                        reverse_arc.free_distance_m,
                        straight_back_m,
                    )
            else:
                reverse_arc = None
        else:
            arc = _BLIND_ARC
            rear_free_m = 0.0
            obstacle_count = 0
            route_remaining = None
            reverse_arc = None
            self._last_route_path = None

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
            reverse_arc=reverse_arc,
        )
        if command.reason == "Arrived at destination" and not self._arrival_logged:
            self._arrival_logged = True
            LOGGER.info(
                "Route check: arrived -- holding with %s of route left",
                "n/a"
                if route_remaining is None
                else f"{route_remaining:.1f} m",
            )
        self._log_driving_telemetry(arc, command, forward_speed, obstacle_count)
        return DrivingPlan(
            arc=arc,
            command=command,
            forward_speed_mps=forward_speed,
            reverse_arc=reverse_arc,
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
            "gain %.2f | defer %.0f m -> %+.4f | route v %s xtrack %s | "
            "block %.1f ms | %s",
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
            "n/a"
            if arc.route_speed_limit_mps is None
            else f"{arc.route_speed_limit_mps * 3.6:.0f}",
            "n/a"
            if arc.route_cross_track_m is None
            else f"{arc.route_cross_track_m:+.1f}",
            # The previous tick's figure, like control_ms: this tick's block
            # is still running when this line prints.
            self._last_drive_block_ms,
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

    def _watch_visual_colours(
        self, name: str, colours: np.ndarray, points: np.ndarray
    ) -> None:
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
        # R==G==B distinguishes an intensity-style grayscale channel from a
        # genuinely rendered scene; the first probe's summary could not tell
        # the two apart, which is exactly what this line has to answer.
        grey = 100.0 * float(
            np.mean(
                (colours[:, 0] == colours[:, 1])
                & (colours[:, 1] == colours[:, 2])
            )
        )
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
            "%.1f%% pure black, %d unique colours, %.1f%% grayscale "
            "(R==G==B), luminance %s, %.2f%% above 160. Dead channel: "
            "near-100%% black. Intensity channel: ~100%% grayscale. "
            "Rendered scene: mostly non-gray colours.",
            len(colours),
            black,
            unique,
            grey,
            spread,
            bright,
        )
        self._dump_colour_probe(points, colours)

    @staticmethod
    def _dump_colour_probe(points: np.ndarray, colours: np.ndarray) -> None:
        """
        Save the probed scan so the channel can be LOOKED at, not just
        summarised: scatter the points coloured by the channel and lane lines
        are either visibly there or visibly not, which settles stage one in a
        way no statistic does. One file, overwritten per attach.
        """
        try:
            target = (
                Path(__file__).parents[2] / "logs" / "road_colour_probe.npz"
            )
            np.savez_compressed(target, points=points, colours=colours)
            LOGGER.info("Colour check: scan dumped to %s", target)
        except Exception:
            LOGGER.debug("Could not dump the colour probe", exc_info=True)

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

    def _poll_route(self) -> None:
        """
        Refresh the cached route on its own cadence.

        Reading the route is a blocking Lua round-trip, and the route only
        changes when the player sets a new destination, so it must not ride
        the display loop. A parseable "no target" clears the cache immediately
        (the player cancelling is data); a TRANSPORT failure keeps the last
        good route for ROUTE_STALE_GRACE_S, because one dropped reply used to
        wipe a perfectly good route for a full poll interval.
        """
        now = time.monotonic()
        if now - self._last_nav_poll_at < NAV_POLL_INTERVAL_MS / 1000.0:
            return
        self._last_nav_poll_at = now
        started = time.perf_counter()
        reply = fetch_route_reply(self._run_lua)
        self._last_nav_rtt_ms = (time.perf_counter() - started) * 1000.0
        if reply is None:
            if (
                self._route is not None
                and now - self._route_fresh_at > ROUTE_STALE_GRACE_S
            ):
                self._route = None
            return
        self._route = parse_route(reply)
        if self._route is not None:
            self._route_fresh_at = now

    def _nav_heading(self, state: dict[str, Any]) -> float | None:
        """Legacy bearing hint from the cached route (see `route_heading`)."""
        self._poll_route()
        return route_heading(self._route, state)

    def _route_context(self, state: dict[str, Any]):
        """
        The cached route as a reference path in the current ego frame.

        Polls on the nav cadence, rebuilds per tick: ~40 samples of interp
        against the 40 ms tick. None whenever no honest path exists, which is
        the signal to fall back to the bearing hint.
        """
        self._poll_route()
        return build_route_path(self._route, state)

    def _route_world_preview(self) -> np.ndarray | None:
        """
        The cached route's world nodes, clipped to the preview reach, for the
        WORLD overlay. None whenever no path is being followed.
        """
        if self._last_route_path is None or self._route is None:
            return None
        nodes = self._route.path_world
        if len(nodes) < 2:
            return None
        chords = np.hypot(*np.diff(nodes[:, :2], axis=0).T)
        along = np.concatenate(([0.0], np.cumsum(chords)))
        count = max(int(np.searchsorted(along, ROUTE_PREVIEW_M)) + 1, 2)
        return np.asarray(nodes[:count], dtype=np.float32)

    def _log_memory(self, now: float, merged_count: int) -> None:
        if now - self._last_memory_log_at < _MEMORY_LOG_INTERVAL_S:
            return
        self._last_memory_log_at = now
        LOGGER.info(
            "Memory check: %d cells (%d vehicle), oldest %.0f m of travel "
            "ago, %d road cells | %d obstacle points to the planner",
            self._memory.cell_count,
            self._memory.vehicle_cell_count,
            self._memory.oldest_age_m(),
            self._memory.road_cell_count,
            merged_count,
        )

    def _log_route_check(self, path) -> None:
        """
        One shot at the first reference path of an engagement: the facts only
        a live game can prove -- above all whether the enriched Lua fields
        (radius, linkCount) actually arrived on this BeamNG version, because
        the chunk defaults them rather than failing and the offline suite
        cannot tell the difference.
        """
        route = self._route
        nodes = 0 if route is None else len(route.path_world)
        spacing = "n/a"
        if route is not None and len(route.path_world) >= 2:
            chords = np.hypot(
                *np.diff(route.path_world[:, :2], axis=0).T
            )
            spacing = f"{chords.min():.1f}-{chords.max():.1f} m"
        with_radius = (
            0
            if route is None or route.half_width_m is None
            else int((route.half_width_m >= 0.0).sum())
        )
        with_links = (
            0
            if route is None or route.link_counts is None
            else int((route.link_counts > 0).sum())
        )
        LOGGER.info(
            "Route check: %d nodes, %.0f m to go (spacing %s) | %d/%d carry "
            "radius, %d/%d carry linkCount | preview v0 %.1f km/h, "
            "cross-track %+.2f m | lua round-trip %.0f ms",
            nodes,
            path.remaining_m,
            spacing,
            with_radius,
            nodes,
            with_links,
            nodes,
            path.speed_limit_mps * 3.6,
            path.cross_track_m,
            self._last_nav_rtt_ms,
        )

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
                "parkingbrake": command.parking_brake,
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
        acceleration term (about a centimetre at full braking). The HEADING is
        advanced too, by the yaw rate measured between successive states
        (2026-08-23): it used to be left alone on the reasoning that it moves
        under a degree over a 40 ms tick, which is true -- but a tick that
        stretches to 100-250 ms turns that into 3-8 degrees at an ordinary
        30 deg/s, and every cloud stamped into WORLD's world-anchored stores
        during a turn was rotated by that much about the car. RAW BEV cannot
        show the error (the same state transforms the cloud both ways and it
        cancels), which is exactly why it was only ever reported from WORLD:
        "the whole world turns with me, then corrects after a few seconds of
        driving" -- the seconds being the road store's memory by the metre.
        """
        future, self._state_future = self._state_future, None
        if future is None:
            state = self._get_vehicle_state()
            self._observe_state_yaw(state, time.perf_counter())
            return state
        try:
            state, done_at = future.result()
        except Exception:
            LOGGER.debug("Prefetched state poll failed; re-polling", exc_info=True)
            state = self._get_vehicle_state()
            self._observe_state_yaw(state, time.perf_counter())
            return state
        age = time.perf_counter() - done_at
        if age > _STATE_PREFETCH_MAX_AGE_S:
            # From before an app stall; extrapolating across it would be worse
            # than the round-trip it saves.
            state = self._get_vehicle_state()
            self._observe_state_yaw(state, time.perf_counter())
            return state
        self._observe_state_yaw(state, done_at)
        if age > 0.0 and "pos" in state:
            position = vec3(state["pos"]) + vec3(
                state.get("vel", (0.0, 0.0, 0.0))
            ) * age
            state["pos"] = tuple(float(value) for value in position)
            if "dir" in state and self._yaw_rate_rps != 0.0:
                state["dir"] = tuple(
                    float(value)
                    for value in rotate_about_up(
                        vec3(state["dir"]), self._yaw_rate_rps * age
                    )
                )
        return state

    def _observe_state_yaw(self, state: dict[str, Any], at: float) -> None:
        """
        Update the yaw-rate estimate from the heading of a freshly polled
        state and the time its round trip finished.

        A plain difference of successive headings over their interval,
        lightly low-passed: the state is polled every tick, so the estimate
        is 25 Hz and the filter only takes the edge off the quantisation. It
        is what `_take_vehicle_state` advances the prefetched heading with and
        what the vision acquisition rewinds each camera frame with.
        """
        direction = state.get("dir")
        if direction is None:
            return
        forward = vec3(direction)
        heading = float(np.arctan2(forward[1], forward[0]))
        previous = self._yaw_observation
        self._yaw_observation = (heading, at)
        if previous is None:
            return
        dt = at - previous[1]
        if dt <= 1e-3 or dt > 1.0:
            return
        delta = (heading - previous[0] + np.pi) % (2.0 * np.pi) - np.pi
        rate = float(delta / dt)
        blend = min(1.0, dt / _YAW_RATE_TAU_S)
        self._yaw_rate_rps += blend * (rate - self._yaw_rate_rps)

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

    def _actor_enrichment_refused(self, now: float) -> bool:
        """
        Whether the simulator recently refused actor enrichment.

        Both halves of it -- the 1 Hz registry (`get_current_info`, measured
        120 ms) and the 10 Hz state poll (`get_states`, 39 ms even when
        rejected) -- are blocking round trips on the worker thread, and
        BeamNG.tech rejects the state poll outright in free-roam. Retrying a
        known refusal ten times a second cost the tick ~0.5 s of every
        second; one refusal now rests both for WORLD_ACTOR_RETRY_S.
        """
        return now - self._actor_refused_at < WORLD_ACTOR_RETRY_S

    def _refresh_actor_registry(self, now: float) -> None:
        if (
            self._bng is None
            or self._actor_enrichment_refused(now)
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
            self._actor_enrichment_refused(now)
            or now - self._last_actor_state_at < WORLD_ACTOR_STATE_INTERVAL_S
        ):
            if now - self._last_actor_success_at > WORLD_ACTOR_FADE_S:
                self._actor_observations = ()
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
        except Exception as exc:
            self._actor_refused_at = now
            if not self._logged_actor_refusal:
                # Once per attach, and without the traceback: in free-roam
                # this is the expected answer, and ten tracebacks a second
                # buried every other line in the log.
                self._logged_actor_refusal = True
                LOGGER.warning(
                    "Actor pose enrichment unavailable (%s); retrying every "
                    "%.0f s. Expected in free-roam -- traffic is drawn from "
                    "the cloud alone.",
                    str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
                    WORLD_ACTOR_RETRY_S,
                )
            else:
                LOGGER.debug("Actor pose enrichment still unavailable")
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
        self._actor_refused_at = -float("inf")
        self._logged_actor_refusal = False
        self._yaw_observation = None
        self._yaw_rate_rps = 0.0

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
        self._disengage_parking_drive("Sensors stopped")
        self._disengage_self_driving("Sensors stopped", announce=False)
        for sensor in reversed(self._sensors):
            try:
                sensor.remove()
            except Exception:
                LOGGER.debug("Could not remove sensor", exc_info=True)
        self._sensors.clear()
        self._sensor_names.clear()
        self._camera_digests = {}
        self._camera_frame_seen = {}
        self._camera_frame_checked = {}
        self._camera_rays = {}
        self._vision_eye_height_m = 0.0
        self._vision_streaming_since = None
        self._logged_vision_check = False
        self._logged_vision_silence = False
        self._logged_unprojection = False
        self._geometry = None
        self._mirrored_geometry = None
        self._memory.clear()
        self._parking_map.clear()
        # The bays described a lot this car was standing in; a re-attach may
        # be a different car in a different place, and a held selection must
        # never survive the evidence for it. The toggle itself is reported
        # off so the button cannot stay lit over a dead scan.
        if self._parking_scan:
            self._parking_scan = False
            self.parking_changed.emit(False)
        self._marking_memory.clear()
        self._parking_bays = ()
        self._parking_seen_at = {}
        self._parking_selected = None
        self._last_parking_scan_at = 0.0
        self._logged_parking_check = False
        self._parking_seen_at: dict[tuple[int, int], float] = {}
        self._last_parking_report: ScanReport | None = None
        self._last_parking_log_at = 0.0
        self._last_parking_bay_count = -1
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
