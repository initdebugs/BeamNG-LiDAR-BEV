from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from beamng_lidar_bev import worker as worker_module
from beamng_lidar_bev.aeb import FORWARD, REVERSE, BrakeEvent
from beamng_lidar_bev.config import (
    AEB_CONFIRM_S,
    DISPLAY_INTERVAL_MS,
    LOOKAHEAD_MAX_M,
    LOOKAHEAD_MIN_M,
)
from beamng_lidar_bev.models import BevFrame, PerceptionSnapshot, VehicleGeometry
from beamng_lidar_bev.semantics import SemanticPalette
from beamng_lidar_bev.worker import BeamNgWorker


class VehicleStub:
    def __init__(self, gear: object = "P") -> None:
        self.polled_names: tuple[str, ...] = ()
        self.controls: list[dict[str, float]] = []
        self.shift_modes: list[str] = []
        self.attached: list[str] = []
        self.state = {
            "pos": (1.0, 2.0, 3.0),
            "dir": (0.0, 1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "vel": (0.0, 0.0, 0.0),
        }
        # Mirrors beamngpy's Sensors mapping. "P" is what an automatic reports
        # at rest, which is the case that used to leave the car in Park.
        self.sensors: dict[str, dict[str, object]] = {"electrics": {"gear": gear}}

    def poll_sensors(self, *names: str) -> None:
        self.polled_names = names

    def attach_sensor(self, name: str, sensor: object) -> None:
        self.attached.append(name)

    def control(self, **kwargs: float) -> None:
        self.controls.append(dict(kwargs))

    def set_shift_mode(self, mode: str) -> None:
        self.shift_modes.append(mode)


class BngStub:
    """Just enough BeamNGpy surface for the navigation Lua round-trip."""

    def __init__(self, reply: str | None = None) -> None:
        self.chunks: list[str] = []
        self._reply = reply
        self.control = SimpleNamespace(queue_lua_command=self._queue_lua_command)

    def _queue_lua_command(self, chunk: str, response: bool = False) -> str | None:
        self.chunks.append(chunk)
        return self._reply


def test_vehicle_state_uses_free_roam_compatible_sensor() -> None:
    vehicle = VehicleStub()
    worker = SimpleNamespace(_vehicle=vehicle, _self_driving=False)

    state = BeamNgWorker._get_vehicle_state(worker)  # type: ignore[arg-type]

    assert vehicle.polled_names == ("state",)
    assert state["pos"] == (1.0, 2.0, 3.0)


class StreamingLidarStub:
    def __init__(self) -> None:
        self.stream_calls = 0

    def stream(self) -> dict[str, np.ndarray]:
        self.stream_calls += 1
        return {
            "pointCloud": np.asarray(((1.0, 12.0, 2.5),), dtype=np.float32),
            "colours": np.asarray(((255, 0, 0),), dtype=np.uint8),
        }

    def poll(self) -> None:
        raise AssertionError("The display loop must not issue blocking LiDAR polls")


class EmptyLidarStub:
    """A sensor that returns nothing, as happens between scans and on reload."""

    def __init__(self, points: np.ndarray | None = None) -> None:
        self._points = (
            np.empty((0, 3), dtype=np.float32) if points is None else points
        )

    def stream(self) -> dict[str, np.ndarray]:
        return {
            "pointCloud": self._points,
            "colours": np.empty((len(self._points), 3), dtype=np.uint8),
        }

    def poll(self) -> None:
        raise AssertionError("The display loop must not issue blocking LiDAR polls")


def _armed_worker(
    sensors: list[object], bng: object | None = None
) -> tuple[BeamNgWorker, list[BevFrame]]:
    worker = BeamNgWorker()
    frames: list[BevFrame] = []
    worker._bng = bng if bng is not None else object()  # type: ignore[assignment]
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._sensors = sensors  # type: ignore[assignment]
    worker._geometry = VehicleGeometry(
        ground_z_vehicle=-0.5,
        left_m=1.0,
        right_m=1.0,
        front_m=2.0,
        rear_m=2.0,
        height_m=1.5,
        mounts={},
    )
    worker._palette = SemanticPalette.from_annotations({"STREET": (255, 0, 0)})
    worker._frame_times = deque(maxlen=60)
    worker.frame_ready.connect(frames.append)
    return worker, frames


def test_lidar_display_loop_reads_shared_memory_streams() -> None:
    sensors = [StreamingLidarStub() for _ in range(4)]
    worker, frames = _armed_worker(sensors)  # type: ignore[arg-type]

    worker._poll_once()

    assert [sensor.stream_calls for sensor in sensors] == [1, 1, 1, 1]
    assert len(frames) == 1
    assert frames[0].raw_point_count == 4


@pytest.mark.parametrize("sensor_count", (1, 3, 4, 5, 6))
def test_the_display_loop_does_not_depend_on_how_many_sensors_there_are(
    sensor_count: int,
) -> None:
    """
    Regression pin for a silently blank display.

    `_poll_once` and both control systems gated on a literal
    `len(self._sensors) != 4`. Adding a fifth mount made every one of those
    guards reject: the display loop returned before its first statement, so no
    frame was emitted, no poll ever failed, the failure budget was never
    touched and NOTHING was logged -- the badge read STREAMING over a frozen,
    empty view. The sensor count is not a precondition for anything here.
    """
    sensors = [StreamingLidarStub() for _ in range(sensor_count)]
    worker, frames = _armed_worker(sensors)  # type: ignore[arg-type]

    worker._poll_once()

    assert [sensor.stream_calls for sensor in sensors] == [1] * sensor_count
    assert len(frames) == 1
    assert frames[0].raw_point_count == sensor_count

    # The same guard gates both control systems, so arming has to survive it
    # too -- that is what reported as "AEB will not arm after attach".
    armed: list[bool] = []
    worker.aeb_changed.connect(armed.append)
    worker.set_aeb(True)
    assert armed == [True]


def test_poll_emits_a_matching_perception_snapshot() -> None:
    worker, _ = _armed_worker([StreamingLidarStub() for _ in range(4)])
    snapshots: list[PerceptionSnapshot] = []
    worker.perception_ready.connect(snapshots.append)

    worker._poll_once()

    assert len(snapshots) == 1
    assert snapshots[0].points_world.shape == (4, 3)
    assert snapshots[0].semantic_groups.shape == (4,)


def test_emits_a_frame_when_every_sensor_returns_no_points() -> None:
    """Regression: the display froze instead of showing an empty scene."""
    worker, frames = _armed_worker([EmptyLidarStub() for _ in range(4)])

    worker._poll_once()

    assert len(frames) == 1
    frame = frames[0]
    assert frame.raw_point_count == 0
    assert frame.road_points.shape == (0, 2)
    assert frame.obstacle_points.shape == (0, 2)
    assert frame.vehicle_geometry is not None


def test_emits_a_frame_when_every_return_is_non_finite() -> None:
    nan_points = np.full((3, 3), np.nan, dtype=np.float32)
    worker, frames = _armed_worker(
        [EmptyLidarStub(nan_points) for _ in range(4)]
    )

    worker._poll_once()

    assert len(frames) == 1
    assert frames[0].raw_point_count == 0


def test_speed_keeps_updating_while_no_points_are_returned() -> None:
    worker, frames = _armed_worker([EmptyLidarStub() for _ in range(4)])
    worker._vehicle.state["vel"] = (0.0, 10.0, 0.0)  # type: ignore[union-attr]

    worker._poll_once()

    assert frames[0].speed_mps == pytest.approx(10.0)


def test_acquisition_fps_reports_zero_when_no_points_are_returned() -> None:
    worker, frames = _armed_worker([EmptyLidarStub() for _ in range(4)])

    worker._poll_once()
    worker._poll_once()

    assert len(frames) == 2
    assert frames[-1].acquisition_fps == pytest.approx(0.0)


class VehiclesApiStub:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.state_requests: list[tuple[str, ...]] = []

    def get_current_info(self, include_config: bool = False) -> dict[str, dict]:
        del include_config
        if self.fail:
            raise RuntimeError("registry unavailable")
        return {
            "ego": {"model": "etk800"},
            "traffic_1": {"model": "pickup"},
            "traffic_2": {"model": "citybus"},
        }

    def get_states(self, vehicles: tuple[str, ...]) -> dict[str, dict]:
        self.state_requests.append(tuple(vehicles))
        return {
            actor_id: {
                "pos": (float(index), 10.0 + index, 0.0),
                "dir": (0.0, 1.0, 0.0),
                "up": (0.0, 0.0, 1.0),
                "vel": (0.0, 2.0, 0.0),
            }
            for index, actor_id in enumerate(vehicles)
        }


def test_actor_states_are_batched_and_player_is_excluded() -> None:
    vehicles = VehiclesApiStub()
    worker = BeamNgWorker()
    worker._bng = SimpleNamespace(vehicles=vehicles)  # type: ignore[assignment]
    worker._player_vid = "ego"

    observations = worker._poll_actor_observations(10.0)

    assert vehicles.state_requests == [("traffic_1", "traffic_2")]
    assert {actor.actor_id for actor in observations} == {
        "traffic_1",
        "traffic_2",
    }


def test_actor_query_failure_is_local_to_visual_enrichment() -> None:
    worker = BeamNgWorker()
    worker._bng = SimpleNamespace(  # type: ignore[assignment]
        vehicles=VehiclesApiStub(fail=True)
    )
    worker._player_vid = "ego"

    assert worker._poll_actor_observations(10.0) == ()


def test_launch_is_refused_when_the_bridge_is_already_reachable(monkeypatch) -> None:
    """Regression: Launch spawned a second BeamNG.tech fighting for the port."""
    spawned: list[bool] = []
    monkeypatch.setattr(worker_module, "bridge_is_reachable", lambda: True)
    monkeypatch.setattr(
        worker_module,
        "start_beamng_process",
        lambda: spawned.append(True),
    )
    worker = BeamNgWorker()
    ready: list[bool] = []
    worker.launch_ready.connect(lambda: ready.append(True))

    worker.launch_beamng()

    assert spawned == []
    assert ready == [True]


def test_bridge_loss_clears_the_connection_and_stops_the_view() -> None:
    worker, _ = _armed_worker([EmptyLidarStub() for _ in range(4)])
    stopped: list[bool] = []
    states: list[str] = []
    worker.sensors_stopped.connect(lambda: stopped.append(True))
    worker.status_changed.connect(lambda state, _msg: states.append(state))

    worker.handle_bridge_lost()

    assert worker._bng is None
    assert stopped == [True]
    assert states == ["OFFLINE"]


# --- self-driving ------------------------------------------------------------


def _is_released(control: dict[str, float]) -> bool:
    """Every actuator the app holds is handed back to the driver."""
    return all(
        control.get(name, 0.0) == 0.0
        for name in ("steering", "throttle", "brake", "parkingbrake")
    )


def _driving_worker(
    bng: object | None = None, gear: object = "P"
) -> tuple[BeamNgWorker, list[BevFrame], VehicleStub]:
    worker, frames = _armed_worker(
        [StreamingLidarStub() for _ in range(4)], bng=bng
    )
    worker._vehicle = VehicleStub(gear=gear)  # type: ignore[assignment]
    worker.set_self_driving(True)
    return worker, frames, worker._vehicle  # type: ignore[return-value]


# --- gearbox handling --------------------------------------------------------


def test_an_automatic_gearbox_is_driven_in_drive_not_park() -> None:
    """
    Regression: forward drive commanded gear 1, which is PARK on every
    automatic-family gearbox. The car applied full throttle and never moved,
    then the stall detector reversed it.
    """
    worker, _, vehicle = _driving_worker(gear="P")

    worker._poll_once()

    assert vehicle.controls[0]["gear"] == 2


def test_a_manual_gearbox_is_driven_in_first() -> None:
    worker, _, vehicle = _driving_worker(gear=0)

    worker._poll_once()

    assert vehicle.controls[0]["gear"] == 1


def test_every_control_message_releases_the_parking_brake() -> None:
    """beamngpy omits None arguments, so an unsent parking brake is never cleared."""
    worker, _, vehicle = _driving_worker()

    worker._poll_once()

    assert vehicle.controls[0]["parkingbrake"] == 0.0


def _tick(worker: BeamNgWorker) -> None:
    """One poll that is guaranteed to actuate, past the CONTROL_INTERVAL_MS gate."""
    worker._last_control_at = 0.0
    worker._poll_once()


def test_the_gear_is_not_resent_once_the_box_is_already_in_it() -> None:
    """shiftToGearIndex has side effects; running it 25x/s churns the clutch."""
    worker, _, vehicle = _driving_worker(gear="P")

    _tick(worker)
    vehicle.sensors["electrics"]["gear"] = "D"
    _tick(worker)

    assert "gear" in vehicle.controls[0]
    assert "gear" not in vehicle.controls[1]


def test_the_gear_is_resent_when_the_box_leaves_drive() -> None:
    worker, _, vehicle = _driving_worker(gear="P")

    _tick(worker)
    vehicle.sensors["electrics"]["gear"] = "D"
    _tick(worker)
    vehicle.sensors["electrics"]["gear"] = "N"
    _tick(worker)

    assert vehicle.controls[2]["gear"] == 2


def test_electrics_is_polled_alongside_state() -> None:
    worker, _, vehicle = _driving_worker()

    worker._poll_once()

    assert set(vehicle.polled_names) == {"state", "electrics"}


def test_an_unreadable_gearbox_still_drives() -> None:
    """Losing electrics must not park the car; default to the automatic index."""
    worker, _, vehicle = _driving_worker()
    vehicle.sensors.pop("electrics")

    worker._poll_once()

    assert vehicle.controls[0]["gear"] == 2
    assert vehicle.controls[0]["throttle"] > 0.0


def test_disengaging_releases_the_parking_brake() -> None:
    worker, _, vehicle = _driving_worker()
    worker._poll_once()

    worker.set_self_driving(False)

    assert vehicle.controls[-1]["parkingbrake"] == 0.0


def test_no_control_is_sent_while_self_driving_is_off() -> None:
    worker, frames = _armed_worker([StreamingLidarStub() for _ in range(4)])

    worker._poll_once()

    assert worker._vehicle.controls == []  # type: ignore[union-attr]
    assert frames[0].plan is None


def test_engaging_self_driving_actuates_the_vehicle() -> None:
    worker, frames, vehicle = _driving_worker()

    worker._poll_once()

    assert frames[0].plan is not None
    assert frames[0].plan.command.mode == "DRIVING"
    assert len(vehicle.controls) == 1
    assert vehicle.controls[0]["throttle"] > 0.0
    assert vehicle.shift_modes == ["realistic_automatic"]


def test_it_brakes_rather_than_driving_blind_when_no_returns_arrive() -> None:
    """An empty cloud looks exactly like a clear road. It must not be trusted."""
    worker, frames = _armed_worker([EmptyLidarStub() for _ in range(4)])
    worker.set_self_driving(True)

    worker._poll_once()

    plan = frames[0].plan
    assert plan is not None
    assert plan.command.mode == "BLOCKED"
    assert plan.command.throttle == 0.0
    assert plan.command.brake > 0.0


def test_engaging_is_refused_without_an_attached_vehicle() -> None:
    worker = BeamNgWorker()
    changed: list[bool] = []
    worker.self_driving_changed.connect(changed.append)

    worker.set_self_driving(True)

    assert changed == [False]
    assert worker._self_driving is False


def test_disengaging_zeroes_the_controls() -> None:
    worker, _, vehicle = _driving_worker()
    worker._poll_once()

    worker.set_self_driving(False)

    assert _is_released(vehicle.controls[-1])


def test_stopping_sensors_zeroes_the_controls() -> None:
    """The car must not drive away with the app detached."""
    worker, _, vehicle = _driving_worker()
    worker._poll_once()

    worker.stop_sensors()

    assert _is_released(vehicle.controls[-1])
    assert worker._self_driving is False


def test_bridge_loss_zeroes_the_controls() -> None:
    worker, _, vehicle = _driving_worker()
    worker._poll_once()

    worker.handle_bridge_lost()

    assert _is_released(vehicle.controls[-1])
    assert worker._self_driving is False


def test_shutdown_zeroes_the_controls() -> None:
    worker, _, vehicle = _driving_worker()
    worker._poll_once()

    worker.shutdown()

    assert _is_released(vehicle.controls[-1])
    assert worker._self_driving is False


def test_a_planner_failure_disengages_instead_of_dropping_the_connection(
    monkeypatch,
) -> None:
    """A controller bug must not look like a lost bridge and tear everything down."""

    def explode(*args, **kwargs):
        raise RuntimeError("planner bug")

    worker, frames, _ = _driving_worker()
    monkeypatch.setattr(worker_module, "plan_arc", explode)

    worker._poll_once()

    assert worker._self_driving is False
    assert worker._bng is not None
    assert worker._poll_failures == 0
    assert len(frames) == 1
    assert frames[0].plan is None


def test_the_lookahead_scales_with_speed() -> None:
    """~3 s of travel at any speed, clamped into the configured window."""
    slow, slow_frames, _ = _driving_worker()
    slow._vehicle.state["vel"] = (0.0, 3.0, 0.0)  # type: ignore[union-attr]
    slow._poll_once()

    fast, fast_frames, _ = _driving_worker()
    fast._vehicle.state["vel"] = (0.0, 11.0, 0.0)  # type: ignore[union-attr]
    fast._poll_once()

    assert slow_frames[0].plan.arc.lookahead_m == pytest.approx(LOOKAHEAD_MIN_M)
    assert fast_frames[0].plan.arc.lookahead_m == pytest.approx(LOOKAHEAD_MAX_M)


def test_the_controller_receives_the_vehicle_heading() -> None:
    worker, _, _ = _driving_worker()

    worker._poll_once()
    worker._poll_once()

    # dir (0, 1, 0) is a constant world heading of pi/2: after two polls the
    # controller has a yaw estimate, and a constant heading reads zero.
    assert worker._controller is not None
    assert worker._controller.measured_curvature == pytest.approx(0.0, abs=1e-9)


def test_the_navigation_route_is_not_polled_every_tick() -> None:
    """Reading the route is a blocking Lua round-trip; the display loop is 25 Hz."""
    bng = BngStub(reply=None)
    worker, _, _ = _driving_worker(bng=bng)

    worker._poll_once()
    worker._poll_once()
    worker._poll_once()

    assert len(bng.chunks) == 1


def test_a_route_ahead_becomes_a_turn_hint() -> None:
    reply = json.dumps(
        {"hasTarget": True, "path": [[-19.0, 22.0, 3.0]], "length": 40.0}
    )
    worker, frames, _ = _driving_worker(bng=BngStub(reply=reply))

    worker._poll_once()

    plan = frames[0].plan
    assert plan is not None
    assert plan.arc.nav_heading_rad is not None
    assert plan.arc.nav_heading_rad > 0.0  # the route bends left


def test_no_destination_leaves_the_planner_without_a_hint() -> None:
    worker, frames, _ = _driving_worker(
        bng=BngStub(reply=json.dumps({"hasTarget": False}))
    )

    worker._poll_once()

    plan = frames[0].plan
    assert plan is not None
    assert plan.arc.nav_heading_rad is None


# --- emergency braking -------------------------------------------------------


def _aeb_worker(
    obstacles: np.ndarray | None,
    monkeypatch,
    speed: float = 11.0,
    self_driving: bool = False,
) -> tuple[BeamNgWorker, list[BevFrame], VehicleStub]:
    """
    A worker with AEB armed and the obstacle cloud stubbed out.

    geometric_obstacles is replaced rather than fed a synthetic point cloud
    because AEB reasons in BEV metres: putting the obstacle there directly is
    what the test is actually about, and the height-band extraction has its own
    coverage in test_planner.
    """
    worker, frames = _armed_worker([StreamingLidarStub() for _ in range(4)])
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    worker._vehicle.state["vel"] = (0.0, speed, 0.0)  # type: ignore[union-attr]
    _stub_obstacles(monkeypatch, obstacles)
    if self_driving:
        worker.set_self_driving(True)
    worker.set_aeb(True)
    return worker, frames, worker._vehicle  # type: ignore[return-value]


def _stub_obstacles(monkeypatch, obstacles: np.ndarray | None) -> None:
    """Hand the same cloud to both floors, since these tests bypass the bands."""
    empty = np.empty((0, 2))
    cloud = empty if obstacles is None else obstacles
    monkeypatch.setattr(
        worker_module,
        "geometric_obstacle_sets",
        lambda bev, heights, ground_z, bands, **kwargs: [
            cloud for _ in bands
        ],
    )


def _wall(distance_m: float) -> np.ndarray:
    lateral = np.linspace(-1.5, 1.5, 15)
    return np.column_stack((lateral, np.full_like(lateral, distance_m)))


def _confirm(worker: BeamNgWorker) -> None:
    """
    Poll until AEB's confirmation window has elapsed.

    A threat has to persist for AEB_CONFIRM_S before the pedal moves, so a
    single tick is never enough -- that is the point of the filter, and these
    tests are about what happens once it has passed.

    `_last_aeb_at` is cleared each time so every tick is charged the nominal
    display interval: the worker takes its dt from the wall clock, which barely
    moves between two back-to-back calls in a test.
    """
    for _ in range(int(AEB_CONFIRM_S / (DISPLAY_INTERVAL_MS / 1000.0)) + 1):
        worker._last_aeb_at.clear()
        _tick(worker)


def test_arming_aeb_is_refused_without_an_attached_vehicle() -> None:
    worker = BeamNgWorker()
    changed: list[bool] = []
    worker.aeb_changed.connect(changed.append)

    worker.set_aeb(True)

    assert changed == [False]
    assert worker._aeb_enabled is False


def test_a_clear_road_leaves_the_driver_alone(monkeypatch) -> None:
    """AEB armed but not firing must not send a single control message."""
    worker, frames, vehicle = _aeb_worker(None, monkeypatch)

    _confirm(worker)

    assert vehicle.controls == []
    assert frames[-1].aeb is not None
    assert frames[-1].aeb.status == "ARMED"


def test_aeb_brakes_without_self_driving(monkeypatch) -> None:
    worker, frames, vehicle = _aeb_worker(_wall(10.0), monkeypatch)

    _confirm(worker)

    assert frames[-1].aeb.status == "BRAKING"
    assert vehicle.controls[-1]["brake"] > 0.0


def test_aeb_alone_touches_the_pedals_and_nothing_else(monkeypatch) -> None:
    """
    A human is holding the wheel, the gearstick and the handbrake. Sending any
    of them would be a takeover, not a brake assist.
    """
    worker, _, vehicle = _aeb_worker(_wall(10.0), monkeypatch)

    _confirm(worker)

    assert set(vehicle.controls[-1]) == {"throttle", "brake"}
    assert vehicle.controls[-1]["throttle"] == 0.0


def test_the_aeb_brake_is_released_exactly_once_when_the_event_ends(
    monkeypatch,
) -> None:
    """A swallowed release leaves the car braking indefinitely."""
    worker, _, vehicle = _aeb_worker(_wall(10.0), monkeypatch)
    _confirm(worker)
    braking = len(vehicle.controls)
    assert braking > 0
    assert vehicle.controls[-1]["brake"] > 0.0

    # The obstacle is gone, and enough time has passed for the minimum hold.
    _stub_obstacles(monkeypatch, None)
    worker._aeb._engaged_for = 10.0
    _tick(worker)
    _tick(worker)
    _tick(worker)

    # One release, and then silence -- the driver has the car back.
    assert len(vehicle.controls) == braking + 1
    assert vehicle.controls[-1]["brake"] == 0.0


def test_aeb_overrides_the_self_driving_throttle(monkeypatch) -> None:
    worker, frames, vehicle = _aeb_worker(
        _wall(9.0), monkeypatch, self_driving=True
    )

    _confirm(worker)

    assert frames[-1].plan is not None
    assert frames[-1].aeb.status == "BRAKING"
    assert vehicle.controls[-1]["throttle"] == 0.0
    assert vehicle.controls[-1]["brake"] >= frames[-1].aeb.brake


def test_aeb_never_softens_the_self_driving_brake(monkeypatch) -> None:
    """The two brake demands combine by max, so neither can undercut the other."""
    worker, frames, vehicle = _aeb_worker(
        _wall(9.0), monkeypatch, self_driving=True
    )

    _confirm(worker)

    plan = frames[-1].plan
    assert plan is not None
    assert vehicle.controls[-1]["brake"] == max(
        plan.command.brake, frames[-1].aeb.brake
    )


def test_disarming_aeb_releases_the_brake(monkeypatch) -> None:
    worker, _, vehicle = _aeb_worker(_wall(10.0), monkeypatch)
    _confirm(worker)

    worker.set_aeb(False)

    assert vehicle.controls[-1]["brake"] == 0.0
    assert vehicle.controls[-1]["throttle"] == 0.0
    assert worker._aeb_enabled is False


def test_stopping_sensors_releases_the_aeb_brake(monkeypatch) -> None:
    """The car must not be left standing on a brake this app applied."""
    worker, _, vehicle = _aeb_worker(_wall(10.0), monkeypatch)
    _confirm(worker)

    worker.stop_sensors()

    assert vehicle.controls[-1]["brake"] == 0.0
    assert worker._aeb_enabled is False


def test_bridge_loss_releases_the_aeb_brake(monkeypatch) -> None:
    worker, _, vehicle = _aeb_worker(_wall(10.0), monkeypatch)
    _confirm(worker)

    worker.handle_bridge_lost()

    assert vehicle.controls[-1]["brake"] == 0.0
    assert worker._aeb_enabled is False


def test_an_aeb_failure_disarms_instead_of_dropping_the_connection(
    monkeypatch,
) -> None:
    worker, frames, _ = _aeb_worker(None, monkeypatch)

    def explode(*args, **kwargs):
        raise RuntimeError("aeb bug")

    monkeypatch.setattr(worker._aeb, "step", explode)

    worker._poll_once()

    assert worker._aeb_enabled is False
    assert worker._bng is not None
    assert worker._poll_failures == 0
    assert len(frames) == 1
    assert frames[0].aeb is None


def test_an_aeb_failure_leaves_self_driving_running(monkeypatch) -> None:
    """The two features are separately isolated; one fault must not stop both."""
    worker, frames, _ = _aeb_worker(None, monkeypatch, self_driving=True)

    def explode(*args, **kwargs):
        raise RuntimeError("aeb bug")

    monkeypatch.setattr(worker._aeb, "step", explode)

    worker._poll_once()

    assert worker._aeb_enabled is False
    assert worker._self_driving is True
    assert frames[0].plan is not None


def test_a_planner_failure_leaves_aeb_running(monkeypatch) -> None:
    worker, frames, _ = _aeb_worker(None, monkeypatch, self_driving=True)

    def explode(*args, **kwargs):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(worker_module, "plan_arc", explode)

    worker._poll_once()

    assert worker._self_driving is False
    assert worker._aeb_enabled is True
    assert frames[0].aeb is not None


def test_aeb_off_leaves_no_trace_on_the_frame() -> None:
    worker, frames = _armed_worker([StreamingLidarStub() for _ in range(4)])

    worker._poll_once()

    assert frames[0].aeb is None


def _reversing_worker(
    obstacles: np.ndarray | None, monkeypatch, speed: float = 3.0
) -> tuple[BeamNgWorker, list[BevFrame], VehicleStub]:
    """Both systems armed, the car reversing at ``speed`` m/s."""
    worker, frames = _armed_worker([StreamingLidarStub() for _ in range(4)])
    worker._vehicle = VehicleStub()  # type: ignore[assignment]
    # Negative along the forward axis: reversing.
    worker._vehicle.state["vel"] = (0.0, -speed, 0.0)  # type: ignore[union-attr]
    _stub_obstacles(monkeypatch, obstacles)
    worker.set_aeb(True)
    worker.set_rear_aeb(True)
    return worker, frames, worker._vehicle  # type: ignore[return-value]


def _wall_behind(distance_m: float) -> np.ndarray:
    lateral = np.linspace(-1.5, 1.5, 15)
    return np.column_stack((lateral, np.full_like(lateral, -distance_m)))


def test_each_aeb_system_gets_its_own_tick_clock(monkeypatch) -> None:
    """
    Regression, reported as "reverse AEB says standby and never brakes".

    Both systems step inside the same tick, and they shared one `_last_aeb_at`.
    Whichever ran second therefore measured a dt of microseconds: its
    confirmation window advanced 1 ms per tick instead of 40, so the rear brake
    needed 4.8 s of continuous threat before it could fire and never did, and
    its yaw rate came out 40x inflated the moment the car turned.
    """
    # Inside the rear scan horizon (~8 m at this speed) but outside the
    # trigger (~3.6 m), so the threat is SEEN every tick without firing.
    worker, _, _ = _reversing_worker(_wall_behind(6.0), monkeypatch)

    ticks = 6
    for _ in range(ticks):
        worker._last_aeb_at.clear()
        _tick(worker)

    nominal = DISPLAY_INTERVAL_MS / 1000.0
    assert worker._rear_aeb._seen_for == pytest.approx(ticks * nominal, rel=0.2)
    assert set(worker._last_aeb_at) == {"AEB", "REAR AEB"}


def test_reversing_into_a_wall_brakes(monkeypatch) -> None:
    worker, frames, vehicle = _reversing_worker(_wall_behind(3.0), monkeypatch)

    _confirm(worker)

    assert frames[-1].rear_aeb.status == "BRAKING"
    assert frames[-1].rear_aeb.rearward is True
    # ...while the forward system correctly sees nothing to do.
    assert frames[-1].aeb.status == "STANDBY"
    assert vehicle.controls[-1]["brake"] > 0.0


def test_reversing_past_a_wall_that_is_still_far_does_not_brake(
    monkeypatch,
) -> None:
    worker, frames, vehicle = _reversing_worker(_wall_behind(6.0), monkeypatch)

    _confirm(worker)

    assert frames[-1].rear_aeb.status == "ARMED"
    assert vehicle.controls == []


def test_both_systems_arm_themselves_once_sensors_are_live() -> None:
    """
    They are protective rather than behavioural, so defaulting them off would
    mean the safety net is missing exactly when nobody thought about it.
    """
    worker, _ = _armed_worker([StreamingLidarStub() for _ in range(4)])
    armed: list[tuple[str, bool]] = []
    worker.aeb_changed.connect(lambda on: armed.append(("aeb", on)))
    worker.rear_aeb_changed.connect(lambda on: armed.append(("rear", on)))

    worker._set_aeb(True, rearward=False)
    worker._set_aeb(True, rearward=True)

    assert armed == [("aeb", True), ("rear", True)]
    assert worker._aeb_enabled and worker._rear_aeb_enabled


def test_disarming_one_system_leaves_the_other_armed(monkeypatch) -> None:
    worker, _, _ = _reversing_worker(None, monkeypatch)

    worker.set_rear_aeb(False)

    assert worker._aeb_enabled is True
    assert worker._rear_aeb_enabled is False


def test_teardown_disarms_both(monkeypatch) -> None:
    worker, _, _ = _reversing_worker(None, monkeypatch)

    worker.stop_sensors()

    assert worker._aeb_enabled is False
    assert worker._rear_aeb_enabled is False


# --- plant diagnostics -------------------------------------------------------
#
# These measure the attached car and change nothing about what it does. Every
# braking figure in config is a property of ONE vehicle, and applying them to a
# heavier one is a question that needs a number.


def _brake_trace(
    worker: BeamNgWorker,
    from_speed: float,
    decel: float,
    dt: float = 0.04,
) -> None:
    """Feed the manual watcher a stop at a constant rate, one tick at a time."""
    now = 0.0
    speed = from_speed
    worker._last_forward_speed = speed
    worker._watch_manual_braking(now)
    while abs(speed) > 0.0:
        speed = (
            max(0.0, speed - decel * dt)
            if speed > 0.0
            else min(0.0, speed + decel * dt)
        )
        now += dt
        worker._last_forward_speed = speed
        worker._watch_manual_braking(now)


def _measurements(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Brake measure:")
    ]


def test_a_hard_manual_stop_is_measured_without_aeb_ever_firing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    How a vehicle other than the one in the config tables gets measured at all:
    drive it, brake hard, read the line. It must not depend on AEB being armed,
    because switching AEB off is the first thing anyone does when it brakes for
    nothing -- which is exactly when the plant needs measuring.
    """
    worker = BeamNgWorker()
    worker._vehicle_model = "pickup"

    with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
        _brake_trace(worker, 20.0, 7.0)

    lines = _measurements(caplog)
    assert len(lines) == 1
    assert lines[0].startswith("Brake measure: MANUAL stop on 'pickup'")
    # The rate the car achieved, beside the one the config assumed.
    assert "achieved 6.9" in lines[0]
    assert f"{FORWARD.braking_decel_mps2:.1f} m/s^2" in lines[0]
    assert worker._manual_event is None


def test_an_aeb_stop_is_not_also_filed_as_a_manual_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The two watchers see the same speed trace. The manual one starts on
    deceleration alone, which does not cross its threshold until a tick or two
    AFTER the pedal goes down -- so checking only where an AEB event begins let
    the same stop be reported twice, once each way.
    """
    worker = BeamNgWorker()
    worker._aeb_events["AEB"] = BrakeEvent(FORWARD)
    worker._aeb_events["AEB"].start(20.0, 0.0, "AEB")

    with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
        _brake_trace(worker, 20.0, 10.0)

    assert _measurements(caplog) == []
    assert worker._manual_event is None


def test_lifting_off_is_not_a_braking_measurement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Engine drag and a trailing throttle are nowhere near the threshold."""
    worker = BeamNgWorker()

    with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
        _brake_trace(worker, 20.0, 1.0)

    assert _measurements(caplog) == []


def test_a_dab_at_the_pedal_is_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hard, but over in a couple of ticks and worth 1 km/h. Nothing to learn."""
    worker = BeamNgWorker()

    now = 0.0
    for speed in (20.0, 19.6, 19.4, 19.4, 19.4):
        worker._last_forward_speed = speed
        with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
            worker._watch_manual_braking(now)
        now += 0.04

    assert _measurements(caplog) == []


def test_a_reverse_stop_is_measured_against_the_reverse_plant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Reversing throws the load onto the rear axle and its smaller brakes, so the
    two directions are different plants -- and a measurement filed against the
    wrong one would read as a fault that is not there.
    """
    worker = BeamNgWorker()

    with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
        _brake_trace(worker, -8.0, 6.5)

    lines = _measurements(caplog)
    assert len(lines) == 1
    assert lines[0].startswith("Brake measure: MANUAL REVERSE stop")
    assert f"{REVERSE.braking_decel_mps2:.1f} m/s^2" in lines[0]
    assert f"{FORWARD.braking_decel_mps2:.1f} m/s^2" not in lines[0]


def test_a_firing_explains_itself_once(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    `AEB: BRAKING` names a distance, and a distance cannot say whether what
    blocked the corridor was a wall or the road surface arriving in the height
    band. One evidence line per firing does -- and only per firing, because it
    costs a ground estimate over the whole cloud.
    """
    worker, frames, _ = _aeb_worker(_wall(10.0), monkeypatch)

    with caplog.at_level("INFO", logger=worker_module.LOGGER.name):
        _confirm(worker)
        # ...and keep braking. The line must not repeat every tick.
        _confirm(worker)

    evidence = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("AEB evidence:")
    ]
    assert frames[-1].aeb.status == "BRAKING"
    assert len(evidence) == 1
    assert "AEB at 10.0 m" in evidence[0]
    assert not worker._pending_evidence
