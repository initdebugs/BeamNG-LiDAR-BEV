"""
Closed-loop driving: the real planner and controller on a kinematic bicycle
plant, over a straight kerbed road and over 90-degree corners.

The straight road is the regression pin for the fan-quantization weave: at a
30 m lookahead one uniform fan step was 3.75 m of lateral offset, so the
immediate family could only "do nothing" or "swerve". The deferred families
then won every keep-right correction ("hold course, bend later"), the
correction never started, and the car wove kerb to kerb until it blocked.

The corners are the pin for "it gets stuck often", and they are the only test
here that exercises obstacle selection, corner-entry braking, the steering
ceiling and the speed loop against each other. Each of those had a defect that
a straight road cannot reach.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from beamng_lidar_bev.aeb import mirror_points, mirrored
from beamng_lidar_bev.config import (
    MAX_LATERAL_ACCEL_MPS2,
    MIN_TURN_RADIUS_M,
    REVERSE_COST_SMOOTHNESS,
    REVERSE_MIN_CLEARANCE_M,
    REVERSE_REQUIRED_FREE_M,
    ROUTE_ARRIVAL_LATCH_M,
    STEERING_SIGN,
)
from beamng_lidar_bev.controller import DrivingController
from beamng_lidar_bev.models import RouteHint, VehicleGeometry
from beamng_lidar_bev.planner import (
    geometric_obstacles,
    plan_arc,
    rear_free_distance,
)
from beamng_lidar_bev.route_model import build_route_path

GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.5,
    left_m=0.9,
    right_m=0.9,
    front_m=2.0,
    rear_m=2.4,
    height_m=1.5,
    mounts={},
)
_K_MAX = 1.0 / MIN_TURN_RADIUS_M
_DT = 0.04
_HALF_ROAD = 2.6  # a 5.2 m road: kerb clip on weave, comfortable otherwise


def _kerbs(length_m: float) -> np.ndarray:
    ys = np.arange(0.0, length_m, 0.5)
    return np.concatenate(
        [
            np.column_stack((np.full_like(ys, -_HALF_ROAD), ys)),
            np.column_stack((np.full_like(ys, _HALF_ROAD), ys)),
        ]
    )


def _drive_straight_road(seconds: float) -> tuple[np.ndarray, list[str]]:
    """Returns the lateral-position trace and the modes seen after settling."""
    kerbs = _kerbs(1200.0)
    controller = DrivingController()
    pos = np.zeros(2)
    psi = np.pi / 2  # facing along the road, on the centreline
    speed = 0.0
    laterals: list[float] = []
    modes: list[str] = []
    for tick in range(int(seconds / _DT)):
        forward = np.asarray((np.cos(psi), np.sin(psi)))
        right = np.asarray((np.sin(psi), -np.cos(psi)))
        rel = kerbs - pos
        bev = np.column_stack((rel @ right, rel @ forward))
        obstacles = bev[np.hypot(bev[:, 0], bev[:, 1]) <= 35.0]

        plan = plan_arc(
            obstacles,
            GEOMETRY,
            previous_curvature=controller.current_curvature,
            lookahead_m=min(30.0, max(16.0, 2.8 * abs(speed))),
        )
        command = controller.step(
            plan,
            speed,
            _DT,
            rear_free_distance_m=rear_free_distance(obstacles, GEOMETRY),
            reported_gear="D",
            heading_rad=psi,
        )

        # Ideal plant: the commanded curvature is driven exactly.
        curvature = (command.steering / STEERING_SIGN) * _K_MAX
        speed = max(
            0.0, speed + (command.throttle * 3.5 - command.brake * 6.0 - 0.1) * _DT
        )
        psi += speed * curvature * _DT
        pos = pos + np.asarray((np.cos(psi), np.sin(psi))) * speed * _DT

        if tick * _DT > 10.0:  # past the pull-away transient
            laterals.append(float(pos[0]))
            modes.append(command.mode)
    return np.asarray(laterals), modes


def test_a_straight_kerbed_road_is_driven_without_weaving() -> None:
    laterals, modes = _drive_straight_road(50.0)

    # The keep-right line sits at +1.25 (right kerb minus half width and
    # margin); a healthy car settles near it and never lunges kerb-ward.
    assert set(modes) == {"DRIVING"}
    assert np.abs(laterals).max() < 1.9  # never within half a metre of a kerb
    settled = laterals[len(laterals) // 2 :]
    # Bounded regulation around wherever it settled, not kerb-to-kerb weave.
    assert float(settled.std()) < 0.35


# --- corners -----------------------------------------------------------------
#
# The straight-road case above cannot reach the failure that actually stopped
# the car in practice. Driving a corner exercises the whole stack together --
# obstacle selection, the arc fan's deferred families, corner-entry braking,
# the steering ceiling and the speed loop -- and every one of them had a defect
# that only showed up here. See the individual pins in test_planner.py and
# test_controller.py; this is the integration guard over all of them.

_CORNER_HALF_ROAD = 3.0
_KERB_HEIGHT_M = 0.15
_VERGE_HEIGHT_M = 0.16


def _corner_road(
    radius_m: float, straight_m: float = 60.0, total_m: float = 700.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    A straight, then a 90-degree constant-radius corner, then straight again.

    Returns (world points, heights above the ground plane) including the ROAD
    SURFACE, not just the kerbs: `ground_rise` estimates the local ground from
    the returns, and fed kerbs alone it correctly concludes the kerb tops are
    the ground and the kerbs stop being obstacles.

    The heading ramp is clipped at 90 degrees. Unclipped it coils the road into
    a spiral that closes on itself inside the planning horizon, and a planner
    that blocks there is right to.
    """
    s = np.arange(0.0, total_m, 0.25)
    psi = np.clip((s - straight_m) / radius_m, 0.0, np.pi / 2)
    cx = np.cumsum(np.sin(psi) * 0.25)
    cy = np.cumsum(np.cos(psi) * 0.25)
    nx, ny = np.cos(psi), -np.sin(psi)

    points: list[np.ndarray] = []
    heights: list[np.ndarray] = []

    def add(offset: float, height: float) -> None:
        points.append(np.column_stack((cx + nx * offset, cy + ny * offset)))
        heights.append(np.full(len(cx), height))

    for offset in np.arange(-_CORNER_HALF_ROAD + 0.15, _CORNER_HALF_ROAD, 0.3):
        add(float(offset), 0.0)
    for side in (-1.0, 1.0):
        add(_CORNER_HALF_ROAD * side, _KERB_HEIGHT_M)
        for extra in (0.4, 0.9, 1.5):
            add((_CORNER_HALF_ROAD + extra) * side, _VERGE_HEIGHT_M)
    return np.vstack(points), np.concatenate(heights)


def _drive_corner(radius_m: float, seconds: float = 30.0) -> dict[str, float]:
    world, world_heights = _corner_road(radius_m)
    controller = DrivingController()
    pos = np.zeros(2)
    psi = np.pi / 2
    speed = 0.0
    distance = 0.0
    peak_lateral = 0.0
    closest = np.inf
    modes: set[str] = set()

    for _ in range(int(seconds / _DT)):
        forward = np.asarray((np.cos(psi), np.sin(psi)))
        right = np.asarray((np.sin(psi), -np.cos(psi)))
        # Coarse box first: the world is 70k points and the projection is two
        # dot products over all of them, which dominates the tick otherwise.
        box = (np.abs(world[:, 0] - pos[0]) <= 36.0) & (
            np.abs(world[:, 1] - pos[1]) <= 36.0
        )
        rel = world[box] - pos
        bev = np.column_stack((rel @ right, rel @ forward)).astype(np.float32)
        in_range = np.hypot(bev[:, 0], bev[:, 1]) <= 35.0
        obstacles = geometric_obstacles(
            bev[in_range],
            (GEOMETRY.ground_z_vehicle + world_heights[box][in_range]).astype(
                np.float32
            ),
            GEOMETRY.ground_z_vehicle,
        )

        plan = plan_arc(
            obstacles,
            GEOMETRY,
            previous_curvature=controller.current_curvature,
            lookahead_m=min(30.0, max(16.0, 2.8 * abs(speed))),
        )
        command = controller.step(
            plan,
            speed,
            _DT,
            rear_free_distance_m=rear_free_distance(obstacles, GEOMETRY),
            reported_gear="D",
            heading_rad=psi,
        )

        curvature = (command.steering / STEERING_SIGN) * _K_MAX / max(
            controller.steering_gain, 1e-6
        )
        speed = max(
            0.0,
            speed + (command.throttle * 3.5 - command.brake * 6.0 - 0.25) * _DT,
        )
        psi += speed * curvature * _DT
        pos = pos + np.asarray((np.cos(psi), np.sin(psi))) * speed * _DT

        distance += speed * _DT
        modes.add(command.mode)
        peak_lateral = max(peak_lateral, speed**2 * abs(curvature))
        if len(obstacles):
            closest = min(
                closest,
                float(np.hypot(obstacles[:, 0], obstacles[:, 1]).min()),
            )

    return {
        "distance_m": distance,
        "peak_lateral": peak_lateral,
        "closest_m": closest,
        "modes": modes,
    }


# --- steered reverse recovery ------------------------------------------------


def _hline(x_from: float, x_to: float, y: float) -> np.ndarray:
    xs = np.arange(x_from, x_to, 0.1)
    return np.column_stack((xs, np.full_like(xs, y)))


def _drive_pocket(seconds: float) -> dict[str, object]:
    """
    A dead end: wall ahead, straight-back and behind-left blocked, open
    behind-right and ahead-left. The escape REQUIRES steering in reverse --
    the tail swings right into the open, the nose rotates left toward the
    gap, and the forward re-plan takes it from there.
    """
    # Wall ahead covering left and centre (open front-right); a parked car
    # half a lane right, dead behind (straight-back room falls short of the
    # 6 m reverse); open behind-left for the diagonal out.
    parked = [
        _hline(0.3, 3.4, y) for y in (-4.5, -5.0, -5.5, -6.0)
    ]
    world = np.concatenate(
        (
            _hline(-12.0, 3.0, 3.5),
            _hline(-12.0, 3.0, 3.8),
            *parked,
        )
    )
    controller = DrivingController()
    mirror_geometry = mirrored(GEOMETRY)
    pos = np.zeros(2)
    psi = np.pi / 2
    speed = 0.0  # signed: reversing reads negative
    modes: list[str] = []
    reverse_steer = 0.0
    straight_back = np.inf

    for tick in range(int(seconds / _DT)):
        forward = np.asarray((np.cos(psi), np.sin(psi)))
        right = np.asarray((np.sin(psi), -np.cos(psi)))
        rel = world - pos
        bev = np.column_stack((rel @ right, rel @ forward)).astype(np.float32)
        obstacles = bev[np.hypot(bev[:, 0], bev[:, 1]) <= 35.0]

        plan = plan_arc(
            obstacles,
            GEOMETRY,
            previous_curvature=controller.current_curvature,
            lookahead_m=min(30.0, max(16.0, 2.8 * abs(speed))),
        )
        rear_free = rear_free_distance(obstacles, GEOMETRY)
        if tick == 0:
            straight_back = rear_free
        # The worker's steered-reverse wiring, mirrored exactly.
        reverse_arc = None
        if controller.mode == "REVERSING" or (
            controller.mode == "BLOCKED" and abs(speed) < 0.3
        ):
            reverse_arc = plan_arc(
                mirror_points(obstacles),
                mirror_geometry,
                previous_curvature=-controller.current_curvature,
                lookahead_m=10.0,
                keep_right=False,
                required_free_m=REVERSE_REQUIRED_FREE_M,
                smoothness_weight=REVERSE_COST_SMOOTHNESS,
            )
            rear_free = float(reverse_arc.free_distance_m)
        command = controller.step(
            plan,
            speed,
            _DT,
            rear_free_distance_m=rear_free,
            reported_gear="D",
            heading_rad=psi,
            reverse_arc=reverse_arc,
        )

        curvature = (command.steering / STEERING_SIGN) * _K_MAX
        direction = 1.0 if command.gear > 0 else -1.0
        speed += direction * command.throttle * 3.5 * _DT
        drop = (command.brake * 6.0 + 0.1) * _DT
        if speed > 0.0:
            speed = max(0.0, speed - drop)
        elif speed < 0.0:
            speed = min(0.0, speed + drop)
        psi += speed * curvature * _DT
        pos = pos + np.asarray((np.cos(psi), np.sin(psi))) * speed * _DT

        modes.append(command.mode)
        if command.mode == "REVERSING":
            reverse_steer = max(reverse_steer, abs(command.steering))

    return {
        "modes": modes,
        "position": pos,
        "reverse_steer": reverse_steer,
        "straight_back_m": straight_back,
    }


def test_a_cornered_car_escapes_with_a_steered_reverse() -> None:
    """
    Backing straight only postpones this pocket: the rear wall sits dead
    behind, so a straight reverse ends still nose-to-wall and the cycle
    repeats forever. The only productive reverse is the diagonal -- tail
    right into the open, nose left toward the gap -- which is exactly what a
    straight-only recovery cannot do.
    """
    result = _drive_pocket(30.0)

    modes = result["modes"]
    assert result["straight_back_m"] >= REVERSE_MIN_CLEARANCE_M  # it MAY back up
    assert "REVERSING" in modes
    assert "STUCK" not in modes
    assert modes[-1] == "DRIVING"
    assert result["reverse_steer"] > 0.05  # it steered, not straight
    assert float(result["position"][1]) > 6.0  # clear of the pocket


# --- route following ---------------------------------------------------------
#
# Open ground, no kerbs: the route reference path is the ONLY thing shaping
# the drive, which is exactly the point -- lateral tracking comes from the
# cross-track/tangent terms and longitudinal shaping from the backward speed
# pass, with no obstacle geometry to hide behind.


def _route_hint(nodes: list[tuple[float, float]]) -> RouteHint:
    array = np.asarray([(x, y, 0.0) for x, y in nodes], dtype=np.float64)
    chords = float(np.sum(np.hypot(*np.diff(array[:, :2], axis=0).T)))
    return RouteHint(path_world=array, remaining_m=chords)


def _drive_route(
    hint: RouteHint, seconds: float
) -> dict[str, object]:
    controller = DrivingController()
    pos = np.zeros(2)
    psi = np.pi / 2
    speed = 0.0
    arrived_hold = False
    speeds: list[float] = []
    brakes: list[float] = []
    cross: list[float] = []
    modes: set[str] = set()
    reason = ""
    no_obstacles = np.empty((0, 2), dtype=np.float32)

    for _ in range(int(seconds / _DT)):
        state = {
            "pos": (float(pos[0]), float(pos[1]), 0.0),
            "dir": (float(np.cos(psi)), float(np.sin(psi)), 0.0),
            "up": (0.0, 0.0, 1.0),
            "vel": (
                float(np.cos(psi) * speed),
                float(np.sin(psi) * speed),
                0.0,
            ),
        }
        route_path = build_route_path(hint, state)
        plan = plan_arc(
            no_obstacles,
            GEOMETRY,
            previous_curvature=controller.current_curvature,
            lookahead_m=min(30.0, max(16.0, 2.8 * abs(speed))),
            route=route_path,
        )
        # The worker's arrival latch, mirrored: near the marker the game (and
        # a consumed polyline) clears the route, and without the latch the
        # speed law would get its full cap back right at the destination.
        if route_path is not None:
            arrived_hold = route_path.remaining_m <= ROUTE_ARRIVAL_LATCH_M
            cross.append(abs(route_path.cross_track_m))
        elif arrived_hold:
            plan = replace(plan, route_speed_limit_mps=0.0)
        command = controller.step(
            plan,
            speed,
            _DT,
            rear_free_distance_m=20.0,
            reported_gear="D",
            heading_rad=psi,
        )

        curvature = (command.steering / STEERING_SIGN) * _K_MAX
        speed = max(
            0.0, speed + (command.throttle * 3.5 - command.brake * 6.0 - 0.1) * _DT
        )
        psi += speed * curvature * _DT
        pos = pos + np.asarray((np.cos(psi), np.sin(psi))) * speed * _DT

        speeds.append(speed)
        brakes.append(command.brake)
        modes.add(command.mode)
        reason = command.reason

    return {
        "speeds": np.asarray(speeds),
        "brakes": np.asarray(brakes),
        "cross": np.asarray(cross),
        "modes": modes,
        "position": pos,
        "reason": reason,
    }


def test_a_route_corner_is_previewed_and_driven_smoothly() -> None:
    """
    A 90-degree right at R = 15 m, 60 m out, with nothing but the route to
    say so. The speed pass must ramp the car down to corner speed before the
    bend -- gently, because the target falls as the sqrt envelope, never as a
    surprise -- and the lateral terms must carry it round and out the far
    side.
    """
    nodes: list[tuple[float, float]] = [(0.0, float(y)) for y in range(0, 61, 10)]
    for angle in np.linspace(np.pi, np.pi / 2, 7)[1:]:
        nodes.append((15.0 + 15.0 * np.cos(angle), 60.0 + 15.0 * np.sin(angle)))
    nodes.extend((float(x), 75.0) for x in range(25, 320, 10))

    result = _drive_route(_route_hint(nodes), seconds=25.0)

    speeds = result["speeds"]
    settled = speeds[int(3.0 / _DT) :]
    corner_speed = np.sqrt(2.8 * 15.0)  # ~6.5 m/s at R = 15
    assert result["modes"] == {"DRIVING"}
    assert float(speeds.max()) > 9.5  # reached near the cap on the straight
    assert float(settled.min()) <= corner_speed + 1.0  # slowed for the bend
    assert float(settled.min()) >= 3.0  # but never crawled or stopped
    assert float(result["position"][0]) > 35.0  # made the exit, heading east
    # Smooth modulation: the ramped target keeps the pedal far below a slam.
    assert float(result["brakes"].max()) <= 0.55
    assert float(result["cross"].max()) < 3.0  # tracked the lane throughout


def test_the_car_stops_at_the_destination_and_stays() -> None:
    result = _drive_route(
        _route_hint([(0.0, float(y)) for y in range(0, 81, 10)]), seconds=30.0
    )

    speeds = result["speeds"]
    assert result["modes"] == {"DRIVING"}
    assert float(speeds.max()) > 9.0  # it drove there, not crept
    assert float(np.max(speeds[-int(3.0 / _DT) :])) < 0.1  # and STAYS stopped
    assert 66.0 < float(result["position"][1]) < 80.0  # short of the marker
    assert result["reason"] == "Arrived at destination"


@pytest.mark.parametrize("radius_m", (60.0, 35.0, 25.0))
def test_a_ninety_degree_corner_is_driven_without_blocking(
    radius_m: float,
) -> None:
    """
    Regression pin for "it gets stuck often".

    Every one of these blocked before: the slope cone hid the kerbs past 12 m
    so the corner was invisible until it was too late; the brake credited
    engine drag it was not getting and never reached the corner-entry speed;
    and the steering ceiling was set to a comfort figure rather than a grip
    one, so the wheel saturated and the car tracked wider every tick until the
    free distance collapsed.
    """
    result = _drive_corner(radius_m)

    assert result["modes"] == {"DRIVING"}
    # 30 s at close to the 40 km/h cap, less the pull-away transient. Before,
    # every one of these stopped inside the first 100 m.
    assert result["distance_m"] > 280.0
    assert result["closest_m"] > 1.0  # never scraping the kerb
    assert result["peak_lateral"] <= MAX_LATERAL_ACCEL_MPS2


# --- getting past something in the lane --------------------------------------


def _drive_past_a_parked_car(
    seconds: float, *, road_half: float = 4.0, car_x: float = 1.4
) -> dict[str, object]:
    """
    A stopped car in the ego's lane on an 8 m road, and enough room to pass.

    Closed loop rather than a single `plan_arc`, because committing to a pass
    is a decision the planner has to keep MAKING: the free-distance reward, the
    guidance taper and the per-tick re-plan all interact, and a planner that
    commits and then changes its mind on the next tick reads as a swerve rather
    than an overtake.
    """
    world = _kerbs_at(road_half, 400.0)
    car = np.asarray(
        [
            (x, y)
            for x in np.arange(car_x - 0.9, car_x + 0.9 + 1e-9, 0.08)
            for y in np.arange(60.0, 64.5, 0.08)
        ]
    )
    world = np.concatenate((world, car))

    controller = DrivingController()
    pos = np.zeros(2)
    psi = np.pi / 2
    speed = 0.0
    modes: list[str] = []
    closest = np.inf
    for _ in range(int(seconds / _DT)):
        forward = np.asarray((np.cos(psi), np.sin(psi)))
        right = np.asarray((np.sin(psi), -np.cos(psi)))
        rel = world - pos
        bev = np.column_stack((rel @ right, rel @ forward))
        obstacles = bev[np.hypot(bev[:, 0], bev[:, 1]) <= 35.0]

        plan = plan_arc(
            obstacles,
            GEOMETRY,
            previous_curvature=controller.current_curvature,
            lookahead_m=min(30.0, max(16.0, 2.8 * abs(speed))),
        )
        command = controller.step(
            plan,
            speed,
            _DT,
            rear_free_distance_m=rear_free_distance(obstacles, GEOMETRY),
            reported_gear="D",
            heading_rad=psi,
        )
        curvature = (command.steering / STEERING_SIGN) * _K_MAX
        speed = max(
            0.0, speed + (command.throttle * 3.5 - command.brake * 6.0 - 0.1) * _DT
        )
        psi += speed * curvature * _DT
        pos = pos + np.asarray((np.cos(psi), np.sin(psi))) * speed * _DT
        modes.append(command.mode)
        if len(car):
            closest = min(closest, float(np.hypot(*(car - pos).T).min()))
    return {
        "modes": modes,
        "position": pos,
        "closest_m": closest,
        "speed": speed,
    }


def _kerbs_at(half_road: float, length_m: float) -> np.ndarray:
    ys = np.arange(0.0, length_m, 0.5)
    return np.concatenate(
        [
            np.column_stack((np.full_like(ys, -half_road), ys)),
            np.column_stack((np.full_like(ys, half_road), ys)),
        ]
    )


def test_a_parked_car_is_driven_around_rather_than_reversed_away_from() -> None:
    """
    Four of the user's six complaints meet here: navigating round an object,
    going for the clear path, not reversing, and using the evidence rather
    than a 4,000-point sample of it. Before this work the car drove up to the
    parked car, blocked, and began the reverse recovery -- and with a
    destination set it did that at EVERY obstacle distance, because the
    saturated route cross-track term outbid the whole free-distance term.

    A wide road on purpose. The narrow case is a separate, still-open defect
    and has its own xfail below; keeping them apart is what stops this test
    silently becoming a test of the candidate model rather than of the cost
    stack.
    """
    result = _drive_past_a_parked_car(30.0, road_half=5.5, car_x=4.4)

    assert set(result["modes"]) == {"DRIVING"}
    # The car sits at 60-64.5 m; this is well past it and still going.
    assert float(result["position"][1]) > 200.0
    assert float(result["speed"]) > 8.0
    # ...and it did not scrape by: the ego half-width is 0.9 m.
    assert float(result["closest_m"]) > 1.5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The candidate model cannot represent a lane change: every candidate is "
        "hold-then-bend with the bend held to the horizon, so on a 7 m road the "
        "only line clearing a parked car is scored against the kerb it then runs "
        "into (measured at a 24 m gap: in-lane blocked by the car at 24.3 m, the "
        "clearing arc by the opposite kerb at 24.8 m). An S-curve family -- bend "
        "out, bend back -- WAS built and measured: it fixes this case, and it "
        "breaks the pocket escape below, taking it from 92.55 m clear to -4.61 m. "
        "The sustained free distance quietly penalises a hard turn in a confined "
        "space, and the S erases that; capping it removes the benefit instead. "
        "Landing it safely needs the planner to COMMIT to the manoeuvre, because "
        "under per-tick re-planning the car drives the outbound half for ever and "
        "the return leg never arrives -- the deferred-family trap in a new place."
    ),
)
def test_a_parked_car_on_an_ordinary_road_is_passed() -> None:
    """
    The narrow case, and the one the candidate model could not do at all.
    Every candidate was hold-then-bend with the bend held to the horizon,
    so on a 7 m road the only line clearing a parked car was scored against
    the kerb it then ran into: measured at a 24 m gap, the in-lane arc was
    blocked by the car at 24.3 m and the clearing arc by the opposite kerb
    at 24.8 m. Nothing rewarded going round. What it takes is an S -- bend
    out, bend back -- which is the manoeuvre a driver actually makes, and
    which is not in the candidate set. See the xfail reason above for why the
    S was built, measured, and not landed.
    """
    result = _drive_past_a_parked_car(30.0, road_half=3.5, car_x=2.4)

    assert set(result["modes"]) == {"DRIVING"}
    assert float(result["position"][1]) > 90.0
    assert float(result["closest_m"]) > 1.1
