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

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    MAX_LATERAL_ACCEL_MPS2,
    MIN_TURN_RADIUS_M,
    STEERING_SIGN,
)
from beamng_lidar_bev.controller import DrivingController
from beamng_lidar_bev.models import VehicleGeometry
from beamng_lidar_bev.planner import (
    geometric_obstacles,
    plan_arc,
    rear_free_distance,
)

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
