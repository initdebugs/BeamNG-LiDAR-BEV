"""World-anchored free/blocked memory used only by automatic parking."""

from __future__ import annotations

import numpy as np

from beamng_lidar_bev.hybrid_astar import BLOCKED, FREE, UNKNOWN
from beamng_lidar_bev.parking_map import ParkingMap

RIGHT = np.asarray((1.0, 0.0, 0.0))
FORWARD = np.asarray((0.0, 1.0, 0.0))


def test_parking_map_stays_anchored_while_the_ego_moves() -> None:
    memory = ParkingMap()
    memory.update(
        np.asarray((100.0, 200.0, 0.0)),
        RIGHT,
        FORWARD,
        blocked_bev=np.asarray(((2.0, 5.0),)),
        free_bev=np.asarray(((-1.0, 4.0),)),
    )

    occupancy = memory.occupancy_bev(
        np.asarray((100.0, 201.0, 0.0)), RIGHT, FORWARD
    )

    assert occupancy.state(2.0, 4.0) == BLOCKED
    assert occupancy.state(-1.0, 3.0) == FREE


def test_parking_map_clears_on_a_pose_jump() -> None:
    memory = ParkingMap()
    memory.update(
        np.asarray((0.0, 0.0, 0.0)),
        RIGHT,
        FORWARD,
        blocked_bev=np.asarray(((2.0, 5.0),)),
        free_bev=np.asarray(((-1.0, 4.0),)),
    )

    memory.update(
        np.asarray((100.0, 100.0, 0.0)),
        RIGHT,
        FORWARD,
        blocked_bev=np.empty((0, 2)),
        free_bev=np.empty((0, 2)),
    )
    occupancy = memory.occupancy_bev(
        np.asarray((100.0, 100.0, 0.0)), RIGHT, FORWARD
    )

    assert occupancy.state(-98.0, -95.0) == UNKNOWN


def test_a_stale_blocked_cell_under_the_body_is_not_served() -> None:
    """
    The ego footprint is trusted: a blocked cell under the body is stale by
    definition (the ego cull means nothing can re-observe ground the car
    covers), and serving it killed every plan from that spot -- measured
    live, four engagements refused UNREACHABLE in ~120 ms each.
    """
    memory = ParkingMap()
    memory.update(
        np.asarray((0.0, 0.0, 0.0)),
        RIGHT,
        FORWARD,
        blocked_bev=np.asarray(((0.2, 0.5), (0.0, 6.0))),
        free_bev=np.empty((0, 2)),
    )

    occupancy = memory.occupancy_bev(
        np.asarray((0.0, 0.0, 0.0)),
        RIGHT,
        FORWARD,
        body=(1.0, 1.0, 1.8, 2.5),
    )

    assert occupancy.state(0.2, 0.5) == UNKNOWN, "under the body: stale"
    assert occupancy.state(0.0, 6.0) == BLOCKED, "clear of the body: kept"
