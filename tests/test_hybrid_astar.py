"""Hybrid A* invariants that keep a planned manoeuvre executable."""

from __future__ import annotations

import numpy as np

from beamng_lidar_bev.hybrid_astar import Occupancy, PlannedPath, Pose, _key


def test_search_state_distinguishes_the_gear_used_to_reach_a_pose() -> None:
    """A direction-change penalty makes arrival gear part of future cost."""
    pose = Pose(1.2, 3.4, 0.25)

    forward = _key(pose, 0.5, 1)
    reverse = _key(pose, 0.5, -1)

    assert forward != reverse


def test_direction_legs_share_the_exact_cusp_pose() -> None:
    """Dropping the cusp makes the executor jump between disconnected legs."""
    path = PlannedPath(
        poses=np.asarray(
            (
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 2.0, 0.0, -1.0),
                (0.5, 1.5, 0.2, -1.0),
                (0.5, 1.5, 0.2, 1.0),
                (1.0, 2.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        expansions=3,
        cost=8.0,
    )

    legs = path.legs()

    assert len(legs) == 3
    assert np.allclose(legs[0][-1, :3], legs[1][0, :3])
    assert np.allclose(legs[1][-1, :3], legs[2][0, :3])
    assert [int(np.sign(leg[-1, 3])) for leg in legs] == [1, -1, 1]


def test_one_sample_direction_change_is_not_deleted() -> None:
    """A short reverse correction must not collapse to forward-then-forward."""
    path = PlannedPath(
        poses=np.asarray(
            (
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.1, 0.9, 0.1, -1.0),
                (0.1, 0.9, 0.1, 1.0),
                (0.2, 1.8, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        expansions=2,
        cost=6.0,
    )

    legs = path.legs()

    assert [int(np.sign(leg[-1, 3])) for leg in legs] == [1, -1, 1]
    assert all(len(leg) >= 2 for leg in legs)


def test_a_primitive_cannot_step_over_a_thin_obstacle() -> None:
    """Checking only the endpoint lets a 0.7 m primitive cross a post."""
    occupancy = Occupancy(
        blocked_bev=np.asarray(((0.0, 0.35),), dtype=np.float64),
        free_bev=np.empty((0, 2), dtype=np.float64),
        cell_m=0.1,
    )

    cost = occupancy.motion_cost(
        Pose(0.0, 0.0, 0.0),
        Pose(0.0, 0.7, 0.0),
        half_width=0.15,
        front=0.1,
        rear=0.1,
    )

    assert cost is None
