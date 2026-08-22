"""Direct tests for the Reeds-Shepp steering-function contract."""

from __future__ import annotations

import numpy as np

from beamng_lidar_bev.reeds_shepp import Segment, integrate


def test_a_reverse_path_is_reverse_from_its_first_sample() -> None:
    """A synthetic forward sample creates a false cusp in Hybrid A*."""
    poses = integrate(
        [Segment(steering=0, gear=-1, length=2.0)],
        radius=6.0,
        step_m=0.25,
    )

    assert np.all(poses[:, 3] == -1.0)
    assert np.allclose(poses[-1, :2], (0.0, -2.0), atol=1e-9)
