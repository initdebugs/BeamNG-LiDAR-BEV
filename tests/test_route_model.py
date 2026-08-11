"""
The route model: reference-path geometry and the backward speed pass.

Everything here is pure offline maths over synthetic `RouteHint`s -- no
simulator, no worker. The scenarios are shaped by the navgraph's two
pathologies: node spacing of tens of metres on straights, and whole turns
encoded as a single vertex between long chords.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev.config import (
    MAX_SPEED_MPS,
    ROUTE_ARRIVAL_MARGIN_M,
    ROUTE_DEFAULT_HALF_WIDTH_M,
    ROUTE_SAMPLE_STEP_M,
)
from beamng_lidar_bev.models import RouteHint
from beamng_lidar_bev.route_model import build_route_path, route_speed_limit

STATE = {
    "pos": (0.0, 0.0, 0.0),
    "dir": (0.0, 1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
}


def _hint(
    points: list[tuple[float, float, float]],
    links: list[int] | None = None,
    widths: list[float] | None = None,
) -> RouteHint:
    array = np.asarray(points, dtype=np.float64)
    chords = float(np.sum(np.hypot(*np.diff(array[:, :2], axis=0).T)))
    return RouteHint(
        path_world=array,
        remaining_m=chords,
        link_counts=None if links is None else np.asarray(links, np.int32),
        half_width_m=None if widths is None else np.asarray(widths, float),
    )


# --- build_route_path geometry ----------------------------------------------


def test_a_sparse_straight_route_resamples_to_uniform_arc_length() -> None:
    """Navgraph nodes tens of metres apart must not reach the consumers raw."""
    path = build_route_path(
        _hint([(0.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 120.0, 0.0)]), STATE
    )

    assert path is not None
    np.testing.assert_allclose(
        np.diff(path.arc_s), ROUTE_SAMPLE_STEP_M, rtol=1e-9
    )
    np.testing.assert_allclose(path.points[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(path.headings, 0.0, atol=1e-9)
    np.testing.assert_allclose(path.curvatures, 0.0, atol=1e-9)
    assert path.speed_limit_mps == pytest.approx(MAX_SPEED_MPS)


def test_a_left_bend_reads_as_positive_curvature() -> None:
    """Positive is LEFT, the planner's convention end to end."""
    path = build_route_path(
        _hint([(0.0, 0.0, 0.0), (0.0, 20.0, 0.0), (-20.0, 40.0, 0.0)]), STATE
    )

    assert path is not None
    assert float(path.curvatures.max()) > 0.01
    assert float(path.headings[-1]) > 0.3


def test_cross_track_is_positive_when_the_ego_is_right_of_the_path() -> None:
    path = build_route_path(
        _hint([(-2.0, -10.0, 0.0), (-2.0, 30.0, 0.0)]), STATE
    )

    assert path is not None
    assert path.cross_track_m == pytest.approx(2.0, abs=1e-6)


def test_too_few_nodes_yield_no_route_path() -> None:
    assert build_route_path(None, STATE) is None
    assert (
        build_route_path(_hint([(0.0, 10.0, 0.0), (0.0, 10.0, 0.0)]), STATE)
        is None
    )


def test_a_route_entirely_behind_the_car_yields_no_route_path() -> None:
    path = build_route_path(
        _hint([(0.0, -30.0, 0.0), (0.0, -5.0, 0.0)]), STATE
    )

    assert path is None


def test_half_width_falls_back_when_the_navgraph_carried_none() -> None:
    nodes = [(0.0, 0.0, 0.0), (0.0, 40.0, 0.0), (0.0, 80.0, 0.0)]

    bare = build_route_path(_hint(nodes), STATE)
    unknown = build_route_path(
        _hint(nodes, widths=[-1.0, -1.0, -1.0]), STATE
    )
    carried = build_route_path(_hint(nodes, widths=[4.5, 4.5, 4.5]), STATE)

    assert bare is not None and unknown is not None and carried is not None
    np.testing.assert_allclose(bare.half_width_m, ROUTE_DEFAULT_HALF_WIDTH_M)
    np.testing.assert_allclose(
        unknown.half_width_m, ROUTE_DEFAULT_HALF_WIDTH_M
    )
    np.testing.assert_allclose(carried.half_width_m, 4.5)


# --- the speed preview -------------------------------------------------------


def test_a_single_vertex_corner_still_slows_the_preview() -> None:
    """A 90-degree turn encoded as ONE vertex between chords: the smoothing
    window must read real curvature out of it, or junctions are invisible."""
    path = build_route_path(
        _hint([(0.0, 0.0, 0.0), (0.0, 15.0, 0.0), (30.0, 15.0, 0.0)]), STATE
    )

    assert path is not None
    assert float(np.abs(path.curvatures).max()) > 0.1
    assert 2.0 < path.speed_limit_mps < 10.0


def test_a_corner_beyond_the_preview_does_not_bind() -> None:
    path = build_route_path(
        _hint(
            [
                (0.0, 0.0, 0.0),
                (0.0, 100.0, 0.0),
                (0.0, 200.0, 0.0),
                (30.0, 230.0, 0.0),
            ]
        ),
        STATE,
    )

    assert path is not None
    assert path.speed_limit_mps == pytest.approx(MAX_SPEED_MPS)


def test_the_destination_ramps_speed_to_zero() -> None:
    near = build_route_path(_hint([(0.0, 0.0, 0.0), (0.0, 20.0, 0.0)]), STATE)
    arrived = build_route_path(
        _hint([(0.0, 0.0, 0.0), (0.0, 4.0, 0.0)]), STATE
    )

    assert near is not None and arrived is not None
    expected = np.sqrt(2.0 * 2.5 * (20.0 - ROUTE_ARRIVAL_MARGIN_M))
    assert near.speed_limit_mps == pytest.approx(expected, rel=0.02)
    assert arrived.speed_limit_mps == pytest.approx(0.0, abs=1e-6)


def test_a_turning_junction_slows_and_a_through_junction_does_not() -> None:
    # A gentle 20-degree kink: too soft for curvature alone to slow much, so
    # the junction cap is what does the work -- if the node is a junction.
    # It sits 12 m out because the cap only binds the speed NOW while the
    # junction is inside the braking distance to it (~15 m from the cap).
    bend = 0.35
    kink = [
        (0.0, 0.0, 0.0),
        (0.0, 12.0, 0.0),
        (-30.0 * np.sin(bend), 12.0 + 30.0 * np.cos(bend), 0.0),
    ]
    straight = [(0.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 60.0, 0.0)]

    turning = build_route_path(_hint(kink, links=[1, 4, 1]), STATE)
    unmarked = build_route_path(_hint(kink, links=[1, 1, 1]), STATE)
    through = build_route_path(_hint(straight, links=[1, 4, 1]), STATE)

    assert turning is not None and unmarked is not None and through is not None
    assert bool(turning.junction_turn.any())
    assert not bool(through.junction_turn.any())
    assert turning.speed_limit_mps < unmarked.speed_limit_mps


def test_the_backward_pass_chains_toward_the_slowest_point() -> None:
    """Pure maths pin: v[i] = min(cap, sqrt(v[i+1]^2 + 2 a ds))."""
    arc_s = np.array([0.0, 10.0, 20.0])
    curvatures = np.array([0.0, 0.0, 0.2])
    flags = np.zeros(3, dtype=bool)

    limit = route_speed_limit(arc_s, curvatures, flags, remaining_m=1000.0)

    corner = np.sqrt(2.8 / 0.2)
    expected = np.sqrt(corner**2 + 2.0 * 2.5 * 20.0)
    assert limit == pytest.approx(min(expected, MAX_SPEED_MPS), rel=1e-6)


def test_an_empty_preview_reads_as_stopped() -> None:
    limit = route_speed_limit(
        np.empty(0), np.empty(0), np.empty(0, dtype=bool), remaining_m=0.0
    )

    assert limit == 0.0


def test_the_lead_never_reads_past_a_corner_dip() -> None:
    """The pass rises again right after every dip, so a point sample at the
    lead inverted the constraint with speed -- a faster ego read a WEAKER
    limit and accelerated mid-corner. The window minimum may never exceed
    the dip while the dip is inside the lead."""
    arc_s = np.arange(0.0, 60.0, 3.0)
    curvatures = np.zeros(len(arc_s))
    curvatures[4:7] = 0.17  # a tight dip at 12-18 m, cap ~4.06 m/s
    flags = np.zeros(len(arc_s), dtype=bool)

    fast = route_speed_limit(arc_s, curvatures, flags, 1000.0, speed_mps=11.1)
    slow = route_speed_limit(arc_s, curvatures, flags, 1000.0, speed_mps=6.0)

    dip = np.sqrt(2.8 / 0.17)
    assert fast <= dip + 0.1
    assert fast <= slow + 1e-9  # never inverts with speed


def test_the_lead_still_reads_ahead_on_a_monotone_ramp() -> None:
    """On a falling stretch the window minimum IS the point at the lead, so
    the documented lag-cancelling behaviour is untouched."""
    arc_s = np.arange(0.0, 60.0, 3.0)
    flags = np.zeros(len(arc_s), dtype=bool)

    limit = route_speed_limit(
        arc_s, np.zeros(len(arc_s)), flags, remaining_m=40.0, speed_mps=8.0
    )

    lead = 8.0 * 1.56
    expected = np.sqrt(2.0 * 2.5 * (40.0 - 5.0 - lead))
    assert limit == pytest.approx(expected, rel=0.02)
