from __future__ import annotations

import json

import numpy as np
import pytest

from beamng_lidar_bev.models import RouteHint
from beamng_lidar_bev.navigation import (
    LUA_ROUTE_CHUNK,
    fetch_route,
    parse_route,
    route_heading,
)

STATE = {
    "pos": (0.0, 0.0, 0.0),
    "dir": (0.0, 1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
}


def _payload(*points: tuple[float, float, float], length: float = 120.0) -> str:
    return json.dumps(
        {
            "hasTarget": True,
            "path": [list(point) for point in points],
            "length": length,
        }
    )


def _hint(*points: tuple[float, float, float]) -> RouteHint:
    return RouteHint(
        path_world=np.asarray(points, dtype=np.float64), remaining_m=120.0
    )


# --- the Lua chunk -----------------------------------------------------------


def test_the_lua_chunk_returns_an_encoded_string() -> None:
    """techCore sends tostring(<return value>), so a table would arrive as
    'table: 0x...'. The chunk must json-encode before returning."""
    assert "jsonEncode" in LUA_ROUTE_CHUNK
    assert "core_groundMarkers" in LUA_ROUTE_CHUNK


# --- parse_route -------------------------------------------------------------


def test_a_route_parses_into_world_positions() -> None:
    route = parse_route(_payload((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))

    assert route is not None
    np.testing.assert_allclose(
        route.path_world, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    assert route.remaining_m == pytest.approx(120.0)


def test_no_destination_yields_no_route() -> None:
    assert parse_route(json.dumps({"hasTarget": False})) is None


def test_an_empty_path_yields_no_route() -> None:
    assert parse_route(_payload()) is None


def test_malformed_json_yields_no_route() -> None:
    assert parse_route("table: 0x0000023f") is None


def test_a_nil_response_yields_no_route() -> None:
    assert parse_route(None) is None
    assert parse_route("") is None


def test_unusable_nodes_are_skipped_rather_than_fatal() -> None:
    raw = json.dumps(
        {"hasTarget": True, "path": [[1.0, 2.0], [3.0, 4.0, 5.0]], "length": 9.0}
    )

    route = parse_route(raw)

    assert route is not None
    np.testing.assert_allclose(route.path_world, [[3.0, 4.0, 5.0]])


# --- fetch_route -------------------------------------------------------------


def test_fetch_route_passes_the_chunk_to_the_runner() -> None:
    seen: list[str] = []

    def run_lua(chunk: str) -> str:
        seen.append(chunk)
        return _payload((0.0, 30.0, 0.0))

    route = fetch_route(run_lua)

    assert seen == [LUA_ROUTE_CHUNK]
    assert route is not None


def test_a_lua_failure_never_propagates() -> None:
    """Navigation is a hint. Losing it must not stop the car driving."""

    def run_lua(chunk: str) -> str:
        raise RuntimeError("bridge went away")

    assert fetch_route(run_lua) is None


# --- route_heading -----------------------------------------------------------


def test_a_route_bending_left_gives_a_positive_heading() -> None:
    heading = route_heading(_hint((-10.0, 20.0, 0.0)), STATE)

    assert heading is not None
    assert heading > 0.0


def test_a_route_bending_right_gives_a_negative_heading() -> None:
    heading = route_heading(_hint((10.0, 20.0, 0.0)), STATE)

    assert heading is not None
    assert heading < 0.0


def test_a_route_straight_ahead_gives_a_zero_heading() -> None:
    heading = route_heading(_hint((0.0, 25.0, 0.0)), STATE)

    assert heading == pytest.approx(0.0, abs=1e-6)


def test_points_inside_the_lookahead_do_not_set_the_heading() -> None:
    """The first metres of the route are behind the car's own turn radius."""
    heading = route_heading(
        _hint((5.0, 3.0, 0.0), (6.0, 8.0, 0.0), (-12.0, 22.0, 0.0)), STATE
    )

    assert heading is not None
    assert heading > 0.0


def test_a_route_entirely_behind_the_car_gives_no_heading() -> None:
    heading = route_heading(_hint((0.0, -20.0, 0.0), (3.0, -30.0, 0.0)), STATE)

    assert heading is None


def test_a_nearby_destination_still_gives_a_heading() -> None:
    """Arriving: nothing is beyond the lookahead, so use the furthest point."""
    heading = route_heading(_hint((-4.0, 6.0, 0.0)), STATE)

    assert heading is not None
    assert heading > 0.0


def test_no_route_gives_no_heading() -> None:
    assert route_heading(None, STATE) is None


def test_the_heading_is_measured_in_the_rotated_vehicle_frame() -> None:
    """The car facing world -X must read a world +Y point as a turn to the right."""
    state = {"pos": (0.0, 0.0, 0.0), "dir": (-1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)}

    heading = route_heading(_hint((-20.0, 20.0, 0.0)), state)

    assert heading is not None
    assert heading < 0.0
