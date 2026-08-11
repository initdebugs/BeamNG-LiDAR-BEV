"""
Reads the destination the player set on the in-game bigmap.

This module only FETCHES and PARSES; `route_model` is what turns the result
into a reference path the planner follows. Guidance never becomes authority:
every route-derived quantity enters `planner.plan_arc` as cost terms among
several, so a blocked arc is still rejected. With no destination set the terms
are simply absent and the car explores on LiDAR alone. `route_heading` remains
as the coarse fallback for routes too short to build a path from.

Qt-free and BeamNGpy-free: the Lua runner arrives as a plain callable, which is
also what makes the whole module testable offline.

The route lives in `core_groundMarkers.routePlanner.path`
(``lua/ge/extensions/core/groundMarkers.lua``), reachable through
``techCore.handleQueueLuaCommandGE``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from .config import NAV_LOOKAHEAD_M
from .geometry import vec3, vehicle_axes
from .models import RouteHint

LOGGER = logging.getLogger(__name__)

# techCore replies with tostring(<the chunk's first return value>), so returning
# a table yields the literal string "table: 0x...". The chunk MUST json-encode
# before returning, and the caller must ask for a response or only get an ACK.
#
# The pcall wrapper matters: core_groundMarkers is absent in the main menu and
# between map loads, and an uncaught Lua error comes back to Python as an
# exception rather than as "no route".
# Each node row is {x, y, z, distToTarget, linkCount, radius}. The last three
# are read total-function style with -1/0 defaults so a game version that lacks
# a field degrades that field rather than the whole route (the pcall would
# otherwise turn one nil lookup into "no destination"). linkCount is the
# navgraph degree (junction detector); radius is the half road width, one hop
# away on the navgraph node via node.wp. Deliberately NOT read:
# links[].speedLimit -- the cap is 40 km/h and the speed preview is curvature
# derived, so the limit would almost never bind, and it would add a
# nested-table walk to a 1 Hz blocking call. Future extension if the cap rises.
LUA_ROUTE_CHUNK = """
local ok, result = pcall(function()
  if core_groundMarkers == nil or not core_groundMarkers.currentlyHasTarget() then
    return {hasTarget = false}
  end
  local graph = (map and map.getMap and map.getMap()) or nil
  local nodes = (graph and graph.nodes) or {}
  local path = {}
  for i, node in ipairs(core_groundMarkers.routePlanner.path) do
    local info = node.wp and nodes[node.wp] or nil
    path[i] = {node.pos.x, node.pos.y, node.pos.z,
               node.distToTarget or -1, node.linkCount or 0,
               (info and info.radius) or -1}
  end
  return {
    hasTarget = true,
    path = path,
    length = core_groundMarkers.getPathLength(),
  }
end)
if not ok then return jsonEncode({hasTarget = false}) end
return jsonEncode(result)
"""


def _node_extra(node: Any, index: int, default: float) -> float:
    """One optional trailing field, defaulting rather than skipping the node."""
    try:
        value = float(node[index])
    except (TypeError, ValueError, IndexError, KeyError):
        return default
    return value if math.isfinite(value) else default


def parse_route(raw: Any) -> RouteHint | None:
    """Turn the Lua chunk's JSON reply into a `RouteHint`, or None."""
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        LOGGER.debug("Navigation route reply was not JSON: %r", raw)
        return None
    if not isinstance(payload, Mapping) or not payload.get("hasTarget"):
        return None

    nodes: list[tuple[float, float, float]] = []
    dists: list[float] = []
    links: list[float] = []
    widths: list[float] = []
    saw_extras = False
    for node in payload.get("path") or ():
        try:
            position = (float(node[0]), float(node[1]), float(node[2]))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        # json.loads accepts NaN literals, and a non-finite node would stitch
        # the polyline with a silently under-read arc length -- every braking
        # point downstream would land at the wrong distance. Skip it like any
        # other unusable node.
        if not all(math.isfinite(value) for value in position):
            continue
        nodes.append(position)
        # A malformed xyz skips the node; a malformed extra only defaults the
        # field, because the position is load-bearing and the extras are not.
        try:
            has_extras = len(node) > 3
        except TypeError:
            has_extras = False
        saw_extras = saw_extras or has_extras
        dists.append(_node_extra(node, 3, -1.0))
        links.append(_node_extra(node, 4, 0.0))
        widths.append(_node_extra(node, 5, -1.0))
    if not nodes:
        return None
    return RouteHint(
        path_world=np.asarray(nodes, dtype=np.float64),
        remaining_m=float(payload.get("length") or 0.0),
        dist_to_target_m=(
            np.asarray(dists, dtype=np.float64) if saw_extras else None
        ),
        link_counts=(
            np.asarray(links, dtype=np.int32) if saw_extras else None
        ),
        half_width_m=(
            np.asarray(widths, dtype=np.float64) if saw_extras else None
        ),
    )


def fetch_route_reply(run_lua: Callable[[str], Any]) -> Any | None:
    """
    The chunk's raw reply, or None ONLY on transport failure.

    The distinction is what lets the worker keep its cached route through a
    transient bridge hiccup while still clearing it the moment the player
    cancels the destination: a parseable "no target" is data, an exception is
    noise.
    """
    try:
        return run_lua(LUA_ROUTE_CHUNK)
    except Exception:
        LOGGER.debug("Could not read the in-game navigation route", exc_info=True)
        return None


def fetch_route(run_lua: Callable[[str], Any]) -> RouteHint | None:
    """
    Ask the simulator for the current bigmap route.

    `run_lua` is anything that takes a Lua chunk and returns its reply -- in the
    worker, ``bng.control.queue_lua_command(chunk, response=True)``. Every
    failure degrades to None: the hint is optional, and losing it must never
    stop the car driving.
    """
    reply = fetch_route_reply(run_lua)
    return None if reply is None else parse_route(reply)


def route_heading(
    route: RouteHint | None,
    state: Mapping[str, Any],
    lookahead_m: float = NAV_LOOKAHEAD_M,
) -> float | None:
    """
    Bearing TO the first route node at least `lookahead_m` away, in radians,
    positive to the left -- a coarse hint, not the route's tangent.

    This is the legacy fallback for routes `route_model.build_route_path`
    cannot turn into a reference path. Matches the planner's
    positive-curvature-is-left convention. Nodes behind the car are ignored;
    if the destination is closer than the lookahead the furthest node ahead is
    used, so the hint survives the final approach.
    """
    if route is None or len(route.path_world) == 0:
        return None

    origin = vec3(state["pos"])
    right, forward, _ = vehicle_axes(state)
    offsets = np.asarray(route.path_world, dtype=np.float64) - origin
    lateral = offsets @ right
    ahead = offsets @ forward

    in_front = ahead > 0.0
    if not in_front.any():
        return None
    lateral = lateral[in_front]
    ahead = ahead[in_front]

    distance = np.hypot(lateral, ahead)
    beyond = np.flatnonzero(distance >= float(lookahead_m))
    index = int(beyond[0]) if len(beyond) else int(np.argmax(distance))
    return float(np.arctan2(-lateral[index], ahead[index]))
