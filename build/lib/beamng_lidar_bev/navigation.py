"""
Reads the destination the player set on the in-game bigmap.

Used for ONE thing: which way to go at a junction. The route never becomes a
path to follow -- `planner.plan_arc` takes the resulting bearing as one cost
term among several, so a blocked arc is still rejected. With no destination set
the hint is simply absent and the car explores on LiDAR alone.

Qt-free and BeamNGpy-free: the Lua runner arrives as a plain callable, which is
also what makes the whole module testable offline.

The route lives in `core_groundMarkers.routePlanner.path`
(``lua/ge/extensions/core/groundMarkers.lua``), reachable through
``techCore.handleQueueLuaCommandGE``.
"""

from __future__ import annotations

import json
import logging
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
LUA_ROUTE_CHUNK = """
local ok, result = pcall(function()
  if core_groundMarkers == nil or not core_groundMarkers.currentlyHasTarget() then
    return {hasTarget = false}
  end
  local path = {}
  for i, node in ipairs(core_groundMarkers.routePlanner.path) do
    path[i] = {node.pos.x, node.pos.y, node.pos.z}
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
    for node in payload.get("path") or ():
        try:
            nodes.append((float(node[0]), float(node[1]), float(node[2])))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    if not nodes:
        return None
    return RouteHint(
        path_world=np.asarray(nodes, dtype=np.float64),
        remaining_m=float(payload.get("length") or 0.0),
    )


def fetch_route(run_lua: Callable[[str], Any]) -> RouteHint | None:
    """
    Ask the simulator for the current bigmap route.

    `run_lua` is anything that takes a Lua chunk and returns its reply -- in the
    worker, ``bng.control.queue_lua_command(chunk, response=True)``. Every
    failure degrades to None: the hint is optional, and losing it must never
    stop the car driving.
    """
    try:
        raw = run_lua(LUA_ROUTE_CHUNK)
    except Exception:
        LOGGER.debug("Could not read the in-game navigation route", exc_info=True)
        return None
    return parse_route(raw)


def route_heading(
    route: RouteHint | None,
    state: Mapping[str, Any],
    lookahead_m: float = NAV_LOOKAHEAD_M,
) -> float | None:
    """
    Bearing of the route at the lookahead, in radians, positive to the left.

    Matches the planner's positive-curvature-is-left convention. Nodes behind
    the car are ignored; if the destination is closer than the lookahead the
    furthest node ahead is used, so the hint survives the final approach.
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
