"""
Hybrid A*: the planner. Searches poses, not points, and can reverse.

This is what turns `reeds_shepp` from a formula into something that gets a car
into a bay. A* over a grid plans for a POINT and produces paths a car cannot
drive; hybrid A* searches the car's actual state -- position AND heading --
by expanding short arcs it could really steer, so every node is reachable by
construction and the answer needs no smoothing to be drivable.

Reeds-Shepp appears twice and both matter:

- as the HEURISTIC, so the search is guided by how far the car really has to
  travel rather than by straight-line distance, which badly underestimates
  whenever the goal is beside you or facing the wrong way; and
- as an ANALYTIC SHORTCUT tried from each expansion -- if a Reeds-Shepp path
  from here to the goal is collision-free, the search is finished exactly,
  with no discretisation error at the one place accuracy matters most.

The two together are what make it terminate quickly instead of grinding out
the last few metres one primitive at a time.

**Obstacles are what it knows and does not know.** `Occupancy` distinguishes
FREE, BLOCKED and UNKNOWN, and the distinction is load-bearing: the existing
stores record where returns came from, so absence of a return has always read
as drivable, and a planner allowed to route through never-observed space will
happily plan through a wall it has not looked at. Unknown space is traversable
at a COST here rather than forbidden -- forbidding it strands the car in a lot
it has only partly seen -- so a route through open tarmac beats a route
through the unseen, and the car prefers what it can actually see.

Qt-free and BeamNGpy-free like every planning module here: config + numpy.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .config import (
    HYBRID_ANALYTIC_INTERVAL,
    HYBRID_CELL_M,
    HYBRID_GEAR_PENALTY,
    HYBRID_GOAL_HEADING_DEG,
    HYBRID_GOAL_RADIUS_M,
    HYBRID_HEADING_BINS,
    HYBRID_MAX_EXPANSIONS,
    HYBRID_REVERSE_PENALTY,
    HYBRID_STEER_PENALTY,
    HYBRID_STEP_M,
    HYBRID_UNKNOWN_PENALTY,
    MIN_TURN_RADIUS_M,
)
from .reeds_shepp import integrate, path_length, shortest_path

FREE = 0
BLOCKED = 1
UNKNOWN = 2


@dataclass(frozen=True)
class Pose:
    right: float
    forward: float
    heading: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.right, self.forward, self.heading


class Occupancy:
    """
    What the car knows about the ground around it: free, blocked, unknown.

    Built from the two things the perception already accumulates -- obstacle
    returns and ROAD returns -- because they answer different questions. An
    obstacle cell is blocked. A road cell is FREE, and positively so: the
    sensors saw ground there. Everything else is UNKNOWN, which is the state
    the rest of this codebase has never had and a planner needs, because
    "nothing was reported here" and "this is clear" are not the same claim.
    """

    def __init__(
        self,
        blocked_bev: np.ndarray | None,
        free_bev: np.ndarray | None,
        cell_m: float = HYBRID_CELL_M,
    ) -> None:
        self.cell_m = float(cell_m)
        self._blocked = self._index(blocked_bev)
        self._free = self._index(free_bev)

    def _index(self, points: np.ndarray | None) -> set[tuple[int, int]]:
        if points is None or not len(points):
            return set()
        cells = np.floor(
            np.asarray(points, dtype=np.float64)[:, :2] / self.cell_m
        ).astype(np.int64)
        return set(map(tuple, cells))

    def state(self, right: float, forward: float) -> int:
        key = (
            int(math.floor(right / self.cell_m)),
            int(math.floor(forward / self.cell_m)),
        )
        if key in self._blocked:
            return BLOCKED
        return FREE if key in self._free else UNKNOWN

    def footprint_cost(
        self, pose: Pose, half_width: float, front: float, rear: float
    ) -> float | None:
        """
        Cost of standing here, or None if the body would hit something.

        The body is sampled as discs along its centreline rather than as a
        rectangle: a rectangle test against a cell grid is a polygon clip per
        cell, and at this cell size the discs are indistinguishable while
        being a handful of lookups.
        """
        span = front + rear
        count = max(2, int(math.ceil(span / max(half_width, 0.4))))
        sin_h, cos_h = math.sin(pose.heading), math.cos(pose.heading)
        unknown = 0
        for index in range(count + 1):
            offset = -rear + span * index / count
            right = pose.right + sin_h * offset
            forward = pose.forward + cos_h * offset
            for lateral in (-half_width, 0.0, half_width):
                probe_r = right + cos_h * lateral
                probe_f = forward - sin_h * lateral
                state = self.state(probe_r, probe_f)
                if state == BLOCKED:
                    return None
                if state == UNKNOWN:
                    unknown += 1
        return unknown * HYBRID_UNKNOWN_PENALTY

    def motion_cost(
        self,
        start: Pose,
        end: Pose,
        half_width: float,
        front: float,
        rear: float,
    ) -> float | None:
        """Average occupancy cost over the complete swept vehicle body."""
        distance = math.hypot(end.right - start.right, end.forward - start.forward)
        heading_delta = (
            end.heading - start.heading + math.pi
        ) % (2.0 * math.pi) - math.pi
        swept_edge = abs(heading_delta) * max(front, rear, half_width)
        spacing = max(self.cell_m * 0.5, 0.02)
        samples = max(1, int(math.ceil(max(distance, swept_edge) / spacing)))
        total = 0.0
        for index in range(samples + 1):
            fraction = index / samples
            pose = Pose(
                start.right + (end.right - start.right) * fraction,
                start.forward + (end.forward - start.forward) * fraction,
                start.heading + heading_delta * fraction,
            )
            cost = self.footprint_cost(pose, half_width, front, rear)
            if cost is None:
                return None
            total += cost
        return total / (samples + 1)


@dataclass(frozen=True)
class PlannedPath:
    """The answer: poses to drive, and where the direction changes."""

    poses: np.ndarray
    """(N, 4) right, forward, heading, gear."""
    expansions: int
    cost: float = 0.0
    """Complete search cost, including reverse, cusp and occupancy penalties."""

    @property
    def length_m(self) -> float:
        steps = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1)
        return float(steps.sum())

    def legs(self) -> list[np.ndarray]:
        """The path split at every direction change -- one leg per cusp."""
        gears = self.poses[:, 3]
        cuts = np.flatnonzero(np.diff(gears) != 0) + 1
        starts = np.concatenate(([0], cuts))
        stops = np.concatenate((cuts, [len(self.poses)]))
        legs: list[np.ndarray] = []
        for start, stop in zip(starts, stops):
            piece = self.poses[start:stop]
            if start:
                # A pose's gear describes the motion that ARRIVED there. The
                # next direction therefore begins at the preceding pose. Keep
                # that exact cusp in both legs and label the duplicate with the
                # new direction so even a one-sample correction remains a real
                # two-pose leg instead of disappearing.
                cusp = self.poses[start - 1].copy()
                cusp[3] = self.poses[start, 3]
                piece = np.concatenate((cusp[None, :], piece))
            if len(piece) >= 2:
                legs.append(piece)
        return legs


def _key(pose: Pose, cell_m: float, gear: int) -> tuple[int, int, int, int]:
    bin_size = 2.0 * math.pi / HYBRID_HEADING_BINS
    return (
        int(math.floor(pose.right / cell_m)),
        int(math.floor(pose.forward / cell_m)),
        int(math.floor((pose.heading % (2.0 * math.pi)) / bin_size)),
        1 if gear >= 0 else -1,
    )


def _advance(pose: Pose, steering: int, gear: int, radius: float) -> Pose:
    """One motion primitive: a short arc the car could actually steer."""
    travelled = HYBRID_STEP_M * gear
    if steering == 0:
        return Pose(
            pose.right + math.sin(pose.heading) * travelled,
            pose.forward + math.cos(pose.heading) * travelled,
            pose.heading,
        )
    rate = -steering / radius
    turned = pose.heading + rate * travelled
    return Pose(
        pose.right + (math.cos(pose.heading) - math.cos(turned)) / rate,
        pose.forward + (math.sin(turned) - math.sin(pose.heading)) / rate,
        turned,
    )


def _heuristic(pose: Pose, goal: Pose, radius: float) -> float:
    """
    How far the car really has to travel, not how far the goal is.

    Straight-line distance is a terrible guide here: it is near zero for a
    goal beside the car facing the wrong way, which is exactly the case that
    needs the most manoeuvring. Reeds-Shepp knows the difference. Where it has
    no word -- it covers the three-segment families, not all 48 -- the
    Euclidean distance stands in, which is still admissible.
    """
    relative = _relative(pose, goal)
    path = shortest_path(relative, radius)
    if path is None:
        return math.hypot(relative[0], relative[1])
    return path_length(path)


def _relative(pose: Pose, goal: Pose) -> tuple[float, float, float]:
    """The goal expressed in `pose`'s own frame."""
    sin_h, cos_h = math.sin(pose.heading), math.cos(pose.heading)
    delta_r = goal.right - pose.right
    delta_f = goal.forward - pose.forward
    return (
        delta_r * cos_h - delta_f * sin_h,
        delta_r * sin_h + delta_f * cos_h,
        goal.heading - pose.heading,
    )


def plan(
    start: Pose,
    goal: Pose,
    occupancy: Occupancy,
    half_width: float,
    front: float,
    rear: float,
    radius: float = MIN_TURN_RADIUS_M,
    start_gear: int = 1,
) -> PlannedPath | None:
    """
    A drivable, collision-free path from `start` to `goal`, or None.

    None means genuinely unreachable given what the car can SEE -- not that
    the geometry was awkward. That distinction is the whole reason for this
    module: the manoeuvre families it replaces refused bays for being in the
    wrong place, where this refuses only for something being in the way.
    """
    start_cost = occupancy.footprint_cost(start, half_width, front, rear)
    if start_cost is None:
        # Standing in something already. Refusing would strand the car, and
        # the manoeuvre's own corridor check is what guards the drive, so the
        # search starts anyway rather than declaring the situation hopeless.
        start_cost = 0.0

    start_gear = 1 if start_gear >= 0 else -1
    start_key = _key(start, occupancy.cell_m, start_gear)
    open_set: list[
        tuple[float, int, float, tuple[int, int, int, int], Pose, int]
    ] = []
    counter = 0
    heapq.heappush(
        open_set,
        (_heuristic(start, goal, radius), counter, 0.0, start_key, start, start_gear),
    )
    came_from: dict[
        tuple[int, int, int, int],
        tuple[Pose, tuple[int, int, int, int] | None, int, int],
    ] = {
        start_key: (start, None, 0, start_gear)
    }
    cost_so_far = {start_key: 0.0}
    goal_heading_tolerance = math.radians(HYBRID_GOAL_HEADING_DEG)

    expansions = 0
    while open_set and expansions < HYBRID_MAX_EXPANSIONS:
        _, _, popped_cost, node, pose, arrival_gear = heapq.heappop(open_set)
        if popped_cost > cost_so_far.get(node, math.inf) + 1e-9:
            continue
        expansions += 1

        # The analytic shortcut. Tried periodically rather than every node --
        # it is the expensive part, and near the goal it succeeds so often
        # that trying it constantly is wasted work far away.
        if expansions % HYBRID_ANALYTIC_INTERVAL == 0 or _close(
            pose, goal, goal_heading_tolerance
        ):
            shot = _analytic(
                pose, goal, occupancy, half_width, front, rear, radius
            )
            if shot is not None:
                tail, tail_cost = shot
                return PlannedPath(
                    poses=_reconstruct(came_from, node, tail),
                    expansions=expansions,
                    cost=cost_so_far[node]
                    + _direction_cost(tail, arrival_gear)
                    + tail_cost,
                )

        for steering in (-1, 0, 1):
            for gear in (1, -1):
                nxt = _advance(pose, steering, gear, radius)
                penalty = occupancy.motion_cost(
                    pose, nxt, half_width, front, rear
                )
                if penalty is None:
                    continue
                step_cost = HYBRID_STEP_M * (
                    HYBRID_REVERSE_PENALTY if gear < 0 else 1.0
                )
                step_cost += abs(steering) * HYBRID_STEER_PENALTY
                if gear != arrival_gear:
                    step_cost += HYBRID_GEAR_PENALTY
                total = cost_so_far[node] + step_cost + penalty
                nxt_key = _key(nxt, occupancy.cell_m, gear)
                if nxt_key in cost_so_far and cost_so_far[nxt_key] <= total:
                    continue
                cost_so_far[nxt_key] = total
                came_from[nxt_key] = (nxt, node, steering, gear)
                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        total + _heuristic(nxt, goal, radius),
                        counter,
                        total,
                        nxt_key,
                        nxt,
                        gear,
                    ),
                )
    return None


def _close(pose: Pose, goal: Pose, heading_tolerance: float) -> bool:
    if math.hypot(goal.right - pose.right, goal.forward - pose.forward) > (
        HYBRID_GOAL_RADIUS_M * 3.0
    ):
        return False
    error = abs((goal.heading - pose.heading + math.pi) % (2 * math.pi) - math.pi)
    return error < heading_tolerance * 3.0


def _analytic(
    pose: Pose,
    goal: Pose,
    occupancy: Occupancy,
    half_width: float,
    front: float,
    rear: float,
    radius: float,
) -> tuple[np.ndarray, float] | None:
    """A Reeds-Shepp shot at the goal, accepted only if it is clear."""
    path = shortest_path(_relative(pose, goal), radius)
    if path is None:
        return None
    poses = integrate(path, radius, start=pose.as_tuple(), step_m=0.25)
    occupancy_cost = 0.0
    for row in poses:
        penalty = occupancy.footprint_cost(
            Pose(row[0], row[1], row[2]), half_width, front, rear
        )
        if penalty is None:
            return None
        occupancy_cost += penalty * 0.25
    return poses, occupancy_cost


def _direction_cost(poses: np.ndarray, initial_gear: int) -> float:
    """Travel/reverse/cusp cost for an already sampled analytic trajectory."""
    if len(poses) < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    gears = np.where(poses[1:, 3] >= 0.0, 1, -1)
    cost = float(
        np.sum(
            distances
            * np.where(gears < 0, HYBRID_REVERSE_PENALTY, 1.0)
        )
    )
    previous = 1 if initial_gear >= 0 else -1
    for gear in gears:
        if int(gear) != previous:
            cost += HYBRID_GEAR_PENALTY
        previous = int(gear)
    return cost


def _reconstruct(
    came_from: dict, node: tuple[int, int, int, int], tail: np.ndarray
) -> np.ndarray:
    """Walk the parents back to the start and append the analytic shot."""
    poses: list[tuple[float, float, float, float]] = []
    cursor: tuple[int, int, int, int] | None = node
    while cursor is not None:
        pose, parent, _, gear = came_from[cursor]
        poses.append((pose.right, pose.forward, pose.heading, float(gear)))
        cursor = parent
    poses.reverse()
    head = np.asarray(poses, dtype=np.float64)
    # `integrate` includes its start pose labelled with the first outgoing
    # gear. The reconstructed head already contains the same physical pose;
    # dropping the duplicate lets `legs()` create one precise shared cusp.
    if len(tail) and np.allclose(head[-1, :3], tail[0, :3], atol=1e-9):
        tail = tail[1:]
    return head if not len(tail) else np.concatenate((head, tail))
