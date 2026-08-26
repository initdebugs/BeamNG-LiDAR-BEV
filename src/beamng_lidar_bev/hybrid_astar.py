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
    HYBRID_HEURISTIC_WEIGHT,
    HYBRID_MAX_EXPANSIONS,
    HYBRID_REVERSE_PENALTY,
    HYBRID_SHOT_PATIENCE,
    HYBRID_STEER_PENALTY,
    HYBRID_STEP_M,
    HYBRID_UNKNOWN_PENALTY,
    MIN_TURN_RADIUS_M,
)
from .reeds_shepp import Segment, all_paths, integrate, path_length, shortest_path

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

    def with_blocked(self, points: np.ndarray | None) -> "Occupancy":
        """
        A view of this occupancy with extra VIRTUAL obstacles added.

        This is how paint becomes a constraint: a bay's side lines return no
        LiDAR points, so nothing in the real map can ever forbid crossing
        them -- the caller supplies the strips as synthetic blocked points
        and the search treats them like any wall. The free set is shared,
        not copied; both objects are read-only once built.
        """
        clone = Occupancy.__new__(Occupancy)
        clone.cell_m = self.cell_m
        clone._free = self._free
        clone._blocked = self._blocked | self._index(points)
        return clone

    def without_start_overlap(
        self, pose: Pose, half_width: float, front: float, rear: float
    ) -> "Occupancy":
        """
        This occupancy with the blocked cells under the START footprint
        exempted -- because the car standing there is PROOF they are
        drivable.

        Without this the search can be born dead: every child expansion
        sweeps the start pose's own footprint, so one blocked cell inside it
        -- a stale cell the body has covered since it was marked (nothing
        can re-observe ground under the car), or a real kerb inside the
        0.18 m clearance the footprint is inflated by -- returns None for
        every move and the whole search exhausts in milliseconds. Measured
        live: four engagements in a row refused UNREACHABLE within 120 ms
        each, from a spot the car had just driven to. Probed denser than
        `footprint_cost`'s own pattern so a cell between its probe discs
        cannot survive to kill the second expansion instead of the first.
        """
        exempt: set[tuple[int, int]] = set()
        span = front + rear
        sin_h, cos_h = math.sin(pose.heading), math.cos(pose.heading)
        stations = max(3, int(math.ceil(span / (self.cell_m * 0.5))))
        laterals = np.linspace(-half_width, half_width, 5)
        for index in range(stations + 1):
            offset = -rear + span * index / stations
            right = pose.right + sin_h * offset
            forward = pose.forward + cos_h * offset
            for lateral in laterals:
                key = (
                    int(math.floor((right + cos_h * lateral) / self.cell_m)),
                    int(
                        math.floor(
                            (forward - sin_h * lateral) / self.cell_m
                        )
                    ),
                )
                if key in self._blocked:
                    exempt.add(key)
        if not exempt:
            return self
        clone = Occupancy.__new__(Occupancy)
        clone.cell_m = self.cell_m
        clone._free = self._free
        clone._blocked = self._blocked - exempt
        return clone

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
        probes = 0
        for index in range(count + 1):
            offset = -rear + span * index / count
            right = pose.right + sin_h * offset
            forward = pose.forward + cos_h * offset
            for lateral in (-half_width, 0.0, half_width):
                probe_r = right + cos_h * lateral
                probe_f = forward - sin_h * lateral
                state = self.state(probe_r, probe_f)
                probes += 1
                if state == BLOCKED:
                    return None
                if state == UNKNOWN:
                    unknown += 1
        # The FRACTION of the body standing in unseen ground, not the count of
        # cells. Charging per probe multiplied the penalty by however many
        # stations the body happens to be sampled at -- fifteen on this car --
        # so a pose entirely in unknown space cost 3.75 against 0.7 for a whole
        # step of distance. That is not the mild preference this is documented
        # to be: at 6.4x, the search will drive 3.8 m out of its way to avoid
        # ONE step of unseen ground, and in a lot where most cells are unknown
        # it plans grand tours. Measured live on a bay 3.5 m away: a four-leg
        # manoeuvre, and on another attempt a single THIRTEEN-METRE reverse.
        return (unknown / probes if probes else 0.0) * HYBRID_UNKNOWN_PENALTY

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


def _advance_by(
    pose: Pose, steering: int, gear: int, radius: float, distance: float
) -> Pose:
    """A constant-steering move of the given arc length, closed form."""
    travelled = distance * gear
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


def _advance(pose: Pose, steering: int, gear: int, radius: float) -> Pose:
    """One motion primitive: a short arc the car could actually steer."""
    return _advance_by(pose, steering, gear, radius, HYBRID_STEP_M)


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
        return HYBRID_HEURISTIC_WEIGHT * math.hypot(relative[0], relative[1])
    return HYBRID_HEURISTIC_WEIGHT * path_length(path)


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
    # The car's presence is evidence: blocked cells its start footprint
    # overlaps are exempted for the whole search, or every child move sweeps
    # them and the search dies before its first expansion. The corridor
    # check guards the actual drive with real returns either way.
    occupancy = occupancy.without_start_overlap(
        start, half_width, front, rear
    )
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
    best_shot: tuple[float, np.ndarray] | None = None
    shot_deadline = HYBRID_MAX_EXPANSIONS
    while open_set and expansions < min(HYBRID_MAX_EXPANSIONS, shot_deadline):
        priority, _, popped_cost, node, pose, arrival_gear = heapq.heappop(
            open_set
        )
        if popped_cost > cost_so_far.get(node, math.inf) + 1e-9:
            continue
        # An admissible heuristic means `priority` under-estimates the cost of
        # ANY completion through this node, so once the frontier's best cannot
        # beat the best shot, the shot is the answer within this primitive set.
        if best_shot is not None and priority >= best_shot[0] - 1e-9:
            break
        expansions += 1

        # The analytic shortcut. Tried periodically rather than every node --
        # it is the expensive part, and near the goal it succeeds so often
        # that trying it constantly is wasted work far away.
        #
        # **Counted from the FIRST expansion, not the eighth.** Starting the
        # count at 8 meant the start pose was never asked "can you simply
        # drive there?", and the first primitive the search had committed came
        # back inside the path: measured on a bay 18 m away down an open aisle,
        # the plan opened with a pointless 0.70 m REVERSE -- exactly one
        # HYBRID_STEP_M -- and the executor then treated that as a leg, hit
        # its cusp almost at once, re-planned, and drew another one.
        #
        # **A shot is PRICED, never accepted on sight.** Taking the first
        # collision-free shot let whichever node happened to be popped when
        # the interval came round donate its whole prefix to the answer -- a
        # topology lottery that returned a different manoeuvre from every
        # slightly different pose. Each successful shot is costed like any
        # other completion (travel, reverse, cusps, occupancy) and the search
        # runs on, bounded by HYBRID_SHOT_PATIENCE, until nothing on the
        # frontier can beat the best one. The candidate keeps its OWN
        # reconstruction: `came_from[node]` may later be rewritten by a
        # cheaper route to the same cell through a different continuous pose,
        # and the stored tail was integrated from this one.
        if (expansions - 1) % HYBRID_ANALYTIC_INTERVAL == 0 or _close(
            pose, goal, goal_heading_tolerance
        ):
            shot = _analytic(
                pose,
                goal,
                occupancy,
                half_width,
                front,
                rear,
                radius,
                arrival_gear=arrival_gear,
                cost_limit=(
                    math.inf
                    if best_shot is None
                    else best_shot[0] - cost_so_far[node]
                ),
            )
            if shot is not None:
                tail, tail_cost = shot
                total = (
                    cost_so_far[node]
                    + _direction_cost(tail, arrival_gear)
                    + tail_cost
                )
                if best_shot is None or total < best_shot[0]:
                    best_shot = (
                        total,
                        _reconstruct(came_from, node, tail, radius),
                    )
                    shot_deadline = min(
                        shot_deadline, expansions + HYBRID_SHOT_PATIENCE
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
                remaining = _heuristic(nxt, goal, radius)
                if best_shot is not None and (
                    total + remaining >= best_shot[0] - 1e-9
                ):
                    continue
                cost_so_far[nxt_key] = total
                came_from[nxt_key] = (nxt, node, steering, gear)
                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        total + remaining,
                        counter,
                        total,
                        nxt_key,
                        nxt,
                        gear,
                    ),
                )
    if best_shot is not None:
        return PlannedPath(
            poses=best_shot[1], expansions=expansions, cost=best_shot[0]
        )
    return None


def _close(pose: Pose, goal: Pose, heading_tolerance: float) -> bool:
    if math.hypot(goal.right - pose.right, goal.forward - pose.forward) > (
        HYBRID_GOAL_RADIUS_M * 3.0
    ):
        return False
    error = abs((goal.heading - pose.heading + math.pi) % (2 * math.pi) - math.pi)
    return error < heading_tolerance * 3.0


def _word_cost(path: list[Segment], arrival_gear: int) -> float:
    """A Reeds-Shepp word priced exactly as the search prices its own steps."""
    cost = 0.0
    previous = 1 if arrival_gear >= 0 else -1
    for segment in path:
        cost += segment.length * (
            HYBRID_REVERSE_PENALTY if segment.gear < 0 else 1.0
        )
        if segment.gear != previous:
            cost += HYBRID_GEAR_PENALTY
        previous = segment.gear
    return cost


def _analytic(
    pose: Pose,
    goal: Pose,
    occupancy: Occupancy,
    half_width: float,
    front: float,
    rear: float,
    radius: float,
    arrival_gear: int = 1,
    cost_limit: float = math.inf,
) -> tuple[np.ndarray, float] | None:
    """
    A Reeds-Shepp shot at the goal, cheapest clear word first.

    Asking only `shortest_path` was the quiet source of shuffly plans: the
    shortest WORD between two poses is very often a reverse-heavy one a few
    metres shorter than a forward-only alternative that is far cheaper once
    the reverse and cusp penalties apply -- and since the shot is the only
    way the search ever terminates, every plan inherited that topology. The
    whole word family is priced with the search's own penalties (the arrival
    gear included, so a continuation of the current direction is correctly
    free) and collision-checked in cost order; only a handful ever need
    checking because the cheap words are the ones that clear.
    """
    words = all_paths(_relative(pose, goal), radius)
    if not words:
        return None
    words.sort(key=lambda path: _word_cost(path, arrival_gear))
    for path in words[:6]:
        # The word cost is a lower bound on the shot's priced total, so a
        # word that cannot beat the best candidate found so far is skipped
        # before the expensive integrate-and-collision pass -- and since the
        # list is sorted, everything after it is too. This is what keeps the
        # periodic shots nearly free once a good candidate exists.
        if _word_cost(path, arrival_gear) >= cost_limit:
            break
        poses = integrate(path, radius, start=pose.as_tuple(), step_m=0.25)
        occupancy_cost = 0.0
        clear = True
        for row in poses:
            penalty = occupancy.footprint_cost(
                Pose(row[0], row[1], row[2]), half_width, front, rear
            )
            if penalty is None:
                clear = False
                break
            occupancy_cost += penalty * 0.25
        if clear:
            return poses, occupancy_cost
    return None


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
    came_from: dict,
    node: tuple[int, int, int, int],
    tail: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Walk the parents back to the start and append the analytic shot.

    Each primitive is re-integrated at a THIRD of its length rather than
    reported as its endpoints. The endpoints alone are 0.7 m chords: the arc
    between them is lost, its whole tangent change lands on one vertex, and
    downstream that vertex reads as a one-sample 0.4 1/m spike -- beyond
    anything the car can steer, and beyond anything the smoother should be
    asked to hide. The sub-poses land exactly on the primitive's own arc, so
    the endpoint chain is unchanged.
    """
    steps: list[tuple[Pose, int, int]] = []
    cursor: tuple[int, int, int, int] | None = node
    start_pose: Pose | None = None
    start_gear = 1
    while cursor is not None:
        pose, parent, steering, gear = came_from[cursor]
        if parent is None:
            start_pose = pose
            start_gear = gear
        else:
            steps.append((pose, steering, gear))
        cursor = parent
    steps.reverse()
    assert start_pose is not None
    poses: list[tuple[float, float, float, float]] = [
        (
            start_pose.right,
            start_pose.forward,
            start_pose.heading,
            float(steps[0][2] if steps else start_gear),
        )
    ]
    # Each primitive is integrated BACKWARD from its own stored child pose,
    # never forward along the chain. `came_from` entries can be rewritten
    # when a cheaper route reaches the same cell through a slightly
    # different continuous pose, so an ancestor's stored pose need not be
    # the one a descendant was generated from -- integrating forward
    # propagates that mismatch down the rest of the path and disconnects
    # the analytic tail (measured, a 0.14 m sideways jump at a cusp).
    # Anchoring on the child keeps every link ending exactly where the
    # search said it ends, which is also what the tail was integrated from.
    for child, steering, gear in steps:
        for fraction in (2.0 / 3.0, 1.0 / 3.0):
            sub = _advance_by(
                child, steering, gear, radius, -HYBRID_STEP_M * fraction
            )
            poses.append((sub.right, sub.forward, sub.heading, float(gear)))
        poses.append((child.right, child.forward, child.heading, float(gear)))
    head = np.asarray(poses, dtype=np.float64)
    # `integrate` includes its start pose labelled with the first outgoing
    # gear. The reconstructed head already contains the same physical pose;
    # dropping the duplicate lets `legs()` create one precise shared cusp.
    if len(tail) and np.allclose(head[-1, :3], tail[0, :3], atol=1e-9):
        tail = tail[1:]
    return head if not len(tail) else np.concatenate((head, tail))
