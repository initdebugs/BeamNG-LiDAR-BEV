"""
Reeds-Shepp paths: the shortest route between two poses for a car that can
reverse.

This is a STEERING FUNCTION, not a planner. It answers one question exactly --
given a minimum turning radius and the ability to go both ways, what is the
shortest path from one pose to another **in empty space** -- and it knows
nothing at all about obstacles. `hybrid_astar` is what turns it into a planner
by searching over collision-free motions and using this both as its heuristic
and as a "can I finish from here in one shot" test.

It exists because the hand-written manoeuvre families it replaces could not be
patched into covering a real lot. Each of those was ONE straight-arc-straight,
which needs `R(1 - cos t)` of lateral room and `R sin t` of longitudinal room
to turn through `t`: a bay 1.9 m to the side needs 56 degrees, so 2.6 m across
and 5.0 m along for the arc alone, and no single arc reaches a setup pose that
is itself beside the car. Reeds-Shepp reaches any pose given room, using as
many direction changes as it needs.

The path is returned in NORMALISED units (turning radius 1) as a list of
`Segment`s and scaled by the caller, which is the form the literature uses and
the form that keeps the word formulas readable.

Coverage: the CSC and CCC families with all four symmetries -- 24 of the 48
words. The omitted ones (CCSC, CCSCC) are rarely the shortest and never the
only option here, and `hybrid_astar` does not depend on this being complete:
a missed word costs a slightly longer path or one more search expansion, never
a wrong one. `test_reeds_shepp.py` pins the property that matters, which is
that every path returned actually ENDS on the goal when you drive it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Steering, as the literature writes it: +1 left, 0 straight, -1 right.
LEFT = 1
STRAIGHT = 0
RIGHT = -1


@dataclass(frozen=True)
class Segment:
    """One constant-steering, constant-direction piece of a path."""

    steering: int
    """LEFT, STRAIGHT or RIGHT."""
    gear: int
    """+1 forwards, -1 backwards."""
    length: float
    """Arc length in turning-radius units; always positive."""


def _mod2pi(angle: float) -> float:
    """Wrap to (-pi, pi]. The word formulas assume this branch."""
    wrapped = math.fmod(angle, 2.0 * math.pi)
    if wrapped < -math.pi:
        wrapped += 2.0 * math.pi
    elif wrapped > math.pi:
        wrapped -= 2.0 * math.pi
    return wrapped


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def path_length(path: list[Segment]) -> float:
    return sum(segment.length for segment in path)


# --- the word families -------------------------------------------------------
#
# Each takes the goal already expressed in the start's frame with unit turning
# radius, and returns the segment lengths for its word, or None.


def _lsl(x: float, y: float, phi: float) -> tuple[float, float, float] | None:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t < 0.0:
        return None
    v = _mod2pi(phi - t)
    if v < 0.0:
        return None
    return t, u, v


def _lsr(x: float, y: float, phi: float) -> tuple[float, float, float] | None:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 * u1 < 4.0:
        return None
    u = math.sqrt(u1 * u1 - 4.0)
    t = _mod2pi(t1 + math.atan2(2.0, u))
    if t < 0.0:
        return None
    v = _mod2pi(t - phi)
    if v < 0.0:
        return None
    return t, u, v


def _lrl(x: float, y: float, phi: float) -> tuple[float, float, float] | None:
    """
    Three arcs, no straight -- the tight-quarters word.

    The published branch for `t` varies between sources and the wrong one is
    silently plausible: it yields a path of the right SHAPE that ends
    somewhere else entirely. Derived here against a path built by driving a
    known LRL and measuring where it finishes, which is the only check that
    catches it -- `theta + u/2`, not `theta + pi/2 + acos(rho/4)`.
    """
    rho, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if rho > 4.0:
        return None
    u = 2.0 * math.asin(rho / 4.0)
    t = _mod2pi(theta + 0.5 * u)
    v = _mod2pi(phi - t + u)
    if t < 0.0 or v < 0.0:
        return None
    return t, u, v


# --- symmetries --------------------------------------------------------------
#
# Every word is generated from the three above by transforming the GOAL and
# then fixing up the answer. This is what keeps the formula count at three
# rather than twenty-four, and it is the standard construction.


def _timeflip(
    x: float, y: float, phi: float
) -> tuple[float, float, float]:
    """Driving the path backwards in time: forwards becomes reverse."""
    return -x, y, -phi


def _reflect(x: float, y: float, phi: float) -> tuple[float, float, float]:
    """Mirroring left and right."""
    return x, -y, -phi


def _backwards(
    x: float, y: float, phi: float
) -> tuple[float, float, float]:
    """
    The start as seen from the GOAL: solve that, then reverse the word.

    The third symmetry, and leaving it out is not a small loss -- with only
    timeflip and reflect, 37% of random poses had no word at all. It costs
    nothing but a transform because a Reeds-Shepp path driven end to end is
    still a Reeds-Shepp path.
    """
    return (
        x * math.cos(phi) + y * math.sin(phi),
        x * math.sin(phi) - y * math.cos(phi),
        phi,
    )


def _build(
    lengths: tuple[float, float, float],
    word: tuple[int, int, int],
    flipped: bool,
    reflected: bool,
    reversed_order: bool,
) -> list[Segment]:
    steering = [-value if reflected else value for value in word]
    gear = -1 if flipped else 1
    segments = [
        Segment(steering=turn, gear=gear, length=abs(length))
        for turn, length in zip(steering, lengths)
        if abs(length) > 1e-9
    ]
    return segments[::-1] if reversed_order else segments


_WORDS = (
    (_lsl, (LEFT, STRAIGHT, LEFT)),
    (_lsr, (LEFT, STRAIGHT, RIGHT)),
    (_lrl, (LEFT, RIGHT, LEFT)),
)


def all_paths(
    goal: tuple[float, float, float], radius: float
) -> list[list[Segment]]:
    """
    Every valid Reeds-Shepp word to `goal`, in METRES, unordered.

    `shortest_path` picks by LENGTH, and that is the wrong criterion for a
    planner that prices reverse travel and direction changes: the shortest
    word between two poses is very often a reverse-heavy one, a few metres
    shorter than a forward-only word that is far cheaper once penalties
    apply. Exposing the whole family lets the caller pick by ITS cost --
    which is how the analytic shot stops proposing shuffles.
    """
    if radius <= 0.0:
        raise ValueError("Turning radius must be positive")
    # Into the literature's frame: it works in (x forward, y left) with
    # heading measured anticlockwise from +x, and in units of the turning
    # radius. Ours is (x right, y forward) with heading clockwise from +y.
    x = float(goal[1]) / radius
    y = -float(goal[0]) / radius
    phi = -float(goal[2])

    found: list[list[Segment]] = []
    for backwards in (False, True):
        for flipped in (False, True):
            for reflected in (False, True):
                probe = (x, y, phi)
                if backwards:
                    probe = _backwards(*probe)
                if flipped:
                    probe = _timeflip(*probe)
                if reflected:
                    probe = _reflect(*probe)
                for solver, word in _WORDS:
                    lengths = solver(*probe)
                    if lengths is None:
                        continue
                    path = _build(
                        lengths, word, flipped, reflected, backwards
                    )
                    if not path:
                        continue
                    found.append(
                        [
                            Segment(
                                steering=segment.steering,
                                gear=segment.gear,
                                length=segment.length * radius,
                            )
                            for segment in path
                        ]
                    )
    return found


def shortest_path(
    goal: tuple[float, float, float], radius: float
) -> list[Segment] | None:
    """
    The shortest Reeds-Shepp path from the origin (heading 0) to `goal`.

    `goal` is `(x, y, heading)` in the START's frame, in metres and radians,
    with heading measured the way the rest of this codebase measures it: from
    +forward toward +right, so a direction is `(sin h, cos h)`. The returned
    segment lengths are in METRES, already scaled by `radius`.
    """
    candidates = all_paths(goal, radius)
    if not candidates:
        return None
    return min(candidates, key=path_length)


def integrate(
    path: list[Segment],
    radius: float,
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step_m: float = 0.2,
) -> np.ndarray:
    """
    Drive the path and return the poses it passes through.

    `(N, 4)`: right, forward, heading, gear. This is both how the path is
    drawn and how it is collision-checked -- there is deliberately no second
    expression of the same geometry to drift out of step with the first.
    """
    poses: list[tuple[float, float, float, float]] = []
    right, forward, heading = start
    first_gear = float(path[0].gear) if path else 1.0
    poses.append((right, forward, heading, first_gear))
    for segment in path:
        steps = max(1, int(math.ceil(segment.length / step_m)))
        piece = segment.length / steps
        for _ in range(steps):
            travelled = piece * segment.gear
            if segment.steering == STRAIGHT:
                right += math.sin(heading) * travelled
                forward += math.cos(heading) * travelled
            else:
                # Positive steering is LEFT, which DECREASES a heading
                # measured from +forward toward +right. The closed form is
                # the same one `parking_drive._straight_arc_straight` uses:
                # with dh/ds constant, position integrates to a chord about
                # the turn centre.
                rate = -segment.steering / radius
                turned = heading + rate * travelled
                right += (math.cos(heading) - math.cos(turned)) / rate
                forward += (math.sin(turned) - math.sin(heading)) / rate
                heading = turned
            poses.append((right, forward, heading, float(segment.gear)))
    return np.asarray(poses, dtype=np.float64)
