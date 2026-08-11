"""
Planning memory: persistence, the two clocks, teleports and bounds.

All offline over synthetic BEV clusters. The scenarios mirror the WORLD
stores' design arguments (forgotten by the metre, vehicles on the wall clock,
teleport guard) because the memory borrows exactly those, at planner grade.
"""

from __future__ import annotations

import time

import numpy as np

from beamng_lidar_bev.config import (
    MEMORY_MAX_CELLS,
    MEMORY_MIN_SUPPORT,
)
from beamng_lidar_bev.planning_map import PlanningMemory

RIGHT = np.asarray((1.0, 0.0, 0.0))
FORWARD = np.asarray((0.0, 1.0, 0.0))
UP_RIGHT = np.asarray((0.0, -1.0, 0.0))  # facing world +X: right is -Y
UP_FORWARD = np.asarray((1.0, 0.0, 0.0))
_EMPTY = np.empty((0, 2), dtype=np.float32)


def _cluster(x: float, y: float, count: int = 6) -> np.ndarray:
    """Several returns inside one 0.4 m cell, as any real surface produces."""
    return np.column_stack(
        (
            np.full(count, x) + np.linspace(0.0, 0.05, count),
            np.full(count, y),
        )
    ).astype(np.float32)


def _tick(
    memory: PlanningMemory,
    pos: tuple[float, float, float],
    obstacles: np.ndarray = _EMPTY,
    vehicles: np.ndarray = _EMPTY,
    now: float = 0.0,
    road: np.ndarray = _EMPTY,
) -> None:
    memory.update(np.asarray(pos), RIGHT, FORWARD, obstacles, vehicles, road, now)


def test_a_kerb_survives_an_occluded_tick_and_lands_in_the_right_place() -> None:
    """World-anchored: after the car moves AND turns, the remembered cell
    reprojects into the new BEV frame within a fraction of a cell."""
    memory = PlanningMemory()
    _tick(memory, (0.0, 0.0, 0.0), obstacles=_cluster(2.0, 10.0))

    moved = memory.obstacles_bev(np.asarray((0.0, 5.0, 0.0)), RIGHT, FORWARD)
    turned = memory.obstacles_bev(
        np.asarray((0.0, 5.0, 0.0)), UP_RIGHT, UP_FORWARD
    )

    assert len(moved) == 1
    np.testing.assert_allclose(moved[0], (2.0, 5.0), atol=0.2)
    # Facing world +X from (0, 5): the kerb at world (2, 10) reads 5 m to the
    # LEFT and 2 m ahead.
    assert len(turned) == 1
    np.testing.assert_allclose(turned[0], (-5.0, 2.0), atol=0.2)


def test_memory_expires_by_metres_travelled_not_seconds() -> None:
    memory = PlanningMemory()
    _tick(memory, (0.0, 0.0, 0.0), obstacles=_cluster(2.0, 10.0), now=0.0)

    for step in range(1, 6):  # 25 m of travel in a fraction of a second
        _tick(memory, (0.0, 5.0 * step, 0.0), now=0.01 * step)

    assert memory.cell_count == 0


def test_a_parked_store_keeps_its_cells() -> None:
    """The WORLD lesson: a stopped car must not watch its map drain away."""
    memory = PlanningMemory()
    _tick(memory, (0.0, 0.0, 0.0), obstacles=_cluster(2.0, 10.0), now=0.0)

    for second in range(1, 60):
        _tick(memory, (0.0, 0.0, 0.0), now=float(second))

    assert memory.cell_count == 1


def test_vehicle_cells_expire_on_the_wall_clock_and_never_smear() -> None:
    memory = PlanningMemory()
    car = _cluster(2.0, 10.0)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=car, vehicles=car, now=0.0)
    assert memory.vehicle_cell_count == 1

    # The car has driven off; 0.3 s later its cell is gone even though the
    # ego never moved a metre.
    _tick(memory, (0.0, 0.0, 0.0), now=0.3)
    assert memory.cell_count == 0

    # A mover leaves no streak: cells older than the TTL drop as it goes.
    for index in range(6):
        spot = _cluster(2.0, 10.0 + 2.0 * index)
        _tick(memory, (0.0, 0.0, 0.0), obstacles=spot, vehicles=spot,
              now=0.1 * index)
    assert memory.cell_count <= 2


def test_seeing_the_ground_evicts_an_unmarked_mover_smear() -> None:
    """On an unannotated map the vehicle TTL never engages (no semantic
    mark), so a crossing car's returns enter as static scenery. The ground
    contradiction is the semantics-independent escape: once the sensors see
    ROAD through those cells -- and only road -- the memory lets go, even
    with the ego parked and the odometer stalled."""
    memory = PlanningMemory()
    smear = _cluster(2.0, 10.0)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=smear, now=0.0)  # unmarked
    assert memory.cell_count == 1

    for tick in range(1, 4):  # the car has left; tarmac shows through
        _tick(memory, (0.0, 0.0, 0.0), road=smear, now=0.04 * tick)

    assert memory.cell_count == 0


def test_an_occluded_cell_is_not_evicted_by_road_elsewhere() -> None:
    """Occlusion is the case memory exists for: a cell receiving NO returns
    of either kind must persist, whatever the rest of the tick saw."""
    memory = PlanningMemory()
    kerb = _cluster(2.0, 10.0)
    elsewhere = _cluster(-4.0, 6.0)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=kerb, now=0.0)

    for tick in range(1, 4):
        _tick(memory, (0.0, 0.0, 0.0), road=elsewhere, now=0.04 * tick)

    assert memory.cell_count == 1


def test_a_visible_kerb_is_protected_from_the_ground_contradiction() -> None:
    """A cell holding road returns AND obstacle returns is a kerb face seen
    beside its own tarmac -- the obstacle returns protect it."""
    memory = PlanningMemory()
    kerb = _cluster(2.0, 10.0)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=kerb, now=0.0)

    for tick in range(1, 4):
        _tick(
            memory, (0.0, 0.0, 0.0), obstacles=kerb, road=kerb, now=0.04 * tick
        )

    assert memory.cell_count == 1


def test_a_marked_cell_still_being_observed_keeps_living() -> None:
    """WORLD's wall-clock semantics: ANY observation refreshes a marked
    cell's stamp, so a once-marked cell observed as static every tick never
    expires mid-observation or loses its support -- it fades only after the
    TTL of NON-observation."""
    memory = PlanningMemory()
    spot = _cluster(2.0, 10.0)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=spot, vehicles=spot, now=0.0)

    for tick in range(1, 10):  # 0.36 s of static observation, past the TTL
        _tick(memory, (0.0, 0.0, 0.0), obstacles=spot, now=0.04 * tick)
    assert memory.cell_count == 1
    assert memory._support[0] > len(spot)  # support kept accumulating

    _tick(memory, (0.0, 0.0, 0.0), now=0.36 + 0.3)  # unobserved past the TTL
    assert memory.cell_count == 0


def test_a_pose_jump_clears_the_store() -> None:
    memory = PlanningMemory()
    _tick(memory, (0.0, 0.0, 0.0), obstacles=_cluster(2.0, 10.0))

    _tick(memory, (100.0, 0.0, 0.0))  # a respawn, not a drive

    assert memory.cell_count == 0
    assert memory.oldest_age_m() == 0.0


def test_low_support_cells_are_withheld() -> None:
    """A speck that survived despeckle once must not block an arc from
    memory; a surface observed across ticks must."""
    memory = PlanningMemory()
    speck = np.asarray([[2.0, 10.0]], dtype=np.float32)
    origin = np.asarray((0.0, 0.0, 0.0))

    for tick in range(MEMORY_MIN_SUPPORT - 1):
        _tick(memory, (0.0, 0.0, 0.0), obstacles=speck, now=0.01 * tick)
    assert len(memory.obstacles_bev(origin, RIGHT, FORWARD)) == 0

    _tick(memory, (0.0, 0.0, 0.0), obstacles=speck, now=0.05)
    assert len(memory.obstacles_bev(origin, RIGHT, FORWARD)) == 1


def test_the_radius_bound_and_cap_hold() -> None:
    memory = PlanningMemory()
    far = np.asarray([[0.0, 60.0]] * 6, dtype=np.float32)  # beyond 50 m
    _tick(memory, (0.0, 0.0, 0.0), obstacles=far)
    assert memory.cell_count == 0

    # More distinct cells than the cap in one tick: a 0.45 m lattice over a
    # 70 m square, all within the 50 m radius.
    axis = np.arange(-35.0, 35.0, 0.45)
    grid = np.column_stack(
        [array.ravel() for array in np.meshgrid(axis, axis)]
    ).astype(np.float32)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=grid)
    assert 0 < memory.cell_count <= MEMORY_MAX_CELLS


def test_memory_update_and_query_stay_inside_the_budget() -> None:
    """The whole memory step must stay a small fraction of the 40 ms tick,
    with the store at its cap and a full-size frame arriving."""
    rng = np.random.default_rng(11)
    memory = PlanningMemory()
    axis = np.arange(-35.0, 35.0, 0.45)
    grid = np.column_stack(
        [array.ravel() for array in np.meshgrid(axis, axis)]
    ).astype(np.float32)
    _tick(memory, (0.0, 0.0, 0.0), obstacles=grid)

    frames = [
        (rng.uniform(-30.0, 30.0, size=(4000, 2)).astype(np.float32))
        for _ in range(10)
    ]
    origin = np.asarray((0.0, 0.0, 0.0))
    started = time.perf_counter()
    for index, frame in enumerate(frames):
        memory.update(
            origin, RIGHT, FORWARD, frame, _EMPTY, frame[::8], 0.01 * index
        )
        memory.obstacles_bev(origin, RIGHT, FORWARD)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(frames)

    assert elapsed_ms < 3.0
