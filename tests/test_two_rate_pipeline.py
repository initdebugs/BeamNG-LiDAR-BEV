"""
The two performance seams: the worker's prefetched state poll, and the scene
build's split into a store refresh (slow, world-anchored) and a per-snapshot
compose (fast, ego-tracking).
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from beamng_lidar_bev.config import WORLD_POSE_JUMP_RESET_M
from beamng_lidar_bev.models import PerceptionSnapshot, VehicleGeometry
from beamng_lidar_bev.scene_worker import SceneWorker
from beamng_lidar_bev.semantics import SCENE_BOUNDARY, SCENE_ROAD
from beamng_lidar_bev.worker import BeamNgWorker
from beamng_lidar_bev.world_scene import WorldSceneAssembler

GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.5,
    left_m=1.0,
    right_m=1.0,
    front_m=2.0,
    rear_m=2.0,
    height_m=1.5,
    mounts={},
)


def _snapshot(
    points: tuple[tuple[float, float, float], ...],
    groups: tuple[int, ...],
    *,
    ego_pos_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    timestamp: float = 0.0,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        points_world=np.asarray(points, dtype=np.float32).reshape(-1, 3),
        semantic_groups=np.asarray(groups, dtype=np.uint8),
        ego_pos_world=ego_pos_world,
        ego_dir_world=(0.0, 1.0, 0.0),
        ego_up_world=(0.0, 0.0, 1.0),
        timestamp=timestamp,
        speed_mps=0.0,
        vehicle_geometry=GEOMETRY,
    )


# --- The prefetched state poll -----------------------------------------------


def test_a_prefetched_state_is_advanced_by_its_own_velocity() -> None:
    future: Future = Future()
    future.set_result(
        (
            {"pos": (100.0, 50.0, 3.0), "vel": (10.0, -10.0, 0.0)},
            time.perf_counter() - 0.1,
        )
    )
    worker = SimpleNamespace(_state_future=future)

    state = BeamNgWorker._take_vehicle_state(worker)  # type: ignore[arg-type]

    # ~0.1 s at (10, -10, 0) m/s; a small allowance for the time the test
    # itself takes between stamping the age and computing the correction.
    assert 100.9 <= state["pos"][0] <= 101.5
    assert 48.5 <= state["pos"][1] <= 49.1
    assert state["pos"][2] == 3.0
    assert worker._state_future is None


def test_a_stale_prefetch_is_discarded_and_repolled() -> None:
    future: Future = Future()
    future.set_result(({"pos": (0.0, 0.0, 0.0)}, time.perf_counter() - 5.0))
    fresh = {"pos": (7.0, 7.0, 7.0)}
    worker = SimpleNamespace(
        _state_future=future, _get_vehicle_state=lambda: fresh
    )

    state = BeamNgWorker._take_vehicle_state(worker)  # type: ignore[arg-type]

    assert state is fresh


def test_a_failed_prefetch_falls_back_to_a_fresh_poll() -> None:
    future: Future = Future()
    future.set_exception(ConnectionError("bridge went away"))
    fresh = {"pos": (1.0, 2.0, 3.0)}
    worker = SimpleNamespace(
        _state_future=future, _get_vehicle_state=lambda: fresh
    )

    state = BeamNgWorker._take_vehicle_state(worker)  # type: ignore[arg-type]

    assert state is fresh
    assert worker._state_future is None


def test_without_a_prefetch_the_state_is_polled_synchronously() -> None:
    fresh = {"pos": (1.0, 2.0, 3.0)}
    worker = SimpleNamespace(
        _state_future=None, _get_vehicle_state=lambda: fresh
    )

    assert BeamNgWorker._take_vehicle_state(worker) is fresh  # type: ignore[arg-type]


def test_dropping_the_prefetch_clears_it_for_teardown() -> None:
    future: Future = Future()
    future.set_result(({}, 0.0))
    worker = SimpleNamespace(_state_future=future)

    BeamNgWorker._drop_state_future(worker)  # type: ignore[arg-type]

    assert worker._state_future is None


# --- The two-rate scene build ------------------------------------------------

_POINTS = (
    (0.0, 5.0, -0.4),
    (0.25, 5.0, -0.4),
    (0.5, 5.0, -0.4),
    (3.0, 5.0, 1.0),
    (3.0, 5.0, 0.2),
    (3.0, 5.0, -0.4),
)
_GROUPS = (
    SCENE_ROAD,
    SCENE_ROAD,
    SCENE_ROAD,
    SCENE_BOUNDARY,
    SCENE_BOUNDARY,
    SCENE_BOUNDARY,
)


def test_a_compose_tick_reproduces_the_full_frame_when_nothing_moved() -> None:
    assembler = WorldSceneAssembler()
    snapshot = _snapshot(_POINTS, _GROUPS)

    full = assembler.update(snapshot)
    composed = assembler.update(snapshot, refresh_stores=False)

    assert len(full.road_vertices)
    assert len(full.boundary_vertices)
    np.testing.assert_array_equal(composed.road_vertices, full.road_vertices)
    np.testing.assert_array_equal(composed.road_colors, full.road_colors)
    np.testing.assert_array_equal(composed.road_indices, full.road_indices)
    np.testing.assert_array_equal(
        composed.boundary_vertices, full.boundary_vertices
    )
    np.testing.assert_array_equal(
        composed.boundary_colors, full.boundary_colors
    )


def test_a_compose_tick_tracks_the_ego_without_touching_the_stores() -> None:
    assembler = WorldSceneAssembler()
    full = assembler.update(_snapshot(_POINTS, _GROUPS))
    road_cells = len(assembler._road_keys)
    voxels = len(assembler._voxel_keys)

    # The ego advances 1 m and the snapshot carries NEW returns; a compose tick
    # must re-aim the cached meshes at the new pose and must NOT ingest the
    # cloud.
    moved = _snapshot(
        ((10.0, 10.0, -0.4),),
        (SCENE_ROAD,),
        ego_pos_world=(0.0, 1.0, 0.0),
        timestamp=0.04,
    )
    composed = assembler.update(moved, refresh_stores=False)

    assert len(assembler._road_keys) == road_cells
    assert len(assembler._voxel_keys) == voxels
    # Render z is -(forward offset): the same world geometry, seen from 1 m
    # further forward, moves 1 m toward +z.
    np.testing.assert_allclose(
        composed.road_vertices[:, 2],
        full.road_vertices[:, 2] + 1.0,
        atol=1e-4,
    )


def test_a_teleport_on_a_compose_tick_still_resets_the_scene() -> None:
    assembler = WorldSceneAssembler()
    assembler.update(_snapshot(_POINTS, _GROUPS))
    assert len(assembler._road_keys)

    jump = float(WORLD_POSE_JUMP_RESET_M) + 10.0
    frame = assembler.update(
        _snapshot((), (), ego_pos_world=(jump, 0.0, 0.0), timestamp=0.04),
        refresh_stores=False,
    )

    # clear() dropped the stores AND the mesh cache, and the forced rebuild
    # drew from the (now empty) stores rather than presenting stale meshes.
    assert not len(assembler._road_keys)
    assert not len(frame.road_vertices)
    assert assembler._travelled_m == 0.0


def test_scene_worker_refreshes_the_stores_on_its_own_clock() -> None:
    class AssemblerStub:
        def __init__(self) -> None:
            self.refreshes: list[bool] = []

        def update(self, snapshot: object, *, refresh_stores: bool = True) -> str:
            self.refreshes.append(refresh_stores)
            return "frame"

        def clear(self) -> None:
            pass

    assembler = AssemblerStub()
    worker = SceneWorker(assembler=assembler)

    for tick in range(3):
        worker.submit(SimpleNamespace(timestamp=float(tick)))  # type: ignore[arg-type]
        worker._process_pending()

    # Three back-to-back snapshots land inside one refresh interval: the first
    # builds the stores, the ones behind it only re-present.
    assert assembler.refreshes == [True, False, False]

    worker.clear()
    worker.submit(SimpleNamespace(timestamp=9.0))  # type: ignore[arg-type]
    worker._process_pending()
    assert assembler.refreshes[-1] is True
