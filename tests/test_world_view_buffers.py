from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev.models import WorldActor
from beamng_lidar_bev.world_view import (
    _VERTEX_STRIDE,
    ActorListModel,
    interleave,
)


def test_vertices_and_colours_interleave_into_the_stride_qt_is_told_about() -> None:
    """
    The buffer layout is a contract with `SceneGeometry.addAttribute`: position
    at offset 0, colour at offset 12, 28 bytes apart. Nothing checks it at
    runtime -- a mismatch renders as garbage geometry rather than raising -- so
    it is pinned here.
    """
    buffer = interleave(
        np.asarray(((1.0, 2.0, 3.0), (-1.0, 0.0, 4.0)), dtype=np.float32),
        np.asarray(((0.1, 0.2, 0.3, 1.0), (0.4, 0.5, 0.6, 1.0)), dtype=np.float32),
    )

    assert buffer.dtype == np.float32
    assert buffer.shape == (2, 7)
    assert buffer.strides[0] == _VERTEX_STRIDE
    np.testing.assert_allclose(buffer[0], (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0))
    np.testing.assert_allclose(buffer[1], (-1.0, 0.0, 4.0, 0.4, 0.5, 0.6, 1.0))
    assert len(buffer.tobytes()) == 2 * _VERTEX_STRIDE


def test_interleave_rejects_a_colour_buffer_that_does_not_match() -> None:
    with pytest.raises(ValueError, match="2 vertices against 1 colours"):
        interleave(np.zeros((2, 3), np.float32), np.zeros((1, 4), np.float32))


def test_actor_model_exposes_stable_scene_roles() -> None:
    model = ActorListModel()
    actor = WorldActor(
        actor_id="traffic",
        kind="car",
        position=(2.0, 0.5, -12.0),
        yaw_deg=15.0,
        scale=(1.9, 1.5, 4.5),
        confidence=0.8,
    )

    model.set_actors((actor,))

    roles = {bytes(name).decode(): role for role, name in model.roleNames().items()}
    index = model.index(0, 0)
    assert model.rowCount() == 1
    assert model.data(index, roles["actorId"]) == "traffic"
    assert model.data(index, roles["kind"]) == "car"
    assert model.data(index, roles["x"]) == 2.0
    assert model.data(index, roles["confidence"]) == 0.8


def test_actor_model_updates_stable_ids_without_resetting_delegates() -> None:
    model = ActorListModel()
    resets: list[None] = []
    changes: list[None] = []
    model.modelReset.connect(lambda: resets.append(None))
    model.dataChanged.connect(lambda *_: changes.append(None))
    first = WorldActor(
        actor_id="traffic",
        kind="car",
        position=(2.0, 0.5, -12.0),
        yaw_deg=15.0,
        scale=(1.9, 1.5, 4.5),
        confidence=0.8,
    )
    moved = WorldActor(
        actor_id="traffic",
        kind="car",
        position=(2.4, 0.5, -12.8),
        yaw_deg=18.0,
        scale=(1.9, 1.5, 4.5),
        confidence=1.0,
    )

    model.set_actors((first,))
    model.set_actors((moved,))

    assert len(resets) == 1
    assert len(changes) == 1
    assert model.data(model.index(0, 0), ActorListModel.XRole) == 2.4
