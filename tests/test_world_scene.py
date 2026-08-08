from __future__ import annotations

import numpy as np
import pytest

from beamng_lidar_bev import config
from beamng_lidar_bev.models import (
    ARMED,
    BRAKING,
    STANDBY,
    ActorObservation,
    AebState,
    PerceptionSnapshot,
    VehicleGeometry,
)
from beamng_lidar_bev.semantics import SCENE_BOUNDARY, SCENE_ROAD, SCENE_VEHICLE
from beamng_lidar_bev.world_scene import (
    WorldSceneAssembler,
    path_ribbon,
    world_to_render,
)

GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.5,
    left_m=1.0,
    right_m=1.0,
    front_m=2.0,
    rear_m=2.0,
    height_m=1.5,
    mounts={},
)


def test_perception_snapshot_rejects_mismatched_semantic_groups() -> None:
    with pytest.raises(ValueError, match="point and semantic-group"):
        PerceptionSnapshot(
            points_world=np.zeros((2, 3), dtype=np.float32),
            semantic_groups=np.zeros(1, dtype=np.uint8),
            ego_pos_world=(0.0, 0.0, 0.0),
            ego_dir_world=(0.0, 1.0, 0.0),
            ego_up_world=(0.0, 0.0, 1.0),
            timestamp=0.0,
            speed_mps=0.0,
            vehicle_geometry=GEOMETRY,
        )


def _snapshot(
    points: tuple[tuple[float, float, float], ...] = (),
    groups: tuple[int, ...] = (),
    *,
    timestamp: float = 0.0,
    actors: tuple[ActorObservation, ...] = (),
    speed_mps: float = 0.0,
    forward_speed_mps: float | None = None,
    ego_pos_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    aeb: object = None,
    rear_aeb: object = None,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        points_world=np.asarray(points, dtype=np.float32).reshape(-1, 3),
        semantic_groups=np.asarray(groups, dtype=np.uint8),
        ego_pos_world=ego_pos_world,
        ego_dir_world=(0.0, 1.0, 0.0),
        ego_up_world=(0.0, 0.0, 1.0),
        timestamp=timestamp,
        speed_mps=speed_mps,
        forward_speed_mps=(
            speed_mps if forward_speed_mps is None else forward_speed_mps
        ),
        vehicle_geometry=GEOMETRY,
        actors=actors,
        aeb=aeb,  # type: ignore[arg-type]
        rear_aeb=rear_aeb,  # type: ignore[arg-type]
    )


def test_world_to_render_is_right_up_negative_forward() -> None:
    snapshot = _snapshot(((0.0, 1.0, 0.0),), (SCENE_ROAD,))

    rendered = world_to_render(snapshot.points_world, snapshot)

    np.testing.assert_allclose(rendered, ((0.0, 0.0, -1.0),))


def test_path_ribbon_has_two_vertices_per_sample_and_triangle_indices() -> None:
    vertices, indices = path_ribbon(
        np.asarray(((0.0, 0.0), (0.0, 5.0), (2.0, 10.0)), dtype=np.float32),
        1.0,
    )

    assert vertices.shape == (6, 3)
    assert indices.shape == (12,)
    assert vertices.dtype == np.float32
    assert indices.dtype == np.uint32
    triangle = vertices[indices[:3]]
    assert np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[1] > 0


def test_temporal_road_cells_expire_after_ttl() -> None:
    assembler = WorldSceneAssembler()
    road = _snapshot(
        ((0.0, 3.0, -0.5), (0.4, 3.0, -0.5), (0.0, 3.4, -0.5)),
        (SCENE_ROAD, SCENE_ROAD, SCENE_ROAD),
    )

    assert assembler.update(road).road_indices.size
    assert not assembler.update(_snapshot(timestamp=1.3)).road_indices.size


def test_road_mesh_is_bounded_by_radius_and_clears_after_pose_jump() -> None:
    assembler = WorldSceneAssembler()
    far = _snapshot(((0.0, 90.0, -0.5),), (SCENE_ROAD,))
    near = _snapshot(((0.0, 3.0, -0.5),), (SCENE_ROAD,), timestamp=0.1)

    assert not assembler.update(far).road_indices.size
    assert assembler.update(near).road_indices.size
    jumped = assembler.update(
        _snapshot(
            timestamp=0.2,
            ego_pos_world=(50.0, 0.0, 0.0),
        )
    )
    assert not jumped.road_indices.size


def test_the_road_is_one_continuous_surface_with_shared_corners() -> None:
    """
    The road used to be `merge_cell_runs` rectangles drawn flat at the mean
    height of the cells behind them. Adjacent rectangles share no vertices, so
    every difference in mean height was a hard step and the surface read as a
    quilt of flat plates -- the "big pixels" complaint. Cells now share their
    lattice corners, which makes the surface continuous by construction: a
    corner holds ONE height, so two neighbouring cells cannot disagree about it.
    """
    step = float(config.WORLD_CELL_SIZE_M)
    points = tuple(
        (x + step * 0.5, y + step * 0.5, -0.5)
        for x in np.arange(-1.0, 1.0, step)
        for y in np.arange(1.0, 3.0, step)
    )
    cells = len(points)

    frame = WorldSceneAssembler().update(_snapshot(points, (SCENE_ROAD,) * cells))

    # Two triangles a cell, but far fewer vertices than 4 per cell, because the
    # interior corners are shared. An 8x8 block has 81 corners, not 256.
    assert frame.road_indices.shape == (cells * 6,)
    assert len(frame.road_vertices) < cells * 4
    assert len(frame.road_vertices) == pytest.approx(
        (np.sqrt(cells) + 1) ** 2, rel=0.05
    )
    # Every index must address a real vertex, or the mesh renders as garbage.
    assert frame.road_indices.max() < len(frame.road_vertices)


def test_a_sloping_road_has_no_steps_between_neighbouring_cells() -> None:
    """
    The quilt's real artefact: two abutting plates at different mean heights
    left a vertical wall between them. Sharing corners removes the failure
    rather than reducing it -- along a ramp, consecutive corner heights must
    advance smoothly instead of jumping by a plate's worth.
    """
    step = float(config.WORLD_CELL_SIZE_M)
    points = tuple(
        (x + step * 0.5, y + step * 0.5, -0.5 + 0.08 * y)
        for x in np.arange(-1.0, 1.0, step)
        for y in np.arange(1.0, 9.0, step)
    )

    frame = WorldSceneAssembler().update(_snapshot(points, (SCENE_ROAD,) * len(points)))

    # Walk one column of the surface from near to far and check the rise per
    # cell never exceeds what the slope itself justifies.
    vertices = frame.road_vertices
    column = vertices[np.isclose(vertices[:, 0], vertices[:, 0].min())]
    ordered = column[np.argsort(-column[:, 2])]
    rise = np.abs(np.diff(ordered[:, 1]))
    assert len(ordered) > 8
    assert rise.max() < 0.08 * step * 2.0


def test_the_camera_pulls_back_and_climbs_with_speed() -> None:
    """
    The view should cover roughly the distance about to be travelled, not a
    fixed patch of road: close in for parking, well back at speed.
    """
    parked = WorldSceneAssembler().update(_snapshot(speed_mps=0.0))
    cruising = WorldSceneAssembler().update(_snapshot(speed_mps=12.0))
    fast = WorldSceneAssembler().update(_snapshot(speed_mps=32.0))

    heights = [frame.camera_position[1] for frame in (parked, cruising, fast)]
    distances = [frame.camera_position[2] for frame in (parked, cruising, fast)]
    assert heights == sorted(heights)
    assert distances == sorted(distances)
    assert distances[2] > distances[0] * 2.5, (
        "the zoom barely responds to speed"
    )
    # Both stay clamped, or a fast straight puts the camera in orbit.
    assert heights[2] <= config.WORLD_CAM_HEIGHT_MAX_M
    assert distances[2] <= config.WORLD_CAM_DISTANCE_MAX_M


def test_the_camera_swings_round_when_reversing_without_self_driving() -> None:
    """
    Reversing is read from the SIGNED forward speed, not from the plan.

    `plan` is None whenever self-driving is off -- which is exactly when a human
    is doing the reversing -- so keying the swing on the plan's REVERSING mode
    meant the camera only ever turned round for the autonomous reverse
    recovery, and never for a driver selecting reverse themselves.
    """
    forward = WorldSceneAssembler().update(
        _snapshot(speed_mps=4.0, forward_speed_mps=4.0)
    )
    backward = WorldSceneAssembler().update(
        _snapshot(speed_mps=4.0, forward_speed_mps=-4.0)
    )

    assert forward.camera_euler[1] == pytest.approx(0.0)
    assert forward.camera_position[2] > 0.0
    assert backward.camera_euler[1] == pytest.approx(180.0)
    assert backward.camera_position[2] < 0.0


def test_creeping_forward_does_not_flip_the_camera() -> None:
    """A car rolling almost imperceptibly forward is not reversing."""
    frame = WorldSceneAssembler().update(
        _snapshot(speed_mps=0.1, forward_speed_mps=-0.05)
    )

    assert frame.camera_euler[1] == pytest.approx(0.0)


def test_ground_truth_actor_is_hidden_until_lidar_corroborates_it() -> None:
    actor = ActorObservation(
        actor_id="traffic",
        kind="car",
        pos_world=(0.0, 6.0, 0.0),
        dir_world=(0.0, 1.0, 0.0),
        velocity_world=(0.0, 0.0, 0.0),
        dimensions_m=(2.0, 1.5, 4.4),
    )
    assembler = WorldSceneAssembler()

    assert assembler.update(_snapshot(actors=(actor,))).actors == ()

    hits = (
        (-0.5, 5.5, 0.2),
        (0.0, 5.7, 0.3),
        (0.5, 5.9, 0.2),
        (-0.3, 6.2, 0.4),
    )
    frame = assembler.update(
        _snapshot(
            hits,
            (SCENE_VEHICLE,) * len(hits),
            timestamp=0.1,
            actors=(actor,),
        )
    )

    assert len(frame.actors) == 1
    assert frame.actors[0].actor_id == "traffic"

    assert assembler.update(
        _snapshot(timestamp=0.4, actors=(actor,))
    ).actors
    assert not assembler.update(
        _snapshot(timestamp=1.0, actors=(actor,))
    ).actors


def _wall(
    x_from: float,
    x_to: float,
    y: float,
    z_low: float,
    z_high: float,
    step: float = 0.25,
) -> np.ndarray:
    """A dense vertical surface, as LiDAR would paint it with no occlusion."""
    xs = np.arange(x_from, x_to, step)
    zs = np.arange(z_low, z_high, step)
    grid_x, grid_z = np.meshgrid(xs, zs)
    return np.column_stack(
        (grid_x.ravel(), np.full(grid_x.size, y), grid_z.ravel())
    )


def _boundary_snapshot(points: np.ndarray, **kwargs: object) -> PerceptionSnapshot:
    return _snapshot(
        tuple(map(tuple, points)),
        (SCENE_BOUNDARY,) * len(points),
        **kwargs,  # type: ignore[arg-type]
    )


def _boxes(frame: object) -> np.ndarray:
    """
    Boundary slabs as an (M, 24, 3) array of render-space corners.

    Twenty-four, not eight: each of the six faces carries its own shade, and a
    shared corner belongs to three faces so it can only hold one colour.
    """
    return frame.boundary_vertices.reshape(-1, 24, 3)  # type: ignore[attr-defined]


def test_a_wall_becomes_one_extruded_slab_not_a_cloud_of_billboards() -> None:
    """
    Boundary returns used to be one 0.16 x 0.32 m card per point, so a wall was
    drawn as confetti with gaps between the confetti and 70-80% of the evidence
    thrown away by the render cap. It is now accumulated into world voxels and
    extruded, which is both solid and far cheaper.
    """
    points = _wall(-5.0, 5.0, 12.0, 0.0, 4.0)
    frame = WorldSceneAssembler().update(_boundary_snapshot(points))

    boxes = _boxes(frame)
    assert len(boxes) == 1
    # Six quads of four corners, against 640 points x 4 verts the old way.
    assert frame.boundary_vertices.shape == (24, 3)
    assert frame.boundary_indices.shape == (36,)
    heights = boxes[0][:, 1]
    assert heights.min() == pytest.approx(0.0, abs=0.3)
    assert heights.max() == pytest.approx(3.75, abs=0.3)


def test_every_face_of_a_slab_is_shaded_separately() -> None:
    """
    The reason a box costs 24 vertices. Unlit geometry of one flat colour has
    no edges whatsoever, so abutting slabs merge into a single silhouette --
    half of the reported "wall and building look like one blob".
    """
    frame = WorldSceneAssembler().update(
        _boundary_snapshot(_wall(-5.0, 5.0, 12.0, 0.0, 4.0))
    )

    assert len(frame.boundary_colors) == len(frame.boundary_vertices)
    faces = frame.boundary_colors.reshape(-1, 6, 4, 4)[0][..., :3]

    # Six faces, six shades, and the gap between them has to dominate the
    # gradient the depth tint puts ACROSS a face -- a 10 m wall's ends really
    # are further away than its middle, which is wanted, but if that gradient
    # were the larger of the two the faces would not read as separate planes.
    means = faces.mean(axis=1)
    within = float(np.ptp(faces, axis=1).max())
    ordered = np.sort(means.mean(axis=1))
    between = float(np.diff(ordered).min())
    assert len(np.unique(np.round(means, 5), axis=0)) == 6, (
        "two faces of one slab share a colour"
    )
    assert between > within, (
        f"faces differ by {between:.4f} but a single face already varies by "
        f"{within:.4f} across its own width"
    )


def test_azimuth_stripe_gaps_are_bridged_but_a_real_opening_is_not() -> None:
    """
    A wall is sampled at r*dazimuth horizontally -- 1.24 m at 20 m against 0.5 m
    columns -- so it arrives as vertical stripes and extruding stripes just
    gives striped buildings. Bridging is bounded by WORLD_COLUMN_BRIDGE_CELLS
    precisely so it can never close a gap the car could drive through.
    """
    dense = _wall(-5.0, 5.0, 12.0, 0.0, 4.0)
    striped = dense[(np.floor(dense[:, 0] / 0.5).astype(int) % 3) == 0]
    bridged = _boxes(WorldSceneAssembler().update(_boundary_snapshot(striped)))
    assert len(bridged) == 1
    assert bridged[0][:, 0].max() - bridged[0][:, 0].min() > 9.0

    gapped = np.concatenate(
        (_wall(-9.0, -2.0, 12.0, 0.0, 4.0), _wall(2.0, 9.0, 12.0, 0.0, 4.0))
    )
    split = _boxes(WorldSceneAssembler().update(_boundary_snapshot(gapped)))
    assert len(split) == 2
    left, right = sorted(split, key=lambda box: box[:, 0].mean())
    assert left[:, 0].max() == pytest.approx(-2.0, abs=0.6)
    assert right[:, 0].min() == pytest.approx(2.0, abs=0.6)


def test_a_kerb_in_front_of_a_facade_does_not_average_into_it() -> None:
    """
    Columns only merge when their slab heights share a bucket. Without that a
    6 m facade and the 0.15 m kerb in front of it would average into one
    waist-high wall across the whole scene.
    """
    mixed = np.concatenate(
        (
            _wall(-5.0, 5.0, 20.0, 0.0, 6.0),
            _wall(-5.0, 5.0, 8.0, 0.0, 0.15, step=0.05),
        )
    )
    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(mixed)))

    by_distance = sorted(boxes, key=lambda box: -box[:, 2].mean())
    kerb, facade = by_distance[0], by_distance[-1]
    assert kerb[:, 1].max() - kerb[:, 1].min() < 0.4
    assert facade[:, 1].max() - facade[:, 1].min() > 5.0


def test_boundary_columns_accumulate_across_frames_and_then_expire() -> None:
    """
    Accumulation is the whole point: ego motion sweeps the azimuth stripes
    across a wall, and only a world-anchored store keeps what earlier frames
    saw. Nothing was accumulated at all before -- the mesh was rebuilt from the
    current snapshot every tick.
    """
    assembler = WorldSceneAssembler()
    assembler.update(_boundary_snapshot(_wall(-5.0, 5.0, 12.0, 0.0, 4.0)))

    empty = np.empty((0, 3), dtype=np.float32)
    assert len(_boxes(assembler.update(_boundary_snapshot(empty, timestamp=2.0))))
    assert not len(
        _boxes(assembler.update(_boundary_snapshot(empty, timestamp=9.0)))
    )


def _canopy(
    x_from: float,
    x_to: float,
    y_from: float,
    y_to: float,
    z_low: float,
    z_high: float,
    step: float = 0.2,
) -> np.ndarray:
    """A block of foliage floating clear of the ground, as a tree really is."""
    xs = np.arange(x_from, x_to, step)
    ys = np.arange(y_from, y_to, step)
    zs = np.arange(z_low, z_high, step)
    grid = np.meshgrid(xs, ys, zs)
    return np.column_stack([axis.ravel() for axis in grid])


def _ground(
    x_from: float, x_to: float, y_from: float, y_to: float, step: float = 0.2
) -> np.ndarray:
    """
    Grass under the tree.

    This is the detail that made the bug: grass is not road, and everything
    that is not road, vehicle or vulnerable is a BOUNDARY return -- so the
    column under a canopy holds returns at ankle height as well as at 5 m.
    """
    xs = np.arange(x_from, x_to, step)
    ys = np.arange(y_from, y_to, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, -0.48))
    )


def test_a_tree_canopy_does_not_reach_down_to_the_ground() -> None:
    """
    Reported from the running app: a tree drew as though its branches and
    leaves ran from the treetop all the way down, which is not what a tree
    looks like.

    The store kept one (min, max) span per XY column and extruded it. Grass
    under the tree is a boundary return, so the column held 0.5 m and 5 m and
    the span smeared the canopy into the grass. Voxels keep the void between
    them, so the canopy is its own run -- and a run with metres of clear air
    beneath it is something the car drives under, so it is not drawn at all.
    """
    scene = np.concatenate(
        (
            _ground(-3.0, 3.0, 9.0, 15.0),
            _canopy(-2.0, 2.0, 10.0, 14.0, 2.8, 5.6),
        )
    )

    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(scene)))

    assert not len(boxes), (
        "the canopy is drivable-under and the grass is too low to be a slab, "
        "so nothing here is an obstacle"
    )


def test_a_trunk_survives_while_the_canopy_above_it_is_dropped() -> None:
    """
    The half of the tree you CAN hit still has to be drawn. The trunk is rooted
    at the ground, so its run starts at the floor and is kept whatever height
    it reaches; only the run floating above clear air is culled.
    """
    scene = np.concatenate(
        (
            _ground(-3.0, 3.0, 9.0, 15.0),
            _canopy(-2.0, 2.0, 10.0, 14.0, 2.8, 5.6),
            _canopy(-0.2, 0.2, 11.8, 12.2, -0.45, 2.8, step=0.1),
        )
    )

    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(scene)))

    assert len(boxes), "the trunk is solid and rooted, so it must be drawn"
    footprint = max(
        box[:, 0].max() - box[:, 0].min() for box in boxes
    )
    assert footprint < 1.0, "the canopy footprint came back, not just the trunk"


def test_a_bridge_deck_over_the_road_is_not_drawn_as_a_wall() -> None:
    """
    The same defect with no ground return at all to reference: under a bridge
    the surface below is ROAD, which never enters the boundary store, so the
    deck's column has no floor of its own. The ego's ground plane stands in,
    and the deck is recognised as something to drive under.
    """
    deck = _canopy(-6.0, 6.0, 18.0, 21.0, 4.6, 5.4)

    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(deck)))

    assert not len(boxes)


def test_a_wall_is_not_culled_just_for_being_tall() -> None:
    """
    The cull tests where a structure STARTS, never where it ends. A building is
    rooted at the ground and stays at full height -- clipping everything to
    roof height would make a facade and a garden wall the same object, which is
    the distinction the view exists to show.
    """
    scene = np.concatenate(
        (_ground(-6.0, 6.0, 9.0, 15.0), _wall(-5.0, 5.0, 14.0, -0.45, 9.0))
    )

    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(scene)))

    assert len(boxes)
    tallest = max(box[:, 1].max() - box[:, 1].min() for box in boxes)
    assert tallest > 8.0, f"a 9 m facade came back only {tallest:.1f} m tall"


def test_a_kerb_and_the_facade_behind_it_stay_two_slabs() -> None:
    """
    Both are rooted, both are hazards, and they must not merge into one mass --
    this is the "wall in front of a building" case in geometry rather than in
    colour. They are separated by the altitude AND height buckets, so a low
    obstacle in front of a tall one keeps its own box.
    """
    scene = np.concatenate(
        (
            _wall(-5.0, 5.0, 10.0, -0.45, 0.4, step=0.05),
            _wall(-5.0, 5.0, 16.0, -0.45, 7.0),
        )
    )

    boxes = _boxes(WorldSceneAssembler().update(_boundary_snapshot(scene)))

    by_distance = sorted(boxes, key=lambda box: -box[:, 2].mean())
    assert len(by_distance) >= 2
    kerb, facade = by_distance[0], by_distance[-1]
    assert kerb[:, 1].max() - kerb[:, 1].min() < 1.2
    assert facade[:, 1].max() - facade[:, 1].min() > 6.0


def _vehicle_boxes(frame: object) -> np.ndarray:
    return frame.vehicle_vertices.reshape(-1, 24, 3)  # type: ignore[attr-defined]


def test_a_car_is_drawn_from_lidar_alone_with_no_ground_truth_actor() -> None:
    """
    Reported from the running app: WORLD showed other cars as nothing at all --
    not as a model, not even as an obstacle.

    Two failures compounded. Vehicle returns were excluded from the geometry on
    the understanding that traffic would be drawn as corroborated actor models
    instead; and that path needs `vehicles.get_states()`, which BeamNG.tech
    REJECTS when the map was loaded through free-roam (see the comment on
    `worker._get_vehicle_state`). In the normal workflow no actor is ever
    confirmed, so a car was drawn by neither route.

    Perception first: a car is a solid the LiDAR saw. The ground-truth model is
    enrichment on top of that, never the thing that makes it visible.
    """
    car = _wall(-0.9, 0.9, 14.0, -0.45, 1.4, step=0.1)
    groups = (SCENE_VEHICLE,) * len(car)

    frame = WorldSceneAssembler().update(
        _snapshot(tuple(map(tuple, car)), groups)
    )

    assert frame.actors == (), "no simulator actor was offered"
    boxes = _vehicle_boxes(frame)
    assert len(boxes), "a car with no confirmed actor is invisible"
    assert boxes[0][:, 1].max() - boxes[0][:, 1].min() > 1.2


def test_traffic_and_scenery_are_separate_meshes_so_a_car_keeps_its_colour(
) -> None:
    """
    Split at the run level rather than meshed together and recoloured. A car
    parked against a wall would otherwise merge into it and take one colour for
    both, and hue is the only thing separating traffic from scenery -- they
    share the single dark obstacle band by design.
    """
    wall = _wall(-6.0, 6.0, 14.5, -0.45, 3.0)
    car = _wall(-0.9, 0.9, 14.0, -0.45, 1.4, step=0.1)
    points = np.concatenate((wall, car))
    groups = (SCENE_BOUNDARY,) * len(wall) + (SCENE_VEHICLE,) * len(car)

    frame = WorldSceneAssembler().update(
        _snapshot(tuple(map(tuple, points)), groups)
    )

    assert len(_boxes(frame)), "the wall vanished"
    assert len(_vehicle_boxes(frame)), "the car was absorbed into the wall"
    # Different meshes, and demonstrably different colours.
    wall_colour = frame.boundary_colors[:, :3].mean(axis=0)
    car_colour = frame.vehicle_colors[:, :3].mean(axis=0)
    assert not np.allclose(wall_colour, car_colour, atol=0.01)


def test_traffic_expires_far_sooner_than_scenery_so_a_car_does_not_smear(
) -> None:
    """
    Everything else in the store is static scenery that only improves with more
    looks. A car MOVES: on the scenery TTL one car is drawn as a four-second
    streak of itself.
    """
    assembler = WorldSceneAssembler()
    car = _wall(-0.9, 0.9, 14.0, -0.45, 1.4, step=0.1)
    wall = _wall(-6.0, 6.0, 25.0, -0.45, 3.0)
    points = np.concatenate((wall, car))
    groups = (SCENE_BOUNDARY,) * len(wall) + (SCENE_VEHICLE,) * len(car)
    assembler.update(_snapshot(tuple(map(tuple, points)), groups))

    later = assembler.update(_snapshot(timestamp=0.5))

    assert not len(_vehicle_boxes(later)), "the car smeared past its own TTL"
    assert len(_boxes(later)), "the scenery expired with the traffic"


def test_a_slab_keeps_the_tallest_look_it_ever_got() -> None:
    """
    The vertical FOV means a wall's observed top FALLS as you approach it, so
    taking the current frame's maximum would shrink buildings as you drove at
    them. The column keeps the highest return seen inside the TTL.
    """
    assembler = WorldSceneAssembler()
    assembler.update(_boundary_snapshot(_wall(-3.0, 3.0, 12.0, 0.0, 6.0)))
    close_up = assembler.update(
        _boundary_snapshot(_wall(-3.0, 3.0, 12.0, 0.0, 1.5), timestamp=0.2)
    )

    assert _boxes(close_up)[0][:, 1].max() > 5.0


def _aeb_state(
    status: str,
    *,
    threat_m: float | None,
    rearward: bool = False,
    curvature: float = 0.0,
) -> AebState:
    return AebState(
        status=status,
        brake=1.0 if status == BRAKING else 0.0,
        curvature=curvature,
        rearward=rearward,
        threat_m=threat_m,
        horizon_m=40.0,
        standoff_m=0.6,
        brake_now_m=12.0,
        corridor_half_width_m=1.2,
        required_decel_mps2=0.0,
        time_to_collision_s=float("inf"),
        reason="test",
    )


def test_standby_aeb_draws_nothing() -> None:
    """Below the arming speed there is no corridor to show."""
    frame = WorldSceneAssembler().update(
        _snapshot(aeb=_aeb_state(STANDBY, threat_m=None))
    )

    assert not frame.aeb_indices.size
    assert not frame.aeb_marker_indices.size


def test_an_armed_aeb_draws_its_corridor_but_no_threat_marker() -> None:
    frame = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=None))
    )

    assert frame.aeb_indices.size, "an armed corridor should be visible"
    assert not frame.aeb_marker_indices.size, "nothing is being braked for"
    # Drawn ahead of the car, and no wider than the corridor that was scanned.
    assert frame.aeb_vertices[:, 2].min() < -1.0
    assert np.abs(frame.aeb_vertices[:, 0]).max() <= 1.2 + 1e-4


def test_the_threat_marker_stands_at_the_measured_threat_distance() -> None:
    """
    Filled only as far as the blockage, never to the horizon: filling to the
    horizon reads as "all 40 m of this is dangerous" when the point that matters
    is the one about to be hit.
    """
    frame = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(BRAKING, threat_m=9.0))
    )

    assert frame.aeb_marker_indices.size
    # Straight ahead, so the panel and its frame stand at -9 m on render z, and
    # nothing in the marker reaches past the thing being braked for.
    assert frame.aeb_marker_vertices[:, 2].min() == pytest.approx(-9.0, abs=0.05)
    # The pool trails back toward the car rather than beyond the threat, so the
    # marker is anchored to a place on the road instead of hanging in the air.
    trailing = frame.aeb_marker_vertices[:, 2].max()
    assert -9.0 < trailing <= -9.0 + config.WORLD_AEB_POOL_LENGTH_M + 0.05
    # The panel stands up off the surface.
    assert (
        frame.aeb_marker_vertices[:, 1].max()
        - frame.aeb_marker_vertices[:, 1].min()
    ) == pytest.approx(config.WORLD_AEB_MARKER_HEIGHT_M, abs=0.05)
    # The WASH stops at the threat, but the RAILS run on to the scanned horizon:
    # how far the corridor was checked is information a filled band cannot give.
    assert frame.aeb_vertices[:, 2].min() == pytest.approx(-40.0, abs=0.5)


def test_the_rear_system_is_drawn_behind_the_car() -> None:
    """
    The rear system reasons in a 180-degree ROTATED frame -- that is what lets
    it share every arc helper in `planner` unchanged -- so the overlay has to
    un-rotate it, exactly as the BEV overlay does.
    """
    frame = WorldSceneAssembler().update(
        _snapshot(
            speed_mps=3.0,
            forward_speed_mps=-3.0,
            rear_aeb=_aeb_state(BRAKING, threat_m=5.0, rearward=True),
        )
    )

    assert frame.aeb_marker_indices.size
    # Mirrored: the panel stands at +5 m and the pool trails back toward the car.
    assert frame.aeb_marker_vertices[:, 2].max() == pytest.approx(5.0, abs=0.05)
    trailing = frame.aeb_marker_vertices[:, 2].min()
    assert 5.0 - config.WORLD_AEB_POOL_LENGTH_M - 0.05 <= trailing < 5.0
    # Nothing of the rear system may stray in front of the car.
    assert frame.aeb_vertices[:, 2].min() >= -0.01


def test_braking_and_armed_are_different_colours() -> None:
    """The colour change IS the event: violet watching, red acting."""
    armed = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=None))
    )
    braking = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(BRAKING, threat_m=9.0))
    )

    assert not np.allclose(
        armed.aeb_colors[0, :3], braking.aeb_colors[0, :3], atol=0.01
    )


def test_both_aeb_systems_can_be_drawn_at_once() -> None:
    """One mesh carries both; the index buffers have to be rebased, not just
    concatenated, or the second corridor renders as garbage."""
    frame = WorldSceneAssembler().update(
        _snapshot(
            speed_mps=6.0,
            aeb=_aeb_state(ARMED, threat_m=None),
            rear_aeb=_aeb_state(ARMED, threat_m=None, rearward=True),
        )
    )

    assert frame.aeb_indices.max() < len(frame.aeb_vertices)
    assert frame.aeb_vertices[:, 2].min() < 0.0
    assert frame.aeb_vertices[:, 2].max() > 0.0


def test_the_rails_show_how_far_the_corridor_was_actually_scanned() -> None:
    """
    A filled wash can only say "this stretch is dangerous". The rails say "and I
    checked this far and found nothing", which is a different and equally
    important statement -- without it a clear corridor and a blind one look the
    same.
    """
    clear = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=None))
    )

    assert clear.aeb_vertices[:, 2].min() == pytest.approx(-40.0, abs=0.5)


def test_the_brake_now_bar_marks_the_last_point_to_brake() -> None:
    """
    The trigger the whole system turns on, and the one number that says how much
    margin is left: the gap between this bar and the threat marker.
    """
    frame = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=None))
    )

    # brake_now_m is 12.0 in the fixture, so a band must exist across the
    # corridor there even though nothing has been detected.
    depths = -frame.aeb_vertices[:, 2]
    near_bar = np.abs(depths - 12.0) < 0.5
    assert near_bar.any(), "no geometry at the brake-now distance"


def test_the_wash_strengthens_with_urgency_while_still_only_watching() -> None:
    """
    The corridor builds as a threat closes instead of snapping on at the moment
    of firing. Hue still carries the STATE -- a colour change is what reads as
    an event -- so urgency rides on alpha alone.
    """
    # Both are WATCHING a threat, not braking for it -- urgency is what happens
    # between "something is there" and "the pedal goes down".
    calm = _aeb_state(ARMED, threat_m=22.0)
    urgent = AebState(**{**calm.__dict__, "required_decel_mps2": 9.0})

    calm_frame = WorldSceneAssembler().update(_snapshot(speed_mps=14.0, aeb=calm))
    urgent_frame = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=urgent)
    )

    # The MEAN, not the max: the rails are a fixed strength in both, so a
    # difference can only have come from the wash they enclose.
    assert urgent_frame.aeb_colors[:, 3].mean() > calm_frame.aeb_colors[:, 3].mean()
    # ...and the hue is unchanged, because nothing has fired yet.
    np.testing.assert_allclose(
        urgent_frame.aeb_colors[0, :3], calm_frame.aeb_colors[0, :3], atol=1e-6
    )


def test_every_aeb_element_stays_inside_the_scanned_corridor() -> None:
    """
    Nothing drawn may imply the system is watching somewhere it is not. Every
    element is built from `AebState`'s own curvature and half-width through
    `aeb.predicted_corridor`, so this holds by construction -- and this pins it,
    because an overlay that overstates the corridor is worse than none.
    """
    state = _aeb_state(BRAKING, threat_m=9.0)
    frame = WorldSceneAssembler().update(_snapshot(speed_mps=14.0, aeb=state))

    for vertices in (frame.aeb_vertices, frame.aeb_marker_vertices):
        assert np.abs(vertices[:, 0]).max() <= state.corridor_half_width_m + 1e-4


def test_the_overlay_vanishes_entirely_when_aeb_is_switched_off() -> None:
    """Both toggles off means no overlay at all, not a faint one."""
    frame = WorldSceneAssembler().update(_snapshot(speed_mps=14.0))

    assert not frame.aeb_indices.size
    assert not frame.aeb_marker_indices.size


def test_a_clear_corridor_gets_rails_but_no_wash() -> None:
    """
    Nothing is filled in when nothing was found. Filling the scanned length says
    "all 48 m of this is dangerous" -- the same error as scoring an empty road
    as an obstacle parked at the horizon -- and over a depth-tinted far road it
    rendered as a solid white band down the whole scene.
    """
    clear = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=None))
    )
    blocked = WorldSceneAssembler().update(
        _snapshot(speed_mps=14.0, aeb=_aeb_state(ARMED, threat_m=20.0))
    )

    assert clear.aeb_indices.size, "the rails should still show the scan extent"
    # The wash is the only thing that differs, so a blocked corridor must carry
    # strictly more geometry than a clear one of the same length.
    assert len(blocked.aeb_vertices) > len(clear.aeb_vertices)
    # The wash is the FAINTEST thing drawn, so its presence shows up as alpha
    # well below anything the rails or the trigger bar ever reach. Comparing
    # means would be wrong in the other direction: adding faint geometry pulls
    # the average down.
    assert clear.aeb_colors[:, 3].min() > 0.2, "a wash crept into a clear corridor"
    assert blocked.aeb_colors[:, 3].min() < 0.1
