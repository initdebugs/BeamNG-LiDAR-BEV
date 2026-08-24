"""
Phase 2, milestone 3: the camera cloud diffed against the LiDAR cloud.

Both rigs on the player's car at once -- the six LiDAR units exactly as the
app builds them, the eight cameras with depth and annotation exactly as the
app builds them -- captured in LOCKSTEP (pause, step, read both), then the
cameras are unprojected through the real `unprojection` module and the two
clouds are compared through the real pipeline: the BEV transform, the
semantic split, and both obstacle bands (`geometric_obstacle_sets`, the same
call `worker._poll_once` makes).

    .venv39\\Scripts\\python tools\\unprojection_oracle.py capture
    .venv39\\Scripts\\python tools\\unprojection_oracle.py analyse ^
        tools\\oracle_data\\scene.npz --png

What the report answers, in order of how much each would cost to be wrong:

* **Handedness.** Whether the simulator's column index runs the way
  `geometry.camera_basis` assumes. The offline suite cannot know; this is the
  one sign in the whole rung that the live scene decides. Measured by the
  planner-band occupancy IoU of the camera cloud against the LiDAR cloud,
  direct AND mirrored -- a mirrored cloud scores worse than the direct one
  on any scene that is not left-right symmetric.
* **The ground band.** Per-ring road height, camera minus LiDAR, out to the
  road radius. This is what phase 1's verdict made permanent, so a bias here
  is a kerb the planner will or will not see.
* **Both obstacle bands.** Occupied planner cells and AEB cells from each
  cloud, with the cells only one rig produces listed by range. A cell the
  cameras produce and the LiDAR does not is a phantom candidate; the other
  way round is a blind spot.
* **Per-camera reach**, the `Unprojection check:` line's numbers with the
  LiDAR's `Sensor reach:` beside them.

A lockstep capture has no frame skew, so a LIVE run that disagrees with this
one by a range-dependent amount is measuring CAMERA_FRAME_STAGING_S -- the
ego-motion milestone -- and that is how the constant gets its number.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from beamng_lidar_bev.config import (  # noqa: E402
    AEB_MIN_VERTICAL_EXTENT_M,
    AEB_OBSTACLE_MIN_HEIGHT_M,
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    LIDAR_RANGE_M,
    OBSTACLE_CELL_M,
    OBSTACLE_MIN_HEIGHT_M,
    PLANNER_HORIZON_M,
    WORLD_ROAD_RADIUS_M,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_camera_rig,
    derive_vehicle_geometry,
    outside_ego_body,
    world_points_to_bev,
)
from beamng_lidar_bev.models import VehicleGeometry  # noqa: E402
from beamng_lidar_bev.planner import ObstacleBand, geometric_obstacle_sets  # noqa: E402
from beamng_lidar_bev.semantics import (  # noqa: E402
    SCENE_ROAD,
    SemanticPalette,
    classify_scene_groups,
)
from beamng_lidar_bev.unprojection import (  # noqa: E402
    build_rig_rays,
    pose_from_state,
    unproject_frame,
)

OUT = Path(__file__).resolve().parent / "oracle_data"
UNKNOWN_RGB = np.asarray((1, 2, 3), dtype=np.uint8)


# --- capture ------------------------------------------------------------------------


def capture(settle_steps: int, out: Path) -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera, Lidar

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_cameras import player_vehicle

    from beamng_lidar_bev.worker import BeamNgWorker

    out.parent.mkdir(parents=True, exist_ok=True)
    bng = BeamNGpy(
        BEAMNG_HOST, BEAMNG_PORT, home=str(BEAMNG_HOME), quit_on_close=False
    )
    bng.open(launch=False)
    sensors: list = []
    paused = False
    try:
        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data
        geometry = derive_vehicle_geometry(state, vehicle.get_bbox())
        prefix = f"oracle_{int(time.monotonic() * 1000)}"

        lidars: dict[str, Lidar] = {}
        for mount in geometry.mounts.values():
            print(f"attaching LiDAR {mount.name}")
            unit = Lidar(
                f"{prefix}_lidar_{mount.name}",
                bng,
                vehicle,
                **BeamNgWorker.lidar_sensor_kwargs(mount),
            )
            sensors.append(unit)
            lidars[mount.name] = unit

        rig = derive_camera_rig(geometry)
        cameras: dict[str, Camera] = {}
        for mount in rig.values():
            print(f"attaching camera {mount.name}")
            camera = Camera(
                f"{prefix}_cam_{mount.name}",
                bng,
                vehicle,
                **BeamNgWorker.camera_sensor_kwargs(mount),
            )
            sensors.append(camera)
            cameras[mount.name] = camera

        try:
            annotations = bng.camera.get_annotations()
        except Exception:
            annotations = {}

        # Lockstep: pause, advance a fixed number of steps, read everything.
        # Every buffer then holds the same simulated instant, so the two
        # clouds differ by SENSING alone -- no frame skew, no ego motion.
        bng.control.pause()
        paused = True
        bng.control.step(settle_steps, wait=True)
        time.sleep(1.5)
        vehicle.poll_sensors("state")
        state = dict(vehicle.sensors["state"].data)

        payload: dict[str, np.ndarray] = {}
        lidar_points, lidar_colours = [], []
        for name, unit in lidars.items():
            reading = unit.stream()
            points = np.asarray(reading.get("pointCloud"), dtype=np.float32)
            points = points.reshape((-1, 3)) if points.size else np.empty((0, 3))
            colours = np.asarray(reading.get("colours"), dtype=np.uint8).reshape(-1)
            colours = colours[: len(points) * 3].reshape((-1, 3))
            if len(colours) < len(points):
                points = points[: len(colours)]
            finite = np.isfinite(points).all(axis=1)
            points, colours = points[finite], colours[finite]
            print(f"  lidar {name}: {len(points)} returns")
            payload[f"lidar_{name}_points"] = points
            payload[f"lidar_{name}_colours"] = colours
            lidar_points.append(points)
            lidar_colours.append(colours)

        rays = build_rig_rays(rig)
        pose = pose_from_state(state, 0.0)
        for name, camera in cameras.items():
            raw = camera.stream_raw()
            result = unproject_frame(
                rays[name], raw.get("depth"), raw.get("annotation"), pose,
                geometry, UNKNOWN_RGB,
            )
            if result is None:
                print(f"  camera {name}: depth buffer did not fill")
                continue
            points, colours, count = result
            print(f"  camera {name}: {count} returns")
            payload[f"camera_{name}_points"] = points
            payload[f"camera_{name}_colours"] = colours
            width, height = camera.resolution
            depth = raw.get("depth")
            if depth is not None and len(depth) == width * height * 4:
                payload[f"camera_{name}_depth"] = (
                    np.frombuffer(depth, dtype=np.float32)
                    .copy()
                    .reshape((height, width))
                )
            colour = raw.get("colour")
            if colour is not None and len(colour) == width * height * 4:
                payload[f"camera_{name}_rgb"] = (
                    np.frombuffer(colour, dtype=np.uint8)
                    .copy()
                    .reshape((height, width, 4))[..., :3]
                )

        if not lidar_points:
            print("no LiDAR returns at all -- nothing to diff against")
            return 1

        np.savez_compressed(
            out,
            state_pos=np.asarray(state["pos"], dtype=np.float64),
            state_dir=np.asarray(state["dir"], dtype=np.float64),
            state_up=np.asarray(state["up"], dtype=np.float64),
            ground_z_vehicle=geometry.ground_z_vehicle,
            body_floor_z=(
                geometry.ground_z_vehicle
                if geometry.body_floor_z is None
                else geometry.body_floor_z
            ),
            left_m=geometry.left_m,
            right_m=geometry.right_m,
            front_m=geometry.front_m,
            rear_m=geometry.rear_m,
            height_m=geometry.height_m,
            roof_z=geometry.mounts["roof"].position_vehicle[2],
            eye_z=max(m.position_vehicle[2] for m in rig.values()),
            annotation_names=np.asarray(list(annotations), dtype=object),
            annotation_rgb=np.asarray(
                [annotations[k] for k in annotations], dtype=np.int64
            ).reshape((-1, 3)) if annotations else np.empty((0, 3), np.int64),
            **payload,
        )
        print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0
    finally:
        for sensor in reversed(sensors):
            try:
                sensor.remove()
            except Exception:
                pass
        if paused:
            try:
                bng.control.resume()
            except Exception:
                pass
        bng.disconnect()


# --- analyse ------------------------------------------------------------------------


def _gather(data, prefix: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    points, colours, counts = [], [], {}
    for key in data.files:
        if key.startswith(prefix) and key.endswith("_points"):
            name = key[len(prefix): -len("_points")]
            p = data[key]
            c = data[f"{prefix}{name}_colours"]
            counts[name] = len(p)
            if len(p):
                points.append(p)
                colours.append(c)
    if not points:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), counts
    world = np.concatenate(points).astype(np.float32)
    rgb = np.concatenate(colours).astype(np.uint8)
    # The LiDAR encodes "no return" as +-3.4e38 -- finite, so the worker's
    # radius cull is what discards it there (after an overflow warning in
    # the float32 cast). Drop it before the transform here.
    sane = (np.abs(world) < 1e6).all(axis=1)
    return world[sane], rgb[sane], counts


def _occupancy(bev: np.ndarray, cell_m: float, radius_m: float) -> set[tuple[int, int]]:
    if not len(bev):
        return set()
    inside = np.einsum("ij,ij->i", bev, bev) <= radius_m**2
    cells = np.floor(bev[inside] / cell_m).astype(np.int64)
    return set(map(tuple, cells.tolist()))


def _iou(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


def _bands(
    bev: np.ndarray, heights: np.ndarray, ground_z: float, eye_z: float
) -> tuple[np.ndarray, np.ndarray]:
    planner, aeb = geometric_obstacle_sets(
        bev,
        heights,
        ground_z,
        (
            ObstacleBand(
                OBSTACLE_MIN_HEIGHT_M,
                PLANNER_HORIZON_M,
                cell_referenced=True,
                reduce_to_cells=True,
                porosity=True,
            ),
            ObstacleBand(
                AEB_OBSTACLE_MIN_HEIGHT_M,
                70.0,
                min_vertical_extent_m=AEB_MIN_VERTICAL_EXTENT_M,
                porosity=True,
            ),
        ),
        sensor_height_m=eye_z,
    )
    return planner, aeb


def _ring_floor(bev, heights, road_mask, lo, hi, half_width=4.0):
    band = (
        road_mask
        & (bev[:, 1] >= lo)
        & (bev[:, 1] < hi)
        & (np.abs(bev[:, 0]) < half_width)
    )
    if band.sum() < 20:
        return None, int(band.sum())
    return float(np.percentile(heights[band], 15.0)), int(band.sum())


def _print_cells(label: str, only: set, cell_m: float, limit: int = 12) -> None:
    if not only:
        print(f"  {label}: none")
        return
    ranges = sorted(
        (np.hypot((x + 0.5) * cell_m, (y + 0.5) * cell_m), x, y) for x, y in only
    )
    shown = ", ".join(
        f"{r:.1f} m ({(x + 0.5) * cell_m:+.1f}, {(y + 0.5) * cell_m:+.1f})"
        for r, x, y in ranges[:limit]
    )
    more = f" ... and {len(ranges) - limit} more" if len(ranges) > limit else ""
    print(f"  {label}: {len(only)} cells, nearest {shown}{more}")


def analyse(path: Path, png: bool) -> int:
    data = np.load(path, allow_pickle=True)
    state = {
        "pos": data["state_pos"],
        "dir": data["state_dir"],
        "up": data["state_up"],
        "vel": (0.0, 0.0, 0.0),
    }
    ground_z = float(data["ground_z_vehicle"])
    roof_z = float(data["roof_z"])
    eye_z = float(data["eye_z"])
    names = [str(n) for n in data["annotation_names"]]
    rgb = data["annotation_rgb"]
    palette = SemanticPalette.from_annotations(
        {name: tuple(int(v) for v in rgb[i]) for i, name in enumerate(names)}
    )

    lidar_world, lidar_rgb, lidar_counts = _gather(data, "lidar_")
    camera_world, camera_rgb, camera_counts = _gather(data, "camera_")
    print(f"{path.name}: LiDAR {len(lidar_world)} returns, cameras {len(camera_world)}")
    print("  per unit:", ", ".join(f"{k} {v}" for k, v in lidar_counts.items()))
    print("  per camera:", ", ".join(f"{k} {v}" for k, v in camera_counts.items()))
    if not len(camera_world):
        print("no camera returns -- nothing to compare")
        return 1

    geometry = VehicleGeometry(
        ground_z_vehicle=ground_z,
        left_m=float(data["left_m"]),
        right_m=float(data["right_m"]),
        front_m=float(data["front_m"]),
        rear_m=float(data["rear_m"]),
        height_m=float(data["height_m"]),
        mounts={},
        body_floor_z=float(data["body_floor_z"]),
    )
    clouds = {}
    for label, world, colours in (
        ("lidar", lidar_world, lidar_rgb),
        ("camera", camera_world, camera_rgb),
    ):
        bev, heights = world_points_to_bev(world, state)
        # The worker's own two culls, in its order: the radius, then the
        # ego's footprint -- the camera rig films its own bodywork by design.
        keep = np.einsum("ij,ij->i", bev, bev) <= LIDAR_RANGE_M**2
        keep &= outside_ego_body(bev, geometry)
        bev, heights, colours = bev[keep], heights[keep], colours[keep]
        groups = classify_scene_groups(colours, heights, ground_z, palette)
        clouds[label] = (bev, heights, groups == SCENE_ROAD)
    mirrored = clouds["camera"][0].copy()
    mirrored[:, 0] *= -1.0
    clouds["camera_mirrored"] = (mirrored, clouds["camera"][1], clouds["camera"][2])

    print()
    print("Reach (furthest return, horizontal):")
    for label in ("lidar", "camera"):
        bev = clouds[label][0]
        r = np.hypot(bev[:, 0], bev[:, 1])
        road = clouds[label][2]
        road_reach = r[road].max() if road.any() else 0.0
        print(
            f"  {label:7s} all {r.max():6.1f} m | road {road_reach:6.1f} m"
            f" | road fraction {road.mean() * 100:5.1f}%"
        )

    # --- handedness -----------------------------------------------------------------
    print()
    print("HANDEDNESS -- planner-band occupancy IoU against the LiDAR cloud:")
    bands = {}
    for label, (bev, heights, _) in clouds.items():
        sensor_height = roof_z if label == "lidar" else eye_z
        bands[label] = _bands(bev, heights, ground_z, sensor_height)
    occ = {
        label: _occupancy(bands[label][0], OBSTACLE_CELL_M, PLANNER_HORIZON_M)
        for label in bands
    }
    direct = _iou(occ["lidar"], occ["camera"])
    flipped = _iou(occ["lidar"], occ["camera_mirrored"])
    print(f"  direct   {direct:.3f}   ({len(occ['camera'])} camera cells vs "
          f"{len(occ['lidar'])} LiDAR cells)")
    print(f"  mirrored {flipped:.3f}")
    if np.isnan(direct) or np.isnan(flipped):
        print("  no obstacle cells in one of the clouds -- park beside something")
    elif direct > flipped:
        print("  -> image columns run the way camera_basis assumes (image right = "
              "vehicle right). Keep the sign.")
    elif flipped > direct:
        print("  -> THE IMAGE IS MIRRORED: negate `right` in geometry.camera_basis.")
    else:
        print("  -> tied; the scene is symmetric, move the car.")

    # --- ground band ----------------------------------------------------------------
    print()
    print("GROUND BAND -- road floor per ring (15th pct height, |right| < 4 m), "
          "camera minus LiDAR:")
    print(f"{'ring':>9s} {'lidar n':>8s} {'camera n':>9s} {'lidar h':>9s} "
          f"{'camera h':>9s} {'bias':>8s}")
    lb, lh, lr = clouds["lidar"]
    cb, ch, cr = clouds["camera"]
    for lo in np.arange(4.0, WORLD_ROAD_RADIUS_M, 4.0):
        hi = lo + 4.0
        lf, ln = _ring_floor(lb, lh, lr, lo, hi)
        cf, cn = _ring_floor(cb, ch, cr, lo, hi)
        if lf is None and cf is None:
            continue
        bias = "" if lf is None or cf is None else f"{cf - lf:+8.3f}"
        print(
            f"{lo:4.0f}-{hi:3.0f} m {ln:8d} {cn:9d} "
            f"{'-' if lf is None else f'{lf:9.3f}'} "
            f"{'-' if cf is None else f'{cf:9.3f}'} {bias:>8s}"
        )

    # --- obstacle bands ---------------------------------------------------------------
    print()
    print("PLANNER BAND -- occupied 0.4 m cells inside the 35 m horizon:")
    _print_cells("camera only (phantom candidates)", occ["camera"] - occ["lidar"],
                 OBSTACLE_CELL_M)
    _print_cells("LiDAR only (camera blind spots)", occ["lidar"] - occ["camera"],
                 OBSTACLE_CELL_M)
    print(f"  shared: {len(occ['lidar'] & occ['camera'])} cells")

    print()
    print("AEB BAND -- occupied 0.4 m cells inside 70 m:")
    aeb_l = _occupancy(bands["lidar"][1], OBSTACLE_CELL_M, 70.0)
    aeb_c = _occupancy(bands["camera"][1], OBSTACLE_CELL_M, 70.0)
    _print_cells("camera only (phantom candidates)", aeb_c - aeb_l, OBSTACLE_CELL_M)
    _print_cells("LiDAR only (camera blind spots)", aeb_l - aeb_c, OBSTACLE_CELL_M)
    print(f"  shared: {len(aeb_l & aeb_c)} cells, IoU {_iou(aeb_l, aeb_c):.3f}")

    if png:
        _write_png(path, clouds, bands)
    return 0


def _write_png(path: Path, clouds, bands) -> None:
    from PIL import Image

    from beamng_lidar_bev.raster import rasterize_points

    size = 900
    panels = []
    for label in ("lidar", "camera"):
        bev, _, road = clouds[label]
        image = rasterize_points(bev[road], bev[~road], size, size, radius_m=60.0)
        rgb = image[..., :3].astype(np.uint8).copy()
        rgb[image[..., 3] == 0] = (24, 26, 28)
        # Planner-band cells in yellow on top.
        planner = bands[label][0]
        if len(planner):
            scale = size / 120.0
            xs = np.clip((planner[:, 0] * scale + size / 2).astype(int), 0, size - 1)
            ys = np.clip((size / 2 - planner[:, 1] * scale).astype(int), 0, size - 1)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    rgb[
                        np.clip(ys + dy, 0, size - 1), np.clip(xs + dx, 0, size - 1)
                    ] = (255, 220, 40)
        panels.append(rgb)
    side = np.concatenate(panels, axis=1)
    out = path.with_suffix(".png")
    Image.fromarray(side).save(out)
    print(
        f"wrote {out} (left: LiDAR, right: cameras; yellow: planner band, 60 m radius)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--settle-steps", type=int, default=30)
    cap.add_argument("--out", type=Path, default=OUT / "scene.npz")
    ana = sub.add_parser("analyse")
    ana.add_argument("path", type=Path)
    ana.add_argument("--png", action="store_true", help="write a side-by-side BEV")
    args = parser.parse_args()
    if args.command == "capture":
        return capture(args.settle_steps, args.out)
    if args.command == "analyse":
        return analyse(args.path, args.png)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
