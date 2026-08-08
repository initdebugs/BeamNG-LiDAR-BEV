"""
Drive the real WORLD pipeline with a synthetic street and grab the framebuffer.

Nothing here is mocked below the snapshot: a PerceptionSnapshot goes into the
real WorldSceneAssembler, the real WorldFrame goes into the real WorldView, and
the real WorldScene.qml renders it on the real GPU. It is the only way to see
the thing short of launching BeamNG.

The scene deliberately contains every case that was reported:
  * a long road surface, to see whether it still reads as big flat pixels
  * a low wall standing in front of a tall building, on the left
  * a tree: trunk, canopy floating clear of the ground, grass underneath
  * a bridge deck crossing overhead
  * kerbs down both sides, and buildings at several ranges
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from beamng_lidar_bev.models import (  # noqa: E402
    ARMED,
    BRAKING,
    AebState,
    PerceptionSnapshot,
    VehicleGeometry,
)
from beamng_lidar_bev.semantics import (  # noqa: E402
    SCENE_BOUNDARY,
    SCENE_ROAD,
    SCENE_UNKNOWN,
    SCENE_VEHICLE,
)
from beamng_lidar_bev.world_scene import WorldSceneAssembler  # noqa: E402
from beamng_lidar_bev.world_view import WorldView  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "braking"
OUT = Path(__file__).with_name(f"world_preview_{MODE}.png")
RNG = np.random.default_rng(3)
GEOMETRY = VehicleGeometry(
    ground_z_vehicle=-0.55,
    left_m=0.95,
    right_m=0.95,
    front_m=2.3,
    rear_m=2.1,
    height_m=1.45,
    mounts={},
)


def slab(x0, x1, y0, y1, z0, z1, step=0.18):
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    # Surfaces only -- a solid block of points is not what a LiDAR returns.
    faces = []
    for x in (xs[0], xs[-1]):
        gy, gz = np.meshgrid(ys, zs)
        faces.append(np.column_stack((np.full(gy.size, x), gy.ravel(), gz.ravel())))
    for y in (ys[0], ys[-1]):
        gx, gz = np.meshgrid(xs, zs)
        faces.append(np.column_stack((gx.ravel(), np.full(gx.size, y), gz.ravel())))
    gx, gy = np.meshgrid(xs, ys)
    faces.append(
        np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, zs[-1])))
    )
    return np.concatenate(faces)


def build_scene():
    road_points, boundary_points = [], []

    # Road surface, sampled as rings the way the sensors really land them.
    radii = np.concatenate([np.arange(2.0, 14.0, 0.10), np.arange(14.0, 44.0, 0.30)])
    azimuth = np.linspace(-np.pi, np.pi, 420)
    rr, aa = np.meshgrid(radii, azimuth)
    x = rr.ravel() * np.cos(aa.ravel())
    y = rr.ravel() * np.sin(aa.ravel())
    on_road = np.abs(x) < 6.5
    road_points.append(
        np.column_stack(
            (
                x[on_road],
                y[on_road],
                -0.55 + 0.05 * np.cos(x[on_road] / 6.5 * np.pi / 2)
                + RNG.normal(0, 0.008, on_road.sum()),
            )
        )
    )
    # Grass either side -- boundary returns, which is what smeared the tree.
    grass = ~on_road & (np.abs(x) < 26.0)
    boundary_points.append(
        np.column_stack(
            (x[grass], y[grass], np.full(grass.sum(), -0.53)
             + RNG.normal(0, 0.01, grass.sum()))
        )
    )

    # Kerbs both sides.
    for side in (-1, 1):
        boundary_points.append(
            slab(side * 6.5 - 0.15, side * 6.5 + 0.15, -20, 42, -0.55, -0.40, 0.07)
        )

    # LEFT: a low wall at 14 m standing in front of a tall building at 22 m.
    # This is the reported "one blob" case.
    boundary_points.append(slab(-13.0, -7.2, 12.0, 12.5, -0.55, 0.85, 0.14))
    boundary_points.append(slab(-19.0, -8.0, 20.0, 30.0, -0.55, 11.0, 0.30))

    # RIGHT: buildings at three ranges, to show depth separating them.
    boundary_points.append(slab(8.0, 17.0, 6.0, 15.0, -0.55, 7.0, 0.26))
    boundary_points.append(slab(8.0, 18.0, 17.0, 27.0, -0.55, 9.5, 0.30))
    boundary_points.append(slab(9.0, 20.0, 29.0, 40.0, -0.55, 12.0, 0.34))

    # A tree on the right verge: trunk, canopy clear of the ground.
    trunk = slab(6.9, 7.25, 8.0, 8.35, -0.55, 2.6, 0.08)
    canopy = slab(5.6, 8.6, 6.7, 9.7, 2.9, 6.1, 0.16)
    boundary_points.extend((trunk, canopy))

    # A bridge deck crossing overhead at 34 m, 5 m up.
    boundary_points.append(slab(-9.0, 9.0, 33.5, 36.5, 4.7, 5.5, 0.22))

    # Two cars ahead in lane, and one oncoming -- SCENE_VEHICLE, with no
    # ground-truth actor offered at all, which is the free-roam case.
    traffic = [
        slab(-1.9, 0.0, 26.0, 30.5, -0.5, 1.45, 0.10),
        slab(-1.7, 0.2, 52.0, 56.5, -0.5, 1.5, 0.12),
        slab(2.6, 4.4, 74.0, 78.5, -0.5, 1.55, 0.14),
    ]

    # Far structure, to show the 150 m reach.
    for near, far, height in ((60, 74, 14.0), (86, 104, 18.0), (118, 140, 22.0)):
        boundary_points.append(
            slab(9.0, 26.0, near, far, -0.55, height, 0.55)
        )
        boundary_points.append(
            slab(-27.0, -10.0, near + 8, far + 6, -0.55, height * 0.8, 0.55)
        )

    unknown = np.column_stack(
        (
            RNG.uniform(-24, 24, 900),
            RNG.uniform(-10, 38, 900),
            RNG.uniform(-0.5, 1.4, 900),
        )
    )

    road = np.concatenate(road_points)
    boundary = np.concatenate(boundary_points)
    vehicles = np.concatenate(traffic)
    points = np.concatenate([road, boundary, vehicles, unknown]).astype(np.float32)
    groups = np.concatenate(
        [
            np.full(len(road), SCENE_ROAD, np.uint8),
            np.full(len(boundary), SCENE_BOUNDARY, np.uint8),
            np.full(len(vehicles), SCENE_VEHICLE, np.uint8),
            np.full(len(unknown), SCENE_UNKNOWN, np.uint8),
        ]
    )
    return points, groups


def aeb_state(status, threat_m, rearward=False):
    return AebState(
        status=status,
        brake=1.0 if status == BRAKING else 0.0,
        curvature=0.0,
        rearward=rearward,
        threat_m=threat_m,
        horizon_m=48.0,
        standoff_m=0.6,
        brake_now_m=22.0,
        corridor_half_width_m=1.35,
        required_decel_mps2=6.2 if status == BRAKING else 3.4,
        time_to_collision_s=1.6,
        reason="preview",
    )


def main() -> int:
    app = QApplication(sys.argv)
    view = WorldView()
    view.resize(1280, 800)
    if not view.is_ready:
        print("QML FAILED:", view.failure_message)
        return 2
    view.show()

    points, groups = build_scene()
    print(f"cloud {len(points):,} points")
    assembler = WorldSceneAssembler()
    frame = None
    for tick in range(6):
        frame = assembler.update(
            PerceptionSnapshot(
                points_world=points,
                semantic_groups=groups,
                ego_pos_world=(0.0, 0.0, 0.0),
                ego_dir_world=(0.0, 1.0, 0.0),
                ego_up_world=(0.0, 0.0, 1.0),
                timestamp=tick * 0.04,
                speed_mps=18.0,
                forward_speed_mps=18.0,
                aeb=(
                    aeb_state(BRAKING, 26.0)
                    if MODE == "braking"
                    else aeb_state(ARMED, None)
                ),
                vehicle_geometry=GEOMETRY,
            )
        )
    print(
        f"road    {len(frame.road_vertices):,} verts / "
        f"{len(frame.road_indices) // 3:,} tris"
    )
    print(
        f"boundary {len(frame.boundary_vertices) // 24:,} slabs, "
        f"{len(frame.boundary_vertices):,} verts"
    )
    view.set_frame(frame)

    def grab() -> None:
        view.grab().save(str(OUT))
        print(f"saved {OUT}")
        view.shutdown()
        app.quit()

    QTimer.singleShot(2500, grab)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
