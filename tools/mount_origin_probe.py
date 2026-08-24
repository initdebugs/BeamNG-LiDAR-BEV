"""
WHAT IS THE ORIGIN of a vehicle-space sensor `pos`?

`derive_vehicle_geometry` builds every mount from extents measured off the
REFERENCE NODE (`state["pos"]`), and CLAUDE.md records one live check of the
Z axis only: "passing pos.z=0.20 puts Lidar.get_position() at bbox_bottom +
0.21 m", i.e. Z is referenced to the vehicle's ground plane and NOT to the
node. X and Y were never checked, and the whole rig assumes they are the node.

This probe settles it by the only measurement that cannot be argued with: put
a sensor at EXACTLY (0, 0, 0) and ask the simulator where it went. It does
that for a Camera and for a Lidar, because the two are separate engine paths
and could disagree, and it reports the answer against three candidate
origins -- the reference node, the bounding-box centre, and the bbox centre
in X/Y with the bbox bottom in Z.

    .venv39\\Scripts\\python tools\\mount_origin_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from beamng_lidar_bev.config import (  # noqa: E402
    BEAMNG_HOST,
    BEAMNG_PORT,
    CAMERA_NEAR_FAR_PLANES,
)
from beamng_lidar_bev.geometry import vec3, vehicle_axes  # noqa: E402

_SETTLE_S = 2.0
_PROBES = (
    ("origin", (0.0, 0.0, 0.0)),
    ("x_plus_one", (1.0, 0.0, 0.0)),
    ("y_plus_one", (0.0, 1.0, 0.0)),
    ("z_plus_one", (0.0, 0.0, 1.0)),
)


def _local(state, world_point) -> np.ndarray:
    """World point -> vehicle frame (+X left, +Y rearward, +Z up), node origin."""
    right, forward, up = vehicle_axes(state)
    offset = np.asarray(world_point, dtype=float) - vec3(state["pos"])
    return np.asarray(
        [offset @ -right, offset @ -forward, offset @ up], dtype=float
    )


def main() -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera, Lidar
    from capture_cameras import player_vehicle

    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=None, quit_on_close=False)
    bng.open(launch=False)
    built: list[tuple[str, object]] = []
    try:
        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)

        prefix = f"originprobe_{int(time.monotonic() * 1000)}"
        for name, pos in _PROBES:
            built.append(
                (
                    f"camera:{name}",
                    Camera(
                        f"{prefix}_cam_{name}",
                        bng,
                        vehicle,
                        requested_update_time=0.1,
                        update_priority=0.0,
                        pos=pos,
                        dir=(0.0, -1.0, 0.0),
                        up=(0.0, 0.0, 1.0),
                        resolution=(64, 48),
                        field_of_view_y=60.0,
                        near_far_planes=CAMERA_NEAR_FAR_PLANES,
                        is_using_shared_memory=True,
                        is_render_colours=True,
                        is_render_annotations=False,
                        is_render_instance=False,
                        is_render_depth=False,
                        is_visualised=False,
                        is_streaming=True,
                        is_static=False,
                        is_snapping_desired=False,
                        is_force_inside_triangle=False,
                        is_dir_world_space=False,
                    ),
                )
            )
        for name, pos in _PROBES:
            built.append(
                (
                    f"lidar:{name}",
                    Lidar(
                        f"{prefix}_lid_{name}",
                        bng,
                        vehicle,
                        requested_update_time=0.1,
                        pos=pos,
                        dir=(0.0, -1.0, 0.0),
                        up=(0.0, 0.0, 1.0),
                        vertical_resolution=16,
                        horizontal_angle=60,
                        is_rotate_mode=False,
                        is_360_mode=False,
                        is_using_shared_memory=True,
                        is_visualised=False,
                        is_streaming=True,
                        is_dir_world_space=False,
                    ),
                )
            )

        time.sleep(_SETTLE_S)

        # ONE state and ONE bbox, read together, so every candidate origin is
        # computed against the same instant -- a settling suspension moves the
        # node relative to the body and would otherwise show up as error.
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data
        bbox = vehicle.get_bbox()
        corners = np.asarray([vec3(point) for point in bbox.values()])
        local_corners = np.asarray([_local(state, c) for c in corners])
        lo = local_corners.min(axis=0)
        hi = local_corners.max(axis=0)
        centre = (lo + hi) / 2.0
        print(
            f"vehicle '{getattr(vehicle, 'model', '?')}'\n"
            f"  bbox in node frame: x [{lo[0]:+.3f}, {hi[0]:+.3f}] "
            f"y [{lo[1]:+.3f}, {hi[1]:+.3f}] z [{lo[2]:+.3f}, {hi[2]:+.3f}]\n"
            f"  bbox centre (node frame): "
            f"({centre[0]:+.3f}, {centre[1]:+.3f}, {centre[2]:+.3f})\n"
            f"  bbox bottom z: {lo[2]:+.3f}"
        )
        print(
            "\n  requested ->            measured (node frame)   "
            "measured - requested"
        )
        for label, sensor in built:
            try:
                world = sensor.get_position()
            except Exception as exc:
                print(f"  {label:22s} get_position failed: {exc}")
                continue
            measured = _local(state, world)
            requested = np.asarray(
                dict(_PROBES)[label.split(":", 1)[1]], dtype=float
            )
            delta = measured - requested
            print(
                f"  {label:22s} ({requested[0]:+.2f},{requested[1]:+.2f},"
                f"{requested[2]:+.2f}) -> "
                f"({measured[0]:+.3f},{measured[1]:+.3f},{measured[2]:+.3f})"
                f"   delta ({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})"
            )
        print(
            "\n  If delta == bbox centre in x,y and bbox bottom in z, the "
            "simulator references sensor pos to the BOUNDING BOX, not the node."
        )
        return 0
    finally:
        for _, sensor in built:
            try:
                sensor.remove()
            except Exception:
                pass
        bng.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
