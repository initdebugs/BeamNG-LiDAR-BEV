"""
Grab real camera frames from a running session and measure them.

"Pixelated" and "washed out" are two different complaints with two different
causes, so this captures the same pose under several variants and writes a PNG
plus a statistics line for each. Per this project's rule about pixel questions:
look at the image, do not reason about it.

    py -3.12 tools/capture_cameras.py --camera all      # the whole rig, once
    py -3.12 tools/capture_cameras.py --camera front_main   # variant sweep

Needs a running BeamNG.tech with a map and player vehicle loaded, and the
app's own connection released (STOP in the app, or close it) -- the bridge
takes one client.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from beamng_lidar_bev.config import (  # noqa: E402
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    CAMERA_NEAR_FAR_PLANES,
    HYBRID_CAMERA_UPDATE_TIME_S,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_hybrid_camera_rig,
    derive_vehicle_geometry,
)
from beamng_lidar_bev.models import CameraMount  # noqa: E402

OUT = Path(__file__).resolve().parent / "camera_probe"

# (label, resolution, {setting: value}). The settings are BeamNG's own graphics
# keys; each variant is applied, captured, then handed back.
VARIANTS: list[tuple[str, tuple[int, int], dict[str, str]]] = [
    ("640x480_asis", (640, 480), {}),
    ("1280x960_asis", (1280, 960), {}),
    ("1280x960_nodof", (1280, 960), {"PostFXDOFGeneralEnabled": "false"}),
    ("1280x960_nobloom", (1280, 960), {"PostFXBloomGeneralEnabled": "false"}),
    ("1280x960_norays", (1280, 960), {"PostFXLightRaysEnabled": "false"}),
    (
        "1280x960_clean",
        (1280, 960),
        {
            "PostFXDOFGeneralEnabled": "false",
            "PostFXBloomGeneralEnabled": "false",
            "PostFXLightRaysEnabled": "false",
        },
    ),
]
POSTFX_KEYS = (
    "PostFXDOFGeneralEnabled",
    "PostFXBloomGeneralEnabled",
    "PostFXLightRaysEnabled",
)


def statistics(rgba: np.ndarray) -> str:
    """Washed out has a signature: a lifted black point and a squeezed range."""
    luma = rgba[..., :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.percentile(luma, (1.0, 99.0))
    blown = float((luma >= 250.0).mean()) * 100.0
    return (
        f"mean {luma.mean():6.1f}  std {luma.std():5.1f}  "
        f"p1 {lo:5.1f}  p99 {hi:5.1f}  span {hi - lo:5.1f}  "
        f"blown {blown:4.1f}%"
    )


def make_camera(camera_cls, bng, vehicle, name: str, mount: CameraMount):
    """Exactly the worker's construction, so a probe cannot flatter the app."""
    return camera_cls(
        name,
        bng,
        vehicle,
        requested_update_time=HYBRID_CAMERA_UPDATE_TIME_S,
        update_priority=0.0,
        pos=mount.position_vehicle,
        dir=mount.direction_vehicle,
        up=(0.0, 0.0, 1.0),
        resolution=mount.resolution,
        field_of_view_y=mount.vertical_fov_deg,
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
    )


def grab(camera, timeout_s: float = 5.0) -> np.ndarray | None:
    """Wait for a frame that is not the zero-filled initial buffer."""
    width, height = camera.resolution
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = camera.stream_raw().get("colour")
        if raw is not None and len(raw) == width * height * 4:
            pixels = np.frombuffer(raw, dtype=np.uint8).copy()
            if pixels.any():
                return pixels.reshape((height, width, 4))
        time.sleep(0.05)
    return None


def capture(camera_cls, bng, vehicle, label: str, mount: CameraMount) -> None:
    camera = make_camera(camera_cls, bng, vehicle, f"probe_{label}", mount)
    try:
        rgba = grab(camera)
        if rgba is None:
            print(f"{label:22s}  NO FRAME (buffer stayed zero-filled)")
            return
        path = OUT / f"{label}.png"
        Image.fromarray(rgba[..., :3]).save(path)
        print(f"{label:22s}  {statistics(rgba)}  -> {path.name}")
    finally:
        camera.remove()


def player_vehicle(bng):
    """
    The vehicle the USER is driving -- not whichever one the dict yields first.

    `get_current()` returns every actor including traffic, so
    `next(iter(...))` picks an arbitrary car: this tool spent a whole
    investigation photographing `clone2`, a `simple_traffic` van parked in a
    car park, while the player sat somewhere else entirely. The worker has
    always used `get_player_vehicle_id`; so does this now, or a probe and the
    app are not looking at the same thing.
    """
    try:
        player = bng.vehicles.get_player_vehicle_id()
        vid = str(player.get("vid", "")).strip()
    except Exception:
        vid = ""
    vehicles = bng.vehicles.get_current()
    if vid and vid in vehicles:
        return vehicles[vid]
    if vid:
        print(f"WARNING: player '{vid}' is not in get_current(); "
              f"available: {sorted(vehicles)}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="front_main")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 960))
    args = parser.parse_args()

    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera

    OUT.mkdir(parents=True, exist_ok=True)
    bng = BeamNGpy(
        BEAMNG_HOST, BEAMNG_PORT, home=str(BEAMNG_HOME), quit_on_close=False
    )
    bng.open(launch=False)
    try:
        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data
        geometry = derive_vehicle_geometry(state, vehicle.get_bbox())
        print(
            f"vehicle: {vehicle.vid} (model {getattr(vehicle, 'model', '?')})"
            f"  bbox {geometry.width_m:.2f} x {geometry.length_m:.2f} x "
            f"{geometry.height_m:.2f}"
        )

        if args.camera == "all":
            rig = derive_hybrid_camera_rig(geometry, tuple(args.resolution))
            for name, mount in rig.items():
                capture(Camera, bng, vehicle, f"rig_{name}", mount)
            return 0

        for label, resolution, settings in VARIANTS:
            for key, value in settings.items():
                bng.settings.change(key, value)
            if settings:
                bng.settings.apply_graphics()
                time.sleep(1.0)
            mount = derive_hybrid_camera_rig(geometry, resolution)[args.camera]
            capture(Camera, bng, vehicle, f"{args.camera}_{label}", mount)

        for key in POSTFX_KEYS:
            bng.settings.change(key, "true")
        bng.settings.apply_graphics()
    finally:
        bng.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
