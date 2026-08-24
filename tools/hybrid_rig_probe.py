"""
Do the two A-pillar cameras see the same car, and are they exposed like it?

This is the instrument that answered "one cam shows more bodywork than the
other". The rig is mirror-symmetric about the body centre ON PAPER, so the
asymmetry had to come from somewhere the arithmetic could not see -- and it
did: the simulator resolves a vehicle-space `pos` from the body centre, not
from the reference node the extents are measured off, so the pair landed
0.32 m askew (see tools/mount_origin_probe.py, which isolates that).

MEASURED on the vivace, before and after the correction:

    ego pixels   left 0.65% / right 6.64%  ->  2.87% / 2.86%
    clipped      left 34%   / right 62%    ->  0.00% / 0.03%
    mean luma    left 231   / right 240    ->  133   / 153

It measures three things at once, and prints them per camera:

* **Where the simulator actually PUT each camera.** `Camera.get_position()`
  is world space; projected back into the vehicle frame it can be compared
  with what was requested. A mount that did not land where it was asked is
  the whole answer, and the printed REQUESTED/MEASURED pair shows it.
* **How much of the frame is the ego car**, per camera, counted from the
  ANNOTATION channel rather than by eye -- the CAR classes, with the pixel
  centroid and column span, so "more bodywork on one" becomes a number.
* **Exposure**, per camera: mean luminance and the fraction of clipped
  pixels, which is the other half of the live report.

Probe-only: it attaches its own cameras with a probe prefix, reads a few
frames and removes them. It never touches the player's controls.

    .venv39\\Scripts\\python tools\\hybrid_rig_probe.py

**The simulator window must be VISIBLE** -- covered, the renderer throttles
to ~2 Hz and every camera reads as stale.
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
    HYBRID_CAMERA_AUTO_EXPOSURE,
    HYBRID_CAMERA_MANUAL_EV,
    HYBRID_CAMERA_UPDATE_TIME_S,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_hybrid_camera_rig,
    derive_vehicle_geometry,
    vec3,
    vehicle_axes,
)

_SETTLE_S = 3.0
_VEHICLE_CLASS_NAMES = ("CAR", "VEHICLE", "TRUCK", "BUS", "TRAILER", "MOTORBIKE")


def _vehicle_frame(state, world_point: np.ndarray) -> tuple[float, float, float]:
    """World point -> the vehicle's (+X left, +Y rearward, +Z up) frame."""
    right, forward, up = vehicle_axes(state)
    offset = world_point - vec3(state["pos"])
    return (
        float(offset @ -right),
        float(offset @ -forward),
        float(offset @ up),
    )


def _annotation_counts(annotation: np.ndarray, targets: set[tuple[int, int, int]]):
    """Pixels whose annotation colour is one of `targets`, and where they are."""
    if not targets:
        return 0, None, None
    packed = (
        annotation[..., 0].astype(np.uint32) << 16
        | annotation[..., 1].astype(np.uint32) << 8
        | annotation[..., 2].astype(np.uint32)
    )
    wanted = np.asarray(
        sorted((r << 16) | (g << 8) | b for r, g, b in targets), dtype=np.uint32
    )
    mask = np.isin(packed, wanted)
    count = int(mask.sum())
    if not count:
        return 0, None, None
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return (
        count,
        (float(cols.mean()) / width, float(rows.mean()) / height),
        (int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())),
    )


def _measure_origin(bng, vehicle, state) -> tuple[float, float, float]:
    """A throwaway camera at (0, 0, 0) reports where the simulator put it."""
    from beamngpy.sensors import Camera

    probe = Camera(
        f"rigprobe_origin_{int(time.monotonic() * 1000)}",
        bng,
        vehicle,
        requested_update_time=10.0,
        update_priority=0.0,
        pos=(0.0, 0.0, 0.0),
        dir=(0.0, -1.0, 0.0),
        up=(0.0, 0.0, 1.0),
        resolution=(16, 16),
        field_of_view_y=60.0,
        near_far_planes=CAMERA_NEAR_FAR_PLANES,
        is_using_shared_memory=False,
        is_render_colours=True,
        is_render_annotations=False,
        is_render_instance=False,
        is_render_depth=False,
        is_visualised=False,
        is_streaming=False,
        is_static=False,
        is_snapping_desired=False,
        is_force_inside_triangle=False,
        is_dir_world_space=False,
    )
    try:
        return _vehicle_frame(
            state, np.asarray(probe.get_position(), dtype=float)
        )
    finally:
        probe.remove()


def _apply_exposure(bng, names: list[str]) -> str:
    """The app's own exposure policy, so the probe measures what it shows."""
    apply = (
        "Research.Camera.clearManualEV(id)"
        if HYBRID_CAMERA_AUTO_EXPOSURE
        else f"Research.Camera.setManualEV(id, {HYBRID_CAMERA_MANUAL_EV!r})"
    )
    wanted = ",".join(f'["{name}"]=true' for name in names)
    chunk = "\n".join(
        (
            f"local wanted = {{{wanted}}}",
            "local ok, res = pcall(function()",
            "  local done = 0",
            "  for _, id in ipairs(Research.Camera.getActiveCameraSensors()) do",
            "    if wanted[tostring(Research.Camera.getSensorName(id))] then",
            f"      {apply}",
            "      done = done + 1",
            "    end",
            "  end",
            "  return done",
            "end)",
            "if ok then return tostring(res) else return 'error: '..tostring(res) end",
        )
    )
    return bng.control.queue_lua_command(chunk, response=True)


def main() -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera
    from capture_cameras import player_vehicle

    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=None, quit_on_close=False)
    bng.open(launch=False)
    cameras: list[tuple[str, Camera]] = []
    try:
        focused = bng.control.queue_lua_command(
            "return tostring(Engine.isProgramFocused())", response=True
        )
        print(f"window focused: {focused}")

        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data
        bbox = vehicle.get_bbox()
        # Exactly what the worker does: measure where the simulator reads a
        # vehicle-space `pos` from BEFORE deriving anything off the extents.
        origin = _measure_origin(bng, vehicle, state)
        print(
            f"\nsensor origin, from the reference node: "
            f"({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f}) m"
        )
        geometry = derive_vehicle_geometry(state, bbox, sensor_origin=origin)

        centre_x = (geometry.left_m - geometry.right_m) / 2.0
        print(
            f"\nvehicle '{getattr(vehicle, 'model', '?')}' | "
            f"left {geometry.left_m:.3f} right {geometry.right_m:.3f} "
            f"front {geometry.front_m:.3f} rear {geometry.rear_m:.3f} "
            f"height {geometry.height_m:.3f}"
        )
        print(
            f"body centre x = {centre_x:+.3f} m (reference node at x = 0); "
            f"body floor z = {geometry.body_floor_z:+.3f}"
        )

        # What the annotation palette calls a vehicle, so ego pixels can be
        # counted rather than eyeballed.
        annotations = bng.camera.get_annotations()
        targets = {
            tuple(int(c) % 256 for c in rgb)
            for name, rgb in annotations.items()
            if name.upper() in _VEHICLE_CLASS_NAMES
        }
        print(f"vehicle annotation colours: {sorted(targets)}")

        rig = derive_hybrid_camera_rig(geometry)
        prefix = f"rigprobe_{int(time.monotonic() * 1000)}"
        for mount in rig.values():
            x, y, z = mount.position_vehicle
            print(
                f"\nREQUESTED {mount.name}: pos=({x:+.3f}, {y:+.3f}, {z:+.3f}) "
                f"| offset from body centre {x - centre_x:+.3f} m "
                f"| dir=({mount.direction_vehicle[0]:+.4f}, "
                f"{mount.direction_vehicle[1]:+.4f}, "
                f"{mount.direction_vehicle[2]:+.4f}) "
                f"| hfov {mount.horizontal_fov_deg:.1f} "
                f"vfov {mount.vertical_fov_deg:.2f} {mount.resolution}"
            )
            camera = Camera(
                f"{prefix}_{mount.name}",
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
                # The probe's whole point: the annotation channel is what
                # turns "more bodywork" into a pixel count.
                is_render_annotations=True,
                is_render_instance=False,
                is_render_depth=True,
                is_visualised=False,
                is_streaming=True,
                is_static=False,
                is_snapping_desired=False,
                is_force_inside_triangle=False,
                is_dir_world_space=False,
                integer_depth=False,
                postprocess_depth=False,
            )
            cameras.append((mount.name, camera))

        applied = _apply_exposure(
            bng, [f"{prefix}_{name}" for name in rig]
        )
        setting = (
            "auto"
            if HYBRID_CAMERA_AUTO_EXPOSURE
            else f"fixed {HYBRID_CAMERA_MANUAL_EV}"
        )
        print(f"\nexposure: {setting} applied to {applied} cameras")

        print(f"\nsettling {_SETTLE_S:.0f} s ...")
        time.sleep(_SETTLE_S)
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data

        rows = []
        for name, camera in cameras:
            world_pos = np.asarray(camera.get_position(), dtype=float)
            world_dir = np.asarray(camera.get_direction(), dtype=float)
            local = _vehicle_frame(state, world_pos)
            right, forward, up = vehicle_axes(state)
            local_dir = (
                float(world_dir @ -right),
                float(world_dir @ -forward),
                float(world_dir @ up),
            )
            raw = camera.stream_raw()
            width, height = camera.resolution
            colour = np.frombuffer(
                memoryview(raw["colour"]), dtype=np.uint8
            ).reshape((height, width, 4))[..., :3].copy()
            annotation = np.frombuffer(
                memoryview(raw["annotation"]), dtype=np.uint8
            ).reshape((height, width, 4))[..., :3].copy()

            count, centroid, span = _annotation_counts(annotation, targets)
            luminance = (
                0.2126 * colour[..., 0].astype(np.float32)
                + 0.7152 * colour[..., 1].astype(np.float32)
                + 0.0722 * colour[..., 2].astype(np.float32)
            )
            rows.append((name, colour))
            print(
                f"\nMEASURED {name}"
                f"\n  actual pos in vehicle frame: "
                f"({local[0]:+.3f}, {local[1]:+.3f}, {local[2]:+.3f}) "
                f"| offset from body centre {local[0] - centre_x:+.3f} m"
                f"\n  actual dir in vehicle frame: "
                f"({local_dir[0]:+.4f}, {local_dir[1]:+.4f}, {local_dir[2]:+.4f})"
                f"\n  ego pixels: {count} ({100.0 * count / (width * height):.2f}%)"
                f" centroid {centroid} col/row span {span}"
                f"\n  luminance: mean {luminance.mean():.1f} "
                f"p50 {np.percentile(luminance, 50):.1f} "
                f"p99 {np.percentile(luminance, 99):.1f} "
                f"| clipped (>=250) {100.0 * (luminance >= 250).mean():.2f}% "
                f"| black (<=5) {100.0 * (luminance <= 5).mean():.2f}%"
            )

        if len(rows) == 2:
            try:
                from PIL import Image

                pair = np.concatenate([rows[0][1], rows[1][1]], axis=1)
                out = Path(__file__).resolve().parent / "hybrid_rig_probe.png"
                Image.fromarray(pair).save(out)
                print(f"\nwrote {out}")
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"could not write the side-by-side PNG: {exc}")
        return 0
    finally:
        for _, camera in cameras:
            try:
                camera.remove()
            except Exception:
                pass
        bng.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
