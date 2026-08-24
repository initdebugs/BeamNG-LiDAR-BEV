"""
What exposure is a tech Camera actually rendering at?

Reported live: the rig's tiles look "brighter and overexposed" against the
stock BeamNG view. Measured on the A-pillar pair, 1280x960 on gridmap: mean
luminance 232 and 241 of 255 with 36% and 64% of pixels at or above 250. That
is not a preference, it is clipping -- the sky and the concrete arrive as flat
white and the detail in them is gone before the buffer is read.

**The cause is the opposite of the obvious one: the sensor does NOT auto-expose.**
A Camera created exactly the way beamngpy creates one reports
`useManualEV=true, manualEV=0.001` -- a FIXED LINEAR exposure multiplier -- while
the game's own view runs its eye-adaptation loop, which had settled around
2^-12.4 = 0.00019 on the same map. So the sensor renders roughly 5x brighter
than what the player sees, and being fixed it cannot adapt to a scene at all.

BeamNG ships the exposure controls but the Lua wrappers are COMMENTED OUT in
`lua/ge/extensions/tech/sensors.lua:438-441`:

    -- local function getCameraUseManualEV(sensorId) ... end
    -- local function setCameraManualEV(sensorId, ev) ... end

and beamngpy's Camera has no exposure argument at all, so neither route
reaches it. The C++ bindings under them (`Research.Camera.setManualEV` and
friends) are live, and `queue_lua_command` reaches those.

This probe finds a camera's engine sensor id by name and sweeps the value, so
the constants in config are measurements. Measured on a bright map:

    manualEV  0.0001 -> mean 73    0% clipped
    manualEV  0.001  -> mean 90    1.4% clipped   (the shipped default)
    manualEV  0.01   -> mean 154   23% clipped
    manualEV  1.0    -> mean 238   58% clipped
    clearManualEV    -> mean 76    0% clipped     (auto: useManualEV false)

Note the scale: anything at or above ~50 saturates the frame outright, so a
sweep in STOPS (which is what `PostEffectLocalExposureObject.manualEV` is,
and what the name suggests) measures nothing but white.

    .venv39\Scripts\python tools\camera_exposure_probe.py

**The simulator window must be VISIBLE**, or the renderer throttles and every
frame is stale.
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
    HYBRID_CAMERA_UPDATE_TIME_S,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_hybrid_camera_rig,
    derive_vehicle_geometry,
)

# The value is a linear multiplier, so the interesting range is small and
# logarithmic. `None` means clearManualEV -- the engine's own auto-exposure.
_EV_SWEEP = (
    None,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.05,
    1.0,
    None,
)
_FRAME_WAIT_S = 1.5
_SETTLE_S = 5.0


def _luminance(colour: np.ndarray) -> np.ndarray:
    return (
        0.2126 * colour[..., 0].astype(np.float32)
        + 0.7152 * colour[..., 1].astype(np.float32)
        + 0.0722 * colour[..., 2].astype(np.float32)
    )


def main() -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera
    from capture_cameras import player_vehicle

    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=None, quit_on_close=False)
    bng.open(launch=False)
    camera: Camera | None = None

    def lua(chunk: str) -> str:
        return bng.control.queue_lua_command(chunk, response=True)

    try:
        print(f"window focused: {lua('return tostring(Engine.isProgramFocused())')}")
        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        geometry = derive_vehicle_geometry(
            vehicle.sensors["state"].data, vehicle.get_bbox()
        )
        mount = next(iter(derive_hybrid_camera_rig(geometry).values()))

        name = f"evprobe_{int(time.monotonic() * 1000)}"
        camera = Camera(
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
        width, height = camera.resolution

        # The engine addresses a camera by an integer id; BeamNGpy addresses it
        # by name and never exposes the id, so it has to be looked up.
        sensor_id = lua(
            "local wanted = '%s'\n"
            "for _, id in ipairs(Research.Camera.getActiveCameraSensors()) do\n"
            "  if tostring(Research.Camera.getSensorName(id)) == wanted then\n"
            "    return tostring(id)\n"
            "  end\n"
            "end\n"
            "return 'nil'" % name
        )
        print(f"engine sensor id for '{name}': {sensor_id}")
        if sensor_id in ("nil", "", None):
            print("Could not resolve the engine sensor id; cannot sweep EV.")
            return 1
        # Eye adaptation is a RATE, so a freshly created render view has not
        # converged yet: measure the settled auto state, not the first frame.
        print(f"settling {_SETTLE_S:.0f} s for eye adaptation ...")
        time.sleep(_SETTLE_S)

        print(
            f"\n  {'EV':>6}  {'mean':>6}  {'p50':>6}  {'p99':>6}  "
            f"{'>=250':>7}  {'<=5':>6}  useManualEV"
        )
        for ev in _EV_SWEEP:
            if ev is None:
                lua(f"Research.Camera.clearManualEV({sensor_id}) return 'ok'")
                label = "auto"
            else:
                lua(f"Research.Camera.setManualEV({sensor_id}, {ev}) return 'ok'")
                label = f"{ev:.1f}"
            time.sleep(_FRAME_WAIT_S)
            raw = camera.stream_raw()
            colour = np.frombuffer(
                memoryview(raw["colour"]), dtype=np.uint8
            ).reshape((height, width, 4))[..., :3].copy()
            lum = _luminance(colour)
            use = lua(
                f"return tostring(Research.Camera.getUseManualEV({sensor_id}))"
            )
            print(
                f"  {label:>6}  {lum.mean():6.1f}  "
                f"{np.percentile(lum, 50):6.1f}  {np.percentile(lum, 99):6.1f}  "
                f"{100.0 * (lum >= 250).mean():6.2f}%  "
                f"{100.0 * (lum <= 5).mean():5.2f}%  {use}"
            )
        lua(f"Research.Camera.clearManualEV({sensor_id}) return 'ok'")
        return 0
    finally:
        if camera is not None:
            try:
                camera.remove()
            except Exception:
                pass
        bng.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
