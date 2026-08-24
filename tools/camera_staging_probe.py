"""
Phase 2, milestone 4: measure CAMERA_FRAME_STAGING_S.

The simulator stages camera frames "a frame or two" behind with no timestamp,
so the worker can only measure a frame's age from when its buffer last
CHANGED; the fixed staging part has to be measured once and written into
config. Two measurements, no driving and no touching the player's car:

* **Stepped**: pause, swing a probe camera's own direction 90 degrees, then
  step ONE frame at a time counting steps until the depth buffer follows.
  The count is the staging in simulator frames.
* **Real time**: resume, swing again, and time how long the buffer takes to
  follow. Includes the render cadence itself, so it is an upper bound on
  what a worker tick can see.

    .venv39\\Scripts\\python tools\\camera_staging_probe.py

**The simulator window must be VISIBLE.** Fully covered, the renderer
throttles to ~2 Hz (measured 2026-08-23: a 320x240 camera AND a LiDAR unit
both under 2 Hz) and every latency here reads as ~700 ms of throttle rather
than staging. The probe checks `Engine.isProgramFocused` and refuses.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from beamng_lidar_bev.config import (  # noqa: E402
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    CAMERA_FRAME_STAGING_S,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_camera_rig,
    derive_vehicle_geometry,
)
from beamng_lidar_bev.worker import BeamNgWorker  # noqa: E402


def main() -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera

    from capture_cameras import player_vehicle

    bng = BeamNGpy(
        BEAMNG_HOST, BEAMNG_PORT, home=str(BEAMNG_HOME), quit_on_close=False
    )
    bng.open(launch=False)
    camera = None
    paused = False
    try:
        focused = bng.control.queue_lua_command(
            "return tostring(Engine.isProgramFocused and "
            "Engine.isProgramFocused() or false)",
            response=True,
        )
        if str(focused).strip().lower() != "true":
            print(
                "REFUSING TO MEASURE: the simulator window is not in the "
                "foreground. Covered, the renderer throttles to ~2 Hz and "
                "every latency below would read as throttle, not staging. "
                "Click the BeamNG window (or put it on a second monitor) "
                "and re-run."
            )
            return 1

        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        geometry = derive_vehicle_geometry(
            vehicle.sensors["state"].data, vehicle.get_bbox()
        )
        mount = derive_camera_rig(geometry)["front_main"]
        camera = Camera(
            f"staging_{int(time.monotonic() * 1000)}",
            bng,
            vehicle,
            **BeamNgWorker.camera_sensor_kwargs(mount),
        )
        time.sleep(1.5)

        def digest() -> bytes:
            raw = camera.stream_raw().get("depth")
            return bytes(np.frombuffer(raw, np.float32)[::1021])

        directions = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ]

        print("== stepped (pause -> swing the probe camera -> step one frame "
              "at a time) ==")
        bng.control.pause()
        paused = True
        bng.control.step(30, wait=True)
        time.sleep(0.5)
        stepped: list[int] = []
        for trial, direction in enumerate(directions):
            bng.control.step(5, wait=True)
            time.sleep(0.3)
            before = digest()
            camera.set_direction(direction)
            steps = None
            for step in range(1, 12):
                bng.control.step(1, wait=True)
                time.sleep(0.05)
                if digest() != before:
                    steps = step
                    break
            print(f"  trial {trial}: buffer followed after "
                  f"{steps if steps is not None else '>11'} frame(s)")
            if steps is not None:
                stepped.append(steps)
        bng.control.resume()
        paused = False
        time.sleep(0.5)

        print("== real time (running sim; includes the render cadence) ==")
        latencies: list[float] = []
        for trial, direction in enumerate(directions):
            before = digest()
            started = time.perf_counter()
            camera.set_direction(direction)
            while time.perf_counter() - started < 3.0:
                if digest() != before:
                    break
                time.sleep(0.002)
            latency = (time.perf_counter() - started) * 1000.0
            latencies.append(latency)
            print(f"  trial {trial}: {latency:.0f} ms")
            time.sleep(0.4)

        if stepped:
            frames = float(np.median(stepped))
            print()
            print(f"staging: {frames:.0f} simulator frame(s) behind.")
            print(
                "At the camera's ~16-18 Hz delivery that is roughly "
                f"{frames * 60:.0f} ms of fixed lag ON TOP of the digest age "
                "the worker already measures. Set CAMERA_FRAME_STAGING_S "
                f"toward {frames * 0.060:.3f} (currently "
                f"{CAMERA_FRAME_STAGING_S}) and re-run the oracle diff before "
                "trusting it -- the constant is applied to every camera on "
                "every tick."
            )
        return 0
    finally:
        if camera is not None:
            try:
                camera.remove()
            except Exception:
                pass
        if paused:
            try:
                bng.control.resume()
            except Exception:
                pass
        bng.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
