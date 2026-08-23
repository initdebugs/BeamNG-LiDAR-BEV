"""
Phase 1: can computed stereo resolve a kerb at 15-30 m?

This is the measurement with veto power over the stereo rung (roadmap phase 1):
if a 0.10-0.15 m kerb face does not separate from the road surface in a stereo
cloud at planner ranges, the stereo rung pivots to a hybrid before the port
rather than after it.

    py -3.12 tools/kerb_experiment.py capture --baseline 0.6
    py -3.12 tools/kerb_experiment.py analyse tools/kerb_data/pair_b0.60.npz

Two design choices worth knowing:

* **The pair needs no rectification.** Both cameras are pinhole, share their
  intrinsics exactly, and are offset along the vehicle's lateral axis with
  identical direction vectors -- so their epipolar lines are already image rows.
  A real rig would have to calibrate and rectify; this one is rectified by
  construction, which is the whole reason a simulator is worth measuring in.

* **The engine's depth channel is the ORACLE, not the answer.** It is captured
  from the left camera in the same lockstep frame purely so the stereo result
  can be diffed against ground truth. That is the ladder's structural advantage
  (spec section 1), and this experiment is the first thing to use it.

One capture yields every range at once: a kerb running away from the car spans
10-40 m in a single frame, so the verdict table comes from one pair rather than
one drive per range.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from beamng_lidar_bev.config import (  # noqa: E402
    BEAMNG_HOME,
    BEAMNG_HOST,
    BEAMNG_PORT,
    CAMERA_NEAR_FAR_PLANES,
    CAMERA_UPDATE_TIME_S,
)
from beamng_lidar_bev.geometry import derive_vehicle_geometry  # noqa: E402

OUT = Path(__file__).resolve().parent / "kerb_data"
STEREO_HFOV_DEG = 60.0
STEREO_RESOLUTION = (1280, 960)


def focal_px(hfov_deg: float, width: int) -> float:
    """Pinhole focal length in pixels from a horizontal aperture."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def vertical_fov_deg(hfov_deg: float, resolution: tuple[int, int]) -> float:
    """The Camera constructor takes a VERTICAL aperture (field_of_view_y)."""
    width, height = resolution
    half = math.tan(math.radians(hfov_deg) / 2.0) * (height / width)
    return math.degrees(2.0 * math.atan(half))


def build_pair(camera_cls, bng, vehicle, geometry, baseline_m: float):
    """
    A rectified-by-construction stereo pair on the vehicle's lateral axis.

    +X is LEFT in the vehicle frame and forward is (0, -1, 0) -- the intuitive
    (0, 1, 0) films the rear seats. Both cameras straddle the BODY centre, not
    the reference node, which is not the same point (measured 0.16 m apart on
    the vivace).
    """
    centre_x = (geometry.left_m - geometry.right_m) / 2.0
    z = 0.90 * geometry.height_m
    y = -0.30 * geometry.front_m
    vfov = vertical_fov_deg(STEREO_HFOV_DEG, STEREO_RESOLUTION)

    def one(name: str, x: float, with_depth: bool):
        return camera_cls(
            name,
            bng,
            vehicle,
            requested_update_time=CAMERA_UPDATE_TIME_S,
            update_priority=0.0,
            pos=(x, y, z),
            dir=(0.0, -1.0, 0.0),
            up=(0.0, 0.0, 1.0),
            resolution=STEREO_RESOLUTION,
            field_of_view_y=vfov,
            near_far_planes=CAMERA_NEAR_FAR_PLANES,
            is_using_shared_memory=True,
            is_render_colours=True,
            is_render_annotations=False,
            is_render_instance=False,
            is_render_depth=with_depth,
            # Both are traps the spec names: the default quantises depth to
            # 0-255 silently, and the postprocess is a 256-iteration Python
            # loop per frame.
            integer_depth=False,
            postprocess_depth=False,
            is_visualised=False,
            is_streaming=True,
            is_static=False,
            is_snapping_desired=False,
            is_force_inside_triangle=False,
            is_dir_world_space=False,
        )

    left = one("kerb_left", centre_x + baseline_m / 2.0, True)
    right = one("kerb_right", centre_x - baseline_m / 2.0, False)
    return left, right


def read_colour(camera) -> np.ndarray | None:
    width, height = camera.resolution
    raw = camera.stream_raw().get("colour")
    if raw is None or len(raw) != width * height * 4:
        return None
    pixels = np.frombuffer(raw, dtype=np.uint8).copy()
    if not pixels.any():
        return None
    return pixels.reshape((height, width, 4))[..., :3]


def read_depth(camera) -> np.ndarray | None:
    """Planar Z in metres. Decodes as raw float32 x far plane (spec section 3)."""
    width, height = camera.resolution
    raw = camera.stream_raw().get("depth")
    if raw is None or len(raw) != width * height * 4:
        return None
    values = np.frombuffer(raw, dtype=np.float32).copy().reshape((height, width))
    if not np.isfinite(values).any() or not values.any():
        return None
    return values * CAMERA_NEAR_FAR_PLANES[1]


def capture(baseline_m: float, settle_steps: int,
            resolution: tuple[int, int] | None = None) -> int:
    global STEREO_RESOLUTION
    if resolution is not None:
        STEREO_RESOLUTION = resolution
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_cameras import player_vehicle

    OUT.mkdir(parents=True, exist_ok=True)
    bng = BeamNGpy(
        BEAMNG_HOST, BEAMNG_PORT, home=str(BEAMNG_HOME), quit_on_close=False
    )
    bng.open(launch=False)
    left = right = None
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
        left, right = build_pair(Camera, bng, vehicle, geometry, baseline_m)
        fx = focal_px(STEREO_HFOV_DEG, STEREO_RESOLUTION[0])
        print(
            f"pair built: baseline {baseline_m:.2f} m, "
            f"{STEREO_RESOLUTION[0]}x{STEREO_RESOLUTION[1]}, "
            f"hfov {STEREO_HFOV_DEG:.0f} deg, fx {fx:.1f} px"
        )

        # Lockstep: pause, advance a fixed number of steps, then read. Both
        # buffers then hold the same simulated instant, which a free-running
        # read cannot promise -- the cameras stage frames independently.
        bng.control.pause()
        paused = True
        bng.control.step(settle_steps, wait=True)
        time.sleep(1.0)

        left_rgb = read_colour(left)
        right_rgb = read_colour(right)
        depth = read_depth(left)
        if left_rgb is None or right_rgb is None:
            print("colour buffers did not fill")
            return 1
        if depth is None:
            print("depth buffer did not fill -- is_render_depth/integer_depth?")
            return 1

        finite = np.isfinite(depth) & (depth > 0.0)
        print(
            f"depth: {finite.mean() * 100:.1f}% valid, "
            f"range {depth[finite].min():.2f}..{depth[finite].max():.1f} m, "
            f"median {np.median(depth[finite]):.2f} m"
        )

        path = OUT / f"pair_b{baseline_m:.2f}_w{STEREO_RESOLUTION[0]}.npz"
        np.savez_compressed(
            path,
            left=left_rgb,
            right=right_rgb,
            depth=depth.astype(np.float32),
            baseline_m=baseline_m,
            focal_px=fx,
            hfov_deg=STEREO_HFOV_DEG,
            resolution=np.asarray(STEREO_RESOLUTION),
            camera_z_m=0.90 * geometry.height_m,
            ground_z_vehicle=geometry.ground_z_vehicle,
        )
        print(f"saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return 0
    finally:
        for camera in (left, right):
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


def heights_and_lateral(
    depth: np.ndarray, focal: float, camera_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel height above the vehicle's ground plane, and lateral offset.

    The camera is level and looks along the vehicle's forward axis, so this is
    the pinhole relation with no rotation in it: a pixel `v` rows below the
    principal point sees a point `(v - cy) * Z / f` below the lens.

    Note what that means for THIS experiment. A depth error enters the height
    only through `(v - cy)/f`, which for a point on the road equals
    `camera_z / Z` -- so at 30 m a 0.34 m depth error is a 1.5 cm height error.
    The road is seen at a grazing angle, and that geometry is what makes a 12 cm
    kerb a plausible target for a sensor whose ranging is metres-coarse.
    """
    height, width = depth.shape
    cx, cy = width / 2.0, height / 2.0
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    above = camera_z - (rows - cy) * depth / focal
    lateral = -(cols - cx) * depth / focal
    return above, lateral


def disparity_search_px(focal: float, baseline: float, nearest_m: float) -> int:
    """
    How far SGBM has to search, rounded up to the multiple of 16 it requires.

    THIS MUST SCALE WITH THE BASELINE and getting it wrong does not look like an
    error: at 1.6 m the nearest road is ~355 px of disparity against a fixed
    160 px window, so the near field simply fails to match and the wide pair
    scored WORSE than the narrow one -- which would have been read as "a wide
    baseline does not help" when it was only ever a search that stopped short.
    """
    needed = int(math.ceil(focal * baseline / nearest_m))
    return max(64, int(math.ceil(needed / 16.0)) * 16)


def stereo_depth(left: np.ndarray, right: np.ndarray, focal: float,
                 baseline: float, max_disparity: int | None = None):
    """SGBM disparity converted to planar Z. Invalid pixels come back as NaN."""
    import cv2

    if max_disparity is None:
        max_disparity = disparity_search_px(focal, baseline, 4.0)
    grey_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    grey_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # SGBM returns 1/16-pixel fixed point; the sub-pixel part is the whole
    # reason a kerb is resolvable at all, so do not round it away.
    disparity = matcher.compute(grey_l, grey_r).astype(np.float32) / 16.0
    valid = disparity > 0.5
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = focal * baseline / disparity[valid]
    return depth, disparity, valid


def road_plane(height: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """
    Fit the road as `h = a + b*Z` over the corridor straight ahead.

    Absolute height above the vehicle's ground plane is NOT flat over range --
    measured on this capture the road climbs 0.077 m at 3.8 m to 0.281 m at
    18.9 m, a steady ~1.5% that is the street's own grade plus whatever the
    suspension is doing. Thresholding on absolute height therefore finds no road
    at all. The LiDAR stack has exactly this problem and answers it the same
    way, with a per-range ground estimate (`planner.ground_rise`); the kerb is a
    step ABOVE the local surface, never a fixed height above the car.

    Fitted on a low percentile per range bin so that cars, poles and the far
    junction cannot pull the surface up.
    """
    corridor = (
        np.isfinite(depth) & (depth > 4.0) & (depth < 45.0) & (np.abs(height) < 3.0)
    )
    lateral_ok = corridor
    ranges, floors = [], []
    for lo in np.arange(4.0, 45.0, 2.0):
        band = lateral_ok & (depth >= lo) & (depth < lo + 2.0)
        if band.sum() < 300:
            continue
        ranges.append(lo + 1.0)
        floors.append(float(np.percentile(height[band], 15.0)))
    if len(ranges) < 3:
        return 0.0, 0.0
    slope, intercept = np.polyfit(np.asarray(ranges), np.asarray(floors), 1)
    return float(intercept), float(slope)


def _profile(values: np.ndarray, lateral: np.ndarray, band: np.ndarray,
             bin_m: float, lo: float, hi: float):
    """
    The GROUND height per lateral bin, for the pixels inside one range band.

    A low percentile rather than a median: a range band is a shell, so it holds
    the wall and the lamp post at that range as well as the surface. The lowest
    returns in a lateral bin are the ground -- which on the pavement side means
    the pavement, and that is the kerb top this experiment is looking for.
    """
    edges = np.arange(lo, hi + bin_m, bin_m)
    centres = edges[:-1] + bin_m / 2.0
    out = np.full(centres.shape, np.nan, dtype=np.float32)
    index = np.digitize(lateral[band], edges) - 1
    picked = values[band]
    finite = np.isfinite(picked)
    for i in range(len(centres)):
        sel = picked[(index == i) & finite]
        if sel.size >= 8:
            out[i] = float(np.percentile(sel, 20.0))
    return centres, out


def _find_kerb(centres: np.ndarray, profile: np.ndarray,
               kerb_min: float, kerb_max: float):
    """
    The carriageway, and the pavement immediately over the kerb beside it.

    Walking OUTWARD from the road edge is what makes this a kerb measurement
    rather than a "something is raised out there" measurement. Selecting every
    raised bin across a 24 m window instead picked up pavements, planters and
    building bases far outboard, and reported a 0.216 m "kerb" -- taller than
    any kerb, which is how the mistake announced itself.
    """
    centre_bin = int(np.argmin(np.abs(centres)))
    flat = np.isfinite(profile) & (np.abs(profile) < 0.04)
    if not flat[centre_bin]:
        near = np.flatnonzero(flat)
        if near.size == 0:
            return None
        centre_bin = int(near[np.argmin(np.abs(near - centre_bin))])

    road = np.zeros_like(flat)
    for step in (1, -1):
        i = centre_bin
        while 0 <= i < len(profile) and flat[i]:
            road[i] = True
            i += step
    if road.sum() < 4:
        return None

    best = None
    for direction in (1, -1):
        edge = int(np.flatnonzero(road)[-1 if direction > 0 else 0])
        i = edge + direction
        # Skip at most a couple of bins of kerb FACE, which read part-height.
        for _ in range(3):
            if not (0 <= i < len(profile)):
                break
            value = profile[i]
            if np.isfinite(value) and kerb_min < value < kerb_max:
                pavement = np.zeros_like(road)
                j, taken = i, 0
                while 0 <= j < len(profile) and taken < 4:
                    v = profile[j]
                    if not (np.isfinite(v) and kerb_min < v < kerb_max):
                        break
                    pavement[j] = True
                    taken += 1
                    j += direction
                if taken >= 2 and (best is None or taken > best[1].sum()):
                    best = (road, pavement)
                break
            i += direction
    return best


def analyse(path: Path, kerb_min: float, kerb_max: float,
            use_oracle: bool = False) -> int:
    data = np.load(path)
    left, right = data["left"], data["right"]
    truth = data["depth"].astype(np.float32)
    focal = float(data["focal_px"])
    baseline = float(data["baseline_m"])
    camera_z = float(data["camera_z_m"])

    depth, disparity, valid = stereo_depth(left, right, focal, baseline)
    print(f"{path.name}: baseline {baseline:.2f} m, fx {focal:.1f} px, "
          f"camera {camera_z:.2f} m above ground")
    print(f"stereo produced disparity for {valid.mean() * 100:.1f}% of pixels")
    if use_oracle:
        # CONTROL: run the engine's own depth through the identical pipeline.
        # The answer is known, so anything other than a clean ~0.12 m step at a
        # huge separation means the MEASUREMENT is broken rather than stereo.
        print("*** CONTROL RUN: oracle depth substituted for stereo ***")
        depth = truth.copy()

    true_h, true_x = heights_and_lateral(truth, focal, camera_z)
    stereo_h, stereo_x = heights_and_lateral(depth, focal, camera_z)

    # Everything below is measured against the LOCAL road surface, never
    # against the car's own ground plane -- see road_plane().
    intercept, slope = road_plane(true_h, truth)
    print(f"road plane: {intercept:+.3f} m at the bumper, {slope * 100:+.2f}% grade")
    true_h = true_h - (intercept + slope * truth)
    stereo_h = stereo_h - (intercept + slope * np.where(np.isfinite(depth), depth, 0.0))

    print()
    print("Depth agreement against the engine oracle, road surface only:")
    print(f"{'range':>7s} {'n':>7s} {'bias':>8s} {'sigma':>8s}")
    for target in (10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        band = (
            np.isfinite(depth)
            & (np.abs(truth - target) < 0.5)
            & (np.abs(true_h) < 0.06)
        )
        if band.sum() < 200:
            print(f"{target:7.0f} {band.sum():7d}   too few road pixels")
            continue
        error = depth[band] - truth[band]
        print(f"{target:7.0f} {band.sum():7d} {np.mean(error):8.3f} "
              f"{np.std(error):8.3f}")

    print()
    print("THE VERDICT -- does a kerb face separate from the road surface?")
    print(f"{'range':>7s} {'true step':>10s} {'stereo step':>12s} "
          f"{'road sigma':>11s} {'separation':>11s}")
    verdicts = []
    for target in (15.0, 20.0, 25.0, 30.0):
        band = np.abs(truth - target) < 0.6
        if band.sum() < 500:
            print(f"{target:7.0f}   too few pixels in the band")
            continue
        centres, true_prof = _profile(true_h, true_x, band, 0.25, -12.0, 12.0)
        _, stereo_prof = _profile(stereo_h, stereo_x, band, 0.25, -12.0, 12.0)

        found = _find_kerb(centres, true_prof, kerb_min, kerb_max)
        if found is None:
            print(f"{target:7.0f}   no kerb found in the oracle at this range")
            continue
        road, pavement = found
        usable = pavement & np.isfinite(stereo_prof)
        if usable.sum() < 2:
            # NO DATA IS A FAILURE, NOT AN EXCLUSION. Skipping these once made
            # the whole run report PASS on the strength of 15 m alone while
            # stereo had produced nothing at all at 20, 25 and 30 -- the
            # question is whether the kerb is resolved, and an empty answer
            # does not resolve it.
            verdicts.append((target, 0.0))
            print(f"{target:7.0f}          -            -           -    "
                  f"NO DATA at the kerb")
            continue

        road_ref = float(np.nanmedian(stereo_prof[road]))
        sigma = float(np.nanstd(stereo_prof[road]))
        true_step = float(np.nanmedian(true_prof[pavement])) - float(
            np.nanmedian(true_prof[road])
        )
        stereo_step = float(np.nanmedian(stereo_prof[usable])) - road_ref
        separation = stereo_step / sigma if sigma > 0 else float("inf")
        verdicts.append((target, separation))
        print(f"{target:7.0f} {true_step:10.3f} {stereo_step:12.3f} "
              f"{sigma:11.3f} {separation:10.1f}x")

    print()
    if verdicts:
        worst = min(s for _, s in verdicts)
        print(f"worst separation across 15-30 m: {worst:.1f} sigma")
        print("PASS -- stereo resolves the kerb" if worst >= 3.0
              else "FAIL -- the kerb does not clear the noise; see roadmap phase 1")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--baseline", type=float, default=0.6)
    cap.add_argument("--settle-steps", type=int, default=30)
    cap.add_argument("--width", type=int, default=1280,
                     help="image width; disparity precision scales with it")
    ana = sub.add_parser("analyse")
    ana.add_argument("path", type=Path)
    ana.add_argument("--kerb-min", type=float, default=0.06)
    ana.add_argument("--kerb-max", type=float, default=0.30)
    ana.add_argument("--use-oracle", action="store_true",
                     help="control: feed the engine depth through the same "
                          "pipeline, where the answer is known")
    args = parser.parse_args()
    if args.command == "capture":
        return capture(args.baseline, args.settle_steps,
                       (args.width, int(args.width * 3 / 4)))
    if args.command == "analyse":
        return analyse(args.path, args.kerb_min, args.kerb_max, args.use_oracle)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
