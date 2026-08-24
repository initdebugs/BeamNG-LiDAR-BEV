"""
Phase 2: per-camera provenance for the WORLD ghosting milestone.

The vision WORLD view drew one pole as several near-copies (reported live
2026-08-24). The candidate mechanisms are all registration error -- the
unmeasured staging constant, the freshness digest's detection jitter, torn
depth reads, and cross-camera disagreement -- and every one of them displaces
a COPY of the object that the 90 m scenery memory then keeps. Before any
fusion radius or mitigation is chosen, the roadmap requires the disagreement
to be MEASURED, per camera, in two cases:

* **stationary** (default): the car at rest near a narrow object. Pose error
  is zero by construction, so whatever spread remains is rig geometry and
  depth quantisation -- the floor every other number sits on.
* **moving** (`--seconds 20 --countdown 5`, then drive past or at the
  object): per-FRAME offsets from the target's REST position, decomposed
  along the velocity. The reference matters: the first moving capture
  measured offsets from its own pooled centre, and with one contributing
  camera that subtracts away exactly the common bias being measured. The
  target position comes from the same run's parked first frame (measured
  cross-camera accuracy +/-1.5 cm), so offsets from it are absolute.

The best moving target is a WALL FACE approached head-on (`--ahead` picks
the structure dead ahead of the parked car): the first capture auto-picked a
thin pole whose only moving returns were the planter kerb beside it sliding
through the cylinder -- the report now warns when a camera's frame spread
exceeds the physical jitter bound (v x one loop tick), because an extended
object's visible window slides with the car and reads as registration error
when it is not. A head-on face is immune: perspective moves the window
ACROSS the face, never through it.

    .venv39\\Scripts\\python tools\\ghosting_probe.py
    ... ghosting_probe.py --ahead --seconds 15 --countdown 5 --tag wall
    ... ghosting_probe.py --analyse tools/oracle_data/ghosting_wall.npz

The target is auto-picked (nearest narrow, tall, isolated structure within
25 m) or given as `--target X Y` in world coordinates. The simulator window
must be visible -- covered, the renderer throttles to ~2 Hz and every age
measurement here reads as throttle.
"""
from __future__ import annotations

import argparse
import math
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
    WORLD_COLUMN_SIZE_M,
)
from beamng_lidar_bev.geometry import (  # noqa: E402
    derive_camera_rig,
    derive_vehicle_geometry,
)
from beamng_lidar_bev.unprojection import (  # noqa: E402
    build_rig_rays,
    pose_from_state,
    sample_depth,
    surface_mask,
    unproject_camera,
)
from beamng_lidar_bev.worker import BeamNgWorker  # noqa: E402

# The worker's own freshness digest stride, so the ages measured here are the
# ages the app would measure.
_DIGEST_STRIDE = 61
_YAW_RATE_TAU_S = 0.08
# Per-tick RMS scatter above which the tracked returns are an EXTENDED
# structure, whose visible window slides with the car -- the along numbers
# then measure perspective, not registration.
_COMPACT_RMS_M = 0.35
_DATA_DIR = Path(__file__).resolve().parent / "oracle_data"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _lattice_depth(camera, rays) -> np.ndarray | None:
    """Depth at the lattice, scoped so no live-buffer view outlives the call.

    Holding the stream_raw() memoryview in a caller-level local is what made
    camera.remove() print `Cannot close shared memory ... exported pointers
    exist` once per buffer at teardown.
    """
    raw = camera.stream_raw()
    return sample_depth(raw.get("depth"), rays)


def _pick_target(
    points: np.ndarray, ego_xy: np.ndarray, ego_ground_z: float
) -> np.ndarray | None:
    """Nearest narrow, tall, isolated structure -- a pole, a bollard, a sign.

    Grid the above-ground points into 0.25 m cells, keep cells with >= 0.7 m
    of vertical extent, and accept the nearest one whose 0.6-2.5 m annulus is
    nearly empty of other above-ground returns: a wall or hedge cell always
    has occupied neighbours along its own run, a pole does not. The annulus
    reaches 2.5 m because the first capture's parked rays put only FOUR
    returns on a planter kerb 1 m from the picked pole -- sparse rings
    under-sample exactly the low neighbours that ruin the moving analysis.
    """
    heights = points[:, 2] - ego_ground_z
    above = points[(heights > 0.3) & (heights < 4.0)]
    if len(above) < 20:
        return None
    cell = 0.25
    cx = np.floor(above[:, 0] / cell).astype(np.int64)
    cy = np.floor(above[:, 1] / cell).astype(np.int64)
    key = cx * np.int64(2_000_003) + cy
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, key_sorted[1:] != key_sorted[:-1]])
    z_sorted = above[order, 2]
    extents = np.maximum.reduceat(z_sorted, starts) - np.minimum.reduceat(
        z_sorted, starts
    )
    centres = np.stack(
        (
            (cx[order][starts] + 0.5) * cell,
            (cy[order][starts] + 0.5) * cell,
        ),
        axis=1,
    )
    tall = extents >= 0.7
    if not tall.any():
        return None
    candidates = centres[tall]
    distances = np.linalg.norm(candidates - ego_xy[None, :], axis=1)
    usable = (distances > 2.0) & (distances < 25.0)
    if not usable.any():
        return None
    xy = above[:, :2]
    for index in np.argsort(distances):
        if not usable[index]:
            continue
        offsets = np.linalg.norm(xy - candidates[index][None, :], axis=1)
        annulus = int(np.count_nonzero((offsets > 0.6) & (offsets < 2.5)))
        core = int(np.count_nonzero(offsets <= 0.4))
        if core >= 10 and annulus <= max(6, core // 10):
            column = above[offsets <= 0.4]
            base_z = float(column[:, 2].min())
            print(
                f"auto-picked target at ({candidates[index][0]:.2f}, "
                f"{candidates[index][1]:.2f}), {distances[index]:.1f} m away, "
                f"{extents[tall][index]:.2f} m tall, {core} returns, "
                f"{annulus} neighbours in the 0.6-2.5 m isolation ring"
            )
            return np.array(
                (candidates[index][0], candidates[index][1], base_z)
            )
    return None


def _capture(args: argparse.Namespace) -> int:
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Camera
    from capture_cameras import player_vehicle

    radius = args.radius
    if radius is None:
        # A drive-by needs the room to catch a badly displaced copy: a
        # compact object with more error than the cylinder margin escapes
        # the capture entirely and reads as "no returns", which is the
        # selection bias the first moving capture could not rule out.
        radius = 3.5 if args.countdown > 0 else 1.5

    bng = BeamNGpy(
        BEAMNG_HOST, BEAMNG_PORT, home=str(BEAMNG_HOME), quit_on_close=False
    )
    bng.open(launch=False)
    cameras: dict[str, object] = {}
    try:
        focused = bng.control.queue_lua_command(
            "return tostring(Engine.isProgramFocused and "
            "Engine.isProgramFocused() or false)",
            response=True,
        )
        if str(focused).strip().lower() != "true":
            print(
                "REFUSING TO MEASURE: the simulator window is not in the "
                "foreground -- covered, the renderer throttles to ~2 Hz and "
                "every frame age below would measure the throttle. Click the "
                "BeamNG window and re-run."
            )
            return 1

        vehicle = player_vehicle(bng)
        if vehicle is None:
            print("No player vehicle. Load a map and spawn one first.")
            return 1
        vehicle.connect(bng)
        vehicle.poll_sensors("state")
        state = vehicle.sensors["state"].data
        geometry = derive_vehicle_geometry(state, vehicle.get_bbox())
        rig = derive_camera_rig(geometry)
        rays = build_rig_rays(rig)
        names = sorted(rig)
        stamp = int(time.monotonic() * 1000)
        for name in names:
            cameras[name] = Camera(
                f"ghost_{name}_{stamp}",
                bng,
                vehicle,
                **BeamNgWorker.camera_sensor_kwargs(rig[name]),
            )
        time.sleep(2.0)

        # One pooled frame from the PARKED car: the target's rest position is
        # the absolute reference every moving offset is measured against.
        pose_now = pose_from_state(state)
        pooled = []
        for name in names:
            depth = _lattice_depth(cameras[name], rays[name])
            if depth is None:
                continue
            keep = surface_mask(depth)
            pooled.append(
                unproject_camera(rays[name], depth, pose_now, geometry, keep)
            )
        if not pooled:
            print("No camera delivered a depth frame; nothing to measure.")
            return 1
        frame = np.concatenate(pooled)
        ego_xy = np.asarray(state["pos"][:2], dtype=np.float64)
        ego_ground_z = float(state["pos"][2]) + geometry.ground_z_vehicle
        if args.target is not None:
            offsets = np.linalg.norm(
                frame[:, :2] - np.asarray(args.target)[None, :], axis=1
            )
            near = frame[offsets < radius]
            base_z = float(near[:, 2].min()) if len(near) else ego_ground_z
            target = np.array((args.target[0], args.target[1], base_z))
        elif args.ahead:
            # The structure dead ahead -- a wall face to drive AT. The
            # isolation picker rejects walls by design (their annulus is
            # full of themselves), and a wall approached head-on is the one
            # target whose along-travel offset is immune to the visible-
            # window slide: perspective moves the window ACROSS the face,
            # never through it.
            forward = np.asarray(state["dir"][:2], dtype=np.float64)
            forward = forward / max(1e-9, np.linalg.norm(forward))
            lateral = np.array((-forward[1], forward[0]))
            rel = frame[:, :2] - ego_xy[None, :]
            ahead = rel @ forward
            beside = rel @ lateral
            corridor = (
                (frame[:, 2] > ego_ground_z + 0.5)
                & (np.abs(beside) < 1.5)
                & (ahead > 3.0)
            )
            if not corridor.any():
                print(
                    "Nothing taller than 0.5 m stands in the forward "
                    "corridor -- park squarely facing the wall and re-run."
                )
                return 1
            nearest = float(ahead[corridor].min())
            slab = corridor & (ahead < nearest + 0.5)
            centre = np.median(frame[slab][:, :2], axis=0)
            base_z = float(frame[slab][:, 2].min())
            target = np.array((centre[0], centre[1], base_z))
            print(
                f"targeting the face dead ahead: ({target[0]:.2f}, "
                f"{target[1]:.2f}), {nearest:.1f} m out, "
                f"{int(slab.sum())} returns in the first half-metre"
            )
        else:
            picked = _pick_target(frame, ego_xy, ego_ground_z)
            if picked is None:
                print(
                    "No isolated narrow structure found within 25 m -- park "
                    "near a pole, bollard or sign, or pass --target X Y (a "
                    "wall face approached head-on works well)."
                )
                return 1
            target = picked

        for remaining in range(int(args.countdown), 0, -1):
            print(f"capturing in {remaining} s ... (start driving now)")
            time.sleep(1.0)

        points_parts: list[np.ndarray] = []
        prov_cam: list[np.ndarray] = []
        prov_tick: list[np.ndarray] = []
        tick_time: list[float] = []
        ego_pos: list[np.ndarray] = []
        ego_vel: list[np.ndarray] = []
        age_rows: list[np.ndarray] = []
        digests: dict[str, bytes] = {}
        seen: dict[str, float] = {}
        checked: dict[str, float] = {}
        heading_prev: float | None = None
        heading_at = 0.0
        yaw_rate = 0.0

        started = time.perf_counter()
        tick = 0
        while time.perf_counter() - started < float(args.seconds):
            vehicle.poll_sensors("state")
            state = vehicle.sensors["state"].data
            now = time.perf_counter()
            forward = np.asarray(state["dir"], dtype=np.float64)
            heading = math.atan2(forward[1], forward[0])
            if heading_prev is not None and now > heading_at:
                dt = now - heading_at
                rate = _wrap(heading - heading_prev) / dt
                yaw_rate += min(1.0, dt / _YAW_RATE_TAU_S) * (rate - yaw_rate)
            heading_prev, heading_at = heading, now

            ages = np.full(len(names), -1.0)
            for cam_index, name in enumerate(names):
                depth = _lattice_depth(cameras[name], rays[name])
                if depth is None:
                    continue
                digest = depth[::_DIGEST_STRIDE].tobytes()
                previous_check = checked.get(name)
                if digests.get(name) != digest:
                    digests[name] = digest
                    # Midpoint of the last two looks -- the worker's own
                    # seen-time centring, mirrored so captures measure the
                    # residual the app actually carries.
                    seen[name] = (
                        (now + previous_check) / 2.0
                        if previous_check is not None
                        else now
                    )
                checked[name] = now
                age = max(0.0, now - seen.get(name, now))
                ages[cam_index] = age
                keep = surface_mask(depth)
                pose = pose_from_state(state, age, yaw_rate)
                points = unproject_camera(
                    rays[name], depth, pose, geometry, keep
                )
                offsets = np.linalg.norm(
                    points[:, :2] - target[None, :2], axis=1
                )
                near = points[offsets < radius]
                if len(near):
                    points_parts.append(near)
                    prov_cam.append(np.full(len(near), cam_index, np.int8))
                    prov_tick.append(np.full(len(near), tick, np.int32))
            tick_time.append(now - started)
            ego_pos.append(np.asarray(state["pos"], dtype=np.float64))
            ego_vel.append(
                np.asarray(state.get("vel", (0.0, 0.0, 0.0)), dtype=np.float64)
            )
            age_rows.append(ages)
            tick += 1

        if not points_parts:
            print("No camera return ever landed inside the target cylinder.")
            return 1
        _DATA_DIR.mkdir(exist_ok=True)
        out = _DATA_DIR / f"ghosting_{args.tag}.npz"
        np.savez_compressed(
            out,
            points=np.concatenate(points_parts),
            prov_cam=np.concatenate(prov_cam),
            prov_tick=np.concatenate(prov_tick),
            camera_names=np.asarray(names),
            tick_time=np.asarray(tick_time),
            ego_pos=np.asarray(ego_pos),
            ego_vel=np.asarray(ego_vel),
            ages=np.asarray(age_rows),
            target=target,
            radius=float(radius),
        )
        print(f"saved {out} ({tick} ticks)")
        _report(np.load(out))
        return 0
    finally:
        for camera in cameras.values():
            try:
                camera.remove()  # type: ignore[attr-defined]
            except Exception:
                pass
        bng.disconnect()


def _frame_ids(age_column: np.ndarray) -> np.ndarray:
    """Group ticks into camera FRAMES: a new frame wherever the measured age
    resets. Registration error is per frame -- every stale re-read of one
    frame lands the same points by design -- so frames, not ticks, are the
    unit the jitter statistics count."""
    ids = np.full(len(age_column), -1, dtype=np.int32)
    current = 0
    previous = -1.0
    for index, age in enumerate(age_column):
        if age < 0.0:
            previous = -1.0
            continue
        if previous >= 0.0 and age < previous - 1e-9:
            current += 1
        ids[index] = current
        previous = age
    return ids


def _ghost_columns(
    points: np.ndarray, prov_tick: np.ndarray, contaminated: bool
) -> None:
    """The store's-eye view: how many distinct voxel columns did the whole
    capture paint, against the median single tick? 1.0x means every look
    agreed; the ratio is the smear the 90 m memory would keep."""

    def _columns(subset: np.ndarray) -> int:
        return len(
            np.unique(
                np.floor(subset[:, 0] / WORLD_COLUMN_SIZE_M).astype(np.int64)
                * np.int64(2_000_003)
                + np.floor(subset[:, 1] / WORLD_COLUMN_SIZE_M).astype(np.int64)
            )
        )

    per_tick_columns = [
        _columns(points[prov_tick == tick]) for tick in np.unique(prov_tick)
    ]
    single = float(np.median(per_tick_columns))
    print(
        f"ghost columns: {_columns(points)} distinct "
        f"{WORLD_COLUMN_SIZE_M:.3f} m columns over the capture against "
        f"{single:.0f} per single tick -- "
        f"{_columns(points) / max(1.0, single):.1f}x"
        + (
            " (contaminated by the extended-structure slide)"
            if contaminated
            else ""
        )
    )


def _report(data: "np.lib.npyio.NpzFile") -> None:
    points = data["points"]
    prov_cam = data["prov_cam"]
    prov_tick = data["prov_tick"]
    names = [str(name) for name in data["camera_names"]]
    ego_pos = data["ego_pos"]
    ego_vel = data["ego_vel"]
    ages = data["ages"]
    target = data["target"]

    # The cylinder contains the GROUND around the object as well, and a
    # camera whose frame misses the object still lands road returns in it --
    # measured on the first capture, front_main's whole contribution was a
    # 3 cm slice of tarmac 1.26 m from the pole, which read as a huge cross
    # offset until the structure was separated from the floor it stands on.
    structure = points[:, 2] > float(target[2]) + 0.3
    if structure.any():
        ground_n = int(len(points) - structure.sum())
        points = points[structure]
        prov_cam = prov_cam[structure]
        prov_tick = prov_tick[structure]
    else:
        ground_n = 0

    speeds = np.linalg.norm(ego_vel[:, :2], axis=1)
    mean_speed = float(speeds.mean())
    moving = mean_speed > 1.0
    print()
    print(
        f"== ghosting report: {len(points)} structure returns "
        f"({ground_n} ground returns set aside) in a "
        f"{float(data['radius']):.1f} m cylinder at "
        f"({target[0]:.2f}, {target[1]:.2f}), "
        f"{'MOVING' if moving else 'STATIONARY'} "
        f"(mean speed {mean_speed:.1f} m/s) =="
    )
    print(
        "offsets are metres FROM THE TARGET'S REST POSITION (the parked "
        "first frame), along x cross the tick's own velocity"
        if moving
        else "offsets are metres FROM THE TARGET'S REST POSITION, along x "
        "cross the mean line of sight"
    )

    # Compactness guard: an extended structure's visible window slides with
    # the car, so its per-frame medians measure perspective, not
    # registration -- the first moving capture's returns were a planter kerb
    # doing exactly that.
    tick_rms: list[float] = []
    for tick in np.unique(prov_tick):
        mine = points[prov_tick == tick][:, :2]
        centre = np.median(mine, axis=0)
        tick_rms.append(
            float(np.sqrt(np.mean(np.sum((mine - centre) ** 2, axis=1))))
        )
    extended = bool(tick_rms) and float(np.median(tick_rms)) > _COMPACT_RMS_M

    # A LINE-LIKE structure (fence, kerb, wall) slides its visible window
    # along its own TANGENT as the car moves -- both early moving captures
    # did exactly this and swamped the velocity-frame numbers. Registration
    # error survives untouched on the structure's NORMAL: no displacement
    # along a straight structure moves it through itself. So when the
    # tracked returns are anisotropic the decomposition axis comes from
    # their SHAPE, and a frame carries timing information only while the
    # car actually crosses that normal.
    deltas = points[:, :2] - np.median(points[:, :2], axis=0)[None, :]
    if len(deltas) > 3:
        cov = (deltas.T @ deltas) / len(deltas)
        evals, evecs = np.linalg.eigh(cov)
        axis_ratio = math.sqrt(float(evals[1]) / max(1e-12, float(evals[0])))
        long_axis_m = math.sqrt(float(evals[1]))
    else:
        axis_ratio, long_axis_m = 1.0, 0.0
        evecs = np.eye(2)
    line_like = moving and axis_ratio > 2.0 and long_axis_m > 0.5

    if extended and not line_like:
        print(
            f"WARNING: tracked returns are EXTENDED (median per-tick RMS "
            f"{float(np.median(tick_rms)):.2f} m > {_COMPACT_RMS_M}) -- the "
            "along numbers below likely measure a visibility window sliding "
            "with the car, not registration. Re-run against a wall face "
            "approached head-on, or a thick post."
        )

    sight = target[:2] - ego_pos[:, :2].mean(axis=0)
    sight = sight / max(1e-9, np.linalg.norm(sight))
    loop_s = (
        float(np.median(np.diff(data["tick_time"])))
        if len(data["tick_time"]) > 1
        else 0.04
    )

    if line_like:
        tangent = evecs[:, 1]
        normal = np.array((-float(tangent[1]), float(tangent[0])))
        print(
            f"tracked structure is LINE-LIKE (axis ratio {axis_ratio:.1f}, "
            f"tangent ({float(tangent[0]):+.2f}, {float(tangent[1]):+.2f})): "
            "offsets are decomposed on its NORMAL, which the visible-window "
            "slide cannot touch. A REGISTRATION error scales with the speed "
            "the car crosses that normal at; a static offset (the tracked "
            "line not being the structure the target sits on) does not; the "
            "regression below separates the two."
        )
        print(
            f"{'camera':>14} {'returns':>7} {'frames':>6} {'used':>4} "
            f"{'normal med':>10} {'med spread':>10} {'period ms':>9}"
        )
        crossings: list[float] = []
        meds: list[float] = []
        for cam_index, name in enumerate(names):
            mask = prov_cam == cam_index
            if not mask.any():
                continue
            frame_of_tick = _frame_ids(ages[:, cam_index])
            frames = frame_of_tick[prov_tick[mask]]
            offsets_n = (points[mask][:, :2] - target[None, :2]) @ normal
            cam_meds: list[float] = []
            used = 0
            for frame in np.unique(frames):
                in_frame = frames == frame
                ticks_here = np.unique(prov_tick[mask][in_frame])
                crossing = float(np.mean(ego_vel[ticks_here][:, :2] @ normal))
                med = float(np.median(offsets_n[in_frame]))
                cam_meds.append(med)
                if abs(crossing) >= 1.5:
                    used += 1
                    crossings.append(crossing)
                    meds.append(med)
            resets = np.flatnonzero(
                np.diff(np.r_[np.int32(-1), frame_of_tick]) > 0
            )
            period = (
                float(np.median(np.diff(data["tick_time"][resets])) * 1000.0)
                if len(resets) > 1
                else float("nan")
            )
            spread_m = (
                max(cam_meds) - min(cam_meds) if len(cam_meds) > 1 else 0.0
            )
            print(
                f"{name:>14} {int(mask.sum()):>7} "
                f"{len(np.unique(frames)):>6} {used:>4} "
                f"{np.median(cam_meds):>+9.3f}m {spread_m:>9.3f}m "
                f"{period:>9.0f}"
            )
        x = np.asarray(crossings)
        y = np.asarray(meds)
        if len(x) >= 5 and float(np.ptp(np.abs(x))) >= 2.0:
            slope, intercept = np.polyfit(x, y, 1)
            residual_rms = float(
                np.sqrt(np.mean((y - (intercept + slope * x)) ** 2))
            )
            slope_se_ms = (
                residual_rms
                / max(1e-9, float(np.std(x)) * math.sqrt(len(x)))
                * 1000.0
            )
            print(
                f"regression over {len(x)} crossing frames: normal offset = "
                f"{intercept:+.2f} m STATIC {slope * 1000.0:+.0f} ms "
                f"(+/-{slope_se_ms:.0f}) x crossing speed | residual rms "
                f"{residual_rms:.2f} m"
            )
            print(
                "the ms term IS the registration age error "
                "(CAMERA_FRAME_STAGING_S plus unmodelled detection "
                "latency); the static term is the tracked line's real "
                "distance from the target, zero when the target sits on "
                "it; the residual is the jitter"
            )
        else:
            span = float(np.ptp(np.abs(x))) if len(x) else 0.0
            print(
                f"only {len(x)} frames crossed the normal above 1.5 m/s "
                f"(crossing-speed span {span:.1f} m/s) -- not enough speed "
                "variation to separate a static offset from a timing "
                "error. Drive AT the structure over a range of speeds."
            )
        _ghost_columns(points, prov_tick, True)
        return

    print(
        f"{'camera':>14} {'returns':>7} {'frames':>6} {'along med':>9} "
        f"{'frame spread':>12} {'cross med':>9} {'drift/frame':>11} "
        f"{'period ms':>9} {'implied ms':>10}"
    )
    for cam_index, name in enumerate(names):
        mask = prov_cam == cam_index
        if not mask.any():
            continue
        frame_of_tick = _frame_ids(ages[:, cam_index])
        frames = frame_of_tick[prov_tick[mask]]
        mine = points[mask][:, :2] - target[None, :2]
        # Decompose each return along ITS tick's velocity when moving.
        if moving:
            vel = ego_vel[prov_tick[mask]][:, :2]
            norms = np.maximum(1e-9, np.linalg.norm(vel, axis=1))
            axes = vel / norms[:, None]
        else:
            axes = np.tile(sight, (len(mine), 1))
        along = np.einsum("ij,ij->i", mine, axes)
        cross = np.einsum("ij,ij->i", mine, np.stack(
            (-axes[:, 1], axes[:, 0]), axis=1
        ))
        frame_alongs = []
        drifts = []
        for frame in np.unique(frames):
            in_frame = frames == frame
            frame_alongs.append(float(np.median(along[in_frame])))
            ticks_here = np.unique(prov_tick[mask][in_frame])
            if len(ticks_here) > 1:
                per_tick = [
                    float(np.median(along[in_frame & (prov_tick[mask] == t)]))
                    for t in ticks_here
                ]
                drifts.append(max(per_tick) - min(per_tick))
        frame_alongs_arr = np.asarray(frame_alongs)
        spread = (
            float(frame_alongs_arr.max() - frame_alongs_arr.min())
            if len(frame_alongs_arr) > 1
            else 0.0
        )
        # Camera delivery period, from the age resets over the whole capture.
        resets = np.flatnonzero(
            np.diff(np.r_[np.int32(-1), _frame_ids(ages[:, cam_index])]) > 0
        )
        if len(resets) > 1:
            period = float(
                np.median(np.diff(data["tick_time"][resets])) * 1000.0
            )
        else:
            period = float("nan")
        speed_here = float(
            np.maximum(0.1, speeds[np.unique(prov_tick[mask])]).mean()
        )
        implied = (
            f"{float(np.median(frame_alongs_arr)) / speed_here * 1000.0:>+10.0f}"
            if moving
            else f"{'n/a':>10}"
        )
        print(
            f"{name:>14} {int(mask.sum()):>7} {len(frame_alongs_arr):>6} "
            f"{np.median(frame_alongs_arr):>+8.3f}m {spread:>11.3f}m "
            f"{np.median(cross):>+8.3f}m "
            f"{(np.median(drifts) if drifts else 0.0):>10.3f}m "
            f"{period:>9.0f} {implied}"
        )
        # Detection jitter is bounded by one probe loop tick of latency, so a
        # frame-to-frame spread beyond v x tick (with slack) cannot be
        # registration at all -- it is an extended structure's visible window
        # sliding with the car. The first moving capture's kerb spread 1.66 m
        # against a 0.25 m bound this way.
        jitter_bound = max(0.30, speed_here * loop_s * 2.0)
        if moving and spread > jitter_bound:
            extended = True
            print(
                f"{'':>14} ^ frame spread {spread:.2f} m exceeds the jitter "
                f"bound {jitter_bound:.2f} m (v x loop tick) -- this camera "
                "is tracking a sliding window, not a registration error"
            )
    if moving:
        print(
            "along med is the SYSTEMATIC registration error: implied ms = "
            "along / speed IS CAMERA_FRAME_STAGING_S if it repeats across "
            "cameras; frame spread is the jitter no constant removes"
        )

    _ghost_columns(points, prov_tick, extended)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--countdown", type=int, default=0)
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="cylinder radius (default 1.5 parked, 3.5 with a countdown)",
    )
    parser.add_argument("--tag", default="stationary")
    parser.add_argument(
        "--target", type=float, nargs=2, metavar=("X", "Y"), default=None
    )
    parser.add_argument(
        "--ahead",
        action="store_true",
        help="target the structure dead ahead (a wall face to drive at)",
    )
    parser.add_argument(
        "--analyse", default=None, help="re-run the report on a saved capture"
    )
    args = parser.parse_args()
    if args.analyse:
        _report(np.load(args.analyse))
        return 0
    return _capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
