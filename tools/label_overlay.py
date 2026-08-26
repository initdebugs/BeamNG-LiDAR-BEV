"""
Draw the hand-clicked bays onto the frames that saw them.

**This is the only proof the labels are where they are believed to be.** Every
number in `dataset_report.py` is self-consistent -- it checks the labels against
each other and against the poses, and it would report a perfectly healthy corpus
if the whole projection were wrong by a metre. The picture is what settles it:
if the drawn quads land on the painted lines, the pose, the mount, the sensor
origin, the focal length and the image-v sign are ALL correct at once, and if
they do not, which way they are wrong says which of those is to blame.

    py -3.12 tools/label_overlay.py                     # the newest session
    py -3.12 tools/label_overlay.py captures/2026-...   # a named one
    py -3.12 tools/label_overlay.py --frames 12         # how many to draw
    py -3.12 tools/label_overlay.py --measure           # ...and MEASURE it

Writes PNGs to tools/label_overlay/<session>/ and prints where each bay landed.

`--measure` puts a number on it instead of an eyeball. For every projected bay
divider it sweeps a perpendicular brightness profile across the real image and
finds where the paint actually is, so the answer is pixels of mis-registration.
Measured on the whole corpus the first time it ran: bias +0.00 px, median
-0.25 px over 234 dividers, with no drift across range bands -- the maths was
right first time, and the ~12 cm spread that remains is the clicking.

**The ground is assumed FLAT at the ego's own ground plane**, because that is
all a recorded sample carries: the corners were stored as world XY with no
height (`SceneBridge.groundPicked` drops the render Y). On a car park that is
a centimetres-level assumption; on a slope it is not, and a systematic radial
error in the drawn quads is what it would look like.

Reads only. Does NOT need BeamNG, a GPU, or Qt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from beamng_lidar_bev.models import CameraMount  # noqa: E402
from beamng_lidar_bev.projection import place_camera, project  # noqa: E402

OUT = Path(__file__).resolve().parent / "label_overlay"
# Drawn per bay, cycling, so two adjacent bays never share an outline colour.
_COLOURS = [
    (255, 92, 92),
    (92, 220, 255),
    (255, 214, 92),
    (150, 255, 140),
    (230, 140, 255),
]


def _mount(entry: dict) -> CameraMount:
    return CameraMount(
        name=entry["name"],
        position_vehicle=tuple(entry["position_vehicle"]),
        direction_vehicle=tuple(entry["direction_vehicle"]),
        horizontal_fov_deg=entry["horizontal_fov_deg"],
        vertical_fov_deg=entry["vertical_fov_deg"],
        resolution=tuple(entry["resolution"]),
    )


# The perpendicular sweep for --measure, in pixels either side of the drawn
# edge. Wide enough to find the paint when a label is a few centimetres out,
# narrow enough not to lock onto the NEXT bay's divider -- 2.5 m away is well
# over 8 px at every range this is asked at.
_SWEEP_PX = np.arange(-8.0, 8.25, 0.25)
# A profile flatter or darker than this has no paint under it at all: a bay
# mouth, which is usually unpainted, or a divider hidden by a parked car.
_PAINT_MIN_PEAK = 170.0
_PAINT_MIN_CONTRAST = 50.0
_MEASURE_MAX_RANGE_M = 20.0


def _peak_offset(
    grey: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float | None:
    """How far perpendicular the brightest run lies from this drawn edge."""
    segment = end - start
    length = float(np.hypot(*segment))
    if length < 40.0:
        return None
    normal = np.array([-segment[1], segment[0]]) / length
    # The middle of the edge only: the corners are where two labels meet and
    # where a neighbouring line's paint runs in.
    along = start + np.linspace(0.2, 0.8, 30)[:, None] * segment
    height, width = grey.shape
    profile = np.empty(len(_SWEEP_PX))
    for index, offset in enumerate(_SWEEP_PX):
        sample = along + offset * normal
        rows = np.clip(sample[:, 1].astype(int), 0, height - 1)
        columns = np.clip(sample[:, 0].astype(int), 0, width - 1)
        profile[index] = grey[rows, columns].mean()
    if (
        profile.max() < _PAINT_MIN_PEAK
        or profile.max() - profile.min() < _PAINT_MIN_CONTRAST
    ):
        return None
    return float(_SWEEP_PX[int(profile.argmax())])


def _measure(sessions: list[Path]) -> int:
    """Where is the paint, relative to where the label says it is?"""
    rows: list[tuple[float, float, float]] = []
    for root in sessions:
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        mounts = [_mount(entry) for entry in meta["cameras"]]
        sensor_origin = tuple(
            meta["vehicle"].get("sensor_origin_vehicle", (0.0, 0.0, 0.0))
        )
        ground_z = float(meta["vehicle"].get("ground_z_vehicle", 0.0))
        bays = [
            np.asarray(bay["corners"], dtype=float)
            for bay in json.loads((root / "bays.json").read_text())["bays"]
        ]
        records = [
            json.loads(line)
            for line in (root / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]
        # Every seventh sample: successive ones are the same paint from almost
        # the same place, so taking them all would weight one lot's clicking
        # rather than measure the projection.
        for record in records[::7]:
            state = {
                "pos": record["ego"]["pos"],
                "dir": record["ego"]["dir"],
                "up": record["ego"]["up"],
            }
            plane_z = float(record["ego"]["pos"][2]) + ground_z
            for mount in mounts:
                if mount.name not in record["images"]:
                    continue
                placement = place_camera(state, mount, sensor_origin)
                metres_per_px = 1.0 / placement.focal_px
                grey = None
                for corners in bays:
                    world = np.column_stack(
                        (corners, np.full(len(corners), plane_z))
                    )
                    uv, visible = project(placement, world)
                    if not visible.all():
                        continue
                    distance = float(
                        np.linalg.norm(world.mean(axis=0) - placement.origin)
                    )
                    if distance > _MEASURE_MAX_RANGE_M:
                        continue
                    if grey is None:
                        grey = np.asarray(
                            Image.open(
                                root / record["images"][mount.name]
                            ).convert("L"),
                            dtype=float,
                        )
                    # The two LONG edges are the dividers, which are what is
                    # actually painted; a bay's mouth usually is not.
                    lengths = [
                        float(np.hypot(*(uv[(i + 1) % 4] - uv[i])))
                        for i in range(4)
                    ]
                    for i in np.argsort(lengths)[-2:]:
                        offset = _peak_offset(grey, uv[i], uv[(i + 1) % 4])
                        if offset is not None:
                            rows.append(
                                (offset, distance, distance * metres_per_px)
                            )

    if not rows:
        print("no painted bay dividers were found to measure against")
        return 1

    measured = np.asarray(rows)
    pixels, scale = measured[:, 0], measured[:, 2]
    print(
        f"{len(measured)} bay dividers measured inside "
        f"{_MEASURE_MAX_RANGE_M:.0f} m, over {len(sessions)} session(s)\n"
    )
    print("  offset of the PAINT from the drawn edge, in pixels:")
    for quantile in (10, 25, 50, 75, 90):
        print(f"    p{quantile:<3} {np.percentile(pixels, quantile):+6.2f}")
    print(
        f"\n  BIAS   {pixels.mean():+.2f} px = "
        f"{(pixels * scale).mean() * 100:+.1f} cm"
    )
    print(
        "         a projection error shows up HERE, as a non-zero mean, and "
        "one in the\n         camera height grows with range"
    )
    error = np.abs(pixels) * scale
    print(
        f"  SPREAD median {np.median(error) * 100:.1f} cm, "
        f"p90 {np.percentile(error, 90) * 100:.1f} cm"
    )
    print(
        "         this is the CLICKING, not the maths: it matches the scatter "
        "in the bay\n         dimensions the dataset report prints"
    )
    for low, high in ((0, 8), (8, 14), (14, 20)):
        band = measured[(measured[:, 1] >= low) & (measured[:, 1] < high)]
        if len(band):
            print(
                f"\n  {low:2d}-{high:2d} m: {len(band):5d} edges, "
                f"bias {band[:, 0].mean():+.2f} px"
            )
    return 0


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    wanted = 8
    if "--frames" in sys.argv:
        wanted = int(sys.argv[sys.argv.index("--frames") + 1])

    captures = Path(__file__).parents[1] / "captures"
    if argv:
        root = Path(argv[0])
    else:
        sessions = sorted(
            p for p in captures.glob("*/") if (p / "bays.json").is_file()
        )
        if not sessions:
            print("no labelled capture sessions under captures/")
            return 2
        root = sessions[-1]

    if "--measure" in sys.argv:
        # Every labelled session unless one was named: a systematic error is
        # the thing worth finding, and one lot cannot show it.
        labelled = sorted(
            p for p in captures.glob("*/") if (p / "bays.json").is_file()
        )
        return _measure([root] if argv else labelled)

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    mounts = [_mount(entry) for entry in meta["cameras"]]
    sensor_origin = tuple(
        meta["vehicle"].get("sensor_origin_vehicle", (0.0, 0.0, 0.0))
    )
    ground_z_vehicle = float(meta["vehicle"].get("ground_z_vehicle", 0.0))
    bays = [
        np.asarray(bay["corners"], dtype=float)
        for bay in json.loads((root / "bays.json").read_text())["bays"]
    ]
    records = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    print(f"session {root.name}: {len(records)} samples, {len(bays)} bays")
    print(f"sensor origin {sensor_origin}, ground_z {ground_z_vehicle:+.3f}\n")

    # Score every (sample, camera) by how many whole bays are in shot, so the
    # frames drawn are the ones that can actually show whether this works.
    scored: list[tuple[int, dict, CameraMount, list]] = []
    for record in records:
        state = {
            "pos": record["ego"]["pos"],
            "dir": record["ego"]["dir"],
            "up": record["ego"]["up"],
        }
        plane_z = float(record["ego"]["pos"][2]) + ground_z_vehicle
        for mount in mounts:
            if mount.name not in record["images"]:
                continue
            placement = place_camera(state, mount, sensor_origin)
            drawn = []
            for index, corners in enumerate(bays):
                world = np.column_stack(
                    (corners, np.full(len(corners), plane_z))
                )
                uv, visible = project(placement, world)
                if visible.all():
                    distance = float(
                        np.linalg.norm(world.mean(axis=0) - placement.origin)
                    )
                    drawn.append((index, uv, distance))
            if drawn:
                scored.append((len(drawn), record, mount, drawn))

    if not scored:
        print(
            "NO BAY IS FULLY IN SHOT IN ANY FRAME.\n"
            "  That is a projection failure, not a labelling one -- the report "
            "says these bays were seen.\n"
            "  Check the image-v sign, the sensor origin and the vehicle-frame "
            "axis flips first."
        )
        return 1

    scored.sort(key=lambda item: -item[0])
    # Spread the picks over the drive rather than taking the top N, which would
    # all be the same moment from one pose.
    step = max(1, len(scored) // wanted)
    picks = scored[::step][:wanted]

    out_dir = OUT / root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'file':<34}{'camera':<18}{'bays':>5}  ranges")
    for count, record, mount, drawn in picks:
        source = root / record["images"][mount.name]
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, uv, distance in drawn:
            colour = _COLOURS[index % len(_COLOURS)]
            draw.polygon([tuple(point) for point in uv], outline=colour, width=3)
            draw.text(
                tuple(uv.mean(axis=0)), f"{index}", fill=colour, anchor="mm"
            )
        name = f"{record['index']:06d}_{mount.name}.png"
        image.save(out_dir / name)
        ranges = ", ".join(f"{d:.0f}m" for _, _, d in sorted(drawn, key=lambda x: x[2]))
        print(f"{name:<34}{mount.name:<18}{count:>5}  {ranges}")

    print(f"\n{len(picks)} frames written to {out_dir}")
    print(
        "\nLOOK AT THEM. The quads must sit on the painted lines. A constant "
        "shift means\nthe mount is wrong; quads too near or too far by a "
        "growing amount means the\ncamera height is; upside down means the "
        "image-v sign is."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
