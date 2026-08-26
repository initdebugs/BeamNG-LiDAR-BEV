"""
Turn the clicked bays into per-pixel training masks.

A label is a world quad; a model trains on an image. This walks every recorded
sample, projects the bays that were in shot, and fills them into a mask the
same size as the frame -- the bridge between `capture` and anything trained.

    py -3.12 tools/build_dataset.py                 # build it
    py -3.12 tools/build_dataset.py --preview 8     # ...and blend 8 to look at
    py -3.12 tools/build_dataset.py --val-share 0.2 # how much to hold out
    py -3.12 tools/build_dataset.py --val-session 2026-08-25_083200

Writes `dataset/` beside `captures/`: mask PNGs, plus an `index.jsonl` that
POINTS AT the original frames rather than copying them (874 MB of JPEG is not
worth duplicating, and a mask of mostly zeros compresses to a few kB).

`--val-session` nominates a session as the validation set, which is what makes
a COMPLETE session worth having: precision measured against partial labels is
unknowable, because a bay the model found and nobody clicked counts as a false
positive. It drops the rest of that session's LOT rather than putting it in
training -- see the rule below -- so a trustworthy measurement costs a few
images, which is a bargain against a number that cannot be read at all.

Five rules, and the first two are why this is a tool rather than a one-liner:

- **The mask carries THREE values, because 12 cm of clicking error means
  different things to a region and to a line.** 0 is background, 1 is the bay
  interior, 2 is a `_DIVIDER_BAND_M` band along the two long edges -- the
  painted dividers. `tools/label_overlay.py --measure` put the labelling
  scatter at a median 11.7 cm, which is nothing across a 3 x 5.5 m rectangle
  and is the whole width of a 0.12 m painted line. Train on 1 (collapsing 2
  into it) for a robust region target; train on 2 alone only knowing the target
  is about as wide as its own error. The band is built in WORLD metres and then
  projected, so it stays perspective-correct rather than a fixed pixel width.
- **The split is by LOT, never by frame.** Samples 0.5 s apart are near
  duplicates, so a random split puts near-copies of the validation frames in
  training and reports a score that means nothing. Lots are found by clustering
  the bay centres, and whole lots go one side or the other.
- **A nominated validation session takes its whole LOT out of training**, and
  the lot's other sessions are DROPPED rather than reused. They are different
  drives past the SAME bays, so training on them is training on the validation
  answers; that is the identical leak lot-splitting exists to stop, and it does
  not stop being a leak because the split was asked for by name.
- **A bay counts when all four corners are in FRONT of the lens**, not when
  they are inside the frame. A bay running off the edge is perfectly good
  signal for the part that shows; a bay behind the camera projects to a
  mirror-image quad that is arithmetically fine and completely wrong.
- **Occlusion is NOT handled, and this is the known hole.** A bay behind a
  parked car still gets filled, which teaches a model to see paint through
  metal. It needs the car positions recorded per frame, which `capture` does
  not do yet; until then the honest mitigation is to prefer lots that were
  empty when they were driven.

Reads the captures, writes the dataset. Does NOT need BeamNG, a GPU, or Qt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from beamng_lidar_bev.models import CameraMount  # noqa: E402
from beamng_lidar_bev.projection import (  # noqa: E402
    ground_points,
    place_camera,
    project,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "captures"
OUT = ROOT / "dataset"

BACKGROUND, BAY, DIVIDER, IGNORE = 0, 1, 2, 255
# How far from a labelled bay the ground is still trusted as BACKGROUND.
#
# **Without this the dataset teaches the opposite of what is wanted.** 177 bays
# were clicked and far more were driven past, so a frame routinely holds a whole
# row of obvious painted bays that nobody labelled -- seen in the very first
# preview, where the mask covered a sliver at the frame edge and left an entire
# painted row supervised as background. Systematic false negatives are the worst
# kind of label noise: the model is told the thing it is looking for is not
# there, in exactly the places it is.
#
# 4 m reaches the aisle in front of a labelled bay and the tarmac either side,
# which is the negative evidence worth having, and stays under a bay's own
# 5.5 m depth so it cannot spill onto the next row. Labelling WHOLE ROWS is what
# keeps it honest: an unlabelled bay beside a labelled one is inside the radius
# and would be mislabelled, which the row tool makes unlikely by construction.
_TRUSTED_BACKGROUND_M = 4.0
# The ignore map is computed on a coarse grid and upsampled: it decides which
# region is supervised, not where an edge falls, so a pixel-exact answer buys
# nothing and costs 16x the rays.
_IGNORE_STRIDE = 4
# Ground this far out is still ground; anything beyond it is ignored anyway.
_IGNORE_RAY_RANGE_M = 200.0
# Width of the divider band, in world metres, centred on the clicked edge, and
# it is DERIVED rather than chosen. The band has to contain the real painted
# line, which is ~0.12 m wide and sits wherever the clicking put the label:
# `label_overlay --measure` puts that offset at a median 11.7 cm, p75 17.8 and
# p90 25.3. Containing the line needs `2 * (offset + 0.06)`, so
#
#   median -> 0.35 m     p75 -> 0.48 m     p90 -> 0.63 m
#
# **At the original 0.30 m the band missed the line MORE THAN HALF THE TIME**,
# which is most of why the divider class scored an IoU of 0.125 while its
# predictions visibly followed the paint. 0.50 contains it at p75. Wider still
# would contain more of the tail and dilute the target with tarmac: a 0.63 m
# band is 22% of a 2.8 m bay.
_DIVIDER_BAND_M = 0.50
# Past this the bay is a handful of pixels and its label is mostly the
# labelling error. Matches the band the visibility histogram says the corpus
# actually covers well.
_MAX_RANGE_M = 40.0
# Two bays closer than this are the same car park. Single-link, so a long row
# chains into one lot, which is what "hold out a lot" should mean.
_LOT_LINK_M = 60.0


def _mount(entry: dict) -> CameraMount:
    return CameraMount(
        name=entry["name"],
        position_vehicle=tuple(entry["position_vehicle"]),
        direction_vehicle=tuple(entry["direction_vehicle"]),
        horizontal_fov_deg=entry["horizontal_fov_deg"],
        vertical_fov_deg=entry["vertical_fov_deg"],
        resolution=tuple(entry["resolution"]),
    )


def _lots(sessions: list[Path]) -> dict[str, int]:
    """Single-link cluster every bay centre; return session -> lot id."""
    centres, owners = [], []
    for session in sessions:
        path = session / "bays.json"
        if not path.is_file():
            continue
        for bay in json.loads(path.read_text(encoding="utf-8"))["bays"]:
            centres.append(np.asarray(bay["corners"], dtype=float).mean(axis=0))
            owners.append(session.name)
    if not centres:
        return {}
    points = np.asarray(centres)
    parent = list(range(len(points)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(points)):
        near = np.flatnonzero(
            np.linalg.norm(points[i + 1 :] - points[i], axis=1) < _LOT_LINK_M
        )
        for offset in near:
            parent[find(i)] = find(i + 1 + int(offset))

    roots = sorted({find(i) for i in range(len(points))})
    lot_of_root = {root: index for index, root in enumerate(roots)}
    session_lot: dict[str, int] = {}
    for index, owner in enumerate(owners):
        # A session's bays are all in one place in practice; first wins, and a
        # session straddling two lots simply joins the first it touched.
        session_lot.setdefault(owner, lot_of_root[find(index)])
    return session_lot


def _band_quad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A `_DIVIDER_BAND_M`-wide world quad centred on the segment a-b."""
    along = b - a
    length = float(np.hypot(*along[:2]))
    if length < 1e-6:
        return np.empty((0, 3))
    normal = np.array([-along[1], along[0], 0.0]) / length * (
        _DIVIDER_BAND_M / 2.0
    )
    return np.asarray([a - normal, b - normal, b + normal, a + normal])


def _draw_sample(
    record: dict,
    mount: CameraMount,
    sensor_origin: tuple[float, float, float],
    ground_z: float,
    bays: list[np.ndarray],
    complete: bool = False,
) -> tuple[np.ndarray | None, int, float]:
    """The mask for one (sample, camera), or None when no bay is in shot."""
    state = {
        "pos": record["ego"]["pos"],
        "dir": record["ego"]["dir"],
        "up": record["ego"]["up"],
    }
    plane_z = float(record["ego"]["pos"][2]) + ground_z
    placement = place_camera(state, mount, sensor_origin)
    width, height = mount.resolution
    mask = Image.new("L", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(mask)

    drawn = 0
    for corners in bays:
        world = np.column_stack((corners, np.full(len(corners), plane_z)))
        distance = float(np.linalg.norm(world.mean(axis=0) - placement.origin))
        if distance > _MAX_RANGE_M:
            continue
        uv, _visible = project(placement, world)
        depth = (world - placement.origin) @ placement.axis
        # In FRONT of the lens, all four. Outside the frame is fine -- PIL
        # clips -- but a corner behind the camera yields a mirror-image quad
        # that is arithmetically perfect and completely wrong.
        if not (depth > 0.05).all():
            continue
        if not _touches_frame(uv, width, height):
            continue
        draw.polygon([tuple(point) for point in uv], fill=BAY)
        drawn += 1

    if not drawn:
        return None, 0, 0.0

    # The dividers go on AFTER every interior, or a neighbour's fill would
    # paint over the band they share.
    for corners in bays:
        world = np.column_stack((corners, np.full(len(corners), plane_z)))
        if float(np.linalg.norm(world.mean(axis=0) - placement.origin)) > (
            _MAX_RANGE_M
        ):
            continue
        depth = (world - placement.origin) @ placement.axis
        if not (depth > 0.05).all():
            continue
        lengths = [
            float(np.linalg.norm(world[(i + 1) % 4] - world[i]))
            for i in range(4)
        ]
        for i in np.argsort(lengths)[-2:]:
            band = _band_quad(world[i], world[(i + 1) % 4])
            if not len(band):
                continue
            band_uv, _ = project(placement, band)
            if ((band - placement.origin) @ placement.axis > 0.05).all():
                draw.polygon(
                    [tuple(point) for point in band_uv], fill=DIVIDER
                )

    array = np.asarray(mask).copy()
    if not complete:
        # A session the labeller did NOT mark complete cannot supervise its own
        # negatives: the tarmac it never labelled may well be painted. Marked
        # complete, every pixel is known and the ignore map is skipped whole --
        # which is the entire point of the flag.
        ignore = _ignore_map(placement, plane_z, bays, width, height)
        array[(array == BACKGROUND) & ignore] = IGNORE
    positive = float(np.count_nonzero((array == BAY) | (array == DIVIDER)))
    return array, drawn, positive / array.size


def _ignore_map(
    placement, plane_z: float, bays: list[np.ndarray], width: int, height: int
) -> np.ndarray:
    """
    Which pixels are ground we know nothing about.

    A pixel is trusted background when its ray misses the ground entirely (sky,
    a building, a car -- none of which is ever a bay) or lands within
    `_TRUSTED_BACKGROUND_M` of a bay somebody actually clicked. Everything else
    is tarmac that may well be painted and was simply never labelled.
    """
    columns = np.arange(0, width, _IGNORE_STRIDE)
    rows = np.arange(0, height, _IGNORE_STRIDE)
    grid = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)
    points, hit = ground_points(
        placement,
        grid.astype(float),
        plane_z,
        max_range_m=_IGNORE_RAY_RANGE_M,
    )

    trusted = ~hit
    if bays:
        centres = np.asarray([bay.mean(axis=0) for bay in bays])
        # Circumradius, so the test is "near this bay" rather than "near its
        # centre" -- generous by design, since the cost of trusting too little
        # is only a smaller training region.
        reach = (
            np.asarray(
                [
                    np.linalg.norm(bay - bay.mean(axis=0), axis=1).max()
                    for bay in bays
                ]
            )
            + _TRUSTED_BACKGROUND_M
        )
        gap = (
            np.linalg.norm(
                points[:, None, :2] - centres[None, :, :], axis=2
            )
            - reach[None, :]
        )
        trusted |= gap.min(axis=1) <= 0.0

    coarse = (~trusted).reshape(len(rows), len(columns))
    return np.repeat(
        np.repeat(coarse, _IGNORE_STRIDE, axis=0), _IGNORE_STRIDE, axis=1
    )[:height, :width]


def _touches_frame(uv: np.ndarray, width: int, height: int) -> bool:
    """Cheap overlap test: the quad's bounding box against the frame's."""
    return (
        uv[:, 0].max() > 0.0
        and uv[:, 0].min() < width
        and uv[:, 1].max() > 0.0
        and uv[:, 1].min() < height
    )


def main() -> int:
    val_share = 0.2
    if "--val-share" in sys.argv:
        val_share = float(sys.argv[sys.argv.index("--val-share") + 1])
    nominated = {
        sys.argv[index + 1]
        for index, value in enumerate(sys.argv)
        if value == "--val-session"
    }
    previews = 0
    if "--preview" in sys.argv:
        previews = int(sys.argv[sys.argv.index("--preview") + 1])

    sessions = sorted(
        p for p in CAPTURES.glob("*/") if (p / "bays.json").is_file()
    )
    if not sessions:
        print("no labelled capture sessions under captures/")
        return 2
    session_lot = _lots(sessions)
    lot_count = len(set(session_lot.values()))
    print(f"{len(sessions)} labelled sessions over {lot_count} lots")

    # Hold out WHOLE lots, largest-first until the share is met, so validation
    # is a place the model has never seen rather than frames next to ones it has.
    per_lot: dict[int, int] = {}
    for session in sessions:
        count = sum(
            1
            for line in (session / "index.jsonl").read_text().splitlines()
            if line.strip()
        )
        per_lot[session_lot[session.name]] = (
            per_lot.get(session_lot[session.name], 0) + count
        )
    dropped_sessions: set[str] = set()
    if nominated:
        unknown = nominated - {session.name for session in sessions}
        if unknown:
            print(f"no such session: {sorted(unknown)}", file=sys.stderr)
            return 2
        for name in sorted(nominated):
            payload = json.loads(
                (CAPTURES / name / "bays.json").read_text(encoding="utf-8")
            )
            if not payload.get("complete", False):
                print(
                    f"WARNING: {name} is NOT marked complete, so holding it "
                    "out gives a precision figure scored against labels known "
                    "to be partial -- which is the number this flag exists to "
                    "make trustworthy. See tools/find_unlabelled.py.",
                    file=sys.stderr,
                )
        held = {session_lot[name] for name in nominated}
        # Everything else in those lots is a different drive past the SAME
        # bays, so training on it is training on the validation answers. That
        # is the identical leak lot-splitting exists to stop, and it does not
        # stop being a leak because the split was asked for by name.
        dropped_sessions = {
            session.name
            for session in sessions
            if session_lot[session.name] in held
            and session.name not in nominated
        }
        held_samples = sum(
            per_lot[lot] for lot in held
        )
        print(
            f"validation is {', '.join(sorted(nominated))} "
            f"(lot(s) {sorted(held)}, {held_samples} samples in those lots)"
        )
        if dropped_sessions:
            print(
                f"dropping {len(dropped_sessions)} session(s) from the same "
                "lot rather than leaking them into training: "
                + ", ".join(sorted(dropped_sessions))
            )
        print()
    else:
        total = sum(per_lot.values())
        # SMALLEST first, stopping BEFORE the share is exceeded. Largest-first
        # overshoots by a whole lot -- with one lot holding a third of the
        # corpus it held out 39% when asked for 20% -- and a lot is
        # indivisible, so the target has to be approached from below.
        held = set()
        running = 0
        for lot, count in sorted(per_lot.items(), key=lambda kv: kv[1]):
            if held and running + count > val_share * total:
                break
            held.add(lot)
            running += count
        print(
            f"holding out lot(s) {sorted(held)} for validation "
            f"({running} of {total} samples, {100 * running / total:.0f}%)\n"
        )

    masks_dir = OUT / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = OUT / "preview"
    if previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = 0
    written_previews = 0
    print(
        f"{'session':<22}{'split':<7}{'kept':>7}{'empty':>7}"
        f"{'cover':>8}{'labels':>10}"
    )
    for session in sessions:
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        mounts = [_mount(entry) for entry in meta["cameras"]]
        sensor_origin = tuple(
            meta["vehicle"].get("sensor_origin_vehicle", (0.0, 0.0, 0.0))
        )
        ground_z = float(meta["vehicle"].get("ground_z_vehicle", 0.0))
        payload = json.loads((session / "bays.json").read_text())
        bays = [
            np.asarray(bay["corners"], dtype=float) for bay in payload["bays"]
        ]
        complete = bool(payload.get("complete", False))
        records = [
            json.loads(line)
            for line in (session / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if session.name in dropped_sessions:
            print(f"{session.name:<22}{'DROPPED -- same lot as validation':<40}")
            continue
        lot = session_lot[session.name]
        if nominated:
            split = "val" if session.name in nominated else "train"
        else:
            split = "val" if lot in held else "train"
        kept, empty, coverages = 0, 0, []
        for record in records:
            for mount in mounts:
                if mount.name not in record["images"]:
                    continue
                array, drawn, coverage = _draw_sample(
                    record, mount, sensor_origin, ground_z, bays, complete
                )
                if array is None:
                    empty += 1
                    skipped += 1
                    continue
                name = f"{session.name}_{record['index']:06d}_{mount.name}.png"
                Image.fromarray(array, mode="L").save(
                    masks_dir / name, optimize=True
                )
                image_path = (session / record["images"][mount.name]).relative_to(
                    ROOT
                )
                rows.append(
                    {
                        "image": image_path.as_posix(),
                        "mask": f"masks/{name}",
                        "session": session.name,
                        "lot": lot,
                        "split": split,
                        "camera": mount.name,
                        "bays": drawn,
                        "coverage": round(coverage, 5),
                    }
                )
                kept += 1
                coverages.append(coverage)
                if (
                    previews
                    and written_previews < previews
                    and len(rows) % 37 == 0
                ):
                    _write_preview(
                        ROOT / image_path, array, preview_dir / name
                    )
                    written_previews += 1
        mean_cover = 100.0 * (sum(coverages) / len(coverages)) if coverages else 0.0
        print(
            f"{session.name:<22}{split:<7}{kept:>7}{empty:>7}"
            f"{mean_cover:>7.1f}%{'COMPLETE' if complete else 'partial':>10}"
        )

    (OUT / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "classes": {
                    "0": "background",
                    "1": "bay interior",
                    "2": f"divider band, {_DIVIDER_BAND_M} m wide",
                    "255": "IGNORE -- contributes no loss",
                },
                "trusted_background_m": _TRUSTED_BACKGROUND_M,
                "max_range_m": _MAX_RANGE_M,
                "val_lots": sorted(held),
                "lot_of_session": session_lot,
                "occlusion_handled": False,
                "note": (
                    "Class 255 must be masked out of the loss. It is ground "
                    "further than trusted_background_m from any labelled bay "
                    "-- tarmac that may well be painted and was never "
                    "labelled. Supervising it as background teaches the model "
                    "that bays are not bays. Bays behind parked cars ARE "
                    "still filled: the capture does not record where the cars "
                    "were. Split is by LOT."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    _report_balance(rows)
    print(
        f"\n{len(rows)} labelled images ({len(train)} train, {len(val)} val), "
        f"{skipped} with no bay in shot"
    )
    if val and train:
        print(
            f"validation is {100 * len(val) / len(rows):.0f}% of the set and "
            f"shares no lot with training"
        )
    if previews:
        print(f"{written_previews} previews in {preview_dir} -- LOOK AT THEM")
    return 0 if rows else 1


def _report_balance(rows: list[dict], sampled: int = 250) -> None:
    """
    What the loss will actually see, which the image count does not say.

    A segmentation set is described by its CLASS BALANCE, not its size: 2,211
    images sounds ample and would be worthless if the positive class were a
    tenth of a percent. Sampled rather than totalled because reading every mask
    back doubles the build for a number that converges in a few hundred.
    """
    if not rows:
        return
    step = max(1, len(rows) // sampled)
    counts = np.zeros(4, dtype=np.int64)
    positives: list[float] = []
    for row in rows[::step]:
        array = np.asarray(Image.open(OUT / row["mask"]))
        bay = int((array == BAY).sum())
        divider = int((array == DIVIDER).sum())
        ignored = int((array == IGNORE).sum())
        counts += (bay, divider, ignored, array.size - bay - divider - ignored)
        positives.append((bay + divider) / array.size)

    total = counts.sum()
    print()
    print("class balance, sampled:")
    for name, count in zip(
        ("bay interior", "divider band", "IGNORED", "background"), counts
    ):
        print(f"  {name:<14}{100 * count / total:5.1f}%")
    share = np.asarray(positives)
    print(
        f"  positive per image: median {100 * np.median(share):.1f}%, "
        f"p10 {100 * np.percentile(share, 10):.1f}%, "
        f"p90 {100 * np.percentile(share, 90):.1f}%"
    )
    print(
        "  a positive class this small wants class weighting or a focal loss; "
        "IGNORED must be masked out of the loss entirely,"
        " never treated as background"
    )


def _write_preview(image_path: Path, mask: np.ndarray, out: Path) -> None:
    """The frame with its mask laid over it, because a mask alone tells you
    nothing about whether it is on the paint."""
    base = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    tint = np.zeros_like(base)
    tint[mask == BAY] = (80, 200, 255)
    tint[mask == DIVIDER] = (255, 90, 90)
    # Ignored ground is drawn too, and darkly: it is the region the model is
    # told nothing about, and how much of the frame it eats is the thing worth
    # seeing.
    tint[mask == IGNORE] = (25, 25, 25)
    blended = np.where(
        (mask > 0)[:, :, None], 0.55 * base + 0.45 * tint, base
    )
    Image.fromarray(blended.astype(np.uint8)).save(out)


if __name__ == "__main__":
    sys.exit(main())
