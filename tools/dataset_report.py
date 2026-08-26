"""
Is a recorded capture session actually usable for training?

Three questions a directory listing cannot answer, and every one of them has
bitten a labelling session:

1. **Are the bays well formed?** Four clicks in the wrong ORDER make a bowtie,
   not a rectangle -- and a bowtie's corners are all in the right places, so it
   looks fine in any list of coordinates. The shoelace area against the convex
   hull's is what separates them.
2. **Was anything labelled twice?** The same bay clicked on two passes is two
   labels at one place. Harmless to a human, and to a training script it is one
   region weighted double.
3. **Do the frames actually SEE the bays?** This is the one that decides whether
   the session is worth anything. A label is only training signal in the frames
   whose camera was pointed at it, so the number that matters is not 51 frames
   and 12 bays but the count of (frame, camera, bay) triples where the bay was
   in shot.

Question 3 is answered with a bearing test, not a full unprojection: the bay
centre is put into the vehicle frame from the recorded pose and its angle
compared against the camera's own axis and half-aperture. That is an upper
bound -- it ignores occlusion and the vertical field -- and it is enough to
tell a session that saw its bays from one that drove past looking away.

    py -3.12 tools/dataset_report.py                    # the newest session
    py -3.12 tools/dataset_report.py captures/2026-...  # a named one
    py -3.12 tools/dataset_report.py --all              # every session + totals
    py -3.12 tools/dataset_report.py --repair           # ...and fix the labels

`--repair` re-winds crossed corners and drops duplicate labels, keeping the
LAST of each group -- a re-click is a correction, so the newer one wins. It
writes `bays.json.bak` first and never touches a frame. Sessions recorded after
2026-08-24 need it only for labels placed before then: `BayLabelStore` now winds
corners on the way in and replaces a re-clicked bay rather than appending it.

Reads only, unless `--repair` is given. Does NOT need BeamNG, a GPU, or Qt.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from beamng_lidar_bev.config import (  # noqa: E402
    PARKING_BAY_MAX_DEPTH_M,
    PARKING_BAY_MIN_DEPTH_M,
    PARKING_BAY_WIDTH_MAX_M,
    PARKING_BAY_WIDTH_MIN_M,
)
from beamng_lidar_bev.geometry import vehicle_axes  # noqa: E402

# Two labels closer than this are the same bay clicked twice. Well under a bay
# width, so genuine neighbours are never merged.
_DUPLICATE_M = 1.5
# A quad this far off its own convex hull's area has crossed corners.
_BOWTIE_AREA_RATIO = 0.9
# Opposite sides differing by more than this mean the quad is a trapezoid, not
# a rectangle. It matters most for the ROW tool: the row is divided evenly, so
# an outer corner clicked long makes every bay in the row taper by the same
# amount. Measured on a real 4-bay row, each bay carried a 0.28 m mismatch and
# the row ran 5.30 m deep at one end and 4.47 at the other.
_SQUARE_TOLERANCE_M = 0.25
# Range bands to report visibility over. Paint thins with distance and the
# far band is where a model earns its keep, so they are reported separately.
_BANDS = [(0.0, 10.0), (10.0, 20.0), (20.0, 35.0), (35.0, 60.0)]


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _hull_area(points: np.ndarray) -> float:
    """Monotone chain, small enough that a dependency would be silly."""
    order = sorted(map(tuple, points))
    if len(order) < 3:
        return 0.0

    def half(seq):
        out: list[tuple[float, float]] = []
        for point in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) > 0:
                    break
                out.pop()
            out.append(point)
        return out[:-1]

    hull = half(order) + half(reversed(order))
    return _shoelace(np.asarray(hull, dtype=float))


def _sides(corners: np.ndarray) -> list[float]:
    return [
        float(np.hypot(*(corners[(i + 1) % 4] - corners[i]))) for i in range(4)
    ]


def _describe_bay(index: int, corners: np.ndarray) -> tuple[str, list[str]]:
    sides = _sides(corners)
    area = _shoelace(corners)
    hull = _hull_area(corners)
    across = (sides[0] + sides[2]) / 2.0
    along = (sides[1] + sides[3]) / 2.0
    width, depth = min(across, along), max(across, along)

    problems: list[str] = []
    if hull > 0.0 and area < _BOWTIE_AREA_RATIO * hull:
        problems.append(
            f"corners crossed (area {area:.1f} vs hull {hull:.1f}) -- "
            "clicked in a Z rather than round the bay"
        )
    if not PARKING_BAY_WIDTH_MIN_M <= width <= PARKING_BAY_WIDTH_MAX_M:
        problems.append(
            f"width {width:.2f} m outside the detector's "
            f"{PARKING_BAY_WIDTH_MIN_M:.1f}-{PARKING_BAY_WIDTH_MAX_M:.1f} m"
        )
    if not PARKING_BAY_MIN_DEPTH_M <= depth <= PARKING_BAY_MAX_DEPTH_M:
        problems.append(
            f"depth {depth:.2f} m outside the detector's "
            f"{PARKING_BAY_MIN_DEPTH_M:.1f}-{PARKING_BAY_MAX_DEPTH_M:.1f} m"
        )
    skew = max(abs(sides[0] - sides[2]), abs(sides[1] - sides[3]))
    if skew > _SQUARE_TOLERANCE_M:
        problems.append(
            f"not square -- opposite sides differ by {skew:.2f} m. On a row "
            "this is one outer corner clicked long, and every bay in the row "
            "tapers by the same amount"
        )
    centre = corners.mean(axis=0)
    line = (
        f"{index:3d}  ({centre[0]:9.1f}, {centre[1]:8.1f})  "
        f"{width:5.2f} x {depth:5.2f} m  area {area:6.1f}"
    )
    return line, problems


def _camera_axis(direction: list[float]) -> np.ndarray:
    """The camera's own axis as (right, forward) in the BEV frame.

    `direction_vehicle` is BeamNG's vehicle convention: +X left, +Y rearward.
    BEV is (right, forward), so both flip sign -- the same relabelling
    `BevWidget._draw_ego` does when it draws the mounts.
    """
    axis = np.array([-direction[0], -direction[1]], dtype=float)
    return axis / max(float(np.hypot(*axis)), 1e-9)


def _repair(root: Path, bays: list[np.ndarray], duplicates) -> None:
    """Re-wind crossed corners, keep the newest of each duplicate group."""
    drop = {i for i, _, _ in duplicates}  # the older of every pair
    kept = [
        {"corners": [list(map(float, c)) for c in _wind(bay)]}
        for index, bay in enumerate(bays)
        if index not in drop
    ]
    path = root / "bays.json"
    path.replace(root / "bays.json.bak")
    path.write_text(json.dumps({"bays": kept}, indent=2), encoding="utf-8")
    print(
        f"\nREPAIRED  {len(bays)} labels -> {len(kept)}, corners re-wound; "
        "the original is in bays.json.bak"
    )


def _wind(corners: np.ndarray) -> np.ndarray:
    centre = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    return corners[np.argsort(angles)]


def report_session(root: Path, repair: bool) -> dict:
    print(f"session {root.name}\n")

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    cameras = meta["cameras"]
    print(f"vehicle  {meta['vehicle'].get('model') or '(unknown)'}")
    print(f"cameras  {', '.join(c['name'] for c in cameras)}")

    # A session recorded but never labelled is a normal thing to find and not
    # an error: the frames are fine, they just have nothing to supervise them.
    # Reported rather than crashed on, because the whole point of this tool is
    # to say what a directory of captures is worth.
    label_path = root / "bays.json"
    if label_path.is_file():
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        bays = [np.asarray(b["corners"], dtype=float) for b in payload["bays"]]
    else:
        bays = []
    records = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"frames   {len(records)} samples, {len(records) * len(cameras)} images")

    if not bays:
        print(
            "\nNO LABELS -- this session was recorded but never labelled, so "
            "its frames supervise nothing.\n  Labels live inside the session "
            "they were clicked in, so they cannot be added later: re-drive the "
            "lot with Label Bays on, or delete this directory."
        )
        return {
            "name": root.name,
            "samples": len(records),
            "images": len(records) * len(cameras),
            "bays": 0,
            "sightings": 0,
            "ranges": [],
            "per_camera": {},
            "faults": 1,
        }

    print(f"\nBAYS ({len(bays)} labelled)")
    print("idx  centre (world x, y)     size            area")
    faults: list[str] = []
    for index, corners in enumerate(bays):
        line, problems = _describe_bay(index, corners)
        print(line + ("   <-- CHECK" if problems else ""))
        faults.extend(f"  bay {index}: {problem}" for problem in problems)

    centres = np.asarray([b.mean(axis=0) for b in bays]) if bays else np.empty((0, 2))
    duplicates: list[tuple[int, int, float]] = []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            gap = float(np.hypot(*(centres[i] - centres[j])))
            if gap < _DUPLICATE_M:
                duplicates.append((i, j, gap))

    if duplicates:
        print(f"\nDUPLICATES ({len(duplicates)} pairs, same bay clicked twice)")
        for i, j, gap in duplicates:
            print(f"  bay {i} and bay {j} are {gap:.2f} m apart")
    if faults:
        print("\nMALFORMED")
        print("\n".join(faults))

    distinct = len(bays) - len({j for _, j, _ in duplicates})
    print(f"\ndistinct bays: {distinct} of {len(bays)} labels")

    # -- what the cameras actually saw ------------------------------------
    print("\nVISIBILITY  (bay centre inside a camera's horizontal aperture)")
    per_camera: dict[str, list[float]] = {c["name"]: [] for c in cameras}
    axes = {
        c["name"]: (_camera_axis(c["direction_vehicle"]),
                    math.radians(c["horizontal_fov_deg"]) / 2.0)
        for c in cameras
    }
    seen_bays: set[int] = set()
    for record in records:
        state = {
            "pos": record["ego"]["pos"],
            "dir": record["ego"]["dir"],
            "up": record["ego"]["up"],
        }
        right, forward, _ = vehicle_axes(state)
        origin = np.asarray(record["ego"]["pos"], dtype=float)[:2]
        right_xy = right[:2] / max(float(np.hypot(*right[:2])), 1e-9)
        forward_xy = forward[:2] / max(float(np.hypot(*forward[:2])), 1e-9)
        for index, centre in enumerate(centres):
            delta = centre - origin
            bev = np.array([float(delta @ right_xy), float(delta @ forward_xy)])
            distance = float(np.hypot(*bev))
            if distance < 1e-6:
                continue
            for name, (axis, half_fov) in axes.items():
                cosine = float(bev @ axis) / distance
                if math.acos(max(-1.0, min(1.0, cosine))) <= half_fov:
                    per_camera[name].append(distance)
                    seen_bays.add(index)

    total = sum(len(v) for v in per_camera.values())
    for name, ranges in per_camera.items():
        print(f"  {name:<16} {len(ranges):5d} sightings", end="")
        print(
            f", nearest {min(ranges):.1f} m, furthest {max(ranges):.1f} m"
            if ranges
            else ", none"
        )
    print(f"\n  {total} (frame, camera, bay) sightings in total")
    print(f"  {len(seen_bays)} of {len(bays)} labelled bays were seen at all")

    every = [r for ranges in per_camera.values() for r in ranges]
    if every:
        print("\n  by range:")
        for low, high in _BANDS:
            count = sum(1 for r in every if low <= r < high)
            share = 100.0 * count / len(every)
            bar = "#" * int(round(share / 2.5))
            print(f"    {low:5.0f}-{high:<5.0f} m {count:5d}  {share:5.1f}% {bar}")

    unseen = sorted(set(range(len(bays))) - seen_bays)
    if unseen:
        print(f"\n  never in shot: bays {unseen}")

    # Only what --repair can actually mend. A quad that is out of size or not
    # square is a CLICKING problem, and offering to "repair" it would promise
    # something this tool cannot do -- those want re-labelling by hand.
    mendable = bool(duplicates) or any("crossed" in fault for fault in faults)
    if mendable:
        if repair:
            _repair(root, bays, duplicates)
        else:
            print(
                "\nrun again with --repair to wind the corners and drop the "
                "duplicates (the original is kept as bays.json.bak)"
            )
    return {
        "name": root.name,
        "samples": len(records),
        "images": len(records) * len(cameras),
        "bays": len(bays),
        "sightings": total,
        "ranges": every,
        "per_camera": {k: len(v) for k, v in per_camera.items()},
        "faults": len(faults) + len(unseen),
    }


def _summarise(reports: list[dict]) -> None:
    """The whole corpus, because a model is trained on all of it at once."""
    print("\n" + "=" * 68)
    print("ALL SESSIONS")
    print(
        f"\n{'session':<20}{'samples':>9}{'images':>8}{'bays':>7}"
        f"{'sightings':>11}"
    )
    for report in reports:
        print(
            f"{report['name']:<20}{report['samples']:>9}{report['images']:>8}"
            f"{report['bays']:>7}{report['sightings']:>11}"
            + ("   <-- CHECK" if report["faults"] else "")
        )
    print(
        f"{'TOTAL':<20}{sum(r['samples'] for r in reports):>9}"
        f"{sum(r['images'] for r in reports):>8}"
        f"{sum(r['bays'] for r in reports):>7}"
        f"{sum(r['sightings'] for r in reports):>11}"
    )

    every = [r for report in reports for r in report["ranges"]]
    if every:
        print("\nrange coverage across every session:")
        for low, high in _BANDS:
            count = sum(1 for r in every if low <= r < high)
            share = 100.0 * count / len(every)
            print(
                f"  {low:5.0f}-{high:<5.0f} m {count:6d}  {share:5.1f}% "
                + "#" * int(round(share / 2.0))
            )

    # Which camera saw the bays. A row driven from one side only is served by
    # one camera, and a model trained on that learns one viewpoint.
    totals: dict[str, int] = {}
    for report in reports:
        for name, count in report["per_camera"].items():
            totals[name] = totals.get(name, 0) + count
    if totals and sum(totals.values()):
        print("\ncamera balance:")
        for name, count in sorted(totals.items()):
            share = 100.0 * count / sum(totals.values())
            print(f"  {name:<18}{count:6d}  {share:5.1f}%")


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    repair = "--repair" in sys.argv
    every = "--all" in sys.argv
    captures = Path(__file__).parents[1] / "captures"

    if every or (argv and Path(argv[0]).resolve() == captures.resolve()):
        roots = sorted(p for p in captures.glob("*/") if (p / "meta.json").is_file())
    elif argv:
        roots = [Path(argv[0])]
    else:
        sessions = sorted(
            p for p in captures.glob("*/") if (p / "meta.json").is_file()
        )
        if not sessions:
            print("no capture sessions under captures/")
            return 2
        roots = [sessions[-1]]

    if not roots:
        print("no capture sessions under captures/")
        return 2

    reports = []
    for index, root in enumerate(roots):
        if index:
            print("\n" + "-" * 68 + "\n")
        reports.append(report_session(root, repair))
    if len(reports) > 1:
        _summarise(reports)
    return 1 if any(r["faults"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
