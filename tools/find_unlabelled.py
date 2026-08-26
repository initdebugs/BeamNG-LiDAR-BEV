"""
Where is there paint that nobody labelled?

The training run measured recall 57% / precision 46%, and the diagnosis was
that the LABELS are the constraint: `build_dataset` has to ignore a third of
every mask because the tarmac away from a clicked bay may well be painted, and
the ignored third is exactly where the unlabelled bays are. The fix is to label
a session completely and tick "Every bay here is labelled" -- but re-driving
everything is a waste, and ticking the box on a session that is NOT complete is
worse than useless.

This answers both questions from the recordings alone. For every session it
walks the frames, finds the ground that is currently IGNORED, looks at what the
camera actually saw there, and reports how much of it is bright enough to be
paint -- plus the world positions of the clusters, so a session that needs more
labelling says WHERE.

    py -3.12 tools/find_unlabelled.py
    py -3.12 tools/find_unlabelled.py --stride 4     # finer, slower
    py -3.12 tools/find_unlabelled.py --mark <session>   # tick the flag

**Brightness alone was measured to be useless here, and the first version of
this tool was wrong because of it.** Rendered onto a real frame, a plain
"brighter than the ground median" test lit up the PAVEMENT and the base of a
building wall -- the flat-ground assumption drops every vertical surface onto
the ground plane, and sunlit concrete beats any threshold that paint does. It
reported 10-38% "unlabelled paint" on every session, essentially all of it
kerbs and buildings.

So a second test does the real work: **paint is a narrow RIDGE, a pavement is a
slab.** A sample counts only when the samples `_RIDGE_STEP` away on BOTH sides
of one image axis are dark. A line a few pixels wide passes; a concrete apron
metres across cannot, however bright it is. It is the same distinction
`parking._stripes` draws in world space, done in image space here because that
is where the sampling is uniform.

**It is still an INDICATOR, not proof** -- a road lane marking, a kerb edge and a
bright window mullion are all narrow bright ridges. A session reading near zero
is strong evidence it is already complete; a session reading high is a place to
go and LOOK.

Reads only. Does NOT need BeamNG, a GPU, or Qt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from beamng_lidar_bev.models import CameraMount  # noqa: E402
from beamng_lidar_bev.projection import ground_points, place_camera  # noqa: E402

CAPTURES = Path(__file__).resolve().parents[1] / "captures"

# Must match build_dataset._TRUSTED_BACKGROUND_M, or this measures a different
# region from the one the dataset actually ignores.
_TRUSTED_BACKGROUND_M = 4.0
# Only where the camera resolves paint well and the flat-ground assumption
# holds. Nearer than 5 m is mostly bonnet and shadow; past 25 m a line is a
# couple of pixels and a bright roof edge looks the same.
_NEAR_M, _FAR_M = 5.0, 25.0
# How far above the frame's own ground median a pixel must sit to be a
# CANDIDATE. Tarmac runs 90-120 and a painted line 200+, so this clears the
# shadow-to-sunlight spread without needing a fixed absolute level. On its own
# it is not enough -- see the ridge test below.
_PAINT_OVER_MEDIAN = 55.0
# How far to look either side for the ridge test, in GRID samples. At stride 8
# that is 24 px: wider than a painted line at every range this looks at
# (5-25 m), and far narrower than a pavement or an apron.
_RIDGE_STEP = 3
# How much darker the shoulders must be for the middle to be a ridge rather
# than the edge of something large.
_RIDGE_DROP = 35.0
# World cell for accumulating hits, and how far apart two clusters must be to
# be different places worth driving to.
_CELL_M = 1.0
_CLUSTER_M = 12.0


def _mount(entry: dict) -> CameraMount:
    return CameraMount(
        name=entry["name"],
        position_vehicle=tuple(entry["position_vehicle"]),
        direction_vehicle=tuple(entry["direction_vehicle"]),
        horizontal_fov_deg=entry["horizontal_fov_deg"],
        vertical_fov_deg=entry["vertical_fov_deg"],
        resolution=tuple(entry["resolution"]),
    )


def _is_ridge(field: np.ndarray) -> np.ndarray:
    """
    Is each sample a narrow bright line rather than part of a bright slab?

    True where the samples `_RIDGE_STEP` away on BOTH sides of one axis are
    `_RIDGE_DROP` darker. A painted line has dark tarmac either side of it; a
    pavement, an apron or a sunlit wall has more of itself. This is the test
    that makes the tool mean anything -- without it the answer was the kerb.
    """
    step = _RIDGE_STEP
    ridge = np.zeros(field.shape, dtype=bool)
    if field.shape[0] > 2 * step:
        middle = field[step:-step, :]
        ridge[step:-step, :] |= (middle - field[: -2 * step, :] > _RIDGE_DROP) & (
            middle - field[2 * step :, :] > _RIDGE_DROP
        )
    if field.shape[1] > 2 * step:
        middle = field[:, step:-step]
        ridge[:, step:-step] |= (middle - field[:, : -2 * step] > _RIDGE_DROP) & (
            middle - field[:, 2 * step :] > _RIDGE_DROP
        )
    return ridge


def _clusters(cells: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Single-link groups of painted cells, biggest first."""
    if not len(cells):
        return []
    parent = list(range(len(cells)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(cells)):
        near = np.flatnonzero(
            np.linalg.norm(cells[i + 1 :] - cells[i], axis=1) < _CLUSTER_M
        )
        for offset in near:
            parent[find(i)] = find(i + 1 + int(offset))
    groups: dict[int, list[int]] = {}
    for index in range(len(cells)):
        groups.setdefault(find(index), []).append(index)
    return sorted(
        ((cells[members].mean(axis=0), len(members)) for members in groups.values()),
        key=lambda item: -item[1],
    )


def _mark(name: str) -> int:
    """Set `complete` on one session's bays.json, in place.

    The flag is only a field, so unlike the labels themselves it CAN be set
    after the fact. It is deliberately a separate, explicit command rather than
    something the scan does on its own: the scan measures BRIGHTNESS and the
    flag asserts COMPLETENESS, and one is not the other.
    """
    path = CAPTURES / name / "bays.json"
    if not path.is_file():
        print(f"no such labelled session: {name}", file=sys.stderr)
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["complete"] = True
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)
    print(
        f"{name} marked COMPLETE ({len(payload['bays'])} bays). "
        "Rebuild the dataset for it to take effect."
    )
    return 0


def main() -> int:
    if "--mark" in sys.argv:
        return _mark(sys.argv[sys.argv.index("--mark") + 1])
    stride = 8
    if "--stride" in sys.argv:
        stride = int(sys.argv[sys.argv.index("--stride") + 1])

    sessions = sorted(
        p for p in CAPTURES.glob("*/") if (p / "bays.json").is_file()
    )
    if not sessions:
        print("no labelled capture sessions under captures/")
        return 2

    print(
        f"{'session':<22}{'bays':>6}{'flag':>10}{'painted':>10}"
        f"{'unlabelled ground':>19}"
    )
    verdicts: list[tuple[str, float, list]] = []
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

        centres = (
            np.asarray([bay.mean(axis=0) for bay in bays])
            if bays
            else np.empty((0, 2))
        )
        reach = (
            np.asarray(
                [
                    np.linalg.norm(bay - bay.mean(axis=0), axis=1).max()
                    for bay in bays
                ]
            )
            + _TRUSTED_BACKGROUND_M
            if bays
            else np.empty(0)
        )

        painted, checked = 0, 0
        hits: list[np.ndarray] = []
        for record in records[::5]:
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
                width, height = mount.resolution
                columns = np.arange(0, width, stride)
                rows = np.arange(0, height, stride)
                grid = np.stack(
                    np.meshgrid(columns, rows), axis=-1
                ).reshape(-1, 2).astype(float)
                points, hit = ground_points(
                    placement, grid, plane_z, max_range_m=_FAR_M
                )
                if not hit.any():
                    continue
                distance = np.linalg.norm(
                    points[:, :2] - placement.origin[None, :2], axis=1
                )
                usable = hit & (distance >= _NEAR_M)
                if len(centres):
                    gap = (
                        np.linalg.norm(
                            points[:, None, :2] - centres[None, :, :], axis=2
                        )
                        - reach[None, :]
                    )
                    usable &= gap.min(axis=1) > 0.0
                if not usable.any():
                    continue

                grey = np.asarray(
                    Image.open(
                        session / record["images"][mount.name]
                    ).convert("L"),
                    dtype=float,
                )
                sampled = grey[
                    np.clip(grid[:, 1].astype(int), 0, height - 1),
                    np.clip(grid[:, 0].astype(int), 0, width - 1),
                ]
                # The frame's OWN ground median, so a shadowed lot and a
                # sunlit one are judged on the same terms.
                floor = np.median(sampled[hit])
                bright = usable & (sampled > floor + _PAINT_OVER_MEDIAN)
                bright &= _is_ridge(
                    sampled.reshape(len(rows), len(columns))
                ).reshape(-1)
                checked += int(usable.sum())
                painted += int(bright.sum())
                if bright.any():
                    hits.append(points[bright][:, :2])

        share = 100.0 * painted / max(checked, 1)
        cells = (
            np.unique(
                np.floor(np.concatenate(hits) / _CELL_M).astype(np.int64),
                axis=0,
            ).astype(float)
            * _CELL_M
            if hits
            else np.empty((0, 2))
        )
        found = _clusters(cells)
        print(
            f"{session.name:<22}{len(bays):>6}"
            f"{'COMPLETE' if complete else 'partial':>10}"
            f"{share:>9.1f}%{f'{len(cells)} cells':>19}"
        )
        verdicts.append((session.name, share, found))

    print("\nVERDICT  (bright is not proof of paint -- go and LOOK)")
    ready = [name for name, share, _ in verdicts if share < 1.0]
    for name, share, found in verdicts:
        if share < 1.0:
            continue
        where = ", ".join(
            f"({centre[0]:.0f}, {centre[1]:.0f}) x{count}"
            for centre, count in found[:3]
        )
        print(f"  {name}: {share:.1f}% unlabelled paint -- {where}")
    if ready:
        print(
            "\nEffectively complete already, and safe to tick "
            '"Every bay here is labelled":'
        )
        for name in ready:
            print(f"  {name}")
    else:
        print("\nNo session reads as already complete.")
    print(
        "\nAdding LABELS to a finished session is impossible -- they are "
        "clicked in the app,\nagainst a live WORLD view. The FLAG is only a "
        "field in bays.json, so a session you\nare confident about can be "
        "marked from here:\n"
        "  py -3.12 tools/find_unlabelled.py --mark <session-name>\n"
        "Confirm it by eye first: a wrong flag turns every missed bay into a "
        "confident\nfalse negative, which is worse than leaving the ground "
        "ignored."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
