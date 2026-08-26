"""
Recording camera frames to disk, and the hand-placed bay labels beside them.

**This module exists because the annotation channel cannot be the training
signal.** Measured on two lots in one session: one annotates its bay dividers
as thin lines, the other annotates whole bay quads as solid slabs, and the
stripe sweep in `parking` finds nothing at all on the second. The camera sees
the same thing on both -- thin white lines on grey tarmac -- so the labels have
to come from a source that does not vary per lot, and the only such source is a
person clicking on the ground.

It lives in the app rather than under `tools/` for one hard reason: **the
BeamNG bridge takes exactly one client.** Every probe under `tools/` carries
"STOP in the app, or close it" in its docstring for that reason, so a separate
capture script cannot run while the app is driving. Recording has to happen
where the connection already is.

Two stores, deliberately separate files under one session directory:

- `CaptureSession` writes frames and poses. It is fed from the 40 ms worker
  tick, so **nothing here may block it**: the queue is bounded and a full
  queue DROPS the sample and counts it rather than waiting. A disk stall must
  never delay `_actuate`, which is the rule the camera digest was already
  written to (digesting whole buffers cost a measured 9.8 ms of the tick and
  sat ahead of every control command).
- `BayLabelStore` holds the world-space quads a person clicked. Tiny, so it is
  rewritten whole on every change and replaced atomically -- a crash mid-drive
  must not cost the labelling that was already done.

Labels live INSIDE the recording session on purpose. A bay is only meaningful
next to the frames that saw it, and pairing them this way sidesteps having to
invent a stable identity for the map: two levels can use overlapping world
coordinates, and nothing in the bridge reports a level name this code can rely
on across BeamNG versions.

Qt-free and BeamNGpy-free, like `planner`, `aeb` and `parking`: config plus
numpy plus the standard library. Pillow is imported lazily on the writer
thread, so importing this module stays cheap and side-effect free -- the same
reason `worker` imports beamngpy inside its methods.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (
    CAPTURE_JPEG_QUALITY,
    CAPTURE_MIN_FREE_MB,
    CAPTURE_QUEUE_DEPTH,
    LABEL_BAY_CORNERS,
    LABEL_DUPLICATE_M,
    LABEL_SNAP_M,
    PARKING_BAY_MAX_DEPTH_M,
    PARKING_BAY_MIN_DEPTH_M,
    PARKING_BAY_WIDTH_MAX_M,
    PARKING_BAY_WIDTH_MIN_M,
)

LOGGER = logging.getLogger(__name__)

# How often the writer re-checks free space, in samples. Every sample would be
# a stat() call per frame for a number that moves at megabytes per minute.
_DISK_CHECK_EVERY = 20


@dataclass(frozen=True)
class CameraPose:
    """One camera's mount, carried per sample so a re-rig cannot desync it.

    Recorded rather than looked up at training time because the rig is derived
    from the VEHICLE (`derive_hybrid_camera_rig` measures from the body faces
    in the simulator's own sensor frame), so a different car gives different
    mounts. A dataset that assumed today's constants would silently mislabel
    every frame shot in another vehicle.
    """

    name: str
    position_vehicle: tuple[float, float, float]
    direction_vehicle: tuple[float, float, float]
    horizontal_fov_deg: float
    vertical_fov_deg: float
    resolution: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position_vehicle": list(self.position_vehicle),
            "direction_vehicle": list(self.direction_vehicle),
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "resolution": list(self.resolution),
        }


@dataclass(frozen=True)
class EgoPose:
    """Where the car was when the shutter fired.

    `pos`/`dir`/`up` are BeamNG world vectors, exactly as `_get_vehicle_state`
    returns them, and are NOT reduced to a yaw here. The unprojection needs the
    full frame, and this project has been bitten twice by a convention guessed
    rather than measured -- so the raw triples are stored and the interpretation
    is left to whatever reads them.
    """

    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    up: tuple[float, float, float]
    speed_mps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pos": list(self.pos),
            "dir": list(self.dir),
            "up": list(self.up),
            "speed_mps": self.speed_mps,
        }


@dataclass(frozen=True)
class _PendingSample:
    index: int
    timestamp: float
    ego: EgoPose
    images: tuple[tuple[str, np.ndarray], ...]


@dataclass
class CaptureStats:
    """What the session has actually done, for the log line and the badge."""

    saved: int = 0
    dropped: int = 0
    errors: int = 0
    bytes_written: int = 0
    stopped_reason: str | None = None

    def describe(self) -> str:
        parts = [f"{self.saved} samples", f"{self.bytes_written / 1e6:.0f} MB"]
        if self.dropped:
            parts.append(f"{self.dropped} DROPPED")
        if self.errors:
            parts.append(f"{self.errors} write errors")
        if self.stopped_reason:
            parts.append(f"stopped: {self.stopped_reason}")
        return ", ".join(parts)


class CaptureSession:
    """One recording run: a directory, a writer thread and a bounded queue."""

    def __init__(
        self,
        root: Path,
        cameras: Sequence[CameraPose],
        vehicle: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._cameras = tuple(cameras)
        self._index_path = self.root / "index.jsonl"
        self._queue: queue.Queue[_PendingSample | None] = queue.Queue(
            maxsize=CAPTURE_QUEUE_DEPTH
        )
        self._lock = threading.Lock()
        self._stats = CaptureStats()
        self._next_index = 0
        self._closed = False
        self._write_meta(vehicle)
        self._thread = threading.Thread(
            target=self._run, name="capture-writer", daemon=True
        )
        self._thread.start()

    # -- worker-thread side ------------------------------------------------

    def offer(
        self, ego: EgoPose, images: Sequence[tuple[str, np.ndarray]]
    ) -> bool:
        """
        Queue one sample. Never blocks, and never raises into the tick.

        Returns False when the sample was dropped -- a full queue (the disk is
        not keeping up) or a session already stopped. The caller does not have
        to care; the count is what the log line reports. The arrays are taken
        BY REFERENCE, which is safe because `worker._acquire_hybrid_camera_images`
        rebinds `_hybrid_camera_frames[name]` to a fresh copy on every refresh
        rather than writing into the array in place, so a held reference can
        never be overwritten from under the writer.
        """
        if self._closed or not images:
            return False
        sample = _PendingSample(
            index=self._next_index,
            timestamp=time.time(),
            ego=ego,
            images=tuple(images),
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1
            return False
        self._next_index += 1
        return True

    @property
    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(**vars(self._stats))

    @property
    def stopped_reason(self) -> str | None:
        with self._lock:
            return self._stats.stopped_reason

    def close(self, timeout_s: float = 5.0) -> CaptureStats:
        """Flush what is queued and join the writer. Bounded, like every wait
        on this project's worker thread -- a hung writer must not hang teardown.
        """
        if self._closed:
            return self.stats
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The sentinel cannot be dropped or the thread never ends, so make
            # room by discarding one pending sample: losing a frame at the end
            # of a session is nothing, a non-joining thread is a leak.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=timeout_s)
        return self.stats

    # -- writer-thread side ------------------------------------------------

    def _run(self) -> None:
        from PIL import Image  # lazy: keeps importing this module cheap

        written_since_check = 0
        with self._index_path.open("a", encoding="utf-8") as index:
            while True:
                sample = self._queue.get()
                if sample is None:
                    index.flush()
                    return
                if written_since_check >= _DISK_CHECK_EVERY:
                    written_since_check = 0
                    if not self._check_disk():
                        index.flush()
                        return
                written_since_check += 1
                try:
                    record = self._write_sample(Image, sample)
                except Exception as exc:  # one bad frame must not end the run
                    with self._lock:
                        self._stats.errors += 1
                    LOGGER.warning("Capture write failed: %s", exc)
                    continue
                index.write(json.dumps(record) + "\n")
                index.flush()

    def _write_sample(self, image_module: Any, sample: _PendingSample) -> dict:
        files: dict[str, str] = {}
        written = 0
        for name, rgba in sample.images:
            # RGBA -> RGB. The fourth byte is NOT opacity on this sensor (see
            # `vision_view`'s Format_RGBX8888), so it is dropped rather than
            # composited -- compositing against it would darken every frame by
            # a channel that means nothing.
            filename = f"{sample.index:06d}_{name}.jpg"
            path = self.frames_dir / filename
            image_module.fromarray(np.ascontiguousarray(rgba[:, :, :3])).save(
                path, format="JPEG", quality=CAPTURE_JPEG_QUALITY
            )
            files[name] = f"frames/{filename}"
            written += path.stat().st_size
        with self._lock:
            self._stats.saved += 1
            self._stats.bytes_written += written
        return {
            "index": sample.index,
            "t": sample.timestamp,
            "ego": sample.ego.as_dict(),
            "images": files,
        }

    def _check_disk(self) -> bool:
        try:
            free_mb = shutil.disk_usage(self.root).free / 1e6
        except OSError:
            return True
        if free_mb >= CAPTURE_MIN_FREE_MB:
            return True
        with self._lock:
            self._stats.stopped_reason = f"only {free_mb:.0f} MB free"
        LOGGER.warning(
            "Capture check: stopping, only %.0f MB free on the capture drive",
            free_mb,
        )
        return False

    def _write_meta(self, vehicle: dict[str, Any] | None) -> None:
        meta = {
            "created": time.time(),
            "cameras": [camera.as_dict() for camera in self._cameras],
            "vehicle": vehicle or {},
            "jpeg_quality": CAPTURE_JPEG_QUALITY,
        }
        (self.root / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


@dataclass(frozen=True)
class LabelledBay:
    """One bay a person clicked, as world XY corners wound round the quad."""

    corners: tuple[tuple[float, float], ...]

    @property
    def centre(self) -> tuple[float, float]:
        xs = [corner[0] for corner in self.corners]
        ys = [corner[1] for corner in self.corners]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def as_dict(self) -> dict[str, Any]:
        return {"corners": [list(corner) for corner in self.corners]}


@dataclass(frozen=True)
class LabelResult:
    """What one completed quad did to the store.

    `replaced_index` is not decoration: a re-click that silently appended is
    exactly how the first real session ended up reporting twelve bays over nine
    places, and the count on screen is the only thing a person labelling has to
    go on.
    """

    bays: tuple[LabelledBay, ...]
    """Every bay the completed quad produced -- one, or a whole divided row."""
    indices: tuple[int, ...]
    replaced_indices: tuple[int, ...] = ()
    snapped_corners: int = 0
    """How many of the four landed on a corner already labelled. A row of bays
    shares its dividers, so on an interior bay this should read 2."""

    @property
    def bay(self) -> LabelledBay:
        return self.bays[0]

    @property
    def index(self) -> int:
        return self.indices[0]


def _lerp(
    a: tuple[float, float], b: tuple[float, float], t: float
) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _plausible(corners: Sequence[tuple[float, float]]) -> bool:
    """Is this quad the size the bay detector would accept?"""
    sides = [
        math.dist(corners[i], corners[(i + 1) % 4]) for i in range(4)
    ]
    across = (sides[0] + sides[2]) / 2.0
    along = (sides[1] + sides[3]) / 2.0
    width, depth = min(across, along), max(across, along)
    return (
        PARKING_BAY_WIDTH_MIN_M <= width <= PARKING_BAY_WIDTH_MAX_M
        and PARKING_BAY_MIN_DEPTH_M <= depth <= PARKING_BAY_MAX_DEPTH_M
    )


def split_row(
    corners: Sequence[tuple[float, float]], count: int
) -> tuple[list[tuple[tuple[float, float], ...]], str | None]:
    """
    Divide one wound quad into `count` bays across the row.

    Returns the bays and, when it cannot, the reason in words.

    **The split axis is not asked for and not inferred from click order.** The
    obvious rule -- divide the longer side -- is wrong at small counts: two
    2.4 m bays are 4.8 m across against a 5.5 m depth, so the longer side is
    the depth and the row would be sliced the wrong way into two 2.75 m-deep
    bays no lot has. Both axes are tried instead and the one whose bays land
    inside the detector's own size bounds wins.

    **At two bays that test does not separate them, and the quad is REFUSED
    rather than guessed.** Measured over the detector's bounds (width 2.1-3.4,
    depth 3.6-7.5): a 4.8x5.5, a 6.0x5.0 and a 6.4x5.5 quad all divide two ways
    that are each entirely plausible, and no tie-break survives contact -- prefer
    the longer side and 4.8x5.5 goes wrong, prefer the squarer bays and 6.0x5.0
    does. At three and above the sizes resolve it uniquely every time, because
    slicing a bay's DEPTH three ways leaves something under a metre wide.
    Two bays is eight clicks; inventing their orientation is not worth it.
    """
    if count <= 1:
        return [tuple(corners)], None

    def strips(a0, a1, b0, b1):
        out = []
        for index in range(count):
            lo, hi = index / count, (index + 1) / count
            out.append(
                (
                    _lerp(a0, a1, lo),
                    _lerp(a0, a1, hi),
                    _lerp(b0, b1, hi),
                    _lerp(b0, b1, lo),
                )
            )
        return out

    c0, c1, c2, c3 = corners
    # Wound order, so c0->c1 is opposite c3->c2, and c1->c2 opposite c0->c3.
    workable = [
        bays
        for bays in (strips(c0, c1, c3, c2), strips(c1, c2, c0, c3))
        if all(_plausible(bay) for bay in bays)
    ]
    if not workable:
        return [], (
            f"{count} bays do not fit that quad at either orientation -- "
            "check the count, or Undo and re-click the row"
        )
    if len(workable) > 1:
        return [], (
            f"that quad could be {count} bays divided either way, and "
            "guessing would invent them -- label them one at a time, or "
            "click a longer row"
        )
    return workable[0], None


def _wound(corners: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Sort corners by bearing about their centroid, so click ORDER cannot matter.

    Four corners clicked in a Z rather than round the bay describe a bowtie:
    every corner is in the right place and the polygon between them crosses
    itself, which no coordinate list looks wrong in. Measured on the first real
    session, one bay of twelve came back with a shoelace area of 0.2 m2 against
    a true 17.5. Sorting by angle always yields the simple polygon for a convex
    set, and a parking bay is convex, so this removes the failure rather than
    detecting it.
    """
    cx = sum(corner[0] for corner in corners) / len(corners)
    cy = sum(corner[1] for corner in corners) / len(corners)
    return tuple(
        sorted(corners, key=lambda c: math.atan2(c[1] - cy, c[0] - cx))
    )


class BayLabelStore:
    """The hand-placed bay quads for one recording session.

    Rewritten whole on every change: the file is a few kilobytes and the write
    is atomic (temp file plus replace), so a crash costs the click in progress
    and never the hour of clicking before it. Appending would be cheaper and
    would leave a truncated line to parse around.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._bays: list[LabelledBay] = []
        self._pending: list[tuple[float, float]] = []
        self._snapped = 0
        self._complete = False
        self.split_refusal: str | None = None
        """Why the last completed quad produced no bays, in words for the user.

        Set beside the None return rather than raised: a refused quad is an
        ordinary outcome of a wrong count, not an error, and the corners stay
        pending so only the number has to change."""

    @property
    def bay_count(self) -> int:
        return len(self._bays)

    @property
    def complete(self) -> bool:
        """Whether EVERY bay this session drove past has been labelled.

        A claim about the session, not about a lot, because that is what the
        frames belong to -- and it is the difference between a dataset that can
        supervise its own negatives and one that cannot. Partial labelling is
        what forces `build_dataset` to ignore a third of every mask, and the
        ignored third is exactly the tarmac where the unlabelled bays are, so
        the model is never told that any particular patch of tarmac is NOT a
        bay. Measured on the first trained run: recall 57%, precision 46%, and
        a precision figure that cannot be trusted because the labels it is
        scored against are known to be incomplete.

        Ticking it when it is not true is worse than leaving it off: it turns
        every unlabelled bay into a confident false negative.
        """
        return self._complete

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def bays(self) -> tuple[LabelledBay, ...]:
        return tuple(self._bays)

    @property
    def pending_corners(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._pending)

    def add_corner(
        self, x: float, y: float, row_bays: int = 1
    ) -> LabelResult | None:
        """
        Drop one corner. Returns the result when the fourth completes the quad.

        `row_bays` above 1 treats the quad as a whole ROW and divides it, which
        is how a lot gets labelled when its interior dividers are not in the
        data to be clicked -- see `split_row`. Returns None (and keeps the
        corners) when the division produces nothing plausible, so a wrong count
        costs a retry rather than a set of invented bays.
        """
        corner, snapped = self._snap((float(x), float(y)))
        self._pending.append(corner)
        self._snapped += int(snapped)
        if len(self._pending) < LABEL_BAY_CORNERS:
            return None

        wound = _wound(self._pending)
        quads, self.split_refusal = split_row(wound, max(1, int(row_bays)))
        if not quads:
            # Deliberately keeps `_pending` intact: the corners are fine, the
            # COUNT was wrong, and re-clicking four corners to fix a number
            # would be the wrong thing to ask for. Undo abandons them.
            return None

        snapped_corners, self._snapped = self._snapped, 0
        self._pending = []
        bays, indices, replaced = [], [], []
        for quad in quads:
            bay = LabelledBay(corners=tuple(quad))
            existing = self._nearest_within(bay.centre, LABEL_DUPLICATE_M)
            if existing is None:
                self._bays.append(bay)
                indices.append(len(self._bays) - 1)
            else:
                self._bays[existing] = bay
                indices.append(existing)
                replaced.append(existing)
            bays.append(bay)
        self._save()
        return LabelResult(
            bays=tuple(bays),
            indices=tuple(indices),
            replaced_indices=tuple(replaced),
            snapped_corners=snapped_corners,
        )

    def _snap(
        self, corner: tuple[float, float]
    ) -> tuple[tuple[float, float], bool]:
        """Land exactly on a corner already labelled, if one is close enough.

        Searches the pending quad as well as the saved bays: the LAST corner of
        a bay is often the one that closes it onto its own first, and a row is
        clicked bay by bay so the neighbour's corners are already saved.
        """
        best: tuple[float, tuple[float, float]] | None = None
        for existing in [
            *self._pending,
            *(c for bay in self._bays for c in bay.corners),
        ]:
            gap = math.hypot(corner[0] - existing[0], corner[1] - existing[1])
            if gap <= LABEL_SNAP_M and (best is None or gap < best[0]):
                best = (gap, existing)
        return (corner, False) if best is None else (best[1], True)

    def _nearest_within(
        self, centre: tuple[float, float], radius_m: float
    ) -> int | None:
        best: tuple[float, int] | None = None
        for index, existing in enumerate(self._bays):
            other = existing.centre
            gap = math.hypot(centre[0] - other[0], centre[1] - other[1])
            if gap <= radius_m and (best is None or gap < best[0]):
                best = (gap, index)
        return None if best is None else best[1]

    def set_complete(self, complete: bool) -> None:
        self._complete = bool(complete)
        self._save()

    def cancel_pending(self) -> bool:
        """Abandon a half-clicked quad. True if there was one."""
        if not self._pending:
            return False
        self._pending = []
        self._snapped = 0
        return True

    def undo(self) -> bool:
        """Drop the corner in progress, or failing that the last whole bay."""
        if self.cancel_pending():
            return True
        if not self._bays:
            return False
        self._bays.pop()
        self._save()
        return True

    def _save(self) -> None:
        payload = {
            "bays": [bay.as_dict() for bay in self._bays],
            # Absent means False, so every session recorded before this
            # existed reads as partially labelled -- which it was.
            "complete": self._complete,
        }
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.path)
