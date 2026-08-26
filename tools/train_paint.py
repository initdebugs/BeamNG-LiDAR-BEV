"""
Train the bay segmenter on the masks `build_dataset.py` produced.

Deliberately OUTSIDE the app and outside its virtualenv. PyTorch is three
gigabytes and slow to import, and it has no business in a process that has to
answer a 40 ms control tick; only the exported model comes back in.

    .venv-train\\Scripts\\python tools/train_paint.py
    .venv-train\\Scripts\\python tools/train_paint.py --epochs 40 --batch 8
    .venv-train\\Scripts\\python tools/train_paint.py --preview 8   # look at it

Writes `dataset/model/` -- the best checkpoint by validation IoU, an ONNX
export for the live path, and (with --preview) blended predictions to look at.

Five things are load-bearing, and the first two are the ones that make the
reported score mean anything:

- **255 is masked out of the loss, not treated as background.** A quarter of
  every mask is ground that may well be painted and was never labelled; see
  `build_dataset`. `ignore_index` does it in the loss AND it has to be done
  again by hand in the IoU, or validation quietly scores the model on pixels
  nobody knows the answer for.
- **Validation is a LOT the model has never seen**, which the dataset already
  arranged. Anything else measures how well a network memorises frames 0.5 s
  apart.
- **Masks resize with NEAREST, images with bilinear.** A bilinear mask
  interpolates 255 against 1 and invents classes that do not exist -- the
  ignore label is a code, not a quantity.
- **Brightness and contrast jitter matter more than usual here.** The cameras
  auto-expose (`HYBRID_CAMERA_AUTO_EXPOSURE`), so the same bay is a different
  brightness depending on how much sky or dark bodywork is in frame; a model
  that has not seen that range will fail on the frame after the one it saw.
- **The positive class is ~5% of pixels**, so the loss is class-weighted.
  Without it the network reaches a fine-looking accuracy by predicting
  background everywhere, and the IoU on the class that matters is zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and torch's ONNX exporter prints a tick
# emoji when it succeeds -- so the export dies with UnicodeEncodeError AFTER
# doing the work, which reads as an export failure and is not one.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as missing:  # pragma: no cover - setup guidance
    raise SystemExit(
        "\n".join(
            (
                f"{missing.name} is not installed in this interpreter.",
                "",
                "  py -3.11 -m venv .venv-train",
                "  .venv-train\\Scripts\\python -m pip install \\",
                "      --index-url https://download.pytorch.org/whl/cu128 \\",
                "      torch torchvision",
                "  .venv-train\\Scripts\\python -m pip install pillow numpy",
                "",
                "cu128 is what an RTX 50-series (Blackwell, sm_120) needs; an",
                "older CUDA wheel imports fine and then fails at the first",
                "kernel.",
            )
        )
    ) from None

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUT = DATASET / "model"

BACKGROUND, BAY, DIVIDER, IGNORE = 0, 1, 2, 255
_REGION_NAMES = ("background", "bay", "divider")
_LINE_NAMES = ("not paint", "paint")

# --target lines is the measured recommendation, and the reason is geometric.
# A bay's INTERIOR is featureless tarmac, identical to the aisle in front of
# it; the only thing that distinguishes them is the paint at the boundary. So
# a region target asks the network to infer an area from its edges, which it
# does raggedly -- measured on a complete-label validation session, precision
# 46.3% with predictions spilling across the aisle while the DIVIDER bands
# visibly followed the real lines. Asking for the lines asks for the thing that
# is actually visible, and `parking.find_bays` over `MarkingMemory` already
# turns lines into bays, is well tested, and is better at it than a U-Net.
#
# The bay interior becomes background under this target, which is simply true:
# it is not paint.
# Half of the captured 1280x960. A bay at 15 m is ~100 px wide at full size and
# ~50 here, which is ample for a REGION target; the divider band is the thing
# that would suffer first, and it is 0.30 m wide by construction for exactly
# that reason. Halving again saves little and starts to cost the far field.
INPUT_SIZE = (640, 480)


def build_model(classes: int = 3, base: int = 32):
    """A small U-Net, written out rather than imported.

    Segmentation-model packages bring their own pretrained weights and a
    dependency tree; this is eighty lines, trains in minutes on 2,000 images,
    and exports to ONNX without argument -- which is what the live path needs.
    """
    def block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = [base, base * 2, base * 4, base * 8]
            self.down = nn.ModuleList()
            channels = 3
            for width in widths:
                self.down.append(block(channels, width))
                channels = width
            self.pool = nn.MaxPool2d(2)
            self.bottom = block(widths[-1], widths[-1] * 2)
            self.up = nn.ModuleList()
            self.merge = nn.ModuleList()
            channels = widths[-1] * 2
            for width in reversed(widths):
                self.up.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
                self.merge.append(block(width * 2, width))
                channels = width
            self.head = nn.Conv2d(widths[0], classes, 1)

        def forward(self, x):
            skips = []
            for stage in self.down:
                x = stage(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottom(x)
            for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
                x = up(x)
                x = merge(torch.cat([skip, x], dim=1))
            return self.head(x)

    return UNet()


class BayDataset(Dataset):
    """
    Module level, and it has to be: Windows starts DataLoader workers with
    `spawn`, which PICKLES the dataset across to them, and a class defined
    inside a factory function has no importable name to pickle by. Nested, this
    dies with "Can't pickle local object" the moment `--workers` is nonzero.
    """

    def __init__(
        self, rows: list[dict], train: bool, target: str = "region"
    ) -> None:
        self.rows = rows
        self.train = train
        self.target = target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(ROOT / row["image"]).convert("RGB")
        mask = Image.open(DATASET / row["mask"])
        image = image.resize(INPUT_SIZE, Image.BILINEAR)
        # NEAREST, always: 255 is a CODE. Interpolating it against 1 invents
        # classes that do not exist and silently poisons the loss.
        mask = mask.resize(INPUT_SIZE, Image.NEAREST)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        labels = np.asarray(mask, dtype=np.int64)
        if self.target == "lines":
            # Paint or not paint. IGNORE survives untouched -- it is a code,
            # and collapsing it here would silently supervise the unknown.
            labels = np.where(
                labels == IGNORE, IGNORE, (labels == DIVIDER).astype(np.int64)
            )

        if self.train:
            if np.random.rand() < 0.5:
                pixels = pixels[:, ::-1].copy()
                labels = labels[:, ::-1].copy()
            # The rig auto-exposes, so the same bay arrives at a different
            # brightness from one frame to the next. This is the variation the
            # model will actually meet.
            pixels = np.clip(
                (pixels - 0.5) * np.random.uniform(0.75, 1.30)
                + 0.5
                + np.random.uniform(-0.18, 0.18),
                0.0,
                1.0,
            )
        return (
            torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            torch.from_numpy(labels),
        )


def class_weights(
    rows: list[dict], target: str, power: float = 0.5, sampled: int = 200
):
    """Inverse-frequency weights, so a small class is not simply ignored."""
    values = _values(target)
    counts = np.zeros(len(values), dtype=np.int64)
    step = max(1, len(rows) // sampled)
    for row in rows[::step]:
        labels = np.asarray(Image.open(DATASET / row["mask"]))
        if target == "lines":
            labels = np.where(
                labels == IGNORE, IGNORE, (labels == DIVIDER).astype(np.int64)
            )
        for value in values:
            counts[value] += int((labels == value).sum())
    frequency = counts / max(counts.sum(), 1)
    # The exponent is the recall/precision dial. Inverse frequency (1.0) is the
    # textbook answer and is far too strong here: at 0.5 the paint class already
    # gets 8x the background's weight, and both trained models over-predicted --
    # region P 0.46 / R 0.69, lines P 0.16 / R 0.82. Lower it to buy precision.
    weights = np.power(np.maximum(frequency, 1e-6), -power)
    return weights / weights.mean(), counts


def _values(target: str) -> tuple[int, ...]:
    return (0, 1) if target == "lines" else (BACKGROUND, BAY, DIVIDER)


def _names(target: str) -> tuple[str, ...]:
    return _LINE_NAMES if target == "lines" else _REGION_NAMES


def evaluate(model, loader, device, target: str = "region") -> dict:
    """Per-class IoU, with 255 excluded BY HAND.

    `ignore_index` covers the loss and nothing else, so an IoU computed over
    the raw tensors scores the model on a quarter of the pixels nobody knows
    the answer for -- and scores it well, because the model has learnt to call
    them background.
    """
    model.eval()
    values = _values(target)
    names = _names(target)
    # Class 1 under BOTH targets: 'bay' for a region, 'paint' for lines.
    # `len(values) - 1` looked equivalent and is not -- it selects DIVIDER for
    # the region target, so the "best" checkpoint was being chosen on the
    # divider IoU while the log column said bay. Caught by a run reporting
    # "best bay IoU 0.193" over epochs that had printed 0.341.
    positive = 1
    intersection = np.zeros(len(values), dtype=np.int64)
    union = np.zeros(len(values), dtype=np.int64)
    hits = misses = spurious = 0
    with torch.no_grad():
        for pixels, labels in loader:
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predicted = model(pixels).argmax(dim=1)
            known = labels != IGNORE
            for value in values:
                hit = (predicted == value) & known
                truth = (labels == value) & known
                intersection[value] += int((hit & truth).sum())
                union[value] += int((hit | truth).sum())
            # Precision and recall on the POSITIVE class, reported beside the
            # IoU because IoU alone cannot say which way a model is wrong --
            # and over- versus under-prediction want opposite fixes.
            found = (predicted >= 1) & known
            real = (labels >= 1) & known
            hits += int((found & real).sum())
            misses += int((~found & real).sum())
            spurious += int((found & ~real).sum())
    iou = intersection / np.maximum(union, 1)
    return {
        "iou": {name: float(iou[i]) for i, name in enumerate(names)},
        "mean_iou": float(iou.mean()),
        # Named for the number the checkpoint is chosen on, which is the
        # POSITIVE class under whichever target is running.
        "bay_iou": float(iou[positive]),
        "precision": hits / max(hits + spurious, 1),
        "recall": hits / max(hits + misses, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--weight-power",
        type=float,
        default=0.5,
        help="0 = no class weighting, 0.5 = inverse sqrt (default), "
        "1 = inverse frequency. Higher buys RECALL with precision.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=("region", "lines"),
        default="region",
        help="what the positive class IS -- see the note above",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="skip training; re-export the saved best.pt to ONNX",
    )
    args = parser.parse_args()

    index = DATASET / "index.jsonl"
    if not index.is_file():
        print("run tools/build_dataset.py first", file=sys.stderr)
        return 2
    rows = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    val_lots = sorted({row["lot"] for row in val_rows})
    print(
        f"{len(train_rows)} train / {len(val_rows)} val images; "
        f"validation is lot(s) {val_lots}, unseen in training"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.export_only:
        model = build_model(len(_values(args.target))).to(device)
        model.load_state_dict(
            torch.load(OUT / f"best_{args.target}.pt")
        )
        _export_onnx(model, device, args.target)
        print(f"exported {OUT}/paint_{args.target}.onnx")
        return 0

    if device.type == "cpu":
        print(
            "WARNING: no CUDA device -- this will take hours rather than "
            "minutes.",
            file=sys.stderr,
        )
    else:
        print(f"device {torch.cuda.get_device_name(0)}")

    weights, counts = class_weights(
        train_rows, args.target, args.weight_power
    )
    print(
        "class pixels "
        + ", ".join(
            f"{name} {100 * c / max(counts.sum(), 1):.1f}%"
            for name, c in zip(_names(args.target), counts)
        )
        + " | weights "
        + ", ".join(f"{w:.2f}" for w in weights)
    )

    loaders = {
        "train": DataLoader(
            BayDataset(train_rows, True, args.target),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        ),
        "val": DataLoader(
            BayDataset(val_rows, False, args.target),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
    }

    model = build_model(len(_values(args.target))).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        # The whole point: a quarter of every mask contributes nothing.
        ignore_index=IGNORE,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    # Per TARGET, because a shared best.pt means the second experiment destroys
    # the first -- which is exactly what happened to the region model when the
    # lines run followed it, leaving nothing to compare against.
    checkpoint = OUT / f"best_{args.target}.pt"
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        for pixels, labels in loaders["train"]:
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(pixels), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            # detach: float() on a grad-tracking tensor keeps the graph
            # alive for the whole epoch, which is a leak as well as a warning.
            running += float(loss.detach())
        schedule.step()

        scores = evaluate(model, loaders["val"], device, args.target)
        elapsed = time.perf_counter() - started
        mean_loss = running / len(loaders["train"])
        history.append({"epoch": epoch, "loss": mean_loss, **scores})
        marker = ""
        if scores["bay_iou"] > best:
            best = scores["bay_iou"]
            torch.save(model.state_dict(), checkpoint)
            marker = "  <-- best"
        print(
            f"epoch {epoch:3d}  loss {mean_loss:.4f}  "
            + "  ".join(
                f"{name} IoU {scores['iou'][name]:.3f}"
                for name in _names(args.target)
            )
            + f"  P {scores['precision']:.2f} R {scores['recall']:.2f}"
            + f"  [{elapsed:.0f}s]{marker}"
        )

    (OUT / f"history_{args.target}.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    model.load_state_dict(torch.load(checkpoint))
    # NEVER fatal. The checkpoint is the hours of work; the export is a
    # convenience for the live path, and torch 2.11's exporter needs a separate
    # `onnxscript` package. Letting a missing optional dependency end the run
    # after the training finished would throw away the whole thing.
    try:
        _export_onnx(model, device, args.target)
    except Exception as failure:
        print(
            f"\nONNX export skipped ({failure}). best.pt is saved and intact; "
            "for the live path run\n  .venv-train\\Scripts\\python -m pip "
            "install onnxscript onnx\nand re-run with --export-only.",
            file=sys.stderr,
        )
    print(f"\nbest bay IoU {best:.3f}; model in {OUT}")
    if args.preview:
        _write_previews(model, val_rows, device, args.preview, args.target)
    print(
        "\nIoU on the BAY class is the number that matters. The mean is "
        "flattered by\nbackground, which is most of every frame and trivially "
        "easy."
    )
    return 0


def _export_onnx(model, device, target: str = "region") -> None:
    """For the live path: ONNX Runtime with the DirectML or CUDA provider is
    what runs this beside a simulator without a second CUDA install."""
    model.eval()
    dummy = torch.zeros(
        1, 3, INPUT_SIZE[1], INPUT_SIZE[0], device=device, dtype=torch.float32
    )
    torch.onnx.export(
        model,
        dummy,
        str(OUT / f"paint_{target}.onnx"),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def _write_previews(
    model, rows: list[dict], device, count: int, target: str = "region"
) -> None:
    """Predictions blended onto the frames, because an IoU cannot show you
    WHERE it is wrong."""
    out_dir = OUT / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    step = max(1, len(rows) // count)
    for row in rows[::step][:count]:
        image = Image.open(ROOT / row["image"]).convert("RGB")
        small = image.resize(INPUT_SIZE, Image.BILINEAR)
        pixels = np.asarray(small, dtype=np.float32) / 255.0
        with torch.no_grad():
            logits = model(
                torch.from_numpy(pixels.transpose(2, 0, 1))[None].to(device)
            )
        predicted = logits.argmax(dim=1)[0].cpu().numpy()
        truth = np.asarray(
            Image.open(DATASET / row["mask"]).resize(INPUT_SIZE, Image.NEAREST)
        )
        base = np.asarray(small, dtype=float)
        tint = np.zeros_like(base)
        if target == "lines":
            # Class 1 IS paint under this target; there is no class 2.
            tint[predicted == 1] = (255, 90, 90)
        else:
            tint[predicted == BAY] = (80, 200, 255)
            tint[predicted == DIVIDER] = (255, 90, 90)
        blended = np.where(
            (predicted > 0)[:, :, None], 0.5 * base + 0.5 * tint, base
        )
        # The truth's outline in green, so a miss is visible as a gap rather
        # than having to be remembered from another window.
        wanted = DIVIDER if target == "lines" else BAY
        edge = (truth == wanted) & ~np.roll(truth == wanted, 1, axis=1)
        blended[edge] = (120, 255, 120)
        Image.fromarray(blended.astype(np.uint8)).save(
            out_dir / Path(row["mask"]).name
        )
    print(f"{count} predictions in {out_dir} -- LOOK AT THEM")


if __name__ == "__main__":
    sys.exit(main())
