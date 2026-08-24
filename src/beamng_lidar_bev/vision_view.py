"""
The camera wall: every camera of the rig, live, in one grid.

Qt-only and worker-agnostic on purpose -- it consumes `VisionFrame` the way
`BevWidget` consumes `BevFrame` and never touches BeamNGpy. The grid geometry
is a pure function (`grid_dimensions`) so the layout arithmetic is testable
without a QApplication, which keeps the whole offline suite Qt-free.

Clicking a tile focuses that camera full-frame; clicking again returns to the
grid. Focus is tracked by camera NAME, not by tile index, so it survives
frames arriving in a different order and dissolves harmlessly (back to the
grid) if the named camera stops appearing -- a rig swap, or a mode switch.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QWidget

from .models import VisionFrame

_BACKGROUND = QColor("#0d0f11")
_CELL_BACKGROUND = QColor("#14171a")
_BORDER = QColor("#30353a")
_LABEL_BG = QColor(13, 15, 17, 190)
_LABEL_FG = QColor("#c7cdd2")
_PLACEHOLDER_FG = QColor("#5c646b")
_HINT_FG = QColor("#6d757c")
_CELL_GAP = 6.0
_LABEL_HEIGHT = 18.0
# RGB**X**, not RGBA: the camera buffer's fourth byte is NOT opacity.
#
# Measured on BeamNG 0.39.4, a single 1280x960 frame: alpha ranges 40..255 with
# only 50.75% of pixels at 255, and what is in there tracks the scene's
# materials. Declared as Format_RGBA8888, Qt dutifully composited every pixel
# against the dark tile background, so wherever that channel dipped the tile
# went dark -- which reads as heavy black speckle on textured surfaces, dark
# outlines around buildings, and a clean sky (where alpha happens to be 255).
# Reported live as "so much noise in the camera compared to BeamNG itself".
#
# It stayed hidden through a long investigation because every probe saved
# `rgba[..., :3]` and threw the channel away, so the captured PNGs were clean
# while the app was not -- exposure, anti-aliasing, SSAO and tile downscaling
# were all measured and none of them was this.
#
# Format_RGBX8888 has the identical byte layout and ignores the fourth byte.
_IMAGE_FORMAT = QImage.Format.Format_RGBX8888


def grid_dimensions(
    count: int, area_w: float, area_h: float, cell_aspect: float = 4.0 / 3.0
) -> tuple[int, int]:
    """
    Rows and columns that give `count` cells of one aspect the most area.

    Tries every row count and keeps the arrangement whose uniformly scaled
    cell is largest -- so a wide window lays 8 cameras out 2x4 and a tall one
    4x2, with no hand-tuned breakpoints. Always satisfies rows * cols >= count.

    A PAIR is the one case where the spatial reading outranks the area: left
    and right cameras belong side by side, and stacking them puts the left one
    above the right one, which reads as nothing at all. That override applies
    only while the pane is at least as wide as it is tall -- forced
    unconditionally it costs real size on a portrait pane (measured on a
    700x1000 pane with 1280x960 frames: 341x256 drawn per tile against
    688x516 for the stacked answer, 3.7x the pixels).
    """
    if count <= 0:
        return (0, 0)
    if area_w <= 0.0 or area_h <= 0.0:
        return (1, count)
    if count == 2 and area_w >= area_h:
        return (1, 2)
    best = (1, count)
    best_width = -1.0
    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)
        cell_width = min(area_w / cols, (area_h / rows) * cell_aspect)
        if cell_width > best_width:
            best_width = cell_width
            best = (rows, cols)
    return best


def wants_prescale(
    source_w: int, source_h: int, target_w: float, target_h: float
) -> bool:
    """
    Should this image be resampled before it is drawn?

    `QPainter.drawImage` under `SmoothPixmapTransform` is BILINEAR, which reads
    a 2x2 neighbourhood however far the image is being shrunk. That is correct
    for magnification and wrong for MINIFICATION: shrinking 1280x960 into a
    400x300 tile steps over roughly 3 source pixels per output pixel, so most
    of them are never sampled and high-frequency texture -- asphalt, gravel,
    foliage, clapboard siding -- aliases into per-pixel speckle. Reported live
    as "so much noise in the camera compared to BeamNG itself", and the giveaway
    was that it ARRIVED when the rig went from 640x480 (upscaled into the pane,
    so merely soft) to 1280x960 (downscaled into it).

    `QImage.scaled(..., SmoothTransformation)` area-averages instead, so the
    detail lands as detail. It costs ~1 ms per tile, which is why it is done
    only when shrinking and only on a size or frame change -- see the cache in
    `VisionView`.
    """
    if source_w <= 0 or source_h <= 0 or target_w <= 0.0 or target_h <= 0.0:
        return False
    return target_w < float(source_w) or target_h < float(source_h)


def toggle_focus(current: str | None, clicked: str | None) -> str | None:
    """
    The focus state machine, pure so it can be pinned offline.

    Any click while focused returns to the grid -- including a click that
    would have landed on another tile, because in the focused view there are
    no other tiles to land on. From the grid, clicking a tile focuses it and
    clicking the gap between tiles does nothing.
    """
    if current is not None:
        return None
    return clicked


class VisionView(QWidget):
    """Paints the most recent `VisionFrame` as a labelled camera grid."""

    def __init__(self) -> None:
        super().__init__()
        self._frame: VisionFrame | None = None
        # QImage wraps the numpy buffer without copying, so the arrays they
        # view must outlive them: the frame reference above is what keeps
        # every buffer alive for exactly as long as its QImage.
        self._images: list[tuple[str, QImage]] = []
        # Which camera fills the view, by name; None means the grid. The tile
        # rectangles are recorded at paint time so the click hit-test agrees
        # with what is actually on screen.
        self._focused_name: str | None = None
        self._cell_rects: list[tuple[str, QRectF]] = []
        # Smooth-scaled copies, keyed by camera name and the size they were
        # made for. Resampling is ~1 ms a tile, far too much to repeat on every
        # paint for eight cameras, and completely unnecessary: it only changes
        # when a new frame arrives or the pane is resized.
        self._scaled: dict[str, tuple[int, int, QImage]] = {}
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_frame(self, frame: VisionFrame) -> None:
        self._frame = frame
        self._images = []
        self._scaled.clear()
        for camera in frame.images:
            height, width, _ = camera.rgba.shape
            image = QImage(
                camera.rgba.data,
                width,
                height,
                width * 4,
                _IMAGE_FORMAT,
            )
            self._images.append((camera.name, image))
        self.update()

    def clear(self) -> None:
        self._frame = None
        self._images = []
        self._focused_name = None
        self._cell_rects = []
        self._scaled.clear()
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        clicked: str | None = None
        position = event.position()
        for name, rect in self._cell_rects:
            if rect.contains(position):
                clicked = name
                break
        self._focused_name = toggle_focus(self._focused_name, clicked)
        self.update()
        event.accept()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        self._cell_rects = []
        if not self._images:
            painter.setPen(_PLACEHOLDER_FG)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "CAMERA VIEW\nAttach to a vehicle to stream the camera rig",
            )
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        font = painter.font()
        font.setPointSizeF(8.0)
        font.setBold(True)
        painter.setFont(font)

        focused = self._focused_image()
        if focused is not None:
            self._paint_focused(painter, *focused)
            painter.end()
            return

        area_w = float(self.width())
        area_h = float(self.height())
        first = self._images[0][1]
        cell_aspect = (
            first.width() / first.height() if first.height() else 4.0 / 3.0
        )
        rows, cols = grid_dimensions(
            len(self._images), area_w, area_h, cell_aspect
        )
        cell_w = (area_w - (cols + 1) * _CELL_GAP) / cols
        cell_h = (area_h - (rows + 1) * _CELL_GAP) / rows
        for index, (name, image) in enumerate(self._images):
            row, col = divmod(index, cols)
            cell = QRectF(
                _CELL_GAP + col * (cell_w + _CELL_GAP),
                _CELL_GAP + row * (cell_h + _CELL_GAP),
                cell_w,
                cell_h,
            )
            self._cell_rects.append((name, cell))
            painter.fillRect(cell, _CELL_BACKGROUND)
            target = self._fit(image, cell)
            self._draw(painter, name, image, target)
            painter.setPen(_BORDER)
            painter.drawRect(cell)
            self._paint_label(painter, target, name)
        painter.end()

    def _focused_image(self) -> tuple[str, QImage] | None:
        """The focused camera's image, or None when the grid should draw --
        including when the focused name stopped arriving (rig change)."""
        if self._focused_name is None:
            return None
        for name, image in self._images:
            if name == self._focused_name:
                return name, image
        return None

    def _paint_focused(
        self, painter: QPainter, name: str, image: QImage
    ) -> None:
        area = QRectF(
            _CELL_GAP,
            _CELL_GAP,
            self.width() - 2.0 * _CELL_GAP,
            self.height() - 2.0 * _CELL_GAP,
        )
        painter.fillRect(area, _CELL_BACKGROUND)
        target = self._fit(image, area)
        self._draw(painter, name, image, target)
        painter.setPen(_BORDER)
        painter.drawRect(area)
        self._paint_label(painter, target, name)
        painter.setPen(_HINT_FG)
        painter.drawText(
            area.adjusted(0.0, 0.0, -8.0, -4.0),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            "click to return to the grid",
        )

    def _draw(
        self, painter: QPainter, name: str, image: QImage, target: QRectF
    ) -> None:
        """Draw `image` into `target`, resampling first when shrinking."""
        width = max(1, int(round(target.width())))
        height = max(1, int(round(target.height())))
        if not wants_prescale(image.width(), image.height(), width, height):
            painter.drawImage(target, image)
            return
        cached = self._scaled.get(name)
        if cached is None or cached[0] != width or cached[1] != height:
            scaled = image.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled[name] = (width, height, scaled)
        else:
            scaled = cached[2]
        painter.drawImage(target, scaled)

    @staticmethod
    def _fit(image: QImage, box: QRectF) -> QRectF:
        """The largest aspect-preserving placement of `image` inside `box`."""
        if image.width() <= 0 or image.height() <= 0:
            return box
        scale = min(box.width() / image.width(), box.height() / image.height())
        draw_w = image.width() * scale
        draw_h = image.height() * scale
        return QRectF(
            box.x() + (box.width() - draw_w) / 2.0,
            box.y() + (box.height() - draw_h) / 2.0,
            draw_w,
            draw_h,
        )

    @staticmethod
    def _paint_label(painter: QPainter, target: QRectF, name: str) -> None:
        label = QRectF(
            target.x(),
            target.bottom() - _LABEL_HEIGHT,
            target.width(),
            _LABEL_HEIGHT,
        )
        painter.fillRect(label, _LABEL_BG)
        painter.setPen(_LABEL_FG)
        painter.drawText(
            label.adjusted(6.0, 0.0, 0.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            name.upper().replace("_", " "),
        )
