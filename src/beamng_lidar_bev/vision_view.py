"""
The Vision-mode camera wall: every camera of the rig, live, in one grid.

Qt-only and worker-agnostic on purpose -- it consumes `VisionFrame` the way
`BevWidget` consumes `BevFrame` and never touches BeamNGpy. The grid geometry
is a pure function (`grid_dimensions`) so the layout arithmetic is testable
without a QApplication, which keeps the whole offline suite Qt-free.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPaintEvent
from PyQt6.QtWidgets import QWidget

from .models import VisionFrame

_BACKGROUND = QColor("#0d0f11")
_CELL_BACKGROUND = QColor("#14171a")
_BORDER = QColor("#30353a")
_LABEL_BG = QColor(13, 15, 17, 190)
_LABEL_FG = QColor("#c7cdd2")
_PLACEHOLDER_FG = QColor("#5c646b")
_CELL_GAP = 6.0
_LABEL_HEIGHT = 18.0


def grid_dimensions(
    count: int, area_w: float, area_h: float, cell_aspect: float = 4.0 / 3.0
) -> tuple[int, int]:
    """
    Rows and columns that give `count` cells of one aspect the most area.

    Tries every row count and keeps the arrangement whose uniformly scaled
    cell is largest -- so a wide window lays 8 cameras out 2x4 and a tall one
    4x2, with no hand-tuned breakpoints. Always satisfies rows * cols >= count.
    """
    if count <= 0:
        return (0, 0)
    if area_w <= 0.0 or area_h <= 0.0:
        return (1, count)
    best = (1, count)
    best_width = -1.0
    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)
        cell_width = min(area_w / cols, (area_h / rows) * cell_aspect)
        if cell_width > best_width:
            best_width = cell_width
            best = (rows, cols)
    return best


class VisionView(QWidget):
    """Paints the most recent `VisionFrame` as a labelled camera grid."""

    def __init__(self) -> None:
        super().__init__()
        self._frame: VisionFrame | None = None
        # QImage wraps the numpy buffer without copying, so the arrays they
        # view must outlive them: the frame reference above is what keeps
        # every buffer alive for exactly as long as its QImage.
        self._images: list[tuple[str, QImage]] = []
        self.setAutoFillBackground(False)

    def set_frame(self, frame: VisionFrame) -> None:
        self._frame = frame
        self._images = []
        for camera in frame.images:
            height, width, _ = camera.rgba.shape
            image = QImage(
                camera.rgba.data,
                width,
                height,
                width * 4,
                QImage.Format.Format_RGBA8888,
            )
            self._images.append((camera.name, image))
        self.update()

    def clear(self) -> None:
        self._frame = None
        self._images = []
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self._images:
            painter.setPen(_PLACEHOLDER_FG)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "VISION MODE\nAttach to a vehicle to stream the camera rig",
            )
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

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        font = painter.font()
        font.setPointSizeF(8.0)
        font.setBold(True)
        painter.setFont(font)
        for index, (name, image) in enumerate(self._images):
            row, col = divmod(index, cols)
            cell = QRectF(
                _CELL_GAP + col * (cell_w + _CELL_GAP),
                _CELL_GAP + row * (cell_h + _CELL_GAP),
                cell_w,
                cell_h,
            )
            painter.fillRect(cell, _CELL_BACKGROUND)
            # Fit the image inside its cell, preserving aspect.
            scale = min(cell.width() / image.width(), cell.height() / image.height())
            draw_w = image.width() * scale
            draw_h = image.height() * scale
            target = QRectF(
                cell.x() + (cell.width() - draw_w) / 2.0,
                cell.y() + (cell.height() - draw_h) / 2.0,
                draw_w,
                draw_h,
            )
            painter.drawImage(target, image)
            painter.setPen(_BORDER)
            painter.drawRect(cell)
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
        painter.end()
