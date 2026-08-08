from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .aeb import corridor_cross_section, predicted_corridor
from .config import DISPLAY_RADIUS_M
from .models import BRAKING, STANDBY, AebState, BevFrame, DrivingPlan, VehicleGeometry
from .planner import arc_polyline, path_polyline
from .raster import rasterize_points

_MODE_COLOURS = {
    "DRIVING": "#53d1b2",
    "BLOCKED": "#ffd36a",
    "REVERSING": "#74bfff",
    "STUCK": "#ff7b82",
}
# Violet is the one hue nothing else in this view uses -- returns are grey, the
# obstacle set red, the ego and plan teal, keep-right amber and the nav hint
# blue -- so the watched corridor reads as an overlay rather than as more data.
# A neutral grey at low alpha was tried first and simply vanished into the road
# points. Braking switches to a hot red because at that moment it should not be
# subtle.
_AEB_ARMED_RGB = (178, 140, 255)
# Fixed, and no longer the LiDAR cull radius: the cull now reaches far past the
# plot so the long-range front unit can feed AEB, and a ring drawn at 190 m
# would simply be off-canvas.
_RING_RADII_M = (25.0, 50.0, 75.0, 100.0)
_AEB_BRAKING_RGB = (255, 96, 86)


def bev_to_screen(rect: QRectF, right_m: float, forward_m: float) -> QPointF:
    """Map BEV (right, forward) metres onto the plot. +forward is up."""
    centre = rect.center()
    scale = rect.width() / (2.0 * DISPLAY_RADIUS_M)
    return QPointF(centre.x() + right_m * scale, centre.y() - forward_m * scale)


def _aeb_to_screen(rect: QRectF, aeb: AebState, x: float, y: float) -> QPointF:
    """
    Map a point from one AEB system's own travel frame onto the plot.

    The rear system reasons in a 180-degree-rotated frame -- that is what lets
    it share every arc helper unchanged -- so un-rotating here is the whole of
    what drawing it backwards takes. See aeb.mirrored.
    """
    return (
        bev_to_screen(rect, -x, -y)
        if aeb.rearward
        else bev_to_screen(rect, x, y)
    )


class BevWidget(QWidget):
    """Fast, fixed-heading top-down LiDAR display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("bevWidget")
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame: Optional[BevFrame] = None
        self._point_image: Optional[QImage] = None
        self._image_side = 0

    def set_frame(self, frame: BevFrame) -> None:
        self._frame = frame
        self._point_image = None
        self.update()

    def clear(self) -> None:
        self._frame = None
        self._point_image = None
        self.update()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        self._point_image = None
        super().resizeEvent(event)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d0f11"))

        plot_rect = self._plot_rect()
        self._draw_grid(painter, plot_rect)

        if self._frame is not None:
            image = self._get_point_image(int(plot_rect.width()))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(plot_rect.topLeft(), image)
            if self._frame.plan is not None:
                self._draw_plan(painter, plot_rect, self._frame.plan)
            # After the plan, so the emergency corridor is never buried under
            # the candidate fan it overrides.
            for state in (self._frame.aeb, self._frame.rear_aeb):
                if state is not None:
                    self._draw_aeb(painter, plot_rect, state)
            self._draw_ego(
                painter,
                plot_rect,
                self._frame.vehicle_geometry,
            )
            if self._frame.plan is not None:
                self._draw_plan_label(painter, plot_rect, self._frame.plan)
            for state in (self._frame.aeb, self._frame.rear_aeb):
                if state is not None:
                    self._draw_aeb_label(painter, plot_rect, state)
        else:
            self._draw_waiting_state(painter, plot_rect)

        painter.end()

    def _plot_rect(self) -> QRectF:
        margin = 24.0
        side = max(1.0, min(self.width(), self.height()) - margin * 2.0)
        left = (self.width() - side) * 0.5
        top = (self.height() - side) * 0.5
        return QRectF(left, top, side, side)

    def _get_point_image(self, side: int) -> QImage:
        if self._point_image is not None and self._image_side == side:
            return self._point_image
        assert self._frame is not None

        pixels = rasterize_points(
            self._frame.road_points,
            self._frame.obstacle_points,
            side,
            side,
        )
        image = QImage(
            pixels.data,
            side,
            side,
            int(pixels.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        self._point_image = image.copy()
        self._image_side = side
        return self._point_image

    def _draw_grid(self, painter: QPainter, rect: QRectF) -> None:
        centre = rect.center()
        scale = rect.width() / (2.0 * DISPLAY_RADIUS_M)

        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor("#111417"))

        axis_pen = QPen(QColor(64, 70, 75, 150), 1.0)
        axis_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(rect.left(), centre.y()), QPointF(rect.right(), centre.y())
        )
        painter.drawLine(
            QPointF(centre.x(), rect.top()), QPointF(centre.x(), rect.bottom())
        )

        ring_pen = QPen(QColor(76, 83, 89, 180), 1.0)
        painter.setPen(ring_pen)
        label_font = QFont(self.font())
        label_font.setPointSize(8)
        painter.setFont(label_font)
        for radius_m in _RING_RADII_M:
            radius_px = radius_m * scale
            ring = QRectF(
                centre.x() - radius_px,
                centre.y() - radius_px,
                radius_px * 2.0,
                radius_px * 2.0,
            )
            painter.drawEllipse(ring)
            painter.setPen(QColor("#737b82"))
            painter.drawText(
                QPointF(centre.x() + 5.0, centre.y() - radius_px - 4.0),
                f"{int(radius_m)} m",
            )
            painter.setPen(ring_pen)

        painter.setPen(QPen(QColor("#2c3136"), 1.0))
        painter.drawRect(rect)
        painter.restore()

        direction_font = QFont(self.font())
        direction_font.setPointSize(8)
        direction_font.setBold(True)
        painter.setFont(direction_font)
        painter.setPen(QColor("#8d959c"))
        painter.drawText(
            QRectF(rect.left(), rect.top() - 20.0, rect.width(), 18.0),
            Qt.AlignmentFlag.AlignCenter,
            "FRONT",
        )
        painter.drawText(
            QRectF(rect.left() + 5.0, rect.top(), 42.0, rect.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "LEFT",
        )
        painter.drawText(
            QRectF(rect.right() - 47.0, rect.top(), 42.0, rect.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            "RIGHT",
        )

    def _draw_ego(
        self, painter: QPainter, rect: QRectF, geometry: VehicleGeometry
    ) -> None:
        centre = rect.center()
        scale = rect.width() / (2.0 * DISPLAY_RADIUS_M)

        def screen(right_m: float, forward_m: float) -> QPointF:
            return QPointF(
                centre.x() + right_m * scale,
                centre.y() - forward_m * scale,
            )

        left = -geometry.left_m
        right = geometry.right_m
        front = geometry.front_m
        rear = -geometry.rear_m
        nose_left = left * 0.68
        nose_right = right * 0.68
        shoulder_y = rear + (front - rear) * 0.72
        body = QPolygonF(
            [
                screen(left, rear),
                screen(right, rear),
                screen(right, shoulder_y),
                screen(nose_right, front),
                screen(nose_left, front),
                screen(left, shoulder_y),
            ]
        )

        painter.save()
        painter.setBrush(QColor("#e8ecef"))
        painter.setPen(QPen(QColor("#53d1b2"), 1.6))
        painter.drawPolygon(body)

        centre_line_pen = QPen(QColor("#252a2e"), 1.0)
        painter.setPen(centre_line_pen)
        painter.drawLine(screen(0.0, rear + 0.3), screen(0.0, front - 0.25))

        for mount in geometry.mounts.values():
            local_x, local_y, _ = mount.position_vehicle
            dir_x, dir_y, _ = mount.direction_vehicle
            origin = screen(-local_x, -local_y)
            direction_right = -dir_x
            direction_forward = -dir_y
            endpoint = screen(
                -local_x + direction_right * 1.6,
                -local_y + direction_forward * 1.6,
            )
            painter.setPen(QPen(QColor(83, 209, 178, 190), 1.0))
            painter.drawLine(origin, endpoint)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#53d1b2"))
            painter.drawEllipse(origin, 2.3, 2.3)
        painter.restore()

    def _draw_plan(
        self, painter: QPainter, rect: QRectF, plan: DrivingPlan
    ) -> None:
        """Candidate fan, chosen arc, free-distance tick and the nav hint."""
        arc = plan.arc
        accent = QColor(_MODE_COLOURS.get(plan.command.mode, "#53d1b2"))

        painter.save()
        painter.setClipRect(rect)

        def polyline(curvature: float, length: float) -> QPolygonF:
            points = arc_polyline(curvature, length, samples=28)
            return QPolygonF(
                [bev_to_screen(rect, float(x), float(y)) for x, y in points]
            )

        # The whole considered fan, so a bad decision is visibly a bad decision
        # rather than an unexplained swerve.
        faint = QColor(accent)
        faint.setAlpha(45)
        painter.setPen(QPen(faint, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for curvature, reach in zip(
            arc.candidate_curvatures, arc.candidate_free_distances
        ):
            painter.drawPolyline(polyline(float(curvature), float(reach)))

        if arc.keep_right_target_m is not None:
            target = bev_to_screen(
                rect, arc.keep_right_target_m, arc.lookahead_m
            )
            painter.setPen(QPen(QColor(255, 211, 106, 150), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target, 4.0, 4.0)

        # The composite path: hold the current curvature through the
        # transition, then bend to the target -- provably what was planned,
        # because it is sampled by the planner's own path_polyline.
        chosen_points = path_polyline(
            arc.curvature,
            arc.transition_distance_m,
            arc.next_curvature,
            max(arc.free_distance_m, 4.0),
            samples=40,
        )
        chosen = QPolygonF(
            [bev_to_screen(rect, float(x), float(y)) for x, y in chosen_points]
        )
        painter.setPen(QPen(accent, 2.2))
        painter.drawPolyline(chosen)
        if chosen.count():
            painter.setPen(QPen(accent, 1.0))
            painter.setBrush(accent)
            painter.drawEllipse(chosen.at(chosen.count() - 1), 3.0, 3.0)

        if arc.nav_heading_rad is not None:
            self._draw_nav_hint(painter, rect, arc.nav_heading_rad)
        painter.restore()

    @staticmethod
    def _draw_nav_hint(
        painter: QPainter, rect: QRectF, heading_rad: float
    ) -> None:
        """The turn the in-game destination asks for, drawn as a bearing spoke."""
        reach = DISPLAY_RADIUS_M * 0.82
        right = -math.sin(heading_rad) * reach
        forward = math.cos(heading_rad) * reach
        tip = bev_to_screen(rect, right, forward)
        base = bev_to_screen(rect, right * 0.86, forward * 0.86)

        painter.setPen(QPen(QColor(116, 191, 255, 190), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(base, tip)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(116, 191, 255, 210))
        painter.drawEllipse(tip, 3.5, 3.5)

    def _draw_aeb(
        self, painter: QPainter, rect: QRectF, aeb: AebState
    ) -> None:
        """
        The corridor AEB is watching, the impact chord, and the stop line.

        Drawn from the state's own curvature and half-width, so what is on
        screen is provably the corridor that was scanned rather than a
        redrawing of it -- the same reason the plan overlay samples the
        planner's own `path_polyline`.
        """
        if aeb.status == STANDBY:
            # Below the arming speed AEB is not watching anything, and drawing
            # a corridor there would claim otherwise.
            return
        braking = aeb.status == BRAKING
        red, green, blue = _AEB_BRAKING_RGB if braking else _AEB_ARMED_RGB

        def polygon(length_m: float) -> QPolygonF:
            return QPolygonF(
                [
                    _aeb_to_screen(rect, aeb, float(x), float(y))
                    for x, y in predicted_corridor(
                        aeb.curvature, length_m, aeb.corridor_half_width_m
                    )
                ]
            )

        painter.save()
        painter.setClipRect(rect)

        # The whole watched corridor, as an outline only. Its length is what
        # AEB is actually scanning, which shrinks with speed.
        outline = QPen(QColor(red, green, blue, 235 if braking else 165), 1.4)
        if not braking:
            outline.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(outline)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon(aeb.horizon_m))

        # The predicted path itself, which is the thing being visualised: the
        # corridor edges disappear into a dense cloud, the centreline does not.
        centre = QPen(QColor(red, green, blue, 130), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(centre)
        painter.drawPolyline(
            QPolygonF(
                [
                    _aeb_to_screen(rect, aeb, float(x), float(y))
                    for x, y in arc_polyline(aeb.curvature, aeb.horizon_m, 28)
                ]
            )
        )

        # The last point to brake. Drawn while armed rather than while braking,
        # because that is when it answers the question the driver actually has:
        # how close does something have to get before this thing acts?
        if not braking and aeb.brake_now_m < aeb.horizon_m:
            self._draw_chord(
                painter,
                rect,
                aeb,
                aeb.brake_now_m,
                QPen(QColor(red, green, blue, 120), 1.2, Qt.PenStyle.DashLine),
            )

        threat = aeb.threat_m
        if threat is not None:
            # Filled only as far as the blockage. Filling to the horizon reads
            # as "all 34 m of this is dangerous" when the point that matters is
            # the one the car is going to hit.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(red, green, blue, 66 if braking else 34))
            painter.drawPolygon(polygon(threat))
            self._draw_chord(
                painter, rect, aeb, threat, QPen(QColor(red, green, blue, 245), 2.6)
            )

        # Where the car has to be stopped by. Only while braking, where it is
        # the target rather than trivia.
        if braking:
            self._draw_chord(
                painter,
                rect,
                aeb,
                aeb.standoff_m,
                QPen(QColor(255, 211, 106, 200), 1.4, Qt.PenStyle.DotLine),
            )
        painter.restore()

    @staticmethod
    def _draw_chord(
        painter: QPainter,
        rect: QRectF,
        aeb: AebState,
        distance_m: float,
        pen: QPen,
    ) -> None:
        """A bar drawn across the corridor at one distance along the path."""
        ends = corridor_cross_section(
            aeb.curvature, distance_m, aeb.corridor_half_width_m
        )
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            _aeb_to_screen(rect, aeb, float(ends[0][0]), float(ends[0][1])),
            _aeb_to_screen(rect, aeb, float(ends[1][0]), float(ends[1][1])),
        )

    def _draw_aeb_label(
        self, painter: QPainter, rect: QRectF, aeb: AebState
    ) -> None:
        name = "REAR AEB" if aeb.rearward else "AEB"
        if aeb.status == STANDBY:
            text = f"{name}  ·  standby"
            colour = QColor(115, 123, 130)
        elif aeb.status == BRAKING:
            ttc = aeb.time_to_collision_s
            impact = f"  ·  {ttc:.1f} s" if math.isfinite(ttc) else ""
            text = f"{name} FULL BRAKE{impact}"
            colour = QColor(*_AEB_BRAKING_RGB)
        else:
            threat = aeb.threat_m
            # Distance still in hand beyond the last point to brake, which is
            # the number that says whether this is about to become an event.
            detail = (
                "clear"
                if threat is None
                else f"{threat:.0f} m, acts at {aeb.brake_now_m:.0f} m"
            )
            text = f"{name} armed  ·  {detail}"
            colour = QColor(*_AEB_ARMED_RGB)

        font = QFont(self.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(colour)
        top = rect.bottom() - 22.0 if aeb.rearward else rect.top() + 6.0
        painter.drawText(
            QRectF(rect.left() + 8.0, top, rect.width() - 16.0, 16.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_plan_label(
        self, painter: QPainter, rect: QRectF, plan: DrivingPlan
    ) -> None:
        font = QFont(self.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(_MODE_COLOURS.get(plan.command.mode, "#53d1b2")))
        painter.drawText(
            QRectF(rect.left() + 8.0, rect.top() + 6.0, rect.width() - 16.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"AUTO {plan.command.mode}  ·  {plan.command.reason}",
        )

    def _draw_waiting_state(self, painter: QPainter, rect: QRectF) -> None:
        centre = rect.center()
        car = QRectF(centre.x() - 7.0, centre.y() - 14.0, 14.0, 28.0)
        painter.setBrush(QColor("#292e32"))
        painter.setPen(QPen(QColor("#4b535a"), 1.0))
        painter.drawRoundedRect(car, 3.0, 3.0)
        painter.setPen(QColor("#737b82"))
        font = QFont(self.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            QRectF(rect.left(), centre.y() + 28.0, rect.width(), 24.0),
            Qt.AlignmentFlag.AlignCenter,
            "NO LIDAR DATA",
        )
