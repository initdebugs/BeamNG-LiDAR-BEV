from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import (
    Q_ARG,
    QAbstractListModel,
    QByteArray,
    QEvent,
    QMetaObject,
    QModelIndex,
    QObject,
    QPointF,
    Qt,
    QUrl,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QCloseEvent, QColor, QMouseEvent, QVector3D, QWheelEvent
from PyQt6.QtQml import QQmlEngine
from PyQt6.QtQuick3D import QQuick3DGeometry
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .models import ParkingSlot, WorldActor, WorldFrame
from .parking import slot_contains
from .world_scene import apply_view_orbit

_POSITION_BYTES = 12
_COLOUR_BYTES = 16
_VERTEX_STRIDE = _POSITION_BYTES + _COLOUR_BYTES


def interleave(vertices: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    Pack positions and linear RGBA colours into one contiguous float32 buffer.

    The layout is what `SceneGeometry` declares to Qt: xyz then rgba, 28 bytes a
    vertex. Kept a free function so the packing can be pinned offline without a
    QApplication or a graphics device.
    """
    positions = np.asarray(vertices, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 vertex array, got {positions.shape}")
    rgba = np.asarray(colors, dtype=np.float32)
    if rgba.ndim != 2 or rgba.shape[1] != 4:
        raise ValueError(f"Expected an Nx4 colour array, got {rgba.shape}")
    if len(rgba) != len(positions):
        raise ValueError(
            f"{len(positions)} vertices against {len(rgba)} colours"
        )
    return np.ascontiguousarray(
        np.concatenate((positions, rgba), axis=1), dtype=np.float32
    )


class SceneGeometry(QQuick3DGeometry):
    """
    A mesh handed to Qt Quick 3D as raw bytes rather than as a QML value list.

    This replaces `ProceduralMesh`, and it is what lets the scene be detailed at
    all. `ProceduralMesh` takes `list<vector3d>`, so feeding it meant building
    one QVector3D per vertex in a Python loop on the GUI thread every frame --
    the cost of the view scaled with its detail, and WORLD_CELL_SIZE_M was set
    by what that loop could carry rather than by what the sensors resolve.
    `setVertexData` takes the numpy buffer verbatim, so the Python cost is O(1)
    in vertex count and the grid is free to follow the data.

    Attributes are declared ONCE, in the constructor: `addAttribute` appends,
    so re-declaring them per frame would grow the attribute list without bound.
    """

    def __init__(self, primitive: QQuick3DGeometry.PrimitiveType) -> None:
        # No Qt parent: QQuick3DGeometry only accepts a QQuick3DObject as one,
        # and the bridge is a plain QObject. The bridge holds a Python
        # reference instead, and ownership is pinned to C++ so the QML engine
        # cannot decide to collect a geometry that is still bound to a Model.
        super().__init__()
        QQmlEngine.setObjectOwnership(self, QQmlEngine.ObjectOwnership.CppOwnership)
        attribute = QQuick3DGeometry.Attribute
        self.setStride(_VERTEX_STRIDE)
        self.setPrimitiveType(primitive)
        self.addAttribute(
            attribute.Semantic.PositionSemantic,
            0,
            attribute.ComponentType.F32Type,
        )
        self.addAttribute(
            attribute.Semantic.ColorSemantic,
            _POSITION_BYTES,
            attribute.ComponentType.F32Type,
        )
        if primitive != QQuick3DGeometry.PrimitiveType.Points:
            self.addAttribute(
                attribute.Semantic.IndexSemantic,
                0,
                attribute.ComponentType.U32Type,
            )
        self.clear_mesh()

    def set_mesh(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        buffer = interleave(vertices, colors)
        if not len(buffer):
            self.clear_mesh()
            return
        self.setVertexData(buffer.tobytes())
        if indices is not None:
            self.setIndexData(
                np.ascontiguousarray(indices, dtype=np.uint32).tobytes()
            )
        low = buffer[:, :3].min(axis=0)
        high = buffer[:, :3].max(axis=0)
        self.setBounds(
            QVector3D(float(low[0]), float(low[1]), float(low[2])),
            QVector3D(float(high[0]), float(high[1]), float(high[2])),
        )
        self.update()

    def clear_mesh(self) -> None:
        self.setVertexData(b"")
        self.setIndexData(b"")
        self.setBounds(QVector3D(), QVector3D())
        self.update()


class ActorListModel(QAbstractListModel):
    ActorIdRole = int(Qt.ItemDataRole.UserRole) + 1
    KindRole = ActorIdRole + 1
    XRole = KindRole + 1
    YRole = XRole + 1
    ZRole = YRole + 1
    YawRole = ZRole + 1
    WidthRole = YawRole + 1
    HeightRole = WidthRole + 1
    LengthRole = HeightRole + 1
    ConfidenceRole = LengthRole + 1

    _ROLE_NAMES = {
        ActorIdRole: QByteArray(b"actorId"),
        KindRole: QByteArray(b"kind"),
        XRole: QByteArray(b"x"),
        YRole: QByteArray(b"y"),
        ZRole: QByteArray(b"z"),
        YawRole: QByteArray(b"yaw"),
        WidthRole: QByteArray(b"actorWidth"),
        HeightRole: QByteArray(b"actorHeight"),
        LengthRole: QByteArray(b"actorLength"),
        ConfidenceRole: QByteArray(b"confidence"),
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._actors: tuple[WorldActor, ...] = ()

    def roleNames(self) -> dict[int, QByteArray]:
        return self._ROLE_NAMES

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._actors)

    def data(self, index: QModelIndex, role: int = 0) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._actors)):
            return None
        actor = self._actors[index.row()]
        width, height, length = actor.scale
        values = {
            self.ActorIdRole: actor.actor_id,
            self.KindRole: actor.kind,
            self.XRole: actor.position[0],
            self.YRole: actor.position[1],
            self.ZRole: actor.position[2],
            self.YawRole: actor.yaw_deg,
            self.WidthRole: width,
            self.HeightRole: height,
            self.LengthRole: length,
            self.ConfidenceRole: actor.confidence,
        }
        return values.get(role)

    def set_actors(self, actors: tuple[WorldActor, ...]) -> None:
        updated = tuple(actors)
        current_ids = tuple(actor.actor_id for actor in self._actors)
        updated_ids = tuple(actor.actor_id for actor in updated)
        if updated_ids == current_ids:
            self._actors = updated
            if updated:
                self.dataChanged.emit(
                    self.index(0, 0),
                    self.index(len(updated) - 1, 0),
                    list(self._ROLE_NAMES),
                )
            return
        self.beginResetModel()
        self._actors = updated
        self.endResetModel()


class SceneBridge(QObject):
    state_changed = pyqtSignal()
    geometry_changed = pyqtSignal()
    parking_slot_clicked = pyqtSignal(float, float)
    """The WORLD centre of the bay picked in the 3D view."""
    parking_selection_cleared = pyqtSignal()
    ground_picked = pyqtSignal(float, float)
    """A click on the drawn ground, in BEV metres (right, forward).

    Reported in the BEV frame and NOT in world, because this object has no
    pose: render space is ego-relative, and the worker owns the pose that
    turns it into world XY. That is the same division the bay pick uses --
    QML answers where in the scene, Python answers what that means.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        triangles = QQuick3DGeometry.PrimitiveType.Triangles
        self._parking_geometry = SceneGeometry(triangles)
        self._parking_slots: tuple[ParkingSlot, ...] = ()
        self._road_geometry = SceneGeometry(triangles)
        self._boundary_geometry = SceneGeometry(triangles)
        self._vehicle_geometry = SceneGeometry(triangles)
        self._aeb_geometry = SceneGeometry(triangles)
        self._aeb_marker_geometry = SceneGeometry(triangles)
        self._path_geometry = SceneGeometry(triangles)
        self._route_geometry = SceneGeometry(triangles)
        self._uncertain_geometry = SceneGeometry(
            QQuick3DGeometry.PrimitiveType.Points
        )
        self._actor_model = ActorListModel(self)
        self._ego_scale = (2.0, 1.5, 4.4)
        self._ego_centre = (0.0, 0.0)
        self._speed_text = "0"
        self._target_speed_text = "—"
        self._autonomy_mode = "OFF"
        self._alert_text = ""
        self._perception_available = False
        # The chase pose as the assembler computed it, kept beside the displayed
        # pose so the user orbit can be re-applied to the latest frame at any
        # time -- including between frames, when the mouse moves but no new
        # snapshot arrives.
        self._chase_position = (0.0, 12.0, 20.0)
        self._chase_euler = (-21.0, 0.0, 0.0)
        self._orbit = (0.0, 0.0, 1.0)  # yaw offset, pitch offset, zoom
        self._camera_position = self._chase_position
        self._camera_euler = self._chase_euler

    @pyqtProperty(QObject, constant=True)
    def actorModel(self) -> QObject:
        return self._actor_model

    # Constant, and deliberately so: the geometry OBJECTS never change, only
    # the buffers inside them. Qt is told about new data by
    # QQuick3DGeometry.update(), so nothing here has to be re-bound per frame.
    @pyqtProperty(QQuick3DGeometry, constant=True)
    def roadGeometry(self) -> QQuick3DGeometry:
        return self._road_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def boundaryGeometry(self) -> QQuick3DGeometry:
        return self._boundary_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def vehicleGeometry(self) -> QQuick3DGeometry:
        return self._vehicle_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def aebGeometry(self) -> QQuick3DGeometry:
        return self._aeb_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def aebMarkerGeometry(self) -> QQuick3DGeometry:
        return self._aeb_marker_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def pathGeometry(self) -> QQuick3DGeometry:
        return self._path_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def routeGeometry(self) -> QQuick3DGeometry:
        return self._route_geometry

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def parkingGeometry(self) -> QQuick3DGeometry:
        return self._parking_geometry

    @pyqtSlot(float, float)
    def parkingPicked(self, render_x: float, render_z: float) -> None:
        """
        Turn a scene point from the QML raycast into the bay that owns it.

        Render space is `(right, height, -forward)`, so dropping the height
        and negating z lands in the BEV frame the slots are already in --
        a relabelling, not a projection, which is why nothing here needs the
        camera or the ego pose.

        Reported as the bay's WORLD centre for the same reason the whole
        selection path uses one: the worker rebuilds the bay set on its own
        cadence, so any index would refer to a different bay by the time it
        arrived.
        """
        hit = next(
            (
                slot
                for slot in self._parking_slots
                if not slot.occupied
                and slot_contains(slot, render_x, -render_z)
            ),
            None,
        )
        if hit is None:
            # The ray met the bay MESH but no bay's rectangle owns the point
            # -- a chevron overhang, or a bay drawn as occupied. Treated as a
            # miss, so it deselects rather than doing nothing.
            self.parking_selection_cleared.emit()
        else:
            self.parking_slot_clicked.emit(*hit.centre_world)

    @pyqtSlot()
    def parkingMissed(self) -> None:
        self.parking_selection_cleared.emit()

    @pyqtSlot(float, float)
    def groundPicked(self, render_x: float, render_z: float) -> None:
        """The labeller's click, relabelled from render space into the BEV.

        Render space is `(right, height, -forward)`, so dropping the height
        and negating z lands in BEV -- a relabelling, not a projection, which
        is why nothing here needs the camera or the ego pose. Exactly what
        `parkingPicked` does with its own hit.
        """
        self.ground_picked.emit(render_x, -render_z)

    @property
    def has_parking_slots(self) -> bool:
        """Whether a click could pick anything, so the filter can stand aside."""
        return bool(self._parking_slots)

    @pyqtProperty(QQuick3DGeometry, constant=True)
    def uncertainGeometry(self) -> QQuick3DGeometry:
        return self._uncertain_geometry

    @pyqtProperty(float, notify=state_changed)
    def egoWidth(self) -> float:
        return self._ego_scale[0]

    @pyqtProperty(float, notify=state_changed)
    def egoHeight(self) -> float:
        return self._ego_scale[1]

    @pyqtProperty(float, notify=state_changed)
    def egoLength(self) -> float:
        return self._ego_scale[2]

    # Where the BODY centre sits in render space. The origin is the vehicle's
    # reference node, which is off-centre in the bounding box, so an ego model
    # left at the origin stands visibly beside the (correct) scene around it.
    @pyqtProperty(float, notify=state_changed)
    def egoCentreX(self) -> float:
        return self._ego_centre[0]

    @pyqtProperty(float, notify=state_changed)
    def egoCentreZ(self) -> float:
        return self._ego_centre[1]

    @pyqtProperty(str, notify=state_changed)
    def speedText(self) -> str:
        return self._speed_text

    @pyqtProperty(str, notify=state_changed)
    def targetSpeedText(self) -> str:
        return self._target_speed_text

    @pyqtProperty(str, notify=state_changed)
    def autonomyMode(self) -> str:
        return self._autonomy_mode

    @pyqtProperty(str, notify=state_changed)
    def alertText(self) -> str:
        return self._alert_text

    @pyqtProperty(bool, notify=state_changed)
    def perceptionAvailable(self) -> bool:
        return self._perception_available

    @pyqtProperty(float, notify=state_changed)
    def cameraX(self) -> float:
        return self._camera_position[0]

    @pyqtProperty(float, notify=state_changed)
    def cameraY(self) -> float:
        return self._camera_position[1]

    @pyqtProperty(float, notify=state_changed)
    def cameraZ(self) -> float:
        return self._camera_position[2]

    @pyqtProperty(float, notify=state_changed)
    def cameraPitch(self) -> float:
        return self._camera_euler[0]

    @pyqtProperty(float, notify=state_changed)
    def cameraYaw(self) -> float:
        return self._camera_euler[1]

    @pyqtSlot(object)
    def set_frame(self, frame: WorldFrame) -> None:
        self._road_geometry.set_mesh(
            frame.road_vertices, frame.road_colors, frame.road_indices
        )
        self._boundary_geometry.set_mesh(
            frame.boundary_vertices, frame.boundary_colors, frame.boundary_indices
        )
        self._vehicle_geometry.set_mesh(
            frame.vehicle_vertices, frame.vehicle_colors, frame.vehicle_indices
        )
        self._aeb_geometry.set_mesh(
            frame.aeb_vertices, frame.aeb_colors, frame.aeb_indices
        )
        self._aeb_marker_geometry.set_mesh(
            frame.aeb_marker_vertices,
            frame.aeb_marker_colors,
            frame.aeb_marker_indices,
        )
        self._path_geometry.set_mesh(
            frame.path_vertices, frame.path_colors, frame.path_indices
        )
        self._route_geometry.set_mesh(
            frame.route_vertices, frame.route_colors, frame.route_indices
        )
        self._parking_geometry.set_mesh(
            frame.parking_vertices, frame.parking_colors, frame.parking_indices
        )
        self._parking_slots = frame.parking_slots
        self._uncertain_geometry.set_mesh(
            frame.uncertain_points, frame.uncertain_colors
        )
        self._actor_model.set_actors(frame.actors)
        self._ego_scale = frame.ego_scale
        self._ego_centre = frame.ego_centre
        self._speed_text = f"{frame.speed_kph:.0f}"
        self._target_speed_text = (
            f"{frame.target_speed_kph:.0f}"
            if frame.autonomy_mode != "OFF"
            else "—"
        )
        self._autonomy_mode = frame.autonomy_mode
        self._alert_text = frame.alert
        self._perception_available = frame.perception_available
        self._chase_position = frame.camera_position
        self._chase_euler = frame.camera_euler
        self._camera_position, self._camera_euler = apply_view_orbit(
            self._chase_position, self._chase_euler, *self._orbit
        )
        self.geometry_changed.emit()
        self.state_changed.emit()

    def set_view_orbit(
        self, yaw_offset_deg: float, pitch_offset_deg: float, zoom: float
    ) -> None:
        """Re-aim the displayed camera without waiting for the next frame."""
        self._orbit = (yaw_offset_deg, pitch_offset_deg, zoom)
        self._camera_position, self._camera_euler = apply_view_orbit(
            self._chase_position, self._chase_euler, *self._orbit
        )
        self.state_changed.emit()

    @pyqtSlot()
    def clear(self) -> None:
        for geometry in (
            self._road_geometry,
            self._boundary_geometry,
            self._vehicle_geometry,
            self._aeb_geometry,
            self._aeb_marker_geometry,
            self._path_geometry,
            self._route_geometry,
            self._parking_geometry,
            self._uncertain_geometry,
        ):
            geometry.clear_mesh()
        self._parking_slots = ()
        self._actor_model.set_actors(())
        self._speed_text = "0"
        self._target_speed_text = "—"
        self._autonomy_mode = "OFF"
        self._alert_text = ""
        self._perception_available = False
        self.geometry_changed.emit()
        self.state_changed.emit()


# Mouse-orbit feel. Degrees of orbit per pixel of right-drag, the wheel's
# magnification per notch, and the zoom range -- 0.5x pulls back to twice the
# chase distance, 4x closes to a quarter of it.
_ORBIT_DEG_PER_PX = 0.35
_ZOOM_PER_WHEEL_NOTCH = 1.18
_ZOOM_MIN = 0.5
_ZOOM_MAX = 4.0
_ORBIT_PITCH_LIMIT_DEG = 88.0


class WorldView(QWidget):
    rendering_failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("worldView")
        self.setMinimumSize(320, 320)
        self.bridge = SceneBridge(self)
        self._ready = False
        self._failure_emitted = False
        self._failure_message = ""
        # Right-drag orbits, the wheel zooms, a right double-click resets.
        # Offsets on the chase camera, not a free camera: no panning by design.
        self._orbit_yaw_deg = 0.0
        self._orbit_pitch_deg = 0.0
        self._orbit_zoom = 1.0
        self._drag_from: QPointF | None = None
        # Redirects the left click from picking a bay to dropping a bay
        # corner. GUI-side only: the worker owns whether labelling is on.
        self._labelling = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._quick = QQuickWidget(self)
        self._quick.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self._quick.setClearColor(QColor("#d5d8da"))
        self._quick.rootContext().setContextProperty("sceneBridge", self.bridge)
        self._quick.statusChanged.connect(self._on_status_changed)
        qml_path = Path(__file__).with_name("qml") / "WorldScene.qml"
        self._quick.setSource(QUrl.fromLocalFile(str(qml_path)))
        layout.addWidget(self._quick)
        # The QQuickWidget receives the mouse; the filter takes the right
        # button and the wheel for the orbit and leaves everything else to QML.
        self._quick.installEventFilter(self)
        self._on_status_changed(self._quick.status())

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def failure_message(self) -> str:
        return self._failure_message

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if watched is self._quick and event is not None:
            if self._handle_view_event(event):
                return True
        return super().eventFilter(watched, event)

    def _handle_view_event(self, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
            notches = event.angleDelta().y() / 120.0
            self._orbit_zoom = min(
                max(
                    self._orbit_zoom * (_ZOOM_PER_WHEEL_NOTCH ** notches),
                    _ZOOM_MIN,
                ),
                _ZOOM_MAX,
            )
            self._push_orbit()
            return True
        if not isinstance(event, QMouseEvent):
            # Swallow the context menu too: the right button is the orbit
            # control here, so a menu popping up mid-drag would be noise.
            return kind == QEvent.Type.ContextMenu
        if (
            event.button() == Qt.MouseButton.LeftButton
            and kind == QEvent.Type.MouseButtonPress
        ):
            # Labelling takes the left button when it is on; otherwise the
            # bay pick gets it, and only when there is something to pick, so
            # with both off the left button reaches QML as it always did.
            if self._labelling:
                if self._pick_ground(event.position()):
                    return True
            elif self._pick_parking(event.position()):
                return True
        if event.button() == Qt.MouseButton.RightButton:
            if kind == QEvent.Type.MouseButtonDblClick:
                self._orbit_yaw_deg = 0.0
                self._orbit_pitch_deg = 0.0
                self._orbit_zoom = 1.0
                self._drag_from = None
                self._push_orbit()
                return True
            if kind == QEvent.Type.MouseButtonPress:
                self._drag_from = event.position()
                return True
            if kind == QEvent.Type.MouseButtonRelease:
                self._drag_from = None
                return True
        if (
            kind == QEvent.Type.MouseMove
            and self._drag_from is not None
            and event.buttons() & Qt.MouseButton.RightButton
        ):
            delta = event.position() - self._drag_from
            self._drag_from = event.position()
            # Drag right orbits round the car's right; drag up climbs toward
            # top-down. The absolute elevation clamp lives in apply_view_orbit;
            # this one only stops the OFFSET winding up past it, which would
            # put dead travel on the way back down.
            self._orbit_yaw_deg += delta.x() * _ORBIT_DEG_PER_PX
            self._orbit_pitch_deg = min(
                max(
                    self._orbit_pitch_deg - delta.y() * _ORBIT_DEG_PER_PX,
                    -_ORBIT_PITCH_LIMIT_DEG,
                ),
                _ORBIT_PITCH_LIMIT_DEG,
            )
            self._push_orbit()
            return True
        return False

    def set_labelling(self, enabled: bool) -> None:
        """While on, a left click drops a bay corner instead of picking a bay.

        The two cannot both own the left button, and labelling wins: the whole
        reason to label by hand is that the detector did not find the bay, so
        there is usually nothing there to select anyway.
        """
        self._labelling = bool(enabled)

    def _pick_ground(self, position: QPointF) -> bool:
        """Ask QML's raycast for the ground point under the click."""
        if not self._ready:
            return False
        root = self._quick.rootObject()
        if root is None:
            return False
        QMetaObject.invokeMethod(
            root,
            "pickGroundPoint",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QVariant", position.x()),
            Q_ARG("QVariant", position.y()),
        )
        return True

    def _pick_parking(self, position: QPointF) -> bool:
        """
        Ask QML's raycast what the click landed on. True if it was handled.

        The pick runs in QML because `View3D.pick` is the only thing that
        knows this camera's projection; the bridge turns the returned scene
        point into a bay. Reproducing the projection in Python would mean
        pinning down Qt Quick 3D's euler convention by hand, which is the
        class of guess this project has measured its way out of twice.
        """
        if not self._ready or not self.bridge.has_parking_slots:
            return False
        root = self._quick.rootObject()
        if root is None:
            return False
        QMetaObject.invokeMethod(
            root,
            "pickParkingBay",
            Qt.ConnectionType.DirectConnection,
            Q_ARG("QVariant", position.x()),
            Q_ARG("QVariant", position.y()),
        )
        return True

    def _push_orbit(self) -> None:
        self.bridge.set_view_orbit(
            self._orbit_yaw_deg, self._orbit_pitch_deg, self._orbit_zoom
        )

    @pyqtSlot(object)
    def set_frame(self, frame: WorldFrame) -> None:
        if self._ready:
            self.bridge.set_frame(frame)

    @pyqtSlot()
    def clear(self) -> None:
        self.bridge.clear()

    @pyqtSlot()
    def shutdown(self) -> None:
        """Unload QML before its context object is destroyed."""
        self._ready = False
        self._quick.setSource(QUrl())

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.shutdown()
        super().closeEvent(event)

    @pyqtSlot(QQuickWidget.Status)
    def _on_status_changed(self, status: QQuickWidget.Status) -> None:
        self._ready = status == QQuickWidget.Status.Ready
        if status != QQuickWidget.Status.Error or self._failure_emitted:
            return
        errors = "; ".join(error.toString() for error in self._quick.errors())
        self._emit_failure(errors or "Qt Quick 3D failed to load")

    def _emit_failure(self, message: str) -> None:
        if self._failure_emitted:
            return
        self._failure_emitted = True
        self._failure_message = message
        self.rendering_failed.emit(message)
