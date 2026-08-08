from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    Qt,
    QUrl,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QCloseEvent, QColor, QVector3D
from PyQt6.QtQuickWidgets import QQuickWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .models import WorldActor, WorldFrame


def qml_vectors(vertices: np.ndarray) -> list[QVector3D]:
    """Convert an Nx3 NumPy buffer to a QML-compatible vector list."""
    positions = np.asarray(vertices, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 vertex array, got {positions.shape}")
    return [
        QVector3D(float(position[0]), float(position[1]), float(position[2]))
        for position in positions
    ]


def qml_indices(indices: np.ndarray) -> list[int]:
    """Convert a NumPy index buffer to the signed integer list QML expects."""
    values = np.asarray(indices, dtype=np.uint32).reshape(-1)
    return [int(value) for value in values]


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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._road_positions: list[QVector3D] = []
        self._road_indices: list[int] = []
        self._boundary_positions: list[QVector3D] = []
        self._boundary_indices: list[int] = []
        self._path_positions: list[QVector3D] = []
        self._path_indices: list[int] = []
        self._uncertain_positions: list[QVector3D] = []
        self._actor_model = ActorListModel(self)
        self._ego_scale = (2.0, 1.5, 4.4)
        self._speed_text = "0"
        self._target_speed_text = "—"
        self._autonomy_mode = "OFF"
        self._alert_text = ""
        self._perception_available = False
        self._camera_position = (0.0, 12.0, 20.0)
        self._camera_euler = (-21.0, 0.0, 0.0)

    @pyqtProperty(QObject, constant=True)
    def actorModel(self) -> QObject:
        return self._actor_model

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def roadPositions(self) -> list[QVector3D]:
        return self._road_positions

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def roadIndices(self) -> list[int]:
        return self._road_indices

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def boundaryPositions(self) -> list[QVector3D]:
        return self._boundary_positions

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def boundaryIndices(self) -> list[int]:
        return self._boundary_indices

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def pathPositions(self) -> list[QVector3D]:
        return self._path_positions

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def pathIndices(self) -> list[int]:
        return self._path_indices

    @pyqtProperty("QVariantList", notify=geometry_changed)
    def uncertainPositions(self) -> list[QVector3D]:
        return self._uncertain_positions

    @pyqtProperty(float, notify=state_changed)
    def egoWidth(self) -> float:
        return self._ego_scale[0]

    @pyqtProperty(float, notify=state_changed)
    def egoHeight(self) -> float:
        return self._ego_scale[1]

    @pyqtProperty(float, notify=state_changed)
    def egoLength(self) -> float:
        return self._ego_scale[2]

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
        self._road_positions = qml_vectors(frame.road_vertices)
        self._road_indices = qml_indices(frame.road_indices)
        self._boundary_positions = qml_vectors(frame.boundary_vertices)
        self._boundary_indices = qml_indices(frame.boundary_indices)
        self._path_positions = qml_vectors(frame.path_vertices)
        self._path_indices = qml_indices(frame.path_indices)
        self._uncertain_positions = qml_vectors(frame.uncertain_points)
        self._actor_model.set_actors(frame.actors)
        self._ego_scale = frame.ego_scale
        self._speed_text = f"{frame.speed_kph:.0f}"
        self._target_speed_text = (
            f"{frame.target_speed_kph:.0f}"
            if frame.autonomy_mode != "OFF"
            else "—"
        )
        self._autonomy_mode = frame.autonomy_mode
        self._alert_text = frame.alert
        self._perception_available = frame.perception_available
        self._camera_position = frame.camera_position
        self._camera_euler = frame.camera_euler
        self.geometry_changed.emit()
        self.state_changed.emit()

    @pyqtSlot()
    def clear(self) -> None:
        self._road_positions = []
        self._road_indices = []
        self._boundary_positions = []
        self._boundary_indices = []
        self._path_positions = []
        self._path_indices = []
        self._uncertain_positions = []
        self._actor_model.set_actors(())
        self._speed_text = "0"
        self._target_speed_text = "—"
        self._autonomy_mode = "OFF"
        self._alert_text = ""
        self._perception_available = False
        self.geometry_changed.emit()
        self.state_changed.emit()


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
        self._on_status_changed(self._quick.status())

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def failure_message(self) -> str:
        return self._failure_message

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
