from __future__ import annotations

import logging
import time
from collections import deque

from PyQt6.QtCore import QMetaObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .bev_widget import BevWidget
from .bridge_monitor import BridgeMonitor
from .config import (
    AEB_MIN_SPEED_MPS,
    AEB_REVERSE_MIN_SPEED_MPS,
    APP_NAME,
    BEAMNG_EXE,
    LIDAR_FRONT_HORIZONTAL_FOV_DEG,
    LIDAR_FRONT_MAX_DISTANCE_M,
    LIDAR_HORIZONTAL_FOV_DEG,
    LIDAR_MAX_DISTANCE_M,
    LIDAR_UPDATE_HZ,
    MAX_SPEED_MPS,
    SENSOR_HEIGHT_ABOVE_GROUND_M,
)
from .models import BevFrame, VehicleGeometry
from .scene_worker import SceneWorker
from .worker import BeamNgWorker
from .world_view import WorldView

LOGGER = logging.getLogger(__name__)


def resolve_visualization(
    requested: str | None, *, world_available: bool
) -> str:
    """Resolve a persisted view choice without selecting a broken renderer."""
    if not world_available:
        return "RAW BEV"
    return "RAW BEV" if requested == "RAW BEV" else "WORLD"


class MainWindow(QMainWindow):
    launch_requested = pyqtSignal()
    attach_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    bridge_lost_requested = pyqtSignal()
    self_drive_requested = pyqtSignal(bool)
    aeb_requested = pyqtSignal(bool)
    rear_aeb_requested = pyqtSignal(bool)
    scene_clear_requested = pyqtSignal()
    streaming_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(640, 560)
        self.resize(1180, 820)
        self._settings = QSettings("Local BeamNG Tools", "BeamNG LiDAR BEV")
        self._bridge_ready = False
        # Explicit phase rather than reusing _last_status: "READY" is emitted
        # both by a successful stop and by a failed attach, so it is ambiguous.
        self._phase = "IDLE"  # IDLE | LAUNCHING | BUSY | STREAMING
        self._display_times: deque[float] = deque(maxlen=60)
        self._last_status = "OFFLINE"
        self._compact_mode: bool | None = None
        self._dense_mode = False
        self._active_visualization = "RAW BEV"

        self._build_ui()
        self._apply_styles()
        if self.world_view.failure_message:
            self._on_world_rendering_failed(self.world_view.failure_message)
        self._start_worker()
        requested_view = self._settings.value("visualization", "WORLD")
        self._select_visualization(
            resolve_visualization(
                str(requested_view) if requested_view is not None else None,
                world_available=self.world_view.is_ready,
            )
        )

        self._set_status("OFFLINE", "Looking for a running BeamNG.tech session")
        self._append_log("Application ready")
        if not BEAMNG_EXE.is_file():
            self._set_launch_enabled(False)
            self._set_status("ERROR", "Configured BeamNG.tech executable is missing")

        saved_geometry = self._settings.value("windowGeometry")
        if saved_geometry is not None:
            self.restoreGeometry(saved_geometry)
        self._update_responsive_layout()

    def _build_ui(self) -> None:
        application_style = self.style()
        assert application_style is not None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(292)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(24, 22, 24, 22)
        sidebar_layout.setSpacing(12)

        brand = QLabel("BEAMNG / LIDAR BEV")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)

        self.status_badge = QLabel("OFFLINE")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setFixedHeight(28)
        sidebar_layout.addWidget(self.status_badge)

        self.status_message = QLabel()
        self.status_message.setObjectName("statusMessage")
        self.status_message.setWordWrap(True)
        self.status_message.setMinimumHeight(42)
        sidebar_layout.addWidget(self.status_message)

        sidebar_layout.addWidget(self._separator())

        exe_title = QLabel("SIMULATOR")
        exe_title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(exe_title)
        exe_name = QLabel(BEAMNG_EXE.name)
        exe_name.setObjectName("pathName")
        exe_name.setToolTip(str(BEAMNG_EXE))
        sidebar_layout.addWidget(exe_name)

        self.launch_button = QPushButton("Launch BeamNG.tech")
        self.launch_button.setObjectName("primaryButton")
        self.launch_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.launch_button.setToolTip("Start BeamNG.tech with the BeamNGpy bridge")
        self.launch_button.clicked.connect(self._request_launch)
        sidebar_layout.addWidget(self.launch_button)

        self.attach_button = QPushButton("Attach to Player Vehicle")
        self.attach_button.setObjectName("attachButton")
        self.attach_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        self.attach_button.setToolTip(
            "Attach four LiDAR sensors to the currently selected vehicle"
        )
        self.attach_button.setEnabled(False)
        self.attach_button.clicked.connect(self._request_attach)
        sidebar_layout.addWidget(self.attach_button)

        self.stop_button = QPushButton("Stop Sensors")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_stop)
        sidebar_layout.addWidget(self.stop_button)

        self.self_drive_button = QPushButton("Self-Driving")
        self.self_drive_button.setObjectName("selfDriveButton")
        self.self_drive_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        self.self_drive_button.setCheckable(True)
        self.self_drive_button.setToolTip(
            f"Drive from the LiDAR cloud, capped at {MAX_SPEED_MPS * 3.6:.0f} km/h. "
            "An in-game destination is used only to pick turns."
        )
        self.self_drive_button.setEnabled(False)
        # clicked, not toggled: setChecked() from the worker's own
        # self_driving_changed would otherwise loop straight back out again.
        self.self_drive_button.clicked.connect(self._request_self_drive)
        sidebar_layout.addWidget(self.self_drive_button)

        self.aeb_button = QPushButton("Emergency Braking (AEB)")
        self.aeb_button.setObjectName("aebButton")
        self.aeb_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        )
        self.aeb_button.setCheckable(True)
        self.aeb_button.setToolTip(
            "Brake automatically when the car's predicted path is about to hit "
            f"something. Armed above {AEB_MIN_SPEED_MPS * 3.6:.0f} km/h, and it "
            "touches the pedals only -- steering stays with the driver, so it "
            "works with or without self-driving."
        )
        self.aeb_button.setEnabled(False)
        self.aeb_button.clicked.connect(self._request_aeb)
        sidebar_layout.addWidget(self.aeb_button)

        self.rear_aeb_button = QPushButton("Reverse AEB")
        self.rear_aeb_button.setObjectName("aebButton")
        self.rear_aeb_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.rear_aeb_button.setCheckable(True)
        self.rear_aeb_button.setToolTip(
            "The same emergency brake pointed backwards, for reversing. Armed "
            f"above {AEB_REVERSE_MIN_SPEED_MPS * 3.6:.0f} km/h -- much lower "
            "than the forward system, because stopping the car before it backs "
            "into something is parking-speed work. The car brakes about 30% "
            "worse in reverse, and the trigger is measured for that."
        )
        self.rear_aeb_button.setEnabled(False)
        self.rear_aeb_button.clicked.connect(self._request_rear_aeb)
        sidebar_layout.addWidget(self.rear_aeb_button)

        sidebar_layout.addWidget(self._separator())
        sensor_title = QLabel("SENSOR ARRAY")
        sensor_title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(sensor_title)

        self._add_spec_row(sidebar_layout, "Units", "Front / Left / Right / Rear")
        self._add_spec_row(
            sidebar_layout,
            "Range",
            f"{LIDAR_FRONT_MAX_DISTANCE_M:.0f} front / "
            f"{LIDAR_MAX_DISTANCE_M:.0f} m",
        )
        self._add_spec_row(
            sidebar_layout,
            "Horizontal FOV",
            f"{LIDAR_FRONT_HORIZONTAL_FOV_DEG:.0f} + "
            f"{LIDAR_HORIZONTAL_FOV_DEG:.0f} deg x 3",
        )
        self._add_spec_row(
            sidebar_layout, "Mount height", f"{SENSOR_HEIGHT_ABOVE_GROUND_M:.2f} m"
        )
        self._add_spec_row(
            sidebar_layout, "Requested rate", f"{LIDAR_UPDATE_HZ:.0f} Hz"
        )

        sidebar_layout.addWidget(self._separator())
        log_title = QLabel("EVENTS")
        log_title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(log_title)

        self.event_log = QPlainTextEdit()
        self.event_log.setObjectName("eventLog")
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(120)
        self.event_log.setMinimumHeight(100)
        sidebar_layout.addWidget(self.event_log, 1)

        root_layout.addWidget(self.sidebar)

        self.content = QFrame()
        self.content.setObjectName("content")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(24, 18, 24, 22)
        self.content_layout.setSpacing(12)

        self.compact_controls = QFrame()
        self.compact_controls.setObjectName("compactControls")
        compact_layout = QHBoxLayout(self.compact_controls)
        compact_layout.setContentsMargins(10, 7, 8, 7)
        compact_layout.setSpacing(8)

        self.compact_status_badge = QLabel("OFFLINE")
        self.compact_status_badge.setObjectName("statusBadge")
        self.compact_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compact_status_badge.setFixedSize(98, 30)
        compact_layout.addWidget(self.compact_status_badge)

        self.compact_status_message = QLabel()
        self.compact_status_message.setObjectName("statusMessage")
        self.compact_status_message.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        compact_layout.addWidget(self.compact_status_message, 1)

        self.compact_launch_button = QPushButton()
        self.compact_launch_button.setObjectName("compactActionButton")
        self.compact_launch_button.setProperty("role", "launch")
        self.compact_launch_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.compact_launch_button.setAccessibleName("Launch BeamNG.tech")
        self.compact_launch_button.setToolTip("Launch BeamNG.tech")
        self.compact_launch_button.setFixedSize(40, 40)
        self.compact_launch_button.clicked.connect(self._request_launch)
        compact_layout.addWidget(self.compact_launch_button)

        self.compact_attach_button = QPushButton()
        self.compact_attach_button.setObjectName("compactActionButton")
        self.compact_attach_button.setProperty("role", "attach")
        self.compact_attach_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        )
        self.compact_attach_button.setAccessibleName("Attach to player vehicle")
        self.compact_attach_button.setToolTip("Attach to player vehicle")
        self.compact_attach_button.setFixedSize(40, 40)
        self.compact_attach_button.setEnabled(False)
        self.compact_attach_button.clicked.connect(self._request_attach)
        compact_layout.addWidget(self.compact_attach_button)

        self.compact_stop_button = QPushButton()
        self.compact_stop_button.setObjectName("compactActionButton")
        self.compact_stop_button.setProperty("role", "stop")
        self.compact_stop_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.compact_stop_button.setAccessibleName("Stop sensors")
        self.compact_stop_button.setToolTip("Stop sensors")
        self.compact_stop_button.setFixedSize(40, 40)
        self.compact_stop_button.setEnabled(False)
        self.compact_stop_button.clicked.connect(self._request_stop)
        compact_layout.addWidget(self.compact_stop_button)

        self.compact_self_drive_button = QPushButton()
        self.compact_self_drive_button.setObjectName("compactActionButton")
        self.compact_self_drive_button.setProperty("role", "selfdrive")
        self.compact_self_drive_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        self.compact_self_drive_button.setAccessibleName("Self-driving")
        self.compact_self_drive_button.setToolTip("Self-driving")
        self.compact_self_drive_button.setCheckable(True)
        self.compact_self_drive_button.setFixedSize(40, 40)
        self.compact_self_drive_button.setEnabled(False)
        self.compact_self_drive_button.clicked.connect(self._request_self_drive)
        compact_layout.addWidget(self.compact_self_drive_button)

        self.compact_aeb_button = QPushButton()
        self.compact_aeb_button.setObjectName("compactActionButton")
        self.compact_aeb_button.setProperty("role", "aeb")
        self.compact_aeb_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        )
        self.compact_aeb_button.setAccessibleName("Emergency braking")
        self.compact_aeb_button.setToolTip("Emergency braking (AEB)")
        self.compact_aeb_button.setCheckable(True)
        self.compact_aeb_button.setFixedSize(40, 40)
        self.compact_aeb_button.setEnabled(False)
        self.compact_aeb_button.clicked.connect(self._request_aeb)
        compact_layout.addWidget(self.compact_aeb_button)

        self.compact_rear_aeb_button = QPushButton()
        self.compact_rear_aeb_button.setObjectName("compactActionButton")
        self.compact_rear_aeb_button.setProperty("role", "aeb")
        self.compact_rear_aeb_button.setIcon(
            application_style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.compact_rear_aeb_button.setAccessibleName("Reverse emergency braking")
        self.compact_rear_aeb_button.setToolTip("Reverse AEB")
        self.compact_rear_aeb_button.setCheckable(True)
        self.compact_rear_aeb_button.setFixedSize(40, 40)
        self.compact_rear_aeb_button.setEnabled(False)
        self.compact_rear_aeb_button.clicked.connect(self._request_rear_aeb)
        compact_layout.addWidget(self.compact_rear_aeb_button)
        self.compact_controls.setVisible(False)
        self.content_layout.addWidget(self.compact_controls)

        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        self.view_title = QLabel("RECONSTRUCTED WORLD")
        self.view_title.setObjectName("viewTitle")
        self.vehicle_label = QLabel("No vehicle attached")
        self.vehicle_label.setObjectName("vehicleLabel")
        title_column.addWidget(self.view_title)
        title_column.addWidget(self.vehicle_label)
        header.addLayout(title_column)
        header.addStretch(1)

        view_toggle = QFrame()
        view_toggle.setObjectName("viewToggle")
        view_toggle_layout = QHBoxLayout(view_toggle)
        view_toggle_layout.setContentsMargins(3, 3, 3, 3)
        view_toggle_layout.setSpacing(2)
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        self.world_view_button = QPushButton("WORLD")
        self.raw_bev_button = QPushButton("RAW BEV")
        for button in (self.world_view_button, self.raw_bev_button):
            button.setObjectName("viewToggleButton")
            button.setCheckable(True)
            button.setFixedHeight(28)
            self.view_button_group.addButton(button)
            view_toggle_layout.addWidget(button)
        self.world_view_button.clicked.connect(
            lambda checked: checked and self._select_visualization("WORLD")
        )
        self.raw_bev_button.clicked.connect(
            lambda checked: checked and self._select_visualization("RAW BEV")
        )
        header.addWidget(view_toggle)

        self.legend_widget = QWidget()
        legend = QHBoxLayout(self.legend_widget)
        legend.setContentsMargins(0, 0, 0, 0)
        legend.setSpacing(14)
        legend.addWidget(self._legend_item("#8f959a", "Road / drivable"))
        legend.addWidget(self._legend_item("#ff444e", "Objects / boundaries"))
        header.addWidget(self.legend_widget)
        self.content_layout.addWidget(self.header_widget)

        self.stats_band = QFrame()
        self.stats_band.setObjectName("statsBand")
        self.stats_layout = QHBoxLayout(self.stats_band)
        self.stats_layout.setContentsMargins(16, 10, 16, 10)
        self.stats_layout.setSpacing(0)
        self.acquisition_container, self.acquisition_value = self._add_metric(
            self.stats_layout, "ACQUISITION", "0.0 Hz"
        )
        self.display_container, self.display_value = self._add_metric(
            self.stats_layout, "DISPLAY", "0.0 fps"
        )
        self.points_container, self.points_value = self._add_metric(
            self.stats_layout, "VISIBLE POINTS", "0"
        )
        self.latency_container, self.latency_value = self._add_metric(
            self.stats_layout, "POLL TIME", "0.0 ms"
        )
        self.speed_container, self.speed_value = self._add_metric(
            self.stats_layout, "EGO SPEED", "0.0 km/h"
        )
        self.autopilot_container, self.autopilot_value = self._add_metric(
            self.stats_layout, "AUTOPILOT", "OFF"
        )
        self.aeb_container, self.aeb_value = self._add_metric(
            self.stats_layout, "AEB", "OFF"
        )
        self.rear_aeb_container, self.rear_aeb_value = self._add_metric(
            self.stats_layout, "REAR AEB", "OFF"
        )
        self.content_layout.addWidget(self.stats_band)

        self.visual_stack = QStackedWidget()
        self.visual_stack.setObjectName("visualStack")
        self.world_view = WorldView()
        self.world_view.rendering_failed.connect(
            self._on_world_rendering_failed
        )
        self.world_view_button.setEnabled(self.world_view.is_ready)
        self.bev = BevWidget()
        self.visual_stack.addWidget(self.world_view)
        self.visual_stack.addWidget(self.bev)
        self.content_layout.addWidget(self.visual_stack, 1)
        root_layout.addWidget(self.content, 1)

    def _start_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker_thread.setObjectName("BeamNgWorkerThread")
        self.worker = BeamNgWorker()
        self.worker.moveToThread(self.worker_thread)

        self.launch_requested.connect(self.worker.launch_beamng)
        self.attach_requested.connect(self.worker.attach_to_player)
        self.stop_requested.connect(self.worker.stop_sensors)
        self.bridge_lost_requested.connect(self.worker.handle_bridge_lost)
        self.self_drive_requested.connect(self.worker.set_self_driving)
        self.aeb_requested.connect(self.worker.set_aeb)
        self.rear_aeb_requested.connect(self.worker.set_rear_aeb)
        self.worker.status_changed.connect(self._set_status)
        self.worker.launch_ready.connect(self._on_launch_ready)
        self.worker.sensors_ready.connect(self._on_sensors_ready)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.sensors_stopped.connect(self._on_sensors_stopped)
        self.worker.self_driving_changed.connect(self._on_self_driving_changed)
        self.worker.aeb_changed.connect(self._on_aeb_changed)
        self.worker.rear_aeb_changed.connect(self._on_rear_aeb_changed)
        self.worker.fatal_error.connect(self._show_error)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()

        self.scene_thread = QThread(self)
        self.scene_thread.setObjectName("WorldSceneWorkerThread")
        self.scene_worker = SceneWorker()
        self.scene_worker.moveToThread(self.scene_thread)
        # DirectConnection only deposits the snapshot into SceneWorker's
        # locked latest-value mailbox; construction is invoked on scene_thread.
        # This prevents Qt's event queue from retaining every large LiDAR array
        # when scene construction is temporarily slower than acquisition.
        self.worker.perception_ready.connect(
            self.scene_worker.submit,
            type=Qt.ConnectionType.DirectConnection,
        )
        self.scene_worker.world_frame_ready.connect(self.world_view.set_frame)
        self.scene_worker.scene_error.connect(self._on_scene_error)
        self.scene_clear_requested.connect(self.scene_worker.clear)
        self.scene_thread.finished.connect(self.scene_worker.deleteLater)
        self.scene_thread.start()

        # The bridge probe blocks on connect(), so it gets its own thread: the
        # GUI thread would stutter and the worker thread is already saturated.
        self.monitor_thread = QThread(self)
        self.monitor_thread.setObjectName("BridgeMonitorThread")
        self.monitor = BridgeMonitor()
        self.monitor.moveToThread(self.monitor_thread)
        self.streaming_changed.connect(self.monitor.set_streaming)
        self.monitor.bridge_up.connect(self._on_bridge_up)
        self.monitor.bridge_down.connect(self._on_bridge_down)
        self.monitor_thread.started.connect(self.monitor.start)
        self.monitor_thread.finished.connect(self.monitor.deleteLater)
        self.monitor_thread.start()

    def _request_launch(self) -> None:
        self._phase = "LAUNCHING"
        self._set_launch_enabled(False)
        self._set_attach_enabled(False)
        self._append_log("Starting BeamNG.tech bridge")
        self.launch_requested.emit()

    def _request_attach(self) -> None:
        self._phase = "BUSY"
        self._set_attach_enabled(False)
        self._set_stop_enabled(False)
        self.bev.clear()
        self.world_view.clear()
        self.scene_clear_requested.emit()
        self._display_times.clear()
        self._append_log("Looking for the current player vehicle")
        self.attach_requested.emit()

    def _request_stop(self) -> None:
        self._set_stop_enabled(False)
        self._append_log("Stopping LiDAR sensors")
        self.stop_requested.emit()

    def _request_self_drive(self, engaged: bool) -> None:
        # Gated on STREAMING because the worker needs a live vehicle and four
        # sensors; it refuses otherwise and would just bounce the button back.
        if engaged and self._phase != "STREAMING":
            self._set_self_drive_checked(False)
            return
        self._set_self_drive_checked(engaged)
        self.self_drive_requested.emit(engaged)

    def _request_aeb(self, armed: bool) -> None:
        # Same gate as self-driving: the worker needs a live vehicle and four
        # sensors, and would only bounce the button back otherwise.
        if armed and self._phase != "STREAMING":
            self._set_aeb_checked(False)
            return
        self._set_aeb_checked(armed)
        self.aeb_requested.emit(armed)

    def _request_rear_aeb(self, armed: bool) -> None:
        if armed and self._phase != "STREAMING":
            self._set_rear_aeb_checked(False)
            return
        self._set_rear_aeb_checked(armed)
        self.rear_aeb_requested.emit(armed)

    def _on_launch_ready(self) -> None:
        # Only means the process was spawned; the bridge monitor decides when
        # the port is actually listening and enables Attach then.
        self._phase = "IDLE"
        self._set_launch_enabled(False)
        self._set_attach_enabled(self._bridge_ready)
        self._append_log("BeamNG.tech process launched; waiting for its bridge")

    def _on_bridge_up(self) -> None:
        self._bridge_ready = True
        if self._phase != "IDLE":
            return
        self._set_attach_enabled(True)
        self._set_launch_enabled(False)
        self.launch_button.setToolTip("BeamNG.tech is already running")
        self._set_status(
            "LAUNCHED", "Found a running BeamNG.tech session; ready to attach"
        )

    def _on_bridge_down(self) -> None:
        self._bridge_ready = False
        if self._phase == "STREAMING":
            self._phase = "BUSY"
            self._append_log("BeamNG.tech disappeared; releasing sensors")
            self.bridge_lost_requested.emit()
            return
        if self._phase != "IDLE":
            return
        self._set_attach_enabled(False)
        self._set_launch_enabled(BEAMNG_EXE.is_file())
        self.launch_button.setToolTip("Start BeamNG.tech with the BeamNGpy bridge")
        self._set_status("OFFLINE", "BeamNG.tech is not running")

    def _on_sensors_ready(self, vehicle_id: str, geometry: VehicleGeometry) -> None:
        self._phase = "STREAMING"
        self.streaming_changed.emit(True)
        self._set_attach_enabled(False)
        self._set_stop_enabled(True)
        self._set_self_drive_enabled(True)
        self._set_aeb_enabled(True)
        self._set_rear_aeb_enabled(True)
        self.vehicle_label.setText(
            f"EGO {vehicle_id}  |  {geometry.length_m:.2f} x {geometry.width_m:.2f} m"
        )
        self._append_log(
            f"Four LiDAR sensors attached to {vehicle_id} at "
            f"{geometry.ground_z_vehicle + SENSOR_HEIGHT_ABOVE_GROUND_M:.2f} m "
            "vehicle-space Z"
        )

    def _on_sensors_stopped(self) -> None:
        self._phase = "IDLE"
        self.streaming_changed.emit(False)
        self._set_attach_enabled(self._bridge_ready)
        self._set_stop_enabled(False)
        # Never leave these checked across a teardown: the next attach would
        # show engaged buttons over a car that is not being driven or watched.
        self._set_self_drive_enabled(False)
        self._set_self_drive_checked(False)
        self._set_aeb_enabled(False)
        self._set_aeb_checked(False)
        self._set_rear_aeb_enabled(False)
        self._set_rear_aeb_checked(False)
        self.vehicle_label.setText("No vehicle attached")
        self.bev.clear()
        self.world_view.clear()
        self.scene_clear_requested.emit()
        self._reset_metrics()
        self._append_log("LiDAR sensors stopped")

    def _on_frame(self, frame: BevFrame) -> None:
        now = time.perf_counter()
        self._display_times.append(now)
        display_fps = 0.0
        if len(self._display_times) >= 2:
            elapsed = self._display_times[-1] - self._display_times[0]
            if elapsed > 0:
                display_fps = (len(self._display_times) - 1) / elapsed

        self.bev.set_frame(frame)
        self.acquisition_value.setText(f"{frame.acquisition_fps:.1f} Hz")
        self.display_value.setText(f"{display_fps:.1f} fps")
        self.points_value.setText(f"{frame.raw_point_count:,}")
        self.latency_value.setText(f"{frame.poll_ms:.1f} ms")
        self.speed_value.setText(f"{frame.speed_mps * 3.6:.1f} km/h")
        self.autopilot_value.setText(
            frame.plan.command.mode if frame.plan is not None else "OFF"
        )
        self.aeb_value.setText(
            frame.aeb.status if frame.aeb is not None else "OFF"
        )
        self.rear_aeb_value.setText(
            frame.rear_aeb.status if frame.rear_aeb is not None else "OFF"
        )

    def _select_visualization(self, requested: str) -> None:
        selected = resolve_visualization(
            requested,
            world_available=self.world_view.is_ready
            and self.world_view_button.isEnabled(),
        )
        self._active_visualization = selected
        world_selected = selected == "WORLD"
        self.visual_stack.setCurrentWidget(
            self.world_view if world_selected else self.bev
        )
        self.world_view_button.setChecked(world_selected)
        self.raw_bev_button.setChecked(not world_selected)
        self.view_title.setText(
            "RECONSTRUCTED WORLD" if world_selected else "EGO BIRD'S-EYE VIEW"
        )
        self.stats_band.setVisible(not world_selected)
        self.legend_widget.setVisible(
            not world_selected and not self._dense_mode
        )
        self._settings.setValue("visualization", selected)

    def _on_world_rendering_failed(self, message: str) -> None:
        self._append_log(f"3D WORLD unavailable: {message}")
        self.world_view_button.setEnabled(False)
        self._select_visualization("RAW BEV")

    def _on_scene_error(self, message: str) -> None:
        self._append_log(message)

    def _set_status(self, state: str, message: str) -> None:
        self._last_status = state
        for badge in (self.status_badge, self.compact_status_badge):
            badge.setText(state)
            badge.setProperty("state", state)
            badge_style = badge.style()
            if badge_style is not None:
                badge_style.unpolish(badge)
                badge_style.polish(badge)
        self.status_message.setText(message)
        self.compact_status_message.setText(message)
        self._append_log(message)

        if state == "ERROR":
            # Do not touch _bridge_ready here: the monitor owns it, and an
            # attach failure says nothing about whether BeamNG.tech is running.
            # Returning to IDLE lets the next probe restore Attach by itself.
            self._phase = "IDLE"
            self.streaming_changed.emit(False)
            self._set_launch_enabled(BEAMNG_EXE.is_file() and not self._bridge_ready)
            self._set_attach_enabled(self._bridge_ready)
            self._set_stop_enabled(False)
            self._set_self_drive_enabled(False)
            self._set_self_drive_checked(False)
            self._set_aeb_enabled(False)
            self._set_aeb_checked(False)
            self._set_rear_aeb_enabled(False)
            self._set_rear_aeb_checked(False)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, APP_NAME, message)

    # The setters are idempotent because a 2 s monitor tick would otherwise
    # reset the hover/focus visuals of whichever button has focus, and because
    # QMessageBox spins a nested event loop that lets queued bridge slots run.
    def _set_launch_enabled(self, enabled: bool) -> None:
        if enabled == self.launch_button.isEnabled():
            return
        self.launch_button.setEnabled(enabled)
        self.compact_launch_button.setEnabled(enabled)

    def _set_attach_enabled(self, enabled: bool) -> None:
        if enabled == self.attach_button.isEnabled():
            return
        self.attach_button.setEnabled(enabled)
        self.compact_attach_button.setEnabled(enabled)

    def _set_stop_enabled(self, enabled: bool) -> None:
        if enabled == self.stop_button.isEnabled():
            return
        self.stop_button.setEnabled(enabled)
        self.compact_stop_button.setEnabled(enabled)

    def _set_self_drive_enabled(self, enabled: bool) -> None:
        if enabled == self.self_drive_button.isEnabled():
            return
        self.self_drive_button.setEnabled(enabled)
        self.compact_self_drive_button.setEnabled(enabled)

    def _set_self_drive_checked(self, checked: bool) -> None:
        for button in (self.self_drive_button, self.compact_self_drive_button):
            if button.isChecked() != checked:
                button.setChecked(checked)
            button.setProperty("state", "engaged" if checked else "idle")
            button_style = button.style()
            if button_style is not None:
                button_style.unpolish(button)
                button_style.polish(button)

    def _set_aeb_enabled(self, enabled: bool) -> None:
        if enabled == self.aeb_button.isEnabled():
            return
        self.aeb_button.setEnabled(enabled)
        self.compact_aeb_button.setEnabled(enabled)

    def _set_aeb_checked(self, checked: bool) -> None:
        for button in (self.aeb_button, self.compact_aeb_button):
            if button.isChecked() != checked:
                button.setChecked(checked)
            button.setProperty("state", "engaged" if checked else "idle")
            button_style = button.style()
            if button_style is not None:
                button_style.unpolish(button)
                button_style.polish(button)

    def _set_rear_aeb_enabled(self, enabled: bool) -> None:
        if enabled == self.rear_aeb_button.isEnabled():
            return
        self.rear_aeb_button.setEnabled(enabled)
        self.compact_rear_aeb_button.setEnabled(enabled)

    def _set_rear_aeb_checked(self, checked: bool) -> None:
        for button in (self.rear_aeb_button, self.compact_rear_aeb_button):
            if button.isChecked() != checked:
                button.setChecked(checked)
            button.setProperty("state", "engaged" if checked else "idle")
            button_style = button.style()
            if button_style is not None:
                button_style.unpolish(button)
                button_style.polish(button)

    def _on_rear_aeb_changed(self, armed: bool) -> None:
        self._set_rear_aeb_checked(armed)
        if not armed:
            self.rear_aeb_value.setText("OFF")

    def _on_self_driving_changed(self, engaged: bool) -> None:
        """The worker owns the truth: it auto-disengages on faults and teardown."""
        self._set_self_drive_checked(engaged)
        if not engaged:
            self.autopilot_value.setText("OFF")

    def _on_aeb_changed(self, armed: bool) -> None:
        self._set_aeb_checked(armed)
        if not armed:
            self.aeb_value.setText("OFF")

    def _reset_metrics(self) -> None:
        self.acquisition_value.setText("0.0 Hz")
        self.display_value.setText("0.0 fps")
        self.points_value.setText("0")
        self.latency_value.setText("0.0 ms")
        self.speed_value.setText("0.0 km/h")
        self.autopilot_value.setText("OFF")
        self.aeb_value.setText("OFF")
        self.rear_aeb_value.setText("OFF")

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "event_log"):
            return
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.appendPlainText(f"{timestamp}  {message}")
        LOGGER.info(message)

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _add_spec_row(layout: QVBoxLayout, name: str, value: str) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 2, 0, 2)
        name_label = QLabel(name)
        name_label.setObjectName("specName")
        value_label = QLabel(value)
        value_label.setObjectName("specValue")
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row_layout.addWidget(name_label)
        row_layout.addWidget(value_label, 1)
        layout.addWidget(row)

    @staticmethod
    def _legend_item(colour: str, text: str) -> QWidget:
        item = QWidget()
        item.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        swatch = QFrame()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background: {colour}; border-radius: 2px;")
        label = QLabel(text)
        label.setObjectName("legendLabel")
        layout.addWidget(swatch)
        layout.addWidget(label)
        return item

    @staticmethod
    def _add_metric(
        layout: QHBoxLayout, label: str, initial: str
    ) -> tuple[QWidget, QLabel]:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 0, 12, 0)
        container_layout.setSpacing(1)
        title = QLabel(label)
        title.setObjectName("metricTitle")
        value = QLabel(initial)
        value.setObjectName("metricValue")
        container_layout.addWidget(title)
        container_layout.addWidget(value)
        layout.addWidget(container, 1)
        return container, value

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if hasattr(self, "speed_container"):
            width = event.size().width() if event is not None else self.width()
            self._update_responsive_layout(width)

    def _update_responsive_layout(self, width: int | None = None) -> None:
        available_width = self.width() if width is None else width
        compact = available_width < 980
        dense = available_width < 720
        self._dense_mode = dense

        if compact != self._compact_mode:
            self._compact_mode = compact
            self.sidebar.setVisible(not compact)
            self.compact_controls.setVisible(compact)

        if compact:
            self.content_layout.setContentsMargins(12, 10, 12, 12)
            self.content_layout.setSpacing(8)
            self.stats_layout.setContentsMargins(6, 7, 6, 7)
        else:
            self.content_layout.setContentsMargins(24, 18, 24, 22)
            self.content_layout.setSpacing(12)
            self.stats_layout.setContentsMargins(16, 10, 16, 10)

        self.legend_widget.setVisible(
            not dense and self._active_visualization == "RAW BEV"
        )
        self.latency_container.setVisible(not dense)
        self.speed_container.setVisible(not dense)
        self.acquisition_container.setVisible(not dense)
        # AUTOPILOT and AEB survive the dense layout: when the car is driving
        # itself or braking itself, those matter more than any diagnostic
        # beside them. ACQUISITION goes instead, to keep the band from
        # crushing now that there are two of them.

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._settings.setValue("windowGeometry", self.saveGeometry())
        self._settings.sync()
        self.world_view.shutdown()
        try:
            if self.scene_thread.isRunning():
                QMetaObject.invokeMethod(
                    self.scene_worker,
                    "shutdown",
                    Qt.ConnectionType.BlockingQueuedConnection,
                )
                self.scene_thread.quit()
                if not self.scene_thread.wait(2000):
                    LOGGER.warning(
                        "Scene worker did not stop; terminating it"
                    )
                    self.scene_thread.terminate()
                    self.scene_thread.wait(1000)
        except Exception:
            LOGGER.exception("Scene worker shutdown failed")
        try:
            if self.monitor_thread.isRunning():
                QMetaObject.invokeMethod(
                    self.monitor, "stop", Qt.ConnectionType.QueuedConnection
                )
                self.monitor_thread.quit()
                self.monitor_thread.wait(2000)
        except Exception:
            LOGGER.exception("Bridge monitor shutdown failed")
        try:
            if self.worker_thread.isRunning():
                # Queued, not BlockingQueued: beamngpy's socket has no timeout
                # (settimeout(None)), so a simulator stuck in a map load would
                # otherwise hang the GUI thread here forever with no way out.
                QMetaObject.invokeMethod(
                    self.worker, "shutdown", Qt.ConnectionType.QueuedConnection
                )
                self.worker_thread.quit()
                if not self.worker_thread.wait(4000):
                    LOGGER.warning("Worker thread did not stop; terminating it")
                    self.worker_thread.terminate()
                    self.worker_thread.wait(1000)
        except Exception:
            LOGGER.exception("Worker shutdown failed")
        if event is not None:
            event.accept()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            * {
                font-family: "Segoe UI";
                letter-spacing: 0px;
            }
            QMainWindow, QWidget#root {
                background: #0d0f11;
                color: #e8ecef;
            }
            QFrame#sidebar {
                background: #191c1f;
                border-right: 1px solid #30353a;
            }
            QFrame#content {
                background: #0d0f11;
            }
            QFrame#compactControls {
                background: #191c1f;
                border: 1px solid #30353a;
                border-radius: 5px;
            }
            QFrame#viewToggle {
                background: #181c20;
                border: 1px solid #343b42;
                border-radius: 7px;
            }
            QPushButton#viewToggleButton {
                min-width: 62px;
                min-height: 28px;
                max-height: 28px;
                padding: 0 10px;
                border: none;
                border-radius: 5px;
                color: #818a92;
                background: transparent;
                font-size: 10px;
                font-weight: 700;
                text-align: center;
            }
            QPushButton#viewToggleButton:hover {
                color: #dfe4e8;
                background: #252a2f;
            }
            QPushButton#viewToggleButton:checked {
                color: #1c2227;
                background: #eef1f3;
            }
            QPushButton#viewToggleButton:disabled {
                color: #50575d;
                background: transparent;
            }
            QLabel#brand {
                color: #f2f4f5;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#statusBadge {
                border: 1px solid #4a5056;
                border-radius: 5px;
                color: #a6adb3;
                background: #22262a;
                font-size: 11px;
                font-weight: 700;
                padding: 2px 8px;
            }
            QLabel#statusBadge[state="STARTING"],
            QLabel#statusBadge[state="ATTACHING"],
            QLabel#statusBadge[state="CONNECTING"] {
                color: #ffd36a;
                border-color: #7d6931;
                background: #2b271b;
            }
            QLabel#statusBadge[state="READY"],
            QLabel#statusBadge[state="LAUNCHED"] {
                color: #74bfff;
                border-color: #315d7d;
                background: #182630;
            }
            QLabel#statusBadge[state="STREAMING"] {
                color: #62ddb9;
                border-color: #2e745f;
                background: #172a24;
            }
            QLabel#statusBadge[state="ERROR"] {
                color: #ff7b82;
                border-color: #813c42;
                background: #2d1b1e;
            }
            QLabel#statusMessage {
                color: #a7afb6;
                font-size: 12px;
            }
            QLabel#sectionTitle, QLabel#metricTitle {
                color: #777f86;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#pathName {
                color: #d6dade;
                font-size: 12px;
            }
            QLabel#specName {
                color: #8c949b;
                font-size: 11px;
            }
            QLabel#specValue {
                color: #d9dde0;
                font-size: 11px;
                font-weight: 600;
            }
            QFrame#separator {
                background: #30353a;
                border: none;
            }
            QPushButton {
                min-height: 38px;
                border-radius: 5px;
                padding: 0 12px;
                font-size: 12px;
                font-weight: 600;
                text-align: left;
            }
            QPushButton#primaryButton {
                color: #111416;
                background: #eef1f3;
                border: 1px solid #eef1f3;
            }
            QPushButton#primaryButton:hover {
                background: #ffffff;
            }
            QPushButton#attachButton {
                color: #071612;
                background: #53d1b2;
                border: 1px solid #53d1b2;
            }
            QPushButton#attachButton:hover {
                background: #68dfc2;
            }
            QPushButton#stopButton {
                color: #e2e6e9;
                background: #282c30;
                border: 1px solid #4a5056;
            }
            QPushButton#stopButton:hover {
                border-color: #ff626c;
                color: #ff8a91;
            }
            QPushButton#compactActionButton {
                min-width: 40px;
                max-width: 40px;
                min-height: 40px;
                max-height: 40px;
                padding: 0px;
                text-align: center;
                color: #e2e6e9;
                background: #282c30;
                border: 1px solid #4a5056;
            }
            QPushButton#compactActionButton[role="launch"]:hover {
                background: #eef1f3;
                border-color: #eef1f3;
            }
            QPushButton#compactActionButton[role="attach"] {
                background: #53d1b2;
                border-color: #53d1b2;
            }
            QPushButton#compactActionButton[role="attach"]:hover {
                background: #68dfc2;
            }
            QPushButton#compactActionButton[role="stop"]:hover {
                border-color: #ff626c;
            }
            QPushButton#selfDriveButton {
                color: #e2e6e9;
                background: #282c30;
                border: 1px solid #4a5056;
            }
            QPushButton#selfDriveButton:hover {
                border-color: #74bfff;
                color: #9ed2ff;
            }
            QPushButton#selfDriveButton[state="engaged"] {
                color: #0d0f11;
                background: #74bfff;
                border-color: #74bfff;
            }
            QPushButton#selfDriveButton[state="engaged"]:hover {
                background: #8fcbff;
            }
            QPushButton#compactActionButton[role="selfdrive"]:hover {
                border-color: #74bfff;
            }
            QPushButton#compactActionButton[role="selfdrive"][state="engaged"] {
                background: #74bfff;
                border-color: #74bfff;
            }
            QPushButton#aebButton {
                color: #e2e6e9;
                background: #282c30;
                border: 1px solid #4a5056;
            }
            QPushButton#aebButton:hover {
                border-color: #ff626c;
                color: #ff8a91;
            }
            QPushButton#aebButton[state="engaged"] {
                color: #1d0e10;
                background: #ff8a91;
                border-color: #ff8a91;
            }
            QPushButton#aebButton[state="engaged"]:hover {
                background: #ff9ea4;
            }
            QPushButton#compactActionButton[role="aeb"]:hover {
                border-color: #ff626c;
            }
            QPushButton#compactActionButton[role="aeb"][state="engaged"] {
                background: #ff8a91;
                border-color: #ff8a91;
            }
            QPushButton:disabled {
                color: #686f75;
                background: #23272a;
                border-color: #30353a;
            }
            QPushButton#primaryButton:disabled,
            QPushButton#attachButton:disabled,
            QPushButton#stopButton:disabled,
            QPushButton#selfDriveButton:disabled,
            QPushButton#aebButton:disabled,
            QPushButton#compactActionButton:disabled {
                color: #686f75;
                background: #23272a;
                border-color: #30353a;
            }
            QPlainTextEdit#eventLog {
                color: #aeb5bb;
                background: #121517;
                border: 1px solid #30353a;
                border-radius: 4px;
                padding: 7px;
                font-family: "Cascadia Mono", "Consolas";
                font-size: 10px;
                selection-background-color: #355d55;
            }
            QLabel#viewTitle {
                color: #f0f2f4;
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#vehicleLabel, QLabel#legendLabel {
                color: #8e969d;
                font-size: 11px;
            }
            QFrame#statsBand {
                background: #181b1e;
                border: 1px solid #30353a;
                border-radius: 5px;
            }
            QLabel#metricValue {
                color: #e6eaed;
                font-size: 15px;
                font-weight: 650;
            }
            QToolTip {
                color: #f0f2f4;
                background: #24282c;
                border: 1px solid #50565c;
                padding: 5px;
            }
            """
        )
