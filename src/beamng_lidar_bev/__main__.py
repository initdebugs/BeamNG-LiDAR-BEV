from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QMessageBox

from .config import APP_NAME, APP_VERSION
from .main_window import MainWindow


class _QuietBeamNGpy(logging.Filter):
    """
    Drop beamngpy's sub-WARNING chatter at the handler.

    beamngpy's Lidar.__init__ calls setLevel(DEBUG) on its own logger, and an
    explicit level on a child logger beats anything set on the "beamngpy"
    parent -- so raising the parent's level does nothing. A new logger is
    created per sensor, so filtering at the handler is the only durable fix.
    Left unchecked this emitted two records per sensor per frame: ~240
    synchronous file writes a second on the worker thread, and 99.8% of a 7 MB
    log file.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith("beamngpy")


def _configure_logging() -> None:
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_dir / "beamng_lidar_bev.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    ]
    # run_app.bat launches through pyw, where sys.stderr is None and a bare
    # StreamHandler raises AttributeError inside emit() for every record.
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())

    quiet = _QuietBeamNGpy()
    for handler in handlers:
        handler.addFilter(quiet)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> int:
    _configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Local BeamNG Tools")
    app.setFont(QFont("Segoe UI", 10))

    try:
        window = MainWindow()
    except Exception as exc:
        logging.exception("Application startup failed")
        QMessageBox.critical(None, APP_NAME, f"Application startup failed:\n{exc}")
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
