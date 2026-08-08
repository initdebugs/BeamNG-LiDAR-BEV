from __future__ import annotations

import socket
import subprocess
from pathlib import Path

from .config import BEAMNG_EXE, BEAMNG_HOST, BEAMNG_PORT


def build_launch_command(
    executable: Path = BEAMNG_EXE,
    host: str = BEAMNG_HOST,
    port: int = BEAMNG_PORT,
) -> list[str]:
    """Build the BeamNGpy-compatible BeamNG.tech command line."""
    return [
        str(executable),
        "-nosteam",
        "-tcom",
        "-tport",
        str(port),
        "-console",
        "-tcom-listen-ip",
        host,
    ]


def start_beamng_process() -> subprocess.Popen[bytes]:
    """Start BeamNG.tech immediately without waiting for its TCP bridge."""
    command = build_launch_command()
    return subprocess.Popen(
        command,
        cwd=str(BEAMNG_EXE.parent),
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def bridge_is_reachable(
    host: str = BEAMNG_HOST,
    port: int = BEAMNG_PORT,
    timeout_s: float = 0.5,
) -> bool:
    """Check bridge readiness without entering BeamNGpy's blocking handshake."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
