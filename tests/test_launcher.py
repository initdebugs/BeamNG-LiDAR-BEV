from __future__ import annotations

import socket
from pathlib import Path

from beamng_lidar_bev.launcher import bridge_is_reachable, build_launch_command


def test_builds_beamngpy_compatible_launch_command() -> None:
    executable = Path(r"C:\BeamNG.tech\BeamNG.tech.exe")

    command = build_launch_command(executable, "127.0.0.1", 64256)

    assert command == [
        str(executable),
        "-nosteam",
        "-tcom",
        "-tport",
        "64256",
        "-console",
        "-tcom-listen-ip",
        "127.0.0.1",
    ]


def test_bridge_readiness_uses_bounded_tcp_probe() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        assert bridge_is_reachable("127.0.0.1", port, timeout_s=0.1)

    assert not bridge_is_reachable("127.0.0.1", port, timeout_s=0.1)
