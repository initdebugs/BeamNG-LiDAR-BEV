from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from unittest import mock

from beamng_lidar_bev import launcher
from beamng_lidar_bev.launcher import (
    bridge_is_reachable,
    build_launch_command,
    capture_setting_warnings,
)


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


def test_the_spawn_never_inherits_the_parents_standard_handles() -> None:
    """
    `run_app.bat` starts the GUI with `pyw`, which has no console: its std
    handles are invalid. Measured against BeamNG.tech 0.39.4, a launcher that
    inherits them aborts with 0xC0000409 after 0.75 s, writes a zero-byte
    launcher log and never spawns the engine -- while `Popen` returns a pid, so
    the app sees a successful launch and waits 300 s for a bridge that cannot
    come. Every handle is therefore passed explicitly.
    """
    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured.update(kwargs)

    with mock.patch.object(launcher.subprocess, "Popen", _FakePopen):
        launcher.start_beamng_process()

    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.DEVNULL


def test_a_healthy_capture_setup_warns_about_nothing() -> None:
    """The measured-good configuration on this machine must stay silent."""
    assert capture_setting_warnings(
        {
            "overall": "Custom",
            "shader": "High",
            "texture": "Normal",
            "lighting": "High",
            "aa": 4,
            "motion_blur": False,
        }
    ) == ()


def test_the_preset_that_returns_black_frames_is_named() -> None:
    """
    The trap this exists for: at the lowest presets the camera buffers come
    back EMPTY while every rate reads healthy, so the rig looks like it works.
    """
    warnings = capture_setting_warnings({"shader": "Lowest", "motion_blur": False})

    assert len(warnings) == 1
    assert "shader" in warnings[0]
    assert "empty" in warnings[0]


def test_the_preset_check_is_case_insensitive_and_covers_every_rung() -> None:
    """'Custom' overall says nothing, so each rung is checked on its own."""
    warnings = capture_setting_warnings(
        {
            "overall": "Custom",
            "shader": "lowest",
            "texture": "Very Low",
            "lighting": "High",
        }
    )

    assert len(warnings) == 2
    assert any("shader" in w for w in warnings)
    assert any("texture" in w for w in warnings)


def test_motion_blur_is_warned_about_because_capture_is_the_point() -> None:
    warnings = capture_setting_warnings({"shader": "High", "motion_blur": True})

    assert len(warnings) == 1
    # The warning must name the setting to change, verbatim -- "motion blur" in
    # prose is not something anyone can search the game's options for.
    assert "PostFXMotionBlurEnabled" in warnings[0]


def test_an_unreadable_setting_is_never_warned_about() -> None:
    """
    A key a game version does not expose is not evidence of a bad setting, and
    warning on it would train the reader to ignore the line.
    """
    assert capture_setting_warnings({}) == ()
    assert capture_setting_warnings({"shader": None, "motion_blur": None}) == ()
    assert capture_setting_warnings({"shader": 0, "motion_blur": "false"}) == ()
