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


# Presets at or below these return EMPTY camera buffers -- a rig that streams
# black frames while every rate reads healthy, which cost a full benchmark
# round to diagnose once already. Compared case-insensitively because the game
# reports them as display strings.
_UNUSABLE_QUALITY = frozenset({"lowest", "very low"})
# Quality keys worth naming individually: a preset of "Custom" says nothing, so
# the individual rungs are what actually have to be checked.
_QUALITY_KEYS = ("overall", "shader", "texture", "lighting")


def capture_setting_warnings(values: dict[str, object]) -> tuple[str, ...]:
    """
    What is wrong with the simulator's renderer settings for CAPTURE.

    Pure so the rules are testable offline; the worker reads the values over
    Lua and logs whatever comes back. This CHECKS rather than SETS, and that is
    deliberate: setting them through `bng.settings.change` + `apply_graphics`
    was measured on 0.39.4 and moved neither the sensor nor the game view, so a
    "pin" would have been a line of code that quietly did nothing. A warning
    that names the setting is worth more than a write that does not land.

    Unknown or unreadable values are NOT warned about -- a missing key means the
    game version does not expose it, which is not evidence of a bad setting.
    """
    warnings: list[str] = []
    for key in _QUALITY_KEYS:
        value = values.get(key)
        if isinstance(value, str) and value.strip().lower() in _UNUSABLE_QUALITY:
            warnings.append(
                f"{key} quality is '{value}' -- at this preset the camera "
                "buffers come back empty (black frames at a healthy frame rate)"
            )
    if values.get("motion_blur") is True:
        warnings.append(
            "PostFXMotionBlurEnabled is on -- it smears every captured frame, "
            "and capture is what this rig exists for"
        )
    if values.get("focused") is False:
        warnings.append(
            "the simulator window is not in the foreground -- when it is "
            "fully covered the renderer throttles to ~2 Hz (measured live "
            "2026-08-23: one 320x240 camera AND a LiDAR unit both delivered "
            "under 2 Hz until the window was visible again). Keep BeamNG "
            "visible, e.g. on a second monitor, while streaming"
        )
    return tuple(warnings)


def start_beamng_process() -> subprocess.Popen[bytes]:
    """
    Start BeamNG.tech immediately without waiting for its TCP bridge.

    **Every standard handle must be given explicitly, and stdout/stderr are
    the ones that matter.** `run_app.bat` starts this GUI with `pyw`, which has
    no console at all -- `sys.stdout` is None and the process's std handles are
    invalid. Inheriting those, the 0.39.4 launcher aborts with
    STATUS_STACK_BUFFER_OVERRUN (0xC0000409) about 0.75 s in, having written a
    zero-byte `beamng-launcher.log` and never spawned the engine. 0.38.5
    tolerated it, which is why this only appeared with the 0.39 upgrade, and it
    is invisible from the app: `Popen` succeeds and returns a pid, so the launch
    looks fine and only the bridge that never opens says otherwise.

    Measured from a windowless parent, spawning the 0.39.4 launcher:

    | stdout/stderr        | outcome                     |
    |----------------------|-----------------------------|
    | inherited            | 0xC0000409 after 0.75 s     |
    | DEVNULL              | boots normally              |
    | CREATE_NEW_CONSOLE   | 0xC0000409 after 0.75 s     |

    Note the middle row is what fixes it and the third is what does not: giving
    the child its own console does not help, because Python still hands the
    parent's invalid handles down. Do not "simplify" these two arguments away.
    """
    command = build_launch_command()
    return subprocess.Popen(
        command,
        cwd=str(BEAMNG_EXE.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
