"""
Does a click in WORLD land where the labeller thinks it does?

The bay labeller turns a screen click into a world XY through three steps, and
only the FIRST has no offline proof: `View3D.pickAll` is the only thing that
knows this camera's projection, so what it returns can be measured and cannot
be reasoned about. (CLAUDE.md, twice over: pixel questions get measured.)

This drives the real `WorldScene.qml` on the real GPU with the same synthetic
street `preview_world.py` builds, fires `pickGroundPoint` at a grid of screen
positions, and prints the BEV metres that come back. Two things it proves:

  * a click on the road returns a point, and the point moves the right way --
    right of centre is +right, low on screen is near, high on screen is far
  * a click on the sky returns NOTHING, so the labeller cannot drop a corner
    on the horizon -- and neither does a click on the ego's own footprint,
    which is a genuine hole in the road mesh (`outside_ego_body` culls every
    return inside the body, so no ground is ever accumulated under the car)

    py -3.12 tools/label_pick_probe.py

Needs a desktop session (the offscreen platform plugin is not QRhi-capable, so
it can compile the QML but never raycast it). Does NOT need BeamNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QPointF, QTimer
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from preview_world import GEOMETRY, build_scene  # noqa: E402

from beamng_lidar_bev.models import PerceptionSnapshot  # noqa: E402
from beamng_lidar_bev.world_scene import WorldSceneAssembler  # noqa: E402
from beamng_lidar_bev.world_view import WorldView  # noqa: E402

WIDTH, HEIGHT = 1280, 800
# Screen fractions to click at, and whether each must land on the ground. The
# centre column walks away down the road; the outer two step off to the sides.
PROBES = [
    ("far centre", 0.50, 0.45, True),
    ("mid centre", 0.50, 0.60, True),
    ("near left", 0.35, 0.78, True),
    ("near right", 0.65, 0.78, True),
    # Both of these MUST miss. The sky is obvious; the ego footprint is not --
    # it is a real hole in the surface, and a labeller that snapped a corner
    # onto the car would be worse than one that refused.
    ("sky", 0.50, 0.08, False),
    ("under the car", 0.50, 0.78, False),
]


def main() -> int:
    app = QApplication(sys.argv)
    view = WorldView()
    view.resize(WIDTH, HEIGHT)
    if not view.is_ready:
        print("QML FAILED:", view.failure_message)
        return 2
    view.show()

    hits: list[tuple[float, float]] = []
    view.bridge.ground_picked.connect(
        lambda right, forward: hits.append((right, forward))
    )

    points, groups = build_scene()
    assembler = WorldSceneAssembler()
    frame = None
    for tick in range(6):
        frame = assembler.update(
            PerceptionSnapshot(
                points_world=points,
                semantic_groups=groups,
                ego_pos_world=(0.0, 0.0, 0.0),
                ego_dir_world=(0.0, 1.0, 0.0),
                ego_up_world=(0.0, 0.0, 1.0),
                timestamp=tick * 0.04,
                speed_mps=0.0,
                forward_speed_mps=0.0,
                vehicle_geometry=GEOMETRY,
            )
        )
    view.set_frame(frame)

    def probe() -> None:
        print(f"{'probe':<14} {'screen':<15} {'right m':>9} {'forward m':>10}")
        seen: dict[str, tuple[float, float] | None] = {}
        failures: list[str] = []
        for name, fx, fy, must_hit in PROBES:
            before = len(hits)
            view._pick_ground(QPointF(fx * WIDTH, fy * HEIGHT))
            got = hits[before:]
            seen[name] = got[-1] if got else None
            shown = (
                f"{got[-1][0]:>9.2f} {got[-1][1]:>10.2f}"
                if got
                else f"{'(no hit)':>20}"
            )
            print(f"{name:<14} ({fx:.2f}, {fy:.2f}){'':<4} {shown}")
            if bool(got) != must_hit:
                failures.append(
                    f"{name}: {'missed' if must_hit else 'hit'} unexpectedly"
                )

        # The signs and the ordering, which is what a labeller actually
        # depends on: a corner clicked left of the car must not be recorded to
        # its right, and a far corner must not be recorded as a near one.
        far, mid = seen["far centre"], seen["mid centre"]
        left, right = seen["near left"], seen["near right"]
        if far and mid and not far[1] > mid[1] > 0.0:
            failures.append("forward does not grow up the screen")
        if left and right and not left[0] < 0.0 < right[0]:
            failures.append("right is mirrored")
        if left and right and abs(abs(left[0]) - abs(right[0])) > 0.2:
            failures.append("the two sides are not symmetric")

        for failure in failures:
            print("  FAIL", failure)
        print("FAILED" if failures else "OK -- the ground picks cleanly")
        view.shutdown()
        app.quit()

    # The same settle the other probes use: QML has to have rendered once
    # before the raycast has a scene to hit.
    QTimer.singleShot(2500, probe)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
