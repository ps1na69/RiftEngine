"""
sandbox/level_demo.py

Loads a .rflevel and runs it through the real renderer with GPU culling --
first real end-to-end test of ActorManager.cull_and_submit().

Usage:
    python sandbox/level_demo.py my_level.rflevel
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rift.core.application import Application
from rift.level import Level
from rift.renderer.gpu_culling import Frustum2D


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python sandbox/level_demo.py <path_to_rflevel>")
        sys.exit(1)

    level_path = sys.argv[1]
    app = Application(width=1280, height=720, title="RiftEngine - Level Demo")

    state = {"level": None, "frame": 0}

    def on_frame(renderer, delta_time: float) -> None:
        if state["level"] is None:
            state["level"] = Level.load(level_path, renderer.culler, renderer.textures)
            print(f"Loaded '{state['level'].name}': {len(state['level'].actors.actors)} actors")

        level = state["level"]
        level.update(delta_time)

        half_w, half_h = renderer.width / 2.0, renderer.height / 2.0
        frustum = Frustum2D.from_ortho_bounds(-half_w, -half_h, half_w, half_h)
        visible = level.cull_and_submit(renderer.sprite_batch, frustum)

        state["frame"] += 1
        if state["frame"] % 60 == 0:
            print(f"FPS: {renderer.fps:.1f}  visible: {visible}/{len(level.actors.actors)}")

    app.run(on_frame=on_frame)


if __name__ == "__main__":
    main()
