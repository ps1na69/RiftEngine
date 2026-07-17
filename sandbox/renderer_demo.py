"""
sandbox/renderer_demo.py

Smoke test for the renderer: draws a grid of instanced sprites through
SpriteBatch + bindless texture handles. sandbox/main.py still exists and
needs no changes -- Application's public API (constructor + run()) is
unchanged, it just no longer opens a raylib window under the hood.

Usage:
    python -m sandbox.renderer_demo path/to/some_sprite.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `rift` importable regardless of how this script is launched -- see
# the same comment in sandbox/main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rift.core.application import Application
from rift.renderer import SpriteInstance
from rift.renderer.bindless_textures import BindlessTextureManager

GRID_SIZE = 10
SPACING = 80.0


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m sandbox.renderer_demo <path_to_png>")
        sys.exit(1)

    sprite_path = sys.argv[1]
    app = Application(width=1280, height=720, title="RiftEngine - Renderer Demo")

    state = {"packed_handle": None, "frame": 0}

    def on_frame(renderer, delta_time: float) -> None:
        if state["packed_handle"] is None:
            texture = renderer.textures.load(sprite_path)
            state["packed_handle"] = (
                BindlessTextureManager.pack_handle_for_vertex(texture.bindless.handle)
                if texture.bindless and texture.bindless.resident
                else (0, 0)
            )

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = (col - GRID_SIZE / 2) * SPACING
                y = (row - GRID_SIZE / 2) * SPACING
                renderer.sprite_batch.submit(SpriteInstance(
                    x=x, y=y,
                    scale_x=64.0, scale_y=64.0,
                    rotation=0.0,
                    uv_rect=(0.0, 0.0, 1.0, 1.0),
                    tex_handle=state["packed_handle"],
                ))

        state["frame"] += 1
        if state["frame"] % 60 == 0:
            print(f"FPS: {renderer.fps:.1f}  actors: {GRID_SIZE * GRID_SIZE}")

    app.run(on_frame=on_frame)


if __name__ == "__main__":
    main()
