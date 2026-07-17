"""
sandbox/character_demo.py

Loads a .rfchar file and plays one of its animations on screen through the
real renderer -- end-to-end test of tools/pack_rfchar.py -> rfchar_serializer
-> Character -> SpriteBatch.

Usage:
    python sandbox/character_demo.py hero.rfchar walk
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rift.character import Character
from rift.core.application import Application


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python sandbox/character_demo.py <path_to_rfchar> <animation_name>")
        sys.exit(1)

    rfchar_path, animation_name = sys.argv[1], sys.argv[2]
    character = Character.load(rfchar_path)
    print(f"Loaded '{character.name}', animations: {', '.join(character.slots)}")
    character.play_animation(animation_name)

    app = Application(width=1280, height=720, title=f"RiftEngine - {character.name}")

    def on_frame(renderer, delta_time: float) -> None:
        character.update(delta_time)
        instance = character.get_sprite_instance(renderer.textures, 0.0, 0.0, scale=4.0)
        if instance is not None:
            renderer.sprite_batch.submit(instance)

    app.run(on_frame=on_frame)


if __name__ == "__main__":
    main()
