"""
tools/pack_rflevel.py

CLI stand-in for the real level editor (Outliner + viewport, PyQt6 --
Phase 3's editor UI, not built yet). Scatters N actors of one type across
a grid so there's something real to test Level / ActorManager / GPU
culling against.

Usage:
    python tools/pack_rflevel.py output.rflevel --name TestLevel \
        --actor-type Sprite --resource some_sprite.png \
        --grid 10x10 --spacing 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rift.level import Actor, save_rflevel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--actor-type", choices=["Sprite", "Character"], default="Sprite")
    parser.add_argument("--resource", required=True, help=".png for Sprite, .rfchar for Character")
    parser.add_argument("--grid", default="10x10", help="WIDTHxHEIGHT actor count, e.g. 10x10")
    parser.add_argument("--spacing", type=float, default=80.0)
    parser.add_argument("--actor-scale", type=float, default=64.0)
    args = parser.parse_args()

    cols, rows = (int(v) for v in args.grid.lower().split("x"))

    actors = []
    for row in range(rows):
        for col in range(cols):
            x = (col - cols / 2) * args.spacing
            y = (row - rows / 2) * args.spacing
            actors.append(Actor(
                name=f"{args.actor_type}_{row}_{col}",
                actor_type=args.actor_type,
                position=(x, y),
                rotation=0.0,
                scale=(args.actor_scale, args.actor_scale),
                resource_path=args.resource,
                custom_data={},
            ))

    level_size = (cols * args.spacing, rows * args.spacing)
    save_rflevel(args.output, args.name, level_size, actors)
    print(f"Wrote {args.output}: {len(actors)} actors, level size {level_size}")


if __name__ == "__main__":
    main()
