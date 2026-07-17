"""
tools/pack_rfchar.py

Minimal CLI stand-in for RiftCharacterCreator (the real PyQt6 tool from the
spec -- not built yet). This just gets you an actual .rfchar file to test
Character / rfchar_serializer against without waiting on a whole GUI app.

Expected folder layout -- one subfolder per animation slot, PNG frames
inside sorted by filename:

    some_folder/
        idle/
            0001.png
            0002.png
        walk/
            0001.png
            0002.png
            0003.png

Usage:
    python tools/pack_rfchar.py some_folder/ output.rfchar --name Hero --fps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from rift.character.animation_slot import AnimationFrame, AnimationSlot
from rift.character.rfchar_serializer import save_rfchar


def _load_slot(slot_dir: Path, fps: float, looping: bool) -> AnimationSlot:
    frame_paths = sorted(slot_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"{slot_dir} has no .png frames")

    duration_ms = 1000.0 / fps
    frames = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGBA")
        frames.append(AnimationFrame(
            duration_ms=duration_ms,
            rgba_bytes=image.tobytes(),
            width=image.width,
            height=image.height,
        ))

    return AnimationSlot(name=slot_dir.name, speed_fps=fps, looping=looping, frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="folder containing one subfolder per animation slot")
    parser.add_argument("output", type=Path, help="output .rfchar path")
    parser.add_argument("--name", required=True, help="character name")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-loop", action="store_true", help="mark all slots as non-looping")
    parser.add_argument("--width", type=float, default=64.0, help="render size, world units")
    parser.add_argument("--height", type=float, default=64.0)
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise SystemExit(f"{args.source_dir} is not a directory")

    slot_dirs = sorted(d for d in args.source_dir.iterdir() if d.is_dir())
    if not slot_dirs:
        raise SystemExit(f"no animation subfolders found in {args.source_dir}")

    slots = {}
    for slot_dir in slot_dirs:
        slot = _load_slot(slot_dir, args.fps, looping=not args.no_loop)
        slots[slot.name] = slot
        print(f"  {slot.name}: {slot.frame_count} frames")

    save_rfchar(args.output, args.name, (args.width, args.height), (0.0, 0.0), slots)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
