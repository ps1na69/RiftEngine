"""
rift/character/animation_slot.py

Animation slot metadata + frame data for the .rfchar format. A "slot" is
one named animation (e.g. "walk", "idle", "attack") -- a sequence of PNG
frames played back at a fixed speed with optional looping.

Deliberately NOT skeletal/bone animation: the .rfchar binary format
described in RIFTENGINE_SPECIFICATION.md is a per-frame PNG sequence
format, not a rig, so the "GPU skeletal blending" language from the
Character System section doesn't have an asset format to operate on.
This implements what the format actually specifies; skeletal animation
would need its own format extension, which is a bigger, separate design
decision worth making explicitly rather than half-building here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnimationFrame:
    duration_ms: float
    rgba_bytes: bytes  # decompressed RGBA8 pixel data, width*height*4 bytes
    width: int
    height: int


@dataclass
class AnimationSlot:
    name: str
    speed_fps: float
    looping: bool
    frames: list[AnimationFrame] = field(default_factory=list)
    icon: bytes | None = None  # raw PNG bytes, for editor thumbnails

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def total_duration_ms(self) -> float:
        return sum(f.duration_ms for f in self.frames)
