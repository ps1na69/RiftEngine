"""
rift/character/character.py

Runtime character: owns a set of AnimationSlots (loaded from a .rfchar
file), tracks current playback state, and hands back render-ready sprite
data each frame. Each animation frame is uploaded as its own individual
bindless texture on first display -- fine for the frame counts a
hand-drawn 2D RPG character actually has. A shared texture atlas (fewer
resident bindless handles, better cache behavior) is a real optimization
worth doing later, but adds a packing step this doesn't need yet.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rift.renderer.bindless_textures import BindlessTextureManager
from rift.renderer.sprite_batch import SpriteInstance
from rift.renderer.texture_manager import TextureManager

from .animation_slot import AnimationSlot
from .rfchar_serializer import load_rfchar

logger = logging.getLogger("rift.character")


class Character:
    def __init__(
        self,
        name: str,
        size: tuple[float, float],
        offset: tuple[float, float],
        slots: dict[str, AnimationSlot],
    ):
        self.name = name
        self.size = size
        self.offset = offset
        self.slots = slots

        self._current_slot_name: str | None = None
        self._frame_index = 0
        self._elapsed_ms = 0.0
        self._finished = False  # true once a non-looping animation reaches its last frame

        # (slot_name, frame_index) -> packed bindless handle, populated
        # lazily as frames are actually displayed.
        self._frame_handles: dict[tuple[str, int], tuple[int, int]] = {}

    @classmethod
    def load(cls, path: str | Path) -> "Character":
        name, size, offset, slots = load_rfchar(path)
        return cls(name, size, offset, slots)

    # -- playback -----------------------------------------------------------

    @property
    def current_animation(self) -> str | None:
        return self._current_slot_name

    @property
    def is_finished(self) -> bool:
        """True once a non-looping animation has reached its last frame.
        Always False for looping animations and while nothing is playing."""
        return self._finished

    def play_animation(self, name: str, *, restart: bool = False) -> None:
        if name not in self.slots:
            logger.warning("Character '%s' has no animation slot '%s' (has: %s)",
                            self.name, name, ", ".join(self.slots))
            return
        if self._current_slot_name == name and not restart:
            return
        self._current_slot_name = name
        self._frame_index = 0
        self._elapsed_ms = 0.0
        self._finished = False

    def stop_animation(self) -> None:
        self._current_slot_name = None
        self._frame_index = 0
        self._elapsed_ms = 0.0
        self._finished = False

    def update(self, delta_time: float) -> None:
        if self._current_slot_name is None or self._finished:
            return
        slot = self.slots[self._current_slot_name]
        if not slot.frames:
            return

        self._elapsed_ms += delta_time * 1000.0

        def _current_frame_duration() -> float:
            explicit = slot.frames[self._frame_index].duration_ms
            return explicit if explicit > 0 else (1000.0 / max(slot.speed_fps, 0.001))

        # A `while`, not `if` -- keeps animation correct even if delta_time
        # briefly spikes (e.g. a hitch) past more than one frame's duration.
        while self._elapsed_ms >= _current_frame_duration():
            self._elapsed_ms -= _current_frame_duration()
            self._frame_index += 1
            if self._frame_index >= slot.frame_count:
                if slot.looping:
                    self._frame_index = 0
                else:
                    self._frame_index = slot.frame_count - 1
                    self._finished = True
                    break

    # -- rendering ------------------------------------------------------------

    def get_sprite_instance(
        self,
        textures: TextureManager,
        x: float,
        y: float,
        *,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> SpriteInstance | None:
        """Returns None if nothing is currently playing -- caller decides
        whether that means "don't draw" or "draw a default pose"."""
        if self._current_slot_name is None:
            return None
        slot = self.slots[self._current_slot_name]
        if not slot.frames:
            return None

        frame = slot.frames[self._frame_index]
        key = (self._current_slot_name, self._frame_index)

        handle = self._frame_handles.get(key)
        if handle is None:
            texture_name = f"{self.name}:{self._current_slot_name}:{self._frame_index}"
            texture = textures.load_from_bytes(texture_name, frame.rgba_bytes, frame.width, frame.height)
            handle = (
                BindlessTextureManager.pack_handle_for_vertex(texture.bindless.handle)
                if texture.bindless and texture.bindless.resident
                else (0, 0)
            )
            self._frame_handles[key] = handle

        return SpriteInstance(
            x=x + self.offset[0], y=y + self.offset[1],
            scale_x=self.size[0] * scale, scale_y=self.size[1] * scale,
            rotation=rotation,
            uv_rect=(0.0, 0.0, 1.0, 1.0),
            tex_handle=handle,
        )
