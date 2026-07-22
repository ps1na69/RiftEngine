"""
rift/renderer/bindless_textures.py

ARB_bindless_texture support via ModernGL's Texture.get_handle(resident).
Confirmed behavior (moderngl docs): calling get_handle(resident=True) then
later get_handle(resident=False) on the *same Texture object* toggles
residency and returns the same handle value both times -- so unlike the
PyOpenGL version, there's no bare integer handle to manage independently;
the Texture object itself is the thing whose residency you toggle. This
class therefore holds a reference to the Texture, not just its handle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import moderngl

logger = logging.getLogger("rift.renderer.bindless")


@dataclass
class BindlessHandle:
    texture: moderngl.Texture
    handle: int
    resident: bool = False


class BindlessTextureManager:
    def __init__(self, supported: bool):
        """`supported` comes from gl_utils.probe_bindless_support(ctx),
        run once at renderer startup -- see gl_renderer.py."""
        self.supported = supported
        self._handles: dict[int, BindlessHandle] = {}  # keyed by Texture.glo
        if not self.supported:
            logger.warning(
                "GL_ARB_bindless_texture not available -- sprite_batch has no "
                "fallback path yet, textures will render with a null handle."
            )

    def acquire(self, texture: moderngl.Texture) -> BindlessHandle:
        """Get (creating if needed) a resident bindless handle for a texture.
        Call once after the texture's data is fully uploaded -- per
        ModernGL's docs, a handle's parameters (filter, wrap, etc.) become
        immutable once the handle is first created."""
        existing = self._handles.get(texture.glo)
        if existing is not None:
            return existing

        if not self.supported:
            handle = BindlessHandle(texture=texture, handle=0, resident=False)
            self._handles[texture.glo] = handle
            return handle

        raw_handle = texture.get_handle(resident=True)
        handle = BindlessHandle(texture=texture, handle=raw_handle, resident=True)
        self._handles[texture.glo] = handle
        return handle

    def release(self, texture: moderngl.Texture) -> None:
        handle = self._handles.pop(texture.glo, None)
        if handle is None or not handle.resident:
            return
        texture.get_handle(resident=False)

    def release_all(self) -> None:
        for handle in list(self._handles.values()):
            self.release(handle.texture)

    @staticmethod
    def pack_handle_for_vertex(handle: int) -> tuple[int, int]:
        """Split a 64-bit handle into two 32-bit uints for the uvec2 vertex
        attribute -- sampler2D reconstruction happens GPU-side in
        sprite.frag via sampler2D(uvec2)."""
        return handle & 0xFFFFFFFF, (handle >> 32) & 0xFFFFFFFF
