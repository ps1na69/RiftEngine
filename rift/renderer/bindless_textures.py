"""
rift/renderer/bindless_textures.py

ARB_bindless_texture support. Bindless handles let sprite_batch.py pack a
texture reference straight into per-instance vertex data (as a uvec2)
instead of juggling texture units, which is what makes batching many
different textures through one instanced draw call possible at all.

Falls back cleanly (supported=False, handle=0) on hardware/drivers without
the extension -- see gl_utils.GLCapabilities.supports_bindless_textures.
Phase 1 does not implement a non-bindless fallback rendering path in
sprite_batch.py itself; on unsupported hardware sprites will draw with an
invalid handle. Worth circling back to before shipping on anything other
than a dev machine you control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from OpenGL.GL.ARB import bindless_texture as _bindless

logger = logging.getLogger("rift.renderer.bindless")


@dataclass
class BindlessHandle:
    texture_id: int
    handle: int
    resident: bool = False


class BindlessTextureManager:
    """Tracks which texture handles are currently GPU-resident. Handles must
    stay resident for as long as any draw call might reference them -- this
    class exists to keep that lifetime explicit instead of scattered across
    texture_manager.py call sites."""

    def __init__(self, capabilities) -> None:
        self.supported = capabilities.supports_bindless_textures
        self._handles: dict[int, BindlessHandle] = {}
        if not self.supported:
            logger.warning(
                "GL_ARB_bindless_texture not available -- sprite_batch has no "
                "fallback path yet, textures will render with a null handle."
            )

    def acquire(self, texture_id: int) -> BindlessHandle:
        """Get (creating if needed) a resident bindless handle for a texture.
        Call once after the texture's storage + data are fully uploaded;
        creating a handle before the final glTextureSubImage* call is legal
        per spec but leaves driver caching behavior murkier than it needs
        to be, so we don't rely on it."""
        existing = self._handles.get(texture_id)
        if existing is not None:
            return existing

        if not self.supported:
            handle = BindlessHandle(texture_id=texture_id, handle=0, resident=False)
            self._handles[texture_id] = handle
            return handle

        raw_handle = _bindless.glGetTextureHandleARB(texture_id)
        _bindless.glMakeTextureHandleResidentARB(raw_handle)
        handle = BindlessHandle(texture_id=texture_id, handle=raw_handle, resident=True)
        self._handles[texture_id] = handle
        return handle

    def release(self, texture_id: int) -> None:
        handle = self._handles.pop(texture_id, None)
        if handle is None or not handle.resident:
            return
        _bindless.glMakeTextureHandleNonResidentARB(handle.handle)

    def release_all(self) -> None:
        for texture_id in list(self._handles):
            self.release(texture_id)

    @staticmethod
    def pack_handle_for_vertex(handle: int) -> tuple[int, int]:
        """Split a 64-bit handle into two 32-bit uints for the uvec2 vertex
        attribute -- sampler2D reconstruction happens GPU-side in
        sprite.frag via sampler2D(uvec2)."""
        return handle & 0xFFFFFFFF, (handle >> 32) & 0xFFFFFFFF
