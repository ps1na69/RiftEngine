"""
rift/renderer/texture_manager.py

Texture loading using DSA (glCreateTextures / glTextureStorage2D /
glTextureSubImage2D), so nothing here ever needs a bound texture unit.
Bindless residency is handled separately by
bindless_textures.BindlessTextureManager -- this module owns *storage*,
that one owns *residency*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from OpenGL import GL
from PIL import Image

from . import gl_utils
from .bindless_textures import BindlessHandle, BindlessTextureManager

logger = logging.getLogger("rift.renderer.textures")


@dataclass
class Texture:
    id: int
    width: int
    height: int
    path: str | None
    bindless: BindlessHandle | None = None


class TextureManager:
    """Owns GPU texture storage. `load()` is synchronous today (Phase 1);
    the lazy-loading/streaming behavior described in the spec is a later
    concern once there's an actual asset pipeline driving *when* a texture
    is needed vs merely referenced."""

    def __init__(self, bindless: BindlessTextureManager):
        self._bindless = bindless
        self._by_path: dict[str, Texture] = {}
        self._all: list[Texture] = []
        self.gpu_bytes_allocated = 0

    def load(self, path: str | Path, *, srgb: bool = False) -> Texture:
        path = str(path)
        if cached := self._by_path.get(path):
            return cached

        image = Image.open(path).convert("RGBA")
        width, height = image.size
        data = image.tobytes()

        texture_id = gl_utils.create_gl_object(GL.glCreateTextures, GL.GL_TEXTURE_2D)
        internal_format = GL.GL_SRGB8_ALPHA8 if srgb else GL.GL_RGBA8

        GL.glTextureStorage2D(texture_id, 1, internal_format, width, height)
        GL.glTextureSubImage2D(texture_id, 0, 0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        bindless_handle = self._bindless.acquire(texture_id)

        texture = Texture(id=texture_id, width=width, height=height, path=path, bindless=bindless_handle)
        self._by_path[path] = texture
        self._all.append(texture)
        self.gpu_bytes_allocated += width * height * 4
        logger.debug("Loaded texture %s (%dx%d)", path, width, height)
        return texture

    def load_from_bytes(self, name: str, raw_rgba: bytes, width: int, height: int) -> Texture:
        """For .rfchar frame data, which arrives as already-decompressed
        RGBA bytes in memory rather than a path on disk."""
        if cached := self._by_path.get(name):
            return cached

        texture_id = gl_utils.create_gl_object(GL.glCreateTextures, GL.GL_TEXTURE_2D)
        GL.glTextureStorage2D(texture_id, 1, GL.GL_RGBA8, width, height)
        GL.glTextureSubImage2D(texture_id, 0, 0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, raw_rgba)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTextureParameteri(texture_id, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

        bindless_handle = self._bindless.acquire(texture_id)
        texture = Texture(id=texture_id, width=width, height=height, path=None, bindless=bindless_handle)
        self._by_path[name] = texture
        self._all.append(texture)
        self.gpu_bytes_allocated += width * height * 4
        return texture

    def unload(self, texture: Texture) -> None:
        self._bindless.release(texture.id)
        GL.glDeleteTextures(1, [texture.id])
        self._all.remove(texture)
        if texture.path:
            self._by_path.pop(texture.path, None)
        self.gpu_bytes_allocated -= texture.width * texture.height * 4

    def unload_all(self) -> None:
        for texture in list(self._all):
            self.unload(texture)
