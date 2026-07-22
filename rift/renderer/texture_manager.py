"""
rift/renderer/texture_manager.py

Texture loading via ctx.texture() -- a single call replaces the old DSA
two-step (glTextureStorage2D + glTextureSubImage2D). dtype='f1' (the
default) means "unsigned byte, normalized 0..1 in-shader," which is
exactly RGBA8 -- matches what Pillow's .convert("RGBA").tobytes() gives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import moderngl
from PIL import Image

from .bindless_textures import BindlessHandle, BindlessTextureManager

logger = logging.getLogger("rift.renderer.textures")


@dataclass
class Texture:
    texture: moderngl.Texture
    width: int
    height: int
    path: str | None
    bindless: BindlessHandle | None = None


class TextureManager:
    """Owns GPU texture storage. `load()` is synchronous today (Phase 1);
    the lazy-loading/streaming behavior described in the spec is a later
    concern once there's an actual asset pipeline driving *when* a texture
    is needed vs merely referenced."""

    def __init__(self, ctx: moderngl.Context, bindless: BindlessTextureManager):
        self.ctx = ctx
        self._bindless = bindless
        self._by_path: dict[str, Texture] = {}
        self._all: list[Texture] = []
        self.gpu_bytes_allocated = 0

    def load(self, path: str | Path) -> Texture:
        path = str(path)
        if cached := self._by_path.get(path):
            return cached

        image = Image.open(path).convert("RGBA")
        width, height = image.size
        data = image.tobytes()

        gl_texture = self.ctx.texture((width, height), 4, data=data)
        gl_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        gl_texture.repeat_x = False
        gl_texture.repeat_y = False

        bindless_handle = self._bindless.acquire(gl_texture)

        texture = Texture(texture=gl_texture, width=width, height=height, path=path, bindless=bindless_handle)
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

        gl_texture = self.ctx.texture((width, height), 4, data=raw_rgba)
        gl_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        bindless_handle = self._bindless.acquire(gl_texture)
        texture = Texture(texture=gl_texture, width=width, height=height, path=None, bindless=bindless_handle)
        self._by_path[name] = texture
        self._all.append(texture)
        self.gpu_bytes_allocated += width * height * 4
        return texture

    def unload(self, texture: Texture) -> None:
        self._bindless.release(texture.texture)
        texture.texture.release()
        self._all.remove(texture)
        if texture.path:
            self._by_path.pop(texture.path, None)
        self.gpu_bytes_allocated -= texture.width * texture.height * 4

    def unload_all(self) -> None:
        for texture in list(self._all):
            self.unload(texture)
