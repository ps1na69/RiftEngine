"""
rift/renderer/gl_renderer.py

GPU-resource + draw orchestration layer: shader compiler, texture manager,
sprite batch, GPU culler, frame timing. Assumes an OpenGL 4.6 core context
is already current -- that's rift.core.window.Window's job (window.create()
must run before this is constructed). This class deliberately never touches
GLFW directly: windowing and rendering are two different concerns now that
raylib (which used to blur that line) is gone from the engine.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from OpenGL import GL

from . import gl_utils
from .shader_compiler import ShaderCompiler
from .bindless_textures import BindlessTextureManager
from .texture_manager import TextureManager
from .sprite_batch import SpriteBatch
from .gpu_culling import GPUCuller

logger = logging.getLogger("rift.renderer")

_SHADER_DIR = Path(__file__).parent / "shaders"


def _build_ortho_matrix(width: float, height: float) -> np.ndarray:
    """Column-major mat4, flattened for a direct byte-copy into a std140
    UBO (no glUniformMatrix transpose flag involved here -- this goes in
    via glNamedBufferSubData, so the memory layout has to already be right).
    World origin sits at the center of the screen."""
    left, right = -width / 2.0, width / 2.0
    bottom, top = -height / 2.0, height / 2.0
    near, far = -1.0, 1.0

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m.flatten(order="F")


class GLRenderer:
    def __init__(self, window) -> None:
        """`window` is a rift.core.window.Window whose create() has already
        run -- i.e. a GL 4.6 context is current on this thread."""
        self.width = window.width
        self.height = window.height
        window.on_resize(self._on_resize)

        self._frame_count = 0
        self._last_frame_time = time.perf_counter()
        self.delta_time = 0.0
        self.fps = 0.0

        self.capabilities = gl_utils.get_gl_capabilities()
        gl_utils.enable_debug_output()
        logger.info(
            "GL %s | %s | %s", self.capabilities.version, self.capabilities.renderer, self.capabilities.vendor
        )
        if not self.capabilities.meets_minimum:
            logger.warning("Context is below OpenGL 4.6 -- bindless textures will be unavailable.")

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.06, 0.06, 0.08, 1.0)
        GL.glViewport(0, 0, self.width, self.height)

        self.shader_compiler = ShaderCompiler(cache_dir=Path.home() / ".riftengine" / "shader_cache")
        self.bindless = BindlessTextureManager(self.capabilities)
        self.textures = TextureManager(self.bindless)

        self.sprite_batch = SpriteBatch(
            self.shader_compiler,
            (_SHADER_DIR / "sprite.vert").read_text(),
            (_SHADER_DIR / "sprite.frag").read_text(),
        )
        self.culler = GPUCuller(self.shader_compiler, (_SHADER_DIR / "cull.comp").read_text())

        self._camera_ubo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._camera_ubo, 64, None, GL.GL_DYNAMIC_STORAGE_BIT)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self._camera_ubo)
        self._update_camera()

    def _on_resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        GL.glViewport(0, 0, width, height)
        self._update_camera()

    def _update_camera(self) -> None:
        matrix = _build_ortho_matrix(float(self.width), float(self.height))
        GL.glNamedBufferSubData(self._camera_ubo, 0, matrix.nbytes, matrix)

    # -- frame loop --------------------------------------------------------

    def begin_frame(self) -> None:
        now = time.perf_counter()
        self.delta_time = now - self._last_frame_time
        self._last_frame_time = now
        self.fps = 1.0 / self.delta_time if self.delta_time > 0 else 0.0

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.sprite_batch.begin()

    def end_frame(self) -> None:
        self.sprite_batch.flush()
        self._frame_count += 1

    def close(self) -> None:
        self.sprite_batch.cleanup()
        self.culler.cleanup()
        self.textures.unload_all()
        self.bindless.release_all()
        self.shader_compiler.cleanup()
        GL.glDeleteBuffers(1, [self._camera_ubo])
