"""
rift/renderer/gl_renderer.py

Top-level renderer: creates the ModernGL context from the GLFW context
Window already made current, and wires together ShaderCompiler /
TextureManager / SpriteBatch / GPUCuller. begin_frame()/end_frame() drive
Application's game loop, same shape as before.

Window itself needed NO changes for this migration -- moderngl.create_context()
just wraps whatever GL context GLFW already made current, exactly like the
raw PyOpenGL calls did. The migration is entirely inside rift/renderer/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import moderngl
import numpy as np

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
    UBO via Buffer.write() -- same layout requirement as before, ModernGL's
    write() is a raw memcpy just like glNamedBufferSubData was. World
    origin sits at the center of the screen."""
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

        self.ctx = moderngl.create_context(require=460)
        gl_utils.log_capabilities(self.ctx)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.viewport = (0, 0, self.width, self.height)

        self.shader_compiler = ShaderCompiler(self.ctx)

        bindless_supported = gl_utils.probe_bindless_support(self.ctx)
        self.bindless = BindlessTextureManager(bindless_supported)
        self.textures = TextureManager(self.ctx, self.bindless)

        sprite_program = self.shader_compiler.get_or_compile(
            "sprite_batch",
            (_SHADER_DIR / "sprite.vert").read_text(),
            (_SHADER_DIR / "sprite.frag").read_text(),
        )
        self.sprite_batch = SpriteBatch(self.ctx, sprite_program)
        self.culler = GPUCuller(self.ctx, self.shader_compiler, (_SHADER_DIR / "cull.comp").read_text())

        self._camera_ubo = self.ctx.buffer(reserve=64, dynamic=True)
        self._update_camera()

    def _on_resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.ctx.viewport = (0, 0, width, height)
        self._update_camera()

    def _update_camera(self) -> None:
        matrix = _build_ortho_matrix(float(self.width), float(self.height))
        self._camera_ubo.write(matrix.tobytes())
        self._camera_ubo.bind_to_uniform_block(0)

    # -- frame loop --------------------------------------------------------

    def begin_frame(self) -> None:
        now = time.perf_counter()
        self.delta_time = now - self._last_frame_time
        self._last_frame_time = now
        self.fps = 1.0 / self.delta_time if self.delta_time > 0 else 0.0

        self.ctx.clear(0.06, 0.06, 0.08, 1.0)
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
        self._camera_ubo.release()
