"""
rift/renderer/sprite_batch.py

Instanced sprite batching backed by a persistently mapped, triple-buffered
instance VBO. The "triple buffered" part matters: with GL_MAP_PERSISTENT_BIT
the CPU pointer stays valid across frames, which means nothing stops you
from overwriting instance data the GPU is still reading from a draw call
issued a frame or two ago. A fence per slice keeps the CPU from writing into
a region until the GPU has actually finished consuming it.

Draws via glDrawElementsInstanced for now rather than the MDI path the spec
calls out -- Multi-Draw Indirect is explicitly a later phase (needs an
indirect-draw-buffer-generating compute pass, which needs gpu_culling.py
wired to a real actor list first). This still gets every sprite in one
draw call per frame, which is the part that actually matters for Phase 1.
"""

from __future__ import annotations

import ctypes
import logging
import struct
from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from . import gl_utils
from .shader_compiler import ShaderCompiler

logger = logging.getLogger("rift.renderer.sprite_batch")

# xy, uv for a unit quad centered on the pivot
_QUAD_VERTS = np.array([
    -0.5, -0.5, 0.0, 0.0,
     0.5, -0.5, 1.0, 0.0,
     0.5,  0.5, 1.0, 1.0,
    -0.5,  0.5, 0.0, 1.0,
], dtype=np.float32)
_QUAD_INDICES = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

# Per-instance layout: pos(2f) scale(2f) rotation(1f) uv_rect(4f) tex_handle(2u)
# = 9 floats + 2 uint32 = 11 * 4 bytes = 44 bytes. Keep this in sync with the
# attribute offsets in _setup_instance_buffer() AND the struct.pack format
# in flush() AND the `in_*` locations in sprite.vert -- there's no single
# source of truth for this layout across Python/GLSL, unfortunately.
_INSTANCE_STRIDE = 44
_MAX_INSTANCES_PER_FRAME = 65536
_BUFFER_SLICE_COUNT = 3  # triple buffering


@dataclass
class SpriteInstance:
    x: float
    y: float
    scale_x: float
    scale_y: float
    rotation: float
    uv_rect: tuple[float, float, float, float]
    tex_handle: tuple[int, int]


class SpriteBatch:
    def __init__(self, shader_compiler: ShaderCompiler, vertex_src: str, fragment_src: str):
        self._compiler = shader_compiler
        self._program = shader_compiler.get_or_compile("sprite_batch", vertex_src, fragment_src)

        self._vao = gl_utils.create_gl_object(GL.glCreateVertexArrays)
        self._setup_static_geometry()
        self._setup_instance_buffer()

        self._pending: list[SpriteInstance] = []
        self._slice_index = 0
        self._fences: list[object | None] = [None] * _BUFFER_SLICE_COUNT

    # -- setup ----------------------------------------------------------

    def _setup_static_geometry(self) -> None:
        self._quad_vbo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._quad_vbo, _QUAD_VERTS.nbytes, _QUAD_VERTS, 0)

        self._ebo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._ebo, _QUAD_INDICES.nbytes, _QUAD_INDICES, 0)

        GL.glVertexArrayVertexBuffer(self._vao, 0, self._quad_vbo, 0, 4 * 4)
        GL.glVertexArrayElementBuffer(self._vao, self._ebo)

        GL.glEnableVertexArrayAttrib(self._vao, 0)  # in_position
        GL.glVertexArrayAttribFormat(self._vao, 0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0)
        GL.glVertexArrayAttribBinding(self._vao, 0, 0)

        GL.glEnableVertexArrayAttrib(self._vao, 1)  # in_uv
        GL.glVertexArrayAttribFormat(self._vao, 1, 2, GL.GL_FLOAT, GL.GL_FALSE, 2 * 4)
        GL.glVertexArrayAttribBinding(self._vao, 1, 0)

    def _setup_instance_buffer(self) -> None:
        total_size = _INSTANCE_STRIDE * _MAX_INSTANCES_PER_FRAME * _BUFFER_SLICE_COUNT
        flags = GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT | GL.GL_MAP_COHERENT_BIT

        self._instance_vbo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._instance_vbo, total_size, None, flags)
        raw_ptr = GL.glMapNamedBufferRange(self._instance_vbo, 0, total_size, flags)

        # Treat the whole mapped region as one big addressable byte array so
        # writes are plain memmove-at-offset instead of typed pointer
        # arithmetic -- simpler to get right than juggling a POINTER(c_float).
        byte_array_type = ctypes.c_ubyte * total_size
        self._instance_buf = ctypes.cast(raw_ptr, ctypes.POINTER(byte_array_type)).contents

        GL.glVertexArrayVertexBuffer(self._vao, 1, self._instance_vbo, 0, _INSTANCE_STRIDE)
        GL.glVertexArrayBindingDivisor(self._vao, 1, 1)  # advance once per instance, not per vertex

        GL.glEnableVertexArrayAttrib(self._vao, 2)  # in_instance_pos (vec2)
        GL.glVertexArrayAttribFormat(self._vao, 2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0)
        GL.glVertexArrayAttribBinding(self._vao, 2, 1)

        GL.glEnableVertexArrayAttrib(self._vao, 3)  # in_instance_scale (vec2)
        GL.glVertexArrayAttribFormat(self._vao, 3, 2, GL.GL_FLOAT, GL.GL_FALSE, 8)
        GL.glVertexArrayAttribBinding(self._vao, 3, 1)

        GL.glEnableVertexArrayAttrib(self._vao, 4)  # in_instance_rotation (float)
        GL.glVertexArrayAttribFormat(self._vao, 4, 1, GL.GL_FLOAT, GL.GL_FALSE, 16)
        GL.glVertexArrayAttribBinding(self._vao, 4, 1)

        GL.glEnableVertexArrayAttrib(self._vao, 5)  # in_instance_uv_rect (vec4)
        GL.glVertexArrayAttribFormat(self._vao, 5, 4, GL.GL_FLOAT, GL.GL_FALSE, 20)
        GL.glVertexArrayAttribBinding(self._vao, 5, 1)

        GL.glEnableVertexArrayAttrib(self._vao, 6)  # in_instance_tex_handle (uvec2)
        GL.glVertexArrayAttribIFormat(self._vao, 6, 2, GL.GL_UNSIGNED_INT, 36)
        GL.glVertexArrayAttribBinding(self._vao, 6, 1)

    # -- per-frame API ----------------------------------------------------

    def begin(self) -> None:
        self._pending.clear()

    def submit(self, instance: SpriteInstance) -> None:
        if len(self._pending) >= _MAX_INSTANCES_PER_FRAME:
            logger.warning(
                "SpriteBatch overflow (%d instances) -- dropping sprite. "
                "Raise _MAX_INSTANCES_PER_FRAME or add multi-flush support.",
                _MAX_INSTANCES_PER_FRAME,
            )
            return
        self._pending.append(instance)

    def flush(self) -> None:
        """Upload pending instances into this frame's slice and draw."""
        count = len(self._pending)
        if count == 0:
            return

        slice_index = self._slice_index
        self._wait_for_slice(slice_index)

        slice_byte_offset = slice_index * _MAX_INSTANCES_PER_FRAME * _INSTANCE_STRIDE
        for i, inst in enumerate(self._pending):
            offset = slice_byte_offset + i * _INSTANCE_STRIDE
            packed = struct.pack(
                "<2f2f1f4f2I",
                inst.x, inst.y,
                inst.scale_x, inst.scale_y,
                inst.rotation,
                *inst.uv_rect,
                *inst.tex_handle,
            )
            ctypes.memmove(ctypes.byref(self._instance_buf, offset), packed, len(packed))

        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)
        GL.glBindVertexBuffer(1, self._instance_vbo, slice_byte_offset, _INSTANCE_STRIDE)

        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(_QUAD_INDICES), GL.GL_UNSIGNED_INT, None, count)

        self._fences[slice_index] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        self._slice_index = (slice_index + 1) % _BUFFER_SLICE_COUNT

    def _wait_for_slice(self, slice_index: int) -> None:
        fence = self._fences[slice_index]
        if fence is None:
            return
        # A 1s timeout is generous on purpose -- if we're actually waiting
        # that long, something upstream is badly stalled and silently
        # hanging here would just hide it.
        GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000)
        GL.glDeleteSync(fence)
        self._fences[slice_index] = None

    def cleanup(self) -> None:
        GL.glUnmapNamedBuffer(self._instance_vbo)
        self._instance_buf = None
        GL.glDeleteBuffers(1, [self._instance_vbo])
        GL.glDeleteBuffers(1, [self._quad_vbo])
        GL.glDeleteBuffers(1, [self._ebo])
        GL.glDeleteVertexArrays(1, [self._vao])
