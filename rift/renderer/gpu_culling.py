"""
rift/renderer/gpu_culling.py

GPU-driven frustum culling via a compute shader. This is the standalone
primitive: given an actor bounds buffer, it writes out the visible indices
and a count, entirely on the GPU. Wiring those visible indices into
SpriteBatch so culled sprites actually get skipped is Phase 3 work per
RIFTENGINE_SPECIFICATION.md's own phase breakdown -- actor_manager.py
doesn't exist yet, and that's what's meant to own the "which actors exist
this frame" list this class culls against. Usable standalone in the
meantime for a debug "N of M actors visible" readout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from . import gl_utils
from .shader_compiler import ShaderCompiler

logger = logging.getLogger("rift.renderer.culling")

_ACTOR_STRIDE = 16  # vec4: center.xy, unused, radius
_LOCAL_SIZE_X = 256  # must match `local_size_x` in cull.comp


@dataclass
class Frustum2D:
    """Four half-plane equations, tested as `dot(plane.xy, center) + plane.w
    >= -radius`. Build one of these from your camera each frame; for a
    simple ortho camera the planes are just its world-space AABB edges."""
    planes: np.ndarray  # shape (4, 4) float32, matches the std140 vec4[4] in cull.comp

    @classmethod
    def from_ortho_bounds(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> "Frustum2D":
        planes = np.array([
            [1.0, 0.0, 0.0, -min_x],   # left:   x >= min_x
            [-1.0, 0.0, 0.0, max_x],   # right:  x <= max_x
            [0.0, 1.0, 0.0, -min_y],   # bottom: y >= min_y
            [0.0, -1.0, 0.0, max_y],   # top:    y <= max_y
        ], dtype=np.float32)
        return cls(planes=planes)


class GPUCuller:
    def __init__(self, shader_compiler: ShaderCompiler, compute_src: str, max_actors: int = 100_000):
        self._program = shader_compiler.get_or_compile_compute("gpu_cull", compute_src)
        self._max_actors = max_actors

        self._actor_ssbo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._actor_ssbo, max_actors * _ACTOR_STRIDE, None, GL.GL_DYNAMIC_STORAGE_BIT)

        self._visible_ssbo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._visible_ssbo, max_actors * 4, None, GL.GL_DYNAMIC_STORAGE_BIT)

        self._count_ssbo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._count_ssbo, 4, None, GL.GL_DYNAMIC_STORAGE_BIT)

        self._frustum_ubo = gl_utils.create_gl_object(GL.glCreateBuffers)
        GL.glNamedBufferStorage(self._frustum_ubo, 4 * 16, None, GL.GL_DYNAMIC_STORAGE_BIT)

        self._actor_count = 0

    def upload_actors(self, centers_xy: np.ndarray, radii: np.ndarray) -> None:
        """centers_xy: (N, 2) float32, radii: (N,) float32."""
        n = len(radii)
        if n > self._max_actors:
            raise ValueError(f"{n} actors exceeds GPUCuller max_actors={self._max_actors}")

        packed = np.zeros((n, 4), dtype=np.float32)
        packed[:, 0:2] = centers_xy
        packed[:, 3] = radii
        GL.glNamedBufferSubData(self._actor_ssbo, 0, packed.nbytes, packed)
        self._actor_count = n

    def cull(self, frustum: Frustum2D) -> None:
        if self._actor_count == 0:
            return

        GL.glNamedBufferSubData(self._frustum_ubo, 0, frustum.planes.nbytes, frustum.planes)

        zero = np.zeros(1, dtype=np.uint32)
        GL.glNamedBufferSubData(self._count_ssbo, 0, 4, zero)

        GL.glUseProgram(self._program)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self._actor_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, self._visible_ssbo)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, self._count_ssbo)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self._frustum_ubo)

        group_count = (self._actor_count + _LOCAL_SIZE_X - 1) // _LOCAL_SIZE_X
        GL.glDispatchCompute(group_count, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

    def read_visible_indices(self) -> np.ndarray:
        """Synchronous readback -- fine for a debug counter, NOT fine to call
        every frame in the real render path (that's a GPU->CPU stall, which
        defeats the point of GPU-driven culling). The real consumer of
        visible_ssbo is meant to be an indirect-draw-buffer-generating
        compute pass, entirely GPU-side -- that's the Phase 4 Multi-Draw
        Indirect work called out separately in the spec."""
        count_bytes = GL.glGetNamedBufferSubData(self._count_ssbo, 0, 4)
        count = int(np.frombuffer(count_bytes, dtype=np.uint32)[0])
        if count == 0:
            return np.empty(0, dtype=np.uint32)
        data = GL.glGetNamedBufferSubData(self._visible_ssbo, 0, count * 4)
        return np.frombuffer(data, dtype=np.uint32).copy()

    def cleanup(self) -> None:
        GL.glDeleteBuffers(1, [self._actor_ssbo])
        GL.glDeleteBuffers(1, [self._visible_ssbo])
        GL.glDeleteBuffers(1, [self._count_ssbo])
        GL.glDeleteBuffers(1, [self._frustum_ubo])
