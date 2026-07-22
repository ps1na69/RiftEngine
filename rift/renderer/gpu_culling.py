"""
rift/renderer/gpu_culling.py

GPU-driven frustum culling via a compute shader. Standalone primitive, same
scope note as before: wiring visible indices into SpriteBatch so culled
sprites actually get skipped is what actor_manager.py (Phase 3) does;
this class doesn't know about actors, just bounds-in/indices-out.

Correctness note specific to this ModernGL migration: the compute shader
writes to SSBOs that are then read back on the CPU via Buffer.read().
ModernGL exposes no memory_barrier()/glMemoryBarrier equivalent anywhere
in its public API (checked Context, Buffer, and ComputeShader -- none of
them have it). ctx.finish() (a full pipeline stall, equivalent to
glFinish) is used here as the correctness-preserving substitute before
every readback. This is heavier than a targeted GL_SHADER_STORAGE_BARRIER
would have been, but read_visible_indices() was already a synchronous
CPU stall in the PyOpenGL version too (documented there as a Phase-4-will-
fix-this-properly item) -- this doesn't make an already-synchronous path
meaningfully worse, it just makes it correct under ModernGL's API surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import moderngl
import numpy as np

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
    def __init__(self, ctx: moderngl.Context, shader_compiler: ShaderCompiler, compute_src: str, max_actors: int = 100_000):
        self.ctx = ctx
        self._program = shader_compiler.get_or_compile_compute("gpu_cull", compute_src)
        self._max_actors = max_actors

        self._actor_ssbo = ctx.buffer(reserve=max_actors * _ACTOR_STRIDE, dynamic=True)
        self._visible_ssbo = ctx.buffer(reserve=max_actors * 4, dynamic=True)
        self._count_ssbo = ctx.buffer(reserve=4, dynamic=True)
        self._frustum_ubo = ctx.buffer(reserve=4 * 16, dynamic=True)

        self._actor_count = 0

    def upload_actors(self, centers_xy: np.ndarray, radii: np.ndarray) -> None:
        """centers_xy: (N, 2) float32, radii: (N,) float32."""
        n = len(radii)
        if n > self._max_actors:
            raise ValueError(f"{n} actors exceeds GPUCuller max_actors={self._max_actors}")

        packed = np.zeros((n, 4), dtype=np.float32)
        packed[:, 0:2] = centers_xy
        packed[:, 3] = radii
        self._actor_ssbo.write(packed.tobytes())
        self._actor_count = n

    def cull(self, frustum: Frustum2D) -> None:
        if self._actor_count == 0:
            return

        self._frustum_ubo.write(frustum.planes.tobytes())
        self._count_ssbo.write(np.zeros(1, dtype=np.uint32).tobytes())

        self._actor_ssbo.bind_to_storage_buffer(0)
        self._visible_ssbo.bind_to_storage_buffer(1)
        self._count_ssbo.bind_to_storage_buffer(2)
        self._frustum_ubo.bind_to_uniform_block(1)

        group_count = (self._actor_count + _LOCAL_SIZE_X - 1) // _LOCAL_SIZE_X
        self._program.run(group_count, 1, 1)
        self.ctx.finish()  # see module docstring -- substitutes for the missing memory_barrier

    def read_visible_indices(self) -> np.ndarray:
        """Already a synchronous readback even before the ctx.finish() this
        migration added -- see module docstring. Fine for a debug counter,
        NOT fine to call every frame in the real render path. The real
        consumer of visible_ssbo is meant to be an indirect-draw-buffer,
        entirely GPU-side -- Phase 4 MDI work, see sprite_batch.py's
        flush_indirect() hook note."""
        count_bytes = self._count_ssbo.read(4)
        count = int(np.frombuffer(count_bytes, dtype=np.uint32)[0])
        if count == 0:
            return np.empty(0, dtype=np.uint32)
        data = self._visible_ssbo.read(count * 4)
        return np.frombuffer(data, dtype=np.uint32).copy()

    def cleanup(self) -> None:
        self._actor_ssbo.release()
        self._visible_ssbo.release()
        self._count_ssbo.release()
        self._frustum_ubo.release()
