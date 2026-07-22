"""
rift/renderer/sprite_batch.py

Instanced sprite batching. Per your decision to drop persistent mapped
buffers (not in ModernGL's public API -- see migration notes), this now
writes each frame's instance data via Buffer.write(), rotating across 3
separate Buffer objects (and therefore 3 separate VertexArray objects,
since ModernGL VAOs bind specific Buffer objects at creation time and
there's no documented way to repoint one at a different buffer afterward).
The rotation still avoids the classic single-buffer stall -- frame N+1's
write() targets a buffer that frame N-2's draw call finished with two
frames ago -- it just relies on the driver's normal buffer-update
synchronization instead of manual fences, since ModernGL doesn't expose
fences/memory barriers either.

MDI hook point, for Phase 4 (not implemented here, per your "design with
MDI in mind" ask): ModernGL's VertexArray.render_indirect(buffer, mode,
count, first) is real and confirmed working -- it consumes a buffer of
5-int draw commands (count, instanceCount, firstIndex, baseVertex,
baseInstance). The natural Phase 4 shape is a sibling method here, e.g.
`flush_indirect(indirect_buffer, draw_count)`, called instead of flush()
once gpu_culling.py's compute pass writes indirect commands directly
instead of ActorManager reading back visible indices on the CPU. flush()
below is deliberately the single place the draw call happens, so that
swap is localized to this file when the time comes.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

import moderngl
import numpy as np

logger = logging.getLogger("rift.renderer.sprite_batch")

# xy, uv for a unit quad centered on the pivot
_QUAD_VERTS = np.array([
    -0.5, -0.5, 0.0, 0.0,
     0.5, -0.5, 1.0, 0.0,
     0.5,  0.5, 1.0, 1.0,
    -0.5,  0.5, 0.0, 1.0,
], dtype=np.float32)
_QUAD_INDICES = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

# Per-instance layout: pos(2f) scale(2f) rotation(1f) uv_rect(4f) tex_handle(2u4)
# = 9 floats + 2 uint32 = 44 bytes. Keep in sync with the ModernGL format
# string below AND the struct.pack format in flush() AND sprite.vert's
# `in_*` locations -- same three-places-in-sync caveat as the PyOpenGL
# version, unavoidable without a shared schema description.
_INSTANCE_STRIDE = 44
_MAX_INSTANCES_PER_FRAME = 65536
_BUFFER_SLICE_COUNT = 3  # rotation depth, replaces the old fence-synced triple buffer

_INSTANCE_FORMAT = "2f 2f 1f 4f 2u4 /i"
_INSTANCE_ATTRS = (
    "in_instance_pos", "in_instance_scale", "in_instance_rotation",
    "in_instance_uv_rect", "in_instance_tex_handle",
)


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
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program):
        self.ctx = ctx
        self._program = program

        self._quad_vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._ebo = ctx.buffer(_QUAD_INDICES.tobytes())

        self._instance_vbos = [
            ctx.buffer(reserve=_INSTANCE_STRIDE * _MAX_INSTANCES_PER_FRAME, dynamic=True)
            for _ in range(_BUFFER_SLICE_COUNT)
        ]
        self._vaos = [
            ctx.vertex_array(
                program,
                [
                    (self._quad_vbo, "2f 2f /v", "in_position", "in_uv"),
                    (instance_vbo, _INSTANCE_FORMAT, *_INSTANCE_ATTRS),
                ],
                index_buffer=self._ebo,
            )
            for instance_vbo in self._instance_vbos
        ]

        self._pending: list[SpriteInstance] = []
        self._slice_index = 0

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
        count = len(self._pending)
        if count == 0:
            return

        slice_index = self._slice_index
        packed = bytearray(count * _INSTANCE_STRIDE)
        offset = 0
        for inst in self._pending:
            struct.pack_into(
                "<2f2f1f4f2I", packed, offset,
                inst.x, inst.y,
                inst.scale_x, inst.scale_y,
                inst.rotation,
                *inst.uv_rect,
                *inst.tex_handle,
            )
            offset += _INSTANCE_STRIDE

        self._instance_vbos[slice_index].write(bytes(packed))
        self._vaos[slice_index].render(mode=moderngl.TRIANGLES, instances=count)

        self._slice_index = (slice_index + 1) % _BUFFER_SLICE_COUNT

    def cleanup(self) -> None:
        for vao in self._vaos:
            vao.release()
        for vbo in self._instance_vbos:
            vbo.release()
        self._quad_vbo.release()
        self._ebo.release()
