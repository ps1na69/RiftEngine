"""
rift/level/actor_manager.py

GPU-driven actor management with SSBO -- this is where gpu_culling.py (built
standalone back in Phase 1) finally gets fed real data and its output
actually used, instead of just being a debug-readout primitive.

Honest caveat, consistent with gpu_culling.py's own docstring: this uses
GPUCuller.read_visible_indices(), which is a synchronous GPU->CPU readback.
That's fine at the actor counts a 2D RPG level actually has, but it is not
the "zero CPU involvement" GPU-driven pipeline the spec's MDI section
describes -- building an indirect-draw buffer entirely on the GPU (no
readback at all) is explicitly Phase 4 work once sprite_batch.py grows an
MDI draw path to consume it.

Bounding radius: .rflevel doesn't carry an explicit collision/culling
radius per actor (see RIFTENGINE_SPECIFICATION.md's actor field list) --
only position/rotation/scale. This derives a conservative bounding-circle
radius from scale (a box's circumscribed circle: half-diagonal). Good
enough for "is this roughly on screen"; not meant as a precise collision
radius.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from rift.character import Character
from rift.renderer.bindless_textures import BindlessTextureManager
from rift.renderer.gpu_culling import Frustum2D, GPUCuller
from rift.renderer.sprite_batch import SpriteInstance
from rift.renderer.texture_manager import TextureManager

logger = logging.getLogger("rift.level.actors")


@dataclass
class Actor:
    name: str
    actor_type: str  # "Character", "Sprite", "Light", ...
    position: tuple[float, float]
    rotation: float
    scale: tuple[float, float]
    resource_path: str
    custom_data: dict = field(default_factory=dict)

    # Runtime-only, not part of .rflevel -- populated by ActorManager on add.
    character: "Character | None" = None

    def bounding_radius(self) -> float:
        half_w, half_h = self.scale[0] / 2.0, self.scale[1] / 2.0
        return math.hypot(half_w, half_h)


class ActorManager:
    def __init__(self, culler: GPUCuller, textures: TextureManager):
        self._culler = culler
        self._textures = textures
        self.actors: list[Actor] = []
        self._bounds_dirty = True

    # -- population ----------------------------------------------------------

    def add_actor(self, actor: Actor) -> None:
        if actor.actor_type == "Character" and actor.character is None and actor.resource_path:
            try:
                actor.character = Character.load(actor.resource_path)
                default_anim = actor.custom_data.get("default_animation")
                if default_anim is None and actor.character.slots:
                    default_anim = next(iter(actor.character.slots))
                if default_anim:
                    actor.character.play_animation(default_anim)
            except (OSError, ValueError) as exc:
                logger.warning("Actor '%s': failed to load %s (%s) -- will not render",
                                actor.name, actor.resource_path, exc)

        self.actors.append(actor)
        self._bounds_dirty = True

    def remove_actor(self, actor: Actor) -> None:
        self.actors.remove(actor)
        self._bounds_dirty = True

    def clear(self) -> None:
        self.actors.clear()
        self._bounds_dirty = True

    # -- per-frame -------------------------------------------------------------

    def update(self, delta_time: float) -> None:
        for actor in self.actors:
            if actor.character is not None:
                actor.character.update(delta_time)

    def _upload_bounds_if_dirty(self) -> None:
        if not self._bounds_dirty:
            return
        if self.actors:
            centers = np.array([a.position for a in self.actors], dtype=np.float32)
            radii = np.array([a.bounding_radius() for a in self.actors], dtype=np.float32)
        else:
            centers = np.empty((0, 2), dtype=np.float32)
            radii = np.empty((0,), dtype=np.float32)
        self._culler.upload_actors(centers, radii)
        self._bounds_dirty = False

    def cull_and_submit(self, sprite_batch, frustum: Frustum2D) -> int:
        """Culls on the GPU, reads back which indices survived, and submits
        a SpriteInstance for each visible actor. Returns the visible count
        (handy for an on-screen "N/M actors visible" debug readout).
        Call after ActorManager.update() and before sprite_batch is
        expected to be flushed for the frame."""
        if not self.actors:
            return 0

        self._upload_bounds_if_dirty()
        self._culler.cull(frustum)
        visible_indices = self._culler.read_visible_indices()

        for idx in visible_indices:
            actor = self.actors[int(idx)]
            instance = self._sprite_instance_for(actor)
            if instance is not None:
                sprite_batch.submit(instance)

        return len(visible_indices)

    def _sprite_instance_for(self, actor: Actor) -> SpriteInstance | None:
        if actor.actor_type == "Character":
            if actor.character is None:
                return None
            return actor.character.get_sprite_instance(
                self._textures, actor.position[0], actor.position[1], rotation=actor.rotation,
            )

        if actor.actor_type == "Sprite":
            if not actor.resource_path:
                return None
            texture = self._textures.load(actor.resource_path)
            handle = (
                BindlessTextureManager.pack_handle_for_vertex(texture.bindless.handle)
                if texture.bindless and texture.bindless.resident
                else (0, 0)
            )
            return SpriteInstance(
                x=actor.position[0], y=actor.position[1],
                scale_x=actor.scale[0], scale_y=actor.scale[1],
                rotation=actor.rotation,
                uv_rect=(0.0, 0.0, 1.0, 1.0),
                tex_handle=handle,
            )

        # "Light" and anything else: not a sprite, no draw instance (yet --
        # a real lighting pass is well beyond Phase 3's scope).
        return None
