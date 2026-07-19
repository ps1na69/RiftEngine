"""
rift/level/level.py

Level class: owns an ActorManager plus level metadata (name, world size),
and the .rflevel load/save lifecycle. Construction needs a GPUCuller and
TextureManager (both live on GLRenderer) since ActorManager needs them to
actually cull and render actors -- a Level isn't really usable detached
from a running renderer, which matches how the spec's usage example treats
it (`Level.load(...)` after `Application` is already up).
"""

from __future__ import annotations

from pathlib import Path

from rift.renderer.gpu_culling import GPUCuller
from rift.renderer.texture_manager import TextureManager

from .actor_manager import Actor, ActorManager
from .rflevel_serializer import load_rflevel, save_rflevel


class Level:
    def __init__(self, name: str, size: tuple[float, float], culler: GPUCuller, textures: TextureManager):
        self.name = name
        self.size = size
        self.actors = ActorManager(culler, textures)

    @classmethod
    def load(cls, path: str | Path, culler: GPUCuller, textures: TextureManager) -> "Level":
        name, size, actors = load_rflevel(path)
        level = cls(name, size, culler, textures)
        for actor in actors:
            level.actors.add_actor(actor)
        return level

    def save(self, path: str | Path) -> None:
        save_rflevel(path, self.name, self.size, self.actors.actors)

    # -- convenience passthroughs ------------------------------------------

    def spawn(self, actor: Actor) -> None:
        self.actors.add_actor(actor)

    def update(self, delta_time: float) -> None:
        self.actors.update(delta_time)

    def cull_and_submit(self, sprite_batch, frustum) -> int:
        return self.actors.cull_and_submit(sprite_batch, frustum)
