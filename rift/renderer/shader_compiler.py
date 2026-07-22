"""
rift/renderer/shader_compiler.py

Shader/program creation via ModernGL's ctx.program() / ctx.compute_shader(),
which compile and link in one call and raise moderngl.Error (with the GLSL
compiler/linker log embedded in the message) on failure -- there's no
separate compile-then-link step to wrap the way the old PyOpenGL version
did, so the old ShaderCompileError/ShaderLinkError distinction collapses
into a single error type here.

Dropped relative to the PyOpenGL version: the on-disk program-binary cache.
ModernGL doesn't expose glGetProgramBinary/glProgramBinary (or any
equivalent) in its public API -- Program has no method for extracting or
reinjecting a compiled binary. What's kept is an in-memory cache keyed by
source hash, so re-requesting the same shader within a session (e.g. two
SpriteBatch instances, or a hot-reload path later) doesn't recompile it,
but there's no cross-session warm start anymore. If that turns out to
matter in practice, the escape hatch would be dropping to raw GL via
Program.glo for just that piece -- not done here since it wasn't asked for.
"""

from __future__ import annotations

import hashlib
import logging

import moderngl

logger = logging.getLogger("rift.renderer.shaders")


class ShaderError(RuntimeError):
    """Wraps a moderngl.Error from program/compute-shader creation with
    which shader this was, since ModernGL's own exception doesn't say."""

    def __init__(self, name: str, original: Exception):
        super().__init__(f"shader '{name}' failed to compile/link:\n{original}")
        self.name = name
        self.original = original


class ShaderCompiler:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._program_cache: dict[str, moderngl.Program] = {}
        self._compute_cache: dict[str, moderngl.ComputeShader] = {}

    @staticmethod
    def _hash_sources(*sources: str) -> str:
        h = hashlib.sha256()
        for s in sources:
            h.update(s.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def get_or_compile(self, name: str, vertex_src: str, fragment_src: str) -> moderngl.Program:
        key = self._hash_sources(vertex_src, fragment_src)
        if program := self._program_cache.get(key):
            return program

        try:
            program = self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        except moderngl.Error as exc:
            raise ShaderError(name, exc) from exc

        self._program_cache[key] = program
        logger.info("Compiled shader program '%s' (%s)", name, key)
        return program

    def get_or_compile_compute(self, name: str, compute_src: str) -> moderngl.ComputeShader:
        key = self._hash_sources(compute_src)
        if program := self._compute_cache.get(key):
            return program

        try:
            program = self.ctx.compute_shader(compute_src)
        except moderngl.Error as exc:
            raise ShaderError(name, exc) from exc

        self._compute_cache[key] = program
        logger.info("Compiled compute program '%s' (%s)", name, key)
        return program

    def cleanup(self) -> None:
        for program in self._program_cache.values():
            program.release()
        for program in self._compute_cache.values():
            program.release()
        self._program_cache.clear()
        self._compute_cache.clear()
