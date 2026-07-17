"""
rift/renderer/shader_compiler.py

GLSL compilation + optional on-disk program binary caching
(GL_ARB_get_program_binary, core since GL 4.1), keyed by a hash of the
source. Caching matters once the editor/character-creator start spinning up
GL contexts repeatedly during a session -- recompiling the cull compute
shader from scratch every time is wasted work.

Note: glGetProgramBinary/glProgramBinary's exact calling convention can
differ slightly between PyOpenGL versions. Both cache paths here are
wrapped in try/except and fail soft into a normal recompile, so a signature
mismatch degrades to "cache doesn't work" rather than a crash -- worth
confirming against your installed PyOpenGL version, since this couldn't be
exercised against a real GPU in the environment that generated it.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
from pathlib import Path

from OpenGL import GL

logger = logging.getLogger("rift.renderer.shaders")

_SHADER_TYPE_NAMES = {
    GL.GL_VERTEX_SHADER: "vertex",
    GL.GL_FRAGMENT_SHADER: "fragment",
    GL.GL_COMPUTE_SHADER: "compute",
    GL.GL_GEOMETRY_SHADER: "geometry",
}


class ShaderCompileError(RuntimeError):
    def __init__(self, stage: str, log: str):
        super().__init__(f"{stage} shader failed to compile:\n{log}")
        self.stage = stage
        self.log = log


class ShaderLinkError(RuntimeError):
    def __init__(self, log: str):
        super().__init__(f"program link failed:\n{log}")
        self.log = log


class ShaderCompiler:
    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._program_cache: dict[str, int] = {}

    # -- compiling individual stages --------------------------------------

    def _compile_stage(self, source: str, stage: int) -> int:
        shader = GL.glCreateShader(stage)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)

        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            if isinstance(log, bytes):
                log = log.decode("utf-8", errors="replace")
            GL.glDeleteShader(shader)
            raise ShaderCompileError(_SHADER_TYPE_NAMES.get(stage, str(stage)), log)

        return shader

    def _link(self, *shaders: int) -> int:
        program = GL.glCreateProgram()
        for shader in shaders:
            GL.glAttachShader(program, shader)
        GL.glLinkProgram(program)

        for shader in shaders:
            GL.glDetachShader(program, shader)
            GL.glDeleteShader(shader)

        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program)
            if isinstance(log, bytes):
                log = log.decode("utf-8", errors="replace")
            GL.glDeleteProgram(program)
            raise ShaderLinkError(log)

        return program

    # -- cache key ----------------------------------------------------------

    @staticmethod
    def _hash_sources(*sources: str) -> str:
        h = hashlib.sha256()
        for s in sources:
            h.update(s.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _load_cached_binary(self, key: str) -> int | None:
        if not self.cache_dir:
            return None
        blob_path = self.cache_dir / f"{key}.bin"
        meta_path = self.cache_dir / f"{key}.fmt"
        if not (blob_path.exists() and meta_path.exists()):
            return None

        try:
            fmt = int(meta_path.read_text())
            data = blob_path.read_bytes()
            program = GL.glCreateProgram()
            GL.glProgramBinary(program, fmt, data, len(data))
            if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
                GL.glDeleteProgram(program)
                return None
            logger.debug("Loaded cached shader binary %s", key)
            return program
        except Exception:
            logger.debug("Cached shader binary %s unusable, recompiling", key, exc_info=True)
            return None

    def _store_cached_binary(self, key: str, program: int) -> None:
        if not self.cache_dir:
            return
        try:
            length = GL.glGetProgramiv(program, GL.GL_PROGRAM_BINARY_LENGTH)
            if length <= 0:
                return
            fmt = GL.GLenum(0)
            buf = (GL.GLubyte * length)()
            written = GL.GLsizei(0)
            GL.glGetProgramBinary(program, length, ctypes.byref(written), ctypes.byref(fmt), buf)
            (self.cache_dir / f"{key}.bin").write_bytes(bytes(buf[: written.value]))
            (self.cache_dir / f"{key}.fmt").write_text(str(fmt.value))
        except Exception:
            logger.debug("Failed to persist shader binary cache for %s", key, exc_info=True)

    # -- public API ----------------------------------------------------------

    def get_or_compile(self, name: str, vertex_src: str, fragment_src: str) -> int:
        key = self._hash_sources(vertex_src, fragment_src)
        if program := self._program_cache.get(key):
            return program

        program = self._load_cached_binary(key)
        if program is None:
            vs = self._compile_stage(vertex_src, GL.GL_VERTEX_SHADER)
            fs = self._compile_stage(fragment_src, GL.GL_FRAGMENT_SHADER)
            program = self._link(vs, fs)
            self._store_cached_binary(key, program)
            logger.info("Compiled shader program '%s' (%s)", name, key)

        self._program_cache[key] = program
        return program

    def get_or_compile_compute(self, name: str, compute_src: str) -> int:
        key = self._hash_sources(compute_src)
        if program := self._program_cache.get(key):
            return program

        program = self._load_cached_binary(key)
        if program is None:
            cs = self._compile_stage(compute_src, GL.GL_COMPUTE_SHADER)
            program = self._link(cs)
            self._store_cached_binary(key, program)
            logger.info("Compiled compute program '%s' (%s)", name, key)

        self._program_cache[key] = program
        return program

    @staticmethod
    def load_source(path: Path | str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def cleanup(self) -> None:
        for program in self._program_cache.values():
            GL.glDeleteProgram(program)
        self._program_cache.clear()
