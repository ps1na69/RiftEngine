"""
rift/renderer/gl_utils.py

Low-level OpenGL 4.6 utilities: extension detection, debug callback setup,
and basic error checking. Nothing here holds engine state -- these are pure
helpers used by gl_renderer.py and friends.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field

from OpenGL import GL

logger = logging.getLogger("rift.renderer")

REQUIRED_MIN_VERSION = (4, 6)

# DSA is core as of 4.5 and compute shaders are core as of 4.3, so on a real
# 4.6 context those two are guaranteed present -- they're still listed below
# so get_gl_capabilities() gives a complete picture, and so the documented
# 4.5 fallback path has something concrete to inspect.
WATCHED_EXTENSIONS = (
    "GL_ARB_bindless_texture",
    "GL_ARB_direct_state_access",
    "GL_ARB_multi_draw_indirect",
    "GL_ARB_compute_shader",
    "GL_ARB_buffer_storage",       # persistent mapped buffers
    "GL_ARB_shader_storage_buffer_object",
)


@dataclass
class GLCapabilities:
    vendor: str
    renderer: str
    version: str
    glsl_version: str
    version_tuple: tuple[int, int]
    extensions: dict[str, bool] = field(default_factory=dict)

    @property
    def supports_bindless_textures(self) -> bool:
        return self.extensions.get("GL_ARB_bindless_texture", False)

    @property
    def supports_indirect_draw(self) -> bool:
        return self.extensions.get("GL_ARB_multi_draw_indirect", False) or self.version_tuple >= (4, 3)

    @property
    def meets_minimum(self) -> bool:
        return self.version_tuple >= REQUIRED_MIN_VERSION


def _query_extensions() -> set[str]:
    """Core-profile-safe extension enumeration. glGetString(GL_EXTENSIONS)
    (the single space-separated string) is removed in core profiles -- you
    have to use the indexed glGetStringi form instead."""
    count = GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)
    found = set()
    for i in range(count):
        name = GL.glGetStringi(GL.GL_EXTENSIONS, i)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        found.add(name)
    return found


def get_gl_capabilities() -> GLCapabilities:
    """Call once, right after the context is made current, before doing
    anything else that depends on extension support."""
    version_str = GL.glGetString(GL.GL_VERSION).decode("utf-8")
    major = GL.glGetIntegerv(GL.GL_MAJOR_VERSION)
    minor = GL.glGetIntegerv(GL.GL_MINOR_VERSION)

    available = _query_extensions()
    extensions = {name: name in available for name in WATCHED_EXTENSIONS}

    caps = GLCapabilities(
        vendor=GL.glGetString(GL.GL_VENDOR).decode("utf-8"),
        renderer=GL.glGetString(GL.GL_RENDERER).decode("utf-8"),
        version=version_str,
        glsl_version=GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode("utf-8"),
        version_tuple=(major, minor),
        extensions=extensions,
    )

    if not caps.meets_minimum:
        logger.warning(
            "GL context is %d.%d, engine targets %d.%d -- "
            "bindless textures and indirect draw will be disabled.",
            major, minor, *REQUIRED_MIN_VERSION,
        )
    missing = [name for name, ok in extensions.items() if not ok]
    if missing:
        logger.warning("Missing optional GL extensions: %s", ", ".join(missing))

    return caps


# --- Debug callback -----------------------------------------------------

_DEBUG_SEVERITY_NAMES = {
    GL.GL_DEBUG_SEVERITY_HIGH: "HIGH",
    GL.GL_DEBUG_SEVERITY_MEDIUM: "MEDIUM",
    GL.GL_DEBUG_SEVERITY_LOW: "LOW",
    GL.GL_DEBUG_SEVERITY_NOTIFICATION: "NOTIFICATION",
}

# Module-level reference to the ctypes callback closure. If this gets
# garbage collected, the driver ends up calling into freed memory the next
# time it wants to log something -- keep it alive for the process lifetime.
_debug_callback_ref = None


def enable_debug_output(*, break_on_error: bool = False) -> None:
    """Route GL_KHR_debug messages to the Python logger instead of silence."""
    global _debug_callback_ref

    GL.glEnable(GL.GL_DEBUG_OUTPUT)
    GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)

    def _callback(source, type_, msg_id, severity, length, message, user_param):
        text = ctypes.string_at(message, length).decode("utf-8", errors="replace")
        severity_name = _DEBUG_SEVERITY_NAMES.get(severity, str(severity))
        if severity == GL.GL_DEBUG_SEVERITY_NOTIFICATION:
            logger.debug("[GL/%s] %s", severity_name, text)
        else:
            logger.warning("[GL/%s] (id=%d) %s", severity_name, msg_id, text)
            if break_on_error and severity in (GL.GL_DEBUG_SEVERITY_HIGH, GL.GL_DEBUG_SEVERITY_MEDIUM):
                raise RuntimeError(f"GL debug message [{severity_name}]: {text}")

    _debug_callback_ref = GL.GLDEBUGPROC(_callback)
    GL.glDebugMessageCallback(_debug_callback_ref, None)
    GL.glDebugMessageControl(GL.GL_DONT_CARE, GL.GL_DONT_CARE, GL.GL_DONT_CARE, 0, None, GL.GL_TRUE)


def check_gl_error(context: str = "") -> None:
    """Cheap error probe for suspect call sites during development. Don't
    call this every frame in a release build -- glGetError forces a sync
    point that will quietly eat your GPU-driven pipeline's performance."""
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:
        raise RuntimeError(f"GL error 0x{err:04X} at {context or '<unknown>'}")


# --- DSA object creation --------------------------------------------------
#
# The real C signature for every DSA "create" function is
# glCreateXxx(..., GLsizei n, GLuint *arrays) -- an output array parameter.
# Some PyOpenGL/PyOpenGL_accelerate builds auto-generate that array for you
# and let you call e.g. glCreateBuffers(1) for a return value; others (per
# a very concrete Windows traceback) insist on the raw signature and reject
# the single-argument form outright. Always using the explicit array form
# here sidesteps the discrepancy -- it's just the literal C signature, so
# it's valid regardless of which wrapping behavior a given install has.

def create_gl_objects(create_func, count: int, *prefix_args) -> list[int]:
    ids = (GL.GLuint * count)()
    create_func(*prefix_args, count, ids)
    return list(ids)


def create_gl_object(create_func, *prefix_args) -> int:
    """Convenience for the common count=1 case, e.g.:
    vao = gl_utils.create_gl_object(GL.glCreateVertexArrays)
    tex = gl_utils.create_gl_object(GL.glCreateTextures, GL.GL_TEXTURE_2D)
    """
    return create_gl_objects(create_func, 1, *prefix_args)[0]
