"""
rift/renderer/gl_utils.py

Capability reporting for the ModernGL context. Much thinner than the old
PyOpenGL version -- ModernGL doesn't expose extension enumeration
(glGetStringi) or the KHR_debug async callback at all in its public API,
so this module can't offer either. What it *can* do:

- Report ctx.version_code (e.g. 460 for GL 4.6) and dump ctx.info wholesale
  for logging, rather than guessing at specific dict key names I haven't
  independently verified.
- Empirically probe bindless-texture support by actually attempting
  Texture.get_handle() on a throwaway 1x1 texture and seeing if it raises,
  rather than checking an extension string -- this is arguably more
  reliable anyway (a driver can advertise an extension string and still
  have a broken implementation) and it's the only option ModernGL's API
  leaves available.

Lost relative to the PyOpenGL version: automatic async GL debug logging
(GL_DEBUG_OUTPUT / glDebugMessageCallback). ModernGL only exposes a
synchronous `ctx.error` poll (equivalent to glGetError), not a callback.
check_gl_error() below is a manual call-it-yourself substitute, same as
it always was -- the thing that's actually gone is the automatic stream
of driver warnings/errors the old debug callback gave you for free.
"""

from __future__ import annotations

import logging

import moderngl

logger = logging.getLogger("rift.renderer")

REQUIRED_MIN_VERSION_CODE = 460  # GL 4.6, matches ctx.version_code's encoding


def log_capabilities(ctx: moderngl.Context) -> None:
    """Call once, right after context creation."""
    logger.info("GL version_code=%d", ctx.version_code)
    logger.info("GL info: %s", dict(ctx.info))
    if ctx.version_code < REQUIRED_MIN_VERSION_CODE:
        logger.warning(
            "GL context is version_code=%d, engine targets %d (GL 4.6) -- "
            "bindless textures may be unavailable.",
            ctx.version_code, REQUIRED_MIN_VERSION_CODE,
        )


def probe_bindless_support(ctx: moderngl.Context) -> bool:
    """Empirically test GL_ARB_bindless_texture support by actually trying
    to make a throwaway texture's handle resident. Returns False (and logs
    a warning) on any failure rather than raising -- callers should treat
    that as "no bindless available" and are responsible for deciding what
    that means for them (sprite_batch.py currently has no non-bindless
    fallback path, same caveat as the PyOpenGL version had)."""
    probe_texture = ctx.texture((1, 1), 4, data=bytes(4))
    try:
        handle = probe_texture.get_handle(resident=True)
        probe_texture.get_handle(resident=False)
        logger.info("Bindless textures supported (probe handle=%d)", handle)
        return True
    except Exception as exc:
        logger.warning("Bindless textures NOT supported on this driver/context: %s", exc)
        return False
    finally:
        probe_texture.release()


def check_gl_error(ctx: moderngl.Context, context: str = "") -> None:
    """Cheap error probe for suspect call sites during development. Don't
    call every frame -- ctx.error forces a sync point like glGetError does."""
    err = ctx.error
    if err and err != "GL_NO_ERROR":
        raise RuntimeError(f"GL error {err} at {context or '<unknown>'}")
