"""
rift/core/window.py

GLFW + OpenGL 4.6 core-profile window. Replaces the earlier raylib-backed
Window -- raylib manages its own GL context internally in a way that isn't
compatible with the DSA / bindless-texture / compute-shader pipeline the
renderer needs, so the engine now owns windowing directly via GLFW (already
the intended dependency per RIFTENGINE_SPECIFICATION.md: PyGLFW==2.7.0).

Window's job stops at "there is a current GL 4.6 context and I can tell you
when it should close / present it." Everything GPU-resource-related (shaders,
textures, draw calls) lives in rift.renderer.GLRenderer, which takes a
Window instance that has already had create() called on it.
"""

from __future__ import annotations

import logging
from typing import Callable

import glfw

logger = logging.getLogger("rift.core.window")

ResizeCallback = Callable[[int, int], None]


class Window:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "Rift Engine", *, vsync: bool = True):
        self.width = width
        self.height = height
        self.title = title
        self.vsync = vsync
        self.handle = None
        self._resize_callbacks: list[ResizeCallback] = []

    def create(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, True)

        self.handle = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.handle:
            glfw.terminate()
            raise RuntimeError(
                "glfw.create_window() failed -- most likely the driver doesn't "
                "support an OpenGL 4.6 core profile. See the 4.5 fallback note "
                "in RIFTENGINE_SPECIFICATION.md (no bindless textures / indirect "
                "draw available in that case; not implemented yet)."
            )

        glfw.make_context_current(self.handle)
        glfw.swap_interval(1 if self.vsync else 0)
        glfw.set_framebuffer_size_callback(self.handle, self._on_resize)
        logger.info("Window created: %dx%d '%s'", self.width, self.height, self.title)

    def _on_resize(self, handle, width: int, height: int) -> None:
        self.width, self.height = width, height
        for callback in self._resize_callbacks:
            callback(width, height)

    def on_resize(self, callback: ResizeCallback) -> None:
        """Register for framebuffer resize events. GLRenderer uses this to
        keep its viewport and camera UBO in sync -- see gl_renderer.py."""
        self._resize_callbacks.append(callback)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.handle)

    def poll_events(self) -> None:
        glfw.poll_events()

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.handle)

    def close(self) -> None:
        if self.handle:
            glfw.destroy_window(self.handle)
        glfw.terminate()
