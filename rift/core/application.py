"""
rift/core/application.py

Top-level app wrapper: owns the Window (GLFW context) and GLRenderer (GPU
resources + draw orchestration), and drives the main loop. This replaces
the earlier raylib-backed version -- there is no raylib dependency left
anywhere in the engine after this change.
"""

from __future__ import annotations

from typing import Callable, Optional

from .window import Window
from rift.renderer.gl_renderer import GLRenderer

# Called once per frame, after the screen is cleared and before it's
# presented. This is a stand-in for the real scripting system
# (script_runner.py / script_api.py / on_update()) described in the spec,
# which is a later phase -- for now it's how sandbox code and tests
# actually get pixels on screen.
FrameCallback = Callable[["GLRenderer", float], None]


class Application:
    """Owns the window + renderer and runs the main loop."""

    def __init__(self, width: int = 1280, height: int = 720, title: str = "Rift Engine"):
        self.window = Window(width, height, title)
        self.renderer: Optional[GLRenderer] = None

    def run(self, on_frame: Optional[FrameCallback] = None) -> None:
        self.window.create()
        self.renderer = GLRenderer(self.window)

        while not self.window.should_close():
            self.window.poll_events()
            self.renderer.begin_frame()

            if on_frame is not None:
                on_frame(self.renderer, self.renderer.delta_time)

            self.renderer.end_frame()
            self.window.swap_buffers()

        self.renderer.close()
        self.window.close()
