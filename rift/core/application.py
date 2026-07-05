from .window import Window


class Application:
    """Simple Rift application wrapper around the core window."""

    def __init__(self, width: int = 1280, height: int = 720, title: str = "Rift Engine"):
        self.window = Window(width, height, title)

    def run(self) -> None:
        self.window.create()

        while not self.window.should_close():
            self.window.begin_draw()
            self.window.clear()
            self.window.end_draw()

        self.window.close()