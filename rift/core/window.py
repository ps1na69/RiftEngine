import raylib as rl


class Window:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "Rift Engine"):
        self.width = width
        self.height = height
        self.title = title

    def create(self) -> None:
        rl.InitWindow(self.width, self.height, self.title.encode("utf-8"))
        rl.SetTargetFPS(60)

    def should_close(self) -> bool:
        return rl.WindowShouldClose()

    def begin_draw(self) -> None:
        rl.BeginDrawing()

    def end_draw(self) -> None:
        rl.EndDrawing()

    def clear(self, color: rl.Color = rl.BLUE) -> None:
        rl.ClearBackground(color)

    def close(self) -> None:
        rl.CloseWindow()