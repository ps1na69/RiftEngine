import sys
from pathlib import Path

# Make `rift` importable regardless of how this script is launched --
# running it by direct file path (as opposed to `python -m sandbox.main`
# from the repo root) doesn't put the repo root on sys.path otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rift.core.application import Application


def main() -> None:
    app = Application(title="Sandbox")
    app.run()


if __name__ == "__main__":
    main()
