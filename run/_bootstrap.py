"""Small helpers shared by the public command-line entry points."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def add_source_path() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def run_module(module: str, args: list[str]) -> None:
    add_source_path()
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__")


def dispatch(commands: dict[str, str], description: str) -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        names = "\n".join(f"  {name:<18} {module}" for name, module in commands.items())
        print(f"{description}\n\nCommands:\n{names}")
        return
    command = sys.argv[1]
    if command not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"Unknown command {command!r}; choose one of: {choices}")
    run_module(commands[command], sys.argv[2:])
