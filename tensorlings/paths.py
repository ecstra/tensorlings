"""Where the course keeps its files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXERCISES_DIR = ROOT / "exercises"
SOLUTIONS_DIR = ROOT / "solutions"
STATE_FILE = ROOT / ".tensorlings-state.json"
DATA_DIR = ROOT / "data"
START_COMMAND = ".venv/bin/python run.py"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
