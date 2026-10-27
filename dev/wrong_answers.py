"""Run common wrong answers through each exercise's checks and print the messages.

Every case must fail, and the script exits with 1 if one passes, since a check
that accepts a wrong answer is testing nothing. Read the printed messages too,
because a beginner sees exactly these.

    .venv/bin/python dev/wrong_answers.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
HEAD = "import torch\nfrom torch import Tensor\n\n"

# Loading run.py would otherwise leave a __pycache__ folder at the repo root.
sys.dont_write_bytecode = True

# (exercise, what is wrong, the file a learner might write)
CASES: list[tuple[str, str, str]] = [
    (
        "tensors1",
        "a list instead of a tensor",
        HEAD + "def make_grid(): return [[1, 2, 3], [4, 5, 6]]\ndef shape_of(v): return v.shape\ndef count_entries(v): return len(v)\n",
    ),
    (
        "tensors1",
        "rows and columns swapped",
        HEAD + "def make_grid(): return torch.tensor([[1, 4], [2, 5], [3, 6]])\ndef shape_of(v): return tuple(v.shape)\ndef count_entries(v): return v.numel()\n",
    ),
    (
        "tensors2",
        "last row instead of last column",
        HEAD + "def first_row(g): return g[0]\ndef last_column(g): return g[-1]\ndef top_left_corner(g, s): return g[:s, :s]\n",
    ),
    (
        "tensors2",
        "corner size fixed at 2",
        HEAD + "def first_row(g): return g[0]\ndef last_column(g): return g[:, -1]\ndef top_left_corner(g, s): return g[:2, :2]\n",
    ),
    (
        "tensors3",
        "first score by index, not by slice",
        HEAD + "def add_bonus(s, b): return s + b\ndef gain_since_first_test(s): return s - s[:, 0]\n",
    ),
    (
        "tensors3",
        "misspelled function name",
        HEAD + "def add_bonus(s, b): return s + b\ndef gain_since_first(s): return s - s[:, 0:1]\n",
    ),
    (
        "tensors4",
        "factors without a new dimension",
        HEAD + "def scale_each_row(s, f): return s * f\n",
    ),
    (
        "tensors5",
        "mean on whole numbers",
        HEAD + "def total_per_row(s): return s.sum(dim=1)\ndef average_per_column(s): return s.mean(dim=0)\ndef share_of_row(s): return s / s.sum(dim=1, keepdim=True)\n",
    ),
    (
        "tensors5",
        "share without keepdim",
        HEAD + "def total_per_row(s): return s.sum(dim=1)\ndef average_per_column(s): return s.float().mean(dim=0)\ndef share_of_row(s): return s / s.sum(dim=1)\n",
    ),
    (
        "matmul1",
        "whole numbers mixed with decimal weights",
        HEAD + "def weighted_sum(i, w): return (i * w).sum()\ndef rents(f, w): return f.long() @ w\n",
    ),
    (
        "matmul1",
        "rents summed into one number",
        HEAD + "def weighted_sum(i, w): return (i * w).sum()\ndef rents(f, w): return (f * w).sum()\n",
    ),
    (
        "matmul1",
        "syntax error",
        HEAD + "def weighted_sum(i, w) return (i * w).sum()\n",
    ),
]


def load_course() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name="run",
        location=REPO / "run.py",
    )
    course = importlib.util.module_from_spec(spec)
    sys.modules["run"] = course
    spec.loader.exec_module(course)

    return course


def main() -> int:
    course = load_course()
    accepted: list[str] = []

    with tempfile.TemporaryDirectory() as folder:
        for number, (name, label, text) in enumerate(CASES):
            exercise = course.exercise_named(name)
            path = Path(folder) / str(number) / f"{name}.py"
            path.parent.mkdir()
            path.write_text(text)
            outcome = course.attempt(
                exercise=exercise,
                path=path,
            )

            print(f"=== {name}, {label}: {'PASSED, which is wrong' if outcome.passed else 'fails as it should'}")

            if outcome.passed:
                accepted.append(f"{name}, {label}")

            if outcome.problem:
                print(f"    {outcome.problem}")

            for result in outcome.results:
                if result.passed:
                    continue

                print(f"  ✗ {result.description}")
                print("    " + result.detail.replace("\n", "\n    "))

    print()

    if accepted:
        print("These wrong answers passed: " + ", ".join(accepted))
        return 1

    print(f"All {len(CASES)} wrong answers fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
