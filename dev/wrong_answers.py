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

TENSORS1_GRID = "def make_grid(): return torch.tensor([[1, 2], [3, 4], [5, 6]])\n"
TENSORS1_REST = "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v.shape)\ndef count_entries(v): return v.numel()\n"
TENSORS2_LAST_TWO = "def last_two_columns(g): return g[:, -2:]\n"
TENSORS3_RESCALE = "def rescale(g, low, high): return (g - low) / (high - low)\n"
TENSORS5_FIRST_TWO = "def column_sums(g): return g.sum(dim=0)\ndef row_means(g): return g.float().mean(dim=1)\n"
MATMUL1_WEIGHTED_SUM = "def weighted_sum(v): return (v * torch.tensor([0.5, -2.0, 3.0])).sum()\n"

# (exercise, what is wrong, the file a learner might write)
CASES: list[tuple[str, str, str]] = [
    (
        "tensors1",
        "a list instead of a tensor",
        HEAD + "def make_grid(): return [[1, 2], [3, 4], [5, 6]]\n" + TENSORS1_REST,
    ),
    (
        "tensors1",
        "2 rows of 3 instead of 3 rows of 2",
        HEAD + "def make_grid(): return torch.tensor([[1, 2, 3], [4, 5, 6]])\n" + TENSORS1_REST,
    ),
    (
        "tensors1",
        "dimensions counted with len of the tensor",
        HEAD
        + TENSORS1_GRID
        + "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v)\ndef count_entries(v): return v.numel()\n",
    ),
    (
        "tensors1",
        "entries as the first two sizes multiplied",
        HEAD
        + TENSORS1_GRID
        + "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v.shape)\n"
        + "def count_entries(v): return v.shape[0] * v.shape[1]\n",
    ),
    (
        "tensors2",
        "last two rows instead of last two columns",
        HEAD + "def last_two_columns(g): return g[-2:]\ndef drop_first_column(g): return g[:, 1:]\ndef inner_part(g): return g[1:-1, 1:-1]\n",
    ),
    (
        "tensors2",
        "first row dropped instead of first column",
        HEAD + TENSORS2_LAST_TWO + "def drop_first_column(g): return g[1:]\ndef inner_part(g): return g[1:-1, 1:-1]\n",
    ),
    (
        "tensors2",
        "edge taken off the rows only",
        HEAD + TENSORS2_LAST_TWO + "def drop_first_column(g): return g[:, 1:]\ndef inner_part(g): return g[1:-1]\n",
    ),
    (
        "tensors3",
        "low left out of the divisor",
        HEAD + "def rescale(g, low, high): return (g - low) / high\ndef subtract_first_column(g): return g - g[:, 0:1]\n",
    ),
    (
        "tensors3",
        "first column by index, not by slice",
        HEAD + TENSORS3_RESCALE + "def subtract_first_column(g): return g - g[:, 0]\n",
    ),
    (
        "tensors3",
        "misspelled function name",
        HEAD + TENSORS3_RESCALE + "def subtract_first(g): return g - g[:, 0:1]\n",
    ),
    (
        "tensors4",
        "times table without a new dimension",
        HEAD + "def times_table(r, c): return r * c\ndef add_row_and_column_values(g, r, c): return g + r[:, None] + c\n",
    ),
    (
        "tensors4",
        "per_row without a new dimension",
        HEAD + "def times_table(r, c): return r[:, None] * c\ndef add_row_and_column_values(g, r, c): return g + r + c\n",
    ),
    (
        "tensors5",
        "column sums along the wrong dimension",
        HEAD
        + "def column_sums(g): return g.sum(dim=1)\ndef row_means(g): return g.float().mean(dim=1)\n"
        + "def rows_to_fractions(g): return g / g.sum(dim=1, keepdim=True)\n",
    ),
    (
        "tensors5",
        "mean on ints",
        HEAD
        + "def column_sums(g): return g.sum(dim=0)\ndef row_means(g): return g.mean(dim=1)\n"
        + "def rows_to_fractions(g): return g / g.sum(dim=1, keepdim=True)\n",
    ),
    (
        "tensors5",
        "fractions without keepdim",
        HEAD + TENSORS5_FIRST_TWO + "def rows_to_fractions(g): return g / g.sum(dim=1)\n",
    ),
    (
        "matmul1",
        "a float returned instead of a tensor",
        HEAD + "def weighted_sum(v): return float((v * torch.tensor([0.5, -2.0, 3.0])).sum())\ndef linear(x, w, b): return x @ w.T + b\n",
    ),
    (
        "matmul1",
        "weight left with outputs as rows",
        HEAD + MATMUL1_WEIGHTED_SUM + "def linear(x, w, b): return x @ w + b\n",
    ),
    (
        "matmul1",
        "bias left out",
        HEAD + MATMUL1_WEIGHTED_SUM + "def linear(x, w, b): return x @ w.T\n",
    ),
    (
        "matmul1",
        "syntax error",
        HEAD + "def weighted_sum(v) return (v * torch.tensor([0.5, -2.0, 3.0])).sum()\n",
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

                print(f"  ✗ {result.function}, {result.description}")
                print("    " + result.detail.replace("\n", "\n    "))

    print()

    if accepted:
        print("These wrong answers passed: " + ", ".join(accepted))
        return 1

    print(f"All {len(CASES)} wrong answers fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
