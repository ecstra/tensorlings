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

TENSORS1_SALES = "def make_sales(): return torch.tensor([[4, 2], [1, 5]])\n"
TENSORS1_REST = "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v.shape)\ndef count_entries(v): return v.numel()\n"
TENSORS2_LAST_TWO = "def last_two_words(s): return s[:, -2:]\n"
TENSORS3_PERCENT = "def as_percent(s, t): return s / t * 100\n"
TENSORS5_FIRST_TWO = "def total_per_test(s): return s.sum(dim=0)\ndef average_per_student(s): return s.float().mean(dim=1)\n"
MATMUL1_GRADE = "def final_grade(m): return (m * torch.tensor([0.2, 0.3, 0.5])).sum()\n"

# (exercise, what is wrong, the file a learner might write)
CASES: list[tuple[str, str, str]] = [
    (
        "tensors1",
        "a list instead of a tensor",
        HEAD + "def make_sales(): return [[4, 2], [1, 5]]\n" + TENSORS1_REST,
    ),
    (
        "tensors1",
        "one row per fruit instead of one per day",
        HEAD + "def make_sales(): return torch.tensor([[4, 1], [2, 5]])\n" + TENSORS1_REST,
    ),
    (
        "tensors1",
        "dimensions counted with len of the tensor",
        HEAD
        + TENSORS1_SALES
        + "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v)\ndef count_entries(v): return v.numel()\n",
    ),
    (
        "tensors1",
        "entries as the first two sizes multiplied",
        HEAD
        + TENSORS1_SALES
        + "def count_rows(v): return v.shape[0]\ndef count_dimensions(v): return len(v.shape)\n"
        + "def count_entries(v): return v.shape[0] * v.shape[1]\n",
    ),
    (
        "tensors2",
        "last two sentences instead of last two words",
        HEAD + "def last_two_words(s): return s[-2:]\ndef drop_first_word(s): return s[:, 1:]\ndef trim_edges(s): return s[1:-1, 1:-1]\n",
    ),
    (
        "tensors2",
        "first sentence dropped instead of first word",
        HEAD + TENSORS2_LAST_TWO + "def drop_first_word(s): return s[1:]\ndef trim_edges(s): return s[1:-1, 1:-1]\n",
    ),
    (
        "tensors2",
        "edges trimmed off the sentences only",
        HEAD + TENSORS2_LAST_TWO + "def drop_first_word(s): return s[:, 1:]\ndef trim_edges(s): return s[1:-1]\n",
    ),
    (
        "tensors3",
        "percent as a fraction",
        HEAD + "def as_percent(s, t): return s / t\ndef gain_since_first_test(s): return s - s[:, 0:1]\n",
    ),
    (
        "tensors3",
        "first score by index, not by slice",
        HEAD + TENSORS3_PERCENT + "def gain_since_first_test(s): return s - s[:, 0]\n",
    ),
    (
        "tensors3",
        "misspelled function name",
        HEAD + TENSORS3_PERCENT + "def gain_since_first(s): return s - s[:, 0:1]\n",
    ),
    (
        "tensors4",
        "times table without a new dimension",
        HEAD + "def times_table(r, c): return r * c\ndef add_both_bonuses(s, b, t): return s + b[:, None] + t\n",
    ),
    (
        "tensors4",
        "student bonus without a new dimension",
        HEAD + "def times_table(r, c): return r[:, None] * c\ndef add_both_bonuses(s, b, t): return s + b + t\n",
    ),
    (
        "tensors5",
        "total along the wrong dimension",
        HEAD
        + "def total_per_test(s): return s.sum(dim=1)\ndef average_per_student(s): return s.float().mean(dim=1)\n"
        + "def share_of_row(s): return s / s.sum(dim=1, keepdim=True)\n",
    ),
    (
        "tensors5",
        "mean on whole numbers",
        HEAD
        + "def total_per_test(s): return s.sum(dim=0)\ndef average_per_student(s): return s.mean(dim=1)\n"
        + "def share_of_row(s): return s / s.sum(dim=1, keepdim=True)\n",
    ),
    (
        "tensors5",
        "share without keepdim",
        HEAD + TENSORS5_FIRST_TWO + "def share_of_row(s): return s / s.sum(dim=1)\n",
    ),
    (
        "matmul1",
        "weights as percents instead of decimals",
        HEAD + "def final_grade(m): return (m * torch.tensor([20.0, 30.0, 50.0])).sum()\ndef grades_under_schemes(m, s): return m @ s.T\n",
    ),
    (
        "matmul1",
        "schemes left as rows",
        HEAD + MATMUL1_GRADE + "def grades_under_schemes(m, s): return m @ s\n",
    ),
    (
        "matmul1",
        "schemes multiplied entry by entry",
        HEAD + MATMUL1_GRADE + "def grades_under_schemes(m, s): return m * s\n",
    ),
    (
        "matmul1",
        "syntax error",
        HEAD + "def final_grade(m) return (m * torch.tensor([0.2, 0.3, 0.5])).sum()\n",
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
