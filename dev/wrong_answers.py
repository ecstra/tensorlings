"""Run common wrong answers through each exercise's checks and print the messages.

Each case takes an exercise's solution and swaps one piece of it for a likely
mistake, so the other functions stay right and only the mistake is tested.
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

# Loading run.py would otherwise leave a __pycache__ folder at the repo root.
sys.dont_write_bytecode = True

# (exercise, what is wrong, a piece of the solution, what a learner might write instead)
CASES: list[tuple[str, str, str, str]] = [
    ("tensors1", "b left as a list", "b = torch.tensor([10, 20, 30, 40])", "b = [10, 20, 30, 40]"),
    ("tensors1", "b with the wrong numbers", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30])"),
    ("tensors1", "d with 2 rows of 3", "d = torch.tensor([[1, 2],\n                  [3, 4],\n                  [5, 6]])", "d = torch.tensor([[1, 2, 3], [4, 5, 6]])"),
    ("tensors1", "e still crashes", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([[1, 2], [3, 4, 5]])"),
    ("tensors1", "e fixed by flattening it", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([1, 2, 3, 4, 5])"),
    ("tensors1", "syntax error", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30, 40]"),
    ("indexing1", "first_column as a Python list", "first_column = grid[:, 0]", "first_column = [row[0] for row in grid]"),
    ("indexing1", "first_column takes the first row", "first_column = grid[:, 0]", "first_column = grid[0]"),
    ("indexing1", "middle_row takes the middle column", "middle_row = grid[1]", "middle_row = grid[:, 1]"),
    ("indexing1", "nine counts from 1", "nine = grid[2, 2]", "nine = grid[3, 3]"),
    ("indexing1", "last_two_columns takes one column", "last_two_columns = grid[:, 1:]", "last_two_columns = grid[:, 2]"),
    ("indexing1", "last_two_columns takes the last two rows", "last_two_columns = grid[:, 1:]", "last_two_columns = grid[1:]"),
    ("math1", "multiplied adds instead", "multiplied = a * b", "multiplied = a + b"),
    ("math1", "squared doubles instead", "squared = a ** 2", "squared = a * 2"),
    ("math1", "negative marks the positives", "negative = c < 0", "negative = c > 0"),
    ("math1", "no_negatives keeps only the negatives", "no_negatives = torch.where(c < 0, 0.0, c)", "no_negatives = torch.where(c < 0, c, 0.0)"),
    ("broadcasting1", "times_row adds instead", "times_row = grid * torch.tensor([1.0, 0.0, -1.0])", "times_row = grid + torch.tensor([1.0, 0.0, -1.0])"),
    ("broadcasting1", "per_row still crashes", "per_row = grid + torch.tensor([100.0, 200.0]).unsqueeze(1)", "per_row = grid + torch.tensor([100.0, 200.0])"),
    ("broadcasting1", "per_row fixed with a third number", "per_row = grid + torch.tensor([100.0, 200.0]).unsqueeze(1)", "per_row = grid + torch.tensor([100.0, 200.0, 300.0])"),
    ("reductions1", "total adds up each row", "total = grid.sum()", "total = grid.sum(dim=1)"),
    ("reductions1", "row_totals adds down the columns", "row_totals = grid.sum(dim=1)", "row_totals = grid.sum(dim=0)"),
    ("reductions1", "column_means adds instead", "column_means = grid.mean(dim=0)", "column_means = grid.sum(dim=0)"),
    ("reductions1", "best_in_column looks along the rows", "best_in_column = scores.argmax(dim=0)", "best_in_column = scores.argmax(dim=1)"),
    ("reductions1", "fractions without keepdim", "fractions = grid / totals", "fractions = grid / grid.sum(dim=1)"),
    ("matmul1", "weighted only multiplies", "weighted = v @ w", "weighted = v * w"),
    ("matmul1", "dot_pq only multiplies", "dot_pq = p @ q", "dot_pq = p * q"),
    ("matmul1", "m_times_w in the wrong order", "m_times_w = m @ w", "m_times_w = w @ m"),
    ("matmul1", "rows_with_rows still crashes", "rows_with_rows = n @ n.T", "rows_with_rows = n @ n"),
    ("matmul1", "rows_with_rows turns the wrong side", "rows_with_rows = n @ n.T", "rows_with_rows = n.T @ n"),
    ("matmul1", "out2 without the bias", "out2 = x2 @ weight.T + bias", "out2 = x2 @ weight.T"),
    ("quiz1", "x left as a list", "x = torch.tensor(data)", "x = data"),
    ("quiz1", "first_two takes the first two rows", "first_two = x[:, 0:2]", "first_two = x[0:2]"),
    ("quiz1", "no_negatives keeps the negatives", "no_negatives = torch.where(first_two < 0, 0.0, first_two)", "no_negatives = first_two"),
    ("quiz1", "shifted adds one number to everything", "shifted = no_negatives + torch.tensor([10.0, 20.0])", "shifted = no_negatives + 10.0"),
    ("quiz1", "row_totals adds down the columns", "row_totals = shifted.sum(dim=1)", "row_totals = shifted.sum(dim=0)"),
    ("quiz1", "scores in the wrong order", "scores = x @ weights", "scores = weights @ x"),
    ("quiz1", "best is the largest score, not its position", "best = scores.argmax()", "best = scores.max()"),
    ("quiz1", "a loop that never ends", "x = torch.tensor(data)", "while True:\n    pass\nx = torch.tensor(data)"),
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
        for number, (name, label, piece, mistake) in enumerate(CASES):
            exercise = course.exercise_named(name)

            # A case whose piece is gone from the solution would test the solution itself.
            if exercise.solution.count(piece) != 1:
                raise SystemExit(f"{name}, {label}: the piece {piece!r} is not in the solution exactly once.")

            path = Path(folder) / str(number) / f"{name}.py"
            path.parent.mkdir()
            path.write_text(exercise.solution.replace(piece, mistake))
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
