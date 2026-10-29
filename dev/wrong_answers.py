"""Run common wrong answers through each exercise's checks and print the messages.

Each case takes an exercise's solution and swaps one piece of it for a likely
mistake, so the other functions stay right and only the mistake is tested.
Every case must fail, and the script exits with 1 if one passes, since a check
that accepts a wrong answer is testing nothing. Read the printed messages too,
because a beginner sees exactly these.

    .venv/bin/python dev/wrong_answers.py
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Loading the course would otherwise leave __pycache__ folders in the package.
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

from tensorlings.course import attempt
from tensorlings.progress import exercise_named

# (exercise, what is wrong, a piece of the solution, what a learner might write instead)
CASES: list[tuple[str, str, str, str]] = [
    ("tensors1", "b left as a list", "b = torch.tensor([10, 20, 30, 40])", "b = [10, 20, 30, 40]"),
    ("tensors1", "b with the wrong numbers", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30])"),
    ("tensors1", "d with 2 rows of 3", "d = torch.tensor([[1, 2],\n                  [3, 4],\n                  [5, 6]])", "d = torch.tensor([[1, 2, 3], [4, 5, 6]])"),
    ("tensors1", "e still crashes", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([[1, 2], [3, 4, 5]])"),
    ("tensors1", "e fixed by flattening it", "e = torch.tensor([[1, 2], [3, 4]])", "e = torch.tensor([1, 2, 3, 4, 5])"),
    ("tensors1", "syntax error", "b = torch.tensor([10, 20, 30, 40])", "b = torch.tensor([10, 20, 30, 40]"),
    ("indexing1", "first_column as a Python list", "return grid[:, 0]", "return [row[0] for row in grid]"),
    ("indexing1", "first_column typed in for the grid in the file", "return grid[:, 0]", "return torch.tensor([1, 4, 7])"),
    ("indexing1", "first_column takes the first row", "return grid[:, 0]", "return grid[0]"),
    ("indexing1", "row_one takes a column", "return grid[1]", "return grid[:, 1]"),
    ("indexing1", "bottom_right only works for a 3 by 3 grid", "return grid[-1, -1]", "return grid[2, 2]"),
    ("indexing1", "last_two_columns only works for 3 columns", "return grid[:, -2:]", "return grid[:, 1:]"),
    ("indexing1", "last_two_columns takes the last two rows", "return grid[:, -2:]", "return grid[-2:]"),
    ("math1", "multiply adds instead", "return a * b", "return a + b"),
    ("math1", "multiply typed in for the tensors in the file", "return a * b", "return torch.tensor([10.0, 40.0, 90.0])"),
    ("math1", "square doubles instead", "return a ** 2", "return a * 2"),
    ("math1", "below_zero counts 0 as below", "return c < 0", "return c <= 0"),
    ("math1", "zero_negatives keeps only the negatives", "return torch.where(c < 0, 0.0, c)", "return torch.where(c < 0, c, 0.0)"),
    ("broadcasting1", "multiply_rows adds instead", "return grid * row", "return grid + row"),
    ("broadcasting1", "add_per_row still crashes", "return grid + numbers.unsqueeze(1)", "return grid + numbers"),
    ("reductions1", "total keeps only the last row", "result = result + row.sum()", "result = row.sum()"),
    ("reductions1", "total typed in for the grid in the file", "return result", "return 21.0"),
    ("reductions1", "row_totals adds down the columns", "return grid.sum(dim=1)", "return grid.sum(dim=0)"),
    ("reductions1", "column_means adds instead", "return grid.mean(dim=0)", "return grid.sum(dim=0)"),
    ("reductions1", "best_in_each_column looks along the rows", "return grid.argmax(dim=0)", "return grid.argmax(dim=1)"),
    ("reductions1", "rows_to_fractions without keepdim", "    totals = grid.sum(dim=1, keepdim=True)", "    totals = grid.sum(dim=1)"),
    ("matmul1", "weighted_sum only multiplies", "return (v @ w).item()", "return v * w"),
    ("matmul1", "weighted_sum typed in for the tensors in the comment", "return (v @ w).item()", "return 320.0"),
    ("matmul1", "row_dots in the wrong order", "return matrix @ w", "return w @ matrix"),
    ("matmul1", "row_pairs still crashes", "return a @ b.T", "return a @ b"),
    ("matmul1", "row_pairs turns the wrong side", "return a @ b.T", "return a.T @ b"),
    ("matmul1", "best_match without .T", "scores = a @ b.T", "scores = a @ b"),
    ("matmul1", "best_match scores b against a", "scores = a @ b.T", "scores = b @ a.T"),
    ("quiz1", "biggest_row adds down the columns", "return grid.sum(dim=1).argmax().item()", "return grid.sum(dim=0).argmax().item()"),
    ("quiz1", "biggest_row typed in for the grid in the comment", "return grid.sum(dim=1).argmax().item()", "return 1"),
    ("quiz1", "clean_fractions without keepdim", "return cleaned / cleaned.sum(dim=1, keepdim=True)", "return cleaned / cleaned.sum(dim=1)"),
    ("quiz1", "center_columns takes each row's mean", "return grid - grid.mean(dim=0)", "return grid - grid.mean(dim=1)"),
    ("quiz1", "weighted_first_columns keeps the first two rows", "first_two = grid[:, 0:2]", "first_two = grid[0:2]"),
    ("quiz1", "weighted_first_columns in the wrong order", "return first_two @ weights", "return weights @ first_two"),
    ("quiz1", "a loop that never ends", "    cleaned = torch.where(grid < 0, 0.0, grid)", "    while True:\n        pass\n    cleaned = torch.where(grid < 0, 0.0, grid)"),
]


def main() -> int:
    accepted: list[str] = []

    with tempfile.TemporaryDirectory() as folder:
        for number, (name, label, piece, mistake) in enumerate(CASES):
            exercise = exercise_named(name)

            # A case whose piece is gone from the solution would test the solution itself.
            if exercise.solution.count(piece) != 1:
                raise SystemExit(f"{name}, {label}: the piece {piece!r} is not in the solution exactly once.")

            path = Path(folder) / str(number) / f"{name}.py"
            path.parent.mkdir()
            path.write_text(exercise.solution.replace(piece, mistake))
            outcome = attempt(
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
