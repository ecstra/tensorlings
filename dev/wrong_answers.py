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
    ("tensors2", "last_columns slices the rows", "grid[:, -count:]", "grid[-count:]"),
    ("tensors2", "last_columns takes one column by index", "grid[:, -count:]", "grid[:, -count]"),
    ("tensors2", "split_last_column keeps the last column's dimension", "table[:, :-1], table[:, -1]", "table[:, :-1], table[:, -1:]"),
    ("tensors2", "split_last_column gives the pair the wrong way round", "table[:, :-1], table[:, -1]", "table[:, -1], table[:, :-1]"),
    ("tensors2", "split_last_column returns one tensor", "table[:, :-1], table[:, -1]", "table[:, :-1]"),
    ("tensors2", "pick_rows picks columns", "table[positions]", "table[:, positions]"),
    ("tensors3", "zero_negatives takes the absolute value", "torch.where(x > 0, x, 0.0)", "x.abs()"),
    ("tensors3", "squared_differences forgets to square", "(a - b) ** 2", "a - b"),
    ("tensors3", "squared_differences squares before subtracting", "(a - b) ** 2", "a ** 2 - b ** 2"),
    ("tensors3", "zero_padding always uses 0 as pad_id", "torch.where(ids == pad_id, 0.0, values)", "torch.where(ids == 0, 0.0, values)"),
    ("tensors3", "zero_padding keeps only the padding", "torch.where(ids == pad_id, 0.0, values)", "torch.where(ids == pad_id, values, 0.0)"),
    ("tensors4", "add_to_rows adds one number per row", "grid + row", "grid + row.unsqueeze(1)"),
    ("tensors4", "rescale_columns forgets the low", "(grid - lows) / (highs - lows)", "grid / highs"),
    ("tensors4", "rescale_columns divides by the high", "(grid - lows) / (highs - lows)", "(grid - lows) / highs"),
    ("tensors4", "divide_rows without unsqueeze", "grid / divisors.unsqueeze(1)", "grid / divisors"),
    ("tensors5", "column_means averages each row", "grid.mean(dim=0)", "grid.mean(dim=1)"),
    ("tensors5", "accuracy takes argmax down the columns", "scores.argmax(dim=1) == answers", "scores.argmax(dim=0) == answers"),
    ("tensors5", "accuracy without .float()", ".float().mean().item()", ".mean().item()"),
    ("tensors5", "rows_to_fractions without keepdim", "grid / grid.sum(dim=1, keepdim=True)", "grid / grid.sum(dim=1)"),
    ("matmul1", "weighted_sums multiplies in the wrong order", "grid @ weights", "weights @ grid"),
    ("matmul1", "dot_all_pairs without .T", "a @ b.T", "a @ b"),
    ("matmul1", "dot_all_pairs in the wrong order", "a @ b.T", "b @ a.T"),
    ("matmul1", "linear without .T", "x @ weight.T + bias", "x @ weight + bias"),
    ("matmul1", "linear without the bias", "x @ weight.T + bias", "x @ weight.T"),
    ("quiz1", "sentence_averages leaves the padding in", "totals = (rows * real).sum(dim=1)", "totals = rows.sum(dim=1)"),
    ("quiz1", "sentence_averages divides by the full length", "lengths = real.sum(dim=1)", "lengths = ids.shape[1]"),
    ("quiz1", "sentence_averages loops for ever", "    rows = table[ids]", "    while True:\n        pass\n    rows = table[ids]"),
    ("quiz1", "predict without .T", "averages @ weight.T + bias", "averages @ weight + bias"),
    ("quiz1", "predict takes argmax down the columns", "class_scores.argmax(dim=1)", "class_scores.argmax(dim=0)"),
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
