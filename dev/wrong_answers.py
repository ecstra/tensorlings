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
    ("tensors1", "make_batch keeps whole numbers as integers", "torch.tensor(examples, dtype=torch.float32)", "torch.tensor(examples)"),
    ("tensors1", "make_batch returns the list", "torch.tensor(examples, dtype=torch.float32)", "examples"),
    ("tensors1", "flatten_images flattens the whole batch", "images.reshape(images.shape[0], -1)", "images.reshape(-1)"),
    ("tensors1", "count_parameters counts the tensors", "sum(tensor.numel() for tensor in tensors)", "len(tensors)"),
    ("tensors1", "count_parameters counts rows", "sum(tensor.numel() for tensor in tensors)", "sum(len(tensor) for tensor in tensors)"),
    ("tensors1", "syntax error", "def make_batch(examples: list[list[float]]) -> Tensor:", "def make_batch(examples: list[list[float]]) -> Tensor"),
    ("tensors2", "last_tokens slices the sentences", "tokens[:, -context:]", "tokens[-context:]"),
    ("tensors2", "last_tokens takes one token by index", "tokens[:, -context:]", "tokens[:, -context]"),
    ("tensors2", "split_labels keeps the label dimension", "table[:, :-1], table[:, -1]", "table[:, :-1], table[:, -1:]"),
    ("tensors2", "split_labels gives the pair the wrong way round", "table[:, :-1], table[:, -1]", "table[:, -1], table[:, :-1]"),
    ("tensors2", "split_labels returns one tensor", "table[:, :-1], table[:, -1]", "table[:, :-1]"),
    ("tensors2", "look_up reads columns", "table[ids]", "table[:, ids]"),
    ("tensors3", "relu takes the absolute value", "torch.where(x > 0, x, 0.0)", "x.abs()"),
    ("tensors3", "squared_error forgets to square", "(predictions - targets) ** 2", "predictions - targets"),
    ("tensors3", "squared_error squares before subtracting", "(predictions - targets) ** 2", "predictions ** 2 - targets ** 2"),
    ("tensors3", "mask_padding always pads with 0", "torch.where(ids == pad_id, 0.0, values)", "torch.where(ids == 0, 0.0, values)"),
    ("tensors3", "mask_padding keeps only the padding", "torch.where(ids == pad_id, 0.0, values)", "torch.where(ids == pad_id, values, 0.0)"),
    ("tensors4", "add_bias adds one number per row", "batch + bias", "batch + bias.unsqueeze(1)"),
    ("tensors4", "rescale forgets the low", "(batch - lows) / (highs - lows)", "batch / highs"),
    ("tensors4", "rescale divides by the high", "(batch - lows) / (highs - lows)", "(batch - lows) / highs"),
    ("tensors4", "average_per_sentence without unsqueeze", "totals / lengths.unsqueeze(1)", "totals / lengths"),
    ("tensors5", "feature_means averages each example", "batch.mean(dim=0)", "batch.mean(dim=1)"),
    ("tensors5", "accuracy takes argmax down the columns", "scores.argmax(dim=1) == labels", "scores.argmax(dim=0) == labels"),
    ("tensors5", "accuracy without .float()", ".float().mean().item()", ".mean().item()"),
    ("tensors5", "to_shares without keepdim", "scores / scores.sum(dim=1, keepdim=True)", "scores / scores.sum(dim=1)"),
    ("matmul1", "scores multiplies in the wrong order", "batch @ weights", "weights @ batch"),
    ("matmul1", "pairwise_scores without .T", "a @ b.T", "a @ b"),
    ("matmul1", "pairwise_scores in the wrong order", "a @ b.T", "b @ a.T"),
    ("matmul1", "linear without .T", "batch @ weight.T + bias", "batch @ weight + bias"),
    ("matmul1", "linear without the bias", "batch @ weight.T + bias", "batch @ weight.T"),
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
