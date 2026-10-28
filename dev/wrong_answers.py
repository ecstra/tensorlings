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
HEAD = (
    "import torch\n"
    "import torch.nn.functional as F\n"
    "from torch import Tensor\n\n"
    'LETTERS = ".abcdefghijklmnopqrstuvwxyz"\n\n'
    'def encode(name): return torch.tensor([LETTERS.index(c) for c in "." + name + "."])\n'
)

# Loading run.py would otherwise leave a __pycache__ folder at the repo root.
sys.dont_write_bytecode = True

TENSORS1_REST = "def empty_counts(): return torch.zeros((27, 27), dtype=torch.int64)\n"
TENSORS2_REST = (
    "def first_letter_counts(c): return c[0]\n"
    "def count_of(c, p): return c[LETTERS.index(p[0]), LETTERS.index(p[1])].item()\n"
)
COUNT_PAIRS = (
    "def count_pairs(names):\n"
    "    counts = torch.zeros((27, 27), dtype=torch.int64)\n"
    "    for name in names:\n"
    "        letters = encode(name)\n"
    "        for first, second in zip(letters, letters[1:]):\n"
    "            counts[{index}] += 1\n"
    "    return counts\n"
)
TO_PROBABILITIES = "def to_probabilities(c): return c / c.sum(dim=1, keepdim=True)\n"
SAMPLE_NAME = (
    "def sample_name(probs, generator):\n"
    "    name = ''\n"
    "    position = 0\n"
    "    while True:\n"
    "        position = torch.multinomial(probs[{row}], num_samples=1, generator=generator).item()\n"
    "        if position == 0:\n"
    "            return name\n"
    "        name += LETTERS[position]\n"
)
LOSS = (
    "def loss(probs, names):\n"
    "    firsts, seconds = [], []\n"
    "    for name in names:\n"
    "        letters = encode(name)\n"
    "        firsts.extend(letters[:-1].tolist())\n"
    "        seconds.extend(letters[1:].tolist())\n"
    "    return {result}\n"
)
ONE_HOT = "def one_hot(p): return F.one_hot(p, num_classes=27).float()\n"

# (exercise, what is wrong, the file a learner might write)
CASES: list[tuple[str, str, str]] = [
    (
        "tensors1",
        "encode without the dots at the ends",
        HEAD.replace('"." + name + "."', "name") + TENSORS1_REST,
    ),
    (
        "tensors1",
        "encode returns a list",
        HEAD.replace("torch.tensor([LETTERS.index(c) for c in \".\" + name + \".\"])", '[LETTERS.index(c) for c in "." + name + "."]')
        + TENSORS1_REST,
    ),
    (
        "tensors1",
        "empty_counts holds floats",
        HEAD + "def empty_counts(): return torch.zeros((27, 27))\n",
    ),
    (
        "tensors1",
        "empty_counts with the wrong size",
        HEAD + "def empty_counts(): return torch.zeros((26, 26), dtype=torch.int64)\n",
    ),
    (
        "tensors2",
        "count_pairs with the row and column swapped",
        HEAD + COUNT_PAIRS.replace("{index}", "second, first") + TENSORS2_REST,
    ),
    (
        "tensors2",
        "first_letter_counts takes a column",
        HEAD
        + COUNT_PAIRS.replace("{index}", "first, second")
        + "def first_letter_counts(c): return c[:, 0]\n"
        + "def count_of(c, p): return c[LETTERS.index(p[0]), LETTERS.index(p[1])].item()\n",
    ),
    (
        "tensors2",
        "count_of with the letters swapped",
        HEAD
        + COUNT_PAIRS.replace("{index}", "first, second")
        + "def first_letter_counts(c): return c[0]\n"
        + "def count_of(c, p): return c[LETTERS.index(p[1]), LETTERS.index(p[0])].item()\n",
    ),
    (
        "tensors3",
        "pairs_starting_with sums the columns",
        HEAD
        + "def pairs_starting_with(c): return c.sum(dim=0)\n"
        + "def most_likely_next(c, l): return LETTERS[c[LETTERS.index(l)].argmax().item()]\n"
        + "def total_pairs(c): return c.sum().item()\n",
    ),
    (
        "tensors3",
        "most_likely_next returns the position, not the letter",
        HEAD
        + "def pairs_starting_with(c): return c.sum(dim=1)\n"
        + "def most_likely_next(c, l): return c[LETTERS.index(l)].argmax().item()\n"
        + "def total_pairs(c): return c.sum().item()\n",
    ),
    (
        "tensors4",
        "row totals without keepdim",
        HEAD + "def to_probabilities(c): return c / c.sum(dim=1)\ndef smoothed_probabilities(c, k): return to_probabilities(c + k)\n",
    ),
    (
        "tensors4",
        "k added after dividing",
        HEAD + TO_PROBABILITIES + "def smoothed_probabilities(c, k): return to_probabilities(c) + k\n",
    ),
    (
        "tensors5",
        "sample_name always draws from row 0",
        HEAD + SAMPLE_NAME.replace("{row}", "0") + LOSS.replace("{result}", "-probs[firsts, seconds].log().mean()"),
    ),
    (
        "tensors5",
        "loss without the minus sign",
        HEAD + SAMPLE_NAME.replace("{row}", "position") + LOSS.replace("{result}", "probs[firsts, seconds].log().mean()"),
    ),
    (
        "tensors5",
        "loss as a sum instead of a mean",
        HEAD + SAMPLE_NAME.replace("{row}", "position") + LOSS.replace("{result}", "-probs[firsts, seconds].log().sum()"),
    ),
    (
        "matmul1",
        "one_hot without .float()",
        HEAD + "def one_hot(p): return F.one_hot(p, num_classes=27)\ndef look_up(p, t): return one_hot(p) @ t\n",
    ),
    (
        "matmul1",
        "look_up multiplies in the wrong order",
        HEAD + ONE_HOT + "def look_up(p, t): return t @ one_hot(p).T\n",
    ),
    (
        "matmul1",
        "syntax error",
        HEAD + "def one_hot(p) return F.one_hot(p, num_classes=27).float()\n",
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
