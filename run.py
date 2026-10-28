"""tensorlings, small exercises that teach deep learning from tensors up.

Set the course up once, then start it.

    .venv/bin/python run.py init
    .venv/bin/python run.py

Each exercise is a Python file in exercises/. This file holds the app, the
lessons, the checks and the solutions. You never need to read it.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

try:
    import torch
    from rich.console import Console, Group, RenderableType
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.screen import ModalScreen, Screen
    from textual.widgets import DataTable, Footer, Input, Static, TabbedContent, TabPane, Tabs
    from textual.widgets import Markdown as MarkdownWidget
except ModuleNotFoundError as missing:
    print(f"tensorlings needs the {missing.name} package. Install the pinned packages first.\n")
    print("    uv pip install --python .venv/bin/python -r requirements.txt\n")
    raise SystemExit(1)

Tensor = torch.Tensor

# Loading the exercise files would otherwise leave a __pycache__ folder in every chapter.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
EXERCISES_DIR = ROOT / "exercises"
SOLUTIONS_DIR = ROOT / "solutions"
STATE_FILE = ROOT / ".tensorlings-state.json"
START_COMMAND = ".venv/bin/python run.py"

CODE_THEME = "monokai"
ACCENT = "cyan"


def prose(text: str) -> str:
    """Dedent a triple quoted block and trim the blank edges."""
    return textwrap.dedent(text).strip("\n")


def source_file(text: str) -> str:
    return prose(text) + "\n"


def terminal_console() -> Console:
    """A console that always asks the terminal for its size. Rich freezes the size when
    COLUMNS or LINES is set, which some shells do, and then resizing the window breaks."""
    return Console(
        _environ={name: value for name, value in os.environ.items() if name not in ("COLUMNS", "LINES")},
    )


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Loading and checking the learner's code
# ---------------------------------------------------------------------------


class Problem(Exception):
    """The file cannot be checked yet. The message is shown to the learner as written."""


class Mismatch(Exception):
    """One test got a different result than it should. The message says how."""


class TookTooLong(Exception):
    """The learner's code ran past the time limit, usually a loop that never ends."""


# Long enough for any correct answer, short enough that a loop that never ends doesn't stall the app.
TIME_LIMIT_SECONDS = 5


def run_with_time_limit(run: Callable[[], object]) -> None:
    """Run code with a time limit. Checks run in a thread, where signals can't interrupt them,
    so a trace function raises TookTooLong inside the learner's code once time is up."""
    deadline = time.monotonic() + TIME_LIMIT_SECONDS

    def stop_when_late(frame: object, event: str, argument: object) -> object:
        if time.monotonic() > deadline:
            raise TookTooLong()

        return stop_when_late

    previous = sys.gettrace()
    sys.settrace(stop_when_late)

    try:
        run()
    finally:
        sys.settrace(previous)


# PyTorch's own wording for common mistakes, each with the same message in plain words.
PLAIN_ERRORS: tuple[tuple[str, str], ...] = (
    (
        r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)",
        "These two shapes cannot be broadcast together. Compared from the right, a size {0} meets a size {1}. "
        "Two sizes only fit when they are equal or one of them is 1.",
    ),
    (
        r"must have the same dtype",
        "@ needs both tensors to hold the same kind of number. One holds whole numbers and the other decimal "
        "numbers. Call .float() on the one with whole numbers.",
    ),
    (
        r"mean\(\): could not infer output dtype",
        ".mean() needs decimal numbers, and this tensor holds whole numbers. Call .float() on it first.",
    ),
    (
        r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)",
        "For @, the size at the end of the first shape must equal the size at the start of the second. Here {1} meets {2}.",
    ),
    (
        r"size mismatch, got input \((\d+)\), mat \((\d+)x(\d+)\), vec \((\d+)\)",
        "For @, every row of the first tensor needs one weight per entry. Rows have {2} entries and there are {3} weights.",
    ),
    (
        r"Dimension out of range",
        "The code asks for a dimension the tensor does not have. Check the tensor's shape.",
    ),
    (
        r"index (-?\d+) is out of bounds for dimension (\d+) with size (\d+)",
        "Position {0} does not exist along dimension {1}, which only has {2} entries. Counting starts at 0.",
    ),
    (
        r"too many indices for tensor of dimension (\d+)",
        "The code picks along more dimensions than the tensor has. It has {0}.",
    ),
    (
        r"tuple index out of range",
        "The code asks a shape for a size it does not have. A shape holds one size per dimension, "
        "so a tensor with fewer dimensions has a shorter shape.",
    ),
    (
        r"len\(\) of a 0-d tensor",
        "len() needs a tensor with at least one dimension, and this one has none.",
    ),
)


def plain_words(error: BaseException) -> str | None:
    for pattern, explanation in PLAIN_ERRORS:
        match = re.search(pattern, str(error))

        if match:
            return explanation.format(*match.groups())

    return None


def describe_crash(
    error: BaseException,
    path: Path,
) -> str:
    learner_frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if Path(frame.filename).resolve() == path.resolve()
    ]
    lines: list[str] = []

    if learner_frames:
        frame = learner_frames[-1]
        lines.append(f"{path.name}, line {frame.lineno}")
        lines.append(f"    {frame.line}")

    explanation = plain_words(error)

    if explanation is not None:
        lines.append(explanation)
        lines.append(f"PyTorch's original message, {type(error).__name__}: {error}")
    else:
        lines.append(f"{type(error).__name__}: {error}")

    return "\n".join(lines)


def load_module(
    path: Path,
    name: str,
) -> ModuleType:
    if not path.exists():
        raise Problem(f"{relative(path)} is missing. Press x to bring back the original.")

    spec = importlib.util.spec_from_file_location(
        name=f"tensorlings_{name}_{time.monotonic_ns()}",
        location=path,
    )
    module = importlib.util.module_from_spec(spec)

    try:
        run_with_time_limit(lambda: spec.loader.exec_module(module))
    except TookTooLong:
        raise Problem(f"{path.name} was still running after {TIME_LIMIT_SECONDS} seconds while loading. Look for a loop that never ends.") from None
    except SyntaxError as error:
        raise Problem(f'Python cannot read line {error.lineno} of {path.name}. Its message is "{error.msg}".') from None
    except Exception as error:
        crash = describe_crash(
            error=error,
            path=path,
        )
        raise Problem(f"{path.name} crashed while loading.\n\n{crash}") from None

    return module


@dataclass(frozen=True)
class TestResult:
    function: str
    description: str
    passed: bool
    detail: str = ""


class Checker:
    """Runs one exercise's tests against the learner's module and keeps every result."""

    def __init__(
        self,
        module: ModuleType,
        path: Path,
    ) -> None:
        self.module = module
        self.path = path
        self.results: list[TestResult] = []

    def test(
        self,
        function: str,
        description: str,
        run: Callable[[Callable[..., object]], None],
    ) -> None:
        """Run one test of one function. A missing function fails only its own tests."""
        found = getattr(self.module, function, None)

        if not callable(found):
            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=f"{self.path.name} has no function called {function}. Check the name in its def line.",
                ),
            )
            return

        try:
            run_with_time_limit(lambda: run(found))
        except TookTooLong:
            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=f"{function} was still running after {TIME_LIMIT_SECONDS} seconds. Look for a loop that never ends.",
                ),
            )
            return
        except Mismatch as mismatch:
            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=str(mismatch),
                ),
            )
            return
        except Exception as error:
            # A TODO left as ... makes code like LETTERS[...] fail, which means the function is not finished.
            if "ellipsis" in str(error):
                detail = f"{function} still has a ... where a TODO goes, so it is {NOT_STARTED}."
            else:
                detail = describe_crash(
                    error=error,
                    path=self.path,
                )

            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=detail,
                ),
            )
            return

        self.results.append(
            TestResult(
                function=function,
                description=description,
                passed=True,
            ),
        )


def shape_text(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"

    return "(" + ", ".join(str(size) for size in shape) + ")"


def value_text(value: object) -> str:
    """A value on one line, a tensor written the way PyTorch prints it."""
    if isinstance(value, Tensor):
        return re.sub(r"\n\s*", " ", str(value))

    return repr(value)


NOT_STARTED = "not written yet"

# A tensor with more entries than this is described by its differences, since printing it whole buries them.
SHOWN_ENTRIES = 12


def number_text(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.4g}"


def differences_text(
    actual: Tensor,
    expected: Tensor,
) -> str:
    different = ~torch.isclose(actual.double(), expected.double(), atol=1e-6)
    positions = different.nonzero().tolist()
    lines = [f"{len(positions)} of {expected.numel()} entries differ. The first ones are"]

    for position in positions[:5]:
        index = tuple(position)
        lines.append(f"  at {list(index)}, got {number_text(actual[index].item())}, expected {number_text(expected[index].item())}")

    return "\n".join(lines)


def reject_missing_return(
    call: str,
    actual: object,
) -> None:
    if actual is None:
        raise Mismatch(f"{call} returned None, which usually means it is {NOT_STARTED}.")


def expect_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} returned a Python {type(actual).__name__}. Turn it into a tensor with torch.tensor(...).")

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} returned a {type(actual).__name__}. It should return a tensor.")

    small = actual.numel() <= SHOWN_ENTRIES and expected.numel() <= SHOWN_ENTRIES

    if tuple(actual.shape) != tuple(expected.shape):
        message = f"{call} has shape {shape_text(tuple(actual.shape))}, but it should have shape {shape_text(tuple(expected.shape))}."

        if small:
            message += f"\ngot       {value_text(actual)}\nexpected  {value_text(expected)}"

        raise Mismatch(message)

    if torch.allclose(actual.double(), expected.double(), atol=1e-6):
        return

    if small:
        raise Mismatch(
            f"{call} has the right shape but different numbers.\n"
            f"got       {value_text(actual)}\n"
            f"expected  {value_text(expected)}",
        )

    raise Mismatch(f"{call} has the right shape but different numbers.\n" + differences_text(actual, expected))


def expect_value(
    call: str,
    actual: object,
    expected: object,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, Tensor) and actual.dim() == 0:
        actual = actual.item()

    if actual != expected:
        raise Mismatch(
            f"{call} returned {value_text(actual)}, but it should return {value_text(expected)}.",
        )


@dataclass(frozen=True)
class Outcome:
    problem: str | None
    results: list[TestResult]
    printed: str

    @property
    def passed(self) -> bool:
        return self.problem is None and bool(self.results) and all(result.passed for result in self.results)


# ---------------------------------------------------------------------------
# Exercises and chapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chapter:
    folder: str
    title: str
    readme: str


@dataclass(frozen=True)
class Exercise:
    name: str
    chapter: Chapter
    title: str
    lesson: str
    source: str
    solution: str
    # One hint per function, shown for the functions that don't pass yet.
    hints: dict[str, str]
    introduces: tuple[str, ...]
    check: Callable[[Checker], None]

    @property
    def path(self) -> Path:
        return EXERCISES_DIR / self.chapter.folder / f"{self.name}.py"

    @property
    def solution_path(self) -> Path:
        return SOLUTIONS_DIR / self.chapter.folder / f"{self.name}.py"


# Two checks never run the learner's code at once. Each one swaps out
# sys.stdout to catch what the code prints, and overlapping swaps would mix up.
CHECK_LOCK = threading.Lock()


def attempt(
    exercise: Exercise,
    path: Path,
) -> Outcome:
    printed = io.StringIO()

    with CHECK_LOCK, contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        try:
            module = load_module(
                path=path,
                name=exercise.name,
            )
        except Problem as problem:
            return Outcome(
                problem=str(problem),
                results=[],
                printed=printed.getvalue(),
            )

        checker = Checker(
            module=module,
            path=path,
        )

        try:
            exercise.check(checker)
        except Problem as problem:
            return Outcome(
                problem=str(problem),
                results=checker.results,
                printed=printed.getvalue(),
            )

    return Outcome(
        problem=None,
        results=checker.results,
        printed=printed.getvalue(),
    )


# ---------------------------------------------------------------------------
# 01_tensors
# ---------------------------------------------------------------------------

TENSORS = Chapter(
    folder="01_tensors",
    title="Tensors",
    readme=source_file(
        """
        # Tensors

        Every number a model reads, stores or produces lives in a tensor. This
        chapter teaches tensors by building the first model of Andrej Karpathy's
        makemore series. It counts which letter follows which in a list of
        names, turns the counts into probabilities and makes up new names.

        ## Further reading

        - [Karpathy, building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
        - [PyTorch, introduction to tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
        - [Dive into Deep Learning, data manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html)
        """,
    ),
)

# "." marks where a name starts and ends. Each other letter is its position in this string.
LETTERS = ".abcdefghijklmnopqrstuvwxyz"

# The names the checks build the model from. Every letter appears, so no row of counts is empty.
TRAINING_NAMES: tuple[str, ...] = (
    "emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia", "amelia", "harper", "evelyn",
    "abigail", "emily", "elizabeth", "mila", "ella", "avery", "sofia", "camila", "aria", "scarlett",
    "victoria", "madison", "luna", "grace", "chloe", "penelope", "layla", "riley", "zoey", "nora",
    "lily", "eleanor", "hannah", "lillian", "addison", "aubrey", "ellie", "stella", "natalie", "zoe",
    "liam", "noah", "oliver", "elijah", "william", "james", "benjamin", "lucas", "henry", "alexander",
    "mason", "michael", "ethan", "daniel", "jacob", "logan", "jackson", "levi", "sebastian", "mateo",
    "jack", "owen", "theodore", "aiden", "samuel", "joseph", "david", "wyatt", "matthew", "luke",
    "quinn", "xavier", "felix", "jude", "kai", "rex", "yusuf", "uma", "quentin", "zane",
)


def letter_positions(name: str) -> list[int]:
    return [LETTERS.index(letter) for letter in "." + name + "."]


def letter_pair_counts(names: tuple[str, ...] | list[str]) -> Tensor:
    """The reference count table the checks compare with, built the way the exercises build it."""
    counts = torch.zeros((len(LETTERS), len(LETTERS)), dtype=torch.int64)

    for name in names:
        positions = letter_positions(name)

        for first, second in zip(positions, positions[1:]):
            counts[first, second] += 1

    return counts


def expect_integers(
    call: str,
    actual: object,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} returned a {type(actual).__name__}. It should return a tensor.")

    if actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds floats. Pass dtype=torch.int64 so it holds integers.")


TENSORS1_LESSON = prose(
    """
    Over these six exercises you build the first model from Andrej Karpathy's
    makemore series. It reads a list of names, learns which letter tends to
    follow which, and makes up new names. Every tensor operation here is one
    that model needs. The last exercise rewrites it as the first piece of a
    neural network, the kind of model the rest of the course builds.

    A **model** is a program that turns input numbers into output numbers.
    **Training** adjusts the model's own numbers until its outputs match the
    examples it learns from. Those numbers live in tensors. A **tensor** is a
    grid of numbers, and `torch.tensor` builds one from a list, or from nested
    lists with one inner list per row.

    ```python
    import torch

    torch.tensor([7, 8, 9]).shape    # torch.Size([3])

    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    x.shape                          # torch.Size([2, 3])
    ```

    The **shape** gives the size of each **dimension**. `x` has 2 rows and 3
    columns, and the flat tensor before it has one dimension of size 3. This
    course writes shapes the short way, as `(2, 3)` and `(3,)`.

    A tensor holds one type of number. `torch.tensor` keeps the type of what
    you give it, so Python ints give a tensor of integers. `torch.zeros` makes a
    tensor of the shape you give it, filled with zeros, and those are floats
    unless you pass `dtype=torch.int64`.

    ```python
    torch.zeros((2, 4))                       # floats, shape (2, 4)
    torch.zeros((2, 4), dtype=torch.int64)    # integers, shape (2, 4)
    ```

    The model works on numbers, so each letter becomes its position in
    `LETTERS`. The `"."` at position 0 marks where a name starts and ends, so
    the name `"ab"` becomes `[0, 1, 2, 0]`.
    """,
)

TENSORS1_SOURCE = source_file(
    '''
    # tensors1.py

    import torch
    from torch import Tensor

    # "." marks where a name starts and ends. Each other letter is its position in this string.
    LETTERS = ".abcdefghijklmnopqrstuvwxyz"


    # TODO: Turn a name into a tensor of letter positions, with "." added at both ends.
    #   encode("ab") is tensor([0, 1, 2, 0]).
    def encode(name: str) -> Tensor:
        ...


    # TODO: Return a table of zeros with one row and one column for every letter in
    #   LETTERS, holding integers. The next exercise counts letter pairs into it.
    def empty_counts() -> Tensor:
        ...


    # What the check tests
    #   encode("ab") is [0, 1, 2, 0] and encode("emma") is [0, 5, 13, 13, 1, 0]
    #   empty_counts() is a 27 by 27 table of integer zeros
    ''',
)

TENSORS1_SOLUTION = source_file(
    '''
    # tensors1.py, a solution for comparison

    import torch
    from torch import Tensor

    LETTERS = ".abcdefghijklmnopqrstuvwxyz"


    def encode(name: str) -> Tensor:
        return torch.tensor([LETTERS.index(letter) for letter in "." + name + "."])


    def empty_counts() -> Tensor:
        return torch.zeros((len(LETTERS), len(LETTERS)), dtype=torch.int64)
    ''',
)


def check_tensors1(checker: Checker) -> None:
    for name in ("ab", "emma"):
        checker.test(
            function="encode",
            description=f'turns "{name}" into {letter_positions(name)}',
            run=lambda encode, name=name: expect_tensor(
                call=f'encode("{name}")',
                actual=encode(name),
                expected=torch.tensor(letter_positions(name)),
            ),
        )

    checker.test(
        function="empty_counts",
        description="is a 27 by 27 table of zeros",
        run=lambda empty_counts: expect_tensor(
            call="empty_counts()",
            actual=empty_counts(),
            expected=torch.zeros((27, 27), dtype=torch.int64),
        ),
    )
    checker.test(
        function="empty_counts",
        description="holds integers",
        run=lambda empty_counts: expect_integers(
            call="empty_counts()",
            actual=empty_counts(),
        ),
    )


TENSORS1 = Exercise(
    name="tensors1",
    chapter=TENSORS,
    title="Letters as numbers",
    lesson=TENSORS1_LESSON,
    source=TENSORS1_SOURCE,
    solution=TENSORS1_SOLUTION,
    hints={
        "encode": 'LETTERS.index(letter) gives a letter\'s position. Collect the positions of every letter in "." + '
        'name + "." in a list, then pass the list to torch.tensor.',
        "empty_counts": "torch.zeros takes the shape as a tuple. len(LETTERS) is the size of each side, and "
        "dtype=torch.int64 makes it hold integers.",
    },
    introduces=("model", "training", "tensor", "shape", "dimension"),
    check=check_tensors1,
)

LETTERS_AND_ENCODE = '''
    # "." marks where a name starts and ends. Each other letter is its position in this string.
    LETTERS = ".abcdefghijklmnopqrstuvwxyz"


    # From tensors1.
    def encode(name: str) -> Tensor:
        return torch.tensor([LETTERS.index(letter) for letter in "." + name + "."])
'''

TENSORS2_LESSON = prose(
    """
    Square brackets read parts of a tensor. Give one position per dimension,
    separated by commas. Positions start at 0, negative positions count from the
    end, and dimensions you leave out are taken whole.

    ```python
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    ```

    | Code | Result | Selects |
    |---|---|---|
    | `x[0]` | `tensor([1, 2, 3])` | row 0 |
    | `x[-1]` | `tensor([7, 8, 9])` | the last row |
    | `x[1, 2]` | `tensor(6)` | row 1, column 2 |
    | `x[:, 0]` | `tensor([1, 4, 7])` | column 0 |

    A single position is an **index**, and a bare `:` takes the whole
    dimension. `start:stop` takes a range and is called a **slice**, so `x[0:2]`
    is rows 0 and 1. An index removes its dimension and a slice keeps it.
    `x[0]` has shape `(3,)`, while `x[0:1]` has shape `(1, 3)`.

    `.item()` turns a tensor with one entry into a plain Python number, so
    `x[1, 2].item()` is `6`. A tensor with one entry also works as a position,
    so `x[torch.tensor(1), 2]` is the same entry as `x[1, 2]`.

    The same brackets write. Assigning to an entry changes it in place, and `+=`
    adds to it.

    ```python
    x[1, 2] = 0     # x[1, 2] is now 0
    x[0, 0] += 10   # x[0, 0] is now 11
    ```

    That is how the model learns from names. Its table has one row and one
    column per letter. Take every pair of neighbouring letters in every name,
    such as `"e"` then `"m"` in `"emma"`. The entry at that row and column goes
    up by one. Row 0 counts the letters that start a name, since every name
    starts with `"."`.
    """,
)

TENSORS2_SOURCE = source_file(
    """
    # tensors2.py

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # A few names to try your functions on. The check uses more.
    NAMES = ["emma", "olivia", "ava", "isabella", "sophia"]


    # TODO: Count every pair of neighbouring letters in the names.
    #   counts[a, b] ends up as the number of times letter b follows letter a.
    def count_pairs(names: list[str]) -> Tensor:
        counts = torch.zeros((len(LETTERS), len(LETTERS)), dtype=torch.int64)

        for name in names:
            letters = encode(name)

            for first, second in zip(letters, letters[1:]):
                # TODO: add one to the entry for this pair
                ...

        return counts


    # TODO: Return how many times each letter starts a name, one count per letter in LETTERS.
    def first_letter_counts(counts: Tensor) -> Tensor:
        ...


    # TODO: Return how many times the second letter of the pair follows the first, as an int.
    #   count_of(counts, "em") is how often "m" comes right after "e".
    def count_of(
        counts: Tensor,
        pair: str,
    ) -> int:
        ...


    # What the check tests
    #   count_pairs(["ab"]) counts the pairs ".a", "ab" and "b."
    #   count_pairs on 84 names
    #   first_letter_counts and count_of on the counts of those names
    ''',
)

TENSORS2_SOLUTION = source_file(
    """
    # tensors2.py, a solution for comparison

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    def count_pairs(names: list[str]) -> Tensor:
        counts = torch.zeros((len(LETTERS), len(LETTERS)), dtype=torch.int64)

        for name in names:
            letters = encode(name)

            for first, second in zip(letters, letters[1:]):
                counts[first, second] += 1

        return counts


    # Every name starts with ".", which is position 0, so row 0 holds the first letters.
    def first_letter_counts(counts: Tensor) -> Tensor:
        return counts[0]


    def count_of(
        counts: Tensor,
        pair: str,
    ) -> int:
        return counts[LETTERS.index(pair[0]), LETTERS.index(pair[1])].item()
    ''',
)


def check_tensors2(checker: Checker) -> None:
    counts = letter_pair_counts(TRAINING_NAMES)

    checker.test(
        function="count_pairs",
        description='counts ".a", "ab" and "b." for the name "ab"',
        run=lambda count_pairs: expect_tensor(
            call='count_pairs(["ab"])',
            actual=count_pairs(["ab"]),
            expected=letter_pair_counts(["ab"]),
        ),
    )
    checker.test(
        function="count_pairs",
        description=f"counts every pair in {len(TRAINING_NAMES)} names",
        run=lambda count_pairs: expect_tensor(
            call=f"count_pairs on {len(TRAINING_NAMES)} names",
            actual=count_pairs(list(TRAINING_NAMES)),
            expected=counts,
        ),
    )
    checker.test(
        function="first_letter_counts",
        description="gives row 0 of the counts",
        run=lambda first_letter_counts: expect_tensor(
            call="first_letter_counts(counts)",
            actual=first_letter_counts(counts),
            expected=counts[0],
        ),
    )

    for pair in ("em", "a.", ".j"):
        checker.test(
            function="count_of",
            description=f'gives the count for "{pair}"',
            run=lambda count_of, pair=pair: expect_value(
                call=f'count_of(counts, "{pair}")',
                actual=count_of(counts, pair),
                expected=counts[LETTERS.index(pair[0]), LETTERS.index(pair[1])].item(),
            ),
        )


TENSORS2 = Exercise(
    name="tensors2",
    chapter=TENSORS,
    title="Counting letter pairs",
    lesson=TENSORS2_LESSON,
    source=TENSORS2_SOURCE,
    solution=TENSORS2_SOLUTION,
    hints={
        "count_pairs": "first and second are already the row and the column of the pair. Use += on that entry of "
        "counts.",
        "first_letter_counts": 'Every name starts with ".", which is position 0. The pairs that start a name are all '
        "in one row.",
        "count_of": "Turn each letter of pair into its position with LETTERS.index, read that entry of counts, and "
        "turn it into an int with .item().",
    },
    introduces=("index", "slice"),
    check=check_tensors2,
)

TENSORS3_LESSON = prose(
    """
    The `.sum()` method adds up entries. With no arguments it adds all of them.
    With `dim` it adds along one dimension, and that dimension disappears from
    the shape.

    ```python
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    x.sum()         # tensor(21)
    x.sum(dim=0)    # tensor([5, 7, 9]), one total per column
    x.sum(dim=1)    # tensor([ 6, 15]), one total per row
    ```

    `keepdim=True` keeps the summed dimension with size 1, so
    `x.sum(dim=1, keepdim=True)` is `tensor([[ 6], [15]])`, shape `(2, 1)`. The
    next exercise needs that.

    `.max()` gives the largest entry and `.argmax()` gives its position.
    `.argmax()` also takes `dim`.

    ```python
    y = torch.tensor([3, 9, 4])
    y.max()            # tensor(9)
    y.argmax()         # tensor(1)
    x.argmax(dim=1)    # tensor([2, 2]), the column of the largest entry in each row
    ```

    `.mean()` takes `dim` the way `.sum()` does, but it only works on floats, so
    `x.float().mean()` is `tensor(3.5000)`.

    Operations like these, which combine many entries into fewer, are called
    **reductions**. In the table of counts, the sum of a row says how many pairs
    start with that letter. The argmax of a row is the position of the letter
    that most often comes next.
    """,
)

TENSORS3_SOURCE = source_file(
    """
    # tensors3.py

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # Every function here gets counts, the table count_pairs made in tensors2.


    # TODO: Return how many pairs start with each letter, one total per letter in LETTERS.
    def pairs_starting_with(counts: Tensor) -> Tensor:
        ...


    # TODO: Return the letter that most often comes right after the given letter.
    #   most_likely_next(counts, "q") is "u" for the names the check uses.
    def most_likely_next(
        counts: Tensor,
        letter: str,
    ) -> str:
        ...


    # TODO: Return the total number of letter pairs counted, as an int.
    def total_pairs(counts: Tensor) -> int:
        ...


    # What the check tests, on the counts of 84 names
    #   pairs_starting_with gives one total per letter
    #   most_likely_next for ".", "q", "e" and "l"
    #   total_pairs gives the number of pairs in all the names
    ''',
)

TENSORS3_SOLUTION = source_file(
    """
    # tensors3.py, a solution for comparison

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # A pair that starts with a letter sits in that letter's row.
    def pairs_starting_with(counts: Tensor) -> Tensor:
        return counts.sum(dim=1)


    def most_likely_next(
        counts: Tensor,
        letter: str,
    ) -> str:
        return LETTERS[counts[LETTERS.index(letter)].argmax().item()]


    def total_pairs(counts: Tensor) -> int:
        return counts.sum().item()
    ''',
)


def check_tensors3(checker: Checker) -> None:
    counts = letter_pair_counts(TRAINING_NAMES)

    # In a table of letter pairs the row sums equal the column sums, since every letter
    # starts as many pairs as it ends. A random table tells the two apart.
    random_counts = torch.randint(0, 10, (len(LETTERS), len(LETTERS)), generator=torch.Generator().manual_seed(0))

    checker.test(
        function="pairs_starting_with",
        description="gives the sum of each row",
        run=lambda pairs_starting_with: expect_tensor(
            call="pairs_starting_with(counts)",
            actual=pairs_starting_with(counts),
            expected=counts.sum(dim=1),
        ),
    )
    checker.test(
        function="pairs_starting_with",
        description="sums rows, not columns, on a table of random counts",
        run=lambda pairs_starting_with: expect_tensor(
            call="pairs_starting_with on a table of random counts",
            actual=pairs_starting_with(random_counts),
            expected=random_counts.sum(dim=1),
        ),
    )

    for letter in (".", "q", "e", "l"):
        checker.test(
            function="most_likely_next",
            description=f'gives the most likely letter after "{letter}"',
            run=lambda most_likely_next, letter=letter: expect_value(
                call=f'most_likely_next(counts, "{letter}")',
                actual=most_likely_next(counts, letter),
                expected=LETTERS[counts[LETTERS.index(letter)].argmax().item()],
            ),
        )

    checker.test(
        function="total_pairs",
        description=f"counts all {counts.sum().item()} pairs",
        run=lambda total_pairs: expect_value(
            call="total_pairs(counts)",
            actual=total_pairs(counts),
            expected=counts.sum().item(),
        ),
    )


TENSORS3 = Exercise(
    name="tensors3",
    chapter=TENSORS,
    title="Adding up the counts",
    lesson=TENSORS3_LESSON,
    source=TENSORS3_SOURCE,
    solution=TENSORS3_SOLUTION,
    hints={
        "pairs_starting_with": "A pair that starts with a letter sits in that letter's row. Add up each row, which "
        "is a sum along dimension 1.",
        "most_likely_next": "Pick the letter's row with LETTERS.index, find where its largest count is with argmax, "
        "and turn that position back into a letter with LETTERS.",
        "total_pairs": "sum with no arguments adds every entry, and .item() turns the result into an int.",
    },
    introduces=("reduction",),
    check=check_tensors3,
)

TENSORS4_LESSON = prose(
    """
    Math between tensors of the same shape works entry by entry. This is
    **elementwise** math, and `+`, `-`, `*` and `/` all work this way. A plain
    number works with a tensor of any shape.

    ```python
    x = torch.tensor([[1, 2],
                      [3, 4]])
    x + x     # tensor([[2, 4], [6, 8]])
    x * 10    # tensor([[10, 20], [30, 40]])
    ```

    The shapes don't have to be equal. PyTorch lines them up from the right, and
    two sizes fit when they are equal or when one of them is 1. A size of 1 is
    stretched to match, and a missing size counts as 1. This is
    **broadcasting**.

    ```text
    (2, 2) with (2,)      the (2,) is stretched down the rows, one value per column
    (2, 2) with (2, 1)    the (2, 1) is stretched across the columns, one value per row
    ```

    ```python
    x / torch.tensor([[1], [10]])    # tensor([[1.0000, 2.0000], [0.3000, 0.4000]])
    ```

    Dividing with `/` always gives floats, even between integers.

    A square tensor hides a trap. `x.sum(dim=1)` has shape `(2,)`, so
    `x / x.sum(dim=1)` runs without an error. But `(2,)` lines up with the
    columns, so each column is divided by one row's total. The numbers are
    wrong and nothing says so. With `keepdim=True` the totals have shape
    `(2, 1)` and line up with the rows. Karpathy hits this exact bug in his
    makemore video.

    Dividing each row of the counts by its own total turns them into
    probabilities. Row `a` then holds the chance of each letter coming right
    after `a`, and every row adds up to 1.

    A count of zero becomes a probability of zero. The next exercise scores the
    model with the log of each probability, and the log of 0 is minus infinity,
    which breaks the score. Adding a small number to every count first keeps
    every probability above zero. This is called smoothing.
    """,
)

TENSORS4_SOURCE = source_file(
    """
    # tensors4.py

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # Every function here gets counts, the table count_pairs made in tensors2.


    # TODO: Turn the counts into probabilities. Divide each row by its own total,
    #   so probs[a, b] is the chance that letter b comes right after letter a.
    def to_probabilities(counts: Tensor) -> Tensor:
        ...


    # TODO: Add k to every count, then turn the result into probabilities.
    def smoothed_probabilities(
        counts: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # What the check tests
    #   to_probabilities of a 3 by 3 table, where a wrong shape still runs
    #   to_probabilities of the counts of 84 names, with every row adding up to 1
    #   smoothed_probabilities of those counts with k = 1
    ''',
)

TENSORS4_SOLUTION = source_file(
    """
    # tensors4.py, a solution for comparison

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # keepdim=True gives the row totals shape (27, 1), so they line up with the rows.
    def to_probabilities(counts: Tensor) -> Tensor:
        return counts / counts.sum(dim=1, keepdim=True)


    def smoothed_probabilities(
        counts: Tensor,
        k: int,
    ) -> Tensor:
        return to_probabilities(counts + k)
    ''',
)


def row_probabilities(counts: Tensor) -> Tensor:
    return counts / counts.sum(dim=1, keepdim=True)


def check_tensors4(checker: Checker) -> None:
    small = torch.tensor([[1, 1, 2], [0, 3, 1], [5, 0, 5]])
    counts = letter_pair_counts(TRAINING_NAMES)

    checker.test(
        function="to_probabilities",
        description="divides each row of a 3 by 3 table by its own total",
        run=lambda to_probabilities: expect_tensor(
            call="to_probabilities(torch.tensor([[1, 1, 2], [0, 3, 1], [5, 0, 5]]))",
            actual=to_probabilities(small),
            expected=row_probabilities(small),
        ),
    )
    checker.test(
        function="to_probabilities",
        description=f"gives rows that add up to 1 for the counts of {len(TRAINING_NAMES)} names",
        run=lambda to_probabilities: expect_tensor(
            call="to_probabilities(counts)",
            actual=to_probabilities(counts),
            expected=row_probabilities(counts),
        ),
    )
    checker.test(
        function="smoothed_probabilities",
        description="adds 1 to every count before dividing",
        run=lambda smoothed_probabilities: expect_tensor(
            call="smoothed_probabilities(counts, 1)",
            actual=smoothed_probabilities(counts, 1),
            expected=row_probabilities(counts + 1),
        ),
    )


TENSORS4 = Exercise(
    name="tensors4",
    chapter=TENSORS,
    title="Counts to probabilities",
    lesson=TENSORS4_LESSON,
    source=TENSORS4_SOURCE,
    solution=TENSORS4_SOLUTION,
    hints={
        "to_probabilities": "Each row's total is a sum along dimension 1. Keep that dimension so the totals have "
        "shape (27, 1), then divide the counts by them.",
        "smoothed_probabilities": "counts + k adds k to every entry. Then divide each row by its total, as "
        "to_probabilities does. You can call it.",
    },
    introduces=("elementwise", "broadcasting"),
    check=check_tensors4,
)

TENSORS5_LESSON = prose(
    """
    A row of probabilities says how likely each next letter is.
    `torch.multinomial` draws positions at random, each as often as its
    probability says. A `torch.Generator` with a fixed seed makes the draws
    repeatable, so the same seed gives the same draws every time.

    ```python
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.0, 0.5])
    torch.multinomial(p, num_samples=8, replacement=True, generator=g)
    # tensor([2, 2, 0, 2, 2, 2, 0, 0]), position 1 never comes up
    ```

    `replacement=True` lets the same position come up more than once. With
    `num_samples=1` it draws a single position, and `.item()` turns that into an
    int. To make up a name, start at position 0, the `"."`, and draw the next
    position from row 0. Then draw from that letter's row, and keep going until
    `"."` comes up again.

    To score the model, look at the probability it gives each pair in real
    names. The chance it gives a whole name is the product of the chances of
    its pairs, and a good model gives real names high chances. A product of
    many probabilities is too small to work with, so models add their logs
    instead. `.log()` takes the natural log of every entry, and the log of a
    probability is 0 or below. `-t` flips the sign of every entry of `t`.

    Indexing with two tensors of positions picks one entry per pair of
    positions. Python lists of positions work the same way, and `.tolist()`
    turns a tensor into a Python list.

    ```python
    q = torch.tensor([[0.5, 0.5],
                      [0.9, 0.1]])
    rows = torch.tensor([0, 1, 1])
    columns = torch.tensor([1, 0, 0])
    q[rows, columns]          # tensor([0.5000, 0.9000, 0.9000])
    q[rows, columns].log()    # tensor([-0.6931, -0.1054, -0.1054])
    ```

    The mean of the negative logs is the model's **loss**, one number saying
    how wrong it is. Lower is better, and training means pushing it down. A
    model that gave every letter the same chance would score `-log(1/27)`, about
    3.30.
    """,
)

TENSORS5_SOURCE = source_file(
    """
    # tensors5.py

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # Every function here gets probs, a table smoothed_probabilities made in tensors4.


    # TODO: Make up a name by drawing one letter at a time from probs.
    def sample_name(
        probs: Tensor,
        generator: torch.Generator,
    ) -> str:
        name = ""
        position = 0

        while True:
            # TODO: draw the next position from the row for the current position, as an int
            position = ...

            if position == 0:
                return name

            name += LETTERS[position]


    # TODO: Return the loss of probs on the names, the mean negative log probability of every pair.
    def loss(
        probs: Tensor,
        names: list[str],
    ) -> Tensor:
        firsts = []
        seconds = []

        for name in names:
            letters = encode(name)
            firsts.extend(letters[:-1].tolist())
            seconds.extend(letters[1:].tolist())

        # TODO: pick the probability of every pair, then return the mean of their negative logs
        ...


    # What the check tests
    #   sample_name on a table where "a" is the only possible name
    #   sample_name makes the same 5 names as the reference with seed 2147483647
    #   loss is log(27), about 3.30, for a table where every letter is equally likely
    #   loss of the smoothed probabilities of 84 names on those names
    ''',
)

TENSORS5_SOLUTION = source_file(
    """
    # tensors5.py, a solution for comparison

    import torch
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    def sample_name(
        probs: Tensor,
        generator: torch.Generator,
    ) -> str:
        name = ""
        position = 0

        while True:
            position = torch.multinomial(probs[position], num_samples=1, generator=generator).item()

            if position == 0:
                return name

            name += LETTERS[position]


    def loss(
        probs: Tensor,
        names: list[str],
    ) -> Tensor:
        firsts = []
        seconds = []

        for name in names:
            letters = encode(name)
            firsts.extend(letters[:-1].tolist())
            seconds.extend(letters[1:].tolist())

        return -probs[firsts, seconds].log().mean()
    ''',
)

# The seed Karpathy uses in makemore.
SAMPLING_SEED = 2147483647


def reference_names(
    probs: Tensor,
    count: int,
) -> list[str]:
    generator = torch.Generator().manual_seed(SAMPLING_SEED)
    names: list[str] = []

    for _ in range(count):
        name = ""
        position = 0

        while True:
            position = torch.multinomial(probs[position], num_samples=1, generator=generator).item()

            if position == 0:
                break

            name += LETTERS[position]

        names.append(name)

    return names


def reference_loss(
    probs: Tensor,
    names: tuple[str, ...],
) -> Tensor:
    pairs = [pair for name in names for pair in zip(letter_positions(name), letter_positions(name)[1:])]
    firsts = [first for first, _ in pairs]
    seconds = [second for _, second in pairs]

    return -probs[firsts, seconds].log().mean()


def check_tensors5(checker: Checker) -> None:
    probs = row_probabilities(letter_pair_counts(TRAINING_NAMES) + 1)
    only_a = torch.zeros((len(LETTERS), len(LETTERS)))
    only_a[0, LETTERS.index("a")] = 1.0
    only_a[LETTERS.index("a"), 0] = 1.0
    even = torch.full((len(LETTERS), len(LETTERS)), 1 / len(LETTERS))

    def five_names(sample_name: Callable[..., object]) -> list[object]:
        generator = torch.Generator().manual_seed(SAMPLING_SEED)
        return [sample_name(probs, generator) for _ in range(5)]

    checker.test(
        function="sample_name",
        description='makes "a" when it is the only possible name',
        run=lambda sample_name: expect_value(
            call="sample_name on a table that only allows the name a",
            actual=sample_name(only_a, torch.Generator().manual_seed(0)),
            expected="a",
        ),
    )
    checker.test(
        function="sample_name",
        description="makes the same 5 names as the reference with the same seed",
        run=lambda sample_name: expect_value(
            call=f"five calls of sample_name with seed {SAMPLING_SEED}",
            actual=five_names(sample_name),
            expected=reference_names(probs, 5),
        ),
    )
    checker.test(
        function="loss",
        description="is about 3.30 when every letter is equally likely",
        run=lambda loss: expect_tensor(
            call="loss on a table where every letter is equally likely",
            actual=loss(even, list(TRAINING_NAMES)),
            expected=reference_loss(even, TRAINING_NAMES),
        ),
    )
    checker.test(
        function="loss",
        description=f"matches the reference on {len(TRAINING_NAMES)} names",
        run=lambda loss: expect_tensor(
            call=f"loss(probs, names) on {len(TRAINING_NAMES)} names",
            actual=loss(probs, list(TRAINING_NAMES)),
            expected=reference_loss(probs, TRAINING_NAMES),
        ),
    )


TENSORS5 = Exercise(
    name="tensors5",
    chapter=TENSORS,
    title="Making up names",
    lesson=TENSORS5_LESSON,
    source=TENSORS5_SOURCE,
    solution=TENSORS5_SOLUTION,
    hints={
        "sample_name": "probs[position] is the row for the current letter. torch.multinomial with num_samples=1 "
        "and generator=generator draws one position, and .item() turns it into an int.",
        "loss": "probs[firsts, seconds] picks one probability per pair. Take .log(), then .mean(), and put a minus "
        "sign in front.",
    },
    introduces=("loss",),
    check=check_tensors5,
)


# ---------------------------------------------------------------------------
# 02_matmul
# ---------------------------------------------------------------------------

MATMUL = Chapter(
    folder="02_matmul",
    title="Matrix multiplication",
    readme=source_file(
        """
        # Matrix multiplication

        A weighted sum multiplies each input by how much it counts and adds the
        results. Matrix multiplication does many weighted sums in one call, and
        it is the operation most of deep learning is built on. This chapter
        rewrites the bigram model from the last chapter as a matrix
        multiplication, which is the first layer of a neural network.

        ## Further reading

        - [Dive into Deep Learning, linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
        - [3Blue1Brown, the essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
        """,
    ),
)

MATMUL1_LESSON = prose(
    """
    A **weighted sum** multiplies each value by its own **weight** and adds the
    results. With tensors that is an elementwise multiply and a sum, and the
    **dot product**, written `@`, does both in one step.

    ```python
    values = torch.tensor([2.0, 3.0])
    weights = torch.tensor([10.0, 100.0])
    (values * weights).sum()    # tensor(320.)
    values @ weights            # tensor(320.)
    ```

    A tensor with two dimensions is also called a **matrix**. With a matrix on
    each side, `@` takes the dot product of every row on the left with every
    column on the right. This is **matrix multiplication**. `(n, k) @ (k, m)`
    gives `(n, m)`, the inner sizes must match, and both sides must have the
    same type.

    ```python
    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0],
                      [0.0, 10.0]])
    a @ b    # tensor([[ 1., 20.], [ 3., 40.]])
    ```

    A one-hot row is all zeros except for a single 1. `torch.nn.functional`
    holds the functions neural networks are built from, and its
    `one_hot(positions, num_classes)` makes one such row per position. It
    returns integers, so add `.float()` before multiplying it with floats.

    ```python
    import torch.nn.functional as F

    F.one_hot(torch.tensor([2, 0]), num_classes=3).float()
    # tensor([[0., 0., 1.],
    #         [1., 0., 0.]])
    ```

    Indexing a matrix with one tensor of positions picks whole rows, so
    `a[torch.tensor([1, 0])]` is row 1 then row 0. A one-hot row times a matrix
    picks out one row too, since every other row gets multiplied by 0. So
    reading row `i` of the probability table is the same as multiplying the
    one-hot row for `i` by the table.

    That is the step from a table to a neural network. A **layer** holds a
    matrix of weights and multiplies its input by it, and a neural network is a
    model made of layers. The bigram model is one layer whose weights were set
    by counting. In the next part the weights are
    learned instead, by pushing the loss down, and the same layer becomes the
    building block of larger models.
    """,
)

MATMUL1_SOURCE = source_file(
    """
    # matmul1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    # TODO: Return the one-hot rows for the positions, as floats, with one column
    #   for every letter in LETTERS.
    def one_hot(positions: Tensor) -> Tensor:
        ...


    # TODO: Return the rows of table at the positions, using one_hot and @ instead
    #   of indexing. look_up(positions, table) must equal table[positions].
    def look_up(
        positions: Tensor,
        table: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   one_hot(torch.tensor([2, 0, 26])) has shape (3, 27) and holds floats
    #   look_up gives the same rows as indexing, for a random 27 by 27 table
    ''',
)

MATMUL1_SOLUTION = source_file(
    """
    # matmul1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor
    """
    + LETTERS_AND_ENCODE
    + '''

    def one_hot(positions: Tensor) -> Tensor:
        return F.one_hot(positions, num_classes=len(LETTERS)).float()


    # (n, 27) @ (27, 27) gives (n, 27), one row of table per position.
    def look_up(
        positions: Tensor,
        table: Tensor,
    ) -> Tensor:
        return one_hot(positions) @ table
    ''',
)


def expect_floats(
    call: str,
    actual: object,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} returned a {type(actual).__name__}. It should return a tensor.")

    if not actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds integers. Add .float() so it holds floats.")


def check_matmul1(checker: Checker) -> None:
    positions = torch.tensor([2, 0, 26])
    table = torch.rand((len(LETTERS), len(LETTERS)), generator=torch.Generator().manual_seed(0))
    name_positions = torch.tensor(letter_positions("emma"))

    checker.test(
        function="one_hot",
        description="gives one row per position, with a single 1 in each",
        run=lambda one_hot: expect_tensor(
            call="one_hot(torch.tensor([2, 0, 26]))",
            actual=one_hot(positions),
            expected=torch.nn.functional.one_hot(positions, num_classes=len(LETTERS)).float(),
        ),
    )
    checker.test(
        function="one_hot",
        description="holds floats",
        run=lambda one_hot: expect_floats(
            call="one_hot(torch.tensor([2, 0, 26]))",
            actual=one_hot(positions),
        ),
    )
    checker.test(
        function="look_up",
        description="gives the same rows as table[positions]",
        run=lambda look_up: expect_tensor(
            call="look_up(torch.tensor([2, 0, 26]), table)",
            actual=look_up(positions, table),
            expected=table[positions],
        ),
    )
    checker.test(
        function="look_up",
        description='gives the rows for the letters of "emma"',
        run=lambda look_up: expect_tensor(
            call='look_up(encode("emma"), table)',
            actual=look_up(name_positions, table),
            expected=table[name_positions],
        ),
    )


MATMUL1 = Exercise(
    name="matmul1",
    chapter=MATMUL,
    title="A lookup is a matrix multiplication",
    lesson=MATMUL1_LESSON,
    source=MATMUL1_SOURCE,
    solution=MATMUL1_SOLUTION,
    hints={
        "one_hot": "F.one_hot needs num_classes=len(LETTERS), and .float() turns the result into floats.",
        "look_up": "one_hot(positions) has shape (n, 27) and table has shape (27, 27). Multiply them with @.",
    },
    introduces=("weight", "weighted sum", "dot product", "matrix", "matrix multiplication", "layer"),
    check=check_matmul1,
)


# ---------------------------------------------------------------------------
# The course, in order
# ---------------------------------------------------------------------------

CHAPTERS: list[Chapter] = [
    TENSORS,
    MATMUL,
]

EXERCISES: list[Exercise] = [
    TENSORS1,
    TENSORS2,
    TENSORS3,
    TENSORS4,
    TENSORS5,
    MATMUL1,
]

COURSE_README = source_file(
    """
    # tensorlings exercises

    Start the course from the project folder.

    ```bash
    .venv/bin/python run.py
    ```

    Each chapter folder has a README with further reading. You can skip them.
    Every lesson you need shows in the app next to the exercise.
    """,
)

WELCOME = prose(
    """
    tensorlings teaches deep learning from the ground up. You start with
    tensors, the grids of numbers every model is made of, and work up to
    building transformers and designing modules of your own.

    **How it works**

    - Each exercise is one Python file in the `exercises` folder.
    - This screen shows the lesson. Keep your editor open beside it.
    - Fill in the code under each `TODO` and save. The app checks your code every time you save.
    - When every function passes, press `n` for the next exercise.

    **Keys**

    - `1`, `2` and `3` show the lesson, the hints and the result. After each save the result shows by itself.
    - The hints tab has one hint for each function that doesn't pass yet. `h` opens it too.
    - The arrow keys and the mouse wheel scroll. `l` lists every exercise, `x` resets the file, and `q` quits.
    - To copy, hold `option` while you drag over the text, then press `cmd+c`. In macOS Terminal hold `fn` instead.
    """,
)

FINISH = prose(
    """
    You finished every exercise that exists so far.

    The course grows in phases, from tensors to transformers to modules you
    design yourself. New exercises show up in the list as soon as they are
    added. Run the app again to continue.
    """,
)


# ---------------------------------------------------------------------------
# Lesson lint, part of the maintainer check
# ---------------------------------------------------------------------------

# Terms a beginner cannot be expected to know. Each must be introduced by a
# lesson, through Exercise.introduces, before any lesson or hint uses it.
JARGON: tuple[str, ...] = (
    "model",
    "training",
    "tensor",
    "shape",
    "dimension",
    "index",
    "slice",
    "elementwise",
    "batch",
    "broadcasting",
    "reduction",
    "weight",
    "weighted sum",
    "dot product",
    "matrix",
    "matrix multiplication",
    "vector",
    "scalar",
    "gradient",
    "derivative",
    "loss",
    "layer",
    "neuron",
    "activation",
    "embedding",
    "softmax",
    "parameter",
    "bias",
    "logit",
    "optimizer",
    "epoch",
    "backpropagation",
    "autograd",
    "attention",
    "transformer",
    "convolution",
)

FILLER: tuple[str, ...] = (
    "simply",
    "just",
    "basically",
    "obviously",
    "clearly",
    "easily",
    "essentially",
    "of course",
    "note that",
    "it is important",
    "delve",
    "leverage",
    "utilize",
    "crucial",
    "robust",
    "seamless",
    "unlock",
    "pivotal",
)

LONGEST_SENTENCE = 25
TALLEST_LESSON = 60
LESSON_WIDTH = 96

# A paragraph that opens with a bold label, like "**Why it exists.**", reads as a filled in template.
BOLD_LABEL = re.compile(r"^\*\*[^*\n]+[.:]\*\*", re.MULTILINE)

# A solution shorter than this many code tokens is too generic to count as copied, like "x @ y".
SHORTEST_ANSWER = 6

CODE_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d*)?|\S")


def prose_sentences(text: str) -> list[str]:
    """The sentences of a lesson, with code blocks and inline code taken out."""
    without_blocks = re.sub(
        pattern=r"```.*?```",
        repl="",
        string=text,
        flags=re.DOTALL,
    )
    without_code = re.sub(
        pattern=r"`[^`]*`",
        repl="CODE",
        string=without_blocks,
    ).replace("**", "")
    sentences: list[str] = []

    for block in re.split(r"\n\s*\n", without_code):
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if lines and lines[0].startswith("|"):
            sentences.extend(line.replace("|", " ") for line in lines if not set(line) <= set("|-: "))
            continue

        if lines and lines[0].startswith("- "):
            sentences.extend(line[2:] for line in lines)
            continue

        sentences.extend(re.split(r"(?<=[.!?])\s+", " ".join(lines)))

    return [sentence for sentence in sentences if sentence.strip()]


def mentions(
    text: str,
    term: str,
) -> bool:
    return re.search(rf"\b{re.escape(term)}(s|es)?\b", text, flags=re.IGNORECASE) is not None


def code_tokens(code: str) -> list[str]:
    """Code as tokens, so spacing and comments don't hide a copy."""
    tokens: list[str] = []

    for line in code.splitlines():
        tokens.extend(CODE_TOKEN.findall(line.split("#")[0]))

    return tokens


def solution_bodies(solution: str) -> dict[str, list[str]]:
    """Each solution function's body as code tokens, with the return keyword left out."""
    bodies: dict[str, list[str]] = {}

    for node in ast.parse(solution).body:
        if not isinstance(node, ast.FunctionDef):
            continue

        tokens: list[str] = []

        for statement in node.body:
            if isinstance(statement, ast.Return) and statement.value is not None:
                statement = statement.value

            tokens.extend(code_tokens(ast.get_source_segment(solution, statement)))

        bodies[node.name] = tokens

    return bodies


def shown_code(text: str) -> str:
    """Every code block and inline code span of a text, as tokens, with a bar between the pieces."""
    pieces = re.findall(r"```.*?```|`[^`\n]+`", text, flags=re.DOTALL)
    tokens: list[str] = []

    for piece in pieces:
        code = piece.strip("`").removeprefix("python").removeprefix("text")
        tokens.extend(code_tokens(code))
        tokens.append("|")

    return " " + " ".join(tokens) + " "


def lint_exercise(
    exercise: Exercise,
    introduced_before: set[str],
    console: Console,
) -> list[str]:
    findings: list[str] = []
    known = introduced_before | set(exercise.introduces)
    # Hints are plain text, so code in them like LETTERS.index is taken out before the prose checks.
    sentences = prose_sentences(exercise.lesson) + [
        sentence for hint in exercise.hints.values() for sentence in prose_sentences(re.sub(r"\w+\.\w+", "CODE", hint))
    ]
    readable = " ".join(sentences)

    for mark in ("—", "–", ";"):
        if mark in readable:
            findings.append(f"uses '{mark}' in prose")

    for sentence in sentences:
        word_count = len(sentence.split())

        if word_count > LONGEST_SENTENCE:
            findings.append(f"a {word_count} word sentence, '{sentence[:70]}...'")

    for word in FILLER:
        if mentions(readable, word):
            findings.append(f"uses the filler word '{word}'")

    for term in JARGON:
        if term not in known and mentions(readable, term):
            findings.append(f"uses '{term}' before a lesson introduces it")

    for label in BOLD_LABEL.findall(exercise.lesson):
        findings.append(f"opens a paragraph with the bold label {label}")

    # The lesson teaches the tools on its own examples. It and the hints never show a task's line as written.
    shown = shown_code(exercise.lesson) + " | " + " ".join(code_tokens(" ".join(exercise.hints.values()))) + " "

    for function, body in solution_bodies(exercise.solution).items():
        if function not in exercise.hints:
            continue

        if len(body) >= SHORTEST_ANSWER and f" {' '.join(body)} " in shown:
            findings.append(f"shows the answer to {function}")

    height = len(
        console.render_lines(
            Markdown(exercise.lesson, code_theme=CODE_THEME),
            console.options.update_width(LESSON_WIDTH),
            pad=False,
        ),
    )

    if height > TALLEST_LESSON:
        findings.append(f"is {height} lines tall at width {LESSON_WIDTH}, more than {TALLEST_LESSON}")

    return findings


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


@dataclass
class Progress:
    done: set[str]
    current: str
    welcomed: bool


def exercise_named(name: str) -> Exercise | None:
    for exercise in EXERCISES:
        if name in (exercise.name, f"{exercise.name}.py", relative(exercise.path)):
            return exercise

    return None


def load_progress() -> Progress:
    names = [exercise.name for exercise in EXERCISES]

    try:
        saved = json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}

    current = saved.get("current")

    return Progress(
        done={name for name in saved.get("done", []) if name in names},
        current=current if current in names else names[0],
        welcomed=bool(saved.get("welcomed", False)),
    )


def save_progress(progress: Progress) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {
                "current": progress.current,
                "done": sorted(progress.done),
                "welcomed": progress.welcomed,
            },
            indent=2,
        )
        + "\n",
    )


def next_pending(progress: Progress) -> Exercise | None:
    """The first pending exercise after the current one, wrapping around, as rustlings does."""
    names = [exercise.name for exercise in EXERCISES]
    current_index = names.index(progress.current)

    for exercise in EXERCISES[current_index + 1 :] + EXERCISES[:current_index]:
        if exercise.name not in progress.done:
            return exercise

    return None


def record_outcome(
    exercise: Exercise,
    outcome: Outcome,
    progress: Progress,
) -> None:
    if not outcome.passed:
        progress.done.discard(exercise.name)
        save_progress(progress)
        return

    progress.done.add(exercise.name)
    save_progress(progress)
    exercise.solution_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    exercise.solution_path.write_text(exercise.solution)


def reset_exercise(
    exercise: Exercise,
    progress: Progress,
) -> None:
    exercise.path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    exercise.path.write_text(exercise.source)
    progress.done.discard(exercise.name)
    save_progress(progress)


def check_all(progress: Progress) -> list[Exercise]:
    failing: list[Exercise] = []

    for exercise in EXERCISES:
        outcome = attempt(
            exercise=exercise,
            path=exercise.path,
        )
        record_outcome(
            exercise=exercise,
            outcome=outcome,
            progress=progress,
        )

        if not outcome.passed:
            failing.append(exercise)

    return failing


def modified_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# ---------------------------------------------------------------------------
# What the app and the run command draw
# ---------------------------------------------------------------------------


def indented(
    text: str,
    style: str = "",
    left: int = 4,
) -> Padding:
    return Padding(
        Text(text, style=style, overflow="fold"),
        pad=(0, 0, 0, left),
    )


def results_by_function(outcome: Outcome) -> dict[str, list[TestResult]]:
    functions: dict[str, list[TestResult]] = {}

    for result in outcome.results:
        functions.setdefault(result.function, []).append(result)

    return functions


def passing_functions(outcome: Outcome) -> int:
    return sum(all(result.passed for result in results) for results in results_by_function(outcome).values())


PRINTED_LINES_SHOWN = 200


def result_view(
    exercise: Exercise,
    outcome: Outcome | None,
    check_again: str,
) -> RenderableType:
    """The result as a table with one row per function, then what went wrong in each failing one.

    The table grows with the functions and not with the checks, and it comes
    before the details, so the learner sees where every function stands first."""
    if outcome is None:
        return Text("Checking your code...", style="dim")

    if outcome.problem is not None and not outcome.results:
        return Group(
            Text("✗ Your file can't be checked yet", style="bold red"),
            Text(""),
            indented(text=outcome.problem, left=2),
        )

    parts: list[RenderableType] = []

    if outcome.passed:
        parts.append(Text("✓ Exercise done. Every function passes.", style="bold green"))
        parts.append(Text.assemble("A solution to compare with is in ", (relative(exercise.solution_path), ACCENT), "."))
        parts.append(Text("Press n for the next exercise.", style="bold"))
        parts.append(Text(""))

    functions = results_by_function(outcome)
    table = Table.grid(padding=(0, 3))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column()
    details: list[RenderableType] = []

    for function, results in functions.items():
        failing = [result for result in results if not result.passed]

        if not failing:
            table.add_row(
                Text("✓", style="bold green"),
                Text(function, style="bold"),
                Text("passes", style="green"),
            )
            continue

        if all(NOT_STARTED in result.detail for result in failing):
            table.add_row(
                Text("✗", style="bold red"),
                Text(function, style="bold"),
                Text("not written yet", style="dim"),
            )
            continue

        status = "fails" if len(results) == 1 else f"{len(failing)} of {len(results)} checks fail"
        table.add_row(
            Text("✗", style="bold red"),
            Text(function, style="bold"),
            Text(status, style="red"),
        )

        # Like a compiler, explain only the first failing check of each function, so the rest stays readable.
        first_failure = failing[0]
        details.append(Text.assemble((function, "bold red"), "  ", first_failure.description))
        details.append(indented(text=first_failure.detail, style="yellow", left=4))
        details.append(Text(""))

    parts.append(table)
    parts.append(Text(""))
    parts.extend(details)

    if outcome.problem is not None:
        parts.append(indented(text=outcome.problem, style="red", left=0))
        parts.append(Text(""))

    if not outcome.passed:
        parts.append(Text(f"{passing_functions(outcome)} of {len(functions)} functions pass. {check_again}", style="dim"))

    if not outcome.printed.strip():
        return Group(*parts)

    printed_lines = outcome.printed.rstrip().splitlines()
    shown_lines = printed_lines[-PRINTED_LINES_SHOWN:]

    if len(shown_lines) == len(printed_lines):
        printed_heading = "What your code printed"
    else:
        printed_heading = f"The last {len(shown_lines)} of the {len(printed_lines)} lines your code printed"

    # A box of its own, so the learner's output never reads as part of the results.
    parts.append(Text(""))
    parts.append(
        Panel(
            Text("\n".join(shown_lines), overflow="fold"),
            title=printed_heading,
            title_align="left",
            border_style="magenta",
        ),
    )

    return Group(*parts)


def status_line(
    exercise: Exercise,
    progress: Progress,
    width: int,
) -> RenderableType:
    done = len(progress.done)
    total = len(EXERCISES)
    cells: list[RenderableType] = [
        Text("Exercise", style="dim"),
        Text(relative(exercise.path), style=ACCENT),
        Text("Progress", style="dim"),
    ]

    # The learner needs the whole file path more than the bar, so on a narrow
    # terminal the bar shrinks, then goes.
    if width >= 80:
        cells.append(
            ProgressBar(
                total=total,
                completed=done,
                width=20 if width >= 100 else 10,
                complete_style="green",
            ),
        )

    cells.append(Text(f"{done} of {total} done"))

    grid = Table.grid(
        expand=True,
        padding=(0, 2),
    )

    for cell in cells:
        grid.add_column(
            ratio=1 if cell is cells[1] else None,
            no_wrap=True,
            overflow="ellipsis",
        )

    grid.add_row(*cells)

    return Padding(
        grid,
        pad=(0, 1, 1, 1),
    )


def hint_view(
    exercise: Exercise,
    outcome: Outcome | None,
) -> RenderableType:
    """The hint for each function that doesn't pass yet, so the learner reads only what they need."""
    passing: set[str] = set()

    if outcome is not None:
        passing = {
            function for function, results in results_by_function(outcome).items() if all(result.passed for result in results)
        }

    parts: list[RenderableType] = []

    for function, hint in exercise.hints.items():
        if function in passing:
            continue

        parts.append(Text(function, style="bold yellow"))
        parts.append(indented(text=hint, left=2))
        parts.append(Text(""))

    if not parts:
        return Text("Every function passes, so there is nothing left to hint at.", style="dim")

    return Group(*parts)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

APP_CSS = """
#status {
    height: 2;
}

#tabs {
    height: 1fr;
}

#tabs ContentSwitcher {
    height: 1fr;
}

#tabs TabPane {
    height: 1fr;
    padding: 0;
}

.pane {
    height: 1fr;
    padding: 1 2;
    scrollbar-size-vertical: 1;
}

#lesson-title {
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}

#exercises {
    height: 1fr;
    border: round $primary;
    border-title-align: left;
    border-subtitle-align: right;
}

#search {
    display: none;
}

#search.-open {
    display: block;
}

ConfirmScreen {
    align: center middle;
}

#confirm {
    width: 70;
    max-width: 90%;
    height: auto;
    padding: 1 2;
    border: round $warning;
    background: $surface;
}

.page {
    height: 1fr;
    padding: 0 2;
    border: round $primary;
    border-title-align: left;
}
"""


class PageScreen(Screen[str]):
    """A page of Markdown in one box, closed by the keys its subclass binds."""

    def __init__(
        self,
        title: str,
        text: str,
    ) -> None:
        super().__init__()
        self.page_title = title
        self.text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="pane page"):
            yield MarkdownWidget(self.text)

        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        page = self.query_one(VerticalScroll)
        page.border_title = self.page_title
        page.focus()

    def action_close(
        self,
        choice: str,
    ) -> None:
        self.dismiss(choice)


class WelcomeScreen(PageScreen):
    BINDINGS = [
        Binding("enter", "close('start')", "start"),
    ]


class FinishScreen(PageScreen):
    BINDINGS = [
        Binding("l", "close('list')", "list"),
        Binding("q", "close('quit')", "quit"),
    ]


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "answer(True)", "yes, reset"),
        Binding("n,escape", "answer(False)", "no, keep my changes"),
    ]

    def __init__(
        self,
        question: str,
    ) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(self.question)

        yield Footer(show_command_palette=False)

    def action_answer(
        self,
        yes: bool,
    ) -> None:
        self.dismiss(yes)


class ListScreen(Screen[str | None]):
    """Every exercise, like rustlings' list. Dismisses with the exercise to continue at, or None."""

    BINDINGS = [
        Binding("c", "continue_here", "continue here"),
        Binding("r", "reset_selected", "reset"),
        Binding("s", "search", "search"),
        Binding("d", "toggle_filter('done')", "done"),
        Binding("p", "toggle_filter('pending')", "pending"),
        Binding("q,escape", "back", "back"),
    ]

    def __init__(
        self,
        progress: Progress,
    ) -> None:
        super().__init__()
        self.progress = progress
        self.only: str | None = None
        self.search_text = ""

    def compose(self) -> ComposeResult:
        yield DataTable(
            id="exercises",
            cursor_type="row",
        )
        yield Input(
            id="search",
            placeholder="part of a name, enter to keep it, escape to clear it",
        )
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.border_title = "Exercises"
        table.add_columns("", "State", "Name", "Title")
        self.fill_rows(select=self.progress.current)
        table.focus()

    def shown_exercises(self) -> list[Exercise]:
        shown: list[Exercise] = []

        for exercise in EXERCISES:
            is_done = exercise.name in self.progress.done

            if self.only == "done" and not is_done:
                continue

            if self.only == "pending" and is_done:
                continue

            if self.search_text and self.search_text not in exercise.name:
                continue

            shown.append(exercise)

        return shown

    def selected_name(self) -> str | None:
        table = self.query_one(DataTable)

        if table.row_count == 0:
            return None

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)

        return row_key.value

    def fill_rows(
        self,
        select: str | None,
    ) -> None:
        table = self.query_one(DataTable)
        table.clear()
        shown = self.shown_exercises()

        for exercise in shown:
            is_done = exercise.name in self.progress.done
            table.add_row(
                Text("▶", style="bold red") if exercise.name == self.progress.current else Text(""),
                Text("DONE", style="green") if is_done else Text("PENDING", style="yellow"),
                exercise.name,
                exercise.title,
                key=exercise.name,
            )

        filters = []

        if self.only is not None:
            filters.append(f"only {self.only}")

        if self.search_text:
            filters.append(f"names with '{self.search_text}'")

        done_text = f"{len(self.progress.done)} of {len(EXERCISES)} done"
        table.border_subtitle = done_text if not filters else f"{', '.join(filters)}   {done_text}"
        names = [exercise.name for exercise in shown]

        if select in names:
            table.move_cursor(row=names.index(select))

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected,
    ) -> None:
        self.dismiss(event.row_key.value)

    def action_continue_here(self) -> None:
        name = self.selected_name()

        if name is None:
            self.notify("Nothing is selected.")
            return

        self.dismiss(name)

    def action_reset_selected(self) -> None:
        name = self.selected_name()

        if name is None:
            self.notify("Nothing is selected.")
            return

        reset_exercise(
            exercise=exercise_named(name),
            progress=self.progress,
        )
        self.notify(f"{name} is back to how it started.")
        self.fill_rows(select=name)

    def action_toggle_filter(
        self,
        which: str,
    ) -> None:
        self.only = None if self.only == which else which
        self.fill_rows(select=self.selected_name())

    def action_search(self) -> None:
        search = self.query_one(Input)
        search.add_class("-open")
        search.value = self.search_text
        search.focus()

    def on_input_changed(
        self,
        event: Input.Changed,
    ) -> None:
        self.search_text = event.value
        self.fill_rows(select=self.selected_name())

    def on_input_submitted(self) -> None:
        self.close_search()

    def close_search(self) -> None:
        self.query_one(Input).remove_class("-open")
        self.query_one(DataTable).focus()

    def action_back(self) -> None:
        search = self.query_one(Input)

        if search.has_focus:
            search.value = ""
            self.close_search()
            return

        self.dismiss(None)


class WatchScreen(Screen[None]):
    """The lesson, the hint and the result of the current exercise, rechecked on every save."""

    BINDINGS = [
        Binding("n", "next_exercise", "next"),
        Binding("r", "check_now", "run"),
        Binding("1", "show_tab('lesson')", "lesson"),
        Binding("2,h", "show_tab('hint')", "hint"),
        Binding("3", "show_tab('result')", "result"),
        Binding("tab", "cycle_tab", "next tab", show=False),
        Binding("l", "open_list", "list"),
        Binding("x", "reset_exercise", "reset"),
        Binding("q", "app.quit", "quit"),
    ]

    TABS = ("lesson", "hint", "result")

    def __init__(
        self,
        progress: Progress,
        manual: bool,
    ) -> None:
        super().__init__()
        self.progress = progress
        self.manual = manual
        self.outcome: Outcome | None = None
        self.latest_check = 0
        self.show_result_when_checked = False
        self.watched_time = 0.0
        self.check_again = "Press r to check again." if manual else "Save the file to check again."

    @property
    def exercise(self) -> Exercise:
        return exercise_named(self.progress.current)

    @property
    def passed(self) -> bool:
        return self.outcome is not None and self.outcome.passed

    def compose(self) -> ComposeResult:
        yield Static(id="status")

        with TabbedContent(id="tabs"):
            with TabPane("Lesson", id="lesson-tab"):
                with VerticalScroll(id="lesson", classes="pane"):
                    yield Static(id="lesson-title")
                    yield MarkdownWidget(id="lesson-text")

            with TabPane("Hint", id="hint-tab"):
                with VerticalScroll(id="hint", classes="pane"):
                    yield Static(id="hint-text")

            with TabPane("Result", id="result-tab"):
                with VerticalScroll(id="result", classes="pane"):
                    yield Static(id="result-text")

        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        # A focused tab bar paints the active tab as a solid block, which flashes on every click.
        # Keys and clicks switch tabs without focus, and the arrow keys stay with the tab's content.
        self.query_one(Tabs).can_focus = False
        self.open_exercise(self.exercise)
        self.set_interval(0.5, self.poll_file)

    def on_tabbed_content_tab_activated(
        self,
        event: TabbedContent.TabActivated,
    ) -> None:
        # Focus the scrolling part of the tab, so the arrow keys scroll it straight away.
        event.pane.query_one(VerticalScroll).focus()

    def action_show_tab(
        self,
        name: str,
    ) -> None:
        self.query_one(TabbedContent).active = f"{name}-tab"

    def action_cycle_tab(self) -> None:
        current = self.query_one(TabbedContent).active.removesuffix("-tab")
        self.action_show_tab(self.TABS[(self.TABS.index(current) + 1) % len(self.TABS)])

    def on_resize(self) -> None:
        self.show_status()

    def open_exercise(
        self,
        exercise: Exercise,
    ) -> None:
        self.progress.current = exercise.name
        save_progress(self.progress)
        self.show_result_when_checked = False
        self.watched_time = modified_time(exercise.path)

        self.query_one("#lesson-title", Static).update(exercise.title)
        self.query_one("#lesson-text", MarkdownWidget).update(exercise.lesson)
        self.query_one("#lesson", VerticalScroll).scroll_home(animate=False)
        self.action_show_tab("lesson")

        self.show_hint()
        self.show_status()
        self.check()

    def check(self) -> None:
        self.outcome = None
        self.latest_check += 1
        self.show_result()
        self.refresh_bindings()
        self.check_in_background(
            exercise=self.exercise,
            number=self.latest_check,
        )

    @work(thread=True, group="check")
    def check_in_background(
        self,
        exercise: Exercise,
        number: int,
    ) -> None:
        outcome = attempt(
            exercise=exercise,
            path=exercise.path,
        )
        self.app.call_from_thread(
            self.finish_check,
            exercise,
            number,
            outcome,
        )

    def finish_check(
        self,
        exercise: Exercise,
        number: int,
        outcome: Outcome,
    ) -> None:
        # A save during a check starts another one. Only the latest gets shown.
        if number != self.latest_check:
            return

        record_outcome(
            exercise=exercise,
            outcome=outcome,
            progress=self.progress,
        )
        self.outcome = outcome
        self.show_result()
        self.show_status()
        self.show_hint()
        self.refresh_bindings()

        # After a save the learner wants the outcome, as rustlings shows it. Opening an exercise shows the lesson.
        if self.show_result_when_checked:
            self.show_result_when_checked = False
            self.action_show_tab("result")

    def poll_file(self) -> None:
        """Textual has no file watcher, so compare the file's time twice a second."""
        if self.manual:
            return

        changed_time = modified_time(self.exercise.path)

        if changed_time == self.watched_time:
            return

        self.watched_time = changed_time
        self.show_result_when_checked = True
        self.check()

    def show_status(self) -> None:
        self.query_one("#status", Static).update(
            status_line(
                exercise=self.exercise,
                progress=self.progress,
                width=self.size.width,
            ),
        )

    def result_label(self) -> str:
        if self.outcome is None:
            return "Result  [dim]checking[/]"

        if self.outcome.passed:
            return "Result  [green]✓ done[/]"

        if not self.outcome.results:
            return "Result  [red]✗[/]"

        total = len(results_by_function(self.outcome))

        return f"Result  [red]✗[/] {passing_functions(self.outcome)} of {total}"

    def show_result(self) -> None:
        self.query_one(TabbedContent).get_tab("result-tab").label = self.result_label()
        self.query_one("#result-text", Static).update(
            result_view(
                exercise=self.exercise,
                outcome=self.outcome,
                check_again=self.check_again,
            ),
        )

    def show_hint(self) -> None:
        self.query_one("#hint-text", Static).update(
            hint_view(
                exercise=self.exercise,
                outcome=self.outcome,
            ),
        )

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],
    ) -> bool | None:
        if action == "next_exercise":
            return self.passed

        if action == "check_now":
            return self.manual

        return True

    def action_next_exercise(self) -> None:
        if not self.passed:
            return

        upcoming = next_pending(self.progress)

        if upcoming is not None:
            self.open_exercise(upcoming)
            return

        self.outcome = None
        self.show_result()
        self.notify("Checking every exercise once more before the finish line.")
        self.check_everything()

    @work(thread=True, group="check")
    def check_everything(self) -> None:
        failing = check_all(self.progress)
        self.app.call_from_thread(self.finish_course, failing)

    def finish_course(
        self,
        failing: list[Exercise],
    ) -> None:
        if failing:
            names = ", ".join(exercise.name for exercise in failing)
            self.open_exercise(failing[0])
            self.notify(
                f"These no longer pass, so you're back at the first one. {names}",
                severity="warning",
                timeout=10,
            )
            return

        self.app.push_screen(
            FinishScreen(
                title="You made it to the finish line",
                text=FINISH,
            ),
            callback=self.after_finish,
        )

    def after_finish(
        self,
        choice: str | None,
    ) -> None:
        if choice == "list":
            self.action_open_list()
            return

        self.app.exit()

    def action_check_now(self) -> None:
        self.show_result_when_checked = True
        self.check()

    def action_open_list(self) -> None:
        self.app.push_screen(
            ListScreen(progress=self.progress),
            callback=self.picked_from_list,
        )

    def picked_from_list(
        self,
        name: str | None,
    ) -> None:
        if name is None:
            self.show_status()
            return

        self.open_exercise(exercise_named(name))

    def action_reset_exercise(self) -> None:
        self.app.push_screen(
            ConfirmScreen(question=f"Put {self.exercise.path.name} back to how it started? Your changes will be lost."),
            callback=self.reset_answered,
        )

    def reset_answered(
        self,
        yes: bool | None,
    ) -> None:
        if not yes:
            return

        reset_exercise(
            exercise=self.exercise,
            progress=self.progress,
        )
        self.watched_time = modified_time(self.exercise.path)
        self.check()
        self.notify(f"{self.exercise.path.name} is back to how it started.")


class TensorlingsApp(App[None]):
    CSS = APP_CSS
    TITLE = "tensorlings"
    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        progress: Progress,
        manual: bool,
    ) -> None:
        super().__init__()
        self.progress = progress
        self.manual = manual

    def on_mount(self) -> None:
        self.push_screen(
            WatchScreen(
                progress=self.progress,
                manual=self.manual,
            ),
        )

        if self.progress.welcomed:
            return

        self.push_screen(
            WelcomeScreen(
                title="Welcome to tensorlings",
                text=WELCOME,
            ),
            callback=self.welcomed,
        )

    def welcomed(
        self,
        choice: str | None,
    ) -> None:
        self.progress.welcomed = True
        save_progress(self.progress)

    def copy_to_clipboard(
        self,
        text: str,
    ) -> None:
        super().copy_to_clipboard(text)

        # macOS Terminal ignores the escape code Textual copies with, so hand the text to pbcopy too.
        pbcopy = shutil.which("pbcopy")

        if pbcopy is None:
            return

        subprocess.run(
            [pbcopy],
            input=text.encode(),
            check=False,
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

USAGE = prose(
    f"""
    tensorlings, small exercises that teach deep learning from tensors up.

    {START_COMMAND} init               set up the exercises folder, once
    {START_COMMAND}                    start the course
    {START_COMMAND} --manual-run       start the course, checking only when you press r
    {START_COMMAND} run [name]         check one exercise and print the result
    {START_COMMAND} hint [name]        print the hints for an exercise
    {START_COMMAND} reset name         reset an exercise file to how it started
    {START_COMMAND} check-all          check every exercise and update your progress
    {START_COMMAND} dev check          maintainer check of every exercise and lesson
    """,
)


def require_setup() -> bool:
    if EXERCISES_DIR.exists():
        return True

    print("The exercises are not set up yet. Run this once first.\n")
    print(f"    {START_COMMAND} init\n")

    return False


def initialize() -> int:
    if EXERCISES_DIR.exists():
        print(f"The exercises folder already exists. Start the course with `{START_COMMAND}`.")
        return 1

    EXERCISES_DIR.mkdir()
    (EXERCISES_DIR / "README.md").write_text(COURSE_README)

    for chapter in CHAPTERS:
        folder = EXERCISES_DIR / chapter.folder
        folder.mkdir()
        (folder / "README.md").write_text(chapter.readme)

    for exercise in EXERCISES:
        exercise.path.write_text(exercise.source)

    print("The exercises are ready in the exercises folder. Start the course with\n")
    print(f"    {START_COMMAND}\n")

    return 0


def run_once(
    exercise: Exercise,
    progress: Progress,
) -> int:
    console = terminal_console()
    outcome = attempt(
        exercise=exercise,
        path=exercise.path,
    )
    record_outcome(
        exercise=exercise,
        outcome=outcome,
        progress=progress,
    )
    console.print(
        Panel(
            result_view(
                exercise=exercise,
                outcome=outcome,
                check_again="Run this command again after you change the file.",
            ),
            title=exercise.title,
            title_align="left",
            border_style="green" if outcome.passed else "red",
        ),
    )

    return 0 if outcome.passed else 1


def dev_check(
    names: list[str],
    verbose: bool,
) -> int:
    console = Console()
    problems = 0
    introduced: set[str] = set()

    with tempfile.TemporaryDirectory() as folder:
        for exercise in EXERCISES:
            findings = lint_exercise(
                exercise=exercise,
                introduced_before=introduced,
                console=console,
            )
            introduced |= set(exercise.introduces)

            if names and exercise.name not in names:
                continue

            outcomes: dict[str, Outcome] = {}

            for version, text in (("stub", exercise.source), ("solution", exercise.solution)):
                path = Path(folder) / version / exercise.chapter.folder / f"{exercise.name}.py"
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                path.write_text(text)
                outcomes[version] = attempt(
                    exercise=exercise,
                    path=path,
                )

            stub = outcomes["stub"]
            solution = outcomes["solution"]
            healthy = not stub.passed and solution.passed and not findings
            problems += 0 if healthy else 1
            mark = "[green]✓[/green]" if healthy else "[red]✗[/red]"
            stub_text = "fails as it should" if not stub.passed else "[red]PASSES[/red]"
            solution_text = "passes" if solution.passed else "[red]FAILS[/red]"
            console.print(f"{mark} {exercise.name:<12} stub {stub_text:<20} solution {solution_text:<8} lesson {len(findings)} findings")

            for finding in findings:
                console.print(f"      [yellow]lesson[/yellow] {finding}")

            if not solution.passed or verbose:
                console.print(
                    textwrap.indent(
                        "\n".join(
                            f"{'pass' if result.passed else 'FAIL'}  {result.function}, {result.description}  {result.detail}"
                            for result in solution.results
                        )
                        + (f"\n{solution.problem}" if solution.problem else ""),
                        "      ",
                    ),
                )

            if verbose:
                console.print(
                    textwrap.indent(
                        "\n".join(
                            f"stub {'pass' if result.passed else 'FAIL'}  {result.function}, {result.description}\n"
                            f"{textwrap.indent(result.detail, '      ')}"
                            for result in stub.results
                        )
                        + (f"\nstub problem: {stub.problem}" if stub.problem else ""),
                        "      ",
                    ),
                )

    console.print()
    console.print("[green]Every exercise is healthy.[/green]" if problems == 0 else f"[red]{problems} exercise(s) need attention.[/red]")

    return 1 if problems else 0


def main(arguments: list[str]) -> int:
    flags = [argument for argument in arguments if argument.startswith("-")]
    words = [argument for argument in arguments if not argument.startswith("-")]
    command = words[0] if words else "watch"
    rest = words[1:]

    if "-h" in flags or "--help" in flags or command == "help":
        print(USAGE)
        return 0

    if command == "init":
        return initialize()

    if command == "dev":
        if rest[:1] != ["check"]:
            print(USAGE)
            return 1

        return dev_check(
            names=rest[1:],
            verbose="--verbose" in flags,
        )

    if not require_setup():
        return 1

    progress = load_progress()

    if command == "watch":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(f"The app needs a terminal. Use `{START_COMMAND} run` to check one exercise without it.")
            return 1

        TensorlingsApp(
            progress=progress,
            manual="--manual-run" in flags,
        ).run()
        print(f"See you next time. Run {START_COMMAND} to continue where you left off.")

        return 0

    if command == "check-all":
        failing = check_all(progress)

        for exercise in EXERCISES:
            state = "PENDING" if exercise in failing else "DONE   "
            print(f"{state}  {exercise.name}")

        return 1 if failing else 0

    if command in ("run", "hint", "reset"):
        if command == "reset" and not rest:
            print(f"Name the exercise to reset, for example `{START_COMMAND} reset tensors1`.")
            return 1

        exercise = exercise_named(rest[0]) if rest else exercise_named(progress.current)

        if exercise is None:
            print(f"There is no exercise called {rest[0]}.")
            return 1

        if command == "hint":
            for function, hint in exercise.hints.items():
                print(f"{function}\n{textwrap.indent(textwrap.fill(hint, width=76), '  ')}\n")

            return 0

        if command == "reset":
            reset_exercise(
                exercise=exercise,
                progress=progress,
            )
            print(f"{exercise.path.name} is back to how it started.")
            return 0

        return run_once(
            exercise=exercise,
            progress=progress,
        )

    print(USAGE)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
