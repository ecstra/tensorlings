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
    from textual.widgets import DataTable, Footer, Input, Static, TabbedContent, TabPane
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
        spec.loader.exec_module(module)
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
            run(found)
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
            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=describe_crash(
                        error=error,
                        path=self.path,
                    ),
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


NOT_STARTED = "returned None"


def reject_missing_return(
    call: str,
    actual: object,
) -> None:
    if actual is None:
        raise Mismatch(f"{call} {NOT_STARTED}, which usually means the function has no return statement yet.")


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

    if tuple(actual.shape) != tuple(expected.shape):
        raise Mismatch(
            f"{call} has shape {shape_text(tuple(actual.shape))}, but it should have shape {shape_text(tuple(expected.shape))}.\n"
            f"got       {value_text(actual)}\n"
            f"expected  {value_text(expected)}",
        )

    if not torch.allclose(actual.double(), expected.double(), atol=1e-6):
        raise Mismatch(
            f"{call} has the right shape but different numbers.\n"
            f"got       {value_text(actual)}\n"
            f"expected  {value_text(expected)}",
        )


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
    # Where to look, the idea, then the answer. Each opens after one more run.
    hints: tuple[str, str, str]
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

        Every number a model reads, stores or produces lives in a tensor, a grid
        of numbers with a fixed shape. This chapter covers making tensors,
        reading their shape, picking out parts, doing math on whole tensors at
        once, and adding up along a dimension.

        ## Further reading

        - [PyTorch, introduction to tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
        - [Dive into Deep Learning, data manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html)
        """,
    ),
)

TENSORS1_LESSON = prose(
    """
    Say you write down the temperature three times a day, in the morning, at
    noon and at night. After two days you have six numbers. Here they are in
    PyTorch.

    ```python
    import torch

    temps = torch.tensor([[14, 22, 17],
                          [12, 25, 19]])
    ```

    You made a **tensor**. Picture a grid. Each inner list became a row, one per
    day, and each row holds that day's three readings. Every number in the grid
    is called an entry.

    Ask PyTorch how big it is.

    ```python
    temps.shape    # torch.Size([2, 3])
    ```

    That's the **shape**, 2 days by 3 readings. Each size in a shape is one
    **dimension**, so `temps` has two. Dimension 0 counts the days, down the
    rows. Dimension 1 counts the readings, across the columns. From here on
    this course writes a shape the short way, like `(2, 3)`.

    A shape works like a Python tuple. Anything you can do with a tuple, you can
    do with a shape.

    Tensors can have any number of dimensions, even none.

    | What you have | Shape |
    |---|---|
    | one reading, `torch.tensor(14)` | `()` |
    | one day, `torch.tensor([14, 22, 17])` | `(3,)` |
    | two days, `temps` | `(2, 3)` |
    | a week of days for 10 cities | `(10, 7, 3)` |

    One reading has no dimensions, so its shape is empty. To count the entries
    in any tensor, multiply the sizes in its shape. `temps` holds 2 × 3 = 6.

    When you save, the app runs your functions on tensors like
    `torch.zeros(4, 5)`, a 4 by 5 grid of zeros. Zeros are fine there, since
    only the shape matters.

    So why start with grids of numbers? A **model**, like a photo tagger or a
    chatbot, is a program that turns numbers into other numbers. It gets good
    through **training**, which keeps nudging its own numbers until its answers
    come out right. Every one of those numbers lives in a tensor, and a lot of
    bugs in model code are shapes that don't fit together.
    """,
)

TENSORS1_SOURCE = source_file(
    '''
    # tensors1.py

    import torch
    from torch import Tensor


    # TODO: A shop sold 4 apples and 2 pears on Monday, then 1 apple and 5 pears
    #   on Tuesday. Return the sales as a tensor with one row per day, apples first.
    def make_sales() -> Tensor:
        ...


    # TODO: Return how many rows the grid has, the size of its dimension 0.
    def count_rows(grid: Tensor) -> int:
        ...


    # TODO: Return how many dimensions the tensor has.
    def count_dimensions(values: Tensor) -> int:
        ...


    # TODO: Return how many entries the tensor holds in total.
    #   It has to work for any tensor, even one with no dimensions, like torch.tensor(5.0).
    def count_entries(values: Tensor) -> int:
        ...


    # What the check tests
    #   make_sales() holds 4 2 above 1 5
    #   count_rows is 4 for torch.zeros(4, 5) and 10 for torch.zeros(10, 3)
    #   count_dimensions is 2, 1, 0 and 3 for tensors of shape (4, 5), (7,), () and (2, 3, 4)
    #   count_entries is 20, 7, 1 and 24 for the same four tensors
    ''',
)

TENSORS1_SOLUTION = source_file(
    '''
    # tensors1.py, a solution for comparison

    import math

    import torch
    from torch import Tensor


    def make_sales() -> Tensor:
        return torch.tensor([[4, 2], [1, 5]])


    def count_rows(grid: Tensor) -> int:
        return grid.shape[0]


    # values.dim() gives the same number.
    def count_dimensions(values: Tensor) -> int:
        return len(values.shape)


    # values.numel() gives the same number. An empty shape multiplies out to 1.
    def count_entries(values: Tensor) -> int:
        return math.prod(values.shape)
    ''',
)

SAMPLE_TENSORS: tuple[tuple[str, Tensor], ...] = (
    ("torch.zeros(4, 5)", torch.zeros(4, 5)),
    ("torch.zeros(7)", torch.zeros(7)),
    ("torch.tensor(5.0)", torch.tensor(5.0)),
    ("torch.zeros(2, 3, 4)", torch.zeros(2, 3, 4)),
)


def check_tensors1(checker: Checker) -> None:
    checker.test(
        function="make_sales",
        description="holds 4 2 for Monday above 1 5 for Tuesday",
        run=lambda make_sales: expect_tensor(
            call="make_sales()",
            actual=make_sales(),
            expected=torch.tensor([[4, 2], [1, 5]]),
        ),
    )
    checker.test(
        function="count_rows",
        description="is 4 for torch.zeros(4, 5) and 10 for torch.zeros(10, 3)",
        run=lambda count_rows: (
            expect_value(
                call="count_rows(torch.zeros(4, 5))",
                actual=count_rows(torch.zeros(4, 5)),
                expected=4,
            ),
            expect_value(
                call="count_rows(torch.zeros(10, 3))",
                actual=count_rows(torch.zeros(10, 3)),
                expected=10,
            ),
        ),
    )
    checker.test(
        function="count_dimensions",
        description="is 2, 1, 0 and 3 for shapes (4, 5), (7,), () and (2, 3, 4)",
        run=lambda count_dimensions: [
            expect_value(
                call=f"count_dimensions({call})",
                actual=count_dimensions(values),
                expected=values.dim(),
            )
            for call, values in SAMPLE_TENSORS
        ],
    )
    checker.test(
        function="count_entries",
        description="is 20, 7, 1 and 24 for the same four tensors",
        run=lambda count_entries: [
            expect_value(
                call=f"count_entries({call})",
                actual=count_entries(values),
                expected=values.numel(),
            )
            for call, values in SAMPLE_TENSORS
        ],
    )


TENSORS1 = Exercise(
    name="tensors1",
    chapter=TENSORS,
    title="What a tensor is",
    lesson=TENSORS1_LESSON,
    source=TENSORS1_SOURCE,
    solution=TENSORS1_SOLUTION,
    hints=(
        "make_sales is about which numbers go in which row. The other three are about the shape. Look again at "
        "what temps.shape gave you and at the table of shapes.",
        "make_sales needs one inner list per day. A shape is a tuple with one size per dimension. Its first item "
        "is the row count and its length is the number of dimensions. For the entries, multiply every size in "
        "the shape. math.prod multiplies a whole tuple, and an empty tuple gives 1.",
        "torch.tensor([[4, 2], [1, 5]]), grid.shape[0], len(values.shape), and math.prod(values.shape) with "
        "import math at the top. values.numel() also counts the entries.",
    ),
    introduces=("tensor", "shape", "dimension", "model", "training"),
    check=check_tensors1,
)

TENSORS2_LESSON = prose(
    """
    When a model writes text, it picks the next word from its output at the
    last position. During training it learns from a few rows of data at a time,
    usually not the whole dataset. Both mean picking parts out of a tensor.
    Here are four days of temperatures to practice on.

    ```python
    temps = torch.tensor([[14, 22, 17],
                          [12, 25, 19],
                          [15, 21, 16],
                          [11, 24, 18]])   # shape (4, 3)
    ```

    Square brackets pick parts out. Give a row, and if you want, a comma and a
    column. Leave the column out and you get the whole row. Positions count
    from 0, and a negative position counts back from the end.

    | Code | You get | Which part |
    |---|---|---|
    | `temps[0]` | `tensor([14, 22, 17])` | day 0 |
    | `temps[-1]` | `tensor([11, 24, 18])` | the last day |
    | `temps[1, 2]` | `tensor(19)` | day 1, the night reading |
    | `temps[:, 0]` | `tensor([14, 12, 15, 11])` | every morning |

    Each position you pick is an **index**. A bare `:` in place of an index
    means all of that dimension, so `temps[:, 0]` reads as every day, reading 0.

    To take a range, write `start:stop`. You get `start` and everything after
    it, up to but not including `stop`. Leave out either end and the range
    runs to the edge. Negative positions work here too. A range like this is a
    **slice**.

    | Code | You get | Shape |
    |---|---|---|
    | `temps[1:3]` | days 1 and 2 | `(2, 3)` |
    | `temps[:2]` | the first two days | `(2, 3)` |
    | `temps[:-1]` | every day but the last | `(3, 3)` |

    Keep an eye on the shapes. `temps[0]` has shape `(3,)`, because an index
    makes its dimension disappear. `temps[0:1]` holds the same day with shape
    `(1, 3)`, because a slice keeps its dimension even when it holds one item.
    That difference comes back in the next two exercises.
    """,
)

TENSORS2_SOURCE = source_file(
    '''
    # tensors2.py

    import torch
    from torch import Tensor

    # Every function here gets a group of sentences. Each row is one sentence
    # and each column is one word position, with every word stored as a number.
    #
    #     sentences = torch.tensor([[11, 12, 13, 14, 15],
    #                               [21, 22, 23, 24, 25],
    #                               [31, 32, 33, 34, 35]])


    # TODO: Return the last two words of every sentence.
    def last_two_words(sentences: Tensor) -> Tensor:
        ...


    # TODO: Return every sentence without its first word.
    def drop_first_word(sentences: Tensor) -> Tensor:
        ...


    # TODO: Drop the first and the last sentence, then drop the first and the
    #   last word of every sentence that is left.
    def trim_edges(sentences: Tensor) -> Tensor:
        ...


    # What the check tests, on the sentences above and on 4 sentences of 6 words
    #   last_two_words(sentences) is [[14, 15], [24, 25], [34, 35]]
    #   drop_first_word(sentences) has shape (3, 4) and starts with [12, 13, 14, 15]
    #   trim_edges(sentences) is [[22, 23, 24]]
    ''',
)

TENSORS2_SOLUTION = source_file(
    '''
    # tensors2.py, a solution for comparison

    import torch
    from torch import Tensor


    def last_two_words(sentences: Tensor) -> Tensor:
        return sentences[:, -2:]


    def drop_first_word(sentences: Tensor) -> Tensor:
        return sentences[:, 1:]


    def trim_edges(sentences: Tensor) -> Tensor:
        return sentences[1:-1, 1:-1]
    ''',
)


def check_tensors2(checker: Checker) -> None:
    sentences = torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]])
    more_sentences = torch.arange(1, 25).reshape(4, 6)

    for function, solve in (
        ("last_two_words", lambda values: values[:, -2:]),
        ("drop_first_word", lambda values: values[:, 1:]),
        ("trim_edges", lambda values: values[1:-1, 1:-1]),
    ):
        checker.test(
            function=function,
            description="is right for the sentences in the file",
            run=lambda learner, solve=solve, function=function: expect_tensor(
                call=f"{function}(sentences)",
                actual=learner(sentences),
                expected=solve(sentences),
            ),
        )
        checker.test(
            function=function,
            description="is right for 4 sentences of 6 words too",
            run=lambda learner, solve=solve, function=function: expect_tensor(
                call=f"{function} on 4 sentences of 6 words, torch.arange(1, 25).reshape(4, 6)",
                actual=learner(more_sentences),
                expected=solve(more_sentences),
            ),
        )


TENSORS2 = Exercise(
    name="tensors2",
    chapter=TENSORS,
    title="Picking out parts of a tensor",
    lesson=TENSORS2_LESSON,
    source=TENSORS2_SOURCE,
    solution=TENSORS2_SOLUTION,
    hints=(
        "Each function picks along both dimensions. Look at how temps[:, 0] took a column and how temps[:-1] "
        "dropped the last day.",
        "Rows come first inside the brackets, then columns. A bare : keeps every row. A slice works on columns "
        "too. A negative start counts from the end, and a negative stop leaves out the end.",
        "sentences[:, -2:], sentences[:, 1:] and sentences[1:-1, 1:-1].",
    ),
    introduces=("index", "slice"),
    check=check_tensors2,
)

TENSORS3_LESSON = prose(
    """
    Two shops sell the same three items. Here are their prices.

    ```python
    prices = torch.tensor([[3, 5, 2],
                           [4, 6, 1]])   # shape (2, 3), one row per shop
    ```

    You can add tax to every price at once, with no loop.

    ```python
    tax = torch.tensor([[1, 1, 0],
                        [1, 2, 0]])
    prices + tax    # tensor([[4, 6, 2], [5, 8, 1]])
    ```

    Each entry got added to the one in the same spot. This is called
    **elementwise** math, and `-`, `*` and `/` work the same way.

    Now a delivery fee for each item, the same in both shops.

    ```python
    fees = torch.tensor([1, 0, 2])   # shape (3,)
    prices + fees    # tensor([[4, 5, 4], [5, 6, 3]])
    ```

    The shapes don't match, and it worked anyway. PyTorch stretched `fees` over
    both rows without copying it. That's **broadcasting**. To see whether two
    shapes fit, line them up from the right. Two sizes fit when they're equal or
    when one of them is 1, and a 1 stretches to match. A shorter shape counts
    its missing sizes as 1.

    ```text
    prices   (2, 3)
    fees        (3,)    3 meets 3, and the missing size counts as 1
    ```

    A plain number has no dimensions at all, so `prices * 2` fits too.

    Here's where it goes wrong. Say each shop adds its own charge, 10 at the
    first and 20 at the second, written as `torch.tensor([10, 20])`.

    ```text
    prices   (2, 3)
    charge      (2,)    2 meets 3, so PyTorch stops with an error
    ```

    The charge has one number per shop, so it should line up with the rows.
    Give it a size 1 at the end and it does.

    ```python
    per_shop = torch.tensor([[10],
                             [20]])     # shape (2, 1)

    prices + per_shop    # tensor([[13, 15, 12], [24, 26, 21]])
    ```

    `(2, 1)` meets `(2, 3)`. The 2s match, and the 1 stretches across the items.

    Models work on a **batch**, many examples at once with one per row. Picture
    a batch of 32 photos. With broadcasting, one line brightens all 32 by the
    same amount, or each one by its own amount.
    """,
)

TENSORS3_SOURCE = source_file(
    '''
    # tensors3.py

    import torch
    from torch import Tensor

    # scores has one row per student and one column per test, like this.
    #
    #     scores = torch.tensor([[45, 60, 40],
    #                            [50, 100, 80]])


    # TODO: Turn the scores into percentages.
    #   Each test has its own top mark, and top_marks holds one per test.
    #   A score of 45 on a test marked out of 50 is 90 percent.
    def as_percent(
        scores: Tensor,
        top_marks: Tensor,
    ) -> Tensor:
        ...


    # TODO: Subtract each student's first test score from all of that student's scores.
    #   A student who scored 50 60 70 gets 0 10 20.
    def gain_since_first_test(scores: Tensor) -> Tensor:
        ...


    # What the check tests
    #   as_percent of the scores above, with top marks 50, 100 and 80
    #   as_percent of 4 students
    #   gain_since_first_test of 2 students, then of 3 students with 3 tests
    ''',
)

TENSORS3_SOLUTION = source_file(
    '''
    # tensors3.py, a solution for comparison

    import torch
    from torch import Tensor


    def as_percent(
        scores: Tensor,
        top_marks: Tensor,
    ) -> Tensor:
        return scores / top_marks * 100


    # A slice keeps its dimension, so scores[:, 0:1] has shape (n, 1) and stretches across the tests.
    def gain_since_first_test(scores: Tensor) -> Tensor:
        return scores - scores[:, 0:1]
    ''',
)


def check_tensors3(checker: Checker) -> None:
    two_students = torch.tensor([[45, 60, 40], [50, 100, 80]])
    four_students = torch.tensor([[45, 60, 40], [50, 100, 80], [10, 50, 20], [25, 75, 60]])
    top_marks = torch.tensor([50, 100, 80])
    gains = torch.tensor([[50, 60, 70], [80, 90, 40]])
    three_by_three = torch.tensor([[1, 2, 3], [40, 50, 60], [700, 800, 900]])

    checker.test(
        function="as_percent",
        description="gives [[90, 60, 50], [100, 100, 100]] for the scores in the file",
        run=lambda as_percent: expect_tensor(
            call="as_percent(scores, torch.tensor([50, 100, 80]))",
            actual=as_percent(two_students, top_marks),
            expected=two_students / top_marks * 100,
        ),
    )
    checker.test(
        function="as_percent",
        description="works for 4 students",
        run=lambda as_percent: expect_tensor(
            call="as_percent with 4 students",
            actual=as_percent(four_students, top_marks),
            expected=four_students / top_marks * 100,
        ),
    )
    checker.test(
        function="gain_since_first_test",
        description="gives [[0, 10, 20], [0, 10, -40]] for scores [[50, 60, 70], [80, 90, 40]]",
        run=lambda gain_since_first_test: expect_tensor(
            call="gain_since_first_test with 2 students",
            actual=gain_since_first_test(gains),
            expected=gains - gains[:, 0:1],
        ),
    )
    checker.test(
        function="gain_since_first_test",
        description="subtracts per student with 3 students and 3 tests",
        run=lambda gain_since_first_test: expect_tensor(
            call="gain_since_first_test with 3 students and 3 tests",
            actual=gain_since_first_test(three_by_three),
            expected=three_by_three - three_by_three[:, 0:1],
        ),
    )


TENSORS3 = Exercise(
    name="tensors3",
    chapter=TENSORS,
    title="Math on whole tensors",
    lesson=TENSORS3_LESSON,
    source=TENSORS3_SOURCE,
    solution=TENSORS3_SOLUTION,
    hints=(
        "as_percent takes two steps from the lesson. For gain_since_first_test, think about which shape each "
        "student's first score needs so it lines up with that student's row.",
        "Divide by the top marks, then multiply by 100. For the gain, the first scores need shape (n, 1), like "
        "per_shop. From the last exercise, a slice keeps its dimension and an index drops it.",
        "scores / top_marks * 100, and scores - scores[:, 0:1].",
    ),
    introduces=("elementwise", "broadcasting", "batch"),
    check=check_tensors3,
)

TENSORS4_LESSON = prose(
    """
    In the last exercise you got numbers of shape `(n, 1)` by slicing, and they
    lined up with the rows. Slicing only works when the numbers already sit in
    a grid. Often they arrive as a tensor with one dimension, a plain list.
    Training does this a lot, with one number per example saying how much that
    example counts.

    ```python
    per_shop = torch.tensor([10, 20])   # shape (2,), and you need (2, 1)
    ```

    Putting `None` inside the brackets adds a new dimension of size 1, right
    where you put it. A `:` keeps a dimension that's already there.

    ```python
    per_shop[:, None]    # tensor([[10], [20]]), shape (2, 1)
    per_shop[None, :]    # tensor([[10, 20]]),   shape (1, 2)
    ```

    Same two numbers, standing up or lying down. Standing up, they line up with
    rows. Lying down, they line up with columns, and so does the plain list.

    Remember the rule from the last exercise. A size of 1 stretches, whichever
    side it's on, so both tensors in one calculation can stretch at once.

    There's one trap. When a grid has as many rows as columns, a plain list
    fits without any `None`.

    ```text
    square   (3, 3)
    list        (3,)    fits, and lines up with the columns
    ```

    You meant one number per row. PyTorch used one per column, and nothing
    complains. The only sign is that the numbers come out wrong. So when you
    mean rows, add the `None` even if the code already runs without it.
    """,
)

TENSORS4_SOURCE = source_file(
    '''
    # tensors4.py

    import torch
    from torch import Tensor


    # TODO: Return a times table. Row i, column j holds rows[i] times columns[j].
    #   times_table(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])) gives
    #
    #     1  2  3   4
    #     2  4  6   8
    #     3  6  9  12
    #
    def times_table(
        rows: Tensor,
        columns: Tensor,
    ) -> Tensor:
        ...


    # TODO: Add each student's own bonus to all of their scores, and add each
    #   test's bonus to every student's score on that test.
    #   scores has one row per student and one column per test.
    #   student_bonus has one number per student, test_bonus one per test.
    def add_both_bonuses(
        scores: Tensor,
        student_bonus: Tensor,
        test_bonus: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   times_table for the 3 by 4 table above, and for 2 rows by 5 columns
    #   add_both_bonuses for 2 students and 3 tests
    #   add_both_bonuses for 3 students and 3 tests, where rows and columns are easy to mix up
    ''',
)

TENSORS4_SOLUTION = source_file(
    '''
    # tensors4.py, a solution for comparison

    import torch
    from torch import Tensor


    # (n, 1) times (m,) stretches both ways into (n, m).
    def times_table(
        rows: Tensor,
        columns: Tensor,
    ) -> Tensor:
        return rows[:, None] * columns


    def add_both_bonuses(
        scores: Tensor,
        student_bonus: Tensor,
        test_bonus: Tensor,
    ) -> Tensor:
        return scores + student_bonus[:, None] + test_bonus
    ''',
)


def check_tensors4(checker: Checker) -> None:
    scores = torch.tensor([[50, 60, 70], [80, 90, 40]])
    square_scores = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    checker.test(
        function="times_table",
        description="gives the 3 by 4 table in the file",
        run=lambda times_table: expect_tensor(
            call="times_table(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]))",
            actual=times_table(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])),
            expected=torch.tensor([1, 2, 3])[:, None] * torch.tensor([1, 2, 3, 4]),
        ),
    )
    checker.test(
        function="times_table",
        description="gives a 2 by 5 table for 2 rows and 5 columns",
        run=lambda times_table: expect_tensor(
            call="times_table(torch.tensor([10, 20]), torch.tensor([1, 2, 3, 4, 5]))",
            actual=times_table(torch.tensor([10, 20]), torch.tensor([1, 2, 3, 4, 5])),
            expected=torch.tensor([10, 20])[:, None] * torch.tensor([1, 2, 3, 4, 5]),
        ),
    )
    checker.test(
        function="add_both_bonuses",
        description="adds both bonuses for 2 students and 3 tests",
        run=lambda add_both_bonuses: expect_tensor(
            call="add_both_bonuses with student bonus [1, 2] and test bonus [10, 20, 30]",
            actual=add_both_bonuses(scores, torch.tensor([1, 2]), torch.tensor([10, 20, 30])),
            expected=scores + torch.tensor([1, 2])[:, None] + torch.tensor([10, 20, 30]),
        ),
    )
    checker.test(
        function="add_both_bonuses",
        description="keeps rows and columns apart for 3 students and 3 tests",
        run=lambda add_both_bonuses: expect_tensor(
            call="add_both_bonuses with student bonus [100, 200, 300] and test bonus [10, 20, 30]",
            actual=add_both_bonuses(square_scores, torch.tensor([100, 200, 300]), torch.tensor([10, 20, 30])),
            expected=square_scores + torch.tensor([100, 200, 300])[:, None] + torch.tensor([10, 20, 30]),
        ),
    )


TENSORS4 = Exercise(
    name="tensors4",
    chapter=TENSORS,
    title="Giving each row its own number",
    lesson=TENSORS4_LESSON,
    source=TENSORS4_SOURCE,
    solution=TENSORS4_SOLUTION,
    hints=(
        "Both functions mix something that goes with rows and something that goes with columns. Look at what None "
        "did to per_shop.",
        "A tensor standing up, shape (n, 1), lines up with rows. One lying flat, shape (m,), lines up with columns. "
        "Combine the two and each size 1 stretches, so you get (n, m).",
        "rows[:, None] * columns, and scores + student_bonus[:, None] + test_bonus.",
    ),
    introduces=(),
    check=check_tensors4,
)

TENSORS5_LESSON = prose(
    """
    At every step of training, a model checks how wrong it is on each example in
    a batch. Then it averages all of those errors into one number to improve on.
    Turning many numbers into fewer is what this exercise is about. Back to four
    days of temperatures.

    ```python
    temps = torch.tensor([[14, 22, 17],
                          [12, 25, 19],
                          [15, 21, 16],
                          [11, 24, 18]])   # 4 days, 3 readings a day
    ```

    `temps.sum()` adds up all twelve numbers into `tensor(214)`. Often that
    squashes too much. Say you want one total per day. Tell `sum` which
    dimension to add along.

    ```python
    temps.sum(dim=1)    # tensor([53, 56, 52, 53]), shape (4,)
    ```

    Each day's three readings got added together. The shape went from `(4, 3)`
    to `(4,)`, and that's the whole rule. The dimension you name is the one
    that disappears.

    Averages work the same way with `.mean()`, with one catch. A tensor holds
    either whole numbers or decimal numbers, never both. `temps` holds whole
    numbers, and `mean` only works on decimals. Unlike `/`, it won't convert
    them for you, so `temps.mean()` stops with an error. Convert first.

    ```python
    temps.float().mean()    # tensor(17.8333)
    ```

    One more option. `keepdim=True` keeps the summed dimension as size 1 instead
    of dropping it. Add up each day's readings that way and the four totals come
    out standing up, shape `(4, 1)`. They're ready to line up with the rows of
    `temps`, and the last two exercises showed why that matters.

    Adding up or averaging along a dimension is called a **reduction**.
    """,
)

TENSORS5_SOURCE = source_file(
    '''
    # tensors5.py

    import torch
    from torch import Tensor

    # scores has one row per student and one column per test.


    # TODO: Return one total per test, adding up every student's score on it.
    def total_per_test(scores: Tensor) -> Tensor:
        ...


    # TODO: Return each student's average score, as decimal numbers.
    def average_per_student(scores: Tensor) -> Tensor:
        ...


    # TODO: Divide every score by its student's total, so each row adds up to 1.
    def share_of_row(scores: Tensor) -> Tensor:
        ...


    # What the check tests, on 2 students with 3 tests and on 3 students with 4 tests
    #   total_per_test of [[50, 60, 70], [80, 90, 40]] is [130, 150, 110]
    #   average_per_student of the same scores is [60.0, 70.0]
    #   share_of_row gives rows that each add up to 1
    ''',
)

TENSORS5_SOLUTION = source_file(
    '''
    # tensors5.py, a solution for comparison

    import torch
    from torch import Tensor


    def total_per_test(scores: Tensor) -> Tensor:
        return scores.sum(dim=0)


    def average_per_student(scores: Tensor) -> Tensor:
        return scores.float().mean(dim=1)


    # keepdim=True leaves the totals standing up, shape (n, 1), so they line up with the rows.
    def share_of_row(scores: Tensor) -> Tensor:
        return scores / scores.sum(dim=1, keepdim=True)
    ''',
)


def check_tensors5(checker: Checker) -> None:
    scores = torch.tensor([[50, 60, 70], [80, 90, 40]])
    bigger = torch.tensor([[1, 2, 3, 4], [10, 0, 10, 0], [5, 5, 5, 5]])

    for function, solve in (
        ("total_per_test", lambda values: values.sum(dim=0)),
        ("average_per_student", lambda values: values.float().mean(dim=1)),
        ("share_of_row", lambda values: values / values.sum(dim=1, keepdim=True)),
    ):
        checker.test(
            function=function,
            description="is right for 2 students with 3 tests",
            run=lambda learner, solve=solve, function=function: expect_tensor(
                call=f"{function}(torch.tensor([[50, 60, 70], [80, 90, 40]]))",
                actual=learner(scores),
                expected=solve(scores),
            ),
        )
        checker.test(
            function=function,
            description="is right for 3 students with 4 tests",
            run=lambda learner, solve=solve, function=function: expect_tensor(
                call=f"{function} with 3 students and 4 tests",
                actual=learner(bigger),
                expected=solve(bigger),
            ),
        )


TENSORS5 = Exercise(
    name="tensors5",
    chapter=TENSORS,
    title="Adding up along a dimension",
    lesson=TENSORS5_LESSON,
    source=TENSORS5_SOURCE,
    solution=TENSORS5_SOLUTION,
    hints=(
        "Each function adds up or averages along one dimension. For each one, ask which dimension should disappear.",
        "Per test means the students go away, so that's dimension 0. Per student means dimension 1. Averages need "
        ".float() first. To divide by row totals, the totals have to stand up, and keepdim=True does that.",
        "scores.sum(dim=0), scores.float().mean(dim=1), and scores / scores.sum(dim=1, keepdim=True).",
    ),
    introduces=("reduction",),
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
        it is the operation most of deep learning is built on.

        ## Further reading

        - [Dive into Deep Learning, linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
        - [3Blue1Brown, the essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
        """,
    ),
)

MATMUL1_LESSON = prose(
    """
    You want to guess the monthly rent of a flat from its size and its number
    of rooms. Your rule of thumb is 10 per square metre plus 100 per room.

    ```text
    size 50, rooms 2

    rent = 10 × 50 + 100 × 2
         = 500 + 200
         = 700
    ```

    The 10 and the 100 say how much each input counts, and they're called
    **weights**. Multiply each input by its weight, add everything up, and you
    have a **weighted sum**. In PyTorch that's an elementwise multiply, then a
    sum. The `.0` makes these decimal numbers, which models compute with.

    ```python
    flat = torch.tensor([50.0, 2.0])
    weights = torch.tensor([10.0, 100.0])

    (flat * weights).sum()    # tensor(700.)
    ```

    Multiplying pairs and adding them up has a name, the **dot product**.
    PyTorch writes it with `@`, so `flat @ weights` gives the same 700. Both
    sides of `@` must hold the same kind of number, whole or decimal, never one
    of each.

    Now price three flats at once, one per row.

    ```python
    flats = torch.tensor([[50.0, 2.0],
                          [80.0, 3.0],
                          [30.0, 1.0]])

    flats @ weights    # tensor([ 700., 1100.,  400.])
    ```

    Each row got its own dot product with the same weights. A grid of numbers
    like `flats` is a **matrix**.

    What if you can't choose between two rules of thumb? Put each rule in its
    own column, and `@` gives every flat a rent under each rule.

    ```python
    rules = torch.tensor([[10.0, 12.0],
                          [100.0, 50.0]])   # one column per rule

    flats @ rules    # tensor([[700., 700.], [1100., 1110.], [400., 410.]])
    ```

    Read the shapes. `(3, 2) @ (2, 2)` gives `(3, 2)`. The inner sizes must
    match, since each row of `flats` needs one weight per input. The outer
    sizes are what you get, one row per flat and one column per rule. This is
    **matrix multiplication**, and most of deep learning runs on it.

    Data won't always arrive the way `@` wants it. `.T` swaps rows and columns,
    so a tensor of shape `(5, 3)` becomes `(3, 5)` and its first row becomes its
    first column.

    A model's weights are numbers like these, and training is the search for
    good ones.
    """,
)

MATMUL1_SOURCE = source_file(
    '''
    # matmul1.py

    import torch
    from torch import Tensor


    # TODO: Return one student's final grade.
    #   marks holds three marks, homework first, then the midterm, then the final exam.
    #   Homework counts 20 percent, the midterm 30 percent and the final exam 50 percent.
    #   Marks of 80, 70 and 90 give 0.2 × 80 + 0.3 × 70 + 0.5 × 90 = 82.
    def final_grade(marks: Tensor) -> Tensor:
        ...


    # TODO: Grade every student under every grading scheme, using @.
    #   marks has one row per student, with the same three marks as above.
    #   schemes has one row per scheme, each holding three weights like 0.2, 0.3 and 0.5.
    #   Return one row per student and one column per scheme.
    def grades_under_schemes(
        marks: Tensor,
        schemes: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   final_grade of marks 80, 70 and 90 is 82, and of 100, 50 and 0 is 35
    #   grades_under_schemes of 2 students under 2 schemes, a (2, 2) result
    #   grades_under_schemes of 3 students under 3 schemes, where a wrong turn still runs
    ''',
)

MATMUL1_SOLUTION = source_file(
    '''
    # matmul1.py, a solution for comparison

    import torch
    from torch import Tensor


    def final_grade(marks: Tensor) -> Tensor:
        return (marks * torch.tensor([0.2, 0.3, 0.5])).sum()


    # schemes.T has one column per scheme, so (students, 3) @ (3, schemes) gives (students, schemes).
    def grades_under_schemes(
        marks: Tensor,
        schemes: Tensor,
    ) -> Tensor:
        return marks @ schemes.T
    ''',
)


def check_matmul1(checker: Checker) -> None:
    two_students = torch.tensor([[80.0, 70.0, 90.0], [100.0, 50.0, 0.0]])
    two_schemes = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.5, 0.0]])
    three_students = torch.tensor([[80.0, 70.0, 90.0], [100.0, 50.0, 0.0], [60.0, 60.0, 60.0]])
    three_schemes = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])

    checker.test(
        function="final_grade",
        description="is 82 for marks 80, 70 and 90",
        run=lambda final_grade: expect_value(
            call="final_grade(torch.tensor([80.0, 70.0, 90.0]))",
            actual=round(float(final_grade(torch.tensor([80.0, 70.0, 90.0]))), 4),
            expected=82.0,
        ),
    )
    checker.test(
        function="final_grade",
        description="is 35 for marks 100, 50 and 0",
        run=lambda final_grade: expect_value(
            call="final_grade(torch.tensor([100.0, 50.0, 0.0]))",
            actual=round(float(final_grade(torch.tensor([100.0, 50.0, 0.0]))), 4),
            expected=35.0,
        ),
    )
    checker.test(
        function="grades_under_schemes",
        description="gives [[82, 75], [35, 75]] for 2 students under 2 schemes",
        run=lambda grades_under_schemes: expect_tensor(
            call="grades_under_schemes with schemes [[0.2, 0.3, 0.5], [0.5, 0.5, 0.0]]",
            actual=grades_under_schemes(two_students, two_schemes),
            expected=two_students @ two_schemes.T,
        ),
    )
    checker.test(
        function="grades_under_schemes",
        description="gives one column per scheme for 3 students under 3 schemes",
        run=lambda grades_under_schemes: expect_tensor(
            call="grades_under_schemes with 3 students and schemes [[0.2, 0.3, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]",
            actual=grades_under_schemes(three_students, three_schemes),
            expected=three_students @ three_schemes.T,
        ),
    )


MATMUL1 = Exercise(
    name="matmul1",
    chapter=MATMUL,
    title="The dot product is a weighted sum",
    lesson=MATMUL1_LESSON,
    source=MATMUL1_SOURCE,
    solution=MATMUL1_SOLUTION,
    hints=(
        "final_grade is a weighted sum where you pick the weights from the TODO. For grades_under_schemes, write "
        "down the shape of marks and the shape of schemes before any code.",
        "The weights are 0.2, 0.3 and 0.5, in a tensor of decimals. For @ the inner sizes must match. marks is "
        "(students, 3) and schemes is (schemes, 3), so the schemes have to become columns first.",
        "(marks * torch.tensor([0.2, 0.3, 0.5])).sum(), and marks @ schemes.T.",
    ),
    introduces=("weight", "weighted sum", "dot product", "matrix", "matrix multiplication"),
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
    - Each exercise has three hints. `2` opens the first, and the next one opens after you save again.
    - The arrow keys and the mouse wheel scroll. `l` lists every exercise, `x` resets the file, and `q` quits.
    - Drag over text to select it. It's copied when you let go.
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
KEPT_NAMES = {"torch", "None", "True", "False"}


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
    """Code as tokens, with every plain name turned into X, so a copy with renamed variables still matches."""
    tokens: list[str] = []

    for line in code.splitlines():
        for token in CODE_TOKEN.findall(line.split("#")[0]):
            is_plain_name = (token[0].isalpha() or token[0] == "_") and token not in KEPT_NAMES
            after_dot = bool(tokens) and tokens[-1] == "."
            tokens.append("X" if is_plain_name and not after_dot else token)

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
    sentences = prose_sentences(exercise.lesson) + [sentence for hint in exercise.hints for sentence in prose_sentences(hint)]
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

    # The lesson and the first two hints must leave the work to the learner. The third hint is the answer.
    shown = shown_code(exercise.lesson) + " | " + " ".join(code_tokens(" ".join(exercise.hints[:2]))) + " "

    for function, body in solution_bodies(exercise.solution).items():
        if len(body) >= SHORTEST_ANSWER and f" {' '.join(body)} " in shown:
            findings.append(f"shows the answer to {function} before the third hint")

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


HINT_LABELS = ("Where to look", "The idea", "The answer")


def hint_view(
    exercise: Exercise,
    hints_open: int,
    next_ready: bool,
    run_again: str,
) -> RenderableType:
    if hints_open == 0:
        return Text("No hint is open yet. Press 2 to open the first one.", style="dim")

    parts: list[RenderableType] = []

    for label, hint in zip(HINT_LABELS, exercise.hints[:hints_open]):
        parts.append(Text(label, style="bold yellow"))
        parts.append(indented(text=hint, left=2))
        parts.append(Text(""))

    if hints_open < len(exercise.hints):
        note = "Press 2 for the next hint." if next_ready else f"The next hint opens after you {run_again}."
        parts.append(Text(note, style="dim"))

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

    # Two bindings share 2 and h, and check_action leaves one of them on, so the footer says whether a new hint is ready.
    BINDINGS = [
        Binding("n", "next_exercise", "next"),
        Binding("r", "check_now", "run"),
        Binding("1", "show_tab('lesson')", "lesson"),
        Binding("2,h", "show_hints", "hint"),
        Binding("2,h", "next_hint", "next hint"),
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
        self.runs = 0
        self.hints_open = 0
        self.runs_at_last_hint = 0
        self.show_result_when_checked = False
        self.watched_time = 0.0
        self.check_again = "Press r to check again." if manual else "Save the file to check again."
        self.run_again = "press r again" if manual else "save again"

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
        self.runs = 0
        self.hints_open = 0
        self.runs_at_last_hint = 0
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
        self.runs += 1
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

    def next_hint_ready(self) -> bool:
        return 0 < self.hints_open < len(self.exercise.hints) and self.runs > self.runs_at_last_hint

    def show_hint(self) -> None:
        label = "Hint" if self.hints_open == 0 else f"Hint {self.hints_open} of {len(self.exercise.hints)}"
        self.query_one(TabbedContent).get_tab("hint-tab").label = label
        self.query_one("#hint-text", Static).update(
            hint_view(
                exercise=self.exercise,
                hints_open=self.hints_open,
                next_ready=self.next_hint_ready(),
                run_again=self.run_again,
            ),
        )

    def open_hint_level(self) -> None:
        self.hints_open += 1
        self.runs_at_last_hint = self.runs
        self.show_hint()
        self.action_show_tab("hint")
        self.query_one("#hint", VerticalScroll).scroll_end(animate=False)
        self.refresh_bindings()

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],
    ) -> bool | None:
        if action == "next_exercise":
            return self.passed

        if action == "check_now":
            return self.manual

        if action == "show_hints":
            return not self.next_hint_ready()

        if action == "next_hint":
            return self.next_hint_ready()

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

    def action_show_hints(self) -> None:
        if self.hints_open == 0:
            self.open_hint_level()
            return

        self.action_show_tab("hint")

    def action_next_hint(self) -> None:
        self.open_hint_level()

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

    def on_text_selected(self) -> None:
        """Copy a selection when the mouse lets go, as Claude Code does. Cmd+C never reaches a terminal app on a Mac."""
        text = self.screen.get_selected_text()

        if not text:
            return

        self.copy_to_clipboard(text)
        self.notify(
            "Copied",
            timeout=1.5,
        )

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
    {START_COMMAND} hint [name]        print the three hints for an exercise
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
            for label, hint in zip(HINT_LABELS, exercise.hints):
                print(f"{label}\n{textwrap.indent(textwrap.fill(hint, width=76), '  ')}\n")

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
