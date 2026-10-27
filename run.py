"""tensorlings, small exercises that teach deep learning from tensors up.

Set the course up once, then start it.

    .venv/bin/python run.py init
    .venv/bin/python run.py

Each exercise is a Python file in exercises/. This file holds the app, the
lessons, the checks and the solutions. You never need to read it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import select
import sys
import tempfile
import termios
import textwrap
import time
import traceback
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

try:
    import torch
    from rich.align import Align
    from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.segment import Segment
    from rich.table import Table
    from rich.text import Text
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

    def function(
        self,
        name: str,
    ) -> Callable[..., object]:
        found = getattr(self.module, name, None)

        if not callable(found):
            raise Problem(
                f"{self.path.name} has no function called {name}. Check the name in its def line. "
                "Pressing x resets the whole file and erases your code.",
            )

        return found

    def test(
        self,
        description: str,
        run: Callable[[], None],
    ) -> None:
        try:
            run()
        except Mismatch as mismatch:
            self.results.append(
                TestResult(
                    description=description,
                    passed=False,
                    detail=str(mismatch),
                ),
            )
            return
        except Exception as error:
            self.results.append(
                TestResult(
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


def reject_missing_return(
    call: str,
    actual: object,
) -> None:
    if actual is None:
        raise Mismatch(f"{call} returned None, which usually means the function has no return statement yet.")


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
    hint: str
    introduces: tuple[str, ...]
    check: Callable[[Checker], None]

    @property
    def path(self) -> Path:
        return EXERCISES_DIR / self.chapter.folder / f"{self.name}.py"

    @property
    def solution_path(self) -> Path:
        return SOLUTIONS_DIR / self.chapter.folder / f"{self.name}.py"


def attempt(
    exercise: Exercise,
    path: Path,
) -> Outcome:
    printed = io.StringIO()

    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
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
    A **model** is a program that takes numbers in and gives numbers out. A
    program that reads a photo and names what is in it is a model. So is one
    that guesses a price, or one that writes the next word of a sentence. A
    model learns by adjusting its own numbers until its outputs are right. That
    process is called **training**.

    Every number a model reads, stores or produces lives in a **tensor**. A
    tensor is a grid of numbers, and each number in it is called an entry. Here
    is a tensor with 2 rows and 3 columns.

    ```python
    import torch

    grid = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
    ```

    Each direction you can move along in a tensor is a **dimension**. This grid
    has two. Dimension 0 counts the rows, and dimension 1 counts the columns.
    The **shape** lists the size of each dimension, so this grid has shape
    `(2, 3)`, 2 rows and 3 columns.

    A tensor can have any number of dimensions.

    | Tensor | Shape |
    |---|---|
    | one number, `torch.tensor(7)` | `()`, no dimensions at all |
    | a list of 3 numbers, `torch.tensor([1, 2, 3])` | `(3,)` |
    | the grid above | `(2, 3)` |
    | 10 grids like it, stacked | `(10, 2, 3)` |

    `grid.shape` gives the shape. It prints as `torch.Size([2, 3])` and behaves
    like the tuple `(2, 3)`, and `tuple(grid.shape)` turns it into a plain
    tuple. `grid.numel()` counts every entry, 6 here. The checks also use
    `torch.zeros(4, 5)`, which makes a tensor of shape `(4, 5)` full of zeros.

    **Why it exists.** A tensor does math on all of its entries at once, fast
    enough for models with millions of numbers. A nested Python list cannot. A
    model's inputs, its own numbers and its outputs are all tensors. Most bugs
    in model code are shapes that do not match, so reading shapes is the first
    skill to learn.

    **Your task.** Finish the three functions in `tensors1.py`.
    """,
)

TENSORS1_SOURCE = source_file(
    '''
    # tensors1.py

    import torch
    from torch import Tensor


    # TODO: Return a tensor that holds this grid, 2 rows and 3 columns.
    #
    #     1  2  3
    #     4  5  6
    #
    def make_grid() -> Tensor:
        ...


    # TODO: Return the shape of the tensor you are given, as a tuple.
    def shape_of(values: Tensor) -> tuple[int, ...]:
        ...


    # TODO: Return how many numbers the tensor holds in total.
    def count_entries(values: Tensor) -> int:
        ...


    # What the check tests
    #   make_grid() holds 1 2 3 in its first row and 4 5 6 in its second
    #   shape_of(torch.zeros(4, 5)) is (4, 5)
    #   shape_of(torch.zeros(7)) is (7,)
    #   count_entries(torch.zeros(4, 5)) is 20
    ''',
)

TENSORS1_SOLUTION = source_file(
    '''
    # tensors1.py, a solution for comparison

    import torch
    from torch import Tensor


    def make_grid() -> Tensor:
        return torch.tensor([[1, 2, 3], [4, 5, 6]])


    def shape_of(values: Tensor) -> tuple[int, ...]:
        return tuple(values.shape)


    def count_entries(values: Tensor) -> int:
        return values.numel()
    ''',
)


def check_tensors1(checker: Checker) -> None:
    make_grid = checker.function("make_grid")
    shape_of = checker.function("shape_of")
    count_entries = checker.function("count_entries")

    checker.test(
        description="make_grid() holds 1 2 3 above 4 5 6",
        run=lambda: expect_tensor(
            call="make_grid()",
            actual=make_grid(),
            expected=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ),
    )
    checker.test(
        description="shape_of(torch.zeros(4, 5)) is (4, 5)",
        run=lambda: expect_value(
            call="shape_of(torch.zeros(4, 5))",
            actual=shape_of(torch.zeros(4, 5)),
            expected=(4, 5),
        ),
    )
    checker.test(
        description="shape_of(torch.zeros(7)) is (7,)",
        run=lambda: expect_value(
            call="shape_of(torch.zeros(7))",
            actual=shape_of(torch.zeros(7)),
            expected=(7,),
        ),
    )
    checker.test(
        description="count_entries(torch.zeros(4, 5)) is 20",
        run=lambda: expect_value(
            call="count_entries(torch.zeros(4, 5))",
            actual=count_entries(torch.zeros(4, 5)),
            expected=20,
        ),
    )


TENSORS1 = Exercise(
    name="tensors1",
    chapter=TENSORS,
    title="What a tensor is",
    lesson=TENSORS1_LESSON,
    source=TENSORS1_SOURCE,
    solution=TENSORS1_SOLUTION,
    hint=(
        "torch.tensor() turns a nested Python list into a tensor, with one inner list per row. "
        "values.shape gives the shape, and tuple(values.shape) turns it into a plain tuple. "
        "values.numel() counts the entries."
    ),
    introduces=("model", "training", "tensor", "shape", "dimension"),
    check=check_tensors1,
)

TENSORS2_LESSON = prose(
    """
    Models often need one part of a tensor. A model that writes text only needs
    its output at the last word. Training feeds a model 32 rows of data at a
    time, out of thousands.

    Take this grid of shape `(3, 4)`.

    ```python
    grid = torch.tensor([[ 1,  2,  3,  4],
                         [ 5,  6,  7,  8],
                         [ 9, 10, 11, 12]])
    ```

    An **index** picks one position along a dimension. Counting starts at 0,
    and negative numbers count back from the end. Inside the brackets you give
    one index per dimension, separated by commas. The first picks the row and
    the second picks the column. A bare `:` means every position along that
    dimension.

    | Code | Result | What it picks |
    |---|---|---|
    | `grid[0]` | `tensor([1, 2, 3, 4])` | row 0 |
    | `grid[-1]` | `tensor([9, 10, 11, 12])` | the last row |
    | `grid[1, 2]` | `tensor(7)` | row 1, column 2 |
    | `grid[:, 0]` | `tensor([1, 5, 9])` | column 0 of every row |

    A **slice** picks a range, written `start:stop`. It includes `start` and
    stops one short of `stop`. Leaving out `start` means from the beginning,
    and leaving out `stop` means to the end.

    | Code | What it picks | Shape |
    |---|---|---|
    | `grid[0:2]` | rows 0 and 1 | `(2, 4)` |
    | `grid[:, 1:3]` | columns 1 and 2 of every row | `(3, 2)` |
    | `grid[:2, :2]` | the top left 2 by 2 corner | `(2, 2)` |

    An index removes its dimension, and a slice keeps it. `grid[0]` has shape
    `(4,)`, while `grid[0:1]` has shape `(1, 4)`. `grid[1, 2]` removes both
    dimensions, which leaves shape `()`.

    **Why it exists.** Picking out parts is how a model selects the data it
    works on, in one line with no loop. The output at the last word is
    `output[-1]`, and a group of 32 rows is `data[0:32]`.

    **Your task.** Finish the three functions in `tensors2.py`.
    """,
)

TENSORS2_SOURCE = source_file(
    '''
    # tensors2.py

    import torch
    from torch import Tensor


    # TODO: Return the first row of the grid.
    def first_row(grid: Tensor) -> Tensor:
        ...


    # TODO: Return the last column of the grid, one number from every row.
    def last_column(grid: Tensor) -> Tensor:
        ...


    # TODO: Return the top left corner of the grid, size rows by size columns.
    def top_left_corner(
        grid: Tensor,
        size: int,
    ) -> Tensor:
        ...


    # What the check tests, on the 3 by 4 grid from the lesson and on one other grid
    #   first_row(grid) is [1, 2, 3, 4]
    #   last_column(grid) is [4, 8, 12]
    #   top_left_corner(grid, 2) is [[1, 2], [5, 6]]
    ''',
)

TENSORS2_SOLUTION = source_file(
    '''
    # tensors2.py, a solution for comparison

    import torch
    from torch import Tensor


    def first_row(grid: Tensor) -> Tensor:
        return grid[0]


    def last_column(grid: Tensor) -> Tensor:
        return grid[:, -1]


    def top_left_corner(
        grid: Tensor,
        size: int,
    ) -> Tensor:
        return grid[:size, :size]
    ''',
)


def check_tensors2(checker: Checker) -> None:
    first_row = checker.function("first_row")
    last_column = checker.function("last_column")
    top_left_corner = checker.function("top_left_corner")
    grid = torch.arange(1, 13).reshape(3, 4)
    other = torch.arange(10, 35).reshape(5, 5)

    checker.test(
        description="first_row(grid) is [1, 2, 3, 4]",
        run=lambda: expect_tensor(
            call="first_row(grid)",
            actual=first_row(grid),
            expected=grid[0],
        ),
    )
    checker.test(
        description="last_column(grid) is [4, 8, 12]",
        run=lambda: expect_tensor(
            call="last_column(grid)",
            actual=last_column(grid),
            expected=grid[:, -1],
        ),
    )
    checker.test(
        description="top_left_corner(grid, 2) is [[1, 2], [5, 6]]",
        run=lambda: expect_tensor(
            call="top_left_corner(grid, 2)",
            actual=top_left_corner(grid, 2),
            expected=grid[:2, :2],
        ),
    )
    checker.test(
        description="the same three functions work on a 5 by 5 grid",
        run=lambda: (
            expect_tensor(
                call="On the 5 by 5 grid, first_row",
                actual=first_row(other),
                expected=other[0],
            ),
            expect_tensor(
                call="On the 5 by 5 grid, last_column",
                actual=last_column(other),
                expected=other[:, -1],
            ),
            expect_tensor(
                call="On the 5 by 5 grid, top_left_corner with size 3",
                actual=top_left_corner(other, 3),
                expected=other[:3, :3],
            ),
        ),
    )


TENSORS2 = Exercise(
    name="tensors2",
    chapter=TENSORS,
    title="Picking out parts of a tensor",
    lesson=TENSORS2_LESSON,
    source=TENSORS2_SOURCE,
    solution=TENSORS2_SOLUTION,
    hint=(
        "grid[0] is the first row. grid[:, -1] takes the last entry of every row. "
        "grid[:size, :size] takes the first size rows and the first size columns."
    ),
    introduces=("index", "slice"),
    check=check_tensors2,
)

TENSORS3_LESSON = prose(
    """
    Math on tensors works entry by entry. Adding two tensors of the same shape
    adds each pair of matching entries. This is called **elementwise** math.

    ```python
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([10, 20, 30])
    a + b    # tensor([11, 22, 33])
    a * b    # tensor([10, 40, 90])
    ```

    Models work on many examples at once. A group of examples handled together
    is called a **batch**, usually one example per row. Here is a batch of 2
    students who each took 3 tests, and a bonus for each test.

    ```python
    scores = torch.tensor([[50, 60, 70],
                           [80, 90, 40]])   # shape (2, 3)
    bonus = torch.tensor([5, 0, 10])         # shape (3,)

    scores + bonus
    # tensor([[55, 60, 80],
    #         [85, 90, 50]])
    ```

    The shapes differ, yet the bonus was added to every row. PyTorch stretched
    the smaller tensor to fit, without copying it. This is called
    **broadcasting**.

    To check whether two shapes fit, line them up from the right. A size
    missing from the shorter shape counts as 1. Two sizes fit when they are
    equal, or when one of them is 1, and a size of 1 is stretched to match.

    ```text
    scores   (2, 3)
    bonus       (3,)     3 meets 3, and the missing size counts as 1
    ```

    A single number has shape `()`, so every size counts as missing and it fits
    any shape. `scores + 5` adds 5 to every entry.

    A slice keeps its dimension, so `scores[:, 0:1]` has shape `(2, 1)`, the
    first test of each student. Lined up with `(2, 3)`, its 1 is stretched
    across all 3 columns. The index `scores[:, 0]` holds the same numbers with
    shape `(2,)`, and then 2 meets 3, so it does not fit.

    **Why it exists.** A model often adds the same numbers to every example in a
    batch. Broadcasting does that in one line, with no loop.

    **Your task.** Finish the two functions in `tensors3.py`.
    """,
)

TENSORS3_SOURCE = source_file(
    '''
    # tensors3.py

    import torch
    from torch import Tensor


    # TODO: Add the bonus for each test to every student's scores.
    #   scores has one row per student and one column per test.
    #   bonus has one number per test.
    def add_bonus(
        scores: Tensor,
        bonus: Tensor,
    ) -> Tensor:
        ...


    # TODO: Subtract each student's first test score from all of that student's scores.
    #   For a student who scored 50 60 70, the answer is 0 10 20.
    def gain_since_first_test(scores: Tensor) -> Tensor:
        ...


    # What the check tests
    #   add_bonus on 2 students and on 4 students
    #   gain_since_first_test on 2 students and on 3 students with 3 tests
    ''',
)

TENSORS3_SOLUTION = source_file(
    '''
    # tensors3.py, a solution for comparison

    import torch
    from torch import Tensor


    def add_bonus(
        scores: Tensor,
        bonus: Tensor,
    ) -> Tensor:
        return scores + bonus


    def gain_since_first_test(scores: Tensor) -> Tensor:
        return scores - scores[:, 0:1]
    ''',
)


def check_tensors3(checker: Checker) -> None:
    add_bonus = checker.function("add_bonus")
    gain_since_first_test = checker.function("gain_since_first_test")
    two_students = torch.tensor([[50, 60, 70], [80, 90, 40]])
    four_students = torch.tensor([[50, 60, 70], [80, 90, 40], [10, 20, 30], [70, 70, 70]])
    three_by_three = torch.tensor([[1, 2, 3], [40, 50, 60], [700, 800, 900]])
    bonus = torch.tensor([5, 0, 10])

    checker.test(
        description="add_bonus adds the bonus to both students",
        run=lambda: expect_tensor(
            call="add_bonus with 2 students",
            actual=add_bonus(two_students, bonus),
            expected=two_students + bonus,
        ),
    )
    checker.test(
        description="add_bonus works for 4 students too",
        run=lambda: expect_tensor(
            call="add_bonus with 4 students",
            actual=add_bonus(four_students, bonus),
            expected=four_students + bonus,
        ),
    )
    checker.test(
        description="gain_since_first_test gives [[0, 10, 20], [0, 10, -40]]",
        run=lambda: expect_tensor(
            call="gain_since_first_test with 2 students",
            actual=gain_since_first_test(two_students),
            expected=two_students - two_students[:, 0:1],
        ),
    )
    checker.test(
        description="gain_since_first_test subtracts per student when there are 3 students and 3 tests",
        run=lambda: expect_tensor(
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
    hint=(
        "add_bonus is one line, scores + bonus. For gain_since_first_test, scores[:, 0:1] is each student's first "
        "score with shape (n, 1). Subtract it from scores and the 1 is stretched across every test. scores[:, 0] "
        "has shape (n,) instead, which lines up with the tests rather than the students."
    ),
    introduces=("elementwise", "batch", "broadcasting"),
    check=check_tensors3,
)

TENSORS4_LESSON = prose(
    """
    Broadcasting lines shapes up from the right, so a tensor with one number per
    row usually does not fit. Say each student gets their own factor.

    ```python
    scores = torch.tensor([[50, 60, 70],
                           [80, 90, 40]])   # shape (2, 3)
    factors = torch.tensor([2, 10])          # shape (2,)
    ```

    ```text
    scores    (2, 3)
    factors      (2,)     2 meets 3, so they do not fit
    ```

    The fix is to give `factors` a second dimension of size 1. In
    `factors[:, None]`, the `:` keeps every factor along dimension 0, and the
    `None` in second place adds a new dimension 1 of size 1.

    ```python
    factors[:, None]             # tensor([[ 2], [10]]), shape (2, 1)
    scores * factors[:, None]    # tensor([[100, 120, 140], [800, 900, 400]])
    ```

    Now `(2, 3)` meets `(2, 1)`. The 1 is stretched across the columns, so
    every score in a row is multiplied by that row's factor.

    Watch out when there are as many rows as columns. With scores of shape
    `(3, 3)` and factors of shape `(3,)`, the shapes fit without `None`. Each
    factor then quietly scales a column instead of a row. No error appears,
    only wrong numbers, so add the dimension anyway.

    **Why it exists.** A model often scales each example by its own amount, such
    as making some examples count more during training. A dimension of size 1
    is how one number per row lines up with the rows.

    **Your task.** Finish the function in `tensors4.py`.
    """,
)

TENSORS4_SOURCE = source_file(
    '''
    # tensors4.py

    import torch
    from torch import Tensor


    # TODO: Multiply every score in a row by that row's factor.
    #   scores has one row per student and one column per test.
    #   factors has one number per student.
    def scale_each_row(
        scores: Tensor,
        factors: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   scale_each_row on the 2 students from the lesson
    #   scale_each_row on 4 students, where the shapes only fit after adding a dimension
    #   scale_each_row on 3 students and 3 tests, where a wrong answer still has the right shape
    ''',
)

TENSORS4_SOLUTION = source_file(
    '''
    # tensors4.py, a solution for comparison

    import torch
    from torch import Tensor


    def scale_each_row(
        scores: Tensor,
        factors: Tensor,
    ) -> Tensor:
        return scores * factors[:, None]
    ''',
)


def check_tensors4(checker: Checker) -> None:
    scale_each_row = checker.function("scale_each_row")
    two_students = torch.tensor([[50, 60, 70], [80, 90, 40]])
    four_students = torch.tensor([[50, 60, 70], [80, 90, 40], [10, 20, 30], [70, 70, 70]])
    three_by_three = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    checker.test(
        description="scale_each_row gives [[100, 120, 140], [800, 900, 400]] for the lesson's students",
        run=lambda: expect_tensor(
            call="scale_each_row with factors [2, 10]",
            actual=scale_each_row(two_students, torch.tensor([2, 10])),
            expected=two_students * torch.tensor([2, 10])[:, None],
        ),
    )
    checker.test(
        description="scale_each_row multiplies each of 4 rows by its own factor",
        run=lambda: expect_tensor(
            call="scale_each_row with 4 students and factors [1, 2, 0, 10]",
            actual=scale_each_row(four_students, torch.tensor([1, 2, 0, 10])),
            expected=four_students * torch.tensor([1, 2, 0, 10])[:, None],
        ),
    )
    checker.test(
        description="scale_each_row scales rows, not columns, on 3 students with 3 tests",
        run=lambda: expect_tensor(
            call="scale_each_row with 3 students, 3 tests and factors [1, 10, 100]",
            actual=scale_each_row(three_by_three, torch.tensor([1, 10, 100])),
            expected=three_by_three * torch.tensor([1, 10, 100])[:, None],
        ),
    )


TENSORS4 = Exercise(
    name="tensors4",
    chapter=TENSORS,
    title="Giving each row its own number",
    lesson=TENSORS4_LESSON,
    source=TENSORS4_SOURCE,
    solution=TENSORS4_SOLUTION,
    hint=(
        "factors has shape (n,) and scores has shape (n, 3). factors[:, None] has shape (n, 1), which lines up "
        "with the rows. Return scores * factors[:, None]."
    ),
    introduces=(),
    check=check_tensors4,
)

TENSORS5_LESSON = prose(
    """
    A **reduction** combines many entries into fewer, for example a sum or an
    average. You choose which dimension to combine with `dim`.

    ```python
    scores = torch.tensor([[50, 60, 70],
                           [80, 90, 40]])   # 2 students, 3 tests

    scores.sum()         # tensor(390), everything
    scores.sum(dim=1)    # tensor([180, 210]), one total per student
    scores.sum(dim=0)    # tensor([130, 150, 110]), one total per test
    ```

    The dimension you name is the one that disappears. `dim=1` adds across the
    columns, so each row becomes one number.

    A tensor holds either whole numbers or decimal numbers, never a mix.
    `torch.tensor([50, 60])` holds whole numbers, and `.mean()` refuses those
    with an error. `.float()` gives a copy with decimal numbers, so
    `scores.float().mean(dim=0)` gives one average per test.

    Sometimes each row has to be divided by its own total. `scores.sum(dim=1)`
    has shape `(2,)`, which does not line up with `(2, 3)`. With
    `keepdim=True`, the summed dimension stays as size 1 instead of
    disappearing, so the totals have shape `(2, 1)` and line up with each row.
    That is the same shape `scores.sum(dim=1)[:, None]` gives, in one step.

    ```python
    totals = scores.sum(dim=1, keepdim=True)   # shape (2, 1)
    scores / totals                             # each row now adds up to 1
    ```

    Division with `/` always gives decimal numbers, so it needs no `.float()`.

    **Why it exists.** Training measures how wrong a model is by averaging its
    errors over a batch, which is a reduction. A model choosing between several
    answers gives each one a share, and the shares must add up to 1. That takes
    a row total, which is a reduction, followed by a division.

    **Your task.** Finish the three functions in `tensors5.py`.
    """,
)

TENSORS5_SOURCE = source_file(
    '''
    # tensors5.py

    import torch
    from torch import Tensor


    # TODO: Return one total per row.
    def total_per_row(scores: Tensor) -> Tensor:
        ...


    # TODO: Return the average of each column, as decimal numbers.
    def average_per_column(scores: Tensor) -> Tensor:
        ...


    # TODO: Divide every row by its own total, so each row adds up to 1.
    def share_of_row(scores: Tensor) -> Tensor:
        ...


    # What the check tests, on the scores from the lesson and on a 3 by 4 grid
    #   total_per_row gives [180, 210]
    #   average_per_column gives [65.0, 75.0, 55.0]
    #   share_of_row gives rows that each add up to 1
    ''',
)

TENSORS5_SOLUTION = source_file(
    '''
    # tensors5.py, a solution for comparison

    import torch
    from torch import Tensor


    def total_per_row(scores: Tensor) -> Tensor:
        return scores.sum(dim=1)


    def average_per_column(scores: Tensor) -> Tensor:
        return scores.float().mean(dim=0)


    def share_of_row(scores: Tensor) -> Tensor:
        return scores / scores.sum(dim=1, keepdim=True)
    ''',
)


def check_tensors5(checker: Checker) -> None:
    total_per_row = checker.function("total_per_row")
    average_per_column = checker.function("average_per_column")
    share_of_row = checker.function("share_of_row")
    scores = torch.tensor([[50, 60, 70], [80, 90, 40]])
    grid = torch.arange(1, 13).reshape(3, 4)

    checker.test(
        description="total_per_row gives one total per student",
        run=lambda: (
            expect_tensor(
                call="total_per_row on the lesson's scores",
                actual=total_per_row(scores),
                expected=scores.sum(dim=1),
            ),
            expect_tensor(
                call="total_per_row on a 3 by 4 grid",
                actual=total_per_row(grid),
                expected=grid.sum(dim=1),
            ),
        ),
    )
    checker.test(
        description="average_per_column gives one average per test",
        run=lambda: (
            expect_tensor(
                call="average_per_column on the lesson's scores",
                actual=average_per_column(scores),
                expected=scores.float().mean(dim=0),
            ),
            expect_tensor(
                call="average_per_column on a 3 by 4 grid",
                actual=average_per_column(grid),
                expected=grid.float().mean(dim=0),
            ),
        ),
    )
    checker.test(
        description="share_of_row makes every row add up to 1",
        run=lambda: (
            expect_tensor(
                call="share_of_row on the lesson's scores",
                actual=share_of_row(scores),
                expected=scores / scores.sum(dim=1, keepdim=True),
            ),
            expect_tensor(
                call="share_of_row on a 3 by 4 grid",
                actual=share_of_row(grid),
                expected=grid / grid.sum(dim=1, keepdim=True),
            ),
        ),
    )


TENSORS5 = Exercise(
    name="tensors5",
    chapter=TENSORS,
    title="Adding up along a dimension",
    lesson=TENSORS5_LESSON,
    source=TENSORS5_SOURCE,
    solution=TENSORS5_SOLUTION,
    hint=(
        "scores.sum(dim=1) gives one total per row. scores.float().mean(dim=0) gives one average per column. "
        "For share_of_row, divide scores by scores.sum(dim=1, keepdim=True)."
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
    Say you guess the monthly rent of a flat from its size and its number of
    rooms. Your rule is 10 per square metre plus 100 per room.

    ```text
    size = 50 square metres, rooms = 2

    rent = 10 × 50 + 100 × 2
         = 500 + 200
         = 700
    ```

    The 10 and the 100 are **weights**. They say how much each input counts.
    Multiplying each input by its weight and adding the results gives a
    **weighted sum**.

    With the inputs in one tensor and the weights in another, a weighted sum is
    an elementwise multiply followed by a sum. The `.0` on each number makes it
    a decimal number, which is what models work with.

    ```python
    inputs = torch.tensor([50.0, 2.0])
    weights = torch.tensor([10.0, 100.0])

    (inputs * weights).sum()    # tensor(700.)
    ```

    This operation is called the **dot product**, and PyTorch writes it with
    the symbol `@`. `inputs @ weights` also gives 700. Unlike `*`, the `@`
    symbol needs both tensors to hold the same kind of number, so do not mix
    whole and decimal numbers.

    `@` also works on many flats at once, one flat per row. `flats @ weights`
    with shapes `(3, 2)` and `(2,)` gives shape `(3,)`, one rent per flat. Each
    row gets its own dot product with the same weights, so the last size of
    `flats` must equal the size of `weights`. A grid of numbers like `flats` is
    also called a **matrix**, and doing many dot products this way is **matrix
    multiplication**.

    ```python
    flats = torch.tensor([[50.0, 2.0],
                          [80.0, 3.0],
                          [30.0, 1.0]])

    flats @ weights    # tensor([ 700., 1100.,  400.])
    ```

    **Why it exists.** A weighted sum is the basic way a model combines several
    inputs into one number, and the weights are the numbers training adjusts.
    Most building blocks in deep learning are matrix multiplications
    underneath, so the rest of the course is built on this one operation.

    **Your task.** Finish the two functions in `matmul1.py`. Write the first with
    `*` and `.sum()`, and the second with `@`.
    """,
)

MATMUL1_SOURCE = source_file(
    '''
    # matmul1.py

    import torch
    from torch import Tensor


    # TODO: Return the weighted sum of the inputs, using * and .sum().
    #   inputs and weights both have shape (n,).
    def weighted_sum(
        inputs: Tensor,
        weights: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return one rent per flat, using @.
    #   flats has one row per flat and one column per input.
    #   weights has one weight per input.
    def rents(
        flats: Tensor,
        weights: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   weighted_sum([50, 2], [10, 100]) is 700
    #   weighted_sum works for 4 inputs too
    #   rents gives [700, 1100, 400] for the three flats in the lesson
    #   rents works for flats with 3 inputs each
    ''',
)

MATMUL1_SOLUTION = source_file(
    '''
    # matmul1.py, a solution for comparison

    import torch
    from torch import Tensor


    def weighted_sum(
        inputs: Tensor,
        weights: Tensor,
    ) -> Tensor:
        return (inputs * weights).sum()


    def rents(
        flats: Tensor,
        weights: Tensor,
    ) -> Tensor:
        return flats @ weights
    ''',
)


def check_matmul1(checker: Checker) -> None:
    weighted_sum = checker.function("weighted_sum")
    rents = checker.function("rents")
    flats = torch.tensor([[50.0, 2.0], [80.0, 3.0], [30.0, 1.0]])
    weights = torch.tensor([10.0, 100.0])
    bigger_flats = torch.tensor([[50.0, 2.0, 1.0], [80.0, 3.0, 0.0]])
    bigger_weights = torch.tensor([10.0, 100.0, 250.0])

    checker.test(
        description="weighted_sum([50, 2], [10, 100]) is 700",
        run=lambda: expect_tensor(
            call="weighted_sum(inputs, weights)",
            actual=weighted_sum(torch.tensor([50.0, 2.0]), weights),
            expected=torch.tensor(700.0),
        ),
    )
    checker.test(
        description="weighted_sum works for 4 inputs",
        run=lambda: expect_tensor(
            call="weighted_sum(torch.tensor([1., 2., 3., 4.]), torch.tensor([4., 3., 2., 1.]))",
            actual=weighted_sum(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([4.0, 3.0, 2.0, 1.0])),
            expected=torch.tensor(20.0),
        ),
    )
    checker.test(
        description="rents gives [700, 1100, 400] for the flats in the lesson",
        run=lambda: expect_tensor(
            call="rents(flats, weights)",
            actual=rents(flats, weights),
            expected=flats @ weights,
        ),
    )
    checker.test(
        description="rents works for flats with 3 inputs each",
        run=lambda: expect_tensor(
            call="rents with 3 inputs per flat",
            actual=rents(bigger_flats, bigger_weights),
            expected=bigger_flats @ bigger_weights,
        ),
    )


MATMUL1 = Exercise(
    name="matmul1",
    chapter=MATMUL,
    title="The dot product is a weighted sum",
    lesson=MATMUL1_LESSON,
    source=MATMUL1_SOURCE,
    solution=MATMUL1_SOLUTION,
    hint=(
        "weighted_sum is (inputs * weights).sum(). rents is flats @ weights, "
        "which runs one dot product per row of flats."
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
    - This screen shows the lesson. Your editor shows the file. Keep them side by side.
    - Find the `TODO` in the file, write the code and save. The app reruns the tests and shows which ones pass.
    - When every test passes, press `n` for the next exercise.

    **Keys**

    - `h` shows or hides a hint
    - `l` lists every exercise
    - `x` resets the file to how it started
    - `↑` and `↓` scroll the lesson
    - `q` quits
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
TALLEST_LESSON = 48
LESSON_WIDTH = 96


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


def lint_exercise(
    exercise: Exercise,
    introduced_before: set[str],
    console: Console,
) -> list[str]:
    findings: list[str] = []
    known = introduced_before | set(exercise.introduces)
    sentences = prose_sentences(exercise.lesson) + prose_sentences(exercise.hint)
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

    for section in ("**Why it exists.**", "**Your task.**"):
        if section not in exercise.lesson:
            findings.append(f"has no {section} paragraph")

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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def key_bar(
    keys: list[tuple[str, str]],
    message: str = "",
) -> Text:
    bar = Text(no_wrap=True, overflow="ellipsis")

    if message:
        bar.append(f"{message}   ", style="bold yellow")

    for key, label in keys:
        bar.append(f" {key} ", style=f"bold black on {ACCENT}")
        bar.append(f" {label}   ")

    return bar


NOT_STARTED = "returned None"


def indented(
    text: str,
    style: str = "",
    left: int = 4,
) -> Padding:
    return Padding(
        Text(text, style=style, overflow="fold"),
        pad=(0, 0, 0, left),
    )


def result_body(
    exercise: Exercise,
    outcome: Outcome | None,
    check_again: str,
) -> RenderableType:
    """The result panel, most important first, since a tall result is cut off at the bottom."""
    parts: list[RenderableType] = []

    if outcome is None:
        parts.append(Text("Checking your code...", style="dim"))
        return Group(*parts)

    if outcome.problem is not None and not outcome.results:
        parts.append(Text("✗ Your file cannot be checked yet", style="bold red"))
        parts.append(indented(text=outcome.problem, left=2))
        return Group(*parts)

    if outcome.passed:
        parts.append(Text("✓ Exercise done. Every test passes.", style="bold green"))
        parts.append(Text.assemble("A solution to compare with is in ", (relative(exercise.solution_path), ACCENT), "."))
        parts.append(Text("Press n for the next exercise.", style="bold"))
        parts.append(Text(""))

    not_started = 0
    explained_one = False

    for result in outcome.results:
        if result.passed:
            parts.append(Text.assemble(("✓ ", "bold green"), result.description))
            continue

        parts.append(Text.assemble(("✗ ", "bold red"), result.description))

        if NOT_STARTED in result.detail:
            not_started += 1
            continue

        # Like a compiler, explain only the first failure in full, so it is never cut off.
        if not explained_one:
            parts.append(indented(text=result.detail, style="yellow"))
            explained_one = True

    if outcome.problem is not None:
        parts.append(Text(""))
        parts.append(indented(text=outcome.problem, style="red", left=0))

    if not outcome.passed:
        passed_count = sum(result.passed for result in outcome.results)
        parts.append(Text(""))

        if not_started:
            parts.append(Text("A function marked ✗ with no message has no return statement yet. Write the code under its TODO."))

        parts.append(Text(f"{passed_count} of {len(outcome.results)} tests pass. {check_again}", style="dim"))

    if outcome.printed.strip():
        printed_lines = outcome.printed.rstrip().splitlines()[-12:]
        parts.append(Text(""))
        parts.append(Text("What your code printed", style="bold"))
        parts.append(indented(text="\n".join(printed_lines), style="dim", left=2))

    return Group(*parts)


@dataclass
class Scroll:
    offset: int = 0
    limit: int = 0


class ScrollView:
    """Shows a window of pre-rendered lines, as tall as the space it is given."""

    def __init__(
        self,
        lines: list[list[Segment]],
        offset: int,
    ) -> None:
        self.lines = lines
        self.offset = offset

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        height = options.height or len(self.lines)

        for line in self.lines[self.offset : self.offset + height]:
            yield from line
            yield Segment.line()


def without_double_blanks(lines: list[list[Segment]]) -> list[list[Segment]]:
    """Rich puts two blank lines around code blocks and tables. One is enough on a small screen."""
    kept: list[list[Segment]] = []
    previous_blank = False

    for line in lines:
        blank = not "".join(segment.text for segment in line).strip()

        if blank and previous_blank:
            continue

        kept.append(line)
        previous_blank = blank

    return kept


HEADER_HEIGHT = 3


def header(
    exercise: Exercise,
    progress: Progress,
) -> RenderableType:
    done = len(progress.done)
    total = len(EXERCISES)
    progress_line = Table.grid(padding=(0, 2))
    progress_line.add_row(
        ProgressBar(
            total=total,
            completed=done,
            width=20,
            complete_style="green",
        ),
        Text(f"{done} of {total} done"),
    )

    grid = Table.grid(
        expand=True,
        padding=(0, 2),
    )
    grid.add_column(
        ratio=1,
        no_wrap=True,
        overflow="ellipsis",
    )
    grid.add_column(no_wrap=True)
    grid.add_row(
        Text("tensorlings", style=f"bold {ACCENT}"),
        Align.right(progress_line),
    )
    grid.add_row(
        Text.assemble((exercise.name, "bold"), " · ", exercise.title),
        Align.right(Text(relative(exercise.path), style=ACCENT)),
    )

    return Padding(
        grid,
        pad=(0, 1, 1, 1),
    )


def render_watch(
    console: Console,
    exercise: Exercise,
    progress: Progress,
    outcome: Outcome | None,
    show_hint: bool,
    check_again: str,
    scroll: Scroll,
    keys: list[tuple[str, str]],
    message: str,
) -> RenderableType:
    width, height = console.size
    body = result_body(
        exercise=exercise,
        outcome=outcome,
        check_again=check_again,
    )
    result_lines = len(console.render_lines(body, console.options.update_width(width - 4), pad=False))
    result_height = min(result_lines + 2, max(6, height * 40 // 100))

    hint = Text(exercise.hint, overflow="fold")
    hint_lines = len(console.render_lines(hint, console.options.update_width(width - 4), pad=False))
    hint_height = min(hint_lines + 2, max(4, height * 25 // 100)) if show_hint else 0

    lesson_height = max(3, height - HEADER_HEIGHT - 1 - hint_height - result_height - 2)
    lesson_lines = without_double_blanks(
        console.render_lines(
            Markdown(exercise.lesson, code_theme=CODE_THEME),
            console.options.update_width(width - 4),
            pad=False,
        ),
    )
    scroll.limit = max(0, len(lesson_lines) - lesson_height)
    scroll.offset = min(max(scroll.offset, 0), scroll.limit)

    if scroll.limit == 0:
        lesson_note = ""
    elif scroll.offset < scroll.limit:
        lesson_note = "↓ more below, scroll with ↑ ↓"
    else:
        lesson_note = "↑ more above"

    if outcome is None:
        result_style = "yellow"
    elif outcome.passed:
        result_style = "green"
    else:
        result_style = "red"

    sections: list[Layout] = [
        Layout(
            header(
                exercise=exercise,
                progress=progress,
            ),
            size=HEADER_HEIGHT,
        ),
        Layout(
            Panel(
                ScrollView(
                    lines=lesson_lines,
                    offset=scroll.offset,
                ),
                title=f"Lesson · {exercise.chapter.title}",
                title_align="left",
                subtitle=lesson_note,
                subtitle_align="right",
                border_style=ACCENT,
            ),
            ratio=1,
        ),
    ]

    if show_hint:
        sections.append(
            Layout(
                Panel(
                    hint,
                    title="Hint",
                    title_align="left",
                    subtitle="more below, make the window taller to see it" if hint_lines + 2 > hint_height else "",
                    subtitle_align="right",
                    border_style="yellow",
                ),
                size=hint_height,
            ),
        )

    sections.append(
        Layout(
            Panel(
                body,
                title="Result",
                title_align="left",
                subtitle="more below, make the window taller to see it" if result_lines + 2 > result_height else "",
                subtitle_align="right",
                border_style=result_style,
            ),
            size=result_height,
        ),
    )
    sections.append(
        Layout(
            key_bar(
                keys=keys,
                message=message,
            ),
            size=1,
        ),
    )

    layout = Layout()
    layout.split_column(*sections)

    return layout


@dataclass
class ListView:
    selected: int
    show: str | None = None
    query: str = ""
    searching: bool = False
    message: str = ""


def visible_exercises(
    view: ListView,
    progress: Progress,
) -> list[Exercise]:
    if view.show == "done":
        return [exercise for exercise in EXERCISES if exercise.name in progress.done]

    if view.show == "pending":
        return [exercise for exercise in EXERCISES if exercise.name not in progress.done]

    return list(EXERCISES)


def render_list(
    console: Console,
    view: ListView,
    rows: list[Exercise],
    progress: Progress,
) -> RenderableType:
    _, height = console.size
    room = max(3, height - 7)
    first_row = min(max(0, view.selected - room + 1), max(0, len(rows) - room))
    table = Table(
        expand=True,
        box=None,
        show_edge=False,
        padding=(0, 1),
        header_style="bold",
    )
    table.add_column("", width=2)
    table.add_column("State", width=8)
    table.add_column("Name", width=14)
    table.add_column("Title", ratio=1, no_wrap=True)
    table.add_column("File", ratio=1, no_wrap=True)

    for index, exercise in enumerate(rows[first_row : first_row + room], start=first_row):
        is_done = exercise.name in progress.done
        name = Text(exercise.name)

        if view.query:
            name.highlight_words([view.query], style="magenta")

        table.add_row(
            Text("▶", style="bold red") if exercise.name == progress.current else Text(""),
            Text("DONE", style="green") if is_done else Text("PENDING", style="yellow"),
            name,
            exercise.title,
            Text(relative(exercise.path), style="dim"),
            style="reverse" if index == view.selected else None,
        )

    done_label = "done" if view.show != "done" else "done (on)"
    pending_label = "pending" if view.show != "pending" else "pending (on)"

    if view.searching:
        footer = Text(f" search  {view.query}▏", style="magenta")
    else:
        footer = key_bar(
            keys=[
                ("↑ ↓", "move"),
                ("c", "continue here"),
                ("r", "reset"),
                ("s", "search"),
                ("d", done_label),
                ("p", pending_label),
                ("q", "back"),
            ],
            message=view.message,
        )

    layout = Layout()
    layout.split_column(
        Layout(
            Panel(
                table,
                title="Exercises",
                title_align="left",
                subtitle=f"{len(progress.done)} of {len(EXERCISES)} done",
                subtitle_align="right",
                border_style=ACCENT,
            ),
            ratio=1,
        ),
        Layout(footer, size=1),
    )

    return layout


def render_page(
    title: str,
    text: str,
    keys: list[tuple[str, str]],
) -> RenderableType:
    layout = Layout()
    layout.split_column(
        Layout(
            Panel(
                Markdown(text, code_theme=CODE_THEME),
                title=title,
                title_align="left",
                border_style=ACCENT,
                padding=(0, 2),
            ),
            ratio=1,
        ),
        Layout(key_bar(keys=keys), size=1),
    )

    return layout


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

ESCAPE_SEQUENCES = {
    b"[A": "up",
    b"[B": "down",
    b"[5~": "page_up",
    b"[6~": "page_down",
    b"[H": "home",
    b"[F": "end",
    b"OH": "home",
    b"OF": "end",
}


class KeyReader:
    """Reads single key presses without Enter while the app runs."""

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self.saved_settings: list | None = None

    def __enter__(self) -> KeyReader:
        if self.enabled:
            self.saved_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

        return self

    def __exit__(self, *_: object) -> None:
        if self.saved_settings is None:
            return

        termios.tcsetattr(
            sys.stdin.fileno(),
            termios.TCSADRAIN,
            self.saved_settings,
        )

    def ready(
        self,
        timeout: float,
    ) -> bool:
        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            timeout,
        )
        return bool(readable)

    def read(
        self,
        timeout: float,
    ) -> str | None:
        if not self.enabled:
            time.sleep(timeout)
            return None

        if not self.ready(timeout):
            return None

        first = os.read(sys.stdin.fileno(), 1)

        if first == b"\x1b":
            return self.read_escape_sequence()

        if first in (b"\r", b"\n"):
            return "enter"

        if first in (b"\x7f", b"\x08"):
            return "backspace"

        return first.decode(errors="ignore")

    def read_escape_sequence(self) -> str:
        """An arrow or paging key after ESC. Reads only up to the sequence's final byte,
        so a held arrow key, which sends sequences back to back, is read one at a time."""
        if not self.ready(0.03):
            return "escape"

        introducer = os.read(sys.stdin.fileno(), 1)

        if introducer not in (b"[", b"O"):
            return "escape"

        sequence = introducer

        while self.ready(0.03):
            byte = os.read(sys.stdin.fileno(), 1)
            sequence += byte

            if 0x40 <= byte[0] <= 0x7E:
                break

        return ESCAPE_SEQUENCES.get(sequence, "escape")


def modified_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


@dataclass
class App:
    progress: Progress
    manual: bool
    console: Console = field(default_factory=terminal_console)
    keys: KeyReader = field(default_factory=KeyReader)
    outcome: Outcome | None = None
    show_hint: bool = False
    scroll: Scroll = field(default_factory=Scroll)
    message: str = ""
    live: Live | None = None

    @property
    def exercise(self) -> Exercise:
        return exercise_named(self.progress.current)

    def draw(
        self,
        renderable: RenderableType,
    ) -> None:
        self.live.update(
            renderable,
            refresh=True,
        )

    def watch_keys(self) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []

        if self.outcome is not None and self.outcome.passed:
            keys.append(("n", "next"))

        if self.manual:
            keys.append(("r", "run"))

        keys.append(("h", "hide hint" if self.show_hint else "hint"))

        keys.extend(
            [
                ("l", "list"),
                ("x", "reset"),
                ("↑ ↓", "scroll"),
                ("q", "quit"),
            ],
        )

        return keys

    def draw_watch(
        self,
        keys: list[tuple[str, str]] | None = None,
        message: str | None = None,
    ) -> None:
        self.draw(
            render_watch(
                console=self.console,
                exercise=self.exercise,
                progress=self.progress,
                outcome=self.outcome,
                show_hint=self.show_hint,
                check_again="Press r to check again." if self.manual else "Save the file to check again.",
                scroll=self.scroll,
                keys=keys if keys is not None else self.watch_keys(),
                message=message if message is not None else self.message,
            ),
        )

    def check(self) -> None:
        self.outcome = None
        self.draw_watch()
        outcome = attempt(
            exercise=self.exercise,
            path=self.exercise.path,
        )
        record_outcome(
            exercise=self.exercise,
            outcome=outcome,
            progress=self.progress,
        )
        self.outcome = outcome

    def open_exercise(
        self,
        exercise: Exercise,
    ) -> None:
        self.progress.current = exercise.name
        save_progress(self.progress)
        self.show_hint = False
        self.scroll = Scroll()
        self.message = ""
        self.check()

    def wait_for_key(
        self,
        watched: Path | None,
    ) -> str:
        """The next key press, or "changed" when the watched file is saved or the window resized."""
        stamp = modified_time(watched) if watched else 0.0
        size = self.console.size

        while True:
            key = self.keys.read(timeout=0.25)

            if key is not None:
                return key

            if watched is not None and modified_time(watched) != stamp:
                # Editors often write a file in two steps. Let the save finish.
                time.sleep(0.15)
                return "changed"

            if self.console.size != size:
                return "resized"

    def show_page(
        self,
        title: str,
        text: str,
        keys: list[tuple[str, str]],
        accepted: set[str],
    ) -> str:
        while True:
            self.draw(
                render_page(
                    title=title,
                    text=text,
                    keys=keys,
                ),
            )
            key = self.wait_for_key(watched=None)

            if key in accepted:
                return key

    def confirm_reset(self) -> bool:
        while True:
            self.draw_watch(
                keys=[
                    ("y", "yes, reset"),
                    ("n", "no, keep my changes"),
                ],
                message=f"Put {self.exercise.path.name} back to how it started? Your changes will be lost.",
            )
            key = self.wait_for_key(watched=None)

            if key in ("y", "n", "escape"):
                return key == "y"

    def run_list(self) -> bool:
        """The exercise list. True when the learner picked an exercise to continue at."""
        names = [exercise.name for exercise in EXERCISES]
        view = ListView(selected=names.index(self.progress.current))

        while True:
            rows = visible_exercises(
                view=view,
                progress=self.progress,
            )
            view.selected = min(max(view.selected, 0), max(len(rows) - 1, 0))
            self.draw(
                render_list(
                    console=self.console,
                    view=view,
                    rows=rows,
                    progress=self.progress,
                ),
            )
            key = self.wait_for_key(watched=None)

            if view.searching:
                if key in ("enter", "escape"):
                    view.searching = False

                    if key == "escape":
                        view.query = ""

                    continue

                if key == "backspace":
                    view.query = view.query[:-1]
                elif len(key) == 1 and key.isprintable():
                    view.query += key

                matches = [index for index, exercise in enumerate(rows) if view.query and view.query in exercise.name]

                if matches:
                    view.selected = matches[0]

                continue

            view.message = ""

            if key in ("j", "down"):
                view.selected += 1
            elif key in ("k", "up"):
                view.selected -= 1
            elif key in ("g", "home"):
                view.selected = 0
            elif key in ("G", "end"):
                view.selected = len(rows) - 1
            elif key in ("c", "enter"):
                if not rows:
                    view.message = "Nothing selected."
                    continue

                self.open_exercise(rows[view.selected])
                return True
            elif key == "r":
                if not rows:
                    view.message = "Nothing selected."
                    continue

                chosen = rows[view.selected]
                reset_exercise(
                    exercise=chosen,
                    progress=self.progress,
                )
                view.message = f"{chosen.name} has been reset."
            elif key == "d":
                view.show = None if view.show == "done" else "done"
            elif key == "p":
                view.show = None if view.show == "pending" else "pending"
            elif key == "s":
                view.searching = True
                view.query = ""
            elif key in ("q", "escape"):
                return False

    def finish(self) -> bool:
        """Check every exercise once more, as rustlings does at the end. True when the learner quits."""
        self.outcome = None
        self.draw(
            render_page(
                title="Checking every exercise",
                text="Checking every exercise once more before the finish line...",
                keys=[],
            ),
        )
        failing = check_all(self.progress)

        if failing:
            names = ", ".join(exercise.name for exercise in failing)
            self.open_exercise(failing[0])
            self.message = f"These no longer pass, so you are back at the first one. {names}"
            return False

        key = self.show_page(
            title="You made it to the finish line",
            text=FINISH,
            keys=[
                ("l", "list"),
                ("q", "quit"),
            ],
            accepted={"l", "q"},
        )

        if key == "q":
            return True

        self.run_list()
        return False

    def handle_watch_key(
        self,
        key: str,
    ) -> str | None:
        """Act on one key in the main screen. Returns "quit" to leave the app."""
        self.message = ""
        passed = self.outcome is not None and self.outcome.passed

        if key in ("changed", "resized"):
            if key == "changed" and not self.manual:
                self.check()

            return None

        if key == "r" and self.manual:
            self.check()
        elif key == "n" and passed:
            upcoming = next_pending(self.progress)

            if upcoming is not None:
                self.open_exercise(upcoming)
            elif self.finish():
                return "quit"
        elif key == "n":
            self.message = "Pass every test first." if self.show_hint else "Pass every test first. Press h for a hint."
        elif key == "h":
            self.show_hint = not self.show_hint
        elif key == "l":
            self.run_list()
        elif key == "x":
            if self.confirm_reset():
                reset_exercise(
                    exercise=self.exercise,
                    progress=self.progress,
                )
                self.check()
                self.message = f"{self.exercise.path.name} is back to how it started."
        elif key in ("up", "k"):
            self.scroll.offset -= 1
        elif key in ("down", "j", " "):
            self.scroll.offset += 1
        elif key == "page_up":
            self.scroll.offset -= 10
        elif key == "page_down":
            self.scroll.offset += 10
        elif key == "q":
            return "quit"

        return None

    def run(self) -> int:
        with self.keys, Live(
            console=self.console,
            screen=True,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            self.live = live

            if not self.progress.welcomed:
                self.show_page(
                    title="Welcome to tensorlings",
                    text=WELCOME,
                    keys=[("enter", "start")],
                    accepted={"enter"},
                )
                self.progress.welcomed = True
                save_progress(self.progress)

            self.check()

            while True:
                self.draw_watch()
                key = self.wait_for_key(watched=None if self.manual else self.exercise.path)

                if self.handle_watch_key(key) == "quit":
                    break

        self.console.print(f"See you next time. Run [bold]{START_COMMAND}[/bold] to continue where you left off.")

        return 0


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
    {START_COMMAND} hint [name]        print the hint for an exercise
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
            result_body(
                exercise=exercise,
                outcome=outcome,
                check_again="Run this command again after you change the file.",
            ),
            title=f"{exercise.name} · {exercise.title}",
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
                        "\n".join(f"{'pass' if result.passed else 'FAIL'}  {result.description}  {result.detail}" for result in solution.results)
                        + (f"\n{solution.problem}" if solution.problem else ""),
                        "      ",
                    ),
                )

            if verbose:
                console.print(
                    textwrap.indent(
                        "\n".join(f"stub {'pass' if result.passed else 'FAIL'}  {result.description}\n{textwrap.indent(result.detail, '      ')}" for result in stub.results)
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

        return App(
            progress=progress,
            manual="--manual-run" in flags,
        ).run()

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
            print(exercise.hint)
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
