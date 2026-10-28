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
        ".mean() needs decimal numbers, and this tensor holds whole numbers or True and False. Call .float() on it first.",
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
        # PyTorch numbers the dimension within the index, not the tensor, so the message leaves it out.
        "Position {0} does not exist, since the dimension it picks from only has {2} entries. Counting starts at 0.",
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
            # A TODO left as ... makes code like x[...] fail, which means the function is not finished.
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
# Checks shared by the exercises
# ---------------------------------------------------------------------------


def expect_floats(
    call: str,
    actual: object,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} returned a Python {type(actual).__name__}. Turn it into a tensor with torch.tensor(...).")

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} returned a {type(actual).__name__}. It should return a tensor.")

    if not actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds integers. Add .float() so it holds floats.")


def random_tensor(
    shape: tuple[int, ...],
    seed: int = 0,
) -> Tensor:
    """Random numbers that are the same on every run, so a failing check fails the same way twice."""
    return torch.randn(
        size=shape,
        generator=torch.Generator().manual_seed(seed),
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
        chapter covers making tensors, reading parts of them, math on whole
        tensors, broadcasting and reductions. Every task is an operation that
        real models run.

        ## Further reading

        - [PyTorch, introduction to tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
        - [PyTorch, broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
        - [Dive into Deep Learning, data manipulation](https://d2l.ai/chapter_preliminaries/ndarray.html)
        """,
    ),
)

TENSORS1_LESSON = prose(
    """
    A **model** is a program that turns input numbers into output numbers.
    **Training** adjusts numbers inside the model until its outputs match the
    examples it learns from. All of these numbers live in tensors.

    A **tensor** is a grid of numbers. `torch.tensor` builds one from a list,
    or from nested lists with one inner list per row.

    ```python
    import torch

    v = torch.tensor([7, 8, 9])
    m = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    v.shape       # torch.Size([3])
    m.shape       # torch.Size([2, 3])
    m.shape[0]    # 2
    ```

    The **shape** gives the size of each **dimension**. `m` has 2 rows and 3
    columns, so its first dimension has size 2. This course writes shapes the
    short way, as `(2, 3)` and `(3,)`.

    Models read many examples at once. A **batch** holds them in one tensor,
    with the examples along the first dimension. A batch of 4 examples, each
    described by 3 numbers called features, has shape `(4, 3)`. A batch of 32
    images of 28 by 28 pixels has shape `(32, 28, 28)`.

    A tensor holds one type of number. Python ints give integers and Python
    floats give floats. Models compute in floats, so integers often need
    converting. Pass `dtype=torch.float32` to `torch.tensor`, or call
    `.float()` on the tensor.

    ```python
    m.dtype            # torch.int64
    m.float().dtype    # torch.float32
    ```

    `.reshape` puts the same entries into a new shape, keeping their order row
    by row. A size of `-1` tells it to work that size out from the others.

    ```python
    t = torch.arange(6)    # tensor([0, 1, 2, 3, 4, 5])
    t.reshape(2, 3)        # tensor([[0, 1, 2], [3, 4, 5]])
    t.reshape(3, -1)       # shape (3, 2)
    ```

    `.numel()` counts the entries, so `m.numel()` is `6`. The numbers that
    training adjusts are the model's **parameters**, and a model's size is how
    many it has. The smallest GPT-2, an early model that writes text, has about
    124 million.
    """,
)

TENSORS1_SOURCE = source_file(
    '''
    # tensors1.py

    import torch
    from torch import Tensor


    # TODO: Turn a list of examples into a batch, a tensor of floats with one row
    #   per example. make_batch([[1, 2], [3, 4], [5, 6]]) has shape (3, 2).
    def make_batch(examples: list[list[float]]) -> Tensor:
        ...


    # TODO: Turn a batch of images of shape (n, height, width) into shape
    #   (n, height * width), so each image becomes one row of pixel values.
    def flatten_images(images: Tensor) -> Tensor:
        ...


    # TODO: Return how many numbers the tensors hold together, as an int.
    #   Models count their size this way.
    def count_parameters(tensors: list[Tensor]) -> int:
        ...


    # What the check tests
    #   make_batch gives floats of the right shape, from whole numbers and from decimals
    #   flatten_images on a (2, 3, 4) batch and a (1, 28, 28) batch
    #   count_parameters on tensors of shape (784, 10) and (10,) gives 7850
    ''',
)

TENSORS1_SOLUTION = source_file(
    '''
    # tensors1.py, a solution for comparison

    import torch
    from torch import Tensor


    def make_batch(examples: list[list[float]]) -> Tensor:
        return torch.tensor(examples, dtype=torch.float32)


    # -1 works out the length of each row, height * width, from the number of entries.
    def flatten_images(images: Tensor) -> Tensor:
        return images.reshape(images.shape[0], -1)


    def count_parameters(tensors: list[Tensor]) -> int:
        return sum(tensor.numel() for tensor in tensors)
    ''',
)


def check_tensors1(checker: Checker) -> None:
    whole_numbers = [[1, 2], [3, 4], [5, 6]]
    decimals = [[0.5, 1.5, 2.5]]

    for examples in (whole_numbers, decimals):
        checker.test(
            function="make_batch",
            description=f"turns {examples} into a batch",
            run=lambda make_batch, examples=examples: expect_tensor(
                call=f"make_batch({examples})",
                actual=make_batch(examples),
                expected=torch.tensor(examples, dtype=torch.float32),
            ),
        )

    checker.test(
        function="make_batch",
        description="holds floats when the examples are whole numbers",
        run=lambda make_batch: expect_floats(
            call=f"make_batch({whole_numbers})",
            actual=make_batch(whole_numbers),
        ),
    )

    image_batches = (
        torch.arange(24).reshape(2, 3, 4),
        torch.arange(784).reshape(1, 28, 28),
    )

    for images in image_batches:
        count, height, width = images.shape
        checker.test(
            function="flatten_images",
            description=f"turns a batch of shape {shape_text(tuple(images.shape))} into shape ({count}, {height * width})",
            run=lambda flatten_images, images=images: expect_tensor(
                call=f"flatten_images on a batch of shape {shape_text(tuple(images.shape))}",
                actual=flatten_images(images),
                expected=images.reshape(images.shape[0], -1),
            ),
        )

    layer_tensors = [torch.zeros((784, 10)), torch.zeros((10,))]
    checker.test(
        function="count_parameters",
        description="counts 7850 numbers in tensors of shape (784, 10) and (10,)",
        run=lambda count_parameters: expect_value(
            call="count_parameters on tensors of shape (784, 10) and (10,)",
            actual=count_parameters(layer_tensors),
            expected=7850,
        ),
    )
    checker.test(
        function="count_parameters",
        description="counts 60 numbers in one tensor of shape (3, 4, 5)",
        run=lambda count_parameters: expect_value(
            call="count_parameters on one tensor of shape (3, 4, 5)",
            actual=count_parameters([torch.zeros((3, 4, 5))]),
            expected=60,
        ),
    )


TENSORS1 = Exercise(
    name="tensors1",
    chapter=TENSORS,
    title="Tensors and shapes",
    lesson=TENSORS1_LESSON,
    source=TENSORS1_SOURCE,
    solution=TENSORS1_SOLUTION,
    hints={
        "make_batch": "torch.tensor builds a tensor from nested lists. Ask it for floats with its dtype argument, or "
        "call .float() on what it gives back.",
        "flatten_images": "Keep the first size, which images.shape[0] gives, and pass -1 as the second size to "
        "reshape so it works out the rest.",
        "count_parameters": ".numel() counts the entries of one tensor. Add up the counts of every tensor in the "
        "list.",
    },
    introduces=("model", "training", "tensor", "shape", "dimension", "batch", "parameter"),
    check=check_tensors1,
)

TENSORS2_LESSON = prose(
    """
    Square brackets read parts of a tensor. Give one position per dimension,
    separated by commas. Positions start at 0, and negative positions count
    from the end.

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
    | `x[0:2]` | rows 0 and 1 | a range of rows |
    | `x[:-1]` | rows 0 and 1 | every row but the last |
    | `x[:, -2:]` | columns 1 and 2 | the last two columns |

    A single position is an **index**. `start:stop` is a **slice**, which
    takes the positions from `start` up to, but not including, `stop`. Leave
    out `start` to begin at 0, or `stop` to run to the end. A bare `:` takes
    the whole dimension.

    An index removes its dimension and a slice keeps it. `x[:, 0]` has shape
    `(3,)`, while `x[:, 0:1]` has shape `(3, 1)`. `.item()` turns a tensor with
    one entry into a plain Python number, so `x[1, 2].item()` is `6`.

    A tensor of positions picks several rows at once, in the order given. The
    result's shape is the shape of the positions, then the shape of one row.

    ```python
    x[torch.tensor([2, 0])]                    # tensor([[7, 8, 9], [1, 2, 3]])
    x[torch.tensor([[2, 0], [1, 1]])].shape    # torch.Size([2, 2, 3])
    ```

    Models that read text split it into **tokens**, short pieces such as
    words, and give each one an id number. A batch of sentences becomes a
    tensor of ids with shape `(sentences, length)`. A model reads a fixed
    number of tokens at a time, so it keeps the most recent ones. It also
    keeps a table with one row of numbers per token id, and reads the row of
    every id in its input.

    A table of training examples often holds each example's label, the answer
    the model should learn to give, in its last column.
    """,
)

TENSORS2_SOURCE = source_file(
    '''
    # tensors2.py

    import torch
    from torch import Tensor


    # TODO: Keep the last `context` tokens of every sentence. tokens has shape
    #   (sentences, length), and the result has shape (sentences, context).
    def last_tokens(
        tokens: Tensor,
        context: int,
    ) -> Tensor:
        ...


    # TODO: Split a table of examples into its inputs, every column but the last,
    #   and its labels, the last column. Return them as a pair, inputs first.
    def split_labels(table: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the row of table for every id in ids, in the same order. ids
    #   can have any shape, and the result has the shape of ids, then the row size.
    def look_up(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   last_tokens keeps the last 2, 1 and 4 tokens of 2 sentences of 4 tokens
    #   split_labels on a table of 4 examples with 2 inputs and a label each
    #   look_up with ids of shape (3,) and (2, 2) on a table of shape (5, 3)
    ''',
)

TENSORS2_SOLUTION = source_file(
    '''
    # tensors2.py, a solution for comparison

    import torch
    from torch import Tensor


    def last_tokens(
        tokens: Tensor,
        context: int,
    ) -> Tensor:
        return tokens[:, -context:]


    # The labels are picked by index, so they lose that dimension and have shape (examples,).
    def split_labels(table: Tensor) -> tuple[Tensor, Tensor]:
        return table[:, :-1], table[:, -1]


    def look_up(
        table: Tensor,
        ids: Tensor,
    ) -> Tensor:
        return table[ids]
    ''',
)


def part_of_pair(
    call: str,
    actual: object,
    position: int,
) -> object:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should return two tensors as a pair, like return inputs, labels.")

    return actual[position]


def check_tensors2(checker: Checker) -> None:
    tokens = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]])

    for context in (2, 1, 4):
        checker.test(
            function="last_tokens",
            description=f"keeps the last {context} of 4 tokens in each sentence",
            run=lambda last_tokens, context=context: expect_tensor(
                call=f"last_tokens(tokens, {context}) for tokens {value_text(tokens)}",
                actual=last_tokens(tokens, context),
                expected=tokens[:, -context:],
            ),
        )

    table = torch.tensor(
        [
            [0.5, 1.0, 0.0],
            [2.0, 3.5, 1.0],
            [1.5, 0.5, 1.0],
            [4.0, 2.0, 0.0],
        ],
    )

    checker.test(
        function="split_labels",
        description="gives every column but the last as the inputs",
        run=lambda split_labels: expect_tensor(
            call="the first tensor split_labels(table) returns",
            actual=part_of_pair(
                call="split_labels(table)",
                actual=split_labels(table),
                position=0,
            ),
            expected=table[:, :-1],
        ),
    )
    checker.test(
        function="split_labels",
        description="gives the last column as the labels, one per example",
        run=lambda split_labels: expect_tensor(
            call="the second tensor split_labels(table) returns",
            actual=part_of_pair(
                call="split_labels(table)",
                actual=split_labels(table),
                position=1,
            ),
            expected=table[:, -1],
        ),
    )

    rows = random_tensor((5, 3))

    for ids in (torch.tensor([3, 0, 3]), torch.tensor([[1, 4], [0, 0]])):
        checker.test(
            function="look_up",
            description=f"reads the rows for ids of shape {shape_text(tuple(ids.shape))}",
            run=lambda look_up, ids=ids: expect_tensor(
                call=f"look_up(table, {value_text(ids)}) for a table of shape (5, 3)",
                actual=look_up(rows, ids),
                expected=rows[ids],
            ),
        )


TENSORS2 = Exercise(
    name="tensors2",
    chapter=TENSORS,
    title="Indexing and slicing",
    lesson=TENSORS2_LESSON,
    source=TENSORS2_SOURCE,
    solution=TENSORS2_SOLUTION,
    hints={
        "last_tokens": "The tokens run along dimension 1. Take every row, and a slice of dimension 1 that starts "
        "context positions before the end.",
        "split_labels": "Take every row both times. The inputs are a slice of the columns that stops before the "
        "last one. The labels are the last column, picked by index.",
        "look_up": "Index table with the whole ids tensor at once. The result already has the shape of ids, then "
        "the row size.",
    },
    introduces=("index", "slice", "token"),
    check=check_tensors2,
)

TENSORS3_LESSON = prose(
    """
    Math between two tensors of the same shape works entry by entry. This is
    **elementwise** math. `+`, `-`, `*`, `/` and `**`, which raises to a power,
    all work this way. A plain number works with a tensor of any shape.

    ```python
    a = torch.tensor([1.0, -2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    a + b     # tensor([5., 3., 9.])
    a * b     # tensor([  4., -10.,  18.])
    a ** 2    # tensor([1., 4., 9.])
    a * 10    # tensor([ 10., -20.,  30.])
    ```

    Comparisons work entry by entry too. They give a tensor of `True` and
    `False`, and `.float()` turns those into 1s and 0s.

    ```python
    a > 0              # tensor([ True, False,  True])
    (a > 0).float()    # tensor([1., 0., 1.])
    b == 5             # tensor([False,  True, False])
    b != 5             # tensor([ True, False,  True])
    ```

    `torch.where(condition, x, y)` takes each entry from `x` where the
    condition is true and from `y` where it is false. A plain number works in
    place of `x` or `y`.

    ```python
    torch.where(b > 4, b, -1.0)    # tensor([-1.,  5.,  6.])
    ```

    Models run elementwise math all the time. Between their steps they often
    replace every negative number with 0, which is called ReLU. Section 3 shows
    why that matters. To see how far their outputs are from the right answers,
    called targets, they square the difference of each pair.

    Sentences in a batch have different lengths. The short ones are filled out
    with a padding id, which means no token here, so the batch keeps one shape.
    The model then sets the values at padding positions to 0, so they count for
    nothing.
    """,
)

TENSORS3_SOURCE = source_file(
    '''
    # tensors3.py

    import torch
    from torch import Tensor


    # TODO: Return x with every negative entry replaced by 0 and every other entry
    #   kept. This is ReLU.
    def relu(x: Tensor) -> Tensor:
        ...


    # TODO: Return the squared difference between each prediction and its target.
    #   predictions and targets have the same shape, and so does the result.
    def squared_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return values with a 0 wherever ids holds pad_id. values and ids both
    #   have shape (sentences, length).
    def mask_padding(
        values: Tensor,
        ids: Tensor,
        pad_id: int,
    ) -> Tensor:
        ...


    # What the check tests
    #   relu on a flat tensor with negatives, a zero and positives, and on a (3, 4) tensor
    #   squared_error on 3 predictions and on a (2, 3) batch of them
    #   mask_padding with pad_id 0 and with pad_id 9
    ''',
)

TENSORS3_SOLUTION = source_file(
    '''
    # tensors3.py, a solution for comparison

    import torch
    from torch import Tensor


    def relu(x: Tensor) -> Tensor:
        return torch.where(x > 0, x, 0.0)


    def squared_error(
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return (predictions - targets) ** 2


    def mask_padding(
        values: Tensor,
        ids: Tensor,
        pad_id: int,
    ) -> Tensor:
        return torch.where(ids == pad_id, 0.0, values)
    ''',
)


def check_tensors3(checker: Checker) -> None:
    mixed = torch.tensor([-2.0, 0.0, 3.5, -0.1])
    grid = random_tensor((3, 4))

    checker.test(
        function="relu",
        description="turns the negatives of a flat tensor into 0",
        run=lambda relu: expect_tensor(
            call=f"relu({value_text(mixed)})",
            actual=relu(mixed),
            expected=torch.tensor([0.0, 0.0, 3.5, 0.0]),
        ),
    )
    checker.test(
        function="relu",
        description="keeps the shape of a (3, 4) tensor",
        run=lambda relu: expect_tensor(
            call="relu on a random (3, 4) tensor",
            actual=relu(grid),
            expected=grid.clamp(min=0),
        ),
    )

    predictions = torch.tensor([2.0, 3.0, 5.0])
    targets = torch.tensor([1.0, 3.0, 7.0])
    batch_predictions = random_tensor((2, 3))
    batch_targets = random_tensor(
        shape=(2, 3),
        seed=1,
    )

    checker.test(
        function="squared_error",
        description="gives [1, 0, 4] for predictions [2, 3, 5] and targets [1, 3, 7]",
        run=lambda squared_error: expect_tensor(
            call=f"squared_error({value_text(predictions)}, {value_text(targets)})",
            actual=squared_error(predictions, targets),
            expected=torch.tensor([1.0, 0.0, 4.0]),
        ),
    )
    checker.test(
        function="squared_error",
        description="works on a (2, 3) batch",
        run=lambda squared_error: expect_tensor(
            call="squared_error on random (2, 3) predictions and targets",
            actual=squared_error(batch_predictions, batch_targets),
            expected=(batch_predictions - batch_targets) ** 2,
        ),
    )

    values = random_tensor((2, 4))
    padded_ids = (
        (torch.tensor([[4, 7, 0, 0], [3, 0, 0, 0]]), 0),
        (torch.tensor([[4, 7, 9, 9], [3, 9, 9, 9]]), 9),
    )

    for ids, pad_id in padded_ids:
        checker.test(
            function="mask_padding",
            description=f"zeroes the values where the ids are {pad_id}",
            run=lambda mask_padding, ids=ids, pad_id=pad_id: expect_tensor(
                call=f"mask_padding(values, {value_text(ids)}, {pad_id})",
                actual=mask_padding(values, ids, pad_id),
                expected=torch.where(ids == pad_id, 0.0, values),
            ),
        )


TENSORS3 = Exercise(
    name="tensors3",
    chapter=TENSORS,
    title="Math on whole tensors",
    lesson=TENSORS3_LESSON,
    source=TENSORS3_SOURCE,
    solution=TENSORS3_SOLUTION,
    hints={
        "relu": "x > 0 marks the entries to keep. torch.where can take each entry from x where that is true and 0.0 "
        "where it is false.",
        "squared_error": "Subtract the targets from the predictions, then raise the result to the power 2 with **.",
        "mask_padding": "ids == pad_id marks the padding positions. torch.where can put 0.0 there and keep values "
        "everywhere else.",
    },
    introduces=("elementwise",),
    check=check_tensors3,
)

TENSORS4_LESSON = prose(
    """
    Elementwise math also works between tensors of different shapes. PyTorch
    lines the shapes up from the right. Two sizes fit when they are equal or
    when one of them is 1, and a missing size counts as 1. A size of 1 is then
    reused to match the other. This is **broadcasting**.

    ```python
    m = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    m + torch.tensor([10.0, 20.0, 30.0])
    # tensor([[11., 22., 33.], [14., 25., 36.]])
    ```

    ```text
    (2, 3) with (3,)      fit, the (3,) is reused for every row
    (2, 3) with (2, 1)    fit, the (2, 1) is reused for every column
    (2, 3) with (2,)      an error, since 3 meets 2
    ```

    To give each row its own number, the numbers need shape `(rows, 1)`.
    `.unsqueeze(1)` adds a dimension of size 1 at position 1, so a tensor of
    shape `(2,)` becomes `(2, 1)`.

    ```python
    r = torch.tensor([1.0, 10.0])
    r.unsqueeze(1)        # tensor([[ 1.], [10.]])
    m / r.unsqueeze(1)    # tensor([[1.0000, 2.0000, 3.0000], [0.4000, 0.5000, 0.6000]])
    ```

    A square tensor hides a trap. With `s` of shape `(3, 3)` and `t` of shape
    `(3,)`, `s / t` runs without an error. The `(3,)` lines up with the
    columns, so each column gets divided when each row was meant to. The
    numbers are wrong and nothing says so.

    A batch of shape `(examples, features)` broadcasts with one number per
    feature, shape `(features,)`. Models add a **bias** this way, one learned
    number per column, added to every example. Data gets prepared this way
    too. Features come in very different ranges, such as pixels from 0 to 255
    next to values from 0 to 1. Models learn faster when every column is
    rescaled to run from 0 to 1.
    """,
)

TENSORS4_SOURCE = source_file(
    '''
    # tensors4.py

    import torch
    from torch import Tensor


    # TODO: Add bias to every row of batch. batch has shape (examples, features)
    #   and bias has shape (features,).
    def add_bias(
        batch: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    # TODO: Rescale every column of batch to run from 0 to 1. lows and highs have
    #   shape (features,) and hold each column's smallest and largest value.
    def rescale(
        batch: Tensor,
        lows: Tensor,
        highs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the average token value of each sentence. totals has shape
    #   (sentences, size) and holds each sentence's token values added up.
    #   lengths has shape (sentences,) and holds how many tokens each sentence has.
    def average_per_sentence(
        totals: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   add_bias on a (3, 2) batch, and on a (2, 2) batch where a wrong shape still runs
    #   rescale on 3 examples of 2 features, with pixels from 0 to 255 in the first
    #   average_per_sentence on 2 sentences, and on 3 where a wrong shape still runs
    ''',
)

TENSORS4_SOLUTION = source_file(
    '''
    # tensors4.py, a solution for comparison

    import torch
    from torch import Tensor


    def add_bias(
        batch: Tensor,
        bias: Tensor,
    ) -> Tensor:
        return batch + bias


    def rescale(
        batch: Tensor,
        lows: Tensor,
        highs: Tensor,
    ) -> Tensor:
        return (batch - lows) / (highs - lows)


    # lengths.unsqueeze(1) has shape (sentences, 1), so each row gets its own length.
    def average_per_sentence(
        totals: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        return totals / lengths.unsqueeze(1)
    ''',
)


def check_tensors4(checker: Checker) -> None:
    batches = (
        (random_tensor((3, 2)), torch.tensor([10.0, -1.0])),
        (random_tensor((2, 2)), torch.tensor([10.0, -1.0])),
    )

    for batch, bias in batches:
        checker.test(
            function="add_bias",
            description=f"adds the bias to every row of a {shape_text(tuple(batch.shape))} batch",
            run=lambda add_bias, batch=batch, bias=bias: expect_tensor(
                call=f"add_bias on a {shape_text(tuple(batch.shape))} batch with bias {value_text(bias)}",
                actual=add_bias(batch, bias),
                expected=batch + bias,
            ),
        )

    pixels_and_values = torch.tensor(
        [
            [0.0, 10.0],
            [255.0, 20.0],
            [51.0, 15.0],
        ],
    )
    lows = torch.tensor([0.0, 10.0])
    highs = torch.tensor([255.0, 20.0])

    checker.test(
        function="rescale",
        description="puts every column of 3 examples between 0 and 1",
        run=lambda rescale: expect_tensor(
            call=f"rescale({value_text(pixels_and_values)}, {value_text(lows)}, {value_text(highs)})",
            actual=rescale(pixels_and_values, lows, highs),
            expected=torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.2, 0.5]]),
        ),
    )

    sentences = (
        (torch.tensor([[2.0, 4.0, 6.0], [8.0, 4.0, 12.0]]), torch.tensor([2, 4])),
        (random_tensor((3, 3)), torch.tensor([1, 2, 5])),
    )

    for totals, lengths in sentences:
        checker.test(
            function="average_per_sentence",
            description=f"divides each of {totals.shape[0]} rows by its own length",
            run=lambda average_per_sentence, totals=totals, lengths=lengths: expect_tensor(
                call=f"average_per_sentence on {shape_text(tuple(totals.shape))} totals with lengths {value_text(lengths)}",
                actual=average_per_sentence(totals, lengths),
                expected=totals / lengths.unsqueeze(1),
            ),
        )


TENSORS4 = Exercise(
    name="tensors4",
    chapter=TENSORS,
    title="Broadcasting",
    lesson=TENSORS4_LESSON,
    source=TENSORS4_SOURCE,
    solution=TENSORS4_SOLUTION,
    hints={
        "add_bias": "The shapes (examples, features) and (features,) already fit, since they line up from the "
        "right. A plain + works.",
        "rescale": "Subtract each column's low first, so every column starts at 0. Then divide by each column's "
        "range, its high minus its low.",
        "average_per_sentence": "Divide each row of totals by its own length. For that, lengths needs shape "
        "(sentences, 1), which .unsqueeze(1) gives.",
    },
    introduces=("broadcasting", "bias"),
    check=check_tensors4,
)

TENSORS5_LESSON = prose(
    """
    `.sum()` adds up entries. With no arguments it adds all of them. With
    `dim` it adds along one dimension, and that dimension disappears from the
    shape.

    ```python
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    x.sum()         # tensor(21.)
    x.sum(dim=0)    # tensor([5., 7., 9.]), one total per column
    x.sum(dim=1)    # tensor([ 6., 15.]), one total per row
    ```

    `.mean()` takes `dim` the same way, so `x.mean(dim=0)` is
    `tensor([2.5000, 3.5000, 4.5000])`. It only works on floats. `.max()` gives
    the largest entry and `.argmax()` gives its position, and both take `dim`
    too.

    ```python
    x.max()            # tensor(6.)
    x.argmax(dim=1)    # tensor([2, 2]), the column of the largest entry in each row
    ```

    Operations like these, which combine many entries into fewer, are called
    **reductions**. `keepdim=True` keeps the reduced dimension with size 1, so
    `x.sum(dim=1, keepdim=True)` has shape `(2, 1)`. That shape broadcasts
    against the rows of `x`, as the last exercise showed.

    A reduction over the examples of a batch, along dimension 0, describes the
    data, such as the average of each feature. A reduction along each row
    gives one answer per example.

    A model that sorts examples into classes, such as which digit an image
    shows, gives each class a score. A batch of scores has shape `(examples,
    classes)`, and the class with the highest score is the model's answer. Its
    accuracy is the share of examples it gets right.

    Dividing each row by its own total makes every row add up to 1. Models turn
    positive scores into shares of 1 this way, one share per class.
    """,
)

TENSORS5_SOURCE = source_file(
    '''
    # tensors5.py

    import torch
    from torch import Tensor


    # TODO: Return the average of each feature over the examples. batch has shape
    #   (examples, features), and the result has shape (features,).
    def feature_means(batch: Tensor) -> Tensor:
        ...


    # TODO: Return the share of examples whose highest scoring class is their
    #   label, as a float. scores has shape (examples, classes) and labels has
    #   shape (examples,).
    def accuracy(
        scores: Tensor,
        labels: Tensor,
    ) -> float:
        ...


    # TODO: Divide each row of scores by its own total, so every row adds up to 1.
    def to_shares(scores: Tensor) -> Tensor:
        ...


    # What the check tests
    #   feature_means on a (4, 3) batch, and on a (3, 3) batch where a wrong dimension still runs
    #   accuracy on 4 examples with 3 right, and on 3 square scores where a wrong dimension still runs
    #   to_shares on 2 rows of 3, and on a (3, 3) table where a wrong shape still runs
    ''',
)

TENSORS5_SOLUTION = source_file(
    '''
    # tensors5.py, a solution for comparison

    import torch
    from torch import Tensor


    def feature_means(batch: Tensor) -> Tensor:
        return batch.mean(dim=0)


    # The comparison gives True and False, and the mean of those as floats is the share that is True.
    def accuracy(
        scores: Tensor,
        labels: Tensor,
    ) -> float:
        return (scores.argmax(dim=1) == labels).float().mean().item()


    # keepdim=True gives the totals shape (examples, 1), so each row gets divided by its own total.
    def to_shares(scores: Tensor) -> Tensor:
        return scores / scores.sum(dim=1, keepdim=True)
    ''',
)


def check_tensors5(checker: Checker) -> None:
    for batch in (random_tensor((4, 3)), random_tensor((3, 3))):
        checker.test(
            function="feature_means",
            description=f"averages each column of a {shape_text(tuple(batch.shape))} batch",
            run=lambda feature_means, batch=batch: expect_tensor(
                call=f"feature_means on a random {shape_text(tuple(batch.shape))} batch",
                actual=feature_means(batch),
                expected=batch.mean(dim=0),
            ),
        )

    four_examples = torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [0.9, 0.05, 0.05],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ],
    )
    # Along dimension 1 every answer is right. Along dimension 0 two of the three are.
    square_scores = torch.tensor(
        [
            [0.5, 0.2, 0.3],
            [0.1, 0.6, 0.3],
            [0.9, 0.05, 0.05],
        ],
    )
    score_cases = (
        (four_examples, torch.tensor([1, 0, 2, 2]), 0.75),
        (square_scores, torch.tensor([0, 1, 0]), 1.0),
    )

    for scores, labels, share in score_cases:
        checker.test(
            function="accuracy",
            description=f"gives {share} for {scores.shape[0]} examples",
            run=lambda accuracy, scores=scores, labels=labels, share=share: expect_value(
                call=f"accuracy(scores, {value_text(labels)}) for scores {value_text(scores)}",
                actual=accuracy(scores, labels),
                expected=share,
            ),
        )

    positive_scores = (
        torch.tensor([[1.0, 1.0, 2.0], [3.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 1.0, 2.0], [0.0, 3.0, 1.0], [5.0, 0.0, 5.0]]),
    )

    for scores in positive_scores:
        checker.test(
            function="to_shares",
            description=f"makes each of {scores.shape[0]} rows add up to 1",
            run=lambda to_shares, scores=scores: expect_tensor(
                call=f"to_shares({value_text(scores)})",
                actual=to_shares(scores),
                expected=scores / scores.sum(dim=1, keepdim=True),
            ),
        )


TENSORS5 = Exercise(
    name="tensors5",
    chapter=TENSORS,
    title="Reductions",
    lesson=TENSORS5_LESSON,
    source=TENSORS5_SOURCE,
    solution=TENSORS5_SOLUTION,
    hints={
        "feature_means": "The examples run along dimension 0. A mean along that dimension leaves one value per "
        "feature.",
        "accuracy": "argmax along dimension 1 gives each example's answer. Compare the answers with labels, turn "
        "the True and False into floats, and take their mean as a plain number.",
        "to_shares": "Each row's total is a sum along dimension 1. Keep that dimension so the totals line up with "
        "the rows, then divide.",
    },
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
        most of the work in deep learning is matrix multiplication. This
        chapter ends with the most common layer in real models.

        ## Further reading

        - [Dive into Deep Learning, linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
        - [3Blue1Brown, the essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
        """,
    ),
)

MATMUL1_LESSON = prose(
    """
    A **weighted sum** multiplies each value by its own **weight** and adds the
    results. With tensors that is an elementwise multiply and a sum. The **dot
    product**, written `@` between two flat tensors, does both in one step.

    ```python
    v = torch.tensor([2.0, 3.0])
    w = torch.tensor([10.0, 100.0])
    (v * w).sum()    # tensor(320.)
    v @ w            # tensor(320.)
    ```

    A tensor with two dimensions is a **matrix**. Put a matrix on the left of
    `@` and a flat tensor on the right. `@` then takes the dot product of every
    row with the flat tensor, so it gives one result per row.

    With a matrix on each side, `@` takes the dot product of every row on the
    left with every column on the right. This is **matrix multiplication**.
    `(n, k) @ (k, m)` gives `(n, m)`, and the two inner sizes must match.

    ```python
    a = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    b = torch.tensor([[1.0, 0.0, 2.0],
                      [0.0, 10.0, 1.0]])
    a @ b                           # tensor([[ 1., 20.,  4.], [ 3., 40., 10.]])
    a @ torch.tensor([1.0, 1.0])    # tensor([3., 7.])
    ```

    `.T` swaps the two dimensions of a matrix, so its rows become columns and
    `b.T` has shape `(3, 2)`. Multiplying by `b.T` takes the dot product of
    every row on the left with every row of `b`.

    A **layer** is a step of a model that holds weights. The most common one
    computes several weighted sums of each example, called its outputs, and
    adds a bias to each. PyTorch stores its weights as a matrix of shape
    `(outputs, inputs)`, one row of weights per output, and its bias with shape
    `(outputs,)`. A batch of shape `(examples, inputs)` comes out with shape
    `(examples, outputs)`. Section 3 builds it as `nn.Linear`.

    Scoring every row of one matrix against every row of another is common
    too. Search engines score a query against every document this way, and
    models that read text compare every token with every other.
    """,
)

MATMUL1_SOURCE = source_file(
    '''
    # matmul1.py

    import torch
    from torch import Tensor


    # TODO: Return one score per example, the weighted sum of its features.
    #   batch has shape (examples, features) and weights has shape (features,).
    def scores(
        batch: Tensor,
        weights: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the dot product of every row of a with every row of b. a has
    #   shape (n, size) and b has shape (m, size), and result[i, j] is a[i] @ b[j].
    def pairwise_scores(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        ...


    # TODO: Run a layer the way PyTorch does. batch has shape (examples, inputs),
    #   weight has shape (outputs, inputs) and bias has shape (outputs,). The
    #   result has shape (examples, outputs).
    def linear(
        batch: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   scores on a (3, 2) batch, and on a (2, 2) batch where a wrong order still runs
    #   pairwise_scores on (2, 3) and (4, 3) rows, and on two (3, 3) matrices where a wrong order still runs
    #   linear matches PyTorch's own torch.nn.functional.linear, from 3 inputs to 2 outputs and to 3
    ''',
)

MATMUL1_SOLUTION = source_file(
    '''
    # matmul1.py, a solution for comparison

    import torch
    from torch import Tensor


    def scores(
        batch: Tensor,
        weights: Tensor,
    ) -> Tensor:
        return batch @ weights


    # b.T has shape (size, m), so (n, size) @ (size, m) gives (n, m).
    def pairwise_scores(
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return a @ b.T


    # Each output is the dot product of an example with one row of weight.
    def linear(
        batch: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        return batch @ weight.T + bias
    ''',
)


def check_matmul1(checker: Checker) -> None:
    batches = (
        (random_tensor((3, 2)), torch.tensor([10.0, -1.0])),
        (random_tensor((2, 2)), torch.tensor([10.0, -1.0])),
    )

    for batch, weights in batches:
        checker.test(
            function="scores",
            description=f"gives one weighted sum per example of a {shape_text(tuple(batch.shape))} batch",
            run=lambda scores, batch=batch, weights=weights: expect_tensor(
                call=f"scores on a random {shape_text(tuple(batch.shape))} batch with weights {value_text(weights)}",
                actual=scores(batch, weights),
                expected=(batch * weights).sum(dim=1),
            ),
        )

    row_pairs = (
        (random_tensor((2, 3)), random_tensor(shape=(4, 3), seed=1)),
        (random_tensor((3, 3)), random_tensor(shape=(3, 3), seed=1)),
    )

    for a, b in row_pairs:
        checker.test(
            function="pairwise_scores",
            description=f"scores {a.shape[0]} rows against {b.shape[0]} rows",
            run=lambda pairwise_scores, a=a, b=b: expect_tensor(
                call=f"pairwise_scores on random {shape_text(tuple(a.shape))} and {shape_text(tuple(b.shape))} rows",
                actual=pairwise_scores(a, b),
                expected=(a.unsqueeze(1) * b.unsqueeze(0)).sum(dim=2),
            ),
        )

    batch = random_tensor((4, 3))

    for outputs in (2, 3):
        weight = random_tensor(
            shape=(outputs, 3),
            seed=1,
        )
        bias = random_tensor(
            shape=(outputs,),
            seed=2,
        )
        checker.test(
            function="linear",
            description=f"matches PyTorch's own layer from 3 inputs to {outputs} outputs",
            run=lambda linear, weight=weight, bias=bias: expect_tensor(
                call=f"linear on a random (4, 3) batch with weight of shape {shape_text(tuple(weight.shape))}",
                actual=linear(batch, weight, bias),
                expected=torch.nn.functional.linear(batch, weight, bias),
            ),
        )


MATMUL1 = Exercise(
    name="matmul1",
    chapter=MATMUL,
    title="Matrix multiplication",
    lesson=MATMUL1_LESSON,
    source=MATMUL1_SOURCE,
    solution=MATMUL1_SOLUTION,
    hints={
        "scores": "Each score is the dot product of one row with weights. @ between a matrix and a flat tensor "
        "does that for every row.",
        "pairwise_scores": "@ pairs the rows on the left with the columns on the right. Turn the rows of b into "
        "columns with .T first.",
        "linear": "Each output is the dot product of an example with one row of weight. Multiply batch by weight "
        "with its dimensions swapped, then add the bias.",
    },
    introduces=("weight", "weighted sum", "dot product", "matrix", "matrix multiplication", "layer"),
    check=check_matmul1,
)


# ---------------------------------------------------------------------------
# quizzes
# ---------------------------------------------------------------------------

QUIZZES = Chapter(
    folder="quizzes",
    title="Quizzes",
    readme=source_file(
        """
        # Quizzes

        A quiz comes after a group of chapters and uses their tools together,
        with nothing new. quiz1 follows the tensor and matrix multiplication
        chapters.
        """,
    ),
)

QUIZ1_LESSON = prose(
    """
    This quiz uses the tools from tensors1 to matmul1 together, and nothing
    new.

    A model that sorts sentences into classes, such as whether a review is
    positive or negative, needs one row of numbers per sentence. It reads the
    table row of every token, sets the rows at padding positions to 0, and
    averages the rows of the real tokens. Each sentence's average then goes
    through a layer that gives one score per class. The class with the highest
    score is the answer.

    Every step uses a tool from an earlier exercise.

    | Step | Taught in |
    |---|---|
    | read a table row for every token id | tensors2 |
    | mark the padding with a comparison | tensors3 |
    | add a dimension of size 1 so the marks line up with the rows | tensors4 |
    | add up along a dimension and divide by a count | tensors4 and tensors5 |
    | run a layer | matmul1 |
    | pick the highest score in each row | tensors5 |
    """,
)

QUIZ1_SOURCE = source_file(
    '''
    # quiz1.py
    #
    # This quiz uses the tools from tensors1 to matmul1 together, and nothing new.

    import torch
    from torch import Tensor


    # TODO: Turn each sentence into the average of the table rows of its tokens,
    #   leaving the padding out. ids has shape (sentences, length) and table has
    #   shape (vocabulary, size). The result has shape (sentences, size).
    def sentence_averages(
        ids: Tensor,
        table: Tensor,
        pad_id: int,
    ) -> Tensor:
        ...


    # TODO: Return the class of each sentence. Run its average through a layer
    #   with weight of shape (classes, size) and bias of shape (classes,), then
    #   pick the class with the highest score. The result has shape (sentences,).
    def predict(
        ids: Tensor,
        table: Tensor,
        pad_id: int,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    # What the check tests
    #   sentence_averages on 2 sentences with padding id 0, worked out by hand
    #   sentence_averages on 4 sentences with padding id 9 and a random table
    #   predict on those 4 sentences with a layer of 3 classes
    ''',
)

QUIZ1_SOLUTION = source_file(
    '''
    # quiz1.py, a solution for comparison

    import torch
    from torch import Tensor


    def sentence_averages(
        ids: Tensor,
        table: Tensor,
        pad_id: int,
    ) -> Tensor:
        rows = table[ids]

        # Shape (sentences, length, 1), so each mark lines up with a whole row.
        real = (ids != pad_id).unsqueeze(2)
        totals = (rows * real).sum(dim=1)
        lengths = real.sum(dim=1)

        return totals / lengths


    def predict(
        ids: Tensor,
        table: Tensor,
        pad_id: int,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        averages = sentence_averages(ids, table, pad_id)
        class_scores = averages @ weight.T + bias

        return class_scores.argmax(dim=1)
    ''',
)


def reference_sentence_averages(
    ids: Tensor,
    table: Tensor,
    pad_id: int,
) -> Tensor:
    """The averages worked out one sentence at a time, apart from how the exercise does it."""
    averages: list[Tensor] = []

    for sentence in ids.tolist():
        real_rows = [table[token] for token in sentence if token != pad_id]
        averages.append(torch.stack(real_rows).mean(dim=0))

    return torch.stack(averages)


def check_quiz1(checker: Checker) -> None:
    # Row 0 is not zero, so leaving the padding in changes the averages.
    small_table = torch.tensor(
        [
            [9.0, 9.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    )
    small_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])

    checker.test(
        function="sentence_averages",
        description="averages the real tokens of 2 sentences with padding id 0",
        run=lambda sentence_averages: expect_tensor(
            call=f"sentence_averages({value_text(small_ids)}, table, 0) for table {value_text(small_table)}",
            actual=sentence_averages(small_ids, small_table, 0),
            expected=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        ),
    )

    table = random_tensor(
        shape=(10, 4),
        seed=3,
    )
    ids = torch.tensor(
        [
            [1, 2, 3, 9, 9, 9],
            [4, 9, 9, 9, 9, 9],
            [5, 6, 7, 8, 0, 9],
            [2, 2, 9, 9, 9, 9],
        ],
    )
    averages = reference_sentence_averages(
        ids=ids,
        table=table,
        pad_id=9,
    )

    checker.test(
        function="sentence_averages",
        description="averages the real tokens of 4 sentences with padding id 9",
        run=lambda sentence_averages: expect_tensor(
            call=f"sentence_averages({value_text(ids)}, table, 9) for a random (10, 4) table",
            actual=sentence_averages(ids, table, 9),
            expected=averages,
        ),
    )

    weight = random_tensor(
        shape=(3, 4),
        seed=4,
    )
    bias = random_tensor(
        shape=(3,),
        seed=5,
    )

    checker.test(
        function="predict",
        description="picks the highest scoring of 3 classes for each of 4 sentences",
        run=lambda predict: expect_tensor(
            call="predict on those 4 sentences with a random layer of 3 classes",
            actual=predict(ids, table, 9, weight, bias),
            expected=torch.nn.functional.linear(averages, weight, bias).argmax(dim=1),
        ),
    )


QUIZ1 = Exercise(
    name="quiz1",
    chapter=QUIZZES,
    title="Sorting sentences into classes",
    lesson=QUIZ1_LESSON,
    source=QUIZ1_SOURCE,
    solution=QUIZ1_SOLUTION,
    hints={
        "sentence_averages": "table[ids] gives one row per token, with shape (sentences, length, size). ids != "
        "pad_id marks the real tokens, and .unsqueeze(2) lines the marks up with the rows. Zero the padding rows, "
        "add up along the tokens, and divide by how many real tokens each sentence has.",
        "predict": "Run the averages through a layer the way linear does in matmul1, then take the argmax of each "
        "row.",
    },
    introduces=(),
    check=check_quiz1,
)


# ---------------------------------------------------------------------------
# The course, in order
# ---------------------------------------------------------------------------

CHAPTERS: list[Chapter] = [
    TENSORS,
    MATMUL,
    QUIZZES,
]

EXERCISES: list[Exercise] = [
    TENSORS1,
    TENSORS2,
    TENSORS3,
    TENSORS4,
    TENSORS5,
    MATMUL1,
    QUIZ1,
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
    "token",
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
    # Hints are plain text, so code in them like images.shape is taken out before the prose checks.
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
