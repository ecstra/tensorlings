"""Running the learner's file and checking what it produces."""

from __future__ import annotations

import ast
import re
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from .paths import relative

Tensor = torch.Tensor


class Problem(Exception):
    """The file cannot be checked yet. The message is shown to the learner as written."""


class Mismatch(Exception):
    """One test got a different result than it should. The message says how."""


class TookTooLong(Exception):
    """The learner's code ran past the time limit, usually a loop that never ends."""


# Long enough for any correct answer, short enough that a loop that never ends doesn't stall the app.
# An exercise that trains on real data sets a longer limit of its own.
TIME_LIMIT_SECONDS = 5


def run_with_time_limit(
    run: Callable[[], object],
    seconds: float = TIME_LIMIT_SECONDS,
) -> None:
    """Run code with a time limit. Checks run in a thread, where signals can't interrupt them,
    so a trace function raises TookTooLong inside the learner's code once time is up."""
    deadline = time.monotonic() + seconds

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
        r"probability tensor contains either",
        "torch.multinomial needs probabilities, numbers from 0 up that it can draw by. A model's scores can be "
        "negative, so pass them through torch.softmax first.",
    ),
    (
        r"list indices must be integers or slices, not tuple",
        "Square brackets with a comma, like x[:, 0], only work on a tensor. This is a Python list, so turn it into "
        "a tensor with torch.tensor first.",
    ),
    (
        r"name '(\w+)' is not defined",
        "{0} was never set, because the line that sets it crashed or is missing. Fix that line first.",
    ),
    (
        r"object is not subscriptable",
        "Square brackets pick parts of something that holds values, like a tensor or a list. Here they come right "
        "after something that holds nothing to pick, such as a function like torch.tensor.",
    ),
    (
        r"expected sequence of length (\d+) at dim \d+ \(got (\d+)\)",
        "Every row of a tensor needs the same length. One row here has {1} numbers, but the rows before it have {0}.",
    ),
    (
        r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)",
        "These two shapes cannot be broadcast together. Compared from the right, a size {0} meets a size {1}. "
        "Two sizes only fit when they are equal or one of them is 1. If each row should get its own number, "
        "give the numbers shape (rows, 1).",
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
        "For @, the size at the end of the first shape must equal the size at the start of the second. Here the "
        "shapes are ({0}, {1}) and ({2}, {3}), so {1} meets {2}. A flat tensor counts as one row when it is on the "
        "left of @.",
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
    (
        r"'(int|float)' object has no attribute 'data'",
        "A plain number has no .data, only a Value does. Wrap the number in a Value first, the way __add__ does.",
    ),
    (
        r"object of type 'NoneType' has no len\(\)",
        "Something here is None, so it has no length. A function with no return line gives None, so check that "
        "every function this line uses is written.",
    ),
    (
        r"'NoneType' object has no attribute '(\w+)'",
        "Something here is None, so it has no .{0}. A tensor's .grad stays None until .backward() runs on a result "
        "made from it, and a function with no return line gives None.",
    ),
)


def plain_words(error: BaseException) -> str | None:
    for pattern, explanation in PLAIN_ERRORS:
        match = re.search(pattern, str(error))

        if match:
            return explanation.format(*match.groups())

    return None


def ellipsis_line(
    error: BaseException,
    path: Path,
) -> int | None:
    """The line of the learner's file where the error was raised, when that line holds a ... of its own."""
    learner_frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if Path(frame.filename).resolve() == path.resolve()
    ]

    if not learner_frames or "..." not in (learner_frames[-1].line or ""):
        return None

    return learner_frames[-1].lineno


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
        lines.append(f"The original message, {type(error).__name__}: {error}")
    else:
        lines.append(f"{type(error).__name__}: {error}")

    return "\n".join(lines)


@dataclass(frozen=True)
class Crash:
    """A top level statement of the learner's file that raised, and the names it would have set."""

    names: tuple[str, ...]
    text: str


def assigned_names(statement: ast.stmt) -> tuple[str, ...]:
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return ()

    return tuple(node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name))


def load_module(
    path: Path,
    name: str,
    seconds: float = TIME_LIMIT_SECONDS,
) -> tuple[ModuleType, list[Crash]]:
    """Run the file one top level statement at a time. A line that crashes on purpose,
    as in a fix the error task, then hides nothing that comes after it."""
    if not path.exists():
        raise Problem(f"{relative(path)} is missing. Press x to bring back the original.")

    try:
        tree = ast.parse(
            path.read_text(),
            filename=str(path),
        )
    except SyntaxError as error:
        raise Problem(f'Python cannot read line {error.lineno} of {path.name}. Its message is "{error.msg}".') from None

    module = ModuleType(f"tensorlings_{name}_{time.monotonic_ns()}")
    module.__file__ = str(path)
    crashes: list[Crash] = []

    for statement in tree.body:
        code = compile(
            ast.Module(body=[statement], type_ignores=[]),
            filename=str(path),
            mode="exec",
        )

        try:
            run_with_time_limit(
                run=lambda: exec(code, module.__dict__),
                seconds=seconds,
            )
        except TookTooLong:
            raise Problem(f"{path.name} was still running after {seconds:g} seconds while loading. Look for a loop that never ends.") from None
        except Exception as error:
            crashes.append(
                Crash(
                    names=assigned_names(statement),
                    text=describe_crash(
                        error=error,
                        path=path,
                    ),
                ),
            )

    return module, crashes


@dataclass(frozen=True)
class TestResult:
    # The function or the name the test is about.
    function: str
    description: str
    passed: bool
    detail: str = ""
    # What a passing name holds, shown in the result in place of print lines.
    shown: str = ""


class Checker:
    """Runs one exercise's tests against the learner's module and keeps every result."""

    def __init__(
        self,
        module: ModuleType,
        path: Path,
        crashes: list[Crash],
        seconds: float = TIME_LIMIT_SECONDS,
    ) -> None:
        self.module = module
        self.path = path
        self.crashes = crashes
        self.seconds = seconds
        self.checked_names: set[str] = set()
        self.results: list[TestResult] = []

    def fail(
        self,
        function: str,
        description: str,
        detail: str,
    ) -> None:
        self.results.append(
            TestResult(
                function=function,
                description=description,
                passed=False,
                detail=detail,
            ),
        )

    def test(
        self,
        function: str,
        description: str,
        run: Callable[[Callable[..., object]], None],
    ) -> None:
        """Run one test of one function. A missing function fails only its own tests."""
        found = getattr(self.module, function, None)

        if not callable(found):
            self.fail(
                function=function,
                description=description,
                detail=f"{self.path.name} has no function called {function}. Check the name in its def line.",
            )
            return

        self.run_test(
            function=function,
            description=description,
            run=lambda: run(found),
        )

    def test_method(
        self,
        class_name: str,
        method: str,
        description: str,
        run: Callable[[type], None],
    ) -> None:
        """Run one test of one method of a class the file defines. The row is named like Value.__mul__,
        and the test gets the class, since a method is tested through the objects it makes."""
        row = f"{class_name}.{method}"
        found = getattr(self.module, class_name, None)

        if not isinstance(found, type):
            self.fail(
                function=row,
                description=description,
                detail=f"{self.path.name} has no class called {class_name}. Check the name in its class line.",
            )
            return

        if not callable(getattr(found, method, None)):
            self.fail(
                function=row,
                description=description,
                detail=f"{class_name} has no method called {method}. Check the name in its def line.",
            )
            return

        self.run_test(
            function=row,
            description=description,
            run=lambda: run(found),
        )

    def check_name(
        self,
        name: str,
        description: str,
        run: Callable[[object], None],
    ) -> None:
        """Run one test of a name the file sets. A line that crashed fails only the tests of the names it sets."""
        self.checked_names.add(name)
        crash = next((crash for crash in self.crashes if name in crash.names), None)

        if crash is not None and "ellipsis" in crash.text:
            self.fail(
                function=name,
                description=description,
                detail=f"The line that sets {name} still has a ... where a TODO goes, so it is {NOT_STARTED}.",
            )
            return

        if crash is not None:
            self.fail(
                function=name,
                description=description,
                detail=crash.text,
            )
            return

        if not hasattr(self.module, name):
            self.fail(
                function=name,
                description=description,
                detail=f"{self.path.name} never sets {name}. Check the spelling on the line that should.",
            )
            return

        value = getattr(self.module, name)

        if value is Ellipsis:
            self.fail(
                function=name,
                description=description,
                detail=f"{name} is still ..., so it is {NOT_STARTED}.",
            )
            return

        self.run_test(
            function=name,
            description=description,
            run=lambda: run(value),
            shown=value_summary(value),
        )

    def run_test(
        self,
        function: str,
        description: str,
        run: Callable[[], None],
        shown: str = "",
    ) -> None:
        try:
            run_with_time_limit(
                run=run,
                seconds=self.seconds,
            )
        except TookTooLong:
            self.results.append(
                TestResult(
                    function=function,
                    description=description,
                    passed=False,
                    detail=f"{function} was still running after {self.seconds:g} seconds. Look for a loop that never ends.",
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
            # The ... can sit in another function this one calls, so the message names its line when the
            # line that failed holds it. A ... set earlier and used later fails on a line without one.
            if "ellipsis" in str(error):
                line = ellipsis_line(
                    error=error,
                    path=self.path,
                )
                detail = f"{function}, or a function it calls, still has a ... where a TODO goes, so it is {NOT_STARTED}."

                if line is not None:
                    detail = f"{function} runs into a ... where a TODO goes, on line {line} of {self.path.name}, so it is {NOT_STARTED}."
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
                shown=shown,
            ),
        )

    def unchecked_crashes(self) -> list[Crash]:
        """Crashes no test has reported, which fail the file as a whole."""
        return [crash for crash in self.crashes if not set(crash.names) & self.checked_names]


@dataclass(frozen=True)
class Outcome:
    problem: str | None
    results: list[TestResult]
    printed: str

    @property
    def passed(self) -> bool:
        return self.problem is None and bool(self.results) and all(result.passed for result in self.results)


def shape_text(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"

    return "(" + ", ".join(str(size) for size in shape) + ")"


@dataclass(frozen=True)
class Described:
    """An input the result shows by its text, such as a function shown as its formula."""

    value: object
    text: str


def value_text(value: object) -> str:
    """A value on one line, a tensor written the way PyTorch prints it."""
    if isinstance(value, Described):
        return value.text

    # Detached, since a module's output would otherwise print its grad_fn, which means nothing to the learner.
    if isinstance(value, Tensor):
        return re.sub(r"\n\s*", " ", str(value.detach()))

    return repr(value)


def value_summary(value: object) -> str:
    """What a name holds, short enough for one row of the result."""
    if not isinstance(value, Tensor):
        return value_text(value)

    shape = f"shape {shape_text(tuple(value.shape))}"

    if value.numel() > SHOWN_ENTRIES:
        return shape

    return f"{value_text(value)}   {shape}"


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
        got = number_text(actual[index].item())
        wanted = number_text(expected[index].item())

        if got == wanted:
            got = f"{actual[index].item():.10g}"
            wanted = f"{expected[index].item():.10g}"

        lines.append(f"  at {list(index)}, got {got}, expected {wanted}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checks shared by the exercises
# ---------------------------------------------------------------------------


def small_tensor(
    seed: int,
    *shape: int,
) -> Tensor:
    """The same tensor of small whole numbers every run, so a check never changes between saves
    and a failing call stays short enough to read."""
    return torch.randint(-3, 4, shape, generator=torch.Generator().manual_seed(seed)).float()


def reject_missing_return(
    call: str,
    actual: object,
) -> None:
    if actual is None:
        raise Mismatch(f"{call} is None, which usually means it is {NOT_STARTED}.")

    # A partly written function hands back the ... its TODO line still holds.
    if actual is Ellipsis:
        raise Mismatch(f"{call} is still ..., so it is {NOT_STARTED}.")


def reject_values_and_positions(
    call: str,
    actual: object,
) -> None:
    """.max(dim=...) and .min(dim=...) give a pair of values and positions, which is a tuple underneath."""
    if type(actual).__module__ == "torch.return_types":
        raise Mismatch(
            f"{call} is the pair that .max or .min with a dim gives, the values and their positions. "
            "For positions alone use .argmax or .argmin, or take .indices of the pair. For values alone take "
            ".values.",
        )


def expect_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )
    reject_values_and_positions(
        call=call,
        actual=actual,
    )

    if isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} is a Python {type(actual).__name__}. Turn it into a tensor with torch.tensor(...).")

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} is a {type(actual).__name__}. It should be a tensor.")

    small = actual.numel() <= SHOWN_ENTRIES and expected.numel() <= SHOWN_ENTRIES

    if tuple(actual.shape) != tuple(expected.shape):
        message = f"{call} has shape {shape_text(tuple(actual.shape))}, but it should have shape {shape_text(tuple(expected.shape))}."

        if small:
            message += f"\ngot       {value_text(actual)}\nexpected  {value_text(expected)}"

        raise Mismatch(message)

    if torch.allclose(actual.double(), expected.double(), atol=1e-6):
        return

    if small:
        got = value_text(actual)
        wanted = value_text(expected)
        message = f"{call} has the right shape but different numbers.\ngot       {got}\nexpected  {wanted}"

        # Values a small constant apart can print the same at PyTorch's 4 decimals.
        if got == wanted:
            gap = (actual.double() - expected.double()).abs().max().item()
            message += f"\nThey agree to the digits shown, but differ by up to {gap:.3g}."

        raise Mismatch(message)

    raise Mismatch(f"{call} has the right shape but different numbers.\n" + differences_text(actual, expected))


def single_entry(
    call: str,
    actual: object,
    expected: object,
    expected_text: str,
) -> object:
    """The plain value a one entry tensor holds, when the reference gives a tensor too.
    A reference that gives a plain value, as its TODO promised, takes a plain value only."""
    if not isinstance(actual, Tensor):
        return actual

    # Comparing a tensor of several numbers with one number gives a tensor, which can't answer a plain if.
    if actual.numel() != 1:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be the single number {expected_text}.")

    if isinstance(expected, Tensor):
        return actual.item()

    kind = "bool" if isinstance(expected, bool) else "number"

    raise Mismatch(
        f"{call} is {value_text(actual)}, a tensor, but it should be the plain {kind} {expected_text}. "
        f"Use .item() to turn a tensor of one entry into a plain {kind}.",
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
    reject_values_and_positions(
        call=call,
        actual=actual,
    )

    if isinstance(actual, Tensor) and isinstance(expected, list):
        raise Mismatch(
            f"{call} is {value_text(actual)}, a tensor, but it should be the Python list {value_text(expected)}. "
            "Use .tolist() to turn a tensor into a list.",
        )

    actual = single_entry(
        call=call,
        actual=actual,
        expected=expected,
        expected_text=value_text(expected.item() if isinstance(expected, Tensor) and expected.numel() == 1 else expected),
    )

    if isinstance(expected, Tensor):
        expected = expected.item()

    if actual != expected:
        raise Mismatch(
            f"{call} is {value_text(actual)}, but it should be {value_text(expected)}.",
        )


def expect_close(
    call: str,
    actual: object,
    expected: float | Tensor,
    tolerance: float = 1e-6,
) -> None:
    """A single number that may differ from the expected one by rounding, or by an estimate's error."""
    reject_missing_return(
        call=call,
        actual=actual,
    )

    actual = single_entry(
        call=call,
        actual=actual,
        expected=expected,
        expected_text=f"{expected:.10g}",
    )

    if isinstance(expected, Tensor):
        expected = expected.item()

    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be the number {expected:.10g}.")

    if abs(actual - expected) <= tolerance:
        return

    raise Mismatch(f"{call} is {actual:.10g}, but it should be {expected:.10g}, give or take {tolerance:g}.")


def expect_parts(*names: str) -> Callable[..., None]:
    """An expect function for a result made of several parts, like two gradients returned together.
    Each part is a tensor or a number, and a message names the part that is wrong."""

    def expect(
        call: str,
        actual: object,
        expected: tuple[object, ...],
    ) -> None:
        reject_missing_return(
            call=call,
            actual=actual,
        )
        listed = ", ".join(names[:-1]) + f" and {names[-1]}"

        if not isinstance(actual, (tuple, list)) or len(actual) != len(names):
            raise Mismatch(f"{call} should give {len(names)} things, {listed}, but it gives {value_text(actual)}.")

        for name, part, wanted in zip(names, actual, expected):
            if isinstance(wanted, Tensor):
                expect_tensor(
                    call=f"{name} from {call}",
                    actual=part,
                    expected=wanted,
                )
                continue

            expect_close(
                call=f"{name} from {call}",
                actual=part,
                expected=wanted,
            )

    return expect


def expect_shape(
    call: str,
    actual: object,
    shape: tuple[int | None, ...],
) -> None:
    """A tensor of the given shape, where None stands for any size."""
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if isinstance(actual, (list, tuple)):
        raise Mismatch(f"{call} is a Python {type(actual).__name__}. Turn it into a tensor with torch.tensor(...).")

    if not isinstance(actual, Tensor):
        raise Mismatch(f"{call} is a {type(actual).__name__}. It should be a tensor.")

    fits = actual.dim() == len(shape) and all(
        wanted is None or size == wanted for size, wanted in zip(actual.shape, shape)
    )

    if fits:
        return

    wanted_text = "(" + ", ".join("any size" if size is None else str(size) for size in shape) + ")"

    raise Mismatch(f"{call} has shape {shape_text(tuple(actual.shape))}, but it should have shape {wanted_text}.")


def check_on_inputs(
    checker: Checker,
    function: str,
    description: str,
    inputs: tuple[tuple[object, ...], ...],
    reference: Callable[..., object],
    expect: Callable[..., None] = expect_tensor,
) -> None:
    """Call the learner's function on inputs the file never shows, so typing an answer in fails."""
    for arguments in inputs:
        call = f"{function}({', '.join(value_text(argument) for argument in arguments)})"
        values = [argument.value if isinstance(argument, Described) else argument for argument in arguments]
        checker.test(
            function=function,
            description=description,
            run=lambda found, values=values, call=call: expect(
                call=call,
                actual=found(*[fresh_copy(value) for value in values]),
                expected=reference(*[fresh_copy(value) for value in values]),
            ),
        )


def fresh_copy(value: object) -> object:
    """A copy of a tensor or a random generator, so the learner's function and the reference each start from
    the same input, and one that changes its input in place or draws from a generator can't change the other's."""
    if isinstance(value, Tensor):
        return value.clone()

    if isinstance(value, torch.Generator):
        copy = torch.Generator()
        copy.set_state(value.get_state())

        return copy

    return value


def seeded(seed: int) -> Described:
    """A random generator with a fixed seed, shown in a failing call by its seed."""
    return Described(
        value=torch.Generator().manual_seed(seed),
        text=f"a generator seeded with {seed}",
    )


class DeviceRecorder(torch.nn.Module):
    """A stand in model on the meta device, which holds no numbers, that notes each input reaching it from the CPU.
    Checks run on the CPU, where a missing .to(device) changes nothing, so this shows it as main's GPU would."""

    def __init__(
        self,
        names: tuple[str, ...],
        output: Callable[..., Tensor],
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), device="meta"))
        self.names = names
        self.output = output
        self.from_cpu: list[str] = []

    def forward(
        self,
        *inputs: Tensor,
    ) -> Tensor:
        for name, value in zip(self.names, inputs):
            if value.device.type != "meta" and name not in self.from_cpu:
                self.from_cpu.append(name)

        return self.output(*[value.to("meta") for value in inputs]) * self.scale


def expect_on_device(
    function: str,
    run: Callable[[], object],
    recorders: list[DeviceRecorder],
    advice: str,
) -> None:
    """Run the learner's function with its models on the meta device, as main runs them on a GPU. A tensor left on
    the CPU either reaches a recorder or crashes against the model's output. Reading a number off the meta device
    fails too, but only once the tensors have met the model, so that ends the run."""
    crash: Exception | None = None

    try:
        run()
    except (RuntimeError, NotImplementedError) as error:
        message = str(error)

        # Reading a number fails with "Tensor.item() cannot be called on meta tensors" or "Cannot copy out of meta
        # tensor; no data!". A tensor on the wrong device fails with a message that names the devices.
        if "meta tensor" not in message and "device" not in message:
            raise

        if "meta tensor" not in message:
            crash = error

    from_cpu = [name for recorder in recorders for name in recorder.from_cpu]

    if from_cpu:
        raise Mismatch(
            f"{function} handed the model {' and '.join(from_cpu)} on the CPU, but main runs the model on a GPU, "
            f"where that crashes. {advice}",
        )

    if crash is not None:
        raise Mismatch(
            f"{function} crashed with the model on another device than the CPU, as main runs it on a GPU. {advice}"
            f"\n\nThe original message, {crash}",
        ) from None
