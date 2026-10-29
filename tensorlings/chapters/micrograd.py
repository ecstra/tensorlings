"""07_micrograd, Section 1's project, finishing the autograd engine and checking it."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..course import Chapter, Exercise
from ..engine import NOT_STARTED, Checker, Described, Mismatch, reject_missing_return, value_text
from ..text import prose, source_file

MICROGRAD = Chapter(
    folder="07_micrograd",
    title="Project: micrograd",
    readme=source_file(
        """
        # Project: micrograd

        Section 1's project. You finish the autograd engine of the autograd
        chapter, as Karpathy's micrograd does, with plain numbers, subtraction,
        powers, division, exp and log. Then you check its gradients against
        nudging and against PyTorch, and use the check to find a bug.

        ## Further reading

        - [Karpathy, micrograd](https://github.com/karpathy/micrograd)
        - [CS231n, gradient checks](https://cs231n.github.io/neural-networks-3/#gradcheck)
        - [PyTorch, torch.autograd.gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html)
        """,
    ),
)


def torch_result(
    build: Callable[..., object],
    numbers: tuple[float, ...],
) -> tuple[float, list[float]]:
    """The value and the gradients PyTorch gives for build at the numbers, in doubles."""
    tensors = [torch.tensor(number, dtype=torch.float64, requires_grad=True) for number in numbers]
    out = build(*tensors)
    out.backward()

    return out.item(), [tensor.grad.item() for tensor in tensors]


def check_expression(
    value_class: type,
    text: str,
    build: Callable[..., object],
    numbers: tuple[float, ...],
) -> None:
    """Build the expression from the learner's Values, run backward, and compare with PyTorch."""
    names = "abc"[: len(numbers)]
    inputs = ", ".join(f"{name} = {number}" for name, number in zip(names, numbers))
    values = [value_class(number) for number in numbers]
    out = build(*values)
    reject_missing_return(
        call=f"{text} with {inputs}",
        actual=out,
    )

    if not isinstance(out, value_class):
        raise Mismatch(f"{text} with {inputs} is {value_text(out)}. It should be a Value.")

    expected_data, expected_grads = torch_result(
        build=build,
        numbers=numbers,
    )

    if not math.isclose(out.data, expected_data, rel_tol=1e-9, abs_tol=1e-12):
        raise Mismatch(f"({text}).data with {inputs} is {out.data!r}, but it should be {expected_data!r}.")

    out.backward()

    for name, value, wanted in zip(names, values, expected_grads):
        if not math.isclose(value.grad, wanted, rel_tol=1e-6, abs_tol=1e-9):
            raise Mismatch(f"After backward on {text} with {inputs}, {name}.grad is {value.grad!r}, but it should be {wanted!r}.")


def check_expressions(
    checker: Checker,
    method: str,
    description: str,
    expressions: tuple[tuple[str, Callable[..., object], tuple[float, ...]], ...],
) -> None:
    for text, build, numbers in expressions:
        checker.test_method(
            class_name="Value",
            method=method,
            description=description,
            run=lambda value_class, text=text, build=build, numbers=numbers: check_expression(
                value_class=value_class,
                text=text,
                build=build,
                numbers=numbers,
            ),
        )


# ---------------------------------------------------------------------------
# micrograd1
# ---------------------------------------------------------------------------

MICROGRAD1_LESSON = prose(
    """
    This is Section 1's project. You finish your engine, as Karpathy's
    micrograd does. Right now a `Value` only adds and multiplies other
    `Value`s, and real expressions need plain numbers, subtraction, powers and
    division.

    Plain numbers first. For `a + 2` to work, a plain number gets wrapped in a
    `Value` before it is used.

    ```python
    other = other if isinstance(other, Value) else Value(other)
    ```

    `2 + a` is harder. Python first tries `2`'s own add, which doesn't know a
    `Value` and gives up. Python then calls `a.__radd__(2)`. Adding gives the
    same either way round, so `__radd__` can hand straight back to `__add__`.

    ```python
    def __radd__(self, other):
        return self + other
    ```

    Many operations need no backward step of their own, since they can be made
    from operations that have one. `-a` is `a * -1`, and `a - b` is
    `a + (-b)`. The graph then holds an extra step or two, and backpropagation
    passes through them like any other.

    A power by a plain number needs its own step, with the power rule from
    derivatives2 as its local derivative. `a ** n` keeps only `a` as a child,
    since `n` is a plain number. Division is then `a * b ** -1`, made from a
    power and a multiplication.
    """,
)

MICROGRAD1_SOURCE = source_file(
    '''
    # micrograd1.py
    #
    # Section 1's project. Finish your autograd engine, one operation at a time.


    class Value:
        """A number that remembers how it was made, and how to pass a gradient back."""

        def __init__(
            self,
            data: float,
            children: tuple["Value", ...] = (),
            op: str = "",
        ) -> None:
            self.data = data
            self.grad = 0.0
            self.children = children
            self.op = op
            self.backward_step = lambda: None

        # a + 2 works too, since a plain number is wrapped in a Value first.
        def __add__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        # TODO: Make a * 2 work too, by wrapping a plain number the way __add__
        #   does, in a new first line.
        def __mul__(self, other: "Value | float") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        # 2 + a runs a.__radd__(2), and adding gives the same either way round.
        def __radd__(self, other: float) -> "Value":
            return self + other

        # TODO: Make 2 * a work, the way __radd__ makes 2 + a work.
        def __rmul__(self, other: float) -> "Value":
            ...

        # TODO: Make -a work, as a times -1.
        def __neg__(self) -> "Value":
            ...

        # TODO: Make a - b work, as a plus the negative of b.
        def __sub__(self, other: "Value | float") -> "Value":
            ...

        # TODO: Finish backward_step, so a ** n, for a plain number n, hands a
        #   its gradient by the power rule.
        def __pow__(self, n: float) -> "Value":
            out = Value(self.data ** n, (self,), f"**{n}")

            def backward_step() -> None:
                self.grad += ...

            out.backward_step = backward_step

            return out

        # TODO: Make a / b work, as a times b to the power -1.
        def __truediv__(self, other: "Value | float") -> "Value":
            ...

        def backward(self) -> None:
            order = topological_order(self)
            self.grad = 1.0

            for value in reversed(order):
                value.backward_step()


    def topological_order(root: Value) -> list[Value]:
        order = []
        seen = set()

        def visit(value: Value) -> None:
            if value in seen:
                return

            seen.add(value)

            for child in value.children:
                visit(child)

            order.append(value)

        visit(root)

        return order
    ''',
)

MICROGRAD1_SOLUTION = source_file(
    '''
    # micrograd1.py, a solution for comparison


    class Value:
        """A number that remembers how it was made, and how to pass a gradient back."""

        def __init__(
            self,
            data: float,
            children: tuple["Value", ...] = (),
            op: str = "",
        ) -> None:
            self.data = data
            self.grad = 0.0
            self.children = children
            self.op = op
            self.backward_step = lambda: None

        def __add__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        def __mul__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        def __radd__(self, other: float) -> "Value":
            return self + other

        def __rmul__(self, other: float) -> "Value":
            return self * other

        def __neg__(self) -> "Value":
            return self * -1

        def __sub__(self, other: "Value | float") -> "Value":
            return self + (-other)

        # The power rule, n * x ** (n - 1), times the gradient handed down.
        def __pow__(self, n: float) -> "Value":
            out = Value(self.data ** n, (self,), f"**{n}")

            def backward_step() -> None:
                self.grad += n * self.data ** (n - 1) * out.grad

            out.backward_step = backward_step

            return out

        # A plain number to the power -1 is a plain number, which __mul__ wraps.
        def __truediv__(self, other: "Value | float") -> "Value":
            return self * other ** -1

        def backward(self) -> None:
            order = topological_order(self)
            self.grad = 1.0

            for value in reversed(order):
                value.backward_step()


    def topological_order(root: Value) -> list[Value]:
        order = []
        seen = set()

        def visit(value: Value) -> None:
            if value in seen:
                return

            seen.add(value)

            for child in value.children:
                visit(child)

            order.append(value)

        visit(root)

        return order
    ''',
)


def check_micrograd1(checker: Checker) -> None:
    for method, description, expressions in (
        (
            "__mul__",
            "multiplies by a plain number too",
            (
                ("a * 3", lambda a: a * 3, (2.0,)),
                ("a * b * -0.5", lambda a, b: a * b * -0.5, (1.5, 4.0)),
            ),
        ),
        (
            "__rmul__",
            "makes a plain number times a Value work",
            (
                ("3 * a", lambda a: 3 * a, (2.0,)),
                ("2 * a * b", lambda a, b: 2 * a * b, (-1.0, 3.0)),
            ),
        ),
        (
            "__neg__",
            "makes -a work",
            (
                ("-a", lambda a: -a, (2.0,)),
                ("-(a * b)", lambda a, b: -(a * b), (1.5, -2.0)),
            ),
        ),
        (
            "__sub__",
            "makes a - b work",
            (
                ("a - b", lambda a, b: a - b, (5.0, 2.0)),
                ("a - 2", lambda a: a - 2, (5.0,)),
                ("a * b - a", lambda a, b: a * b - a, (3.0, 4.0)),
            ),
        ),
        (
            "__pow__",
            "gives a power by a plain number its gradient by the power rule",
            (
                ("a ** 3", lambda a: a ** 3, (2.0,)),
                ("a ** -1", lambda a: a ** -1, (4.0,)),
                ("(a * b) ** 0.5", lambda a, b: (a * b) ** 0.5, (2.0, 8.0)),
            ),
        ),
        (
            "__truediv__",
            "makes a / b work",
            (
                ("a / b", lambda a, b: a / b, (6.0, 3.0)),
                ("a / 4", lambda a: a / 4, (6.0,)),
                ("(a + b) / (a * b)", lambda a, b: (a + b) / (a * b), (2.0, -0.5)),
            ),
        ),
    ):
        check_expressions(
            checker=checker,
            method=method,
            description=description,
            expressions=expressions,
        )


MICROGRAD1 = Exercise(
    name="micrograd1",
    chapter=MICROGRAD,
    title="Project: more operations",
    lesson=MICROGRAD1_LESSON,
    source=MICROGRAD1_SOURCE,
    solution=MICROGRAD1_SOLUTION,
    hints={
        "Value.__mul__": "Copy the first line of __add__ to the top of __mul__.",
        "Value.__rmul__": "Multiplying gives the same either way round, so hand back to __mul__ the way __radd__ "
        "hands back to __add__.",
        "Value.__neg__": "Return self times -1. __mul__ wraps the -1 in a Value.",
        "Value.__sub__": "Return self plus the negative of other. -other uses __neg__ when other is a Value, and is "
        "a plain number otherwise.",
        "Value.__pow__": "The power rule gives n * x ** (n - 1), with x being self.data. Multiply it by out.grad, "
        "as every step multiplies its local derivative by the gradient handed down.",
        "Value.__truediv__": "Return self times other to the power -1. Both pieces already work.",
    },
    introduces=(),
    check=check_micrograd1,
)


# ---------------------------------------------------------------------------
# micrograd2
# ---------------------------------------------------------------------------

MICROGRAD2_LESSON = prose(
    """
    `exp` and `log` are the last pieces. Each gets its own backward step, with
    its local derivative from the table in derivatives2. `exp` is its own
    derivative, and the result's data already holds it, so the step reuses
    `out.data`.

    ```python
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward_step():
            self.grad += out.data * out.grad

        out.backward_step = backward_step
        return out
    ```

    `log`'s local derivative is `1 / x`, taken at its input `x`, which is
    `self.data` inside the step.

    A method is called on a value, as in `a.exp()`, and chains like any
    other. `log` undoes `exp`, so this gives `a * b` back.

    ```python
    a, b = Value(2.0), Value(3.0)
    c = (a * b).exp().log()
    c.data      # 6.0
    ```

    The log of a sum of exps shows up again in Section 2. For a list of
    numbers it comes out a little above the largest of them, so it works as a
    smooth maximum. For 1 and 3 it is about 3.127.

    With these two your engine has every operation Section 2 builds its
    calculations from.
    """,
)

MICROGRAD2_CLASS = '''

    class Value:
        """A number that remembers how it was made, and how to pass a gradient back."""

        def __init__(
            self,
            data: float,
            children: tuple["Value", ...] = (),
            op: str = "",
        ) -> None:
            self.data = data
            self.grad = 0.0
            self.children = children
            self.op = op
            self.backward_step = lambda: None

        def __add__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        def __mul__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        def __radd__(self, other: float) -> "Value":
            return self + other

        def __rmul__(self, other: float) -> "Value":
            return self * other

        def __neg__(self) -> "Value":
            return self * -1

        def __sub__(self, other: "Value | float") -> "Value":
            return self + (-other)

        def __pow__(self, n: float) -> "Value":
            out = Value(self.data ** n, (self,), f"**{n}")

            def backward_step() -> None:
                self.grad += n * self.data ** (n - 1) * out.grad

            out.backward_step = backward_step

            return out

        def __truediv__(self, other: "Value | float") -> "Value":
            return self * other ** -1
'''

MICROGRAD_BACKWARD = '''
        def backward(self) -> None:
            order = topological_order(self)
            self.grad = 1.0

            for value in reversed(order):
                value.backward_step()


    def topological_order(root: Value) -> list[Value]:
        order = []
        seen = set()

        def visit(value: Value) -> None:
            if value in seen:
                return

            seen.add(value)

            for child in value.children:
                visit(child)

            order.append(value)

        visit(root)

        return order
'''

MICROGRAD2_SOURCE = source_file(
    "\n    # micrograd2.py\n\n    import math\n"
    + MICROGRAD2_CLASS
    + '''
        # exp is its own derivative, and out.data already holds exp(self.data).
        def exp(self) -> "Value":
            out = Value(math.exp(self.data), (self,), "exp")

            def backward_step() -> None:
                self.grad += out.data * out.grad

            out.backward_step = backward_step

            return out

        # TODO: Finish backward_step, so log hands its input the gradient times
        #   the local derivative of log, which is 1 / x at its input x.
        def log(self) -> "Value":
            out = Value(math.log(self.data), (self,), "log")

            def backward_step() -> None:
                self.grad += ...

            out.backward_step = backward_step

            return out
'''
    + MICROGRAD_BACKWARD
    + '''

    # TODO: Return the log of the sum of exp(v) for every Value v in values,
    #   built with the Value methods, so backward works through it. values
    #   holds at least one Value.
    def log_sum_exp(values: list[Value]) -> Value:
        ...
    ''',
)

MICROGRAD2_SOLUTION = source_file(
    "\n    # micrograd2.py, a solution for comparison\n\n    import math\n"
    + MICROGRAD2_CLASS
    + '''
        def exp(self) -> "Value":
            out = Value(math.exp(self.data), (self,), "exp")

            def backward_step() -> None:
                self.grad += out.data * out.grad

            out.backward_step = backward_step

            return out

        def log(self) -> "Value":
            out = Value(math.log(self.data), (self,), "log")

            def backward_step() -> None:
                self.grad += (1 / self.data) * out.grad

            out.backward_step = backward_step

            return out
'''
    + MICROGRAD_BACKWARD
    + '''

    def log_sum_exp(values: list[Value]) -> Value:
        total = values[0].exp()

        for value in values[1:]:
            total = total + value.exp()

        return total.log()
    ''',
)


def check_log_sum_exp(
    found: Callable[..., object],
    value_class: type,
) -> None:
    for numbers in ((1.0, 3.0), (-2.0, 0.5, 1.0), (4.0,)):
        names = ", ".join("abc"[: len(numbers)])
        check_expression(
            value_class=value_class,
            text=f"log_sum_exp([{names}])",
            build=lambda *inputs: found(list(inputs)) if isinstance(inputs[0], value_class) else torch.stack(inputs).logsumexp(dim=0),
            numbers=numbers,
        )


def check_micrograd2(checker: Checker) -> None:
    check_expressions(
        checker=checker,
        method="log",
        description="hands its input the gradient times 1 / x",
        expressions=(
            ("a.log()", lambda a: a.log(), (2.0,)),
            ("(a * b).log()", lambda a, b: (a * b).log(), (0.5, 3.0)),
            ("(a.exp() + 1).log()", lambda a: (a.exp() + 1).log(), (-1.0,)),
        ),
    )
    checker.test(
        function="log_sum_exp",
        description="builds the log of exp(a) + exp(b) from Values",
        run=lambda found: check_log_sum_exp(
            found=found,
            value_class=checker.module.Value,
        ),
    )


MICROGRAD2 = Exercise(
    name="micrograd2",
    chapter=MICROGRAD,
    title="Project: exp and log",
    lesson=MICROGRAD2_LESSON,
    source=MICROGRAD2_SOURCE,
    solution=MICROGRAD2_SOLUTION,
    hints={
        "Value.log": "The local derivative is 1 / self.data. Multiply it by out.grad, as exp's step multiplies its "
        "own by out.grad.",
        "log_sum_exp": "Start a total with the exp of the first Value. Add the exp of each one after it in a loop, "
        "then call .log() on the total.",
    },
    introduces=(),
    check=check_micrograd2,
)


# ---------------------------------------------------------------------------
# micrograd3
# ---------------------------------------------------------------------------

MICROGRAD3_LESSON = prose(
    """
    An engine that runs is not the same as an engine that is right. A backward
    step with a wrong sign runs without a complaint. So check the gradients
    two ways, against nudging from derivatives1, and against PyTorch.

    Nudging only needs the forward pass, which is hard to get wrong. Nudge
    one input both ways while the rest stay put, and take the central
    difference of the output's `data`.

    PyTorch's tensors have the same operators and the same `.exp()` and
    `.log()` methods. So a function written for your `Value` runs on tensors
    unchanged, and PyTorch's autograd gives a third answer.

    ```python
    build = lambda a, b: (a * b + 1).log()
    build(Value(2.0), Value(3.0)).data       # 1.9459101490553132
    t = torch.tensor(2.0, dtype=torch.float64)
    build(t, torch.tensor(3.0, dtype=torch.float64)).item()    # 1.9459101490553132
    ```

    `dtype=torch.float64` keeps about 16 digits, as a Python float does.
    Without it a tensor keeps about 7, and the answers differ by rounding.

    The three agree up to rounding when the engine is right. Nudging has a
    small error of its own, so a gap around `1e-6` from it is normal. A gap
    from PyTorch should be far smaller, and a gap near 1 means a backward step
    is wrong. PyTorch ships the same check as `torch.autograd.gradcheck`.

    `print` works in the exercise files, and what it prints shows in the
    Result tab, which helps while you hunt for a bug.
    """,
)

MICROGRAD3_TOOLS = '''

    # TODO: Return the gradient your engine gives each input, in a list. Wrap each
    #   number in a Value, build the output from them, and run backward.
    def engine_gradients(
        build: Callable[..., Value],
        numbers: list[float],
    ) -> list[float]:
        ...


    # TODO: Finish the line in the loop, so this returns the central difference
    #   of the output for each input, nudged by 1e-6 each way.
    def numeric_gradients(
        build: Callable[..., Value],
        numbers: list[float],
    ) -> list[float]:
        output = lambda values: build(*[Value(number) for number in values]).data
        gradients = []

        for i in range(len(numbers)):
            up = numbers.copy()
            down = numbers.copy()
            up[i] += 1e-6
            down[i] -= 1e-6
            gradients.append(...)

        return gradients


    # TODO: Return PyTorch's gradient for each input, in a list. Use tensors
    #   with dtype=torch.float64.
    def torch_gradients(
        build: Callable[..., torch.Tensor],
        numbers: list[float],
    ) -> list[float]:
        ...
'''

MICROGRAD3_SOURCE = source_file(
    "\n    # micrograd3.py\n    #\n    # TODO: One backward step in this Value class is wrong. Write the three\n"
    "    #   functions below the class, use them to find the wrong step, then fix it.\n\n"
    "    import math\n    from collections.abc import Callable\n\n    import torch\n"
    + MICROGRAD2_CLASS
    + '''
        def exp(self) -> "Value":
            out = Value(math.exp(self.data), (self,), "exp")

            def backward_step() -> None:
                self.grad += out.data * out.grad

            out.backward_step = backward_step

            return out

        def log(self) -> "Value":
            out = Value(math.log(self.data), (self,), "log")

            def backward_step() -> None:
                self.grad += (1 / out.data) * out.grad

            out.backward_step = backward_step

            return out
'''
    + MICROGRAD_BACKWARD
    + MICROGRAD3_TOOLS,
)

MICROGRAD3_SOLUTION = source_file(
    "\n    # micrograd3.py, a solution for comparison\n\n"
    "    import math\n    from collections.abc import Callable\n\n    import torch\n"
    + MICROGRAD2_CLASS
    + '''
        def exp(self) -> "Value":
            out = Value(math.exp(self.data), (self,), "exp")

            def backward_step() -> None:
                self.grad += out.data * out.grad

            out.backward_step = backward_step

            return out

        # The step divided by out.data, the log's result, where the local derivative needs its input.
        def log(self) -> "Value":
            out = Value(math.log(self.data), (self,), "log")

            def backward_step() -> None:
                self.grad += (1 / self.data) * out.grad

            out.backward_step = backward_step

            return out
'''
    + MICROGRAD_BACKWARD
    + '''

    def engine_gradients(
        build: Callable[..., Value],
        numbers: list[float],
    ) -> list[float]:
        values = [Value(number) for number in numbers]
        build(*values).backward()

        return [value.grad for value in values]


    def numeric_gradients(
        build: Callable[..., Value],
        numbers: list[float],
    ) -> list[float]:
        output = lambda values: build(*[Value(number) for number in values]).data
        gradients = []

        for i in range(len(numbers)):
            up = numbers.copy()
            down = numbers.copy()
            up[i] += 1e-6
            down[i] -= 1e-6
            gradients.append((output(up) - output(down)) / 2e-6)

        return gradients


    def torch_gradients(
        build: Callable[..., torch.Tensor],
        numbers: list[float],
    ) -> list[float]:
        tensors = [torch.tensor(number, dtype=torch.float64, requires_grad=True) for number in numbers]
        build(*tensors).backward()

        return [tensor.grad.item() for tensor in tensors]
    ''',
)

CHECKED_BUILDS: tuple[tuple[Described, list[float]], ...] = (
    (Described(lambda a, b: (a * b + 1).log(), "(a * b + 1).log()"), [2.0, 3.0]),
    (Described(lambda a, b, c: ((a - b) ** 2 / c).exp(), "((a - b) ** 2 / c).exp()"), [1.5, 0.5, 4.0]),
    (Described(lambda a, b: a.exp() * b - a / (b + 2), "a.exp() * b - a / (b + 2)"), [0.3, -1.2]),
)


def expect_numbers(tolerance: float) -> Callable[..., None]:
    """An expect function for a list of numbers, each allowed to differ by the tolerance."""

    def expect(
        call: str,
        actual: object,
        expected: list[float],
    ) -> None:
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise Mismatch(f"{call} should be a list of {len(expected)} numbers, one per input, but it is {value_text(actual)}.")

        for position, (got, wanted) in enumerate(zip(actual, expected)):
            if got is Ellipsis:
                raise Mismatch(f"{call} still holds a ... where a TODO goes, so it is {NOT_STARTED}.")

            got = got.item() if isinstance(got, torch.Tensor) else got

            if not isinstance(got, (int, float)) or abs(got - wanted) > tolerance:
                raise Mismatch(f"Entry {position} of {call} is {value_text(got)}, but it should be {wanted!r}.")

    return expect


def own_engine_gradients(
    value_class: type,
    build: Callable[..., object],
    numbers: list[float],
) -> list[float]:
    values = [value_class(number) for number in numbers]
    build(*values).backward()

    return [value.grad for value in values]


def own_numeric_gradients(
    value_class: type,
    build: Callable[..., object],
    numbers: list[float],
) -> list[float]:
    gradients = []

    for i in range(len(numbers)):
        up = list(numbers)
        down = list(numbers)
        up[i] += 1e-6
        down[i] -= 1e-6
        gradients.append((build(*map(value_class, up)).data - build(*map(value_class, down)).data) / 2e-6)

    return gradients


def check_micrograd3(checker: Checker) -> None:
    # The references read the learner's Value only when a test runs, so a file without one fails its tests
    # instead of the whole check.
    for function, description, reference, tolerance in (
        (
            "engine_gradients",
            "gives the gradients your engine finds",
            lambda build, numbers: own_engine_gradients(checker.module.Value, build, numbers),
            1e-9,
        ),
        (
            "numeric_gradients",
            "gives the central difference for each input",
            lambda build, numbers: own_numeric_gradients(checker.module.Value, build, numbers),
            1e-6,
        ),
        (
            "torch_gradients",
            "gives PyTorch's gradients in doubles",
            lambda build, numbers: torch_result(build, tuple(numbers))[1],
            1e-9,
        ),
    ):
        for build, numbers in CHECKED_BUILDS:
            call = f"{function}({build.text}, {numbers})"
            checker.test(
                function=function,
                description=description,
                run=lambda found, build=build, numbers=numbers, call=call, reference=reference, tolerance=tolerance: expect_numbers(tolerance)(
                    call=call,
                    actual=found(build.value, list(numbers)),
                    expected=reference(build.value, numbers),
                ),
            )

    for build, numbers in CHECKED_BUILDS:
        checker.test(
            function="Value",
            description="gives the same gradients as PyTorch for every operation",
            run=lambda found, build=build, numbers=numbers: check_expression(
                value_class=found,
                text=build.text,
                build=build.value,
                numbers=tuple(numbers),
            ),
        )


MICROGRAD3 = Exercise(
    name="micrograd3",
    chapter=MICROGRAD,
    title="Project: check your engine",
    lesson=MICROGRAD3_LESSON,
    source=MICROGRAD3_SOURCE,
    solution=MICROGRAD3_SOLUTION,
    hints={
        "engine_gradients": "Make a list of Values from numbers, call build with them, call .backward() on the "
        "result, and collect each Value's .grad.",
        "numeric_gradients": "output(up) is the output with input i nudged up. Take output(up) minus output(down), "
        "and divide by the whole gap, 2e-6.",
        "torch_gradients": "Do what engine_gradients does, with torch.tensor(number, dtype=torch.float64, "
        "requires_grad=True) in place of Value(number), and .grad.item() for each gradient.",
        "Value": "Compare engine_gradients with torch_gradients on functions that use one operation each, like "
        "lambda a: a.log() or lambda a: a.exp(), and print what they give. The step whose numbers differ is the "
        "wrong one.",
    },
    introduces=(),
    check=check_micrograd3,
)
