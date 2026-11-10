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

        This is the project for Section 1. You finish the autograd engine you
        built in the autograd chapter. First you make it handle plain numbers,
        subtraction, powers and division, the way Karpathy's micrograd does.
        Then you give exp its own backward step, as his lecture does, and give
        log one too. Last you check the engine's gradients against nudging and
        against PyTorch, and use that check to find a bug.

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
    This is the project for Section 1. You finish your autograd engine so it
    handles the same operations as Karpathy's micrograd. Right now a `Value`
    can only be added to or multiplied by another `Value`. Real expressions
    also need plain numbers, subtraction, powers and division.

    Start with plain numbers. To make `a + 2` work, `__add__` first wraps a
    plain number in a `Value`. After that line, both inputs are `Value`s.

    ```python
    other = other if isinstance(other, Value) else Value(other)
    ```

    `2 + a` is harder. Python first calls the `__add__` method of the int 2.
    That method does not know about `Value`, so it returns `NotImplemented`.
    Python then calls `a.__radd__(2)` instead. Addition gives the same result
    in either order, so `__radd__` can reuse `__add__`.

    ```python
    def __radd__(self, other):
        return self + other
    ```

    Many operations need no backward step of their own, because they can be
    built from operations that already have one. `-a` is `a * -1`, and
    `a - b` is `a + (-b)`. The computation graph then holds an extra step or
    two, and backpropagation goes through those steps like any others.

    Raising a `Value` to a plain number, as in `a ** n`, needs its own
    backward step. Its local derivative comes from the power rule in
    derivatives2. The result of `a ** n` has only `a` as a child, because `n`
    is a plain number, not a `Value`. Division can then be built from a power
    and a multiplication, since `a / b` equals `a * (b ** -1)`.
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

        # a + 2 works too, because the first line wraps a plain number in a Value.
        def __add__(self, other: "Value | float") -> "Value":
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        # TODO: Make a * 2 work too. Add a new first line that wraps a plain
        #   number in a Value, the same way the first line of __add__ does.
        #   Example: (Value(2.0) * 3).data is 6.0.
        def __mul__(self, other: "Value | float") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        # 2 + a calls a.__radd__(2). Addition gives the same result in either
        # order, so __radd__ reuses __add__.
        def __radd__(self, other: float) -> "Value":
            return self + other

        # TODO: Make 2 * a work, the same way __radd__ makes 2 + a work. Python
        #   calls a.__rmul__(2) for 2 * a.
        def __rmul__(self, other: float) -> "Value":
            ...

        # TODO: Make -a work. Compute it as a multiplied by -1.
        def __neg__(self) -> "Value":
            ...

        # TODO: Make a - b work. Compute it as a plus the negative of b.
        def __sub__(self, other: "Value | float") -> "Value":
            ...

        # TODO: Finish backward_step for a ** n, where n is a plain number. Add
        #   to self.grad the local derivative from the power rule, taken at
        #   self.data, multiplied by out.grad.
        #   Example: for a = Value(2.0), (a ** 3).backward() sets a.grad to 12.0.
        def __pow__(self, n: float) -> "Value":
            out = Value(self.data ** n, (self,), f"**{n}")

            def backward_step() -> None:
                self.grad += ...

            out.backward_step = backward_step

            return out

        # TODO: Make a / b work. Compute it as a times b to the power -1.
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

        # The local derivative from the power rule, n * (x ** (n - 1)), times out.grad.
        def __pow__(self, n: float) -> "Value":
            out = Value(self.data ** n, (self,), f"**{n}")

            def backward_step() -> None:
                self.grad += n * self.data ** (n - 1) * out.grad

            out.backward_step = backward_step

            return out

        # When other is a plain number, other ** -1 is a plain number too, and
        # __mul__ wraps it in a Value.
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
            "gives a ** n, for a plain number n, its gradient from the power rule",
            (
                ("a ** 3", lambda a: a ** 3, (2.0,)),
                ("a ** -1", lambda a: a ** -1, (4.0,)),
                ("(a * b) ** 0.5", lambda a, b: (a * b) ** 0.5, (2.0, 8.0)),
                # The power is not the last step here, so its gradient handed down is not 1.
                ("(a ** 2) * 3", lambda a: a ** 2 * 3, (1.5,)),
                # a feeds the power and the sum, so a step that sets a.grad with = wipes out the sum's share.
                ("a ** 2 + a", lambda a: a ** 2 + a, (1.5,)),
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
        "Value.__rmul__": "Multiplication gives the same result in either order, so __rmul__ can reuse __mul__, "
        "the way __radd__ reuses __add__.",
        "Value.__neg__": "Return self times -1. __mul__ wraps the -1 in a Value.",
        "Value.__sub__": "Return self plus the negative of other. When other is a Value, -other calls __neg__. When "
        "other is a plain number, -other is a plain number too.",
        "Value.__pow__": "The power rule gives the local derivative n * (x ** (n - 1)), where x is self.data. Multiply "
        "it by out.grad, as every backward step does with its local derivative.",
        "Value.__truediv__": "Return self times other to the power -1. Both of those operations already work.",
    },
    introduces=(),
    check=check_micrograd1,
)


# ---------------------------------------------------------------------------
# micrograd2
# ---------------------------------------------------------------------------

MICROGRAD2_LESSON = prose(
    """
    `exp` and `log` are the last two operations. Each gets its own backward
    step, and each local derivative comes from the table in derivatives2. The
    derivative of `exp(x)` is `exp(x)` itself. `out.data` already holds
    `exp(x)`, so the backward step reuses `out.data`.

    ```python
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward_step():
            self.grad += out.data * out.grad

        out.backward_step = backward_step
        return out
    ```

    The local derivative of `log(x)` is `1 / x`, where `x` is the input to
    `log`. Inside the backward step, that input is `self.data`.

    These are methods, so you call them on a value, as in `a.exp()`. You can
    call another method on the result. `log` undoes `exp`, so this example
    gives back the value of `a * b`.

    ```python
    a, b = Value(2.0), Value(3.0)
    c = (a * b).exp().log()
    c.data      # 6.0
    ```

    Section 2 uses the log of a sum of exps again, the value
    `log(exp(x1) + exp(x2) + ...)` for a list of numbers. The result is a
    little above the largest number in the list. It also changes smoothly as
    the numbers change. For 1 and 3 it is about 3.127.
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
        # The derivative of exp is exp itself, and out.data already holds
        # exp(self.data).
        def exp(self) -> "Value":
            out = Value(math.exp(self.data), (self,), "exp")

            def backward_step() -> None:
                self.grad += out.data * out.grad

            out.backward_step = backward_step

            return out

        # TODO: Finish backward_step for log. Add to self.grad the local
        #   derivative of log at its input, multiplied by out.grad.
        #   Example: for a = Value(2.0), a.log().backward() sets a.grad to 0.5.
        def log(self) -> "Value":
            out = Value(math.log(self.data), (self,), "log")

            def backward_step() -> None:
                self.grad += ...

            out.backward_step = backward_step

            return out
'''
    + MICROGRAD_BACKWARD
    + '''

    # TODO: Finish the line in the loop and the last line, so this returns
    #   log(exp(v1) + exp(v2) + ...) for the Values v1, v2, ... in values.
    #   Build it from Value operations, so backward can pass gradients
    #   through it.
    #   Example: log_sum_exp([Value(1.0), Value(3.0)]).data is about 3.127.
    def log_sum_exp(values: list[Value]) -> Value:
        total = values[0].exp()

        for value in values[1:]:
            total = ...

        return ...
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
        description="adds out.grad times 1 / x to its input's gradient",
        expressions=(
            ("a.log()", lambda a: a.log(), (2.0,)),
            ("(a * b).log()", lambda a, b: (a * b).log(), (0.5, 3.0)),
            ("(a.exp() + 1).log()", lambda a: (a.exp() + 1).log(), (-1.0,)),
            # log is not the last step here, so its gradient handed down is not 1.
            ("(a.log() + b) * b", lambda a, b: (a.log() + b) * b, (2.0, 3.0)),
            # a feeds the log and the sum, so a step that sets a.grad with = wipes out the sum's share.
            ("a.log() + a", lambda a: a.log() + a, (2.0,)),
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
        "Value.log": "The local derivative is 1 / self.data. Multiply it by out.grad, the same way the backward "
        "step of exp multiplies its local derivative by out.grad.",
        "log_sum_exp": "In the loop, add the exp of each value to total. At the end, call .log() on the total.",
    },
    introduces=(),
    check=check_micrograd2,
)


# ---------------------------------------------------------------------------
# micrograd3
# ---------------------------------------------------------------------------

MICROGRAD3_LESSON = prose(
    """
    A backward step with a mistake in it, such as a wrong sign, still runs
    without any error. So you check the engine's gradients in two ways. You
    compare them with estimates from nudging, as in derivatives1, and with the
    gradients PyTorch computes.

    Nudging only uses the forward pass, the computation of the output's
    `data`, which is hard to get wrong. Nudge one input up and down by a small
    step while the other inputs stay fixed. Then take the central difference
    of the output's `data`.

    PyTorch tensors support the same operators and the same `.exp()` and
    `.log()` methods as your `Value`. So a function written for `Value`s also
    runs on tensors without changes. PyTorch's autograd then gives a third set
    of gradients to compare.

    ```python
    build = lambda a, b: (a * b + 1).log()
    build(Value(2.0), Value(3.0)).data       # 1.9459101490553132
    t = torch.tensor(2.0, dtype=torch.float64)
    build(t, torch.tensor(3.0, dtype=torch.float64)).item()    # 1.9459101490553132
    ```

    `dtype=torch.float64` makes a tensor keep about 16 significant digits,
    like a Python float. Without it, a tensor keeps about 7, and its answers
    differ from your engine's by rounding error.

    When the engine is right, the three sets of gradients agree up to
    rounding. Nudging has a small error of its own, around `1e-10` on the
    functions here. The difference from PyTorch's gradients is smaller still.
    A difference near `1e-6` or larger means a backward step is wrong. PyTorch
    has the same check built in, as `torch.autograd.gradcheck`.

    You can use `print` in the exercise files, and its output shows in the
    Result tab. Use it to find the bug.
    """,
)

MICROGRAD3_TOOLS = '''

    # TODO: Finish the middle line, so this returns a list with your engine's
    #   gradient for each input. The middle line builds the output from the
    #   Values and runs backward on it.
    #   Example: engine_gradients(lambda a, b: a * b, [2.0, 3.0]) returns
    #   [3.0, 2.0].
    def engine_gradients(
        build: Callable[..., Value],
        numbers: list[float],
    ) -> list[float]:
        values = [Value(number) for number in numbers]
        ...

        return [value.grad for value in values]


    # TODO: Finish the line in the loop, so this returns a list with the
    #   central difference estimate of the gradient for each input, with a
    #   step of 1e-6 each way.
    #   Example: numeric_gradients(lambda a, b: a * b, [2.0, 3.0]) returns
    #   about [3.0, 2.0].
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


    # TODO: Finish the last two lines, so this returns a list with PyTorch's
    #   gradient for each input, as plain Python floats. It works like
    #   engine_gradients, with tensors in place of Values.
    #   Example: torch_gradients(lambda a, b: a * b, [2.0, 3.0]) returns
    #   [3.0, 2.0].
    def torch_gradients(
        build: Callable[..., torch.Tensor],
        numbers: list[float],
    ) -> list[float]:
        tensors = [torch.tensor(number, dtype=torch.float64, requires_grad=True) for number in numbers]
        ...

        return ...
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

        # The bug: the step divided by out.data, which is log(x). The local
        # derivative 1 / x needs the input x, which is self.data.
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
    (Described(lambda a, b, c: ((a - b) ** 2 / c).exp(), "(((a - b) ** 2) / c).exp()"), [1.5, 0.5, 4.0]),
    (Described(lambda a, b: a.exp() * b - a / (b + 2), "a.exp() * b - a / (b + 2)"), [0.3, -1.2]),
    # a feeds the log and the product, so a log step that sets a.grad with = in place of += fails here.
    (Described(lambda a: a.log() * a, "a.log() * a"), [2.0]),
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
            "gives your engine's gradient for each input",
            lambda build, numbers: own_engine_gradients(checker.module.Value, build, numbers),
            1e-9,
        ),
        # The same central difference agrees to about 1e-10, while a one sided difference or a nudge of 1e-3 is off
        # by 1e-8 to 1e-6 on these builds.
        (
            "numeric_gradients",
            "gives the central difference for each input",
            lambda build, numbers: own_numeric_gradients(checker.module.Value, build, numbers),
            1e-9,
        ),
        (
            "torch_gradients",
            "gives PyTorch's gradients, computed in float64",
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
        "engine_gradients": "Call build with the Values as separate arguments, build(*values), and call "
        ".backward() on the result.",
        "numeric_gradients": "output(up) is the output with input i nudged up by 1e-6, and output(down) is the output "
        "with it nudged down. Subtract output(down) from output(up), then divide by 2e-6, the distance between "
        "the two inputs.",
        "torch_gradients": "Run build on the tensors and call .backward(), as engine_gradients does. Then call "
        ".grad.item() on each tensor, so the list holds plain numbers.",
        "Value": "Compare engine_gradients with torch_gradients on functions that use one operation each, like "
        "lambda a: a.log() or lambda a: a.exp(), and print both results. The operation whose gradients differ has "
        "the wrong backward step.",
    },
    introduces=(),
    check=check_micrograd3,
)
