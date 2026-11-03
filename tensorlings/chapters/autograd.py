"""05_autograd, a small autograd engine built step by step, then PyTorch's own."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from .chain_rule import autograd_partials
from .tensors import fastest_seconds, pinned_threads

AUTOGRAD = Chapter(
    folder="05_autograd",
    title="Autograd",
    readme=source_file(
        """
        # Autograd

        Autograd records every step of a calculation as it runs, then walks the
        steps backward to find every gradient. This chapter builds a small
        autograd engine for single numbers, one piece per exercise, in the
        style of Karpathy's micrograd. It ends with PyTorch's own, which works
        the same way on whole tensors.

        ## Further reading

        - [Karpathy, micrograd](https://github.com/karpathy/micrograd)
        - [PyTorch, a gentle introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
        """,
    ),
)


# ---------------------------------------------------------------------------
# Checks on the learner's Value class
# ---------------------------------------------------------------------------


def expect_made_from(
    call: str,
    made: object,
    value_class: type,
    data: float,
    children: tuple[object, ...],
    op: str,
    either_order: bool = False,
) -> None:
    """A new Value with the given data, children and op. With either_order, the two children may come either way
    round, since a + or a * gives the same Value whichever input comes first."""
    reject_missing_return(
        call=call,
        actual=made,
    )

    if not isinstance(made, value_class):
        raise Mismatch(f"{call} is {value_text(made)}. It should be a new Value.")

    if not math.isclose(made.data, data, abs_tol=1e-9):
        raise Mismatch(f"({call}).data is {made.data!r}, but it should be {data!r}.")

    same_count = len(made.children) == len(children)
    in_order = same_count and all(got is wanted for got, wanted in zip(made.children, children))
    swapped = either_order and same_count and all(got is wanted for got, wanted in zip(made.children, reversed(children)))

    if not (in_order or swapped):
        order = "" if either_order else ", in the order they were used"
        raise Mismatch(f"({call}).children should hold the Values it was made from{order}.")

    if made.op != op:
        raise Mismatch(f"({call}).op is {made.op!r}, but it should be {op!r}.")


def expect_grads(
    call: str,
    values: dict[str, object],
    expected: dict[str, float],
) -> None:
    for name, wanted in expected.items():
        got = values[name].grad

        if not math.isclose(got, wanted, abs_tol=1e-6):
            raise Mismatch(f"After {call}, {name}.grad is {got!r}, but it should be {wanted!r}.")


def check_mul_makes_value(value_class: type) -> None:
    for left, right in ((2.0, -3.0), (0.5, 4.0)):
        a = value_class(left)
        b = value_class(right)
        expect_made_from(
            call=f"Value({left}) * Value({right})",
            made=a * b,
            value_class=value_class,
            data=left * right,
            children=(a, b),
            op="*",
        )


def check_build(
    found: Callable[..., object],
    value_class: type,
) -> None:
    probe = value_class(2.0) * value_class(3.0)

    if probe is None or probe is Ellipsis:
        raise Mismatch("build multiplies two Values, so it needs Value.__mul__. Write that first.")

    for numbers in ((2.0, 3.0, 4.0), (-1.0, 5.0, 0.5)):
        a, b, c = (value_class(number) for number in numbers)
        call = "build({}, {}, {})".format(*(f"Value({number})" for number in numbers))
        made = found(a, b, c)
        reject_missing_return(
            call=call,
            actual=made,
        )

        if not isinstance(made, value_class):
            raise Mismatch(f"{call} is {value_text(made)}. It should be the Value of a * b + c.")

        if len(made.children) != 2:
            raise Mismatch(
                f"{call} is a Value made from {len(made.children)} Values, but a * b + c is made from two, the "
                "Value of a * b and c. Build it with * and + on the Values passed in.",
            )

        # c + a * b and b * a + c build the same Value, so the product can sit on either side of the +.
        product = made.children[1] if made.children[0] is c else made.children[0]
        expect_made_from(
            call=call,
            made=made,
            value_class=value_class,
            data=numbers[0] * numbers[1] + numbers[2],
            children=(product, c),
            op="+",
            either_order=True,
        )
        expect_made_from(
            call=f"the product inside {call}",
            made=product,
            value_class=value_class,
            data=numbers[0] * numbers[1],
            children=(a, b),
            op="*",
            either_order=True,
        )


# ---------------------------------------------------------------------------
# autograd1
# ---------------------------------------------------------------------------

AUTOGRAD1_LESSON = prose(
    """
    Backprop by hand works, but writing the backward lines for every
    calculation gets old fast. Instead, let every number remember how it was
    made. A program can then walk the graph backward by itself. This is
    **autograd**, and the next exercises build a small one.

    Start with a class that holds a number and the values it came from.

    ```python
    class Value:
        def __init__(self, data, children=(), op=""):
            self.data = data
            self.grad = 0.0
            self.children = children
            self.op = op

        def __add__(self, other):
            return Value(self.data + other.data, (self, other), "+")
    ```

    Python calls `__add__` when it meets `+` between two `Value`s, so `a + b`
    runs `a.__add__(b)`. In the same way `a * b` runs `a.__mul__(b)`.

    ```python
    a = Value(2.0)
    b = Value(-3.0)
    c = a + b
    c.data                  # -1.0
    c.children == (a, b)    # True, c holds a and b themselves
    c.op                    # '+'
    ```

    `c` knows it is the sum of `a` and `b`. The values a value was made from
    are its children, and it is their parent. Every value keeps a link to its
    children, so the whole computation graph hangs off the last value.

    `grad` starts at 0. It holds the value's gradient once backpropagation
    fills it in. The next exercises build that part.
    """,
)

AUTOGRAD1_SOURCE = source_file(
    '''
    # autograd1.py


    class Value:
        """A number that remembers the Values it was made from."""

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

        # a + b makes a new Value that remembers a, b and the "+".
        def __add__(self, other: "Value") -> "Value":
            return Value(self.data + other.data, (self, other), "+")

        # TODO: Make a * b work the same way, remembering both Values and "*".
        def __mul__(self, other: "Value") -> "Value":
            ...


    # TODO: Return a * b + c, built from the Values passed in, so the result
    #   remembers how it was made.
    def build(
        a: Value,
        b: Value,
        c: Value,
    ) -> Value:
        ...
    ''',
)

AUTOGRAD1_SOLUTION = source_file(
    '''
    # autograd1.py, a solution for comparison


    class Value:
        """A number that remembers the Values it was made from."""

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

        def __add__(self, other: "Value") -> "Value":
            return Value(self.data + other.data, (self, other), "+")

        def __mul__(self, other: "Value") -> "Value":
            return Value(self.data * other.data, (self, other), "*")


    # Python runs a * b first, then adds c to that new Value.
    def build(
        a: Value,
        b: Value,
        c: Value,
    ) -> Value:
        return a * b + c
    ''',
)


def check_autograd1(checker: Checker) -> None:
    checker.test_method(
        class_name="Value",
        method="__mul__",
        description="makes a Value that remembers both Values and '*'",
        run=check_mul_makes_value,
    )
    checker.test(
        function="build",
        description="builds a * b + c from the Values passed in",
        run=lambda found: check_build(
            found=found,
            value_class=checker.module.Value,
        ),
    )


AUTOGRAD1 = Exercise(
    name="autograd1",
    chapter=AUTOGRAD,
    title="A number that remembers",
    lesson=AUTOGRAD1_LESSON,
    source=AUTOGRAD1_SOURCE,
    solution=AUTOGRAD1_SOLUTION,
    hints={
        "Value.__mul__": "Follow __add__, with * in place of + for the data and '*' as the op.",
        "build": "Use * and + on the Values as you would on numbers. Python calls __mul__ and __add__ for you.",
    },
    introduces=("autograd",),
    check=check_autograd1,
)


# ---------------------------------------------------------------------------
# autograd2
# ---------------------------------------------------------------------------

AUTOGRAD2_LESSON = prose(
    """
    chain3 walked a graph backward with two rules. An addition passes its
    gradient to each input unchanged. A multiplication gives each input the
    gradient times the other input.

    Each `Value` can carry its own rule as a small function, set up when the
    value is made. The function reads the result's `grad` and adds to the
    inputs' `grad`.

    ```python
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def backward_step():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_step = backward_step
        return out
    ```

    `backward_step` is defined inside `__add__`, so it can still use `self`,
    `other` and `out` after `__add__` returns. A value that no operation made
    keeps a step that does nothing.

    To run a step by hand, set the result's gradient and call its step.

    ```python
    a, b = Value(2.0), Value(-3.0)
    c = a + b
    c.grad = 1.0
    c.backward_step()
    a.grad, b.grad      # (1.0, 1.0)
    ```

    A step adds with `+=` and never sets with `=`. In `a + a` the step hands
    `a` a part through each use, and the second part must add to the first,
    as chain2 showed. With `=` the second would wipe out the first.

    The order of the steps matters too. A value can only pass its gradient on
    once every value made from it has handed it its part.
    """,
)

AUTOGRAD2_SOURCE = source_file(
    '''
    # autograd2.py


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

        # An addition passes its gradient to each input unchanged.
        def __add__(self, other: "Value") -> "Value":
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        # TODO: Finish the two lines of backward_step, so a * b hands each input
        #   the result's gradient times the other input's data.
        def __mul__(self, other: "Value") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += ...
                other.grad += ...

            out.backward_step = backward_step

            return out


    # TODO: Finish the walk backward. L's step has run, so q has its gradient.
    #   Call the backward steps of q and p, in the order that gives each one
    #   all of its gradient before it passes it on.
    def walk_backward(
        a: Value,
        b: Value,
        c: Value,
        r: Value,
    ) -> tuple[float, float, float, float]:
        p = a * b
        q = p + c
        L = q * r

        L.grad = 1.0
        L.backward_step()
        ...
        ...

        return a.grad, b.grad, c.grad, r.grad
    ''',
)

AUTOGRAD2_SOLUTION = source_file(
    '''
    # autograd2.py, a solution for comparison


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

        def __add__(self, other: "Value") -> "Value":
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        def __mul__(self, other: "Value") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out


    # q is made from p, so q's step must run first to hand p its gradient.
    def walk_backward(
        a: Value,
        b: Value,
        c: Value,
        r: Value,
    ) -> tuple[float, float, float, float]:
        p = a * b
        q = p + c
        L = q * r

        L.grad = 1.0
        L.backward_step()
        q.backward_step()
        p.backward_step()

        return a.grad, b.grad, c.grad, r.grad
    ''',
)


def check_mul_step(value_class: type) -> None:
    for left, right, upstream in ((2.0, -3.0, 1.0), (0.5, 4.0, 2.0)):
        a = value_class(left)
        b = value_class(right)
        c = a * b
        c.grad = upstream
        c.backward_step()
        expect_grads(
            call=f"c = Value({left}) * Value({right}), c.grad = {upstream} and c's backward step",
            values={"a": a, "b": b},
            expected={"a": right * upstream, "b": left * upstream},
        )

    # The same Value on both sides catches a step that sets with = instead of adding with +=.
    a = value_class(3.0)
    c = a * a
    c.grad = 1.0
    c.backward_step()
    expect_grads(
        call="c = a * a with a = Value(3.0), c.grad = 1.0 and c's backward step",
        values={"a": a},
        expected={"a": 6.0},
    )


def check_walk_backward(
    found: Callable[..., object],
    value_class: type,
) -> None:
    for numbers in ((2.0, -3.0, 10.0, -2.0), (1.5, 2.0, -4.0, 0.5)):
        call = "walk_backward({}, {}, {}, {})".format(*(f"Value({number})" for number in numbers))
        expect_parts("a.grad", "b.grad", "c.grad", "r.grad")(
            call=call,
            actual=found(*(value_class(number) for number in numbers)),
            expected=autograd_partials(lambda a, b, c, r: (a * b + c) * r, *numbers),
        )


def check_autograd2(checker: Checker) -> None:
    checker.test_method(
        class_name="Value",
        method="__mul__",
        description="hands each input the gradient times the other input",
        run=check_mul_step,
    )
    checker.test(
        function="walk_backward",
        description="runs the backward steps in an order that gives every gradient",
        run=lambda found: check_walk_backward(
            found=found,
            value_class=checker.module.Value,
        ),
    )


AUTOGRAD2 = Exercise(
    name="autograd2",
    chapter=AUTOGRAD,
    title="Each operation's backward step",
    lesson=AUTOGRAD2_LESSON,
    source=AUTOGRAD2_SOURCE,
    solution=AUTOGRAD2_SOLUTION,
    hints={
        "Value.__mul__": "self gets other.data times out.grad, and other gets self.data times out.grad. Keep the +=, "
        "so a * a gets both parts.",
        "walk_backward": "p is a child of q, so p only has its whole gradient after q's step has run.",
    },
    introduces=(),
    check=check_autograd2,
)


# ---------------------------------------------------------------------------
# autograd3
# ---------------------------------------------------------------------------

AUTOGRAD3_LESSON = prose(
    """
    autograd2 ran each step by hand, in an order you picked, so that every
    value had its whole gradient before passing it on. A program has to find
    that order by itself.

    A value is ready to pass its gradient on once every parent, every value
    made from it, has run its step. So list the values with each one after its children,
    then run the steps from the end of the list back to the start. A list
    with every value after its children is a **topological order**.

    Take this graph.

    ```python
    p = a * b
    q = p + c
    L = q * r
    ```

    One topological order is `a, b, p, c, q, r, L`. Read backward, it runs
    `L`, then `r`, then `q`, before `p`, so `p` has its whole gradient when its
    turn comes. Values with no children, like `a`, run a step that does
    nothing.

    A recursive walk finds the order. It visits a value's children first, and
    adds the value to the list only after them. A set of values already seen
    stops the walk from adding a value twice when it is used twice.

    `backward` then sets the last value's gradient to 1, the way chain3 began,
    and runs every step in reverse topological order.
    """,
)

AUTOGRAD3_SOURCE = source_file(
    '''
    # autograd3.py


    # TODO: Finish the line in the loop, so every child of a value is visited,
    #   and so added to order, before the value itself.
    def topological_order(root: "Value") -> list["Value"]:
        order = []
        seen = set()

        def visit(value: "Value") -> None:
            if value in seen:
                return

            seen.add(value)

            for child in value.children:
                ...

            order.append(value)

        visit(root)

        return order


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

        def __add__(self, other: "Value") -> "Value":
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        def __mul__(self, other: "Value") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        # TODO: This runs the steps in the wrong order, so values pass their
        #   gradient on before they have all of it. Read the lesson, then fix
        #   the loop. It needs topological_order, above, finished first.
        def backward(self) -> None:
            order = topological_order(self)
            self.grad = 1.0

            for value in order:
                value.backward_step()
    ''',
)

AUTOGRAD3_SOLUTION = source_file(
    '''
    # autograd3.py, a solution for comparison


    def topological_order(root: "Value") -> list["Value"]:
        order = []
        seen = set()

        def visit(value: "Value") -> None:
            if value in seen:
                return

            seen.add(value)

            for child in value.children:
                visit(child)

            order.append(value)

        visit(root)

        return order


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

        def __add__(self, other: "Value") -> "Value":
            out = Value(self.data + other.data, (self, other), "+")

            def backward_step() -> None:
                self.grad += out.grad
                other.grad += out.grad

            out.backward_step = backward_step

            return out

        def __mul__(self, other: "Value") -> "Value":
            out = Value(self.data * other.data, (self, other), "*")

            def backward_step() -> None:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

            out.backward_step = backward_step

            return out

        # Children come before their parents in order, so reversed runs every parent first.
        def backward(self) -> None:
            order = topological_order(self)
            self.grad = 1.0

            for value in reversed(order):
                value.backward_step()
    ''',
)

# Graphs of + and * alone, which work the same on the learner's Values and on PyTorch tensors.
# The last ones use a value twice.
GRAPHS: tuple[tuple[str, Callable[..., object], tuple[str, ...], tuple[float, ...]], ...] = (
    ("(a * b + c) * f", lambda a, b, c, f: (a * b + c) * f, ("a", "b", "c", "f"), (2.0, -3.0, 10.0, -2.0)),
    ("(a + b) * (b * c)", lambda a, b, c: (a + b) * (b * c), ("a", "b", "c"), (1.5, -2.0, 3.0)),
    ("a * a + a * b", lambda a, b: a * a + a * b, ("a", "b"), (3.0, -1.0)),
    ("(a * b + a) * (a * b + b)", lambda a, b: (a * b + a) * (a * b + b), ("a", "b"), (0.5, 2.0)),
)


def graph_values(root: object) -> list[object]:
    """Every Value under root, each once, found without the learner's code."""
    found: list[object] = []
    stack = [root]

    while stack:
        value = stack.pop()

        if any(value is seen for seen in found):
            continue

        found.append(value)
        stack.extend(value.children)

    return found


def check_topological_order(
    found: Callable[..., object],
    value_class: type,
) -> None:
    for text, graph, _, numbers in GRAPHS:
        root = graph(*(value_class(number) for number in numbers))
        call = f"topological_order on {text}"
        order = found(root)
        reject_missing_return(
            call=call,
            actual=order,
        )
        expected = graph_values(root)

        if len(order) != len(expected) or any(not any(value is listed for listed in order) for value in expected):
            raise Mismatch(f"{call} should list each of the {len(expected)} Values in the graph once, but it lists {len(order)}.")

        position = {id(value): index for index, value in enumerate(order)}

        for value in expected:
            for child in value.children:
                if position[id(child)] > position[id(value)]:
                    raise Mismatch(f"In {call}, a Value comes before one of its children. Every child has to come first.")


def check_backward(value_class: type) -> None:
    for text, graph, names, numbers in GRAPHS:
        values = [value_class(number) for number in numbers]
        graph(*values).backward()
        expected = autograd_partials(graph, *numbers)
        expect_grads(
            call=f"backward on {text} with " + ", ".join(f"{name} = {number}" for name, number in zip(names, numbers)),
            values=dict(zip(names, values)),
            expected=dict(zip(names, expected)),
        )


def check_autograd3(checker: Checker) -> None:
    checker.test(
        function="topological_order",
        description="lists every Value after its children, each once",
        run=lambda found: check_topological_order(
            found=found,
            value_class=checker.module.Value,
        ),
    )
    checker.test_method(
        class_name="Value",
        method="backward",
        description="fills in every gradient of the graph",
        run=check_backward,
    )


AUTOGRAD3 = Exercise(
    name="autograd3",
    chapter=AUTOGRAD,
    title="Backward through the whole graph",
    lesson=AUTOGRAD3_LESSON,
    source=AUTOGRAD3_SOURCE,
    solution=AUTOGRAD3_SOLUTION,
    hints={
        "topological_order": "visit calls itself on each child. When it returns, that child and everything under it "
        "are in order.",
        "Value.backward": "order lists every child before its parents. Loop over it from the end with reversed(order), "
        "so every parent runs its step before its children do.",
    },
    introduces=("topological order",),
    check=check_autograd3,
)


# ---------------------------------------------------------------------------
# autograd4
# ---------------------------------------------------------------------------

AUTOGRAD4_LESSON = prose(
    """
    PyTorch has the same machinery built in. Give a tensor
    `requires_grad=True`, and every operation on it records how its result was
    made. `.backward()` on the final value fills `.grad` on each tensor you
    made with `requires_grad=True`. The results of operations, like `L` below,
    get no `.grad` of their own.

    ```python
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    L = (a * b + 10.0) * -2.0
    L.backward()
    a.grad, b.grad    # (tensor(6.), tensor(-4.))
    ```

    The numbers match chain3 and your own `Value`. A tensor made by an
    operation keeps the step that made it in `.grad_fn`, much as your `Value`
    keeps `backward_step`. `.item()` turns a tensor that holds one number into
    a plain Python number.

    `.backward()` works on whole tensors too, as long as the last value is a
    single number. The gradient has the shape of its tensor, one partial per
    entry. `.requires_grad_()` switches recording on for a tensor you already
    have.

    ```python
    v = torch.tensor([1.0, 2.0, 3.0])
    v.requires_grad_()
    (v ** 2).sum().backward()
    v.grad            # tensor([2., 4., 6.]), which is 2 * v
    ```

    This is far faster and more exact than nudging. Take
    `f(v) = ((X @ v) ** 2).mean()`, with `X` a random float32 matrix of shape
    `(64, 10000)`. derivatives3's `numeric_gradient` calls `f` twice per entry,
    20,000 calls, and took 0.25 to 0.31 seconds in three runs. One
    `.backward()` took 0.03 to 0.04 milliseconds, over 7,000 times less. The
    nudged gradient was also off by 1.0 to 1.3 percent of its largest entry,
    and autograd's by under 5 in 10 million.

    PyTorch's gradients add up across calls, as your `+=` does. A second
    `.backward()` adds onto the `.grad` already there. That is a trap when two
    separate calculations each call `.backward()`, as the last TODO shows.
    Clear the gradient in between with `v.grad = None`.
    """,
)

AUTOGRAD4_SOURCE = source_file(
    '''
    # autograd4.py

    from collections.abc import Callable

    import torch
    from torch import Tensor

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    L = (a * b + 10.0) * -2.0
    L.backward()
    grads = (a.grad, b.grad)                      # (tensor(6.), tensor(-4.))


    # TODO: Finish the two middle lines, so this returns the derivative of f at
    #   x from PyTorch's autograd, with no nudging.
    def autograd_slope(
        f: Callable[[Tensor], Tensor],
        x: float,
    ) -> float:
        t = torch.tensor(x, requires_grad=True)
        ...
        ...

        return t.grad.item()


    # TODO: Finish the middle line, so this returns the gradient of f at v, a
    #   tensor with the shape of v. f gives a single number.
    def autograd_gradient(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> Tensor:
        v.requires_grad_()
        ...

        return v.grad


    # TODO: This should return the derivative of the second calculation alone,
    #   which is 2 * x for x ** 2, but it returns more. Fix it.
    def second_gradient(x: float) -> float:
        t = torch.tensor(x, requires_grad=True)

        first = t ** 3
        first.backward()

        second = t ** 2
        second.backward()

        return t.grad.item()
    ''',
)

AUTOGRAD4_SOLUTION = source_file(
    '''
    # autograd4.py, a solution for comparison

    from collections.abc import Callable

    import torch
    from torch import Tensor

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    L = (a * b + 10.0) * -2.0
    L.backward()
    grads = (a.grad, b.grad)


    def autograd_slope(
        f: Callable[[Tensor], Tensor],
        x: float,
    ) -> float:
        t = torch.tensor(x, requires_grad=True)
        y = f(t)
        y.backward()

        return t.grad.item()


    def autograd_gradient(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> Tensor:
        v.requires_grad_()
        f(v).backward()

        return v.grad


    # The first backward leaves 3 * x ** 2 in t.grad, and the second adds onto it.
    def second_gradient(x: float) -> float:
        t = torch.tensor(x, requires_grad=True)

        first = t ** 3
        first.backward()
        t.grad = None

        second = t ** 2
        second.backward()

        return t.grad.item()
    ''',
)


def torch_slope(
    f: Callable[[Tensor], Tensor],
    x: float,
) -> float:
    t = torch.tensor(x, requires_grad=True)
    f(t).backward()

    return t.grad.item()


def torch_gradient(
    f: Callable[[Tensor], Tensor],
    v: Tensor,
) -> Tensor:
    v = v.clone().requires_grad_()
    f(v).backward()

    return v.grad


def check_autograd4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="autograd_slope",
        description="gives the derivative from autograd",
        inputs=(
            (Described(lambda t: t ** 3 - 2 * t, "x ** 3 - 2 * x"), 1.5),
            (Described(lambda t: torch.exp(t) * t, "exp(x) * x"), 0.5),
            (Described(lambda t: torch.log(t ** 2 + 1), "log(x ** 2 + 1)"), -2.0),
        ),
        reference=torch_slope,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="autograd_gradient",
        description="gives the gradient from autograd, shaped like v",
        inputs=(
            (Described(lambda v: (v ** 2).sum(), "(v ** 2).sum()"), torch.tensor([1.0, -2.0, 3.0])),
            (Described(lambda v: v[0] * v[1] + v[1] ** 2, "v[0] * v[1] + v[1] ** 2"), torch.tensor([2.0, 3.0])),
            (Described(lambda v: (v * v.sum()).mean(), "(v * v.sum()).mean()"), torch.tensor([[1.0, 2.0], [0.5, -1.0]])),
        ),
        reference=torch_gradient,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="second_gradient",
        description="gives the derivative of x ** 2 alone",
        inputs=(
            (3.0,),
            (-1.5,),
        ),
        reference=lambda x: 2 * x,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


AUTOGRAD4 = Exercise(
    name="autograd4",
    chapter=AUTOGRAD,
    title="PyTorch's autograd",
    lesson=AUTOGRAD4_LESSON,
    source=AUTOGRAD4_SOURCE,
    solution=AUTOGRAD4_SOLUTION,
    hints={
        "autograd_slope": "Call f on t to get the result, then call .backward() on the result.",
        "autograd_gradient": "Call f on v, then call .backward() on what it gives.",
        "second_gradient": "The first .backward() leaves its gradient in t.grad. Clear t.grad before the second "
        "calculation runs its .backward().",
    },
    introduces=(),
    check=check_autograd4,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def nudged_gradient(
    f: Callable[[Tensor], Tensor],
    v: Tensor,
) -> Tensor:
    """derivatives3's numeric_gradient, a central difference with a nudge of 0.001 for every entry."""
    gradient = torch.zeros_like(v)

    for i in range(len(v)):
        nudge = torch.zeros_like(v)
        nudge[i] = 0.001
        gradient[i] = (f(v + nudge) - f(v - nudge)) / 0.002

    return gradient


def measure_nudging_against_autograd(
    size: int = 10_000,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float]]:
    """autograd4's comparison: the gradient of ((X @ v) ** 2).mean(), with X a random matrix of shape (64, size), in
    float32. Per seed, the calls of f that nudging makes, each way's seconds, and each way's worst error as a fraction
    of the largest partial, measured against autograd in float64."""
    results: list[dict[str, float]] = []

    with pinned_threads():
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            X = torch.randn(64, size, generator=generator, dtype=torch.float64) / size ** 0.5
            v = torch.randn(size, generator=generator, dtype=torch.float64)
            exact = torch_gradient(lambda v: ((X @ v) ** 2).mean(), v)
            largest = exact.abs().max().item()
            X32 = X.float()
            v32 = v.float()
            calls = 0

            def f(v: Tensor) -> Tensor:
                nonlocal calls
                calls += 1

                return ((X32 @ v) ** 2).mean()

            start = time.perf_counter()
            nudged = nudged_gradient(f, v32)
            nudging_seconds = time.perf_counter() - start
            nudging_calls = calls

            autograd_seconds = fastest_seconds(
                run=lambda: torch_gradient(f, v32),
                repeats=5,
            )
            from_autograd = torch_gradient(f, v32)

            results.append(
                {
                    "nudging calls": nudging_calls,
                    "nudging seconds": nudging_seconds,
                    "autograd seconds": autograd_seconds,
                    "nudging error": (nudged.double() - exact).abs().max().item() / largest,
                    "autograd error": (from_autograd.double() - exact).abs().max().item() / largest,
                },
            )

    return results
