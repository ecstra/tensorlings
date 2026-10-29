"""04_chain_rule, derivatives through steps fed into each other, values used twice, and backprop by hand."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..course import Chapter, Exercise
from ..engine import Checker, check_on_inputs, expect_close, expect_parts
from ..text import prose, source_file

CHAIN_RULE = Chapter(
    folder="04_chain_rule",
    title="The chain rule",
    readme=source_file(
        """
        # The chain rule

        A model is a long chain of small steps, each fed into the next. The chain
        rule gives the derivative of the whole chain from each step's own
        derivative. This chapter uses it on one path, on values that take
        several paths, and then by hand on a small calculation, the way
        training does it.

        ## Further reading

        - [Karpathy, the spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
        - [Dive into Deep Learning, automatic differentiation](https://d2l.ai/chapter_preliminaries/autograd.html)
        """,
    ),
)


def autograd_partials(
    f: Callable[..., torch.Tensor],
    *values: float,
) -> tuple[float, ...]:
    """The partial derivatives of f at the values, from PyTorch's own autograd."""
    inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in values]
    f(*inputs).backward()

    return tuple(value.grad.item() for value in inputs)


CHAIN1_LESSON = prose(
    """
    Most functions in deep learning are made by feeding one function into
    another. Take `h(x) = (3 * x + 1) ** 2`. An inner step `u = 3 * x + 1`
    feeds an outer step `u ** 2`.

    Nudge `x` a tiny amount. The inner step moves `u` by 3 times the nudge,
    since its derivative is 3. The outer step moves the output by `2 * u`
    times the move in `u`. So the output moves `2 * u * 3` times the nudge.

    ```python
    x = 2.0
    u = 3 * x + 1      # 7.0
    outer = 2 * u      # 14.0, the derivative of u ** 2 with respect to u
    inner = 3          # the derivative of 3 * x + 1 with respect to x
    outer * inner      # 42.0, the derivative of h at 2
    ```

    This is the **chain rule**. The derivative of steps fed into each other is
    the product of each step's own derivative, each taken at the value that
    step sees. A longer chain multiplies more of them.

    A step's own derivative is its **local derivative**. It only needs that
    step's input. So each step can work out its local derivative on its own,
    and the chain multiplies them, however many steps there are.

    The rules of derivatives2 give the local derivatives. Here is
    `exp(x ** 2)` at 1, where the inner step is `x ** 2` and the outer is
    `exp`.

    ```python
    u = 1.0 ** 2
    math.exp(u) * (2 * 1.0)    # 5.43656365691809
    ```
    """,
)

CHAIN1_SOURCE = source_file(
    '''
    # chain1.py

    import math


    # The derivative of (3 * x + 1) ** 2, by the chain rule.
    def d_square_of_line(x: float) -> float:
        u = 3 * x + 1
        outer = 2 * u
        inner = 3

        return outer * inner


    # TODO: Finish the two middle lines, so this returns the derivative of
    #   (2 * x - 5) ** 3 at x.
    def d_cube_of_line(x: float) -> float:
        u = 2 * x - 5
        outer = ...
        inner = ...

        return outer * inner


    # TODO: Finish the three lines, so this returns the derivative of
    #   log(x ** 2 + 1) at x. The inner step is x ** 2 + 1 and the outer step is
    #   log.
    def d_log_of_square(x: float) -> float:
        u = ...
        outer = ...
        inner = ...

        return outer * inner


    # TODO: Finish the last line, so this returns the derivative of
    #   exp(3 * x) ** 2 at x. It has three steps, 3 * x, then exp, then
    #   squaring, so the chain multiplies three local derivatives. a and b hold
    #   what the first two steps give.
    def d_three_steps(x: float) -> float:
        a = 3 * x
        b = math.exp(a)

        return ...
    ''',
)

CHAIN1_SOLUTION = source_file(
    '''
    # chain1.py, a solution for comparison

    import math


    def d_square_of_line(x: float) -> float:
        u = 3 * x + 1
        outer = 2 * u
        inner = 3

        return outer * inner


    def d_cube_of_line(x: float) -> float:
        u = 2 * x - 5
        outer = 3 * u ** 2
        inner = 2

        return outer * inner


    def d_log_of_square(x: float) -> float:
        u = x ** 2 + 1
        outer = 1 / u
        inner = 2 * x

        return outer * inner


    # The square's local derivative is 2 * b, exp's is exp(a), and 3 * x's is 3.
    def d_three_steps(x: float) -> float:
        a = 3 * x
        b = math.exp(a)

        return 2 * b * math.exp(a) * 3
    ''',
)

CHAIN_POINTS: tuple[tuple[float], ...] = (
    (2.0,),
    (-0.5,),
    (0.25,),
)


def check_chain1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="d_cube_of_line",
        description="gives the derivative of (2 * x - 5) ** 3",
        inputs=CHAIN_POINTS,
        reference=lambda x: 3 * (2 * x - 5) ** 2 * 2,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="d_log_of_square",
        description="gives the derivative of log(x ** 2 + 1)",
        inputs=CHAIN_POINTS,
        reference=lambda x: 2 * x / (x ** 2 + 1),
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="d_three_steps",
        description="gives the derivative of exp(3 * x) ** 2",
        inputs=CHAIN_POINTS,
        reference=lambda x: 6 * math.exp(6 * x),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-6 * max(1.0, abs(expected)),
        ),
    )


CHAIN1 = Exercise(
    name="chain1",
    chapter=CHAIN_RULE,
    title="The chain rule",
    lesson=CHAIN1_LESSON,
    source=CHAIN1_SOURCE,
    solution=CHAIN1_SOLUTION,
    hints={
        "d_cube_of_line": "outer is the derivative of u ** 3 with respect to u. inner is the derivative of "
        "2 * x - 5 with respect to x.",
        "d_log_of_square": "u is the inner step's value, as in d_square_of_line. outer is the derivative of log at "
        "u, from the table in derivatives2, and inner is the derivative of x ** 2 + 1.",
        "d_three_steps": "Multiply the three local derivatives. The square's is twice what it squares, which is b. "
        "exp's is exp of its input, and the first step's is the number x is multiplied by.",
    },
    introduces=("chain rule",),
    check=check_chain1,
)

CHAIN2_LESSON = prose(
    """
    The chain rule follows one path from an input to the output. Often an
    input reaches the output along more than one path. In
    `f(x, y) = x * y + x`, `x` feeds both the product and the sum.

    Nudge `x` and both paths move the output at once. The product moves it by
    `y` per unit of `x`, and the plain `+ x` moves it by 1. The moves add up,
    so the partial with respect to `x` is `y + 1`.

    ```python
    x, y = 2.0, 5.0
    d_x = y + 1    # 6.0, both paths of x added up
    d_y = x        # 2.0, y has one path
    ```

    So when a value feeds several places, its derivative is the sum over its
    paths, with each path's part found by the chain rule. Along one path, hold
    the value's other uses still. So the path through one factor of a product
    gives the other factors.

    `x * x` is the plain case. The path through the first `x` gives the second
    `x`, and the path through the second gives the first. Together they give
    `2 * x`, as the power rule says.

    The same goes for a step whose result is used more than once. With
    `a = x + y` and `f = a * a + a`, the derivative of `f` with respect to `a`
    adds up the three paths through `a`. The chain rule then carries it back to
    `x` and `y`.

    ```python
    a = x + y          # 7.0
    d_a = a + a + 1    # 15.0, two paths through a * a and one through + a
    d_x = d_a * 1      # 15.0, since a moves one for one with x
    ```
    """,
)

CHAIN2_SOURCE = source_file(
    '''
    # chain2.py


    # In f(x, y) = x * y + x, x reaches f along two paths, so its partial
    # derivative adds them up.
    def partials_of_product_plus_x(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        d_x = y + 1
        d_y = x

        return d_x, d_y


    # TODO: Return the partial derivatives of f(x, y) = x * y * x, first with
    #   respect to x, then with respect to y.
    def partials_of_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        ...


    # TODO: Finish the line that sets d_a. Here f(x, y) = a * a + 3 * a, where
    #   a = x * y, and d_a is the derivative of f with respect to a, adding up
    #   the paths through a. The lines after it carry d_a back to x and y.
    def partials_through_a(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        a = x * y
        d_a = ...
        d_x = d_a * y
        d_y = d_a * x

        return d_x, d_y


    # TODO: Return the partial derivatives of f(x, y) = (x + y) * x + y, first
    #   with respect to x, then with respect to y. x and y each reach f along
    #   two paths.
    def partials_of_two_paths(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        ...
    ''',
)

CHAIN2_SOLUTION = source_file(
    '''
    # chain2.py, a solution for comparison


    def partials_of_product_plus_x(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        d_x = y + 1
        d_y = x

        return d_x, d_y


    # Each x in the product gives the rest of it, x * y, and the two paths add up.
    def partials_of_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return 2 * x * y, x * x


    # a * a gives a through each of its two a's, and 3 * a gives 3.
    def partials_through_a(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        a = x * y
        d_a = 2 * a + 3
        d_x = d_a * y
        d_y = d_a * x

        return d_x, d_y


    # x gives x + y through the outer x and x through the x in x + y.
    # y gives x through x + y and 1 on its own.
    def partials_of_two_paths(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return (x + y) + x, x + 1
    ''',
)

PAIRS: tuple[tuple[float, float], ...] = (
    (2.0, 3.0),
    (-1.5, 4.0),
    (0.5, -2.0),
)

EXPECT_TWO_PARTIALS = expect_parts("the x partial", "the y partial")


def check_chain2(checker: Checker) -> None:
    for function, description, f in (
        ("partials_of_reuse", "gives both partials of x * y * x", lambda x, y: x * y * x),
        ("partials_through_a", "gives both partials of a * a + 3 * a with a = x * y", lambda x, y: (x * y) * (x * y) + 3 * (x * y)),
        ("partials_of_two_paths", "gives both partials of (x + y) * x + y", lambda x, y: (x + y) * x + y),
    ):
        check_on_inputs(
            checker=checker,
            function=function,
            description=description,
            inputs=PAIRS,
            reference=lambda x, y, f=f: autograd_partials(f, x, y),
            expect=EXPECT_TWO_PARTIALS,
        )


CHAIN2 = Exercise(
    name="chain2",
    chapter=CHAIN_RULE,
    title="Values that take several paths",
    lesson=CHAIN2_LESSON,
    source=CHAIN2_SOURCE,
    solution=CHAIN2_SOLUTION,
    hints={
        "partials_of_reuse": "x reaches f twice, once through each x of the product. Each path gives the rest of the "
        "product, x * y, and the paths add up. y has one path.",
        "partials_through_a": "a reaches f three times, twice in a * a and once in 3 * a. Add up what each path "
        "gives.",
        "partials_of_two_paths": "Through the outer x, the path of x gives x + y, and through the x inside x + y it "
        "gives x. y gives x through x + y, and 1 on its own.",
    },
    introduces=(),
    check=check_chain2,
)

CHAIN3_LESSON = prose(
    """
    Write a calculation as small named steps, and the steps with the values
    flowing between them form a **computation graph**. Here is one, run
    forward from its inputs.

    ```python
    a, b, c, r = 2.0, -3.0, 10.0, -2.0
    p = a * b      # -6.0
    q = p + c      # 4.0
    L = q * r      # -8.0
    ```

    Now walk it backward from `L` to find how `L` moves with every value. How
    `L` moves with `a` is the partial derivative of `L` with respect to `a`.
    Deep learning calls it the gradient of `a`, written here as `d_a`. The
    gradient of `L` itself is 1, since `L` moves one for one with itself. Each
    step hands gradients to its inputs by the chain rule.

    - An addition passes its gradient to each input unchanged, since each input moves the sum one for one.
    - A multiplication gives each input the gradient times the other input.

    ```python
    d_L = 1.0
    d_q = d_L * r    # -2.0, from L = q * r
    d_r = d_L * q    # 4.0
    d_p = d_q        # -2.0, from q = p + c
    d_c = d_q        # -2.0
    d_a = d_p * b    # 6.0, from p = a * b
    d_b = d_p * a    # -4.0
    ```

    Walking backward this way is **backpropagation**. Each step only needs its
    own inputs and the gradient handed down to it. When a value feeds two
    steps, its gradient is the sum of what both hand it, as in chain2.
    """,
)

CHAIN3_SOURCE = source_file(
    '''
    # chain3.py


    # L = a * b is a multiplication, so each input gets the gradient of L times
    # the other input.
    def backward_product(
        a: float,
        b: float,
    ) -> tuple[float, float]:
        L = a * b

        d_L = 1.0
        d_a = d_L * b
        d_b = d_L * a

        return d_a, d_b


    # TODO: Finish the backward lines, so this returns the gradients of L with
    #   respect to x, y and z. The first two lines run the graph forward.
    def backward_small(
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        s = x + y
        L = s * z

        d_L = 1.0
        d_s = ...
        d_z = ...
        d_x = ...
        d_y = ...

        return d_x, d_y, d_z


    # TODO: Finish the backward lines, so this returns the gradients of L with
    #   respect to x and y. q is used twice in q * q, and x feeds two steps, so
    #   their gradients add up what each use hands them.
    def backward_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        p = x * y
        q = p + x
        L = q * q

        d_L = 1.0
        d_q = ...
        d_p = ...
        d_x = ...
        d_y = ...

        return d_x, d_y
    ''',
)

CHAIN3_SOLUTION = source_file(
    '''
    # chain3.py, a solution for comparison


    def backward_product(
        a: float,
        b: float,
    ) -> tuple[float, float]:
        L = a * b

        d_L = 1.0
        d_a = d_L * b
        d_b = d_L * a

        return d_a, d_b


    def backward_small(
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        s = x + y
        L = s * z

        d_L = 1.0
        d_s = d_L * z
        d_z = d_L * s
        d_x = d_s
        d_y = d_s

        return d_x, d_y, d_z


    # q * q hands q its gradient once through each q. x gets a part from p = x * y
    # and a part from q = p + x.
    def backward_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        p = x * y
        q = p + x
        L = q * q

        d_L = 1.0
        d_q = d_L * q + d_L * q
        d_p = d_q
        d_x = d_p * y + d_q
        d_y = d_p * x

        return d_x, d_y
    ''',
)


def check_chain3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="backward_small",
        description="gives the gradients of (x + y) * z",
        inputs=(
            (2.0, -3.0, 4.0),
            (1.5, 0.5, -2.0),
        ),
        reference=lambda x, y, z: autograd_partials(lambda x, y, z: (x + y) * z, x, y, z),
        expect=expect_parts("d_x", "d_y", "d_z"),
    )
    check_on_inputs(
        checker=checker,
        function="backward_reuse",
        description="gives the gradients of (x * y + x) ** 2",
        inputs=PAIRS,
        reference=lambda x, y: autograd_partials(lambda x, y: (x * y + x) * (x * y + x), x, y),
        expect=expect_parts("d_x", "d_y"),
    )


CHAIN3 = Exercise(
    name="chain3",
    chapter=CHAIN_RULE,
    title="Backpropagation by hand",
    lesson=CHAIN3_LESSON,
    source=CHAIN3_SOURCE,
    solution=CHAIN3_SOLUTION,
    hints={
        "backward_small": "L = s * z is a multiplication, so s gets d_L times z, and z gets d_L times s. "
        "s = x + y is an addition, so x and y each get d_s unchanged.",
        "backward_reuse": "q * q hands q the gradient times the other q, once for each q, and the two add up. "
        "x gets d_p * y from p = x * y and d_q from q = p + x.",
    },
    introduces=("computation graph", "backpropagation"),
    check=check_chain3,
)
