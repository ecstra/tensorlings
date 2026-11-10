"""03_derivatives, the slope of a function, the rules that give it exactly, and the gradient."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import Checker, Described, check_on_inputs, expect_close, expect_tensor
from ..text import prose, source_file

DERIVATIVES = Chapter(
    folder="03_derivatives",
    title="Derivatives",
    readme=source_file(
        """
        # Derivatives

        A derivative says how fast a function's output changes when its input
        changes a little. Training rests on it, since it tells each number in a
        model which way to move. This chapter estimates derivatives by nudging
        the input, works them out exactly with a few rules, and then does both
        for functions of many inputs.

        ## Further reading

        - [3Blue1Brown, the essence of calculus](https://www.3blue1brown.com/topics/calculus)
        - [Dive into Deep Learning, calculus](https://d2l.ai/chapter_preliminaries/calculus.html)
        """,
    ),
)

DERIVATIVES1_LESSON = prose(
    """
    Try the first TODO before you read on.

    The slope of a function at a point says how fast its output changes when
    its input changes. You can estimate it with a small step `h`. Increase the
    input by `h`, see how much the output changed, and divide that change by
    `h`. Adding a small step to the input like this is called nudging it.

    ```text
    slope ≈ (f(x + h) - f(x)) / h
    ```

    Here it is for `f(x) = x ** 2` at `x = 3`, with two step sizes.

    ```python
    def f(x):
        return x ** 2

    (f(3.001) - f(3)) / 0.001          # 6.000999999999479
    (f(3.000001) - f(3)) / 0.000001    # 6.000001000927568
    ```

    As the step gets smaller, the answer gets closer to 6. That limit is the
    **derivative** of `f` at 3. It means that near `x = 3`, the output changes
    about 6 times as much as the input. A negative derivative means the output
    goes down as the input goes up. A derivative of 0 means the function is
    flat at that point.

    Training depends on derivatives. A model's error depends on the numbers
    inside the model. The derivative of the error with respect to one of those
    numbers says whether increasing it makes the error go up or down. So
    training knows which way to change each number.

    Stepping only forward measures the slope a little to the right of `x`,
    which is why it gave 6.001 instead of 6. A better estimate steps both
    ways, to `x + h` and `x - h`, and divides by the distance between them,
    `2 * h`. This is called the **central difference**.

    ```python
    (f(3.001) - f(2.999)) / 0.002      # 5.999999999999339
    ```

    For `x ** 2` the central difference is exact apart from rounding. For
    `x ** 3` at 2, where the true derivative is 12, it is still much closer
    than the forward step.

    ```python
    g = lambda x: x ** 3
    (g(2.001) - g(2)) / 0.001          # 12.006000999997823
    (g(2.001) - g(1.999)) / 0.002      # 12.000000999998317
    ```

    The step can also be too small. A Python float keeps only about 16
    significant digits, so adding `1e-15` to 3 gets rounded.

    ```python
    (3 + 1e-15) - 3                    # 8.881784197001252e-16
    ```

    The step that actually happened is about 11 percent smaller than `1e-15`,
    so the slope comes out about 11 percent wrong too. A step around `1e-6` is
    small enough to be accurate and big enough to avoid this rounding.
    """,
)

DERIVATIVES1_SOURCE = source_file(
    '''
    # derivatives1.py

    from collections.abc import Callable


    # TODO: Try this before reading the lesson. Estimate how fast f changes at
    #   x. Compute f(x + 0.001) - f(x), which is how much the output changes
    #   when x increases by 0.001, then divide that change by 0.001.
    #   Example: for f(x) = x ** 2 and x = 3, nudge_slope returns about 6.001.
    def nudge_slope(
        f: Callable[[float], float],
        x: float,
    ) -> float:
        ...


    # The TODOs below use what the lesson teaches.


    def square(x: float) -> float:
        return x ** 2


    # The central difference of square at 3, with a step of 0.001 each way.
    slope_at_three = (square(3.001) - square(2.999)) / 0.002    # 5.999999999999339


    # TODO: Return the central difference of f at x with a step of h. Evaluate f
    #   at x + h and at x - h, and divide the difference by 2 * h.
    #   Example: central_slope(square, 3.0, 0.001) returns about 6.0.
    def central_slope(
        f: Callable[[float], float],
        x: float,
        h: float,
    ) -> float:
        ...


    # TODO: This returns a slope far from the true derivative. For square at 3
    #   it returns about 5.33 instead of 6. Read the last paragraph of the
    #   lesson, then fix it.
    def careful_slope(
        f: Callable[[float], float],
        x: float,
    ) -> float:
        h = 1e-15

        return (f(x + h) - f(x - h)) / (2 * h)
    ''',
)

DERIVATIVES1_SOLUTION = source_file(
    '''
    # derivatives1.py, a solution for comparison

    from collections.abc import Callable


    def nudge_slope(
        f: Callable[[float], float],
        x: float,
    ) -> float:
        return (f(x + 0.001) - f(x)) / 0.001


    def square(x: float) -> float:
        return x ** 2


    slope_at_three = (square(3.001) - square(2.999)) / 0.002


    def central_slope(
        f: Callable[[float], float],
        x: float,
        h: float,
    ) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)


    # A step of 1e-15 is lost to rounding. Anything from about 1e-4 to 1e-8 works.
    def careful_slope(
        f: Callable[[float], float],
        x: float,
    ) -> float:
        h = 1e-6

        return (f(x + h) - f(x - h)) / (2 * h)
    ''',
)

SQUARE = Described(
    value=lambda x: x ** 2,
    text="x ** 2",
)
LINE = Described(
    value=lambda x: 3 * x + 1,
    text="3 * x + 1",
)
CUBIC = Described(
    value=lambda x: x ** 3 - 2 * x,
    text="x ** 3 - 2 * x",
)
QUADRATIC = Described(
    value=lambda x: 5 * x ** 2 + x,
    text="5 * x ** 2 + x",
)


def check_derivatives1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="nudge_slope",
        description="divides f(x + 0.001) - f(x) by 0.001",
        inputs=(
            (SQUARE, 3.0),
            (LINE, 2.0),
            (CUBIC, -1.0),
        ),
        reference=lambda f, x: (f(x + 0.001) - f(x)) / 0.001,
        expect=expect_close,
    )
    # With h as big as 0.1, a one sided nudge on a cubic misses by enough to fail.
    check_on_inputs(
        checker=checker,
        function="central_slope",
        description="divides f(x + h) - f(x - h) by 2 * h",
        inputs=(
            (SQUARE, 3.0, 0.001),
            (CUBIC, 1.5, 0.1),
            (QUADRATIC, -2.0, 0.01),
        ),
        reference=lambda f, x, h: (f(x + h) - f(x - h)) / (2 * h),
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="careful_slope",
        description="comes close to the derivative",
        inputs=(
            (SQUARE, 3.0),
            (CUBIC, -1.0),
            (QUADRATIC, 0.5),
        ),
        reference=lambda f, x: (f(x + 1e-5) - f(x - 1e-5)) / 2e-5,
        # A nudge of 0.01 misses the cubic by 1e-4, so this tolerance fails it, while 1e-3 to 1e-11 pass.
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


DERIVATIVES1 = Exercise(
    name="derivatives1",
    chapter=DERIVATIVES,
    title="The slope of a function",
    lesson=DERIVATIVES1_LESSON,
    source=DERIVATIVES1_SOURCE,
    solution=DERIVATIVES1_SOLUTION,
    hints={
        "nudge_slope": "Call f at x + 0.001 and at x, take the difference, and divide it by 0.001.",
        "central_slope": "Copy the pattern of the line that sets slope_at_three, using x instead of 3 and h instead "
        "of 0.001. The distance from x - h to x + h is 2 * h.",
        "careful_slope": "A step of 1e-15 is lost to rounding. Change h to the step size the lesson recommends.",
    },
    introduces=("derivative",),
    check=check_derivatives1,
)

DERIVATIVES2_LESSON = prose(
    """
    Nudging gives only an estimate, and it costs two calls of the function for
    every input. For most functions, a few rules give the exact derivative
    instead. The derivative of `f` is written `f'`.

    | Function | Derivative | Rule |
    |---|---|---|
    | `c`, a constant | `0` | a constant never changes |
    | `x` | `1` | power, since `x` is `x ** 1` |
    | `x ** n` | `n * x ** (n - 1)` | power |
    | `c * f(x)` | `c * f'(x)` | a constant factor stays |
    | `f(x) + g(x)` | `f'(x) + g'(x)` | sum |
    | `f(x) * g(x)` | `f'(x) * g(x) + f(x) * g'(x)` | product |
    | `exp(x)` | `exp(x)` | exp is its own derivative |
    | `log(x)` | `1 / x` | log |

    `exp(x)` is the number e, about 2.718, raised to the power `x`, and `log`
    undoes it. Both turn up all through deep learning. In Python they are
    `math.exp` and `math.log`.

    The expressions here are written as Python code. Python computes `**`
    first, then `*` and `/`, then `+` and `-`. So `x ** 3 + 4 * x` means
    `(x ** 3) + (4 * x)`. It is two terms added together.

    The sum rule says to take the derivative of each term and add the results.
    The derivative of `x ** 3` is `3 * x ** 2`, by the power rule. The
    derivative of `4 * x` is `4 * 1`, by the constant factor rule. So the
    derivative of `x ** 3 + 4 * x` is `3 * x ** 2 + 4`. Here it is at `x = 2`,
    by the rules and by nudging.

    ```python
    3 * 2 ** 2 + 4                                  # 16, by the rules
    f = lambda x: x ** 3 + 4 * x
    (f(2.000001) - f(1.999999)) / 0.000002          # 16.00000000046009
    ```

    The product rule handles two parts multiplied together. Python reads
    `x ** 2 * exp(x)` as `(x ** 2) * exp(x)`, so the two parts are
    `f(x) = x ** 2` and `g(x) = exp(x)`. Their derivatives are `f'(x) = 2 * x`
    and `g'(x) = exp(x)`. The product rule gives
    `2 * x * exp(x) + x ** 2 * exp(x)`. At `x = 1` that is `2 * e + e`, about
    8.15.
    """,
)

DERIVATIVES2_SOURCE = source_file(
    '''
    # derivatives2.py

    import math


    # The derivative of x ** 3 + 4 * x, by the power and sum rules.
    def d_cubic(x: float) -> float:
        return 3 * x ** 2 + 4


    # TODO: Return the derivative of 5 * x ** 4 - 2 * x + 7 at x. Python reads
    #   this as (5 * (x ** 4)) - (2 * x) + 7, so it has three terms. Take the
    #   derivative of each term, then add or subtract them the same way.
    #   Example: d_quartic(2.0) returns 158.0.
    def d_quartic(x: float) -> float:
        ...


    # The derivative of x ** 2 * exp(x), by the product rule.
    def d_square_times_exp(x: float) -> float:
        return 2 * x * math.exp(x) + x ** 2 * math.exp(x)


    # TODO: Return the derivative of x ** 3 * log(x) at x, by the product rule.
    #   Python reads this as (x ** 3) * log(x), so the two parts are x ** 3
    #   and log(x). x is always above 0, where log is defined.
    #   Example: d_cube_times_log(1.0) returns 1.0.
    def d_cube_times_log(x: float) -> float:
        ...


    # TODO: Return the derivative of exp(x) + log(x) + x ** 0.5 at x. The three
    #   terms are exp(x), log(x) and x ** 0.5. x ** 0.5 is the square root of
    #   x, and the power rule works for it too, with n = 0.5.
    #   Example: d_three_terms(1.0) returns about 4.218.
    def d_three_terms(x: float) -> float:
        ...
    ''',
)

DERIVATIVES2_SOLUTION = source_file(
    '''
    # derivatives2.py, a solution for comparison

    import math


    def d_cubic(x: float) -> float:
        return 3 * x ** 2 + 4


    def d_quartic(x: float) -> float:
        return 20 * x ** 3 - 2


    def d_square_times_exp(x: float) -> float:
        return 2 * x * math.exp(x) + x ** 2 * math.exp(x)


    # x ** 3 * (1 / x) is x ** 2.
    def d_cube_times_log(x: float) -> float:
        return 3 * x ** 2 * math.log(x) + x ** 2


    def d_three_terms(x: float) -> float:
        return math.exp(x) + 1 / x + 0.5 * x ** -0.5
    ''',
)

POSITIVE_POINTS: tuple[tuple[float], ...] = (
    (0.5,),
    (1.0,),
    (2.5,),
)


def check_derivatives2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="d_quartic",
        description="gives the derivative of 5 * x ** 4 - 2 * x + 7",
        inputs=(
            (2.0,),
            (-1.5,),
            (0.0,),
        ),
        reference=lambda x: 20 * x ** 3 - 2,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="d_cube_times_log",
        description="gives the derivative of x ** 3 * log(x)",
        inputs=POSITIVE_POINTS,
        reference=lambda x: 3 * x ** 2 * math.log(x) + x ** 2,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="d_three_terms",
        description="gives the derivative of exp(x) + log(x) + x ** 0.5",
        inputs=POSITIVE_POINTS,
        reference=lambda x: math.exp(x) + 1 / x + 0.5 * x ** -0.5,
        expect=expect_close,
    )


DERIVATIVES2 = Exercise(
    name="derivatives2",
    chapter=DERIVATIVES,
    title="Rules for exact derivatives",
    lesson=DERIVATIVES2_LESSON,
    source=DERIVATIVES2_SOURCE,
    solution=DERIVATIVES2_SOLUTION,
    hints={
        "d_quartic": "Take each term's derivative with the power rule and add them up. A constant on its own has "
        "derivative 0.",
        "d_cube_times_log": "The two parts are x ** 3 and log(x), with derivatives 3 * x ** 2 and 1 / x. Put them "
        "into the product rule the way d_square_times_exp does.",
        "d_three_terms": "Take each term's derivative from the table and add them. The power rule turns x ** 0.5 "
        "into 0.5 * x ** -0.5.",
    },
    introduces=(),
    check=check_derivatives2,
)

DERIVATIVES3_LESSON = prose(
    """
    A model's error depends on many numbers at once. Take a function of two
    inputs, `f(x, y) = x * y + y ** 2`. If you hold `y` fixed and change only
    `x`, the derivative you get is the **partial derivative** of `f` with
    respect to `x`. It treats every other input as a constant, so the rules
    from derivatives2 work unchanged.

    For the partial with respect to `x`, treat `y` as a constant. Then `x * y`
    is the constant `y` times `x`, whose derivative is `y`. And `y ** 2` is a
    constant, whose derivative is 0. So at `x = 2, y = 3` the partial is 3.

    For the partial with respect to `y`, treat `x` as a constant. `x * y` has
    derivative `x`, and `y ** 2` has derivative `2 * y`. So the partial is
    `x + 2 * y`, which is 8 at the same point.

    Put the inputs in one tensor `v`, and put the partial derivatives in a
    tensor of the same shape. That tensor of partials is the **gradient** of
    `f` at `v`.

    ```python
    def f(v):
        return v[0] * v[1] + v[1] ** 2

    v = torch.tensor([2.0, 3.0])    # the gradient of f here is tensor([3., 8.])
    ```

    To estimate one partial derivative in code, nudge one entry of `v` and
    leave the others alone. `torch.zeros_like(v)` makes a tensor of zeros with
    the shape of `v`. So the central difference along entry 1 is

    ```python
    nudge = torch.zeros_like(v)
    nudge[1] = 0.01
    (f(v + nudge) - f(v - nudge)) / 0.02    # tensor(8.0000)
    ```

    A tensor like `v` holds float32 numbers, which keep about 7 significant
    digits instead of a Python float's 16. So a tensor needs a bigger step.
    For this partial, whose true value is 8, a step of `1e-6` gives 7.63 and a
    step of `1e-7` gives 0. A step of 0.01 gives 7.99999, so this file uses
    steps of 0.01 and 0.001.

    The gradient points in the direction where `f` increases fastest. Its
    negative points where `f` decreases fastest. Training changes a model's
    numbers in the direction of the negative gradient of its error.

    In `(v ** 2).sum()`, each entry appears only in its own square. So the
    partial for each entry is the derivative of its square, `2 * v[i]`, and
    the whole gradient is `2 * v`.
    """,
)

DERIVATIVES3_SOURCE = source_file(
    '''
    # derivatives3.py

    from collections.abc import Callable

    import torch
    from torch import Tensor


    def f(v: Tensor) -> Tensor:
        return v[0] * v[1] + v[1] ** 2


    v = torch.tensor([2.0, 3.0])

    # Nudging only entry 0 estimates the partial derivative with respect to
    # v[0], which is v[1], so 3 here.
    nudge = torch.zeros_like(v)
    nudge[0] = 0.01
    partial_0 = (f(v + nudge) - f(v - nudge)) / 0.02      # tensor(3.0000)


    # TODO: Finish the second and last lines, so this returns the central
    #   difference of f at v along entry i with a step of h. That is the
    #   partial derivative of f with respect to v[i], estimated.
    #   Example: partial(f, v, 1, 0.01) returns about tensor(8.).
    def partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
        i: int,
        h: float,
    ) -> Tensor:
        nudge = torch.zeros_like(v)
        nudge[i] = ...

        return ...


    # TODO: Finish the line in the loop, so this returns the estimated gradient
    #   of f at v. Entry i of the gradient is the partial derivative along
    #   entry i, from your partial function with a step of 0.001.
    #   Example: numeric_gradient(f, v) returns about tensor([3., 8.]).
    def numeric_gradient(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> Tensor:
        gradient = torch.zeros_like(v)

        for i in range(len(v)):
            gradient[i] = ...

        return gradient


    # TODO: Return the exact gradient of (v ** 3).sum() at v, worked out with
    #   the derivative rules instead of nudging. v can have any shape.
    #   Example: gradient_of_cubes(torch.tensor([1.0, 2.0])) returns
    #   tensor([ 3., 12.]).
    def gradient_of_cubes(v: Tensor) -> Tensor:
        ...
    ''',
)

DERIVATIVES3_SOLUTION = source_file(
    '''
    # derivatives3.py, a solution for comparison

    from collections.abc import Callable

    import torch
    from torch import Tensor


    def f(v: Tensor) -> Tensor:
        return v[0] * v[1] + v[1] ** 2


    v = torch.tensor([2.0, 3.0])

    nudge = torch.zeros_like(v)
    nudge[0] = 0.01
    partial_0 = (f(v + nudge) - f(v - nudge)) / 0.02


    def partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
        i: int,
        h: float,
    ) -> Tensor:
        nudge = torch.zeros_like(v)
        nudge[i] = h

        return (f(v + nudge) - f(v - nudge)) / (2 * h)


    def numeric_gradient(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> Tensor:
        gradient = torch.zeros_like(v)

        for i in range(len(v)):
            gradient[i] = partial(f, v, i, 0.001)

        return gradient


    # Each entry's partial is the derivative of its own cube.
    def gradient_of_cubes(v: Tensor) -> Tensor:
        return 3 * v ** 2
    ''',
)


def central_partial(
    f: object,
    v: Tensor,
    i: int,
    h: float,
) -> Tensor:
    nudge = torch.zeros_like(v)
    nudge[i] = h

    return (f(v + nudge) - f(v - nudge)) / (2 * h)


# Doubles, so the estimates come out the same however the learner orders the arithmetic.
PRODUCT = Described(
    value=lambda v: v[0] * v[1] + v[1] ** 2,
    text="v[0] * v[1] + v[1] ** 2",
)
SQUARES = Described(
    value=lambda v: (v ** 2).sum(),
    text="(v ** 2).sum()",
)
MIXED = Described(
    value=lambda v: v[0] ** 3 * v[1] - v[2],
    text="v[0] ** 3 * v[1] - v[2]",
)


def check_derivatives3(checker: Checker) -> None:
    # With h as big as 0.1, a one sided nudge on a cube misses by enough to fail.
    check_on_inputs(
        checker=checker,
        function="partial",
        description="estimates one partial derivative with a central difference",
        inputs=(
            (PRODUCT, torch.tensor([2.0, 3.0], dtype=torch.float64), 1, 0.001),
            (SQUARES, torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64), 2, 0.01),
            (MIXED, torch.tensor([1.5, 2.0, 4.0], dtype=torch.float64), 0, 0.1),
        ),
        reference=central_partial,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="numeric_gradient",
        description="estimates every partial derivative",
        inputs=(
            (PRODUCT, torch.tensor([2.0, 3.0], dtype=torch.float64)),
            (MIXED, torch.tensor([1.5, 2.0, 4.0], dtype=torch.float64)),
        ),
        reference=lambda f, v: torch.stack([central_partial(f, v, i, 0.001) for i in range(len(v))]),
    )
    check_on_inputs(
        checker=checker,
        function="gradient_of_cubes",
        description="gives the gradient of (v ** 3).sum()",
        inputs=(
            (torch.tensor([1.0, -2.0, 0.5]),),
            (torch.tensor([[2.0, 0.0], [-1.0, 3.0]]),),
        ),
        reference=lambda v: 3 * v ** 2,
        expect=expect_tensor,
    )


DERIVATIVES3 = Exercise(
    name="derivatives3",
    chapter=DERIVATIVES,
    title="Several inputs and the gradient",
    lesson=DERIVATIVES3_LESSON,
    source=DERIVATIVES3_SOURCE,
    solution=DERIVATIVES3_SOLUTION,
    hints={
        "partial": "Set entry i of nudge to h. Then copy the pattern of the line that sets partial_0, using 2 * h "
        "instead of 0.02.",
        "numeric_gradient": "Call partial with f, v, i and 0.001.",
        "gradient_of_cubes": "Each entry appears only in its own cube, so its partial is the derivative of x ** 3 "
        "from the table in derivatives2. Arithmetic on a tensor applies to every entry at once.",
    },
    introduces=("gradient",),
    check=check_derivatives3,
)
