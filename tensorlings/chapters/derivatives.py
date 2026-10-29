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

    Nudge the input of a function a little and see how far the output moves.
    The output's move divided by the input's move is the slope there. Take
    `f(x) = x ** 2` at `x = 3`.

    ```python
    def f(x):
        return x ** 2

    (f(3.001) - f(3)) / 0.001          # 6.000999999999479
    (f(3.000001) - f(3)) / 0.000001    # 6.000001000927568
    ```

    As the nudge shrinks, the slope settles on 6. That value is the
    **derivative** of `f` at 3. Near `x = 3` the output moves about six times
    as far as the input. A negative derivative means the output falls as the
    input grows, and 0 means the function is flat there.

    Training rests on this. A model's error depends on the numbers inside it,
    and the derivative of the error tells each number which way to move.

    A nudge one way measures the slope a little to that side of `x`, so it
    gives 6.001 instead of 6. Nudging both ways and dividing by the whole gap
    is far closer for the same nudge. This is the **central difference**.

    ```python
    (f(3.001) - f(2.999)) / 0.002      # 5.999999999999339
    ```

    A nudge can also be too small. A float keeps about 16 significant digits,
    so a nudge of `1e-15` gets rounded.

    ```python
    (3 + 1e-15) - 3                    # 8.881784197001252e-16
    ```

    The nudge that actually happened is about 10 percent off, so the slope
    comes out about 10 percent off too. A nudge around `1e-6` is a safe
    middle.
    """,
)

DERIVATIVES1_SOURCE = source_file(
    '''
    # derivatives1.py

    from collections.abc import Callable


    # TODO: Try this before reading the lesson. Return how far f(x) moves when x
    #   grows by 0.001, divided by 0.001. For f(x) = x ** 2 at x = 3 that is
    #   about 6.001.
    def nudge_slope(
        f: Callable[[float], float],
        x: float,
    ) -> float:
        ...


    # The TODOs below use what the lesson shows.


    def square(x: float) -> float:
        return x ** 2


    # The central difference nudges both ways and divides by the whole gap.
    slope_at_three = (square(3.001) - square(2.999)) / 0.002    # 5.999999999999339


    # TODO: Return the central difference of f at x, nudging by h each way.
    def central_slope(
        f: Callable[[float], float],
        x: float,
        h: float,
    ) -> float:
        ...


    # TODO: This gives a slope far from the derivative. For square at 3 it gives
    #   about 5.33 instead of 6. Read the last paragraph of the lesson, then fix
    #   it.
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


    # A nudge of 1e-15 is lost in rounding. Anything from about 1e-4 to 1e-8 works.
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
        description="divides the move of f by a nudge of 0.001",
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
        description="nudges both ways by h and divides by the whole gap",
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
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
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
        "central_slope": "Follow the line that sets slope_at_three, with x in place of 3 and h in place of 0.001. "
        "The whole gap is 2 * h.",
        "careful_slope": "A nudge of 1e-15 is lost in rounding. Change h to a nudge the lesson calls safe.",
    },
    introduces=("derivative",),
    check=check_derivatives1,
)

DERIVATIVES2_LESSON = prose(
    """
    Nudging gives an estimate and costs two calls per input. For most
    functions a few rules give the derivative exactly. Write the derivative of
    `f` as `f'`.

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

    Here is the derivative of `x ** 3 + 4 * x` at 2, by the rules and by
    nudging.

    ```python
    3 * 2 ** 2 + 4                                  # 16, by the power, constant factor and sum rules
    f = lambda x: x ** 3 + 4 * x
    (f(2.000001) - f(1.999999)) / 0.000002          # 16.00000000046009
    ```

    The product rule handles two parts multiplied together. For
    `x ** 2 * exp(x)` the parts are `x ** 2` and `exp(x)`, so the derivative
    at 1 is `2 * 1 * e + 1 * e`, about 8.15.
    """,
)

DERIVATIVES2_SOURCE = source_file(
    '''
    # derivatives2.py

    import math


    # The derivative of x ** 3 + 4 * x, by the power and sum rules.
    def d_cubic(x: float) -> float:
        return 3 * x ** 2 + 4


    # TODO: Return the derivative of 5 * x ** 4 - 2 * x + 7 at x.
    def d_quartic(x: float) -> float:
        ...


    # The derivative of x ** 2 * exp(x), by the product rule.
    def d_square_times_exp(x: float) -> float:
        return 2 * x * math.exp(x) + x ** 2 * math.exp(x)


    # TODO: Return the derivative of x ** 3 * log(x) at x, by the product rule.
    #   x is always above 0, where log works.
    def d_cube_times_log(x: float) -> float:
        ...


    # TODO: Return the derivative of exp(x) + log(x) + x ** 0.5 at x. x ** 0.5 is
    #   the square root of x, and the power rule covers it too.
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
    inputs, `f(x, y) = x * y + y ** 2`. Hold `y` still and nudge only `x`, and
    you get the **partial derivative** of `f` with respect to `x`. A partial
    treats every other input as a constant, so the rules of derivatives2 apply
    as they are.

    For `x`, hold `y` still. Then `x * y` is the constant `y` times `x`, with
    derivative `y`. `y ** 2` is a constant, with derivative 0. At `x = 2,
    y = 3` the partial is 3.

    For `y`, hold `x` still. `x * y` gives `x`, and `y ** 2` gives `2 * y`, so
    the partial is `x + 2 * y`, which is 8.

    Keep the inputs in one tensor `v`, and the partials in a tensor of the same
    shape. That tensor is the **gradient** of `f` at `v`.

    ```python
    def f(v):
        return v[0] * v[1] + v[1] ** 2

    v = torch.tensor([2.0, 3.0])    # the gradient of f here is tensor([3., 8.])
    ```

    To estimate one partial in code, nudge one entry and leave the rest.
    `torch.zeros_like(v)` makes a tensor of zeros with the shape of `v`, so a
    nudge along entry 1 is

    ```python
    nudge = torch.zeros_like(v)
    nudge[1] = 0.01
    (f(v + nudge) - f(v - nudge)) / 0.02    # tensor(8.0000)
    ```

    The gradient points the way `f` grows fastest, and its negative the way `f`
    falls fastest. Training moves a model's numbers along that negative.

    In `(v ** 2).sum()` each entry only touches its own square. So each
    entry's partial is the derivative of its square, and the gradient is
    `2 * v`.
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

    # Nudging entry 0 only estimates the partial derivative with respect to v[0],
    # which is v[1].
    nudge = torch.zeros_like(v)
    nudge[0] = 0.01
    partial_0 = (f(v + nudge) - f(v - nudge)) / 0.02      # tensor(3.0000)


    # TODO: Finish the second and last lines, so this returns the central
    #   difference of f at v along entry i, nudging by h each way.
    def partial(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
        i: int,
        h: float,
    ) -> Tensor:
        nudge = torch.zeros_like(v)
        nudge[i] = ...

        return ...


    # TODO: Finish the line in the loop, so this returns the gradient of f at v,
    #   with each entry's partial estimated by partial and a nudge of 0.001.
    def numeric_gradient(
        f: Callable[[Tensor], Tensor],
        v: Tensor,
    ) -> Tensor:
        gradient = torch.zeros_like(v)

        for i in range(len(v)):
            gradient[i] = ...

        return gradient


    # TODO: Return the exact gradient of (v ** 3).sum() at v, without nudging.
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
        "partial": "Put h at entry i of nudge. Then follow the line that sets partial_0, with 2 * h in place of "
        "0.02.",
        "numeric_gradient": "Call partial with f, v, i and 0.001.",
        "gradient_of_cubes": "Each entry only touches its own cube, so its partial is the derivative of x ** 3 from "
        "the table in derivatives2. Math on a tensor does it to every entry at once.",
    },
    introduces=("gradient",),
    check=check_derivatives3,
)
