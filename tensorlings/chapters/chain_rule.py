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

        A model is a long chain of small steps, where the output of each step is
        the input of the next. The chain rule gives the derivative of the whole
        chain from the derivatives of the single steps. This chapter applies it
        first to one chain of steps, then to values that reach the output along
        several paths. Last, you work out every derivative of a small
        calculation by hand, the way training does it.

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
    Most functions in deep learning are built by passing the output of one
    function into another function. Take `h(x) = (3 * x + 1) ** 2`. It is made
    of two steps. The inner step computes `u = 3 * x + 1`. The outer step
    squares `u`, which gives `u ** 2`.

    Suppose `x` increases by a tiny amount. The inner step has derivative 3,
    so `u` increases by 3 times that amount. The outer step has derivative
    `2 * u` with respect to `u`, so the output changes by `2 * u` times the
    change in `u`. Combining both steps, the output changes by `2 * u * 3`
    times the change in `x`. So `2 * u * 3` is the derivative of `h`.

    ```python
    x = 2.0
    u = 3 * x + 1      # 7.0
    outer = 2 * u      # 14.0, the derivative of u ** 2 with respect to u
    inner = 3          # the derivative of 3 * x + 1 with respect to x
    outer * inner      # 42.0, the derivative of h at 2
    ```

    This is the **chain rule**. It applies when steps are chained, so that the
    output of each step is the input of the next. The derivative of the whole
    chain is then the product of the derivatives of the steps. Each step's
    derivative is taken at the input that step receives. In the example, the
    outer derivative `2 * u` is taken at `u = 7`, the value the inner step
    produced. A longer chain multiplies more derivatives together.

    The derivative of one step with respect to its own input is called its
    **local derivative**. Computing it needs only that step's input and
    nothing else from the chain. So each step can compute its local
    derivative separately, and the chain rule multiplies them together,
    however many steps there are.

    The rules table in derivatives2 gives the local derivatives. Here is the
    derivative of `exp(x ** 2)` at `x = 1`. The inner step is `x ** 2`, with
    local derivative `2 * x`. The outer step is `exp`, with local derivative
    `exp(u)`.

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


    # The derivative of (3 * x + 1) ** 2 at x, by the chain rule. outer is the
    # derivative of the outer step u ** 2 with respect to u. inner is the
    # derivative of the inner step 3 * x + 1 with respect to x.
    def d_square_of_line(x: float) -> float:
        u = 3 * x + 1
        outer = 2 * u
        inner = 3

        return outer * inner


    # TODO: Finish the two middle lines, so this returns the derivative of
    #   (2 * x - 5) ** 3 at x. The inner step computes u = 2 * x - 5, and the
    #   outer step computes u ** 3.
    #   Example: d_cube_of_line(4.0) returns 54.0.
    def d_cube_of_line(x: float) -> float:
        u = 2 * x - 5
        outer = ...
        inner = ...

        return outer * inner


    # TODO: Finish the three lines, so this returns the derivative of
    #   log(x ** 2 + 1) at x. The inner step computes x ** 2 + 1, and the outer
    #   step applies log to that result.
    #   Example: d_log_of_square(2.0) returns 0.8.
    def d_log_of_square(x: float) -> float:
        u = ...
        outer = ...
        inner = ...

        return outer * inner


    # TODO: Finish the last line, so this returns the derivative of
    #   exp(3 * x) ** 2 at x. This function has three steps. The first computes
    #   a = 3 * x, the second computes b = exp(a), and the third squares b. So
    #   the derivative is the product of three local derivatives.
    #   Example: d_three_steps(0.0) returns 6.0.
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


    # Squaring b has local derivative 2 * b, exp has exp(a), and 3 * x has 3.
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
        "d_cube_of_line": "Set outer to the derivative of u ** 3 with respect to u, using the power rule. Set inner "
        "to the derivative of 2 * x - 5 with respect to x.",
        "d_log_of_square": "Set u to the value of the inner step, the same way d_square_of_line does. outer is the "
        "derivative of log at u, which the table in derivatives2 gives. inner is the derivative of x ** 2 + 1.",
        "d_three_steps": "Multiply the three local derivatives together. Squaring has local derivative 2 times the "
        "value being squared, which is b. exp is its own derivative, so its local derivative is exp of its input "
        "a. The local derivative of 3 * x is 3.",
    },
    introduces=("chain rule",),
    check=check_chain1,
)

CHAIN2_LESSON = prose(
    """
    The chain rule follows one path from an input to the output. Often an
    input reaches the output along more than one path. In
    `f(x, y) = x * y + x`, `x` appears in two places. It is a factor of the
    product `x * y`, and it is also added on its own.

    When `x` increases, both places change the output at the same time.
    Through the product, the output changes by `y` for each unit of change in
    `x`. Through the `+ x` term, it changes by 1 for each unit. The two
    changes add up, so the partial derivative with respect to `x` is `y + 1`.

    ```python
    x, y = 2.0, 5.0
    d_x = y + 1    # 6.0, both paths of x added up
    d_y = x        # 2.0, y has one path
    ```

    In general, when a value is used in several places, its derivative is the
    sum of the contributions of all its paths. The chain rule gives the
    contribution of each path. To work out one path, treat the value's other
    uses as constants. That is why the path through one factor of a product
    contributes the product of the other factors.

    `x * x` is the simplest example. The path through the first `x`
    contributes the second `x`, and the path through the second `x`
    contributes the first. Together they give `2 * x`, which matches the
    power rule.

    The same applies to an intermediate result that is used more than once.
    Take `a = x + y` and `f = a * a + a`. The derivative of `f` with respect
    to `a` adds up three paths, two through `a * a` and one through `+ a`.
    Then the chain rule carries that derivative back to `x` and `y`. Since
    `a = x + y`, the local derivative of `a` with respect to each input is 1.

    ```python
    a = x + y          # 7.0
    d_a = a + a + 1    # 15.0, two paths through a * a and one through + a
    d_x = d_a * 1      # 15.0, since the derivative of a with respect to x is 1
    ```
    """,
)

CHAIN2_SOURCE = source_file(
    '''
    # chain2.py


    # In f(x, y) = x * y + x, x reaches f along two paths, so its partial
    # derivative is the sum of the contributions of the two paths.
    def partials_of_product_plus_x(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        d_x = y + 1
        d_y = x

        return d_x, d_y


    # TODO: Return the partial derivatives of f(x, y) = x * y * x, first with
    #   respect to x, then with respect to y.
    #   Example: partials_of_reuse(2.0, 3.0) returns (12.0, 4.0).
    def partials_of_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        ...


    # TODO: Finish the line that sets d_a. Here f(x, y) = a * a + 3 * a, where
    #   a = x * y. d_a is the derivative of f with respect to a, which adds up
    #   the contributions of all the paths through a. The lines after it use
    #   the chain rule to turn d_a into the partials for x and y.
    #   Example: partials_through_a(2.0, 3.0) returns (45.0, 30.0).
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
    #   Example: partials_of_two_paths(2.0, 3.0) returns (7.0, 3.0).
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


    # The path through each x contributes the other two factors, x * y, and
    # the two paths add up.
    def partials_of_reuse(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return 2 * x * y, x * x


    # Each of the two paths through a * a contributes a, and 3 * a contributes 3.
    def partials_through_a(
        x: float,
        y: float,
    ) -> tuple[float, float]:
        a = x * y
        d_a = 2 * a + 3
        d_x = d_a * y
        d_y = d_a * x

        return d_x, d_y


    # x contributes x + y through the outer x, and x through the x inside x + y.
    # y contributes x through x + y, and 1 through the final + y.
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
    title="Values used in more than one place",
    lesson=CHAIN2_LESSON,
    source=CHAIN2_SOURCE,
    solution=CHAIN2_SOLUTION,
    hints={
        "partials_of_reuse": "x appears twice in the product, so it has two paths. The path through each x "
        "contributes the other two factors, which multiply to x * y. Add the two paths. y appears once, so it has "
        "one path.",
        "partials_through_a": "a appears three times in f, twice in a * a and once in 3 * a. Add up the "
        "contributions of the three paths.",
        "partials_of_two_paths": "For x, the path through the outer x contributes x + y, and the path through the x "
        "inside x + y contributes x. For y, the path through x + y contributes x, and the final + y contributes 1.",
    },
    introduces=(),
    check=check_chain2,
)

CHAIN3_LESSON = prose(
    """
    Any calculation can be written as a sequence of small named steps. The
    steps, together with the values passed between them, form a
    **computation graph**. Here is a small one, computed forward from its
    inputs.

    ```python
    a, b, c, r = 2.0, -3.0, 10.0, -2.0
    p = a * b      # -6.0
    q = p + c      # 4.0
    L = q * r      # -8.0
    ```

    Now go through the graph backward, starting from `L`, to find how much `L`
    changes when each value changes. For `a`, that is the partial derivative
    of `L` with respect to `a`. Deep learning calls it the gradient of `a`,
    and this code writes it as `d_a`. The gradient of `L` itself is 1,
    because the derivative of any value with respect to itself is 1.

    Each step then passes a gradient back to each of its inputs. By the chain
    rule, an input's gradient is the gradient of the step's result times the
    step's local derivative for that input.

    - An addition passes the gradient of its result unchanged to each input, because its local derivatives are all 1.
    - A multiplication gives each input the gradient of its result times the other input, which is the local derivative.

    ```python
    d_L = 1.0
    d_q = d_L * r    # -2.0, from L = q * r
    d_r = d_L * q    # 4.0
    d_p = d_q        # -2.0, from q = p + c
    d_c = d_q        # -2.0
    d_a = d_p * b    # 6.0, from p = a * b
    d_b = d_p * a    # -4.0
    ```

    Computing gradients backward through the graph like this is called
    **backpropagation**. Each step needs only its own inputs and the gradient
    of its result. When a value is used by two steps, its gradient is the sum
    of the gradients both steps pass back to it. This is the same adding up
    of paths as in chain2.
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
    #   respect to x, y and z. The first two lines compute s and L, going
    #   forward through the graph. Each backward line computes one gradient
    #   from the gradients above it.
    #   Example: backward_small(2.0, -3.0, 4.0) returns (4.0, 4.0, -1.0).
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
    #   respect to x and y. q is used twice in q * q, and x is an input to both
    #   p = x * y and q = p + x. So the gradient of q is the sum of what its
    #   two uses pass back, and the same goes for the gradient of x.
    #   Example: backward_reuse(2.0, 3.0) returns (64.0, 32.0).
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


    # q * q passes d_L * q back to q once for each of its two q's. x gets one
    # part from p = x * y and one part from q = p + x.
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
        "backward_reuse": "q * q is a multiplication, so each of its two q's gets d_L times the other q. Add those "
        "two parts to get d_q. x gets d_p * y from p = x * y and d_q from q = p + x.",
    },
    introduces=("computation graph", "backpropagation"),
    check=check_chain3,
)
