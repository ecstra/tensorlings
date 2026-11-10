"""25_adaptive, RMSProp, Adam and AdamW built from their papers and matched with torch.optim, and when to use which."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .initialization import expect_numbers, relative
from .names import context_sets, names_model, reference_validation_loss
from .sgd import check_optimizer_class

ADAPTIVE = Chapter(
    folder="25_adaptive",
    title="Adaptive optimizers",
    readme=source_file(
        """
        # Adaptive optimizers

        An adaptive optimizer gives every parameter its own step size,
        computed from that parameter's past gradients. This chapter builds
        RMSProp, Adam (following Algorithm 1 of the Adam paper) and AdamW, and
        checks each one against PyTorch's version. It also shows why Adam
        needs its bias correction, and which optimizer and learning rate suit
        which kind of model.

        ## Further reading

        - [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980)
        - [Loshchilov and Hutter, decoupled weight decay regularization](https://arxiv.org/abs/1711.05101)
        - [Zhang and others, which algorithmic choices matter at which batch sizes](https://arxiv.org/abs/1907.04164)
        - [Hinton, Coursera lecture 6e, RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
        - [Ruder, an overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)
        """,
    ),
)

ADAM_CLASS = '''
class MyAdam:
    def __init__(
        self,
        parameters,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]

    def step(self) -> None:
        self.t += 1

        with torch.no_grad():
            for position, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                self.m[position] = self.beta1 * self.m[position] + (1 - self.beta1) * grad
                self.v[position] = self.beta2 * self.v[position] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[position] / (1 - self.beta1 ** self.t)
                v_hat = self.v[position] / (1 - self.beta2 ** self.t)
                parameter -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_ADAM = textwrap.indent(ADAM_CLASS.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# rmsprop1
# ---------------------------------------------------------------------------

RMSPROP1_LESSON = prose(
    """
    SGD uses the same learning rate for every parameter. So a parameter whose
    gradients are always large takes big steps, and a parameter whose
    gradients are always small barely changes. **RMSProp**, from Hinton's 2012
    lecture notes, divides each parameter's step by the root mean square of
    that parameter's recent gradients. The root mean square is the square root
    of the average of the squared gradients.

    ```text
    s = alpha * s + (1 - alpha) * (g ** 2)
    parameter = parameter - (lr * g) / (sqrt(s) + eps)
    ```

    Here `g` is the gradient and `s` is a running average of its square. Each
    step, `s` keeps the fraction `alpha` of its old value and adds
    `1 - alpha` times the new `g ** 2`.

    So each parameter changes by about the learning rate per step, whatever
    the size of its gradients. In momentum2's valley, the large gradients in
    the steep y direction get divided by their large root mean square. That
    shrinks the bouncing across the valley. The small gradients in the gentle
    x direction get divided by a small root mean square, so the steps along
    the valley get bigger.

    On that valley, RMSProp with a learning rate of 0.1 brought the loss below
    0.001 and kept it there from step 125 on. Plain SGD at its best learning
    rate needed 301 steps for the same, and momentum needed 109. Above a rate
    of 0.1, RMSProp kept bouncing back above 0.001. That is because its steps
    stay about as big as the learning rate even near the bottom.

    PyTorch's default `alpha` is 0.99, which averages over roughly the last
    100 steps. The `eps` of `1e-8` prevents a division by 0 when the average
    is 0. Each parameter has its own average, and every average starts at 0.
    `torch.optim.RMSprop(params, lr=0.01)` does all of this.

    Starting the averages at 0 has a side effect. After one step the average
    is only `(1 - alpha) * (g ** 2)`, a hundredth of the first squared gradient.
    Its square root is a tenth of the gradient's size. So the first step comes
    out about 10 times the learning rate.
    """,
)

RMSPROP1_SOURCE = source_file(
    '''
    # rmsprop1.py

    import torch
    from torch import Tensor

    grad = torch.tensor([0.001, 1000.0])
    sizes = grad.abs()                               # the absolute value of each entry


    # TODO: Return the new running average of the squared gradient, which is
    #   alpha times square_average plus (1 - alpha) times grad ** 2.
    #   Example: square_average_update(torch.tensor([1.0, 4.0]),
    #   torch.tensor([2.0, -1.0]), 0.99) returns tensor([1.0300, 3.9700]).
    def square_average_update(
        square_average: Tensor,
        grad: Tensor,
        alpha: float,
    ) -> Tensor:
        ...


    # TODO: Finish the two lines with ... in step. The first sets the
    #   parameter's average to the result of square_average_update. The second
    #   subtracts self.lr * grad / (sqrt(average) + self.eps) from the
    #   parameter, using the new average. Add eps after taking the square root.
    class MyRMSprop:
        def __init__(
            self,
            parameters,
            lr: float = 0.01,
            alpha: float = 0.99,
            eps: float = 1e-8,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr
            self.alpha = alpha
            self.eps = eps
            self.square_averages = [torch.zeros_like(parameter) for parameter in self.parameters]

        def step(self) -> None:
            with torch.no_grad():
                for position, parameter in enumerate(self.parameters):
                    if parameter.grad is None:
                        continue

                    self.square_averages[position] = ...
                    parameter -= ...

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    # TODO: Return the size of RMSProp's first step for each entry of grad, as
    #   a positive number. The average starts at 0, so the first step uses the
    #   average after one update from 0.
    #   Example: first_step(grad, 0.01, 0.99, 1e-8) with the grad above returns
    #   about tensor([0.1000, 0.1000]).
    def first_step(
        grad: Tensor,
        lr: float,
        alpha: float,
        eps: float,
    ) -> Tensor:
        ...
    ''',
)

RMSPROP1_SOLUTION = source_file(
    '''
    # rmsprop1.py, a solution for comparison

    import torch
    from torch import Tensor

    grad = torch.tensor([0.001, 1000.0])
    sizes = grad.abs()


    def square_average_update(
        square_average: Tensor,
        grad: Tensor,
        alpha: float,
    ) -> Tensor:
        return alpha * square_average + (1 - alpha) * grad ** 2


    class MyRMSprop:
        def __init__(
            self,
            parameters,
            lr: float = 0.01,
            alpha: float = 0.99,
            eps: float = 1e-8,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr
            self.alpha = alpha
            self.eps = eps
            self.square_averages = [torch.zeros_like(parameter) for parameter in self.parameters]

        def step(self) -> None:
            with torch.no_grad():
                for position, parameter in enumerate(self.parameters):
                    if parameter.grad is None:
                        continue

                    self.square_averages[position] = square_average_update(self.square_averages[position], parameter.grad, self.alpha)
                    parameter -= self.lr * parameter.grad / (self.square_averages[position].sqrt() + self.eps)

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    # The first average is (1 - alpha) * (grad ** 2), so the step is about lr / sqrt(1 - alpha).
    def first_step(
        grad: Tensor,
        lr: float,
        alpha: float,
        eps: float,
    ) -> Tensor:
        average = (1 - alpha) * grad ** 2

        return lr * grad.abs() / (average.sqrt() + eps)
    ''',
)


def check_my_rmsprop(found: type) -> None:
    # An eps of 1e-8 can't tell sqrt(average) + eps from sqrt(average + eps), so the last round uses 0.1.
    for alpha, eps in ((0.99, 1e-8), (0.9, 1e-8), (0.99, 0.1)):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.01, "alpha": alpha, "eps": eps},
            reference=torch.optim.RMSprop,
            reference_arguments={"lr": 0.01, "alpha": alpha, "eps": eps},
            name="MyRMSprop",
        )


def check_rmsprop1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="square_average_update",
        description="moves the average toward the squared gradient",
        inputs=(
            (torch.tensor([1.0, 4.0]), torch.tensor([2.0, -1.0]), 0.99),
            (small_tensor(520, 2, 3).abs(), small_tensor(521, 2, 3), 0.9),
        ),
        reference=lambda square_average, grad, alpha: alpha * square_average + (1 - alpha) * grad ** 2,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="MyRMSprop",
        method="step",
        description="updates the parameters like torch.optim.RMSprop",
        run=check_my_rmsprop,
    )
    check_on_inputs(
        checker=checker,
        function="first_step",
        description="gives the size of the first step from an average of 0",
        inputs=(
            (torch.tensor([0.001, 1000.0, -3.0]), 0.01, 0.99, 1e-8),
            (torch.tensor([0.5, 0.0, -2.0]), 0.1, 0.9, 1e-8),
        ),
        reference=lambda grad, lr, alpha, eps: lr * grad.abs() / (((1 - alpha) * grad ** 2).sqrt() + eps),
        expect=expect_tensor,
    )


RMSPROP1 = Exercise(
    name="rmsprop1",
    chapter=ADAPTIVE,
    title="RMSProp",
    lesson=RMSPROP1_LESSON,
    source=RMSPROP1_SOURCE,
    solution=RMSPROP1_SOLUTION,
    hints={
        "square_average_update": "Multiply square_average by alpha, then add (1 - alpha) times grad ** 2.",
        "MyRMSprop.step": "The new average is square_average_update called on the old average, parameter.grad and "
        "self.alpha. Then subtract from the parameter self.lr times parameter.grad, divided by the new average's "
        ".sqrt() plus self.eps.",
        "first_step": "The first average is (1 - alpha) * (grad ** 2). Divide lr times grad.abs() by the square root "
        "of that average plus eps.",
    },
    introduces=(),
    check=check_rmsprop1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# adam1
# ---------------------------------------------------------------------------

ADAM1_LESSON = prose(
    """
    **Adam**, from Kingma and Ba in 2015, combines momentum and RMSProp. It
    keeps two running averages for each parameter. `m` is a running average of
    the gradient, and `v` is a running average of its square. Each step
    changes the parameter by about the learning rate times `m` divided by the
    square root of `v`. Here is one step of Algorithm 1 from the paper, where
    `t` counts the steps starting from 1.

    ```text
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - (beta1 ** t))
    v_hat = v / (1 - (beta2 ** t))
    parameter = parameter - (lr * m_hat) / (sqrt(v_hat) + eps)
    ```

    `m` plays the part of momentum's velocity, but as an average instead of a
    sum. `v` is the same average of squared gradients that RMSProp keeps. Both
    averages start at 0, which pulls them toward 0 during the first steps.
    `m_hat` and `v_hat`, called the hats, undo that pull, and adam2 is about
    them. The paper's defaults, which PyTorch also uses, are a `beta1` of 0.9,
    a `beta2` of 0.999 and an `eps` of `1e-8`.

    On the first step, `m_hat` equals the gradient and `v_hat` equals its
    square. So the first step is the learning rate times the sign of the
    gradient, which is almost exactly the same size for every parameter.

    `torch.optim.Adam(params, lr=0.001)` does all of this. The usual starting
    learning rate is between 0.0003 and 0.001.
    """,
)

ADAM1_SOURCE = source_file(
    '''
    # adam1.py

    import torch
    from torch import Tensor

    t = 3
    correction = 1 - 0.9 ** t                        # 0.271, so after 3 steps m is 0.271 times a steady gradient


    # TODO: Finish step, so it runs the five lines of Algorithm 1 from the
    #   lesson for each parameter whose .grad is not None. Store each
    #   parameter's m and v in self.m and self.v, at the parameter's position.
    #   self.t is the step count t, and it is already 1 on the first step.
    class MyAdam:
        def __init__(
            self,
            parameters,
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr
            self.beta1, self.beta2 = betas
            self.eps = eps
            self.t = 0
            self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
            self.v = [torch.zeros_like(parameter) for parameter in self.parameters]

        def step(self) -> None:
            self.t += 1

            with torch.no_grad():
                for position, parameter in enumerate(self.parameters):
                    if parameter.grad is None:
                        continue

                    ...

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    # TODO: Return Adam's first step for each entry of grad, keeping its sign.
    #   The step is the amount subtracted from the parameter,
    #   lr * m_hat / (sqrt(v_hat) + eps), with m_hat and v_hat from step 1.
    #   Example: adam_first_step(torch.tensor([0.001, 1000.0, -3.0]), 0.01,
    #   1e-8) returns about tensor([ 0.0100,  0.0100, -0.0100]).
    def adam_first_step(
        grad: Tensor,
        lr: float,
        eps: float,
    ) -> Tensor:
        ...
    ''',
)

ADAM1_SOLUTION = source_file(
    '''
    # adam1.py, a solution for comparison

    import torch
    from torch import Tensor

    t = 3
    correction = 1 - 0.9 ** t


''' + INDENTED_ADAM + '''


    # On the first step m_hat is grad and v_hat is grad squared.
    def adam_first_step(
        grad: Tensor,
        lr: float,
        eps: float,
    ) -> Tensor:
        return lr * grad / (grad.abs() + eps)
    ''',
)


def check_my_adam(found: type) -> None:
    # An eps of 1e-8 can't tell sqrt(v_hat) + eps from sqrt(v_hat + eps), so the last round uses 0.1.
    for betas, eps in (((0.9, 0.999), 1e-8), ((0.8, 0.99), 1e-8), ((0.9, 0.999), 0.1)):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.01, "betas": betas, "eps": eps},
            reference=torch.optim.Adam,
            reference_arguments={"lr": 0.01, "betas": betas, "eps": eps},
            name="MyAdam",
        )


def check_adam1(checker: Checker) -> None:
    checker.test_method(
        class_name="MyAdam",
        method="step",
        description="updates the parameters like torch.optim.Adam",
        run=check_my_adam,
    )
    check_on_inputs(
        checker=checker,
        function="adam_first_step",
        description="gives the first step, the learning rate times the sign",
        inputs=(
            (torch.tensor([0.001, 1000.0, -3.0]), 0.01, 1e-8),
            (torch.tensor([0.5, 0.0, -2.0]), 0.001, 1e-8),
        ),
        reference=lambda grad, lr, eps: lr * grad / (grad.abs() + eps),
        expect=expect_tensor,
    )


ADAM1 = Exercise(
    name="adam1",
    chapter=ADAPTIVE,
    title="Adam from its paper",
    lesson=ADAM1_LESSON,
    source=ADAM1_SOURCE,
    solution=ADAM1_SOLUTION,
    hints={
        "MyAdam.step": "Write the five lines of Algorithm 1 with self.m[position], self.v[position], self.beta1, "
        "self.beta2, self.t, self.lr and self.eps, with parameter.grad as the gradient. Store the new m and v back "
        "at the same position, and subtract the step from the parameter with -=.",
        "adam_first_step": "On the first step, m_hat equals grad and v_hat equals grad squared. The square root of "
        "v_hat is grad.abs().",
    },
    introduces=(),
    check=check_adam1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# adam2
# ---------------------------------------------------------------------------

ADAM2_LESSON = prose(
    """
    Both averages start at 0. After one step, `m` is a tenth of the gradient,
    because `1 - beta1` is 0.1. And `v` is a thousandth of the squared
    gradient, because `1 - beta2` is 0.001. Without the hats, the first step
    would be `(lr * 0.1 * grad) / sqrt(0.001 * (grad ** 2))`, which is about
    3.2 times the learning rate.

    Suppose the gradient stays the same at every step. Then the uncorrected
    step is the learning rate times
    `(1 - (beta1 ** t)) / sqrt(1 - (beta2 ** t))`.
    That factor rises to about 6.6 near step 12, and it comes back down near
    1 only after a few thousand steps. So without the hats, Adam's early steps
    are several times too big. That happens right when the weights are random
    and the gradients are large.

    Here the names model of Section 3 was trained with Adam for 10 steps, with
    and without the hats, using three seeds. Its validation loss started near
    3.29.

    | Learning rate | With the hats | Without the hats |
    |---|---|---|
    | 0.003 | 2.89 to 3.00 | 3.14 to 3.36 |
    | 0.01 | 2.85 to 3.02 | 4.17 to 4.37 |
    | 0.03 | 3.44 to 3.93 | 5.28 to 6.89 |

    Without the hats, at rates of 0.01 and 0.03, the loss rose above its
    starting value. At 0.03 it rose with the hats too, so 0.03 is too high a
    rate for this model either way.

    The damage from those big early steps can last. Here a deeper names model,
    with 16 post norm blocks like the ones in residual4, was trained with Adam
    at 0.001. It ran for 1500 steps with a batch size of 64, using three
    seeds. The learning rate decreased smoothly to 0 by the end.

    | Adam | Validation loss |
    |---|---|
    | with the hats | 2.219 to 2.229 |
    | without the hats | 2.820 to 2.821 |

    Without the hats, the model got stuck at 2.82 on every seed, which is
    worse than the bigram model. The Adam paper warns about the same problem.
    Without the correction, the first steps are much larger, and training
    often diverges, which means the loss blows up. The fix costs only two
    divisions per step.
    """,
)

ADAM2_SOURCE = source_file(
    '''
    # adam2.py

    import math

    import torch
    from torch import Tensor

    first = 0.1 / math.sqrt(0.001)                   # 3.16, the first uncorrected step divided by lr


    # TODO: Suppose the gradient has been the same for t steps. Return the size
    #   of Adam's uncorrected step at step t divided by the learning rate, for
    #   the given betas.
    #   Example: uncorrected_ratio(0.9, 0.999, 1) returns about 3.162.
    def uncorrected_ratio(
        beta1: float,
        beta2: float,
        t: int,
    ) -> float:
        ...


    # TODO: Compute uncorrected_ratio for every t from 1 to steps. Return the t
    #   where it is largest, and that largest ratio, as a tuple.
    #   Example: peak_ratio(0.9, 0.999, 100) returns about (12, 6.569).
    def peak_ratio(
        beta1: float,
        beta2: float,
        steps: int,
    ) -> tuple[int, float]:
        ...


    # TODO: Finish the line that sets step. adam_step returns the new m, the new
    #   v, and step, the amount Adam subtracts from the parameter at step t,
    #   with an eps of 1e-8. The lines above apply the hats when corrected is
    #   True and skip them when it is False.
    #   Example: with g = torch.tensor([0.5, -2.0, 1.0]),
    #   adam_step(torch.zeros(3), torch.zeros(3), g, 1, 0.01, True) returns a
    #   step of about tensor([ 0.0100, -0.0100,  0.0100]).
    def adam_step(
        m: Tensor,
        v: Tensor,
        grad: Tensor,
        t: int,
        lr: float,
        corrected: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2

        if corrected:
            m_hat, v_hat = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        else:
            m_hat, v_hat = m, v

        step = ...

        return m, v, step
    ''',
)

ADAM2_SOLUTION = source_file(
    '''
    # adam2.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    first = 0.1 / math.sqrt(0.001)


    def uncorrected_ratio(
        beta1: float,
        beta2: float,
        t: int,
    ) -> float:
        return (1 - beta1 ** t) / math.sqrt(1 - beta2 ** t)


    def peak_ratio(
        beta1: float,
        beta2: float,
        steps: int,
    ) -> tuple[int, float]:
        ratios = [uncorrected_ratio(beta1, beta2, t) for t in range(1, steps + 1)]
        largest = max(ratios)

        return ratios.index(largest) + 1, largest


    def adam_step(
        m: Tensor,
        v: Tensor,
        grad: Tensor,
        t: int,
        lr: float,
        corrected: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2

        if corrected:
            m_hat, v_hat = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        else:
            m_hat, v_hat = m, v

        step = lr * m_hat / (v_hat.sqrt() + 1e-8)

        return m, v, step
    ''',
)


def reference_ratio(
    beta1: float,
    beta2: float,
    t: int,
) -> float:
    return (1 - beta1 ** t) / math.sqrt(1 - beta2 ** t)


def reference_peak(
    beta1: float,
    beta2: float,
    steps: int,
) -> tuple[int, float]:
    ratios = [reference_ratio(beta1, beta2, t) for t in range(1, steps + 1)]
    largest = max(ratios)

    return ratios.index(largest) + 1, largest


def reference_adam_step(
    m: Tensor,
    v: Tensor,
    grad: Tensor,
    t: int,
    lr: float,
    corrected: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad ** 2

    if corrected:
        m_hat, v_hat = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
    else:
        m_hat, v_hat = m, v

    return m, v, lr * m_hat / (v_hat.sqrt() + 1e-8)


def expect_peak(
    call: str,
    actual: object,
    expected: tuple[int, float],
) -> None:
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} should return two things, the step and the ratio, but it returns {actual!r}.")

    expect_value(
        call=f"the step from {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_close(
        call=f"the ratio from {call}",
        actual=actual[1],
        expected=expected[1],
        tolerance=1e-9,
    )


def check_adam2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="uncorrected_ratio",
        description="gives the uncorrected step divided by the learning rate",
        inputs=(
            (0.9, 0.999, 1),
            (0.9, 0.999, 12),
            (0.9, 0.999, 3000),
            (0.8, 0.99, 5),
        ),
        reference=reference_ratio,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-9,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="peak_ratio",
        description="finds the step where the uncorrected ratio is largest",
        inputs=(
            (0.9, 0.999, 100),
            (0.8, 0.99, 50),
        ),
        reference=reference_peak,
        expect=expect_peak,
    )
    check_on_inputs(
        checker=checker,
        function="adam_step",
        description="gives the new averages and the step, with or without the hats",
        inputs=(
            (torch.zeros(3), torch.zeros(3), torch.tensor([0.5, -2.0, 1.0]), 1, 0.01, True),
            (torch.zeros(3), torch.zeros(3), torch.tensor([0.5, -2.0, 1.0]), 1, 0.01, False),
            (small_tensor(530, 3) / 10, small_tensor(531, 3).abs() / 100, small_tensor(532, 3), 4, 0.001, True),
        ),
        reference=reference_adam_step,
        expect=expect_parts("the new m", "the new v", "the step"),
    )


ADAM2 = Exercise(
    name="adam2",
    chapter=ADAPTIVE,
    title="Why the bias correction",
    lesson=ADAM2_LESSON,
    source=ADAM2_SOURCE,
    solution=ADAM2_SOLUTION,
    hints={
        "uncorrected_ratio": "After t steps of the same gradient, m is (1 - (beta1 ** t)) times the gradient. And v "
        "is (1 - (beta2 ** t)) times its square. Divide the first factor by the square root of the second.",
        "peak_ratio": "Compute uncorrected_ratio for every t from 1 to steps and find the largest. Return its t, "
        "counted from 1, and the ratio itself.",
        "adam_step": "The step is lr times m_hat, divided by the square root of v_hat plus 1e-8.",
    },
    introduces=(),
    check=check_adam2,
)


# ---------------------------------------------------------------------------
# adamw1
# ---------------------------------------------------------------------------

ADAMW1_LESSON = prose(
    """
    Weight decay in Section 2 added `strength * (w ** 2)` to the loss, which
    adds `2 * strength * w` to the gradient. With SGD, that shrinks every
    weight toward 0 by the same fraction each step. PyTorch's
    `torch.optim.Adam(params, weight_decay=0.01)` does the same thing. It adds
    `weight_decay * w` to the gradient, so `weight_decay` plays the part of
    `2 * strength`. Adding the decay to the gradient like this is called L2.

    Adam then divides the whole gradient, decay term included, by its root
    mean square. So a weight with large gradients has its decay divided by a
    large number, and it gets almost no decay. A weight with small gradients
    has its decay divided by a small number, and it gets a lot. So with Adam,
    L2 does not shrink every weight by the same fraction the way it does with
    SGD.

    **AdamW**, from Loshchilov and Hutter in 2019, takes the decay out of the
    gradient. In PyTorch's version, each step first shrinks every weight by
    the fraction `lr * weight_decay`, which means multiplying it by
    `1 - lr * weight_decay`. Then it takes Adam's usual step with the gradient
    alone, without any decay term.

    ```python
    torch.optim.Adam(params, lr=0.001, weight_decay=0.01)     # decay in the gradient, L2
    torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)    # decay on the weight, decoupled
    ```

    Here two weights both start at 1, and their gradients are pure random
    noise. The noise has a standard deviation of 1 for the first weight and
    0.001 for the second. Each optimizer ran 1000 steps at a learning rate of
    0.001. The table shows where each weight ended, averaged over 20 seeds.

    | Gradient standard deviation | Adam, L2 0.01 | Adam, L2 0.1 | AdamW, decay 1.0 |
    |---|---|---|---|
    | 1 | 0.992 | 0.908 | 0.369 |
    | 0.001 | 0.262 | 0.258 | 0.370 |

    With L2, the weight with small gradients shrank to about a quarter, while
    the weight with large gradients barely changed. AdamW shrank both weights
    by the same fraction. Loshchilov and Hutter tested AdamW on ResNets with
    BatchNorm. There it let Adam do about as well as SGD with momentum on
    image classification.

    In the paper's own version, the shrink does not include Adam's learning
    rate. In PyTorch's version, the shrink is set by the product
    `lr * weight_decay`. So if you change the learning rate, you need to
    change `weight_decay` too to keep the same shrink. AdamW is the default
    optimizer for language models.

    A class can build on another class with `class MyAdamW(MyAdam):`, which
    is called inheritance. Inside it, `super().step()` runs the `step` method
    of the class it builds on.
    """,
)

ADAMW1_SOURCE = source_file(
    '''
    # adamw1.py

    import torch
    from torch import Tensor


    # MyAdam from adam1.
''' + INDENTED_ADAM + '''


    weight = torch.tensor([2.0, -4.0])
    shrunk = weight * (1 - 0.1 * 0.5)                # tensor([ 1.9000, -3.8000])


    # TODO: Return the gradient that Adam sees with L2 decay, which is grad plus
    #   weight_decay times parameter.
    #   Example: l2_gradient(torch.tensor([0.5, -1.0]), weight, 0.01) returns
    #   tensor([ 0.5200, -1.0400]).
    def l2_gradient(
        grad: Tensor,
        parameter: Tensor,
        weight_decay: float,
    ) -> Tensor:
        ...


    # TODO: Return parameter shrunk by the fraction lr * weight_decay.
    #   Example: decoupled_decay(weight, 0.1, 0.5) returns
    #   tensor([ 1.9000, -3.8000]), the same as shrunk.
    def decoupled_decay(
        parameter: Tensor,
        lr: float,
        weight_decay: float,
    ) -> Tensor:
        ...


    # TODO: Finish step. First shrink every parameter whose .grad is not None by
    #   the fraction self.lr * self.weight_decay, in place. Then take MyAdam's
    #   step.
    class MyAdamW(MyAdam):
        def __init__(
            self,
            parameters,
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
        ) -> None:
            super().__init__(parameters, lr, betas, eps)
            self.weight_decay = weight_decay

        def step(self) -> None:
            ...
    ''',
)

ADAMW1_SOLUTION = source_file(
    '''
    # adamw1.py, a solution for comparison

    import torch
    from torch import Tensor


''' + INDENTED_ADAM + '''


    weight = torch.tensor([2.0, -4.0])
    shrunk = weight * (1 - 0.1 * 0.5)


    def l2_gradient(
        grad: Tensor,
        parameter: Tensor,
        weight_decay: float,
    ) -> Tensor:
        return grad + weight_decay * parameter


    def decoupled_decay(
        parameter: Tensor,
        lr: float,
        weight_decay: float,
    ) -> Tensor:
        return parameter * (1 - lr * weight_decay)


    class MyAdamW(MyAdam):
        def __init__(
            self,
            parameters,
            lr: float = 0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
        ) -> None:
            super().__init__(parameters, lr, betas, eps)
            self.weight_decay = weight_decay

        def step(self) -> None:
            with torch.no_grad():
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter *= 1 - self.lr * self.weight_decay

            super().step()
    ''',
)


def check_my_adamw(found: type) -> None:
    for weight_decay in (0.1, 0.5):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.01, "weight_decay": weight_decay},
            reference=torch.optim.AdamW,
            reference_arguments={"lr": 0.01, "weight_decay": weight_decay},
            name="MyAdamW",
        )

    # A parameter left out of the loss has no gradient, and torch.optim.AdamW neither shrinks nor steps it.
    unused = nn.Parameter(torch.ones(2))
    used = nn.Parameter(torch.ones(2))
    optimizer = found([used, unused], lr=0.1, weight_decay=0.5)
    (used * 2).sum().backward()
    optimizer.step()

    if not torch.equal(unused.detach(), torch.ones(2)):
        raise Mismatch("MyAdamW.step shrank a parameter whose .grad is None. Shrink only the parameters that have a gradient.")


def check_adamw1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="l2_gradient",
        description="adds the decay times the weight to the gradient",
        inputs=(
            (torch.tensor([0.5, -1.0]), torch.tensor([2.0, -4.0]), 0.01),
            (small_tensor(540, 2, 3), small_tensor(541, 2, 3), 0.1),
        ),
        reference=lambda grad, parameter, weight_decay: grad + weight_decay * parameter,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="decoupled_decay",
        description="multiplies the weight by 1 - lr * weight_decay",
        inputs=(
            (torch.tensor([2.0, -4.0]), 0.1, 0.5),
            (small_tensor(542, 2, 3), 0.01, 0.1),
        ),
        reference=lambda parameter, lr, weight_decay: parameter * (1 - lr * weight_decay),
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="MyAdamW",
        method="step",
        description="updates the parameters like torch.optim.AdamW",
        run=check_my_adamw,
    )


ADAMW1 = Exercise(
    name="adamw1",
    chapter=ADAPTIVE,
    title="AdamW",
    lesson=ADAMW1_LESSON,
    source=ADAMW1_SOURCE,
    solution=ADAMW1_SOLUTION,
    hints={
        "l2_gradient": "Add weight_decay times parameter to grad.",
        "decoupled_decay": "Multiply parameter by 1 minus lr times weight_decay.",
        "MyAdamW.step": "Inside a with torch.no_grad() block, loop over self.parameters. Multiply each parameter "
        "that has a .grad by 1 - self.lr * self.weight_decay in place, using *=. Then call super().step().",
    },
    introduces=(),
    check=check_adamw1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# adamw2
# ---------------------------------------------------------------------------

ADAMW2_LESSON = prose(
    """
    Here the names model was trained for 5000 steps with a constant learning
    rate. Each optimizer was tried at three learning rates with two seeds, and
    the table shows its best rate.

    | Optimizer | Best rate | Validation loss |
    |---|---|---|
    | SGD | 0.1 | 2.301 to 2.304 |
    | SGD, momentum 0.9 | 0.01 | 2.305 to 2.307 |
    | Adam | 0.001 | 2.269 to 2.276 |

    Each best rate is 10 times smaller than the one above it in the table.
    Momentum's velocity grows to 10 times the gradient, so its best rate is 10
    times smaller than SGD's. Adam's steps are about the size of the learning
    rate, whatever the size of the gradient. So Adam's learning rate is
    roughly how much each weight changes per step.

    With batches of 32, momentum did about as well as plain SGD, as
    momentum2's noisy valley predicts. A batch of 32 gives noisy gradients,
    and the noise removes most of momentum's advantage. Zhang and others found
    the same on real models in 2019. In their experiments, momentum helped
    only with large batches.

    - AdamW: the default for language models and most new work, at rates from 0.0001 to 0.001 and weight decay from 0.01 to 0.1.
    - SGD with momentum 0.9: image models like ResNet, starting at a rate of 0.1 that is lowered during training, with weight decay near 0.0001.
    - Adam without decay: small models and quick experiments.

    GPT-3 used AdamW with betas of 0.9 and 0.95 and a weight decay of 0.1.

    Weight decay should apply to the weight matrices and embedding tables,
    which are the parameters with two or more dimensions. nanoGPT and most
    language model code do not decay biases or the weights of norm layers. To
    give parameters different settings, pass the optimizer a list of
    parameter groups. Each group is a dict with its `"params"` and its own
    settings.

    ```python
    torch.optim.AdamW([{"params": [w], "weight_decay": 0.1}, {"params": [b], "weight_decay": 0.0}], lr=0.001)
    ```
    """,
)

ADAMW2_SOURCE = source_file(
    '''
    # adamw2.py

    import torch
    import torch.nn as nn

    layer = nn.Linear(3, 2)
    dimensions = [parameter.dim() for parameter in layer.parameters()]    # [2, 1], the weight and the bias


    # TODO: Return a new optimizer for parameters with learning rate lr, of the
    #   kind that name picks. "sgd" is plain SGD, "momentum" is SGD with
    #   momentum 0.9, "adam" is Adam and "adamw" is AdamW with weight decay
    #   0.01.
    def make_optimizer(
        name: str,
        parameters,
        lr: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Return a list of two parameter groups for model. The first holds the
    #   parameters with two or more dimensions and has weight_decay. The second
    #   holds the rest and has a weight decay of 0.0. Each group is a dict with
    #   the keys "params" and "weight_decay", as in the lesson.
    def decay_groups(
        model: nn.Module,
        weight_decay: float,
    ) -> list[dict]:
        ...


    # TODO: results maps each (optimizer name, rate) pair to its validation
    #   loss. Return the pair with the lowest loss.
    #   Example: best_setting({("sgd", 0.3): 2.31, ("sgd", 0.1): 2.30}) returns
    #   ("sgd", 0.1).
    def best_setting(results: dict[tuple[str, float], float]) -> tuple[str, float]:
        ...
    ''',
)

ADAMW2_SOLUTION = source_file(
    '''
    # adamw2.py, a solution for comparison

    import torch
    import torch.nn as nn

    layer = nn.Linear(3, 2)
    dimensions = [parameter.dim() for parameter in layer.parameters()]


    def make_optimizer(
        name: str,
        parameters,
        lr: float,
    ) -> torch.optim.Optimizer:
        if name == "sgd":
            return torch.optim.SGD(parameters, lr=lr)

        if name == "momentum":
            return torch.optim.SGD(parameters, lr=lr, momentum=0.9)

        if name == "adam":
            return torch.optim.Adam(parameters, lr=lr)

        return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)


    def decay_groups(
        model: nn.Module,
        weight_decay: float,
    ) -> list[dict]:
        decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]

        return [{"params": decayed, "weight_decay": weight_decay}, {"params": kept, "weight_decay": 0.0}]


    def best_setting(results: dict[tuple[str, float], float]) -> tuple[str, float]:
        return min(results, key=results.get)
    ''',
)


def check_make_optimizer(found: object) -> None:
    wanted = {
        "sgd": (torch.optim.SGD, {"momentum": 0}),
        "momentum": (torch.optim.SGD, {"momentum": 0.9}),
        "adam": (torch.optim.Adam, {"weight_decay": 0}),
        "adamw": (torch.optim.AdamW, {"weight_decay": 0.01}),
    }

    for name, (kind, settings) in wanted.items():
        call = f"make_optimizer({name!r}, a Linear layer's parameters, 0.05)"
        optimizer = found(name, nn.Linear(3, 2).parameters(), 0.05)

        if type(optimizer) is not kind:
            raise Mismatch(f"{call} is {type(optimizer).__name__}, but it should be torch.optim.{kind.__name__}.")

        group = optimizer.param_groups[0]

        if group["lr"] != 0.05:
            raise Mismatch(f"{call} has a learning rate of {group['lr']}, but it should be 0.05.")

        for key, value in settings.items():
            if group[key] != value:
                raise Mismatch(f"{call} has {key} {group[key]}, but it should be {value}.")


def small_model(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Embedding(5, 3), nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))


def check_decay_groups(found: object) -> None:
    model = small_model(550)
    call = "decay_groups on an Embedding, a Linear, a LayerNorm and a Linear, with weight decay 0.1"
    groups = found(model, 0.1)

    if not isinstance(groups, list) or len(groups) != 2 or not all(isinstance(group, dict) for group in groups):
        raise Mismatch(f"{call} should return a list of two dicts.")

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    wanted = (
        ([name for name, parameter in model.named_parameters() if parameter.dim() >= 2], 0.1),
        ([name for name, parameter in model.named_parameters() if parameter.dim() < 2], 0.0),
    )

    for position, (group, (expected, decay)) in enumerate(zip(groups, wanted)):
        got = [names.get(id(parameter), "something else") for parameter in group.get("params", [])]

        if got != expected:
            raise Mismatch(f"Group {position} of {call} should hold {', '.join(expected)}, but it holds {', '.join(got) or 'nothing'}.")

        if group.get("weight_decay") != decay:
            raise Mismatch(f"Group {position} of {call} should have a weight_decay of {decay}.")


def check_adamw2(checker: Checker) -> None:
    checker.test(
        function="make_optimizer",
        description="makes each optimizer with its settings",
        run=check_make_optimizer,
    )
    checker.test(
        function="decay_groups",
        description="decays the parameters with two or more dimensions only",
        run=check_decay_groups,
    )
    check_on_inputs(
        checker=checker,
        function="best_setting",
        description="picks the setting with the lowest loss",
        inputs=(
            ({("sgd", 0.1): 2.301, ("momentum", 0.01): 2.305, ("adam", 0.001): 2.276, ("adamw", 0.001): 2.278},),
            ({("sgd", 0.3): 2.31, ("sgd", 0.1): 2.30},),
        ),
        reference=lambda results: min(results, key=results.get),
        expect=expect_value,
    )


ADAMW2 = Exercise(
    name="adamw2",
    chapter=ADAPTIVE,
    title="Which optimizer when",
    lesson=ADAMW2_LESSON,
    source=ADAMW2_SOURCE,
    solution=ADAMW2_SOLUTION,
    hints={
        "make_optimizer": "Write one if per name. Each returns the right torch.optim class called with parameters "
        "and lr=lr, plus momentum=0.9 or weight_decay=0.01 where the name asks for it.",
        "decay_groups": "Split model.parameters() by .dim() into two lists, then return two dicts, each with "
        "\"params\" and \"weight_decay\".",
        "best_setting": "Call min on results, which loops over its keys, with key=results.get so the keys are "
        "compared by their loss.",
    },
    introduces=(),
    check=check_adamw2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """CPU results move in the third decimal with the thread count, so every measurement runs on 4."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def train_names_rows(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: Tensor,
    Y: Tensor,
    steps: int,
    generator: torch.Generator,
) -> None:
    model.train()

    for _ in range(steps):
        rows = torch.randint(0, len(X), (32,), generator=generator)
        loss = F.cross_entropy(model(X[rows]), Y[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


BEST_RATE_GRID: dict[str, tuple[float, ...]] = {
    "SGD": (0.03, 0.1, 0.3),
    "SGD, momentum 0.9": (0.003, 0.01, 0.03),
    "Adam": (0.0003, 0.001, 0.003),
}


def optimizer_named(
    name: str,
    parameters: list[nn.Parameter],
    lr: float,
) -> torch.optim.Optimizer:
    if name == "SGD":
        return torch.optim.SGD(parameters, lr=lr)

    if name == "SGD, momentum 0.9":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)

    return torch.optim.Adam(parameters, lr=lr)


def measure_best_rates(seeds: tuple[int, ...] = (0, 1)) -> dict[tuple[str, float], list[float]]:
    """adamw2's first table: the names model for 5000 steps at batch 32 and a fixed rate, each optimizer at each rate of
    its grid. The validation loss per seed."""
    X, Y = context_sets()["train"]
    validation_x, validation_y = context_sets()["validation"]
    results: dict[tuple[str, float], list[float]] = {}

    with pinned_threads():
        for name, rates in BEST_RATE_GRID.items():
            for rate in rates:
                for seed in seeds:
                    model = names_model(seed)
                    train_names_rows(
                        model=model,
                        optimizer=optimizer_named(name, list(model.parameters()), rate),
                        X=X,
                        Y=Y,
                        steps=5000,
                        generator=torch.Generator().manual_seed(seed + 1),
                    )
                    results.setdefault((name, rate), []).append(reference_validation_loss(model, validation_x, validation_y))

    return results


def measure_decay_by_gradient(seeds: tuple[int, ...] = tuple(range(20))) -> dict[tuple[str, float], list[float]]:
    """adamw1's two weight table: two weights start at 1.0 and get only noise for a gradient, of size 1 and
    of size 0.001, for 1000 steps at a rate of 0.001. Where each weight ends, averaged over the seeds, under Adam with
    L2 and under AdamW."""
    results: dict[tuple[str, float], list[float]] = {}

    with pinned_threads():
        for name, strength in (("Adam", 0.0), ("Adam with L2", 0.01), ("Adam with L2", 0.1), ("AdamW", 1.0)):
            finals = []

            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                weights = nn.Parameter(torch.ones(2))
                sizes = torch.tensor([1.0, 0.001])
                optimizer = (
                    torch.optim.AdamW([weights], lr=0.001, weight_decay=strength)
                    if name == "AdamW"
                    else torch.optim.Adam([weights], lr=0.001, weight_decay=strength)
                )

                for _ in range(1000):
                    optimizer.zero_grad()
                    weights.grad = sizes * torch.randn(2, generator=generator)
                    optimizer.step()

                finals.append(weights.detach().clone())

            results[(name, strength)] = torch.stack(finals).mean(dim=0).tolist()

    return results


class HatsAdam:
    """Adam from Algorithm 1 with the hats switched on or off, for adam2's runs. It reads its rate from param_groups,
    so a schedule can set it."""

    def __init__(
        self,
        parameters: Iterator[nn.Parameter],
        lr: float,
        corrected: bool,
    ) -> None:
        self.parameters = list(parameters)
        self.param_groups = [{"lr": lr}]
        self.corrected = corrected
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]

    def step(self) -> None:
        self.t += 1
        lr = self.param_groups[0]["lr"]

        with torch.no_grad():
            for position, parameter in enumerate(self.parameters):
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                self.m[position] = self.beta1 * self.m[position] + (1 - self.beta1) * grad
                self.v[position] = self.beta2 * self.v[position] + (1 - self.beta2) * grad ** 2
                m_hat, v_hat = self.m[position], self.v[position]

                if self.corrected:
                    m_hat = m_hat / (1 - self.beta1 ** self.t)
                    v_hat = v_hat / (1 - self.beta2 ** self.t)

                parameter -= lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None


def measure_bias_correction(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[tuple[float, bool], list[float]]:
    """adam2's first table: the names model after 10 steps of Adam at batch 32, with and without the hats, at each
    rate. The validation loss per seed, from a start near log(27)."""
    X, Y = context_sets()["train"]
    validation_x, validation_y = context_sets()["validation"]
    results: dict[tuple[float, bool], list[float]] = {}

    with pinned_threads():
        for rate in (0.003, 0.01, 0.03):
            for corrected in (True, False):
                for seed in seeds:
                    model = names_model(seed)
                    train_names_rows(
                        model=model,
                        optimizer=HatsAdam(model.parameters(), rate, corrected),
                        X=X,
                        Y=Y,
                        steps=10,
                        generator=torch.Generator().manual_seed(seed + 1),
                    )
                    results.setdefault((rate, corrected), []).append(reference_validation_loss(model, validation_x, validation_y))

    return results


def measure_deep_bias_correction(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[bool, list[float]]:
    """adam2's second table: names2's model with 16 post norm blocks, trained with Adam at 0.001 for 1500 steps at
    batch 64, the rate falling along a cosine to 0, with and without the hats. The validation loss per seed."""
    # The deep model and its schedule live with warmup1, which imports this module, so they load here.
    from .schedules import WARMUP_STEPS, DeepNamesModel, names_validation, reference_cosine_rate, train_names_on_schedule

    results: dict[bool, list[float]] = {}

    with pinned_threads():
        for corrected in (True, False):
            for seed in seeds:
                torch.manual_seed(seed)
                model = DeepNamesModel(16)
                train_names_on_schedule(
                    model=model,
                    optimizer=HatsAdam(model.parameters(), 0.001, corrected),
                    rate_at=lambda step: reference_cosine_rate(0.001, step, WARMUP_STEPS, 0.0),
                    steps=WARMUP_STEPS,
                    batch=64,
                    generator=torch.Generator().manual_seed(seed + 1),
                )
                results.setdefault(corrected, []).append(names_validation(model))

    return results


def steps_to_settle(
    make: Callable[[list[Tensor], float], torch.optim.Optimizer],
    lr: float,
    limit: int = 3000,
) -> int | None:
    """Steps on momentum2's valley from (-10, 1) until the loss stays below 0.001 for the rest of limit, or None."""
    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = make([point], lr)
    last_above = -1

    for step in range(limit):
        loss = point[0] ** 2 + 100 * point[1] ** 2

        if not math.isfinite(loss.item()):
            return None

        if loss.item() >= 0.001:
            last_above = step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return last_above + 1 if last_above + 1 < limit else None


def measure_valley_settling() -> dict[str, tuple[int, float]]:
    """rmsprop1's numbers: on momentum2's valley, each optimizer's fewest steps to settle below 0.001 over a grid of
    rates, and the rate that gave them."""
    even = [round(0.0005 * k, 4) for k in range(1, 21)]
    spread = [round(10 ** (-3 + 3 * k / 30), 5) for k in range(31)]
    makers: dict[str, tuple[Callable[[list[Tensor], float], torch.optim.Optimizer], list[float]]] = {
        "SGD": (lambda parameters, lr: torch.optim.SGD(parameters, lr=lr), even),
        "SGD, momentum 0.9": (lambda parameters, lr: torch.optim.SGD(parameters, lr=lr, momentum=0.9), even),
        "RMSprop": (lambda parameters, lr: torch.optim.RMSprop(parameters, lr=lr), spread),
    }
    results: dict[str, tuple[int, float]] = {}

    with pinned_threads():
        for name, (make, rates) in makers.items():
            settled = [(steps, rate) for rate in rates if (steps := steps_to_settle(make, rate)) is not None]
            results[name] = min(settled)

    return results
