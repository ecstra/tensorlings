"""25_adaptive, RMSProp, Adam and AdamW built from their papers and matched with torch.optim, and when to use which."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Iterator

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
from .names import CHARACTERS, ReferenceNamesMLP, context_sets, name_splits, names_model, reference_contexts, reference_validation_loss
from .sgd import check_optimizer_class

ADAPTIVE = Chapter(
    folder="25_adaptive",
    title="Adaptive optimizers",
    readme=source_file(
        """
        # Adaptive optimizers

        An adaptive optimizer gives every parameter a step size of its own,
        worked out from that parameter's gradients. This chapter builds
        RMSProp, Adam from Algorithm 1 of its paper and AdamW, matches each
        with PyTorch's, shows what Adam's bias correction is for, and sets out
        which optimizer and learning rate fits which model.

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
    SGD takes one learning rate for every parameter. A parameter whose
    gradients are always large takes big steps, and one whose gradients are
    small barely moves. **RMSProp**, from Hinton's lecture notes of 2012,
    divides each parameter's step by a running root mean square of its own
    gradients.

    ```text
    s = alpha * s + (1 - alpha) * g ** 2
    parameter = parameter - lr * g / (sqrt(s) + eps)
    ```

    Here `g` is the gradient and `s` the running average of its square.

    Each parameter then steps about the learning rate, whatever the size of
    its gradients. In momentum2's valley, the steep direction's large
    gradients get divided by their large average, so the bounce shrinks while
    the gentle direction speeds up.

    `alpha` is 0.99 in PyTorch, an average over about the last 100 steps. The
    `eps` of `1e-8` keeps an average of 0 from dividing by 0. The averages
    start at 0, one per parameter, and `torch.optim.RMSprop(params, lr=0.01)`
    does all this.

    Starting at 0 has a side effect. After one step the average holds only
    `1 - alpha` of the first gradient's square, so the first step comes out
    about 10 times the learning rate.
    """,
)

RMSPROP1_SOURCE = source_file(
    '''
    # rmsprop1.py

    import torch
    from torch import Tensor

    grad = torch.tensor([0.001, 1000.0])
    sizes = grad.abs()                               # each gradient's size, without its sign


    # TODO: Return the new running average of the squared gradient, the old one
    #   times alpha plus 1 - alpha times grad squared.
    def square_average_update(
        square_average: Tensor,
        grad: Tensor,
        alpha: float,
    ) -> Tensor:
        ...


    # TODO: Finish the two lines with ... in step, so each parameter's average
    #   is updated with square_average_update, and the parameter moves against
    #   its gradient divided by the root of the average plus self.eps, scaled
    #   by self.lr.
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


    # TODO: Return the size of RMSProp's first step for each entry of grad,
    #   without its sign. That is the step taken from an average of 0.
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


    # The first average is (1 - alpha) * grad ** 2, so the step is about lr / sqrt(1 - alpha).
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
    for alpha in (0.99, 0.9):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.01, "alpha": alpha},
            reference=torch.optim.RMSprop,
            reference_arguments={"lr": 0.01, "alpha": alpha},
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
        description="moves the parameters like torch.optim.RMSprop",
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
        "square_average_update": "alpha times square_average, plus (1 - alpha) times grad ** 2.",
        "MyRMSprop.step": "The average is square_average_update of the old one, parameter.grad and self.alpha. The "
        "parameter moves by self.lr times its grad, divided by the root of the new average plus self.eps.",
        "first_step": "The first average is (1 - alpha) * grad ** 2. Divide lr times the size of grad, "
        "grad.abs(), by its root plus eps.",
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
    **Adam**, from Kingma and Ba in 2015, puts momentum and RMSProp together.
    It keeps two running averages for each parameter, `m` of the gradient and
    `v` of its square. It steps by `m` divided by the root of `v`. Here is
    Algorithm 1 of the paper, for one step, with `t` counting the steps from 1.

    ```text
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    parameter = parameter - lr * m_hat / (sqrt(v_hat) + eps)
    ```

    `m` is momentum's velocity, as an average, and `v` is RMSProp's. The two
    hats undo the pull toward 0 that starting both averages at 0 gives, and
    adam2 is about them. The paper's defaults, which PyTorch keeps, are a
    `beta1` of 0.9, a `beta2` of 0.999 and an `eps` of `1e-8`.

    On the first step the hats give back the gradient and its square exactly.
    So the first step is the learning rate times the gradient's sign, about
    the same size for every parameter.

    `torch.optim.Adam(params, lr=0.001)` does this. A learning rate of 0.001
    to 0.0003 is the usual start.
    """,
)

ADAM1_SOURCE = source_file(
    '''
    # adam1.py

    import torch
    from torch import Tensor

    t = 3
    correction = 1 - 0.9 ** t                        # 0.271, the share m has built up after 3 steps


    # TODO: Finish step, so it runs Algorithm 1 of the paper for each parameter
    #   that has a gradient, keeping m and v in self.m and self.v by position.
    #   self.t counts the steps from 1.
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


    # TODO: Return Adam's first step for each entry of grad, with its sign.
    #   That is the amount taken away from the parameter.
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
    for betas in ((0.9, 0.999), (0.8, 0.99)):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.01, "betas": betas},
            reference=torch.optim.Adam,
            reference_arguments={"lr": 0.01, "betas": betas},
            name="MyAdam",
        )


def check_adam1(checker: Checker) -> None:
    checker.test_method(
        class_name="MyAdam",
        method="step",
        description="moves the parameters like torch.optim.Adam",
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
        "by position, and take the step with -=.",
        "adam_first_step": "On the first step m_hat is grad and v_hat is grad squared, whose root is grad.abs().",
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
    Both averages start at 0. After one step, `m` holds a tenth of the
    gradient and `v` a thousandth of its square. Without the hats, the first
    step would be `0.1 * grad / sqrt(0.001 * grad ** 2)`, about 3.2 times the
    learning rate.

    For a steady gradient, the uncorrected step is the learning rate times
    `(1 - beta1 ** t) / sqrt(1 - beta2 ** t)`. That climbs to about 6.6 near
    step 12, and falls back near 1 only after a few thousand steps. So without
    the hats, Adam's early steps are several times too big, right when the
    weights are random and the gradients large.

    Here the names model of Section 3 trained with Adam, with and without the
    hats, starting from a validation loss of 3.29.

    | Learning rate | 10 steps, with | 10 steps, without | 2000 steps, with | 2000 steps, without |
    |---|---|---|---|---|
    | 0.003 | 2.89 | 3.24 | 2.330 | 2.332 |
    | 0.01 | 2.85 | 4.37 | 2.347 | 2.354 |
    | 0.03 | 3.77 | 5.96 | 2.437 | 2.461 |

    Without the hats, the loss at 0.01 and 0.03 jumped above where it started
    before it recovered. The paper warns of the same, steps much larger at
    first. The fix costs two divisions.
    """,
)

ADAM2_SOURCE = source_file(
    '''
    # adam2.py

    import math

    import torch
    from torch import Tensor

    first = 0.1 / math.sqrt(0.001)                   # 3.16, the first uncorrected step over lr


    # TODO: Return how many times the learning rate an uncorrected Adam step is
    #   after t steps of the same gradient, for the given betas.
    def uncorrected_ratio(
        beta1: float,
        beta2: float,
        t: int,
    ) -> float:
        ...


    # TODO: Return the step t, from 1 to steps, where uncorrected_ratio is
    #   largest, and that ratio.
    def peak_ratio(
        beta1: float,
        beta2: float,
        steps: int,
    ) -> tuple[int, float]:
        ...


    # TODO: Finish the line that sets step, so this returns the new m, the new v
    #   and the amount Adam takes away from a parameter at step t, with the hats
    #   when corrected is True and without them when it is False.
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
        raise Mismatch(f"{call} should give two things, the step and the ratio, but it gives {actual!r}.")

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
        description="gives the uncorrected step over the learning rate",
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
        description="finds the step where the uncorrected ratio peaks",
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
        "uncorrected_ratio": "After t steps of the same gradient, m is (1 - beta1 ** t) of it and v is "
        "(1 - beta2 ** t) of its square. Divide the first by the square root of the second.",
        "peak_ratio": "Work out uncorrected_ratio for every t from 1 to steps, find the largest, and give its t, "
        "counted from 1, and the ratio.",
        "adam_step": "The step is lr times m_hat, divided by the root of v_hat plus 1e-8.",
    },
    introduces=(),
    check=check_adam2,
)


# ---------------------------------------------------------------------------
# adamw1
# ---------------------------------------------------------------------------

ADAMW1_LESSON = prose(
    """
    Weight decay in Section 2 added `strength * w ** 2` to the loss, so its
    gradient added `2 * strength * w`. With SGD, that pulls every weight
    toward 0 by the same fraction each step. PyTorch's
    `torch.optim.Adam(params, weight_decay=0.01)` does the same, adding
    `weight_decay` times the weight to the gradient. So `weight_decay` plays
    the part of `2 * strength`. This is called L2.

    Adam then divides the whole gradient by its root mean square, the decay
    included. A weight with large gradients gets almost no decay, and one with
    small gradients gets a lot. So with Adam, L2 is not the steady pull it is
    with SGD.

    Loshchilov and Hutter's **AdamW**, from 2019, takes the decay out of the
    gradient. In PyTorch's version, each step first shrinks every weight by
    the fraction `lr * weight_decay`, then takes Adam's step on the plain
    gradient.

    ```python
    torch.optim.Adam(params, lr=0.001, weight_decay=0.01)     # decay in the gradient, L2
    torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)    # decay on the weight, decoupled
    ```

    Their paper found that decoupling the decay improved Adam's
    generalization enough to compete with SGD and momentum on image
    classification. adamw2 shows it on the names model. Their own version
    leaves Adam's
    learning rate out of the shrink, so the best decay stays put when the
    rate changes. With PyTorch's, the product `lr * weight_decay` is what
    matters, so a new rate can call for a new decay. AdamW is the default for
    language models.

    A class can build on another with `class MyAdamW(MyAdam):`, and
    `super().step()` runs the step of the class it builds on.
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


    # TODO: Return the gradient Adam sees with L2 decay, grad plus weight_decay
    #   times parameter.
    def l2_gradient(
        grad: Tensor,
        parameter: Tensor,
        weight_decay: float,
    ) -> Tensor:
        ...


    # TODO: Return parameter shrunk by the fraction lr * weight_decay.
    def decoupled_decay(
        parameter: Tensor,
        lr: float,
        weight_decay: float,
    ) -> Tensor:
        ...


    # TODO: Finish step, so it shrinks every parameter that has a gradient by the
    #   fraction self.lr * self.weight_decay, in place, then takes MyAdam's step.
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
        description="shrinks the weight by lr times the decay",
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
        description="moves the parameters like torch.optim.AdamW",
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
        "l2_gradient": "grad plus weight_decay times parameter.",
        "decoupled_decay": "Multiply parameter by 1 minus lr times weight_decay.",
        "MyAdamW.step": "Inside a with torch.no_grad() block, loop over self.parameters. Multiply each one with a "
        ".grad in place with *= by 1 - self.lr * self.weight_decay. Then call super().step().",
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
    Here the names model trained for 5000 steps at a learning rate that never
    changed. Each optimizer ran at three rates on two seeds, and the table
    gives its best.

    | Optimizer | Best rate | Validation loss |
    |---|---|---|
    | SGD | 0.1 | 2.301 to 2.304 |
    | SGD, momentum 0.9 | 0.01 | 2.305 to 2.307 |
    | Adam | 0.001 | 2.269 to 2.276 |

    The best rates sit 10 times apart. Momentum's velocity builds to 10 times
    the gradient, so its rate comes out 10 times smaller. Adam's steps are
    about the rate itself, whatever the gradient, so its rate is a distance in
    the weights' own units.

    Momentum matched plain SGD at these batches of 32. Zhang and others found
    the same in 2019: momentum gives no benefit at small batch sizes, and it
    speeds training up at large ones.

    Weight decay needs a model with room to memorize. So here a wider names
    model, with embeddings of 24 and 1000 hidden units, trained on the first
    2,000 names alone. It ran at 0.001 for 5000 steps on three seeds.

    | Optimizer | Training loss | Validation loss |
    |---|---|---|
    | Adam | 1.975 to 1.990 | 2.383 to 2.412 |
    | AdamW, decay 0.1 | 2.004 to 2.018 | 2.367 to 2.394 |
    | AdamW, decay 0.3 | 2.052 to 2.073 | 2.343 to 2.372 |

    On every seed, more decay gave up training loss and won more back on new
    names, as Loshchilov and Hutter found.

    - AdamW: the default for language models and most new work, at rates from 0.0001 to 0.001 and weight decay from 0.01 to 0.1.
    - SGD with momentum 0.9: image models like ResNet, at a rate of 0.1 that a schedule lowers, with weight decay near 0.0001.
    - Adam without decay: small models and quick tries.

    GPT-3 used AdamW with betas of 0.9 and 0.95 and a decay of 0.1.

    Decay belongs on the weight matrices and embedding tables, the parameters
    with two or more dimensions. nanoGPT and most language model code leave
    biases and norm weights out of it. An optimizer takes a list of groups,
    each a dict with its `"params"` and its own settings.

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


    # TODO: Return the optimizer named by name for parameters at lr. "sgd" is
    #   plain SGD, "momentum" is SGD with momentum 0.9, "adam" is Adam and
    #   "adamw" is AdamW with weight decay 0.01.
    def make_optimizer(
        name: str,
        parameters,
        lr: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Return two parameter groups for model, the parameters with two or
    #   more dimensions with weight_decay, then the rest with a weight decay
    #   of 0.0.
    def decay_groups(
        model: nn.Module,
        weight_decay: float,
    ) -> list[dict]:
        ...


    # TODO: Return the (optimizer name, rate) pair from results with the lowest
    #   validation loss. results maps each pair to its loss.
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
        raise Mismatch(f"{call} should give a list of two dicts.")

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
        "make_optimizer": "One if per name, each returning the torch.optim class called with parameters and lr=lr, "
        "plus momentum=0.9 or weight_decay=0.01 where the name asks for it.",
        "decay_groups": "Split model.parameters() by .dim() into two lists, then return two dicts, each with "
        "\"params\" and \"weight_decay\".",
        "best_setting": "min over the keys of results, with key=results.get to compare by the loss.",
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


# Few enough names that a wide model overfits them, so weight decay has work to do.
DECAY_NAMES = 2000


def measure_decay(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[tuple[float, float]]]:
    """adamw2's second table: a names model with embeddings of 24 and 1000 hidden units on the first 2,000 training
    names, for 5000 steps at batch 32 and a rate of 0.001. Adam against AdamW, the training then the validation loss
    per seed."""
    names = list(name_splits()["train"])[:DECAY_NAMES]
    X, Y = reference_contexts(names, CHARACTERS, 3)
    validation_x, validation_y = context_sets()["validation"]
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for label, decay in (("Adam", 0.0), ("AdamW, decay 0.1", 0.1), ("AdamW, decay 0.3", 0.3)):
            for seed in seeds:
                torch.manual_seed(seed)
                model = ReferenceNamesMLP(27, 24, 3, 1000)
                decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
                kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]
                optimizer = (
                    torch.optim.Adam(model.parameters(), lr=0.001)
                    if decay == 0.0
                    else torch.optim.AdamW([{"params": decayed, "weight_decay": decay}, {"params": kept, "weight_decay": 0.0}], lr=0.001)
                )
                train_names_rows(
                    model=model,
                    optimizer=optimizer,
                    X=X,
                    Y=Y,
                    steps=5000,
                    generator=torch.Generator().manual_seed(seed + 1),
                )
                results.setdefault(label, []).append(
                    (
                        reference_validation_loss(model, X, Y),
                        reference_validation_loss(model, validation_x, validation_y),
                    ),
                )

    return results
