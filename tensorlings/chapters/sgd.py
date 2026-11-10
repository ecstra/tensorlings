"""24_sgd, the optimizer object, SGD built and matched, and momentum built and seen in a narrow valley."""

from __future__ import annotations

import random
import statistics

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    NOT_STARTED,
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .initialization import relative

SGD = Chapter(
    folder="24_sgd",
    title="SGD and momentum",
    readme=source_file(
        """
        # SGD and momentum

        Section 4 is about the update step, the rule that changes the
        parameters once the gradients are known. This chapter moves the update
        from Section 2 into an optimizer object and builds your own copy of
        PyTorch's SGD. Then it adds momentum and shows how momentum helps in a
        long, narrow valley.

        ## Further reading

        - [PyTorch, torch.optim](https://docs.pytorch.org/docs/2.14/optim.html)
        - [Goh, why momentum really works](https://distill.pub/2017/momentum/)
        - [Zhang and others, which algorithmic choices matter at which batch sizes](https://arxiv.org/abs/1907.04164)
        - [CS231n, parameter updates](https://cs231n.github.io/neural-networks-3/#update)
        """,
    ),
)


def seeded_line(seed: int) -> nn.Linear:
    torch.manual_seed(seed)

    return nn.Linear(3, 2)


def regression_data(seed: int) -> tuple[Tensor, Tensor]:
    return small_tensor(seed, 8, 3), small_tensor(seed + 1, 8, 2)


def mse_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    steps: int,
) -> float:
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def expect_same_parameters(
    call: str,
    mine: nn.Module,
    theirs: nn.Module,
) -> None:
    for (name, actual), expected in zip(mine.named_parameters(), theirs.parameters()):
        expect_tensor(
            call=f"the {name} after {call}",
            actual=actual.detach(),
            expected=expected.detach(),
        )


def check_optimizer_class(
    found: type,
    arguments: dict[str, float],
    reference: type,
    reference_arguments: dict[str, float],
    name: str,
) -> None:
    """Five steps of the learner's optimizer and PyTorch's on the same model and data must land on the same weights."""
    x, y = regression_data(500)
    mine, theirs = seeded_line(501), seeded_line(501)
    arguments_text = ", ".join(f"{key}={value}" for key, value in arguments.items())
    call = f"5 steps of {name}(model.parameters(), {arguments_text}) on a Linear layer"

    before = [parameter.detach().clone() for parameter in mine.parameters()]
    mse_steps(mine, found(mine.parameters(), **arguments), x, y, 5)

    if all(torch.equal(old, new.detach()) for old, new in zip(before, mine.parameters())):
        raise Mismatch(
            f"After {call} the weights did not change. Either {name}.step is {NOT_STARTED}, or it sets parameter = "
            "..., which only makes the loop variable refer to a new tensor and leaves the real parameter as it was. "
            "Change the parameter in place with -= or *= inside torch.no_grad().",
        )

    mse_steps(theirs, reference(theirs.parameters(), **reference_arguments), x, y, 5)
    expect_same_parameters(call, mine, theirs)


# ---------------------------------------------------------------------------
# sgd1
# ---------------------------------------------------------------------------

SGD1_LESSON = prose(
    """
    Every training loop so far updated the parameters by hand. PyTorch puts
    that update in an **optimizer**, an object that holds the parameters and
    knows how to update them.

    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

    `zero_grad` clears every gradient. `backward` computes the new gradients
    and stores them in each parameter's `.grad`. `step` updates every
    parameter. SGD, short for stochastic gradient descent, is the update from
    Section 2. It replaces each parameter with `parameter - lr * grad`, the
    parameter minus the learning rate times its gradient. It is called
    stochastic, which means random, because each minibatch gives a noisy
    estimate of the gradient.

    Switching to another optimizer changes only the line that creates it. So
    the rest of this section builds several optimizers one at a time and puts
    each one into this same loop.

    `model.parameters()` returns a generator, which can be looped over only
    once. So an optimizer stores the parameters in a list. A parameter that
    was not used to compute the loss has a `.grad` of `None`, and the step
    skips it. `zero_grad` sets every `.grad` to `None`, which is also what
    PyTorch's own `zero_grad` does.
    """,
)

SGD1_SOURCE = source_file(
    '''
    # sgd1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameters = list(model.parameters())            # the weight and the bias


    # TODO: Finish step and zero_grad. step changes every parameter whose .grad
    #   is not None to parameter - self.lr * parameter.grad, in place. It skips
    #   parameters whose .grad is None. zero_grad sets the .grad of every
    #   parameter to None.
    class MySGD:
        def __init__(
            self,
            parameters,
            lr: float,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr

        def step(self) -> None:
            ...

        def zero_grad(self) -> None:
            ...


    # TODO: Train model for steps steps with optimizer. Each step computes the
    #   mean squared error of model(x) against y, clears the gradients, calls
    #   backward on the loss and then calls optimizer.step(). Return the loss of
    #   the last step as a float, the one computed before its optimizer.step().
    #   Don't compute a new loss after the loop.
    def train(
        model: nn.Module,
        optimizer,
        x: Tensor,
        y: Tensor,
        steps: int,
    ) -> float:
        ...
    ''',
)

SGD1_SOLUTION = source_file(
    '''
    # sgd1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameters = list(model.parameters())


    class MySGD:
        def __init__(
            self,
            parameters,
            lr: float,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr

        def step(self) -> None:
            with torch.no_grad():
                for parameter in self.parameters:
                    if parameter.grad is None:
                        continue

                    parameter -= self.lr * parameter.grad

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    def train(
        model: nn.Module,
        optimizer,
        x: Tensor,
        y: Tensor,
        steps: int,
    ) -> float:
        for _ in range(steps):
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)


def check_my_sgd_step(found: type) -> None:
    check_optimizer_class(
        found=found,
        arguments={"lr": 0.05},
        reference=torch.optim.SGD,
        reference_arguments={"lr": 0.05},
        name="MySGD",
    )

    # A parameter left out of the loss has no gradient, and the step has to skip it.
    unused = nn.Parameter(torch.ones(2))
    used = nn.Parameter(torch.ones(2))
    optimizer = found([used, unused], lr=0.5)
    (used * 2).sum().backward()
    optimizer.step()

    if not torch.equal(unused.detach(), torch.ones(2)):
        raise Mismatch("MySGD.step changed a parameter whose .grad is None. Skip those parameters.")


def check_my_sgd_zero_grad(found: type) -> None:
    layer = seeded_line(502)
    optimizer = found(layer.parameters(), lr=0.1)
    layer(torch.ones(1, 3)).sum().backward()
    optimizer.zero_grad()

    for name, parameter in layer.named_parameters():
        if parameter.grad is not None and parameter.grad.abs().max().item() > 0:
            raise Mismatch(f"After MySGD.zero_grad the {name} still has a gradient. Set every .grad to None.")


def check_train(found: object) -> None:
    for steps, lr in ((5, 0.05), (20, 0.01)):
        x, y = regression_data(503 + steps)
        mine, theirs = seeded_line(505), seeded_line(505)
        expect_close(
            call=f"train with torch.optim.SGD at lr={lr} for {steps} steps on a Linear layer",
            actual=found(mine, torch.optim.SGD(mine.parameters(), lr=lr), x, y, steps),
            expected=mse_steps(theirs, torch.optim.SGD(theirs.parameters(), lr=lr), x, y, steps),
            tolerance=1e-5,
        )


def check_sgd1(checker: Checker) -> None:
    checker.test_method(
        class_name="MySGD",
        method="step",
        description="updates the parameters exactly like torch.optim.SGD",
        run=check_my_sgd_step,
    )
    checker.test_method(
        class_name="MySGD",
        method="zero_grad",
        description="clears every gradient",
        run=check_my_sgd_zero_grad,
    )
    checker.test(
        function="train",
        description="runs zero_grad, backward and step every step",
        run=check_train,
    )


SGD1 = Exercise(
    name="sgd1",
    chapter=SGD,
    title="The optimizer object",
    lesson=SGD1_LESSON,
    source=SGD1_SOURCE,
    solution=SGD1_SOLUTION,
    hints={
        "MySGD.step": "Inside a with torch.no_grad() block, loop over self.parameters. Skip a parameter whose "
        ".grad is None, and subtract self.lr times its .grad from it with -=.",
        "MySGD.zero_grad": "Loop over self.parameters and set each one's .grad to None.",
        "train": "Follow the loop in the lesson, then return the last loss as a float with .item().",
    },
    introduces=("optimizer",),
    check=check_sgd1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# momentum1
# ---------------------------------------------------------------------------

MOMENTUM1_LESSON = prose(
    """
    SGD uses only the current gradient and ignores every earlier step.
    **Momentum**, from Polyak in 1964, also keeps a velocity for each
    parameter. The velocity is a tensor with the same shape as its parameter.
    Each step multiplies the old velocity by a number called the momentum,
    usually 0.9, and adds the new gradient. Then the parameter is updated with
    the velocity in place of the gradient, so it changes by minus the learning
    rate times the velocity.

    When the gradient keeps the same sign step after step, the velocity keeps
    growing. With a momentum of 0.9 and a constant gradient, the velocity
    grows toward 10 times the gradient. That is because
    `1 + 0.9 + 0.81 + ...` adds up to `1 / (1 - 0.9)`, which is 10. When the
    gradient flips sign every step, the terms mostly cancel and the velocity
    stays small.

    `torch.optim.SGD(params, lr=0.1, momentum=0.9)` does this, and it starts
    every velocity at zero. `torch.zeros_like` makes a tensor of zeros with
    the same shape as its input.

    ```python
    torch.zeros_like(torch.ones(2, 3))    # zeros of shape (2, 3)
    ```

    Sutskever and others wrote momentum with the learning rate multiplied into
    the velocity instead. When the learning rate never changes, the two
    versions make exactly the same updates.
    """,
)

MOMENTUM1_SOURCE = source_file(
    '''
    # momentum1.py

    import torch
    from torch import Tensor

    velocity = torch.tensor([1.0, -2.0])
    grad = torch.tensor([0.5, 0.5])
    shrunk = 0.9 * velocity                          # tensor([ 0.9000, -1.8000])


    # TODO: Return the new velocity, which is momentum times the old velocity,
    #   plus grad. velocity and grad have the same shape.
    #   Example: momentum_update(velocity, grad, 0.9) with the tensors above
    #   returns tensor([ 1.4000, -1.3000]).
    def momentum_update(
        velocity: Tensor,
        grad: Tensor,
        momentum: float,
    ) -> Tensor:
        ...


    # TODO: Finish the two lines with ... in step. The first sets the
    #   parameter's velocity to the result of momentum_update on its old
    #   velocity, its gradient and self.momentum. The second subtracts self.lr
    #   times that new velocity from the parameter.
    class MyMomentumSGD:
        def __init__(
            self,
            parameters,
            lr: float,
            momentum: float,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr
            self.momentum = momentum
            self.velocities = [torch.zeros_like(parameter) for parameter in self.parameters]

        def step(self) -> None:
            with torch.no_grad():
                for position, parameter in enumerate(self.parameters):
                    if parameter.grad is None:
                        continue

                    self.velocities[position] = ...
                    parameter -= ...

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    # TODO: Start with a velocity of 0.0 and apply the momentum update steps
    #   times, with the same gradient grad every time. Return the final
    #   velocity as a float.
    #   Example: velocity_after(1.0, 0.9, 50) returns about 9.948.
    def velocity_after(
        grad: float,
        momentum: float,
        steps: int,
    ) -> float:
        ...
    ''',
)

MOMENTUM1_SOLUTION = source_file(
    '''
    # momentum1.py, a solution for comparison

    import torch
    from torch import Tensor

    velocity = torch.tensor([1.0, -2.0])
    grad = torch.tensor([0.5, 0.5])
    shrunk = 0.9 * velocity


    def momentum_update(
        velocity: Tensor,
        grad: Tensor,
        momentum: float,
    ) -> Tensor:
        return momentum * velocity + grad


    class MyMomentumSGD:
        def __init__(
            self,
            parameters,
            lr: float,
            momentum: float,
        ) -> None:
            self.parameters = list(parameters)
            self.lr = lr
            self.momentum = momentum
            self.velocities = [torch.zeros_like(parameter) for parameter in self.parameters]

        def step(self) -> None:
            with torch.no_grad():
                for position, parameter in enumerate(self.parameters):
                    if parameter.grad is None:
                        continue

                    self.velocities[position] = momentum_update(self.velocities[position], parameter.grad, self.momentum)
                    parameter -= self.lr * self.velocities[position]

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None


    def velocity_after(
        grad: float,
        momentum: float,
        steps: int,
    ) -> float:
        velocity = 0.0

        for _ in range(steps):
            velocity = momentum * velocity + grad

        return velocity
    ''',
)


def check_my_momentum(found: type) -> None:
    for momentum in (0.9, 0.5):
        check_optimizer_class(
            found=found,
            arguments={"lr": 0.02, "momentum": momentum},
            reference=torch.optim.SGD,
            reference_arguments={"lr": 0.02, "momentum": momentum},
            name="MyMomentumSGD",
        )


def reference_velocity_after(
    grad: float,
    momentum: float,
    steps: int,
) -> float:
    velocity = 0.0

    for _ in range(steps):
        velocity = momentum * velocity + grad

    return velocity


def check_momentum1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="momentum_update",
        description="multiplies the velocity by momentum and adds the gradient",
        inputs=(
            (torch.tensor([1.0, -2.0]), torch.tensor([0.5, 0.5]), 0.9),
            (small_tensor(510, 2, 3), small_tensor(511, 2, 3), 0.5),
        ),
        reference=lambda velocity, grad, momentum: momentum * velocity + grad,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="MyMomentumSGD",
        method="step",
        description="updates the parameters like SGD with momentum",
        run=check_my_momentum,
    )
    check_on_inputs(
        checker=checker,
        function="velocity_after",
        description="gives the velocity after steps updates with the same gradient",
        inputs=(
            (1.0, 0.9, 1),
            (1.0, 0.9, 50),
            (-2.0, 0.5, 3),
        ),
        reference=reference_velocity_after,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=relative(expected),
        ),
    )


MOMENTUM1 = Exercise(
    name="momentum1",
    chapter=SGD,
    title="Momentum",
    lesson=MOMENTUM1_LESSON,
    source=MOMENTUM1_SOURCE,
    solution=MOMENTUM1_SOLUTION,
    hints={
        "momentum_update": "Multiply velocity by momentum, then add grad.",
        "MyMomentumSGD.step": "The new velocity is momentum_update called on the old velocity, parameter.grad and "
        "self.momentum. Subtract self.lr times that new velocity from the parameter with -=.",
        "velocity_after": "Start a velocity at 0.0 and update it steps times with the same grad. Or add up the "
        "terms grad, momentum times grad, momentum squared times grad, and so on, one per step.",
    },
    introduces=(),
    check=check_momentum1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# momentum2
# ---------------------------------------------------------------------------

MOMENTUM2_LESSON = prose(
    """
    Take the loss `x ** 2 + 100 * y ** 2`. Its graph is a long, narrow valley.
    Changing y raises the loss 100 times as much as changing x by the same
    amount. So the valley is steep across, in y, and gentle along its floor,
    in x. Each optimizer below started at `(-10, 1)` and ran until the loss
    fell below 0.001. Each was tried with 20 learning rates from 0.0005 to
    0.01, and the table shows its best one.

    | Optimizer | Best learning rate | Steps |
    |---|---|---|
    | SGD | 0.0095 | 301 |
    | momentum 0.9 | 0.006 | 85 |
    | momentum 0.99 | 0.002 | 523 |

    The steep y direction limits SGD's learning rate. Above 0.01, each step
    in y jumps past the valley floor and ends farther from it than it started.
    So y grows every step and the loss blows up. With the learning rate kept
    below 0.01, the steps in the gentle x direction are tiny.

    Momentum gets around this limit. As the point bounces across the valley,
    the gradient in y flips sign every step, so it mostly cancels in the
    velocity. The gradient in x keeps the same sign, so it builds up and the
    steps along the floor grow. At a momentum of 0.99 the velocity keeps too
    much of its past, and the point overshoots the bottom again and again.

    The steeper the valley, the bigger the difference. For the loss
    `x ** 2 + 1000 * y ** 2`, with 40 learning rates from 0.00005 to 0.002,
    SGD took 3027 steps and momentum 0.9 took 97.

    Those runs used exact gradients. A minibatch gives a noisy gradient, and
    a smaller batch gives more noise. The next table adds random noise to
    every gradient on the same valley. The noise has the standard deviation
    in the first column, so a bigger number means noisier gradients. Each run
    stopped when the loss fell below 0.1. Each count is the median over 20
    random seeds, at that optimizer's best learning rate.

    | Noise standard deviation | SGD | Momentum 0.9 |
    |---|---|---|
    | 0 | 181 | 36 |
    | 10 | 204 | 44 |
    | 30 | 225.5 | 96 |
    | 100 | 359.5 | 290.5 |
    | 300 | 777 | 585.5 |

    With exact gradients, momentum needed about a fifth of SGD's steps. With
    heavy noise it saved only a quarter of the steps or less. So momentum
    helps most with large batches, which Zhang and others also found in 2019.
    Real losses have narrow valleys like this in many directions. That is why
    SGD with momentum 0.9 is still the usual choice among the simple
    optimizers.
    """,
)

MOMENTUM2_SOURCE = source_file(
    '''
    # momentum2.py

    import torch
    from torch import Tensor

    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = torch.optim.SGD([point], lr=0.01, momentum=0.9)    # an optimizer that updates one tensor


    # TODO: Return the valley loss x ** 2 + 100 * y ** 2, where x and y are the
    #   two entries of point, a tensor of shape (2,).
    #   Example: valley(torch.tensor([-10.0, 1.0])) returns tensor(200.).
    def valley(point: Tensor) -> Tensor:
        ...


    # TODO: Finish the loop. steps_to_reach starts a point at (-10, 1) and moves
    #   it down the valley with torch.optim.SGD. It returns the number of steps
    #   taken before the loss fell below target, or limit if the loss never
    #   did. The missing part takes one optimizer step.
    #   Example: steps_to_reach(0.0095, 0.0, 0.001, 1000) returns 301.
    def steps_to_reach(
        lr: float,
        momentum: float,
        target: float,
        limit: int,
    ) -> int:
        point = torch.tensor([-10.0, 1.0], requires_grad=True)
        optimizer = torch.optim.SGD([point], lr=lr, momentum=momentum)

        for step in range(limit):
            loss = valley(point)

            if loss.item() < target:
                return step

            ...

        return limit


    # TODO: Return the learning rate from rates that brings the loss below
    #   target in the fewest steps, as counted by steps_to_reach. If several
    #   rates tie, return the first of them.
    #   Example: best_rate([0.002, 0.005, 0.0095], 0.0, 0.001, 2000) returns
    #   0.0095.
    def best_rate(
        rates: list[float],
        momentum: float,
        target: float,
        limit: int,
    ) -> float:
        ...
    ''',
)

MOMENTUM2_SOLUTION = source_file(
    '''
    # momentum2.py, a solution for comparison

    import torch
    from torch import Tensor

    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = torch.optim.SGD([point], lr=0.01, momentum=0.9)


    def valley(point: Tensor) -> Tensor:
        return point[0] ** 2 + 100 * point[1] ** 2


    def steps_to_reach(
        lr: float,
        momentum: float,
        target: float,
        limit: int,
    ) -> int:
        point = torch.tensor([-10.0, 1.0], requires_grad=True)
        optimizer = torch.optim.SGD([point], lr=lr, momentum=momentum)

        for step in range(limit):
            loss = valley(point)

            if loss.item() < target:
                return step

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return limit


    def best_rate(
        rates: list[float],
        momentum: float,
        target: float,
        limit: int,
    ) -> float:
        steps = [steps_to_reach(rate, momentum, target, limit) for rate in rates]

        return rates[steps.index(min(steps))]
    ''',
)


def reference_valley(point: Tensor) -> Tensor:
    return point[0] ** 2 + 100 * point[1] ** 2


def reference_steps_to_reach(
    lr: float,
    momentum: float,
    target: float,
    limit: int,
) -> int:
    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = torch.optim.SGD([point], lr=lr, momentum=momentum)

    for step in range(limit):
        loss = reference_valley(point)

        if loss.item() < target:
            return step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return limit


def check_momentum2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="valley",
        description="gives x squared plus 100 times y squared",
        inputs=(
            (torch.tensor([-10.0, 1.0]),),
            (torch.tensor([2.0, -0.5]),),
        ),
        reference=reference_valley,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="steps_to_reach",
        description="counts the steps until the loss falls below target",
        inputs=(
            (0.0095, 0.0, 0.001, 1000),
            (0.006, 0.9, 0.001, 1000),
            (0.03, 0.0, 0.001, 50),
        ),
        reference=reference_steps_to_reach,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="best_rate",
        description="picks the rate that takes the fewest steps",
        inputs=(
            ([0.002, 0.005, 0.0095], 0.0, 0.001, 2000),
            ([0.001, 0.006, 0.02], 0.9, 0.001, 2000),
            # Both rates stop at the limit of 50, so the first one has to win the tie.
            ([0.001, 0.002], 0.0, 0.001, 50),
        ),
        reference=lambda rates, momentum, target, limit: min(rates, key=lambda rate: reference_steps_to_reach(rate, momentum, target, limit)),
        expect=expect_value,
    )


MOMENTUM2 = Exercise(
    name="momentum2",
    chapter=SGD,
    title="Why momentum",
    lesson=MOMENTUM2_LESSON,
    source=MOMENTUM2_SOURCE,
    solution=MOMENTUM2_SOLUTION,
    hints={
        "valley": "point[0] is x and point[1] is y.",
        "steps_to_reach": "The missing part is the three lines of the optimizer loop from sgd1, which call "
        "zero_grad, backward and step.",
        "best_rate": "Call steps_to_reach for each rate, then return the rate at the position of the smallest "
        "count.",
    },
    introduces=(),
    check=check_momentum2,
    time_limit=20,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def valley_steps(
    lr: float,
    momentum: float,
    steepness: int,
    limit: int,
) -> int:
    """steps_to_reach on a valley steepness times steeper across than along, down to a loss of 0.001."""
    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = torch.optim.SGD([point], lr=lr, momentum=momentum)

    for step in range(limit):
        loss = point[0] ** 2 + steepness * point[1] ** 2

        if loss.item() < 0.001:
            return step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return limit


def measure_valley(
    steepness: int = 100,
    rates: tuple[float, ...] = tuple(round(0.0005 * k, 4) for k in range(1, 21)),
    momenta: tuple[float, ...] = (0.0, 0.9, 0.99),
    limit: int = 20000,
) -> dict[float, tuple[float, int]]:
    """momentum2's first table, each momentum's best rate on the valley and the steps it took. Its 1000 times valley
    is steepness=1000 with the 40 rates round(0.00005 * k, 5) and momenta of 0.0 and 0.9."""
    results: dict[float, tuple[float, int]] = {}

    for momentum in momenta:
        steps = [valley_steps(rate, momentum, steepness, limit) for rate in rates]
        results[momentum] = (rates[steps.index(min(steps))], min(steps))

    return results


def noisy_valley_steps(
    lr: float,
    momentum: float,
    noise: float,
    seed: int,
    limit: int = 20000,
) -> int:
    """Steps on momentum2's valley until the loss falls below 0.1, with Gaussian noise of size noise added to each
    gradient, as a smaller batch adds. Plain Python numbers, since a run takes thousands of tiny steps."""
    draws = random.Random(seed)
    x, y, x_velocity, y_velocity = -10.0, 1.0, 0.0, 0.0

    for step in range(limit):
        if x * x + 100 * y * y < 0.1:
            return step

        x_gradient = 2 * x + noise * draws.gauss(0, 1)
        y_gradient = 200 * y + noise * draws.gauss(0, 1)
        x_velocity = momentum * x_velocity + x_gradient
        y_velocity = momentum * y_velocity + y_gradient
        x -= lr * x_velocity
        y -= lr * y_velocity

        if abs(x) > 1e6 or abs(y) > 1e6:
            return limit

    return limit


def measure_noisy_valley(
    noises: tuple[float, ...] = (0.0, 10.0, 30.0, 100.0, 300.0),
    seeds: tuple[int, ...] = tuple(range(20)),
) -> dict[tuple[float, float], tuple[float, float]]:
    """momentum2's second table: for each size of gradient noise, plain SGD's and momentum 0.9's best rate of a grid,
    as (median steps over the seeds to bring the loss below 0.1, rate)."""
    rates = [rate / scale for rate in (round(0.0005 * k, 4) for k in range(1, 21)) for scale in (1, 10)]
    results: dict[tuple[float, float], tuple[float, float]] = {}

    for noise in noises:
        for momentum in (0.0, 0.9):
            best: tuple[float, float] | None = None

            for rate in rates:
                median = statistics.median(noisy_valley_steps(rate, momentum, noise, seed) for seed in seeds)

                if best is None or median < best[0]:
                    best = (median, rate)

            results[(noise, momentum)] = best

    return results
