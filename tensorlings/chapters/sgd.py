"""24_sgd, the optimizer object, SGD built and matched, and momentum built and seen in a narrow valley."""

from __future__ import annotations

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

        Section 4 is about the step itself. This chapter moves the update of
        Section 2 into an optimizer object, builds PyTorch's SGD, then adds
        momentum and shows what it does in a long narrow valley.

        ## Further reading

        - [PyTorch, torch.optim](https://docs.pytorch.org/docs/2.14/optim.html)
        - [Goh, why momentum really works](https://distill.pub/2017/momentum/)
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
        raise Mismatch(f"After {call} the weights did not move, which usually means {name}.step is {NOT_STARTED}.")

    mse_steps(theirs, reference(theirs.parameters(), **reference_arguments), x, y, 5)
    expect_same_parameters(call, mine, theirs)


# ---------------------------------------------------------------------------
# sgd1
# ---------------------------------------------------------------------------

SGD1_LESSON = prose(
    """
    Every loop so far stepped the parameters by hand. PyTorch puts the step in
    an **optimizer**, an object that holds the parameters and knows how to
    move them.

    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

    `zero_grad` clears every gradient, `backward` fills them, and `step` moves
    every parameter. SGD, stochastic gradient descent, is the step of Section
    2, each parameter minus the learning rate times its gradient. It is called
    stochastic because each minibatch gives a noisy estimate of the gradient.

    Swapping in another optimizer changes one line, so the rest of this
    section builds them one by one and swaps them in.

    `model.parameters()` can be gone through only once, so an optimizer keeps
    the parameters in a list. A parameter that took no part in the loss has
    `.grad` of `None`, and the step skips it. `zero_grad` sets every `.grad`
    to `None`, which PyTorch does too.
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


    # TODO: Finish step and zero_grad. step moves every parameter that has a
    #   gradient, against that gradient, scaled by self.lr. zero_grad sets every
    #   gradient to None.
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


    # TODO: Run steps steps of optimizer on the mean squared error of model on
    #   x against y, clearing the gradients before each backward. Return the
    #   loss of the last step, the one computed before its optimizer.step().
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
        description="moves the parameters like torch.optim.SGD",
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
        "MySGD.step": "Inside a with torch.no_grad() block, loop over self.parameters, skip one whose .grad is "
        "None, and take self.lr times its .grad away from it with -=.",
        "MySGD.zero_grad": "Loop over self.parameters and set each one's .grad to None.",
        "train": "Follow the loop in the lesson, then return the last loss with .item().",
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
    SGD forgets every earlier step. **Momentum**, from Polyak in 1964, keeps a
    running velocity instead, one per parameter with the same shape. Each step
    shrinks the old velocity by a factor, the momentum, usually 0.9, and adds
    the new gradient. The parameter then moves by the velocity, times the
    learning rate.

    Gradients that keep pointing the same way pile up. With a momentum of 0.9
    the steps along them grow to ten times the gradient, since
    `1 + 0.9 + 0.81 + ...` adds up to `1 / (1 - 0.9)`. Gradients that keep
    flipping sign cancel out. Picture a ball rolling downhill, gathering speed
    along a valley floor while its bounce between the walls dies down.

    `torch.optim.SGD(params, lr=0.1, momentum=0.9)` does this, with every
    velocity starting at zero.

    ```python
    torch.zeros_like(torch.ones(2, 3))    # zeros of shape (2, 3)
    ```

    Sutskever and others wrote it with the learning rate inside the velocity.
    With a learning rate that never changes, the two forms take the same
    steps.
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


    # TODO: Return the new velocity, the old one shrunk by momentum, plus grad.
    def momentum_update(
        velocity: Tensor,
        grad: Tensor,
        momentum: float,
    ) -> Tensor:
        ...


    # TODO: Finish the two lines with ... in step, so each parameter's velocity
    #   is updated with momentum_update, and the parameter moves against its
    #   velocity, scaled by self.lr.
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


    # TODO: Return the velocity after steps steps that all have the same
    #   gradient grad, starting from a velocity of 0.
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
        description="shrinks the velocity and adds the gradient",
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
        description="moves the parameters like SGD with momentum",
        run=check_my_momentum,
    )
    check_on_inputs(
        checker=checker,
        function="velocity_after",
        description="gives the velocity a steady gradient builds up",
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
        "momentum_update": "momentum times velocity, plus grad.",
        "MyMomentumSGD.step": "The velocity is momentum_update of the old velocity, parameter.grad and "
        "self.momentum. The parameter moves by self.lr times that new velocity.",
        "velocity_after": "Start a velocity at 0.0 and update it steps times with the same grad, or add up the "
        "shrinking terms.",
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
    Here is a long narrow valley, the loss `x ** 2 + 100 * y ** 2`. It is 100
    times steeper across, in y, than along, in x. Each optimizer started at
    `(-10, 1)` and ran until the loss fell below 0.001. Each got its best of
    20 learning rates, from 0.0005 to 0.01.

    | Optimizer | Best learning rate | Steps |
    |---|---|---|
    | SGD | 0.0095 | 301 |
    | momentum 0.9 | 0.006 | 85 |
    | momentum 0.99 | 0.002 | 523 |

    SGD's learning rate is capped by the steep direction. Above 0.01 each step
    across overshoots the valley floor by more than it started, and the loss
    blows up. That cap makes its steps along the gentle floor tiny.

    Momentum gets around the cap. The bounce across flips sign every step and
    cancels in the velocity, while the steps along the floor pile up. At 0.99
    the velocity remembers too much, and overshoots the bottom again and
    again.

    The gap grows with the valley. At 1000 times steeper, with the rates tried
    more finely, SGD took 3110 steps and momentum 0.9 took 95. Real losses
    have such valleys in many directions, so SGD with momentum 0.9 is the
    usual plain optimizer.
    """,
)

MOMENTUM2_SOURCE = source_file(
    '''
    # momentum2.py

    import torch
    from torch import Tensor

    point = torch.tensor([-10.0, 1.0], requires_grad=True)
    optimizer = torch.optim.SGD([point], lr=0.01, momentum=0.9)    # an optimizer of one tensor


    # TODO: Return the valley loss x ** 2 + 100 * y ** 2, where x and y are the
    #   two entries of point.
    def valley(point: Tensor) -> Tensor:
        ...


    # TODO: Finish the loop, so this steps a point from (-10, 1) down the valley
    #   with torch.optim.SGD, and returns how many steps it took for the loss to
    #   fall below target, or limit if it never did.
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


    # TODO: Return the learning rate from rates that reaches target in the fewest
    #   steps, the first one if several tie.
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
        description="counts the steps down to the target",
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
        "steps_to_reach": "The missing part is the three lines of the optimizer loop from sgd1, zero_grad, "
        "backward and step.",
        "best_rate": "Call steps_to_reach for each rate, then return the rate at the position of the smallest "
        "count.",
    },
    introduces=(),
    check=check_momentum2,
    time_limit=20,
    imports_first=OPTIMIZER_IMPORTS,
)
