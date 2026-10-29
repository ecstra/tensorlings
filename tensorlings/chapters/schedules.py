"""26_schedules, the learning rate as a function of the step, warmup, and a range test to pick the rate."""

from __future__ import annotations

import math

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
    expect_value,
)
from ..text import prose, source_file
from .dropout import labelled_points
from .initialization import expect_numbers, relative

SCHEDULES = Chapter(
    folder="26_schedules",
    title="Learning rate schedules",
    readme=source_file(
        """
        # Learning rate schedules

        The best learning rate changes while a model trains. This chapter
        writes step and cosine decay as functions of the step and matches them
        with PyTorch's schedulers, adds a warmup at the start, and picks a rate
        with a range test.

        ## Further reading

        - [Loshchilov and Hutter, SGDR, cosine decay](https://arxiv.org/abs/1608.03983)
        - [Goyal and others, large minibatch SGD, warmup](https://arxiv.org/abs/1706.02677)
        - [Smith, cyclical learning rates, the range test](https://arxiv.org/abs/1506.01186)
        - [PyTorch, how to adjust the learning rate](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)
        """,
    ),
)


def reference_step_rate(
    base: float,
    step: int,
    step_size: int,
    gamma: float,
) -> float:
    return base * gamma ** (step // step_size)


def reference_cosine_rate(
    base: float,
    step: int,
    total: int,
    minimum: float,
) -> float:
    return minimum + (base - minimum) * (1 + math.cos(math.pi * step / total)) / 2


def rate_tolerance(expected: float) -> float:
    return 1e-12 + 1e-9 * abs(expected)


def expect_rate(
    call: str,
    actual: object,
    expected: float,
) -> None:
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=rate_tolerance(expected),
    )


def one_parameter_optimizer(lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=lr)


# ---------------------------------------------------------------------------
# schedule1
# ---------------------------------------------------------------------------

SCHEDULE1_LESSON = prose(
    """
    A rate that suits the start, while the loss falls fast, is too big near
    the end. There the steps bounce around the bottom instead of settling into
    it. A **learning rate schedule** lowers the rate as training goes.

    - step decay: multiply the rate by `gamma` every `step_size` steps. ResNet divided it by 10 twice.
    - cosine decay: glide from the base rate down to a minimum along half a cosine wave. Most language models use it.

    Cosine decay comes from Loshchilov and Hutter's SGDR, 2017. At step `t` of
    `total` the rate is the minimum plus `(base - minimum)` times
    `(1 + cos(pi * t / total)) / 2`.

    Here the names model trained for 5000 steps, with each schedule. Step
    decay divided the rate by 10 for the last quarter, and cosine went down
    to 0.

    | Optimizer | Constant | Step decay | Cosine |
    |---|---|---|---|
    | SGD at 0.1 | 2.301 | 2.258 | 2.273 |
    | Adam at 0.003 | 2.281 | 2.233 | 2.239 |

    Lowering the rate at the end helped every time, by more than the choice
    of optimizer did.

    A scheduler from `torch.optim.lr_scheduler` changes the optimizer's rate
    each time its `.step()` runs, once per training step after
    `optimizer.step()`. The rate the next step uses is in
    `optimizer.param_groups[0]["lr"]`.

    ```python
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0)
    ```

    In `CosineAnnealingLR`, `T_max` is the cosine's `total` and `eta_min` is
    its `minimum`.
    """,
)

SCHEDULE1_SOURCE = source_file(
    '''
    # schedule1.py

    import math

    import torch

    halfway = (1 + math.cos(math.pi * 0.5)) / 2      # 0.5, the cosine factor halfway through


    # TODO: Return step decay's rate at step, base times gamma once for every
    #   full step_size steps done.
    def step_rate(
        base: float,
        step: int,
        step_size: int,
        gamma: float,
    ) -> float:
        ...


    # TODO: Return cosine decay's rate at step, going from base at step 0 down to
    #   minimum at step total.
    def cosine_rate(
        base: float,
        step: int,
        total: int,
        minimum: float,
    ) -> float:
        ...


    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    current = optimizer.param_groups[0]["lr"]        # 0.1


    # TODO: Return the rate optimizer uses at each of steps steps, as a list.
    #   Read it before each optimizer.step(), and call scheduler.step() after.
    def scheduler_rates(
        optimizer: torch.optim.Optimizer,
        scheduler,
        steps: int,
    ) -> list[float]:
        ...
    ''',
)

SCHEDULE1_SOLUTION = source_file(
    '''
    # schedule1.py, a solution for comparison

    import math

    import torch

    halfway = (1 + math.cos(math.pi * 0.5)) / 2


    def step_rate(
        base: float,
        step: int,
        step_size: int,
        gamma: float,
    ) -> float:
        return base * gamma ** (step // step_size)


    def cosine_rate(
        base: float,
        step: int,
        total: int,
        minimum: float,
    ) -> float:
        return minimum + (base - minimum) * (1 + math.cos(math.pi * step / total)) / 2


    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    current = optimizer.param_groups[0]["lr"]


    def scheduler_rates(
        optimizer: torch.optim.Optimizer,
        scheduler,
        steps: int,
    ) -> list[float]:
        rates = []

        for _ in range(steps):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        return rates
    ''',
)


def check_scheduler_rates(found: object) -> None:
    cases = (
        ("StepLR(optimizer, step_size=3, gamma=0.5) from a rate of 0.1", lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5), lambda step: reference_step_rate(0.1, step, 3, 0.5)),
        ("CosineAnnealingLR(optimizer, T_max=8, eta_min=0.01) from a rate of 0.1", lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=0.01), lambda step: reference_cosine_rate(0.1, step, 8, 0.01)),
    )

    for name, make, closed_form in cases:
        optimizer = one_parameter_optimizer(0.1)
        actual = found(optimizer, make(optimizer), 9)
        expected = [closed_form(step) for step in range(9)]
        expect_numbers(
            call=f"scheduler_rates for 9 steps of {name}",
            actual=actual,
            expected=expected,
        )


def check_schedule1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="step_rate",
        description="multiplies by gamma every step_size steps",
        inputs=(
            (0.1, 0, 3, 0.5),
            (0.1, 5, 3, 0.5),
            (0.1, 6, 3, 0.5),
            (0.01, 7500, 2500, 0.1),
        ),
        reference=reference_step_rate,
        expect=expect_rate,
    )
    check_on_inputs(
        checker=checker,
        function="cosine_rate",
        description="glides from base to minimum along half a cosine",
        inputs=(
            (0.1, 0, 100, 0.0),
            (0.1, 50, 100, 0.0),
            (0.1, 100, 100, 0.001),
            (0.003, 1250, 5000, 0.0003),
        ),
        reference=reference_cosine_rate,
        expect=expect_rate,
    )
    checker.test(
        function="scheduler_rates",
        description="reads the rate before each step and steps the scheduler after",
        run=check_scheduler_rates,
    )


SCHEDULE1 = Exercise(
    name="schedule1",
    chapter=SCHEDULES,
    title="Step and cosine decay",
    lesson=SCHEDULE1_LESSON,
    source=SCHEDULE1_SOURCE,
    solution=SCHEDULE1_SOLUTION,
    hints={
        "step_rate": "step // step_size counts the full step_size stretches done. Raise gamma to that power and "
        "multiply by base.",
        "cosine_rate": "Follow the formula in the lesson, with math.cos and math.pi.",
        "scheduler_rates": "In a loop of steps, append optimizer.param_groups[0][\"lr\"] to a list, then call "
        "optimizer.step() and scheduler.step().",
    },
    introduces=(),
    check=check_schedule1,
    imports_first=OPTIMIZER_IMPORTS,
)


def small_classifier(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))


# ---------------------------------------------------------------------------
# warmup1
# ---------------------------------------------------------------------------

WARMUP1_LESSON = prose(
    """
    The first steps are the riskiest. The weights are random, the gradients
    large, and Adam's averages hold only a few steps. A big rate there can
    throw the weights somewhere training never climbs out of. **Warmup**
    raises the rate from near 0 to its base over the first few hundred steps,
    and then the schedule takes over.

    Here a deeper names model, eight post norm blocks like residual4's,
    trained with Adam for 2000 steps. The 0.01 rows ran on three seeds.

    | Rate | Warmup | Validation loss |
    |---|---|---|
    | 0.003 | none | 2.310 |
    | 0.003 | 200 steps | 2.312 |
    | 0.01 | none | 2.76 to 2.83 |
    | 0.01 | 200 steps | 2.36 to 2.37 |

    At 0.003, warmup changed nothing. At 0.01 the runs without it fell faster
    for the first steps, then stuck near 2.8, worse than the bigram. With
    warmup, all three reached about 2.37. So warmup makes a larger rate safe,
    and post norm models need it most, as Xiong and others found. Here 0.01
    with warmup still lost to 0.003, with or without it. Warmup matters when
    a larger rate is needed, as with big batches.

    Goyal and others warmed up for 5 epochs to train ResNet with huge batches.
    nanoGPT warms up for 2000 of its 600,000 steps by default, and for 100 in
    its Shakespeare character config. Then it decays along a cosine to a tenth
    of the base rate.

    Setting `group["lr"]` for each group in `optimizer.param_groups` changes
    the rate by hand, before `optimizer.step()`.
    """,
)

WARMUP1_SOURCE = source_file(
    '''
    # warmup1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    climbing = [0.1 * (step + 1) / 4 for step in range(4)]    # [0.025, 0.05, 0.075, 0.1]


    # TODO: Return the rate at step during a linear warmup over warmup steps,
    #   base times (step + 1) / warmup, and base itself once warmup is over.
    def warmup_rate(
        base: float,
        step: int,
        warmup: int,
    ) -> float:
        ...


    # TODO: Return the rate at step of a schedule shaped like nanoGPT's. During
    #   the first warmup steps it is warmup_rate. After that it falls along a
    #   cosine from base at step warmup to minimum at step total.
    def warmup_cosine(
        base: float,
        step: int,
        warmup: int,
        total: int,
        minimum: float,
    ) -> float:
        ...


    # TODO: Finish the line in the inner loop, so each step trains with Adam at
    #   the rate rate_at(step) gives. It returns the last loss.
    def train_scheduled(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        rate_at,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters())

        for step in range(steps):
            for group in optimizer.param_groups:
                ...

            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)

WARMUP1_SOLUTION = source_file(
    '''
    # warmup1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    climbing = [0.1 * (step + 1) / 4 for step in range(4)]


    def warmup_rate(
        base: float,
        step: int,
        warmup: int,
    ) -> float:
        return base * min(1.0, (step + 1) / warmup)


    def warmup_cosine(
        base: float,
        step: int,
        warmup: int,
        total: int,
        minimum: float,
    ) -> float:
        if step < warmup:
            return warmup_rate(base, step, warmup)

        progress = (step - warmup) / (total - warmup)

        return minimum + (base - minimum) * (1 + math.cos(math.pi * progress)) / 2


    def train_scheduled(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        rate_at,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters())

        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = rate_at(step)

            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
    ''',
)


def reference_warmup_rate(
    base: float,
    step: int,
    warmup: int,
) -> float:
    return base * min(1.0, (step + 1) / warmup)


def reference_warmup_cosine(
    base: float,
    step: int,
    warmup: int,
    total: int,
    minimum: float,
) -> float:
    if step < warmup:
        return reference_warmup_rate(base, step, warmup)

    return reference_cosine_rate(base, step - warmup, total - warmup, minimum)


def reference_train_scheduled(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    rate_at,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = rate_at(step)

        batch = torch.randint(0, len(X), (32,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def check_train_scheduled(found: object) -> None:
    X, Y = labelled_points(560, 200, 6, 3)
    rate_at = lambda step: reference_warmup_cosine(0.01, step, 10, 60, 0.001)
    expect_close(
        call="train_scheduled for 60 steps with 10 steps of warmup to 0.01, then cosine decay",
        actual=found(small_classifier(561), X, Y, 60, rate_at, torch.Generator().manual_seed(562)),
        expected=reference_train_scheduled(small_classifier(561), X, Y, 60, rate_at, torch.Generator().manual_seed(562)),
        tolerance=1e-5,
    )


def check_warmup1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="warmup_rate",
        description="climbs to base over the warmup steps, then stays there",
        inputs=(
            (0.1, 0, 4),
            (0.1, 2, 4),
            (0.1, 3, 4),
            (0.003, 500, 200),
        ),
        reference=reference_warmup_rate,
        expect=expect_rate,
    )
    check_on_inputs(
        checker=checker,
        function="warmup_cosine",
        description="warms up, then falls along a cosine to the minimum",
        inputs=(
            (0.01, 4, 10, 110, 0.001),
            (0.01, 10, 10, 110, 0.001),
            (0.01, 60, 10, 110, 0.001),
            (0.01, 110, 10, 110, 0.001),
        ),
        reference=reference_warmup_cosine,
        expect=expect_rate,
    )
    checker.test(
        function="train_scheduled",
        description="sets the rate from rate_at before every step",
        run=check_train_scheduled,
    )


WARMUP1 = Exercise(
    name="warmup1",
    chapter=SCHEDULES,
    title="Warmup",
    lesson=WARMUP1_LESSON,
    source=WARMUP1_SOURCE,
    solution=WARMUP1_SOLUTION,
    hints={
        "warmup_rate": "base times the smaller of 1.0 and (step + 1) / warmup. min gives the smaller of two numbers.",
        "warmup_cosine": "Before warmup, return warmup_rate. After it, the cosine of schedule1 with the progress "
        "(step - warmup) / (total - warmup) in place of step / total.",
        "train_scheduled": "Set group[\"lr\"] to rate_at(step).",
    },
    introduces=(),
    check=check_warmup1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# schedule2
# ---------------------------------------------------------------------------

SCHEDULE2_LESSON = prose(
    """
    A range test, from Leslie Smith's 2015 paper on cyclical rates, finds a
    good rate in one short run. Start with a tiny rate and raise it a little
    every step, while recording the loss. The loss falls slowly, then fast,
    then bottoms out and blows up. Smith raised the rate by the same amount
    each step. This test follows the fast.ai learning rate finder, which
    raises it by the same factor each step.

    Here the names model ran 300 steps of Adam, with the rate climbing from
    0.00001 to 1, on minibatches of 256.

    | Rate | 0.00001 | 0.0001 | 0.001 | 0.0057 | 0.058 | 0.18 | 0.33 | 0.58 |
    |---|---|---|---|---|---|---|---|---|
    | Loss | 3.29 | 3.23 | 2.64 | 2.42 | 2.45 | 3.01 | 13.1 | 99.9 |

    One step per rate makes the losses noisy. So they get a running average
    first, corrected at the start the way Adam's hats are. The smoothed loss
    was lowest near 0.0095 and doubled past 0.33.

    The usual rule picks a tenth of the lowest point, about 0.001 here, the
    rate adamw2's sweep found best. On a second seed the lowest point came out
    at 0.026, a tenth of it 0.0026, still a good rate. The test gives a place
    to start, for 300 steps instead of a sweep of full runs.

    The rates are `start * (end / start) ** (k / (count - 1))` for `k` from 0
    to `count - 1`, spread evenly on a log scale.
    """,
)

SCHEDULE2_SOURCE = source_file(
    '''
    # schedule2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    factor = (1.0 / 0.00001) ** (1 / 299)            # 1.04, the climb per step over 300 steps


    # TODO: Return count rates from start to end, each the same factor above the
    #   one before.
    def exponential_rates(
        start: float,
        end: float,
        count: int,
    ) -> list[float]:
        ...


    # TODO: Return the losses smoothed by a running average with beta, which
    #   starts at 0. Keep the running average itself uncorrected. At the k-th
    #   loss, counted from 1, append that average divided by 1 - beta ** k, as
    #   Adam's hats do.
    def smooth(
        losses: list[float],
        beta: float,
    ) -> list[float]:
        ...


    # TODO: Return a tenth of the rate where smoothed is lowest.
    def pick_rate(
        rates: list[float],
        smoothed: list[float],
    ) -> float:
        ...


    # TODO: Finish the line that sets the rate, so each step of the loop trains
    #   at the next of rates. It returns the loss of every step.
    def range_test(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        rates: list[float],
        generator: torch.Generator,
    ) -> list[float]:
        optimizer = torch.optim.Adam(model.parameters())
        losses = []

        for rate in rates:
            optimizer.param_groups[0]["lr"] = ...
            batch = torch.randint(0, len(X), (256,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses
    ''',
)

SCHEDULE2_SOLUTION = source_file(
    '''
    # schedule2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    factor = (1.0 / 0.00001) ** (1 / 299)


    def exponential_rates(
        start: float,
        end: float,
        count: int,
    ) -> list[float]:
        return [start * (end / start) ** (k / (count - 1)) for k in range(count)]


    def smooth(
        losses: list[float],
        beta: float,
    ) -> list[float]:
        average = 0.0
        smoothed = []

        for k, loss in enumerate(losses, start=1):
            average = beta * average + (1 - beta) * loss
            smoothed.append(average / (1 - beta ** k))

        return smoothed


    def pick_rate(
        rates: list[float],
        smoothed: list[float],
    ) -> float:
        return rates[smoothed.index(min(smoothed))] / 10


    def range_test(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        rates: list[float],
        generator: torch.Generator,
    ) -> list[float]:
        optimizer = torch.optim.Adam(model.parameters())
        losses = []

        for rate in rates:
            optimizer.param_groups[0]["lr"] = rate
            batch = torch.randint(0, len(X), (256,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses
    ''',
)


def reference_smooth(
    losses: list[float],
    beta: float,
) -> list[float]:
    average = 0.0
    smoothed = []

    for k, loss in enumerate(losses, start=1):
        average = beta * average + (1 - beta) * loss
        smoothed.append(average / (1 - beta ** k))

    return smoothed


def reference_range_test(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    rates: list[float],
    generator: torch.Generator,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters())
    losses = []

    for rate in rates:
        optimizer.param_groups[0]["lr"] = rate
        batch = torch.randint(0, len(X), (256,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def check_range_test(found: object) -> None:
    X, Y = labelled_points(570, 300, 6, 3)
    rates = [0.0001 * 100 ** (k / 19) for k in range(20)]
    expect_numbers(
        call="range_test on a small classifier, with 20 rates from 0.0001 to 0.01",
        actual=found(small_classifier(571), X, Y, rates, torch.Generator().manual_seed(572)),
        expected=reference_range_test(small_classifier(571), X, Y, rates, torch.Generator().manual_seed(572)),
    )


def check_schedule2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="exponential_rates",
        description="spreads the rates evenly on a log scale",
        inputs=(
            (0.001, 1.0, 4),
            (0.00001, 1.0, 6),
        ),
        reference=lambda start, end, count: [start * (end / start) ** (k / (count - 1)) for k in range(count)],
        expect=expect_numbers,
    )
    check_on_inputs(
        checker=checker,
        function="smooth",
        description="gives a running average corrected at the start",
        inputs=(
            ([3.0, 2.0, 2.5, 1.0], 0.9),
            ([1.0, 1.0, 1.0], 0.5),
        ),
        reference=reference_smooth,
        expect=expect_numbers,
    )
    check_on_inputs(
        checker=checker,
        function="pick_rate",
        description="takes a tenth of the rate at the lowest smoothed loss",
        inputs=(
            ([0.001, 0.01, 0.1, 1.0], [2.6, 2.4, 2.5, 9.0]),
            ([0.0001, 0.001, 0.01], [3.0, 2.9, 3.5]),
        ),
        reference=lambda rates, smoothed: rates[smoothed.index(min(smoothed))] / 10,
        expect=expect_rate,
    )
    checker.test(
        function="range_test",
        description="trains one step at each rate and records the losses",
        run=check_range_test,
    )


SCHEDULE2 = Exercise(
    name="schedule2",
    chapter=SCHEDULES,
    title="A range test",
    lesson=SCHEDULE2_LESSON,
    source=SCHEDULE2_SOURCE,
    solution=SCHEDULE2_SOLUTION,
    hints={
        "exponential_rates": "Follow the formula at the end of the lesson in a list comprehension over k.",
        "smooth": "Keep an average starting at 0.0, and count the losses from 1 with enumerate(losses, start=1). "
        "Update the average as beta * average + (1 - beta) * loss, then append it divided by 1 - beta ** k.",
        "pick_rate": "Find the position of the smallest smoothed loss, take the rate there, and divide by 10.",
        "range_test": "Set it to rate, the rate of this pass through the loop.",
    },
    introduces=(),
    check=check_schedule2,
    imports_first=OPTIMIZER_IMPORTS,
)
