"""26_schedules, the learning rate as a function of the step, warmup, and a range test to pick the rate."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

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
from .adaptive import pinned_threads
from .dropout import labelled_points
from .initialization import expect_numbers, relative
from .names import context_sets, names_model, reference_validation_loss

SCHEDULES = Chapter(
    folder="26_schedules",
    title="Learning rate schedules",
    readme=source_file(
        """
        # Learning rate schedules

        The best learning rate changes during training. This chapter writes
        two common schedules, step decay and cosine decay, as functions of the
        step number, and checks them against PyTorch's schedulers. Then it adds
        a warmup at the start of training, and uses a range test to pick a
        good learning rate.

        ## Further reading

        - [Loshchilov and Hutter, SGDR, cosine decay](https://arxiv.org/abs/1608.03983)
        - [Hägele and others, a constant rate with a cooldown against cosine](https://arxiv.org/abs/2405.18392)
        - [Goyal and others, large minibatch SGD, warmup](https://arxiv.org/abs/1706.02677)
        - [Xiong and others, layer normalization in the transformer, warmup for post norm](https://arxiv.org/abs/2002.04745)
        - [Kalra and Barkeshli, why warmup the learning rate](https://arxiv.org/abs/2406.09405)
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
    A learning rate that works well at the start, while the loss falls fast,
    is too big near the end. Near the end, the parameters keep jumping around
    the minimum of the loss instead of getting closer to it. A **learning rate
    schedule** lowers the learning rate as training goes on.

    - step decay: multiply the rate by `gamma` every `step_size` steps. ResNet's training divided it by 10 twice.
    - cosine decay: lower the rate smoothly from the base rate to a minimum, following half a cosine wave. Most language models use it.

    Cosine decay comes from SGDR, a 2017 paper by Loshchilov and Hutter. At
    step `t` out of `total` steps, the rate is

    ```text
    rate = minimum + (base - minimum) * (1 + cos(pi * t / total)) / 2
    ```

    The factor `(1 + cos(pi * t / total)) / 2` goes from 1 at step 0 down to
    0 at step `total`.

    Here the names model was trained for 5000 steps with each schedule, with
    three seeds each. Step decay divided the rate by 10 for the last quarter
    of training. Cosine decay went down to 0. The table shows the range of
    validation losses over the seeds.

    | Optimizer | Constant | Step decay | Cosine |
    |---|---|---|---|
    | SGD at 0.1 | 2.297 to 2.304 | 2.255 to 2.258 | 2.267 to 2.273 |
    | Adam at 0.003 | 2.274 to 2.281 | 2.230 to 2.233 | 2.234 to 2.239 |

    Lowering the rate at the end helped on every seed. It helped more than
    switching from SGD to Adam did. Step decay's single late drop did as well
    as cosine decay, and slightly better. Hägele and others found the same for
    language models in 2024. A constant rate with a cooldown at the end does
    as well as cosine as models get larger.

    PyTorch's schedulers are in `torch.optim.lr_scheduler`. A scheduler
    changes the optimizer's learning rate each time you call
    `scheduler.step()`, which goes after `optimizer.step()`. PyTorch's
    examples call it once per epoch, but this course calls it once per
    training step. The rate that the next step will use is stored in
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


    # TODO: Return the step decay rate at step. Start from base and multiply by
    #   gamma once for every full step_size steps completed so far.
    #   Example: step_rate(0.1, 6, 3, 0.5) returns 0.025.
    def step_rate(
        base: float,
        step: int,
        step_size: int,
        gamma: float,
    ) -> float:
        ...


    # TODO: Return the cosine decay rate at step, with the formula from the
    #   lesson. It goes from base at step 0 down to minimum at step total.
    #   Example: cosine_rate(0.1, 50, 100, 0.0) returns about 0.05.
    def cosine_rate(
        base: float,
        step: int,
        total: int,
        minimum: float,
    ) -> float:
        ...


    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    current = optimizer.param_groups[0]["lr"]        # 0.1


    # TODO: Run steps steps and return, as a list, the learning rate optimizer
    #   uses at each step. At each step, read the rate before optimizer.step(),
    #   then call scheduler.step() after it.
    #   Example: with StepLR(optimizer, step_size=3, gamma=0.5) on the optimizer
    #   above, 7 steps return [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025].
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

        # The rates read come out the same in either order, so the order shows only in PyTorch's warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = found(optimizer, make(optimizer), 9)

        if any("before `optimizer.step()`" in str(warning.message) for warning in caught):
            raise Mismatch(f"scheduler_rates for {name} calls scheduler.step() before optimizer.step(). Step the optimizer first, then the scheduler.")

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
        description="lowers the rate from base to minimum along half a cosine",
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
        "step_rate": "step // step_size is the number of full blocks of step_size steps completed so far. Raise "
        "gamma to that power and multiply by base.",
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
    The first steps of training are the riskiest. The weights are random and
    the gradients are large. Adam's running averages are built from only a
    few steps, so they are still noisy. A large learning rate at this point
    can push the weights into a bad region that training never gets out of.
    **Warmup** raises the learning rate from near 0 to its base value over
    the first few hundred steps. After that, the usual schedule takes over.

    Here a deeper names model was trained with Adam for 1500 steps, with a
    batch size of 64. It has 32 post norm blocks like the ones in residual4.
    After the warmup, if there was one, the rate followed a cosine down to 0.
    Every row was run with three seeds.

    | Blocks | Rate | Warmup | Validation loss |
    |---|---|---|---|
    | post norm | 0.0003 | none | 2.255 to 2.265 |
    | post norm | 0.0005 | none | 2.258 to 2.300 |
    | post norm | 0.0007 | none | 2.738 to 2.788 |
    | post norm | 0.0007 | 500 steps | 2.249 to 2.255 |
    | post norm | 0.001 | none | 2.820 to 2.821 |
    | post norm | 0.001 | 150 steps | 2.820 to 2.821 |
    | pre norm | 0.001 | none | 2.294 to 2.311 |

    Without warmup, a rate of 0.0007 got stuck near 2.75. That is worse than
    the bigram model's 2.465. With 500 steps of warmup, the same rate trained
    well. On every seed it ended lower than 0.0003, the best rate without
    warmup. Kalra and Barkeshli found the same benefit in 2024. Warmup lets a
    larger learning rate train at all, and the larger rate ends at a lower
    loss.

    Warmup has to be long enough. At a rate of 0.001, 150 steps of warmup
    left the model as stuck as no warmup at all. The pre norm stack trained
    at 0.001 with no warmup. Xiong and others found that post norm models
    need warmup the most.

    Large batches need large learning rates, so they need warmup too. Goyal
    and others warmed up for 5 epochs to train ResNet with huge batches.
    nanoGPT warms up for the first 2000 of its 600,000 steps by default, and
    for 100 steps in its Shakespeare character config. After the warmup, its
    rate decays along a cosine to a tenth of the base rate.

    To change the rate by hand, set `group["lr"]` for each `group` in
    `optimizer.param_groups`, before calling `optimizer.step()`.
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


    # TODO: Return the learning rate at step for a linear warmup that lasts
    #   warmup steps. During the warmup the rate is base * (step + 1) / warmup.
    #   Once the warmup is over, the rate is base.
    #   Example: warmup_rate(0.1, 2, 4) returns about 0.075, as in climbing.
    def warmup_rate(
        base: float,
        step: int,
        warmup: int,
    ) -> float:
        ...


    # TODO: Return the learning rate at step for a schedule shaped like
    #   nanoGPT's. For the first warmup steps, the rate is warmup_rate. After
    #   that, it falls along a cosine from base at step warmup to minimum at
    #   step total.
    #   Example: warmup_cosine(0.01, 60, 10, 110, 0.001) returns about 0.0055.
    def warmup_cosine(
        base: float,
        step: int,
        warmup: int,
        total: int,
        minimum: float,
    ) -> float:
        ...


    # TODO: Finish the line in the inner loop, so that each step trains with
    #   Adam at the learning rate rate_at(step). rate_at is a function from the
    #   step number to a rate. train_scheduled returns the loss of the last step.
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
        description="rises to base over the warmup steps, then stays at base",
        inputs=(
            (0.1, 0, 4),
            (0.1, 2, 4),
            (0.1, 3, 4),
            (0.1, 4, 4),
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
        "warmup_rate": "Multiply base by the smaller of 1.0 and (step + 1) / warmup. Python's min returns the "
        "smaller of two numbers.",
        "warmup_cosine": "While step is below warmup, return warmup_rate. After that, use the cosine formula from "
        "schedule1 with the progress (step - warmup) / (total - warmup) in place of step / total.",
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
    A range test finds a good learning rate in one short run. It comes from
    Leslie Smith's 2015 paper on cyclical learning rates. You start with a
    tiny rate and raise it a little at every step, while recording the loss.
    The loss falls slowly at first, then fast, then reaches its lowest point
    and blows up. Smith added the same amount to the rate at each step. This
    test follows the fast.ai learning rate finder instead, which multiplies
    the rate by the same factor at each step.

    Here the names model was trained for 300 steps of Adam on minibatches of
    256, with the rate rising from 0.00001 to 1.

    | Rate | 0.00001 | 0.0001 | 0.001 | 0.0057 | 0.058 | 0.18 | 0.33 | 0.58 |
    |---|---|---|---|---|---|---|---|---|
    | Loss | 3.29 | 3.23 | 2.64 | 2.42 | 2.45 | 3.01 | 13.1 | 99.9 |

    Each rate gets only one step, so the losses are noisy. The test first
    smooths them with a running average with beta 0.9. The average is
    corrected at the start the same way Adam's hats correct its averages. The
    smoothed loss was lowest at a rate near 0.0095. Past a rate of 0.33 it
    was more than double its lowest value.

    The usual rule picks a tenth of the rate at the lowest point, which is
    about 0.001 here. That is the rate the sweep in adamw2 found best. Over
    five seeds, the picked rates ranged from 0.00095 to 0.0020. Adam trained
    at each picked rate for 5000 steps reached a validation loss of 2.265 to
    2.277. That is as low as the best rate in the adamw2 sweep. So one range
    test of 300 steps can replace a sweep of many full training runs.

    The rates are `start * (end / start) ** (k / (count - 1))` for `k` from 0
    to `count - 1`. Each rate is the previous one times the same factor, so
    the rates are spread evenly on a log scale.
    """,
)

SCHEDULE2_SOURCE = source_file(
    '''
    # schedule2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    factor = (1.0 / 0.00001) ** (1 / 299)            # about 1.04, the ratio from one rate to the next


    # TODO: Return a list of count rates from start to end, where each rate is
    #   the previous one times the same factor. The first rate is start and the
    #   last is end.
    #   Example: exponential_rates(0.001, 1.0, 4) returns about
    #   [0.001, 0.01, 0.1, 1.0].
    def exponential_rates(
        start: float,
        end: float,
        count: int,
    ) -> list[float]:
        ...


    # TODO: Return the losses smoothed by a running average with factor beta.
    #   The average starts at 0, and each loss updates it to
    #   beta * average + (1 - beta) * loss. Keep the running average itself
    #   uncorrected. At the k-th loss, counted from 1, append the average
    #   divided by 1 - beta ** k, the same correction Adam's hats make.
    #   Example: smooth([3.0, 2.0, 2.5, 1.0], 0.9) returns about
    #   [3.0, 2.474, 2.483, 2.052].
    def smooth(
        losses: list[float],
        beta: float,
    ) -> list[float]:
        ...


    # TODO: smoothed[i] is the smoothed loss at the rate rates[i]. Return a
    #   tenth of the rate where the smoothed loss is lowest.
    #   Example: pick_rate([0.001, 0.01, 0.1, 1.0], [2.6, 2.4, 2.5, 9.0])
    #   returns 0.001.
    def pick_rate(
        rates: list[float],
        smoothed: list[float],
    ) -> float:
        ...


    # TODO: Finish the line that sets the learning rate, so that each pass
    #   through the loop trains one step at the next rate in rates. range_test
    #   returns the list of every step's loss.
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
        "range_test": "Set the learning rate to rate, the loop variable that holds this step's rate.",
    },
    introduces=(),
    check=check_schedule2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def train_names_on_schedule(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rate_at: Callable[[int], float],
    steps: int,
    batch: int,
    generator: torch.Generator,
) -> None:
    X, Y = context_sets()["train"]
    model.train()

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = rate_at(step)

        rows = torch.randint(0, len(X), (batch,), generator=generator)
        loss = F.cross_entropy(model(X[rows]), Y[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def names_validation(model: nn.Module) -> float:
    validation_x, validation_y = context_sets()["validation"]

    return reference_validation_loss(model, validation_x, validation_y)


SCHEDULE_STEPS = 5000


def schedule_rate(
    schedule: str,
    base: float,
    step: int,
) -> float:
    if schedule == "constant":
        return base

    if schedule == "step":
        return base if step < SCHEDULE_STEPS * 3 // 4 else base / 10

    return reference_cosine_rate(base, step, SCHEDULE_STEPS, 0.0)


def measure_schedules(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[tuple[str, str], list[float]]:
    """schedule1's table: the names model for 5000 steps at batch 32, each optimizer under each schedule. Step decay
    divides the rate by 10 for the last quarter, and cosine goes down to 0. The validation loss per seed."""
    results: dict[tuple[str, str], list[float]] = {}

    with pinned_threads():
        for name, make, base in (("SGD at 0.1", torch.optim.SGD, 0.1), ("Adam at 0.003", torch.optim.Adam, 0.003)):
            for schedule in ("constant", "step", "cosine"):
                for seed in seeds:
                    model = names_model(seed)
                    train_names_on_schedule(
                        model=model,
                        optimizer=make(model.parameters(), lr=base),
                        rate_at=lambda step: schedule_rate(schedule, base, step),
                        steps=SCHEDULE_STEPS,
                        batch=32,
                        generator=torch.Generator().manual_seed(seed + 1),
                    )
                    results.setdefault((name, schedule), []).append(names_validation(model))

    return results


class PostNormBlock(nn.Module):
    """residual4's post norm block, the norm of x plus an MLP of x, where Vaswani and others put the norm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.down(torch.relu(self.up(x))))


class PreNormBlock(nn.Module):
    """residual3's pre norm block, x plus an MLP of the norm of x."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(torch.relu(self.up(self.norm(x))))


class DeepNamesModel(nn.Module):
    """names2's model with post norm blocks, or pre norm blocks and a last norm as GPT-2 has, between its hidden
    layer and its head."""

    def __init__(
        self,
        blocks: int,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(27, 10)
        self.hidden = nn.Linear(30, 200)
        self.norm = nn.LayerNorm(200)
        block = PreNormBlock if pre_norm else PostNormBlock
        self.blocks = nn.ModuleList([block(200) for _ in range(blocks)])
        self.last_norm = nn.LayerNorm(200) if pre_norm else nn.Identity()
        self.head = nn.Linear(200, 27)

        with torch.no_grad():
            self.head.weight *= 0.1
            self.head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.tanh(self.norm(self.hidden(self.embedding(x).reshape(len(x), -1))))

        for block in self.blocks:
            hidden = block(hidden)

        return self.head(self.last_norm(hidden))


WARMUP_BLOCKS = 32
WARMUP_STEPS = 1500


def measure_warmup(
    runs: tuple[tuple[str, float, int], ...] = (
        ("post", 0.0003, 0),
        ("post", 0.0005, 0),
        ("post", 0.0007, 0),
        ("post", 0.0007, 500),
        ("post", 0.001, 0),
        ("post", 0.001, 150),
        ("pre", 0.001, 0),
    ),
    seeds: tuple[int, ...] = (0, 1, 2),
    blocks: int = WARMUP_BLOCKS,
) -> dict[tuple[str, float, int], list[float]]:
    """warmup1's table: a stack of post norm blocks, or pre norm blocks, trained with Adam for 1500 steps at batch 64,
    the rate falling along a cosine to 0, with and without a warmup. Each run is a (norm, rate, warmup steps) triple.
    The validation loss per seed."""
    results: dict[tuple[str, float, int], list[float]] = {}

    with pinned_threads():
        for norm, base, warmup in runs:
            for seed in seeds:
                torch.manual_seed(seed)
                model = DeepNamesModel(blocks, pre_norm=norm == "pre")
                train_names_on_schedule(
                    model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=base),
                    rate_at=lambda step: reference_warmup_cosine(base, step, warmup, WARMUP_STEPS, 0.0),
                    steps=WARMUP_STEPS,
                    batch=64,
                    generator=torch.Generator().manual_seed(seed + 1),
                )
                results.setdefault((norm, base, warmup), []).append(names_validation(model))

    return results


RANGE_STEPS = 300

# A heavier average, such as fast.ai's 0.98, lags the losses by tens of steps over a climb of only 300, so its
# lowest point lands at a rate several times too high.
RANGE_BETA = 0.9

# The steps whose losses schedule2's table shows, spread from the smallest rate to past the blowup.
RANGE_POINTS = (0, 60, 120, 165, 225, 255, 270, 285)


def measure_range_tests(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, object]]:
    """schedule2's runs: per seed, a range test of 300 Adam steps on the names model with the rate climbing from
    0.00001 to 1 at batch 256, smoothed with RANGE_BETA. Then the rate it picks, and the validation loss of Adam
    trained at that rate for 5000 steps at batch 32."""
    X, Y = context_sets()["train"]
    rates = [0.00001 * (1.0 / 0.00001) ** (k / (RANGE_STEPS - 1)) for k in range(RANGE_STEPS)]
    results: list[dict[str, object]] = []

    with pinned_threads():
        for seed in seeds:
            losses = reference_range_test(
                names_model(seed),
                X,
                Y,
                rates,
                torch.Generator().manual_seed(seed + 1),
            )
            smoothed = reference_smooth(losses, RANGE_BETA)
            bottom = smoothed.index(min(smoothed))
            doubled = next((k for k in range(bottom, RANGE_STEPS) if smoothed[k] > 2 * smoothed[bottom]), None)
            pick = rates[bottom] / 10
            model = names_model(seed)
            train_names_on_schedule(
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=pick),
                rate_at=lambda step: pick,
                steps=SCHEDULE_STEPS,
                batch=32,
                generator=torch.Generator().manual_seed(seed + 1),
            )
            results.append(
                {
                    "rates": [rates[k] for k in RANGE_POINTS],
                    "losses": [losses[k] for k in RANGE_POINTS],
                    "lowest": rates[bottom],
                    "doubled": None if doubled is None else rates[doubled],
                    "pick": pick,
                    "trained": names_validation(model),
                },
            )

    return results
