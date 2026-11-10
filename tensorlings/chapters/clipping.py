"""27_clipping, the global gradient norm, clipping it, and one huge gradient held in check."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    NOT_STARTED,
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .adaptive import pinned_threads
from .dropout import labelled_points
from .initialization import expect_numbers
from .names import context_sets, names_model, reference_validation_loss

CLIPPING = Chapter(
    folder="27_clipping",
    title="Gradient clipping",
    readme=source_file(
        """
        # Gradient clipping

        Now and then one minibatch gives a gradient far larger than the
        others. A single update with that gradient can undo much of the
        training that came before it. This chapter measures the size of a
        gradient and limits that size the way PyTorch's `clip_grad_norm_`
        does. Then it shows the limit protecting a training run from one huge
        gradient, and how to choose the limit from the gradient sizes logged
        during training.

        ## Further reading

        - [Pascanu and others, on the difficulty of training recurrent networks](https://arxiv.org/abs/1211.5063)
        - [PyTorch, clip_grad_norm_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)
        - [Google, deep learning tuning playbook, on picking the cap](https://github.com/google-research/tuning_playbook)
        """,
    ),
)


def model_with_gradients(
    seed: int,
    scale: float,
) -> nn.Sequential:
    """A small model after one backward pass, its loss scaled so the gradients come out as large as asked."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    x, y = labelled_points(seed, 20, 4, 3)
    (scale * F.cross_entropy(model(x), y)).backward()

    return model


def reference_global_norm(parameters: list[nn.Parameter]) -> float:
    return math.sqrt(sum((parameter.grad ** 2).sum().item() for parameter in parameters if parameter.grad is not None))


# ---------------------------------------------------------------------------
# clipping1
# ---------------------------------------------------------------------------

CLIPPING1_LESSON = prose(
    """
    The size of a gradient is measured by its **norm**, the square root of the
    sum of the squares of its entries. If you picture the gradient as an
    arrow, the norm is the arrow's length. This norm is only a measure of
    size, and it has nothing to do with the norm layers of Section 3. The
    global norm treats all the gradients in the model as one long vector. You
    square every entry of every gradient, add them all up, and take the square
    root of the total.

    ```python
    grads = [torch.tensor([3.0]), torch.tensor([0.0, 4.0])]
    math.sqrt(sum((g ** 2).sum().item() for g in grads))    # 5.0
    ```

    **Gradient clipping** puts a cap, an upper limit, on that size. When the
    global norm is above `max_norm`, every gradient is multiplied by
    `max_norm / norm`, which brings the global norm down to `max_norm`. All
    the gradients are scaled by the same factor, so their direction stays the
    same and only their length shrinks. When the global norm is at or below
    the cap, nothing changes.

    `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` does
    this in place. You call it after `backward` and before `step`. It returns
    the global norm from before clipping, which is worth logging. It actually
    divides by `norm + 1e-6` instead of `norm`, but that difference is too
    small to matter.

    Pascanu and others proposed clipping the norm in 2013. They worked on
    models that read long sequences, whose gradients can explode, which means
    grow huge. The earlier method capped each entry separately, as
    `torch.nn.utils.clip_grad_value_` still does. Capping entries separately
    changes the direction of the gradient. Scaling every entry by the same
    factor keeps the direction. GPT-3 and nanoGPT clip the global norm at 1.0.
    """,
)

CLIPPING1_SOURCE = source_file(
    '''
    # clipping1.py

    import math

    import torch
    import torch.nn as nn

    grads = [torch.tensor([3.0]), torch.tensor([0.0, 4.0])]
    length = math.sqrt(sum((g ** 2).sum().item() for g in grads))    # 5.0


    # TODO: Return the global norm of the gradients of parameters as a float.
    #   Square every entry of every .grad, add them all up, and take the square
    #   root of the total. Skip any parameter whose .grad is None. The line that
    #   sets length above does the same for the plain tensors in grads.
    def global_norm(parameters: list[nn.Parameter]) -> float:
        ...


    # TODO: Finish the line in the loop. When the global norm is above max_norm,
    #   every gradient must be multiplied in place by max_norm / norm. The
    #   function returns the global norm from before clipping.
    def clip_gradients_(
        parameters,
        max_norm: float,
    ) -> float:
        parameters = list(parameters)
        norm = global_norm(parameters)

        if norm > max_norm:
            for parameter in parameters:
                if parameter.grad is not None:
                    ...

        return norm


    # TODO: norms holds the global norm of each training step. Return the
    #   fraction of those steps whose norm is above max_norm, which are the
    #   steps that clipping would change.
    #   Example: clipped_fraction([0.5, 1.5, 0.9, 3.0], 1.0) returns 0.5.
    def clipped_fraction(
        norms: list[float],
        max_norm: float,
    ) -> float:
        ...
    ''',
)

CLIPPING1_SOLUTION = source_file(
    '''
    # clipping1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn

    grads = [torch.tensor([3.0]), torch.tensor([0.0, 4.0])]
    length = math.sqrt(sum((g ** 2).sum().item() for g in grads))


    def global_norm(parameters: list[nn.Parameter]) -> float:
        return math.sqrt(sum((parameter.grad ** 2).sum().item() for parameter in parameters if parameter.grad is not None))


    def clip_gradients_(
        parameters,
        max_norm: float,
    ) -> float:
        parameters = list(parameters)
        norm = global_norm(parameters)

        if norm > max_norm:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad *= max_norm / norm

        return norm


    def clipped_fraction(
        norms: list[float],
        max_norm: float,
    ) -> float:
        return sum(norm > max_norm for norm in norms) / len(norms)
    ''',
)


def check_global_norm(found: object) -> None:
    for seed, scale in ((580, 1.0), (581, 50.0)):
        model = model_with_gradients(seed, scale)
        expected = reference_global_norm(list(model.parameters()))
        expect_close(
            call=f"global_norm of a small model's parameters, with a global norm near {expected:.3g}",
            actual=found(list(model.parameters())),
            expected=expected,
            tolerance=1e-4 * expected,
        )

    unused = nn.Parameter(torch.ones(3))
    used = nn.Parameter(torch.ones(2))
    (3 * used).sum().backward()
    expect_close(
        call="global_norm of two parameters, one of them without a gradient",
        actual=found([used, unused]),
        expected=math.sqrt(18),
        tolerance=1e-5,
    )


def check_clip_gradients(
    found: object,
    global_norm: object,
) -> None:
    if callable(global_norm) and global_norm(list(model_with_gradients(582, 50.0).parameters())) is None:
        raise Mismatch(f"global_norm returns None, which usually means it is {NOT_STARTED}. clip_gradients_ needs it, so write it first.")

    for seed, scale, max_norm in ((582, 50.0, 2.0), (583, 1.0, 100.0)):
        mine = model_with_gradients(seed, scale)
        theirs = model_with_gradients(seed, scale)
        call = f"clip_gradients_ with max_norm {max_norm} on a small model's gradients"
        returned = found(mine.parameters(), max_norm)
        expected = torch.nn.utils.clip_grad_norm_(theirs.parameters(), max_norm).item()
        expect_close(
            call=f"the norm {call} returns",
            actual=returned,
            expected=expected,
            tolerance=1e-4 * expected,
        )

        for (name, parameter), reference in zip(mine.named_parameters(), theirs.parameters()):
            expect_tensor(
                call=f"the gradient of {name} after {call}",
                actual=parameter.grad,
                expected=reference.grad,
            )


def check_clipping1(checker: Checker) -> None:
    checker.test(
        function="global_norm",
        description="gives the global norm of all the gradients together",
        run=check_global_norm,
    )
    checker.test(
        function="clip_gradients_",
        description="scales the gradients like clip_grad_norm_",
        run=lambda found: check_clip_gradients(
            found=found,
            global_norm=getattr(checker.module, "global_norm", None),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="clipped_fraction",
        description="gives the fraction of steps whose norm is above max_norm",
        inputs=(
            ([0.5, 1.5, 0.9, 3.0], 1.0),
            ([0.2, 0.3], 1.0),
        ),
        reference=lambda norms, max_norm: sum(norm > max_norm for norm in norms) / len(norms),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
        ),
    )


CLIPPING1 = Exercise(
    name="clipping1",
    chapter=CLIPPING,
    title="Clipping the global norm",
    lesson=CLIPPING1_LESSON,
    source=CLIPPING1_SOURCE,
    solution=CLIPPING1_SOLUTION,
    hints={
        "global_norm": "Add up (parameter.grad ** 2).sum().item() over the parameters whose .grad isn't None, then "
        "take math.sqrt of the total.",
        "clip_gradients_": "Multiply parameter.grad by max_norm / norm in place, using *=.",
        "clipped_fraction": "Count the norms above max_norm, then divide by how many norms there are.",
    },
    introduces=(),
    check=check_clipping1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# clipping2
# ---------------------------------------------------------------------------

CLIPPING2_LESSON = prose(
    """
    Here the names model was trained for 3000 steps with three different
    seeds. It used SGD with a learning rate of 0.1 or Adam with a learning
    rate of 0.003. At step 1000, the loss of that one step was multiplied by
    100 or by 1000. This imitates one bad minibatch, such as one with a
    corrupt example, whose loss spiked. The table shows the validation loss
    right after that step, 10 steps later, 100 steps later, and at the end.
    Each cell gives the range over the three seeds.

    | Run | After | 10 later | 100 later | End |
    |---|---|---|---|---|
    | SGD, no spike | 2.41 to 2.42 | 2.40 to 2.45 | 2.39 to 2.41 | 2.335 to 2.352 |
    | SGD, 100 times | 12.0 to 16.4 | 6.14 to 6.35 | 2.81 to 2.97 | 2.377 to 2.429 |
    | SGD, 1000 times | 88.7 to 116 | 35.1 to 41.9 | 6.49 to 15.6 | 2.861 to 3.361 |
    | SGD, 1000 times, clipped at 1.9 | 2.41 to 2.42 | 2.40 to 2.45 | 2.39 to 2.41 | 2.335 to 2.352 |
    | Adam, no spike | 2.37 to 2.38 | 2.38 | 2.36 to 2.38 | 2.291 to 2.306 |
    | Adam, 1000 times | 2.40 to 2.42 | 3.77 to 3.95 | 2.93 to 3.07 | 2.330 to 2.338 |
    | Adam, 1000 times, clipped at 1.6 | 2.37 to 2.38 | 2.38 | 2.36 to 2.38 | 2.291 to 2.305 |

    For SGD, the one spiked step undid all 1000 steps of training before it,
    and more. The 2000 steps after it did not repair the damage. Adam's step
    size stays close to its learning rate even for a huge gradient. But the
    spike filled Adam's running averages with huge values, and those averages
    made the next steps go wrong.

    Ordinary SGD gradients had global norms near 1.7. The spiked gradients
    had norms near 170 and 1700. With clipping, the spiked gradient was scaled
    down to the size of an ordinary one. So the runs came out as if there had
    been no spike.

    You choose the cap from the norms logged during training. Google's tuning
    playbook sets it near the 90th percentile of those norms, the value that
    9 in 10 norms stay below. Here that was 1.9 for SGD and 1.6 for Adam.
    With those caps, `clipped_fraction` from clipping1 gave 0.09 to 0.13 on
    the logged norms. So only the unusually large gradients were clipped.

    For SGD, a cap of 1.0 also protected the run from the spike, but
    `clipped_fraction` came out at 1.0. Every step was clipped, which acts
    like a smaller learning rate. When more than half the steps get clipped,
    the playbook lowers the learning rate instead.

    Logging the norm at each step also shows which steps had spikes, so you
    can find the bad batch and inspect it. A spike is a norm far above the
    median of the norms, their middle value when sorted. A few spikes pull
    the mean up a lot, but they barely change the median. For an even number
    of values, PyTorch's `median()` returns the lower of the two middle
    values.
    """,
)

CLIPPING2_SOURCE = source_file(
    '''
    # clipping2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    norms = torch.tensor([1.2, 1.7, 160.0, 1.5])
    middle = norms.median()                          # tensor(1.5000)


    # TODO: Finish the line that sets norm. Each step uses
    #   torch.nn.utils.clip_grad_norm_ to clip the global norm of the gradients
    #   at cap. When max_norm is None, cap is infinity, so the call only
    #   measures the norm. train_with_spike returns the last loss and the list
    #   of every step's norm.
    def train_with_spike(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        spike_step: int,
        max_norm: float | None,
        generator: torch.Generator,
    ) -> tuple[float, list[float]]:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        norms = []

        for step in range(steps):
            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])

            if step == spike_step:
                loss = loss * 1000

            optimizer.zero_grad()
            loss.backward()
            cap = float("inf") if max_norm is None else max_norm
            norm = ...
            norms.append(norm.item())
            optimizer.step()

        return loss.item(), norms


    # TODO: norms holds the global norm of each step. Return the list of step
    #   numbers, counted from 0, whose norm is more than factor times the
    #   median of norms.
    #   Example: spike_steps([1.2, 1.7, 160.0, 1.5, 1.4], 10.0) returns [2].
    def spike_steps(
        norms: list[float],
        factor: float,
    ) -> list[int]:
        ...
    ''',
)

CLIPPING2_SOLUTION = source_file(
    '''
    # clipping2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    norms = torch.tensor([1.2, 1.7, 160.0, 1.5])
    middle = norms.median()


    def train_with_spike(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        steps: int,
        spike_step: int,
        max_norm: float | None,
        generator: torch.Generator,
    ) -> tuple[float, list[float]]:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        norms = []

        for step in range(steps):
            batch = torch.randint(0, len(X), (32,), generator=generator)
            loss = F.cross_entropy(model(X[batch]), Y[batch])

            if step == spike_step:
                loss = loss * 1000

            optimizer.zero_grad()
            loss.backward()
            cap = float("inf") if max_norm is None else max_norm
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cap)
            norms.append(norm.item())
            optimizer.step()

        return loss.item(), norms


    def spike_steps(
        norms: list[float],
        factor: float,
    ) -> list[int]:
        middle = torch.tensor(norms).median().item()

        return [step for step, norm in enumerate(norms) if norm > factor * middle]
    ''',
)


def small_names_like(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))


def reference_train_with_spike(
    model: nn.Module,
    X: Tensor,
    Y: Tensor,
    steps: int,
    spike_step: int,
    max_norm: float | None,
    generator: torch.Generator,
) -> tuple[float, list[float]]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    norms = []

    for step in range(steps):
        batch = torch.randint(0, len(X), (32,), generator=generator)
        loss = F.cross_entropy(model(X[batch]), Y[batch])

        if step == spike_step:
            loss = loss * 1000

        optimizer.zero_grad()
        loss.backward()
        cap = float("inf") if max_norm is None else max_norm
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), cap).item())
        optimizer.step()

    return loss.item(), norms


def check_train_with_spike(found: object) -> None:
    X, Y = labelled_points(640, 200, 6, 3)

    for max_norm in (None, 1.0):
        call = f"train_with_spike for 40 steps with a spike at step 20 and max_norm {max_norm}"
        actual = found(small_names_like(641), X, Y, 40, 20, max_norm, torch.Generator().manual_seed(642))
        expected = reference_train_with_spike(small_names_like(641), X, Y, 40, 20, max_norm, torch.Generator().manual_seed(642))

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should give two things, the last loss and the list of norms.")

        expect_numbers(
            call=f"the norms from {call}",
            actual=actual[1],
            expected=expected[1],
        )
        expect_close(
            call=f"the last loss from {call}",
            actual=actual[0],
            expected=expected[0],
            tolerance=1e-4,
        )


def check_clipping2(checker: Checker) -> None:
    checker.test(
        function="train_with_spike",
        description="clips each step's gradient and records its norm",
        run=check_train_with_spike,
    )
    check_on_inputs(
        checker=checker,
        function="spike_steps",
        description="finds the steps whose norm is more than factor times the median",
        inputs=(
            ([1.2, 1.7, 160.0, 1.5, 1.4], 10.0),
            ([1.0, 30.0, 1.0, 2.0, 25.0], 5.0),
        ),
        reference=lambda norms, factor: [step for step, norm in enumerate(norms) if norm > factor * torch.tensor(norms).median().item()],
        expect=expect_value,
    )


CLIPPING2 = Exercise(
    name="clipping2",
    chapter=CLIPPING,
    title="One huge gradient",
    lesson=CLIPPING2_LESSON,
    source=CLIPPING2_SOURCE,
    solution=CLIPPING2_SOLUTION,
    hints={
        "train_with_spike": "Call torch.nn.utils.clip_grad_norm_ with model.parameters() and cap. It clips the "
        "gradients in place and returns the global norm from before clipping.",
        "spike_steps": "Take the median with torch.tensor(norms).median().item(). Then keep each step whose norm is "
        "above factor times that median, using enumerate to get the step numbers.",
    },
    introduces=(),
    check=check_clipping2,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

# Each run is (optimizer, how many times the loss at the spike step is scaled, the cap or None for no clipping). SGD's
# cap of 1.9 and Adam's of 1.6 are the 90th percentiles of their norms without a spike, as measure_spike reports them.
SPIKE_RUNS: tuple[tuple[str, float, float | None], ...] = (
    ("SGD", 1.0, None),
    ("SGD", 100.0, None),
    ("SGD", 1000.0, None),
    ("SGD", 1000.0, 1.9),
    ("SGD", 1000.0, 1.0),
    ("Adam", 1.0, None),
    ("Adam", 1000.0, None),
    ("Adam", 1000.0, 1.6),
)


def measure_spike(
    runs: tuple[tuple[str, float, float | None], ...] = SPIKE_RUNS,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 3000,
    spike_step: int = 1000,
) -> dict[tuple[str, float, float | None], list[dict[str, float]]]:
    """clipping2's table: the names model trained with minibatches of 32 by SGD at 0.1 or Adam at 0.003, one loss
    scaled at spike_step, clipped at a cap or not. Per seed, the validation loss right after the spike step, 10 and 100
    steps later and at the end, the spike's norm, the median and 90th percentile of the other steps' norms, and the
    share of steps above the cap."""
    X, Y = context_sets()["train"]
    validation_x, validation_y = context_sets()["validation"]
    marks = {spike_step: "after", spike_step + 10: "10 later", spike_step + 100: "100 later", steps - 1: "end"}
    results: dict[tuple[str, float, float | None], list[dict[str, float]]] = {}

    with pinned_threads():
        for optimizer_name, factor, cap in runs:
            for seed in seeds:
                model = names_model(seed)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1) if optimizer_name == "SGD" else torch.optim.Adam(model.parameters(), lr=0.003)
                generator = torch.Generator().manual_seed(seed + 1)
                norms: list[float] = []
                row: dict[str, float] = {}

                for step in range(steps):
                    model.train()
                    batch = torch.randint(0, len(X), (32,), generator=generator)
                    loss = F.cross_entropy(model(X[batch]), Y[batch])

                    if step == spike_step:
                        loss = loss * factor

                    optimizer.zero_grad()
                    loss.backward()
                    norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if cap is None else cap).item())
                    optimizer.step()

                    if step in marks:
                        row[marks[step]] = reference_validation_loss(model, validation_x, validation_y)

                others = torch.tensor(norms[:spike_step] + norms[spike_step + 1:])
                row["spike norm"] = norms[spike_step]
                row["median norm"] = others.median().item()
                row["90th percentile"] = others.quantile(0.9).item()
                row["share clipped"] = 0.0 if cap is None else sum(norm > cap for norm in norms) / len(norms)
                results.setdefault((optimizer_name, factor, cap), []).append(row)

    return results
