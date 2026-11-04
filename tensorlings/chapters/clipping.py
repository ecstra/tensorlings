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

        Now and then one minibatch gives a gradient far larger than the rest,
        and one step on it can undo much of the training before it. This
        chapter measures a gradient's size, caps it the way PyTorch's
        `clip_grad_norm_` does, and shows the cap holding through one huge
        gradient, with the cap picked from the logged norms.

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
    The size of a gradient is its **norm**, the square root of the sum of its
    squares, like the length of an arrow. This norm is a size, nothing to do
    with the norm layers of Section 3. The global norm treats every gradient
    in the model as one long vector. Square every entry of every gradient, add
    them all up, and take the root.

    ```python
    grads = [torch.tensor([3.0]), torch.tensor([0.0, 4.0])]
    math.sqrt(sum((g ** 2).sum().item() for g in grads))    # 5.0
    ```

    **Gradient clipping** caps that size. When the global norm is above
    `max_norm`, every gradient is multiplied by `max_norm / norm`. The
    direction stays and only the length shrinks. Below the cap, nothing
    changes.

    `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` does
    this in place, between `backward` and `step`. It returns the norm it found
    before clipping, which is worth logging. It divides by `norm + 1e-6`, a
    difference too small to matter.

    Pascanu and others proposed clipping the norm in 2013, for models that
    read long sequences, whose gradients can explode. The earlier way capped
    each entry on its own, as `torch.nn.utils.clip_grad_value_` still does.
    That turns the direction, and scaling the norm keeps it. GPT-3 and
    nanoGPT clip the norm at 1.0.
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


    # TODO: Return the global norm of the gradients of parameters, as a plain
    #   number, skipping any parameter whose .grad is None.
    def global_norm(parameters: list[nn.Parameter]) -> float:
        ...


    # TODO: Finish the line in the loop, so when the global norm is above
    #   max_norm every gradient is scaled in place by max_norm / norm. It
    #   returns the norm from before.
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


    # TODO: Return the share of the steps whose norm in norms is above max_norm,
    #   the steps clipping would change.
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
        raise Mismatch(f"global_norm gives None, which usually means it is {NOT_STARTED}. clip_gradients_ needs it, so write it first.")

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
        description="gives the size of all the gradients together",
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
        description="gives the share of steps above the cap",
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
        "clip_gradients_": "Multiply parameter.grad in place with *= by max_norm / norm.",
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
    Here the names model trained for 3000 steps on three seeds, with SGD at
    0.1 or Adam at 0.003. At step 1000 one loss was multiplied by 100 or by
    1000. That stands in for one bad minibatch, like a corrupt example or a
    loss that spiked. The table gives the validation loss right after that
    step, 10 and 100 steps later, and at the end.

    | Run | After | 10 later | 100 later | End |
    |---|---|---|---|---|
    | SGD, no spike | 2.41 to 2.42 | 2.40 to 2.45 | 2.39 to 2.41 | 2.335 to 2.352 |
    | SGD, 100 times | 12.0 to 16.4 | 6.14 to 6.35 | 2.81 to 2.97 | 2.377 to 2.429 |
    | SGD, 1000 times | 88.7 to 116 | 35.1 to 41.9 | 6.49 to 15.6 | 2.861 to 3.361 |
    | SGD, 1000 times, clipped at 1.9 | 2.41 to 2.42 | 2.40 to 2.45 | 2.39 to 2.41 | 2.335 to 2.352 |
    | Adam, no spike | 2.37 to 2.38 | 2.38 | 2.36 to 2.38 | 2.291 to 2.306 |
    | Adam, 1000 times | 2.40 to 2.42 | 3.77 to 3.95 | 2.93 to 3.07 | 2.330 to 2.338 |
    | Adam, 1000 times, clipped at 1.6 | 2.37 to 2.38 | 2.38 | 2.36 to 2.38 | 2.291 to 2.305 |

    One step undid all 1000 steps of SGD and more, and 2000 more didn't
    repair it. Adam's step stays about its rate, but the spike filled its
    averages and threw the next steps off.

    Ordinary SGD gradients had norms near 1.7, and the spikes near 170 and
    1700. Clipped, the spike's gradient shrank to the size of an ordinary
    one, and the runs came out as if there were no spike.

    The cap comes from the logged norms. Google's tuning playbook puts it
    near their 90th percentile, the value 9 in 10 norms stay below. Here that
    was 1.9 for SGD and 1.6 for Adam. At those caps clipping1's
    `clipped_fraction` of the norms was 0.09 to 0.13, so only the steps that
    stand out were cut.

    For SGD a cap of 1.0 held the spike too, but `clipped_fraction` came out
    at 1.0. It cut every step, which works as a smaller learning rate. When
    more than half the steps get clipped, the playbook lowers the rate
    instead.

    Logging the norm each step also shows where the spikes are, so a bad
    batch can be found and looked at. A spike stands far above the median of
    the norms, their middle value.
    Spikes don't pull the median up the way they pull up a mean. For an even
    count, PyTorch's `median()` takes the lower of the two middle values.
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


    # TODO: Finish the line that sets norm, so each step clips the global norm at
    #   max_norm with torch.nn.utils.clip_grad_norm_, or only measures it when
    #   max_norm is None. This returns the last loss and every step's norm.
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


    # TODO: Return the steps, counted from 0, whose norm is more than factor
    #   times the median of norms.
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
        description="finds the steps far above the median norm",
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
        "train_with_spike": "Call torch.nn.utils.clip_grad_norm_ with model.parameters() and cap. It clips in place "
        "and returns the norm from before.",
        "spike_steps": "Take the median with torch.tensor(norms).median().item(), then keep each step whose norm is "
        "above factor times it, using enumerate for the step.",
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
