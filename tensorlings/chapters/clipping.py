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
from .dropout import labelled_points
from .initialization import expect_numbers

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
        gradient.

        ## Further reading

        - [Pascanu and others, on the difficulty of training recurrent networks](https://arxiv.org/abs/1211.5063)
        - [PyTorch, clip_grad_norm_](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)
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
    read long sequences, whose gradients can explode. GPT-3 and nanoGPT clip
    at 1.0, and most language models do the same.
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
    Here the names model trained for 3000 steps, and at step 1000 one loss was
    multiplied by 100 or by 1000. That stands in for one bad minibatch, like a
    corrupt example or a loss that spiked. The table gives the validation loss
    right after that step, 10 and 100 steps later, and at the end.

    | Optimizer | Spike | After | 10 later | 100 later | End |
    |---|---|---|---|---|---|
    | SGD at 0.1 | none | 2.41 | 2.43 | 2.40 | 2.335 |
    | SGD at 0.1 | 100 times | 15.8 | 6.59 | 2.80 | 2.377 |
    | SGD at 0.1 | 1000 times | 88.7 | 43.3 | 8.69 | 3.361 |
    | Adam at 0.003 | none | 2.38 | 2.38 | 2.37 | 2.291 |
    | Adam at 0.003 | 1000 times | 2.40 | 3.67 | 3.08 | 2.338 |

    One step undid most of 1000 steps of SGD, and 2000 more didn't fully
    repair it. Adam's step stays about its rate, but the spike filled its
    averages and threw the next steps off.

    With the global norm clipped at 1.0, both spikes changed nothing. The
    runs came out the same as without a spike, since the spike's gradient
    shrank to the length of an ordinary one. Ordinary gradients here had
    norms near 1.7, and the spikes 174 and 1742.

    Clipping at 1.0 caught ordinary steps too, and that cost nothing here.
    Logging the norm each step shows where the spikes are, so a bad batch can
    be found and looked at.

    A spike stands far above the median of the norms, their middle value.
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
