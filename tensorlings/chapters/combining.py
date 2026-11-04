"""104_combining, how the course's pieces join into a module of your own: FiLM, gating against adding, a new branch that starts as do nothing, mixing across and within positions, and picking the structure for a problem you haven't seen."""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module

COMBINING = Chapter(
    folder="104_combining",
    title="Combining",
    readme=source_file(
        """
        # Combining

        Section 14 is about building modules of your own. Every structure
        assumes something about the problem, and a module is a choice of
        which assumptions to make. This chapter joins the pieces the course
        built: a side input that scales features, a gate against a sum, a
        new branch on a trained model, mixing across and within positions,
        and a guide to picking the structure for a problem you haven't seen.

        ## Further reading

        - [Perez and colleagues, FiLM](https://arxiv.org/abs/1709.07871)
        - [Tolstikhin and colleagues, MLP-Mixer](https://arxiv.org/abs/2105.01601)
        - [Zhang and colleagues, ControlNet](https://arxiv.org/abs/2302.05543)
        - [Bachlechner and colleagues, ReZero](https://arxiv.org/abs/2003.04887)
        - [Dauphin and colleagues, gated linear units](https://arxiv.org/abs/1612.08083)
        """,
    ),
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count moves the last digits of a run, so the measurements pin it."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def seeded_model(
    make: Callable[..., nn.Module],
    *arguments: object,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def recording_hook(
    model: nn.Module,
    seen: list[bool],
) -> torch.utils.hooks.RemovableHandle:
    """Notes on every call whether autograd was recording, since a model's outputs can't show it."""
    return model.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))


# ---------------------------------------------------------------------------
# combine1
# ---------------------------------------------------------------------------

POSITIONS = 6
FEATURES = 4
INSTRUCTIONS = 8

# Instructions 0 to 3 read features 0 to 3, and 4 to 7 read them with the sign flipped.
SIGNS = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


def instruction_data(
    count: int,
    generator: torch.Generator,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs of 6 positions of 4 features, a one hot instruction each, and the answer the instruction asks for."""
    x = (torch.rand(count, POSITIONS, FEATURES, generator=generator) * 2 - 1) * scale
    instruction = torch.randint(0, INSTRUCTIONS, (count,), generator=generator)
    condition = F.one_hot(instruction, INSTRUCTIONS).float()
    picked = x.mean(dim=1).gather(1, (instruction % FEATURES).unsqueeze(1)).squeeze(1)

    return x, condition, SIGNS[instruction] * picked


class ReferenceFiLM(nn.Module):
    def __init__(
        self,
        condition_size: int,
        channels: int,
    ) -> None:
        super().__init__()
        self.to_scale = nn.Linear(condition_size, channels)
        self.to_shift = nn.Linear(condition_size, channels)

    def forward(
        self,
        features: Tensor,
        condition: Tensor,
    ) -> Tensor:
        return self.to_scale(condition).unsqueeze(1) * features + self.to_shift(condition).unsqueeze(1)


def reference_concatenated(
    features: Tensor,
    condition: Tensor,
) -> Tensor:
    return torch.cat([features, condition.unsqueeze(1).expand(-1, features.shape[1], -1)], dim=-1)


class ReferenceFilmNet(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURES, hidden)
        self.film = ReferenceFiLM(INSTRUCTIONS, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tensor:
        h = torch.relu(self.film(self.embed(x), condition))

        return self.head(h.mean(dim=1)).squeeze(1)


class ConcatNet(nn.Module):
    """The condition joined onto every position, one hidden layer or two."""

    def __init__(
        self,
        hidden: int = 16,
        deep: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURES + INSTRUCTIONS, hidden)
        self.middle = nn.Linear(hidden, hidden) if deep else None
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tensor:
        h = torch.relu(self.embed(reference_concatenated(x, condition)))

        if self.middle is not None:
            h = torch.relu(self.middle(h))

        return self.head(h.mean(dim=1)).squeeze(1)


def train_conditioned(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    y: Tensor,
    steps: int,
    seed: int,
    lr: float,
) -> nn.Module:
    """The runs behind combine1's and combine2's tables, Adam on batches of 128."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        batch = torch.randint(0, len(y), (128,), generator=generator)
        loss = F.mse_loss(model(*(part[batch] for part in inputs)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def conditioned_error(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    y: Tensor,
) -> float:
    with torch.no_grad():
        return F.mse_loss(model(*inputs), y).item()


COMBINE1_LESSON = prose(
    """
    Section 14 is about building modules of your own. Each structure assumes
    something about the problem, and a module is a choice of assumptions.

    Often a second input should change what a model does with the first. A
    question changes what to look for in an image. An instruction changes
    which number to read.

    The plain way is to join the condition onto every position's features.
    Perez and colleagues point out what that buys. A Linear on the joined
    input adds a bias per feature, a shift that depends on the condition.

    Their **FiLM** layer scales as well. A generator computes a scale and a
    shift from the condition, one pair per feature, the same at every position.

    ```text
    FiLM(h) = scale(condition) * h + shift(condition)
    ```

    A scale can do what a shift can't. A scale of 0 turns a feature off, and
    a negative one flips it, however large the feature is. The paper put a
    ReLU after each FiLM and found 36% of its scales negative. On the CLEVR
    reasoning benchmark it roughly halved the best error of methods without
    extra supervision, from relation networks' 4.5% to 2.3%.

    Here the condition is one of 8 instructions. Each input holds 6
    positions of 4 features. The instruction says which feature to average,
    and whether to flip its sign. Each model trained for 1,500 steps, on
    three seeds, then met new inputs, and new inputs 3 times larger.

    | Model | Weights | Error | Error, inputs 3 times larger |
    |---|---|---|---|
    | joined, one hidden layer of 16 | 225 | 0.0004 | 0.14 to 0.22 |
    | joined, two hidden layers of 32 | 1,505 | 0.0002 to 0.0003 | 0.11 to 0.22 |
    | FiLM, one hidden layer of 16 | 385 | 0.00003 to 0.00011 | 0.043 to 0.050 |

    Guessing the average scores 0.053 and 0.47. In range every model fits. On
    larger inputs FiLM's error is 2 to 5 times lower than either joined model's.

    Which half does the work? The paper's test holds one half at its average
    over the training data, so the condition no longer moves it. Here it is on
    the trained FiLM model, on new inputs, three seeds.

    | Held at its average | Error |
    |---|---|
    | nothing | 0.00003 to 0.00011 |
    | the shift | 0.013 to 0.047 |
    | the scale | 0.054 to 0.097 |

    Without its scale the model is no better than a guess. The condition works
    mostly through the scale, as on CLEVR, where the averaged scale dropped
    accuracy by 65.4% and the averaged shift by 1.0%.

    The paper's appendix makes the scale 1 plus a change, so no feature starts
    switched off, and found it made no real difference. combine3 comes back to
    such starts.
    """,
)

COMBINE1_SOURCE = source_file(
    '''
    # combine1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    scale = torch.tensor([[2.0, 0.0]])                  # one row per input, one number per channel
    spread = scale.unsqueeze(1) * torch.ones(1, 3, 2)   # tensor([[[2., 0.], [2., 0.], [2., 0.]]]), at all 3 positions


    # TODO: Finish forward. features has shape (batch, positions, channels) and
    #   condition (batch, condition_size). Compute a scale with to_scale and a
    #   shift with to_shift, one number per channel. Return scale times
    #   features plus shift, the same scale and shift at every position.
    class FiLM(nn.Module):
        def __init__(
            self,
            condition_size: int,
            channels: int,
        ) -> None:
            super().__init__()
            self.to_scale = nn.Linear(condition_size, channels)
            self.to_shift = nn.Linear(condition_size, channels)

        def forward(
            self,
            features: Tensor,
            condition: Tensor,
        ) -> Tensor:
            ...


    copies = torch.ones(2, 3).unsqueeze(1).expand(-1, 4, -1)    # shape (2, 4, 3), each row copied to 4 positions


    # TODO: Return features, of shape (batch, positions, channels), with the
    #   condition, of shape (batch, size), joined onto every position's
    #   channels. The result has shape (batch, positions, channels + size).
    def concatenated(
        features: Tensor,
        condition: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Embed x, of shape (batch, positions, 4), run film on
    #   it with the condition, then a ReLU. Average over the positions and run
    #   head, giving one answer per input, of shape (batch,).
    class FilmNet(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.embed = nn.Linear(4, hidden)
            self.film = FiLM(8, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            condition: Tensor,
        ) -> Tensor:
            ...


    # The joined model of the lesson's first row, for comparison.
    class ConcatNet(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.embed = nn.Linear(12, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            condition: Tensor,
        ) -> Tensor:
            h = torch.relu(self.embed(concatenated(x, condition)))

            return self.head(h.mean(dim=1)).squeeze(1)
    ''',
)

COMBINE1_SOLUTION = source_file(
    '''
    # combine1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    scale = torch.tensor([[2.0, 0.0]])
    spread = scale.unsqueeze(1) * torch.ones(1, 3, 2)


    class FiLM(nn.Module):
        def __init__(
            self,
            condition_size: int,
            channels: int,
        ) -> None:
            super().__init__()
            self.to_scale = nn.Linear(condition_size, channels)
            self.to_shift = nn.Linear(condition_size, channels)

        def forward(
            self,
            features: Tensor,
            condition: Tensor,
        ) -> Tensor:
            scale = self.to_scale(condition).unsqueeze(1)
            shift = self.to_shift(condition).unsqueeze(1)

            return scale * features + shift


    copies = torch.ones(2, 3).unsqueeze(1).expand(-1, 4, -1)


    def concatenated(
        features: Tensor,
        condition: Tensor,
    ) -> Tensor:
        repeated = condition.unsqueeze(1).expand(-1, features.shape[1], -1)

        return torch.cat([features, repeated], dim=-1)


    class FilmNet(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.embed = nn.Linear(4, hidden)
            self.film = FiLM(8, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            condition: Tensor,
        ) -> Tensor:
            h = torch.relu(self.film(self.embed(x), condition))

            return self.head(h.mean(dim=1)).squeeze(1)


    class ConcatNet(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.embed = nn.Linear(12, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            condition: Tensor,
        ) -> Tensor:
            h = torch.relu(self.embed(concatenated(x, condition)))

            return self.head(h.mean(dim=1)).squeeze(1)
    ''',
)


def check_film(found: type) -> None:
    for condition_size, channels, batch, positions, seed in ((3, 5, 2, 4, 941), (2, 3, 3, 2, 942)):
        layer = expect_module(f"FiLM({condition_size}, {channels})", seeded_model(found, condition_size, channels))
        features = small_tensor(seed, batch, positions, channels) / 3
        condition = small_tensor(seed + 10, batch, condition_size) / 3

        with torch.no_grad():
            expected = seeded_model(ReferenceFiLM, condition_size, channels)(features, condition)
            actual = layer(features, condition)

        expect_tensor(
            call=f"FiLM({condition_size}, {channels}) on features of shape {tuple(features.shape)} and a condition of {condition_size}",
            actual=actual,
            expected=expected,
        )


def check_film_net(found: type) -> None:
    model = expect_module("FilmNet()", seeded_model(found))
    x, condition, _ = instruction_data(5, torch.Generator().manual_seed(943))

    with torch.no_grad():
        expected = seeded_model(ReferenceFilmNet)(x, condition)
        actual = model(x, condition)

    expect_tensor(
        call="FilmNet() on 5 inputs of 6 positions and their instructions",
        actual=actual,
        expected=expected,
    )


def check_combine1(checker: Checker) -> None:
    checker.test_method(
        class_name="FiLM",
        method="forward",
        description="scales and shifts each channel by amounts the condition sets",
        run=check_film,
    )
    check_on_inputs(
        checker=checker,
        function="concatenated",
        description="joins the condition onto every position",
        inputs=(
            (small_tensor(944, 2, 3, 4), small_tensor(945, 2, 2)),
            (small_tensor(946, 3, 5, 1), small_tensor(947, 3, 3)),
        ),
        reference=reference_concatenated,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="FilmNet",
        method="forward",
        description="embeds, runs FiLM and a ReLU, averages, then reads one answer",
        run=check_film_net,
    )


COMBINE1 = Exercise(
    name="combine1",
    chapter=COMBINING,
    title="FiLM, a condition that scales",
    lesson=COMBINE1_LESSON,
    source=COMBINE1_SOURCE,
    solution=COMBINE1_SOLUTION,
    hints={
        "FiLM.forward": "self.to_scale(condition) has shape (batch, channels). Give it a dimension for the positions "
        "with .unsqueeze(1) so it multiplies every position, and the same for the shift.",
        "concatenated": "Give the condition a positions dimension with .unsqueeze(1), copy it to features.shape[1] "
        "positions with .expand, then torch.cat along the last dimension.",
        "FilmNet.forward": "Run self.embed on x, pass that and the condition to self.film, take torch.relu, average over "
        "dim=1, run self.head and squeeze the last dimension.",
    },
    introduces=(),
    check=check_combine1,
)


# ---------------------------------------------------------------------------
# combine2
# ---------------------------------------------------------------------------

# The two linear maps the control picks between in combine2's task.
MAP_A = torch.tensor([[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, 1.0, -0.5]])
MAP_B = torch.tensor([[-0.5, 0.0, 1.0, 1.0], [1.0, 0.5, 0.0, -1.0]])


def switch_data(
    count: int,
    generator: torch.Generator,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs of 4 numbers, a one hot control of 2, and the input through MAP_A or MAP_B as the control says."""
    x = (torch.rand(count, 4, generator=generator) * 2 - 1) * scale
    which = torch.randint(0, 2, (count,), generator=generator)
    control = F.one_hot(which, 2).float()

    return x, control, torch.where(which.unsqueeze(1) == 0, x @ MAP_A.T, x @ MAP_B.T)


# The second input's own map in combine2's sum task, where both inputs add to the answer.
MAP_SECOND = torch.tensor([[1.0, -0.5], [0.5, 1.0]])


def sum_data(
    count: int,
    generator: torch.Generator,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs of 4 numbers and a second input of 2, and the answer MAP_A gives the first plus MAP_SECOND the second."""
    x = (torch.rand(count, 4, generator=generator) * 2 - 1) * scale
    second = (torch.rand(count, 2, generator=generator) * 2 - 1) * scale

    return x, second, x @ MAP_A.T + second @ MAP_SECOND.T


class ReferenceGated(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.value = nn.Linear(4, hidden)
        self.gate = nn.Linear(2, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(
        self,
        x: Tensor,
        control: Tensor,
    ) -> Tensor:
        return self.head(torch.sigmoid(self.gate(control)) * self.value(x))


class ReferenceAdded(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.value = nn.Linear(4, hidden)
        self.from_control = nn.Linear(2, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(
        self,
        x: Tensor,
        control: Tensor,
    ) -> Tensor:
        return self.head(torch.relu(self.value(x) + self.from_control(control)))


class AddedLinear(nn.Module):
    """combine2's first row, the control added with no hidden layer at all."""

    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Linear(4, 2)
        self.from_control = nn.Linear(2, 2)

    def forward(
        self,
        x: Tensor,
        control: Tensor,
    ) -> Tensor:
        return self.value(x) + self.from_control(control)


def reference_effective_weights(
    model: nn.Module,
    control: Tensor,
) -> Tensor:
    with torch.no_grad():
        gates = torch.sigmoid(model.gate(control))

        return model.head.weight @ (gates.unsqueeze(-1) * model.value.weight)


COMBINE2_LESSON = prose(
    """
    Adding and multiplying combine two signals in different ways. A sum
    weighs several pieces of evidence for one answer. A product lets one
    signal decide what happens to another.

    Here a control of 2 picks one of two fixed linear maps, A or B, for an
    input of 4 numbers. Three models, trained for 1,000 steps on three seeds.

    - added: a Linear of the input plus a Linear of the control, no hidden layer.
    - added, hidden: the two Linears summed inside a hidden layer of 16, then a ReLU and a head.
    - gated: gating1's gate from the control, times a Linear of the input, then a head.

    | Model | Weights | Error | Error, inputs 4 times larger |
    |---|---|---|---|
    | added | 16 | 0.30 | 4.66 |
    | added, hidden | 162 | 0.00001 to 0.00002 | 1.56 to 1.91 |
    | gated | 162 | 0.0000 | 0.0000 |

    Guessing the average scores 0.75 and 12.1. The gated model's errors
    stayed below 0.0000000000002, which is rounding.

    The sum alone has the same weights for every input, so it can't switch
    maps. With a hidden layer, the control's shift pushes units on or off
    through the ReLU. A large enough input overrides that shift, as it did
    FiLM's joined model in combine1.

    For a fixed control, the gated model is one linear map of the input. Its
    weights are the head's weights times each unit's gate times the value's
    weights. The control writes those **effective weights** anew on every pass.

    ```text
    W(control) = W_head @ diag(sigmoid(gate(control))) @ W_value
    ```

    After training, the control for A wrote weights within 0.0000003 of A,
    and the control for B within 0.0000002 of B, on every seed.

    Now flip the task. The answer is A of the input plus another fixed map of
    a second input of 2 numbers. Both are evidence for the same answer, and
    neither picks anything. The same three models, trained the same way.

    | Model | Error | Error, inputs 4 times larger |
    |---|---|---|
    | added | 0.0000 | 0.0000 |
    | added, hidden | 0.00002 to 0.0001 | 0.60 to 1.12 |
    | gated | 0.0003 to 0.0004 | 1.18 to 1.29 |

    Guessing the average scores 1.18 and 18.5. The plain sum is exact, below
    0.000000000004, since the answer is a sum. The gate can only scale the
    first input's units by numbers from 0 to 1. It fakes the sum in range and
    misses once the inputs grow.

    So multiply when one signal should control another, with a gate, FiLM or
    attention's weights. Add when signals are evidence for the same answer,
    as a residual adds a block's change to what came before.
    """,
)

COMBINE2_SOURCE = source_file(
    '''
    # combine2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    gates = torch.sigmoid(torch.tensor([0.0, 10.0]))                # tensor([0.5000, 1.0000])
    rows = torch.tensor([[2.0], [3.0]]) * torch.ones(2, 4)          # each row times its own number
    batched = torch.ones(5, 2, 3) @ torch.ones(5, 3, 4)             # shape (5, 2, 4), one product per input


    # TODO: Finish forward. Gate each of value's hidden units by sigmoid of gate
    #   on the control, then run head. x has shape (batch, 4) and control
    #   (batch, 2). The result has shape (batch, 2).
    class Gated(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.value = nn.Linear(4, hidden)
            self.gate = nn.Linear(2, hidden)
            self.head = nn.Linear(hidden, 2)

        def forward(
            self,
            x: Tensor,
            control: Tensor,
        ) -> Tensor:
            ...


    # TODO: Finish forward. Add from_control of the control to value of x,
    #   take a ReLU, then run head. The result has shape (batch, 2).
    class Added(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.value = nn.Linear(4, hidden)
            self.from_control = nn.Linear(2, hidden)
            self.head = nn.Linear(hidden, 2)

        def forward(
            self,
            x: Tensor,
            control: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the effective weights a Gated model gets from each control,
    #   of shape (batch, 2, 4): head's weight times each hidden unit's gate
    #   times value's weight. control has shape (batch, 2). Record nothing for
    #   autograd.
    def effective_weights(
        model: Gated,
        control: Tensor,
    ) -> Tensor:
        ...
    ''',
)

COMBINE2_SOLUTION = source_file(
    '''
    # combine2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    gates = torch.sigmoid(torch.tensor([0.0, 10.0]))
    rows = torch.tensor([[2.0], [3.0]]) * torch.ones(2, 4)
    batched = torch.ones(5, 2, 3) @ torch.ones(5, 3, 4)


    class Gated(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.value = nn.Linear(4, hidden)
            self.gate = nn.Linear(2, hidden)
            self.head = nn.Linear(hidden, 2)

        def forward(
            self,
            x: Tensor,
            control: Tensor,
        ) -> Tensor:
            opened = torch.sigmoid(self.gate(control))

            return self.head(opened * self.value(x))


    class Added(nn.Module):
        def __init__(self, hidden: int = 16) -> None:
            super().__init__()
            self.value = nn.Linear(4, hidden)
            self.from_control = nn.Linear(2, hidden)
            self.head = nn.Linear(hidden, 2)

        def forward(
            self,
            x: Tensor,
            control: Tensor,
        ) -> Tensor:
            summed = self.value(x) + self.from_control(control)

            return self.head(torch.relu(summed))


    def effective_weights(
        model: Gated,
        control: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            opened = torch.sigmoid(model.gate(control))
            gated_value = opened.unsqueeze(-1) * model.value.weight

            return model.head.weight @ gated_value
    ''',
)


def check_two_input_module(
    found: type,
    reference: type,
    name: str,
) -> None:
    for seed, count in ((948, 5), (949, 3)):
        model = expect_module(f"{name}()", seeded_model(found))
        x, control, _ = switch_data(count, torch.Generator().manual_seed(seed))

        with torch.no_grad():
            expected = seeded_model(reference)(x, control)
            actual = model(x, control)

        expect_tensor(
            call=f"{name}() on {count} inputs and their controls",
            actual=actual,
            expected=expected,
        )


def check_effective_weights(found: object) -> None:
    for seed, controls in ((950, torch.tensor([[1.0, 0.0], [0.0, 1.0]])), (951, small_tensor(952, 3, 2) / 3)):
        model = seeded_model(ReferenceGated)
        torch.manual_seed(seed)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter))

        seen: list[bool] = []
        hook = model.gate.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))

        try:
            actual = found(model, controls)
        finally:
            hook.remove()

        if any(seen):
            raise Mismatch("effective_weights recorded for autograd. Work inside torch.no_grad().")

        expect_tensor(
            call=f"effective_weights of a Gated model for {len(controls)} controls",
            actual=actual,
            expected=reference_effective_weights(model, controls),
        )


def check_combine2(checker: Checker) -> None:
    checker.test_method(
        class_name="Gated",
        method="forward",
        description="gates the input's hidden units by the control",
        run=lambda found: check_two_input_module(found, ReferenceGated, "Gated"),
    )
    checker.test_method(
        class_name="Added",
        method="forward",
        description="adds the control to the input's hidden units before a ReLU",
        run=lambda found: check_two_input_module(found, ReferenceAdded, "Added"),
    )
    checker.test(
        function="effective_weights",
        description="gives the linear map each control writes",
        run=check_effective_weights,
    )


COMBINE2 = Exercise(
    name="combine2",
    chapter=COMBINING,
    title="Gating against adding",
    lesson=COMBINE2_LESSON,
    source=COMBINE2_SOURCE,
    solution=COMBINE2_SOLUTION,
    hints={
        "Gated.forward": "torch.sigmoid(self.gate(control)) gives one gate per hidden unit. Multiply it by "
        "self.value(x), then run self.head.",
        "Added.forward": "Sum self.value(x) and self.from_control(control), take torch.relu, then run self.head.",
        "effective_weights": "Inside torch.no_grad(), the gates have shape (batch, hidden). .unsqueeze(-1) lets them "
        "scale the rows of model.value.weight, and model.head.weight @ that gives (batch, 2, 4).",
    },
    introduces=(),
    check=check_combine2,
)


# ---------------------------------------------------------------------------
# combine3
# ---------------------------------------------------------------------------

WIDTH = 32


def side_data(
    count: int,
    generator: torch.Generator,
    useful: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs of 8 numbers, a side input of 4, and an answer the side input explains part of, or none of."""
    x = torch.rand(count, 8, generator=generator) * 2 - 1
    side = torch.rand(count, 4, generator=generator) * 2 - 1
    base_part = torch.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2] - 0.5 * x[:, 3]
    side_part = 0.5 * torch.tanh(2 * side[:, 0] - side[:, 1]) * (1 + x[:, 4])

    return x, side, base_part + (side_part if useful else 0 * side_part)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.f = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))

    def forward(self, h: Tensor) -> Tensor:
        return h + self.f(self.norm(h))


class Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(8, WIDTH)
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.embed(x)

        for block in self.blocks:
            h = block(h)

        return self.head(self.norm(h)).squeeze(1)


def make_branch(hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, WIDTH))


def reference_zero_last_(branch: nn.Sequential) -> None:
    with torch.no_grad():
        branch[-1].weight.zero_()
        branch[-1].bias.zero_()


class ReferenceWithBranch(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        branch: nn.Module,
        at: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.branch = branch
        self.at = at

    def forward(
        self,
        x: Tensor,
        side: Tensor,
    ) -> Tensor:
        h = self.base.embed(x)

        for index, block in enumerate(self.base.blocks):
            if index == self.at:
                h = h + self.branch(side)

            h = block(h)

        return self.base.head(self.base.norm(h)).squeeze(1)


def reference_start_gap(
    model: nn.Module,
    base: nn.Module,
    x: Tensor,
    side: Tensor,
) -> float:
    with torch.no_grad():
        return (model(x, side) - base(x)).abs().max().item()


COMBINE3_LESSON = prose(
    """
    A model that already works sometimes needs a new piece, like a branch
    that reads a new input. ControlNet adds such a branch to a trained image
    model, and lora1 added a low rank one.

    The new branch can start random, or start as do nothing. With its last
    Linear at zero, the model at step 0 gives exactly what it gave before.
    residual3 used this start for a deep stack with no norm. Here it protects
    a trained model.

    Zero only the last layer. With every layer at zero, the first layer's
    gradient is multiplied by the last layer's zeros, and the last weight's by
    the first's. Only the last bias moves, so the branch never reads its input.

    ControlNet's authors gave the zero start's reason: no harmful noise
    reaches the trained model while the new piece starts to learn. So measure
    the error along the way, checked every 50 steps, not only at the end.

    Here a residual model of 4 blocks was trained first. Then a branch read
    4 new numbers and added its output to the stream before the second block.
    The branch, starting from nothing, trained at 0.01, three seeds each.

    - signal: the new numbers explain part of the answer, with 4,000 examples.
    - no signal: the new numbers explain nothing, with only 64 examples.

    With signal the branch is wide, 512 hidden units, as ControlNet's is a
    full copy of its model. The trained model trained too, at 0.001. Without
    signal the branch had 64 units, and the trained model stayed frozen, as
    ControlNet locks its own.

    | Branch start | Signal, step 0 | Signal, step 1,500 | No signal, step 1,500 |
    |---|---|---|---|
    | last layer zero | 0.252 | 0.0033 to 0.0037 | 0.00073 to 0.00075 |
    | PyTorch's default | 0.23 to 0.28 | 0.0035 to 0.0048 | 0.0012 to 0.0017 |
    | Kaiming | 0.33 to 0.37 | 0.0042 to 0.0049 | 0.063 to 0.081 |

    The trained model alone scored 0.252 with signal and 0.0005 without.

    The zero start began exactly at the trained model and never rose above it
    at any check. Every random start began somewhere else, the loud Kaiming
    one up to 0.37. All of them learned the new input, ending 52 to 76 times
    below the model alone, and the zero start ended lowest on every seed.

    Without signal, every branch learned some noise from its 64 examples. The
    zero start kept that smallest, ending about 1.5 times the model alone, and
    no check passed 0.0011. PyTorch's default began at 0.017 to 0.030 and
    ended 2.4 to 3.3 times the model alone, and Kaiming 124 to 160 times. With
    every layer at zero, the signal run ended at 0.29 to 0.30, worse than the
    model alone. That branch could never read its input.

    So use the zero start when the model already works and you don't know the
    piece will help. In ControlNet's own test, starting those layers at random
    pulled its results down to about those of its lite version.
    """,
)

COMBINE3_SOURCE = source_file(
    '''
    # combine3.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    WIDTH = 32
    last = nn.Linear(3, 2)
    last.weight.shape    # torch.Size([2, 3]), its .bias has shape (2,)


    # A trained model's block, a pre norm residual MLP.
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(WIDTH)
            self.f = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))

        def forward(self, h: Tensor) -> Tensor:
            return h + self.f(self.norm(h))


    # The trained model the branch joins.
    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(8, WIDTH)
            self.blocks = nn.ModuleList([Block() for _ in range(4)])
            self.norm = nn.LayerNorm(WIDTH)
            self.head = nn.Linear(WIDTH, 1)

        def forward(self, x: Tensor) -> Tensor:
            h = self.embed(x)

            for block in self.blocks:
                h = block(h)

            return self.head(self.norm(h)).squeeze(1)


    # TODO: Set the weight and the bias of branch's last layer to zero, in
    #   place, without recording for autograd. Leave every other layer as it is.
    def zero_last_(branch: nn.Sequential) -> None:
        ...


    # TODO: Finish forward. Run the base model on x, of shape (batch, 8), as
    #   Base.forward does, but add branch of side, of shape (batch, 4), to the
    #   stream just before self.base.blocks[at] runs, so at=1 adds it before
    #   the second block. Return one answer per input.
    class WithBranch(nn.Module):
        def __init__(
            self,
            base: Base,
            branch: nn.Module,
            at: int,
        ) -> None:
            super().__init__()
            self.base = base
            self.branch = branch
            self.at = at

        def forward(
            self,
            x: Tensor,
            side: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the largest change, as a plain number, between model's
    #   answers for x and side and base's answers for x alone. Record nothing
    #   for autograd.
    def start_gap(
        model: nn.Module,
        base: nn.Module,
        x: Tensor,
        side: Tensor,
    ) -> float:
        ...
    ''',
)

COMBINE3_SOLUTION = source_file(
    '''
    # combine3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    WIDTH = 32
    last = nn.Linear(3, 2)
    last.weight.shape


    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(WIDTH)
            self.f = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))

        def forward(self, h: Tensor) -> Tensor:
            return h + self.f(self.norm(h))


    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(8, WIDTH)
            self.blocks = nn.ModuleList([Block() for _ in range(4)])
            self.norm = nn.LayerNorm(WIDTH)
            self.head = nn.Linear(WIDTH, 1)

        def forward(self, x: Tensor) -> Tensor:
            h = self.embed(x)

            for block in self.blocks:
                h = block(h)

            return self.head(self.norm(h)).squeeze(1)


    def zero_last_(branch: nn.Sequential) -> None:
        with torch.no_grad():
            branch[-1].weight.zero_()
            branch[-1].bias.zero_()


    class WithBranch(nn.Module):
        def __init__(
            self,
            base: Base,
            branch: nn.Module,
            at: int,
        ) -> None:
            super().__init__()
            self.base = base
            self.branch = branch
            self.at = at

        def forward(
            self,
            x: Tensor,
            side: Tensor,
        ) -> Tensor:
            h = self.base.embed(x)

            for index, block in enumerate(self.base.blocks):
                if index == self.at:
                    h = h + self.branch(side)

                h = block(h)

            return self.base.head(self.base.norm(h)).squeeze(1)


    def start_gap(
        model: nn.Module,
        base: nn.Module,
        x: Tensor,
        side: Tensor,
    ) -> float:
        with torch.no_grad():
            return (model(x, side) - base(x)).abs().max().item()
    ''',
)


def check_zero_last(found: object) -> None:
    for seed in (953, 954):
        torch.manual_seed(seed)
        branch = make_branch()
        before = copy.deepcopy(branch)
        result = found(branch)

        if result is not None:
            raise Mismatch(f"zero_last_ gave {value_text(result)}. It should change branch in place and give nothing back.")

        if branch[-1].weight.abs().max() != 0 or branch[-1].bias.abs().max() != 0:
            raise Mismatch("After zero_last_, the last layer's weight and bias should both be all zeros.")

        if not torch.equal(branch[0].weight, before[0].weight) or not torch.equal(branch[0].bias, before[0].bias):
            raise Mismatch(
                "zero_last_ changed the first layer too. Zero only the last one, or no gradient can ever reach "
                "either layer.",
            )

        if not branch[-1].weight.requires_grad or not branch[-1].bias.requires_grad:
            raise Mismatch("After zero_last_ the last layer no longer trains. Change its values, and keep requires_grad.")


def seeded_base() -> nn.Module:
    torch.manual_seed(0)

    return Base()


def check_with_branch(found: type) -> None:
    x, side, _ = side_data(5, torch.Generator().manual_seed(955), True)

    for at in (0, 1, 3):
        torch.manual_seed(956 + at)
        branch = make_branch()
        model = expect_module(f"WithBranch(base, branch, {at})", found(seeded_base(), branch, at))

        with torch.no_grad():
            expected = ReferenceWithBranch(seeded_base(), branch, at)(x, side)
            actual = model(x, side)

        expect_tensor(
            call=f"WithBranch(base, branch, {at}) on 5 inputs with their side inputs",
            actual=actual,
            expected=expected,
        )


class LoweredBase(nn.Module):
    """A stand in whose answers all sit below the base's, so the largest change is a negative one."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        x: Tensor,
        side: Tensor,
    ) -> Tensor:
        return self.base(x) - 1 - side.abs().sum(dim=1)


def check_start_gap(found: object) -> None:
    x, side, _ = side_data(6, torch.Generator().manual_seed(958), True)

    for name, start in (("a random branch", "random"), ("a branch with its last layer zero", "zero"), ("answers all lowered", "lowered")):
        torch.manual_seed(959)
        branch = make_branch()

        if start == "zero":
            reference_zero_last_(branch)

        base = seeded_base()
        model = LoweredBase(seeded_base()) if start == "lowered" else ReferenceWithBranch(seeded_base(), branch, 1)
        seen: list[bool] = []
        hook = recording_hook(model, seen)

        try:
            actual = found(model, base, x, side)
        finally:
            hook.remove()

        if any(seen):
            raise Mismatch("start_gap recorded for autograd. Run the models inside torch.no_grad().")

        expect_close(
            call=f"start_gap of a model with {name}, on 6 inputs",
            actual=actual,
            expected=reference_start_gap(model, base, x, side),
            tolerance=1e-6,
        )


def check_combine3(checker: Checker) -> None:
    checker.test(
        function="zero_last_",
        description="zeroes the last layer alone, in place",
        run=check_zero_last,
    )
    checker.test_method(
        class_name="WithBranch",
        method="forward",
        description="adds the branch to the stream before block at",
        run=check_with_branch,
    )
    checker.test(
        function="start_gap",
        description="measures how far the new piece moves the answers",
        run=check_start_gap,
    )


COMBINE3 = Exercise(
    name="combine3",
    chapter=COMBINING,
    title="A new branch that starts as do nothing",
    lesson=COMBINE3_LESSON,
    source=COMBINE3_SOURCE,
    solution=COMBINE3_SOLUTION,
    hints={
        "zero_last_": "branch[-1] is the last layer. Inside torch.no_grad(), call .zero_() on its weight and on its "
        "bias.",
        "WithBranch.forward": "Start from self.base.embed(x). Loop with enumerate over self.base.blocks, add "
        "self.branch(side) when the index equals self.at, then run the block. End with the base's norm and head.",
        "start_gap": "Inside torch.no_grad(), take model(x, side) minus base(x), then .abs().max().item().",
    },
    introduces=(),
    check=check_combine3,
)


# ---------------------------------------------------------------------------
# combine4
# ---------------------------------------------------------------------------

STEPS_ALONG = 8


def neighbour_data(
    count: int,
    generator: torch.Generator,
    length: int = STEPS_ALONG,
) -> tuple[Tensor, Tensor]:
    """8 positions of 2 channels, or length, and at each position how far its channel 0 is from the next one's
    channel 1."""
    x = torch.rand(count, length, 2, generator=generator) * 2 - 1

    return x, (x[:, :, 0] - x.roll(-1, dims=1)[:, :, 1]).abs()


class ReferenceTokenMix(nn.Module):
    def __init__(
        self,
        width: int,
        positions: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(positions, hidden), nn.GELU(), nn.Linear(hidden, positions))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x).transpose(1, 2)).transpose(1, 2)


class ReferenceChannelMix(nn.Module):
    def __init__(
        self,
        width: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.norm(x))


class AttentionMix(nn.Module):
    """combine4's attention row: one head of attention across the positions in place of the token MLP, each place
    given a learned position as it enters, with the same norm and skip."""

    def __init__(
        self,
        width: int,
        positions: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.place = nn.Embedding(positions, width)
        self.qkv = nn.Linear(width, 3 * hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x) + self.place(torch.arange(x.shape[1]))
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        return x + self.proj(F.scaled_dot_product_attention(q, k, v))


class ConvMix(nn.Module):
    """combine4's convolution row: each channel mixes its value with its two neighbours', wrapping around, with the
    same norm and skip. It has no weight tied to a place, so it runs at any length."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=1, padding_mode="circular", groups=width)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)


class Mixer(nn.Module):
    """An embedding of 2 channels into width, a block per kind in order, and one answer per position."""

    def __init__(
        self,
        kinds: list[str],
        width: int = 16,
        token_mix: Callable[[int, int, int], nn.Module] = ReferenceTokenMix,
        channel_mix: Callable[[int, int], nn.Module] = ReferenceChannelMix,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(2, width)
        self.blocks = nn.Sequential(
            *(token_mix(width, STEPS_ALONG, 32) if kind == "across" else channel_mix(width, 32) for kind in kinds),
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.norm(self.blocks(self.embed(x)))).squeeze(-1)


def reference_reach(
    model: nn.Module,
    x: Tensor,
    position: int,
) -> Tensor:
    x = x.clone().requires_grad_()
    gradient = torch.autograd.grad(model(x)[:, position].sum(), x)[0]

    return gradient.abs().sum(dim=(0, 2)) > 0


COMBINE4_LESSON = prose(
    """
    Inputs with positions, like a sequence or an image's patches, can be
    mixed in two ways.

    - across positions: each channel mixes its values from every position, one channel at a time.
    - within a position: the channels mix, one position at a time.

    A transformer block does both, with attention across and its MLP within.
    Tolstikhin and colleagues' MLP-Mixer does both with MLPs alone. A token
    mixing MLP acts on each channel's column of positions, shared by every
    channel. A channel mixing MLP acts on each position's row. The two are
    interleaved, each with a layer norm and a skip connection. Its largest
    model, pre-trained on 300 million images, reached 87.94% on ImageNet.

    The task has 8 positions of 2 channels. The answer at position p is
    `|x[p, 0] - x[p + 1, 1]|`, wrapping around at the end. It needs the next
    position's channel 1 beside this one's channel 0, then a curved mix.

    Each model embeds the 2 channels into 16, runs 4 blocks of the kinds
    below, and reads one answer per position. Each trained for 1,500 steps,
    three seeds each.

    | Blocks | Weights | Error |
    |---|---|---|
    | within only | 4,513 | 0.200 to 0.205 |
    | across only | 2,433 | 0.053 to 0.123 |
    | across, within, across, within | 3,473 | 0.0002 to 0.0005 |
    | within, across, within, across | 3,473 | 0.0003 to 0.0005 |

    Guessing the average scores 0.221. Within only never sees position p + 1,
    so it can't do much better than a guess. Across only mixes along the
    positions, one channel at a time. Its embedding and head mix channels only
    in straight lines, and its layer norms only center and rescale them. So it
    got part of the way. Alternating gets both, in either order.

    A position's **reach** is the set of input positions that can change its
    output. A gradient finds it, since an input with a nonzero gradient
    changes the output when it moves.

    `torch.autograd.grad(out, x)` gives the gradient of `out` with respect to
    `x`, without touching any `.grad`.
    """,
)

COMBINE4_SOURCE = source_file(
    '''
    # combine4.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    table = torch.arange(6.0).reshape(1, 3, 2)    # 1 input, 3 positions, 2 channels
    columns = table.transpose(1, 2)                 # shape (1, 2, 3), one row of positions per channel
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    slopes = torch.autograd.grad((x[0] * 3).sum(), x)[0]    # tensor([3., 0.]), x[1] can't move the output


    # TODO: Finish forward. x has shape (batch, positions, width). Normalize
    #   it, run mlp on each channel's column of positions, and add the result
    #   back to x. The result has x's shape.
    class TokenMix(nn.Module):
        def __init__(
            self,
            width: int,
            positions: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(positions, hidden), nn.GELU(), nn.Linear(hidden, positions))

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. Normalize x, run mlp on each position's channels,
    #   and add the result back to x.
    class ChannelMix(nn.Module):
        def __init__(
            self,
            width: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

        def forward(self, x: Tensor) -> Tensor:
            ...


    # The lesson's models: an embedding, a block per kind in order, and one
    #   answer per position.
    class Mixer(nn.Module):
        def __init__(
            self,
            kinds: list[str],
            width: int = 16,
        ) -> None:
            super().__init__()
            self.embed = nn.Linear(2, width)
            self.blocks = nn.Sequential(
                *(TokenMix(width, 8, 32) if kind == "across" else ChannelMix(width, 32) for kind in kinds),
            )
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 1)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(self.norm(self.blocks(self.embed(x)))).squeeze(-1)


    # TODO: Return the reach of position in model, a tensor of True or False,
    #   one per input position: whether that position's numbers have a nonzero
    #   gradient for the sum of model(x)[:, position]. x has shape (batch,
    #   positions, channels). Leave x itself as it is.
    def reach(
        model: nn.Module,
        x: Tensor,
        position: int,
    ) -> Tensor:
        ...
    ''',
)

COMBINE4_SOLUTION = source_file(
    '''
    # combine4.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    table = torch.arange(6.0).reshape(1, 3, 2)
    columns = table.transpose(1, 2)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    slopes = torch.autograd.grad((x[0] * 3).sum(), x)[0]


    class TokenMix(nn.Module):
        def __init__(
            self,
            width: int,
            positions: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(positions, hidden), nn.GELU(), nn.Linear(hidden, positions))

        def forward(self, x: Tensor) -> Tensor:
            columns = self.norm(x).transpose(1, 2)

            return x + self.mlp(columns).transpose(1, 2)


    class ChannelMix(nn.Module):
        def __init__(
            self,
            width: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

        def forward(self, x: Tensor) -> Tensor:
            return x + self.mlp(self.norm(x))


    class Mixer(nn.Module):
        def __init__(
            self,
            kinds: list[str],
            width: int = 16,
        ) -> None:
            super().__init__()
            self.embed = nn.Linear(2, width)
            self.blocks = nn.Sequential(
                *(TokenMix(width, 8, 32) if kind == "across" else ChannelMix(width, 32) for kind in kinds),
            )
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 1)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(self.norm(self.blocks(self.embed(x)))).squeeze(-1)


    def reach(
        model: nn.Module,
        x: Tensor,
        position: int,
    ) -> Tensor:
        watched = x.clone().requires_grad_()
        out = model(watched)[:, position].sum()
        gradient = torch.autograd.grad(out, watched)[0]

        return gradient.abs().sum(dim=(0, 2)) > 0
    ''',
)


def check_mix_block(
    found: type,
    reference: type,
    name: str,
    arguments: tuple[int, ...],
) -> None:
    for seed, batch, positions in ((960, 2, 5), (961, 3, 3)):
        made_arguments = arguments if name == "ChannelMix" else (arguments[0], positions, arguments[1])
        block = expect_module(f"{name}{made_arguments}", seeded_model(found, *made_arguments))
        x = small_tensor(seed, batch, positions, arguments[0]) / 3

        with torch.no_grad():
            expected = seeded_model(reference, *made_arguments)(x)
            actual = block(x)

        expect_tensor(
            call=f"{name}{made_arguments} on x of shape {tuple(x.shape)}",
            actual=actual,
            expected=expected,
        )


def check_reach(found: object) -> None:
    x, _ = neighbour_data(3, torch.Generator().manual_seed(962))

    for kinds, position in ((["within"], 2), (["across"], 5), (["within", "across"], 0)):
        model = seeded_model(Mixer, kinds)
        given = x.clone()
        actual = found(model, x, position)

        if not torch.equal(x, given) or x.requires_grad:
            raise Mismatch("reach changed x itself. Work on x.clone().requires_grad_() and leave x as it is.")

        if not isinstance(actual, Tensor) or actual.dtype != torch.bool:
            raise Mismatch(f"reach gave {value_text(actual)}. It should give a tensor of True or False, one per position.")

        expect_tensor(
            call=f"reach of position {position} in a Mixer of {', '.join(kinds)} blocks",
            actual=actual,
            expected=reference_reach(model, x, position),
        )


def check_combine4(checker: Checker) -> None:
    checker.test_method(
        class_name="TokenMix",
        method="forward",
        description="mixes each channel across the positions, with a skip",
        run=lambda found: check_mix_block(found, ReferenceTokenMix, "TokenMix", (4, 6)),
    )
    checker.test_method(
        class_name="ChannelMix",
        method="forward",
        description="mixes the channels within each position, with a skip",
        run=lambda found: check_mix_block(found, ReferenceChannelMix, "ChannelMix", (4, 6)),
    )
    checker.test(
        function="reach",
        description="finds the input positions that can change an output",
        run=check_reach,
    )


COMBINE4 = Exercise(
    name="combine4",
    chapter=COMBINING,
    title="Mixing across and within positions",
    lesson=COMBINE4_LESSON,
    source=COMBINE4_SOURCE,
    solution=COMBINE4_SOLUTION,
    hints={
        "TokenMix.forward": "Normalize x, then .transpose(1, 2) so each channel's positions sit in the last "
        "dimension, where a Linear acts. Run self.mlp, turn it back with the same transpose, and add x.",
        "ChannelMix.forward": "A Linear already acts on the last dimension, the channels, so run self.mlp on "
        "self.norm(x) and add x.",
        "reach": "Make a copy of x that records gradients with .clone().requires_grad_(). Sum the output at position "
        "over the batch, take torch.autograd.grad of it, and see which positions have any nonzero entry.",
    },
    introduces=(),
    check=check_combine4,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# combine5
# ---------------------------------------------------------------------------

BUMP = torch.tensor([0.5, 1.0, 0.5])
IDS = 16

# Each signal sits on its own level, so only a detector of a bump's shape can count, not a sum of the signal.
LEVEL = 0.3


def bump_signals(
    count: int,
    length: int,
    most: int,
    generator: torch.Generator,
    twin: bool = False,
) -> tuple[Tensor, Tensor]:
    """Noisy signals on a level of their own, with up to most bumps of 0.5, 1, 0.5, at most one in each 4 places,
    and how many each holds. The twin task's bumps point down."""
    slots = length // 4
    counts = torch.randint(0, most + 1, (count,), generator=generator)
    chosen = torch.rand(count, slots, generator=generator).argsort(dim=1) < counts.unsqueeze(1)
    shifts = torch.randint(0, 2, (count, slots), generator=generator)
    centers = torch.zeros(count, length).scatter(1, torch.arange(slots) * 4 + 1 + shifts, chosen.float())
    bumps = F.conv1d(centers.unsqueeze(1), BUMP.view(1, 1, 3), padding=1).squeeze(1)
    levels = (torch.rand(count, 1, generator=generator) * 2 - 1) * LEVEL
    noise = torch.randn(count, length, generator=generator) * 0.1

    return levels + (-bumps if twin else bumps) + noise, counts.float()


def baskets(
    count: int,
    items: int,
    generator: torch.Generator,
    twin: bool = False,
) -> tuple[Tensor, Tensor]:
    """Baskets of items, each a price from 0.5 to 2 and a count of 1 to 3, and each basket's total. The twin task
    adds up each item's count over its price, so no one fixed formula fits both."""
    price = 0.5 + 1.5 * torch.rand(count, items, generator=generator)
    how_many = torch.randint(1, 4, (count, items), generator=generator).float()
    totals = (how_many / price).sum(dim=1) if twin else (price * how_many).sum(dim=1)

    return torch.stack([price, how_many], dim=-1), totals


def lookups(
    count: int,
    pairs: int,
    generator: torch.Generator,
    twin: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pairs of a key and a value, both ids from 0 to 15 with no key twice, a query that is one of the keys, and its
    value. The twin task's query is one more than the key it asks for, wrapping at 16."""
    keys = torch.rand(count, IDS, generator=generator).argsort(dim=1)[:, :pairs]
    values = torch.randint(0, IDS, (count, pairs), generator=generator)
    which = torch.randint(0, pairs, (count,), generator=generator)
    rows = torch.arange(count)
    query = (keys[rows, which] + 1) % IDS if twin else keys[rows, which]

    return keys, values, query, values[rows, which]


class ReferenceBumpCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detect = nn.Conv1d(1, 8, 3, padding=1)
        self.score = nn.Conv1d(8, 1, 1)

    def forward(self, signals: Tensor) -> Tensor:
        return self.score(torch.relu(self.detect(signals.unsqueeze(1)))).squeeze(1).sum(dim=1)


class ReferenceBasketTotal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.item = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, items: Tensor) -> Tensor:
        return self.item(items).squeeze(-1).sum(dim=1)


class ReferenceKeyLookup(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.key_table = nn.Embedding(IDS, width)
        self.query_table = nn.Embedding(IDS, width)
        self.value_table = nn.Embedding(IDS, width)
        self.head = nn.Linear(width, IDS)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        q = self.query_table(query).unsqueeze(1)
        k = self.key_table(keys)
        weights = torch.softmax((q * k).sum(dim=-1) / math.sqrt(k.shape[-1]), dim=-1)

        return self.head((weights.unsqueeze(-1) * self.value_table(values)).sum(dim=1))


class MeanBumpCounter(ReferenceBumpCounter):
    def forward(self, signals: Tensor) -> Tensor:
        return self.score(torch.relu(self.detect(signals.unsqueeze(1)))).squeeze(1).mean(dim=1)


class MeanBasketTotal(ReferenceBasketTotal):
    def forward(self, items: Tensor) -> Tensor:
        return self.item(items).squeeze(-1).mean(dim=1)


class GRULookup(nn.Module):
    """attention6's way, every pair read in order into one hidden state."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.table = nn.Embedding(IDS, 16)
        self.gru = nn.GRU(32, width, batch_first=True)
        self.from_query = nn.Linear(16, width)
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, IDS))

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        _, h = self.gru(torch.cat([self.table(keys), self.table(values)], dim=-1))

        return self.head(torch.cat([h[0], self.from_query(self.table(query))], dim=-1))


def train_bumps(
    model: nn.Module,
    seed: int,
    twin: bool = False,
    steps: int = 300,
) -> nn.Module:
    """Signals of 16, 32 and 48 in turn, so no answer can lean on a length."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(seed)

    for step in range(steps):
        length = (16, 32, 48)[step % 3]
        signals, counts = bump_signals(64, length, length // 8 - 1 if length > 16 else 2, generator, twin)
        loss = F.mse_loss(model(signals), counts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_baskets(
    model: nn.Module,
    seed: int,
    twin: bool = False,
    steps: int = 500,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        items, totals = baskets(64, 5, generator, twin)
        loss = F.mse_loss(model(items), totals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_lookups(
    model: nn.Module,
    seed: int,
    twin: bool = False,
    steps: int = 400,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        keys, values, query, answer = lookups(64, 4, generator, twin)
        loss = F.cross_entropy(model(keys, values, query), answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def bump_miss(
    model: nn.Module,
    twin: bool = False,
) -> float:
    """The average miss in bumps on 1,000 signals of 96, with up to 9 bumps each."""
    signals, counts = bump_signals(1000, 96, 9, torch.Generator().manual_seed(101), twin)

    with torch.no_grad():
        return (model(signals) - counts).abs().mean().item()


def basket_miss(
    model: nn.Module,
    twin: bool = False,
) -> float:
    """The average miss on 1,000 baskets of 12 items, as a share of the average total."""
    items, totals = baskets(1000, 12, torch.Generator().manual_seed(101), twin)

    with torch.no_grad():
        return ((model(items) - totals).abs().mean() / totals.mean()).item()


def lookup_share(
    model: nn.Module,
    twin: bool = False,
) -> float:
    """The share of 1,000 lookups among 12 pairs answered right."""
    keys, values, query, answer = lookups(1000, 12, torch.Generator().manual_seed(112), twin)

    with torch.no_grad():
        return (model(keys, values, query).argmax(dim=-1) == answer).float().mean().item()


# The change each problem's answer ignores. A signal moved along by whole places of 4 keeps every bump whole, and a
# basket's items or a lookup's pairs can come in any order.
PROMISES = {
    "bumps": "shift",
    "baskets": "shuffle",
    "lookups": "shuffle",
}


def bump_shift_gap(model: nn.Module) -> float:
    """How far the counts move, on average, when the signals of 96 roll along by 4 places."""
    signals, _ = bump_signals(1000, 96, 9, torch.Generator().manual_seed(101))

    with torch.no_grad():
        return (model(torch.roll(signals, 4, dims=1)) - model(signals)).abs().mean().item()


def basket_shuffle_gap(model: nn.Module) -> float:
    """How far the totals move when each basket's 12 items come in a new order, as a share of the average total."""
    items, totals = baskets(1000, 12, torch.Generator().manual_seed(101))
    order = torch.randperm(12, generator=torch.Generator().manual_seed(5))

    with torch.no_grad():
        return ((model(items[:, order]) - model(items)).abs().mean() / totals.mean()).item()


def lookup_shuffle_gap(model: nn.Module) -> float:
    """The share of answers that change when the 12 pairs come in a new order, each key kept with its value."""
    keys, values, query, _ = lookups(1000, 12, torch.Generator().manual_seed(112))
    order = torch.randperm(12, generator=torch.Generator().manual_seed(5))

    with torch.no_grad():
        before = model(keys, values, query).argmax(dim=-1)
        after = model(keys[:, order], values[:, order], query).argmax(dim=-1)

        return (before != after).float().mean().item()


COMBINE5_LESSON = prose(
    """
    Each structure the course built assumes something about the problem.
    Pick it by what the problem gives you.

    | Structure | What it assumes | Shown in |
    |---|---|---|
    | Linear or MLP | nothing, a weight for every input | mlp1 |
    | Convolution | a local pattern means the same anywhere | conv4 |
    | Pooling by sum, mean or max | the order of the items means nothing | sets2 |
    | Attention | items are found by content, at any distance | attention6 |
    | RNN, LSTM or state space | one state of fixed size, carried in order | lstm1, ssm1 |
    | Gate or FiLM | one signal controls another | gating1, combine1, combine2 |
    | Residual and norm | the model is deep | residual2, residual3 |
    | Equivariant layer | a symmetry, like a turn, a move or a mirror | equivariance2 |
    | Router to experts | each input needs a few of many skills | moe1 |
    | Message passing | the answer follows a graph's edges | message1 |
    | None of these | derive one from the exact algorithm | derive1, paths3 |

    The right structure matters most where the test differs from training.
    conv4's MLP missed a moved pattern, and attention6's GRU lost pairs as
    their number grew.

    Here are three problems you haven't seen. Build a model for each. The
    check trains it on small inputs, then tests it on bigger ones it never saw.

    - bumps: count the bumps in a noisy signal on a level of its own. Training signals are 16 to 48 long, and test signals 96.
    - baskets: each item has a price and a count, and the answer is the basket's total. Training baskets hold 5 items, and test baskets 12.
    - lookups: pairs of a key and a value, then a query, and the answer is the query's value. Training uses 4 pairs, and testing 12.

    Ask of each problem what stays true as the input grows. Does the answer
    grow with it? Does the order matter? Does one item need to find another?

    Each structure also makes a promise, a change its output ignores. Name in
    `PROMISES` the change each problem's true answer ignores. The check makes
    that change to the test inputs, and your trained model must barely move.
    A GRU reading the lookup pairs in order changed over half its answers when
    the pairs were shuffled.

    A model made of the right pieces needs no hand written answer. The check
    also trains it on a twin of each problem with another rule. The bumps point
    down, each item adds its count over its price, and the query is one more
    than its key. Learned pieces fit both, and a formula written by hand fits
    one.
    """,
)

COMBINE5_SOURCE = source_file(
    '''
    # combine5.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    signals = torch.zeros(2, 96)                     # bumps: 2 signals of 96, answers of shape (2,)
    items = torch.zeros(2, 12, 2)                     # baskets: 2 baskets of 12 items, each a price and a count
    keys = torch.zeros(2, 12, dtype=torch.long)       # lookups: ids from 0 to 15, values the same shape
    query = torch.zeros(2, dtype=torch.long)          # one query per row, the answer is one of 16 ids
    moved = torch.roll(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), 4, dims=1)    # tensor([[2., 3., 4., 5., 1.]])


    # TODO: For each problem, name the change its true answer ignores, "shift"
    #   or "shuffle". A shift moves a signal along by 4 places, wrapping
    #   around, as moved shows. A shuffle puts a basket's items, or a lookup's
    #   pairs, in a new order.
    PROMISES = {
        "bumps": ...,
        "baskets": ...,
        "lookups": ...,
    }


    # TODO: Build a model that counts the bumps in signals of shape (batch,
    #   length), for any length. forward gives one count per signal, of shape
    #   (batch,). The check trains it on lengths 16 to 48 and tests it on 96.
    class BumpCounter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ...

        def forward(self, signals: Tensor) -> Tensor:
            ...


    # TODO: Build a model that gives each basket's total, for items of shape
    #   (batch, items, 2), each a price and a count, for any number of items.
    #   forward gives one total per basket, of shape (batch,). The check
    #   trains it on baskets of 5 and tests it on baskets of 12.
    class BasketTotal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ...

        def forward(self, items: Tensor) -> Tensor:
            ...


    # TODO: Build a model that answers a lookup. keys and values have shape
    #   (batch, pairs) and query (batch,), all ids from 0 to 15. forward gives
    #   16 scores per row, of shape (batch, 16), the highest for the query's
    #   value. The check trains it on 4 pairs and tests it on 12.
    class KeyLookup(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ...

        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            ...
    ''',
)

COMBINE5_SOLUTION = source_file(
    '''
    # combine5.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    signals = torch.zeros(2, 96)
    items = torch.zeros(2, 12, 2)
    keys = torch.zeros(2, 12, dtype=torch.long)
    query = torch.zeros(2, dtype=torch.long)
    moved = torch.roll(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), 4, dims=1)


    PROMISES = {
        "bumps": "shift",
        "baskets": "shuffle",
        "lookups": "shuffle",
    }


    class BumpCounter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detect = nn.Conv1d(1, 8, 3, padding=1)
            self.score = nn.Conv1d(8, 1, 1)

        def forward(self, signals: Tensor) -> Tensor:
            found = torch.relu(self.detect(signals.unsqueeze(1)))

            return self.score(found).squeeze(1).sum(dim=1)


    class BasketTotal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, items: Tensor) -> Tensor:
            return self.item(items).squeeze(-1).sum(dim=1)


    class KeyLookup(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.key_table = nn.Embedding(16, 16)
            self.query_table = nn.Embedding(16, 16)
            self.value_table = nn.Embedding(16, 16)
            self.head = nn.Linear(16, 16)

        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            q = self.query_table(query).unsqueeze(1)
            k = self.key_table(keys)
            weights = torch.softmax((q * k).sum(dim=-1) / math.sqrt(k.shape[-1]), dim=-1)

            return self.head((weights.unsqueeze(-1) * self.value_table(values)).sum(dim=1))
    ''',
)

# The bars sit far from both sides, on the task and on its twin. Over seeds 0 to 2 see measure_problems for the
# reference against mean pooling, a plain sum, a convolution with no ReLU, a GRU and the hand written rules.
BUMP_BAR = 1.0
BASKET_BAR = 0.05
LOOKUP_BAR = 0.9

# How far a trained model's answers may move under the change its problem ignores.
BUMP_SHIFT_BAR = 0.25
BASKET_SHUFFLE_BAR = 0.01
LOOKUP_SHUFFLE_BAR = 0.02


def trained_or_crash(
    found: type,
    name: str,
    train: Callable[[nn.Module, int, bool], nn.Module],
    twin: bool = False,
) -> tuple[nn.Module, nn.Module]:
    untrained = expect_module(f"{name}()", seeded_model(found))

    if not any(parameter.requires_grad for parameter in untrained.parameters()):
        raise Mismatch(f"{name}() has no weights to train. Build it from layers the check can train, not a fixed rule.")

    try:
        trained = train(seeded_model(found), 7, twin)
    except RuntimeError as error:
        raise Mismatch(f"Training {name} on the small inputs crashed: {error}") from None

    return untrained, trained


def twin_or_rule(
    found: type,
    name: str,
    train: Callable[[nn.Module, int, bool], nn.Module],
    measure: Callable[[nn.Module, bool], float],
    passes: Callable[[float], bool],
    twin_task: str,
    cause: str = "a fixed rule is doing the work. Build it from layers that learn the rule.",
) -> None:
    """The same model trained on the problem's twin, where the answer follows another rule. Learned pieces fit
    both, and a hand written formula with a learned scale fits only one."""
    _, trained = trained_or_crash(found, name, train, twin=True)
    score = tested_or_crash(lambda model: measure(model, True), trained, name, "the twin task")

    if not passes(score):
        raise Mismatch(f"{name} learns the task but not its twin, where {twin_task}. Pieces that learn fit both, so {cause}")


def kept_promise(
    gap: float,
    bar: float,
    name: str,
    change: str,
    text: str,
) -> None:
    if gap > bar:
        raise Mismatch(
            f"Trained, {name}'s answers move by {text.format(gap)} when {change}, which leaves the true answer alone. "
            "Its structure should promise that, so pick layers that can't see the change.",
        )


def tested_or_crash(
    measure: Callable[[nn.Module], float],
    model: nn.Module,
    name: str,
    bigger: str,
) -> float:
    try:
        return measure(model)
    except RuntimeError as error:
        raise Mismatch(
            f"{name} crashed on {bigger}, bigger than training had: {error}. A model for this problem has to take "
            "any size, so its weights can't be tied to one length.",
        ) from None


def check_bump_counter(found: type) -> None:
    untrained, trained = trained_or_crash(found, "BumpCounter", train_bumps)
    miss = tested_or_crash(bump_miss, trained, "BumpCounter", "signals of 96")
    untrained_miss = tested_or_crash(bump_miss, untrained, "BumpCounter", "signals of 96")

    if untrained_miss <= BUMP_BAR:
        raise Mismatch(f"Untrained, BumpCounter already misses only {untrained_miss:.2f} bumps. It should learn the count, not compute it by a fixed rule.")

    if miss > BUMP_BAR:
        raise Mismatch(
            f"Trained on signals of 16 to 48, BumpCounter misses {miss:.2f} bumps on average on signals of 96. It "
            f"should miss fewer than {BUMP_BAR}. Ask what the count does as the signal grows, and what a bump looks "
            "like whatever level its signal sits on.",
        )

    kept_promise(bump_shift_gap(trained), BUMP_SHIFT_BAR, "BumpCounter", "the signals move along by 4 places", "{:.2f} bumps")
    twin_or_rule(found, "BumpCounter", train_bumps, bump_miss, lambda score: score <= BUMP_BAR, "the bumps point down")


def check_basket_total(found: type) -> None:
    untrained, trained = trained_or_crash(found, "BasketTotal", train_baskets)
    miss = tested_or_crash(basket_miss, trained, "BasketTotal", "baskets of 12")
    untrained_miss = tested_or_crash(basket_miss, untrained, "BasketTotal", "baskets of 12")

    if untrained_miss <= BASKET_BAR:
        raise Mismatch("Untrained, BasketTotal already gets the totals right. It should learn them, not compute them by a fixed rule.")

    if miss > BASKET_BAR:
        raise Mismatch(
            f"Trained on baskets of 5, BasketTotal misses {miss:.1%} of the average total on baskets of 12. It "
            f"should miss less than {BASKET_BAR:.0%}. Ask what a total does as the basket grows.",
        )

    kept_promise(basket_shuffle_gap(trained), BASKET_SHUFFLE_BAR, "BasketTotal", "each basket's items come in a new order", "{:.1%} of a total")
    twin_or_rule(found, "BasketTotal", train_baskets, basket_miss, lambda score: score <= BASKET_BAR, "each item adds its count over its price")


def check_key_lookup(found: type) -> None:
    untrained, trained = trained_or_crash(found, "KeyLookup", train_lookups)
    share = tested_or_crash(lookup_share, trained, "KeyLookup", "12 pairs")
    untrained_share = tested_or_crash(lookup_share, untrained, "KeyLookup", "12 pairs")

    if untrained_share >= LOOKUP_BAR:
        raise Mismatch("Untrained, KeyLookup already answers the lookups. It should learn them, not compute them by a fixed rule.")

    if share < LOOKUP_BAR:
        raise Mismatch(
            f"Trained on 4 pairs, KeyLookup answers {share:.1%} of lookups among 12 pairs right. It should pass "
            f"{LOOKUP_BAR:.0%}. Ask whether the query has to find its key among the pairs.",
        )

    kept_promise(lookup_shuffle_gap(trained), LOOKUP_SHUFFLE_BAR, "KeyLookup", "the pairs come in a new order", "{:.1%} of its answers")
    twin_or_rule(
        found,
        "KeyLookup",
        train_lookups,
        lookup_share,
        lambda score: score >= LOOKUP_BAR,
        "the query is one more than its key",
        "either a fixed rule is doing the work, or the query and the keys share one table, which can only match a "
        "query to its own id. Give the query a table or a projection of its own.",
    )


PROMISE_REASONS = {
    ("bumps", "shuffle"): "Shuffling a signal's places breaks its bumps apart, so the count changes. Which change keeps every bump whole?",
    ("baskets", "shift"): "A basket's items have no order at all, so any new order keeps its total, not only a shift.",
    ("lookups", "shift"): "A lookup's pairs have no order at all, so any new order keeps the answer, not only a shift.",
}


def check_promises(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(PROMISES):
        raise Mismatch(f"PROMISES is {value_text(value)}, but it should be a dict with the keys {sorted(PROMISES)}.")

    for problem, promise in PROMISES.items():
        given = value[problem]

        if given not in ("shift", "shuffle"):
            raise Mismatch(f"PROMISES[{problem!r}] is {value_text(given)}, but it should be \"shift\" or \"shuffle\".")

        if given != promise:
            raise Mismatch(f"PROMISES[{problem!r}] is {given!r}. {PROMISE_REASONS[(problem, given)]}")


def check_combine5(checker: Checker) -> None:
    checker.check_name(
        name="PROMISES",
        description="names the change each problem's answer ignores",
        run=check_promises,
    )
    checker.test_method(
        class_name="BumpCounter",
        method="forward",
        description="trained on short signals, counts the bumps in long ones",
        run=check_bump_counter,
    )
    checker.test_method(
        class_name="BasketTotal",
        method="forward",
        description="trained on baskets of 5, totals baskets of 12",
        run=check_basket_total,
    )
    checker.test_method(
        class_name="KeyLookup",
        method="forward",
        description="trained on 4 pairs, answers lookups among 12",
        run=check_key_lookup,
    )


COMBINE5 = Exercise(
    name="combine5",
    chapter=COMBINING,
    title="Picking the structure",
    lesson=COMBINE5_LESSON,
    source=COMBINE5_SOURCE,
    solution=COMBINE5_SOLUTION,
    hints={
        "PROMISES": "Ask of each problem which change leaves its true answer alone. A shuffled signal's bumps fall "
        "apart. A basket's total and a lookup's answer don't care what order their parts come in.",
        "BumpCounter.forward": "A bump means the same anywhere, so slide a small detector along the signal with "
        "nn.Conv1d, as conv4 did. A count grows with the signal, so add up a score per place rather than averaging.",
        "BasketTotal.forward": "The order of the items means nothing, as in sets1. Run one small MLP on every item, "
        "then pool. A total grows with the basket, so the pool is a sum.",
        "KeyLookup.forward": "The query has to find its key, as in attention6. Score each key against the query with "
        "two embeddings, softmax the scores, and take the weighted sum of the values' embeddings, then a head.",
    },
    introduces=(),
    check=check_combine5,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_film(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """combine1's table: error on new inputs and on inputs 3 times larger, after 1,500 steps."""
    x, condition, y = instruction_data(4000, torch.Generator().manual_seed(0))
    test = instruction_data(2000, torch.Generator().manual_seed(1))
    larger = instruction_data(2000, torch.Generator().manual_seed(2), 3.0)
    makers = {
        "joined, 16": lambda: ConcatNet(),
        "joined, two of 32": lambda: ConcatNet(32, True),
        "FiLM, 16": lambda: ReferenceFilmNet(),
    }
    results: dict[str, list[tuple[float, float]]] = {"guess": [(test[2].var().item(), larger[2].var().item())]}

    with pinned_threads():
        for name, make in makers.items():
            results[name] = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = train_conditioned(make(), (x, condition), y, 1500, seed, 3e-3)
                results[name].append((conditioned_error(model, test[:2], test[2]), conditioned_error(model, larger[:2], larger[2])))

    return results


def film_error_with_mean(
    model: ReferenceFilmNet,
    inputs: tuple[Tensor, Tensor],
    y: Tensor,
    means: tuple[Tensor, Tensor],
    averaged: str,
) -> float:
    """The paper's test time ablation: the scale or the shift put back to its mean over the training data, so the
    condition no longer moves it."""
    x, condition = inputs
    mean_scale, mean_shift = means

    with torch.no_grad():
        scale = model.film.to_scale(condition)
        shift = model.film.to_shift(condition)

        if averaged == "scale":
            scale = mean_scale.expand_as(scale)

        if averaged == "shift":
            shift = mean_shift.expand_as(shift)

        h = torch.relu(scale.unsqueeze(1) * model.embed(x) + shift.unsqueeze(1))

        return F.mse_loss(model.head(h.mean(dim=1)).squeeze(1), y).item()


def measure_film_ablation(seeds: range = range(3)) -> dict[str, list[float]]:
    """combine1's second table: the trained FilmNet's error on new inputs with its scale, or its shift, held at
    the mean over the training data, as in Perez and colleagues' ablation."""
    x, condition, y = instruction_data(4000, torch.Generator().manual_seed(0))
    test_x, test_condition, test_y = instruction_data(2000, torch.Generator().manual_seed(1))
    results: dict[str, list[float]] = {"guess": [test_y.var().item()]}

    with pinned_threads():
        for seed in seeds:
            torch.manual_seed(seed)
            model = train_conditioned(ReferenceFilmNet(), (x, condition), y, 1500, seed, 3e-3)

            with torch.no_grad():
                means = (model.film.to_scale(condition).mean(dim=0), model.film.to_shift(condition).mean(dim=0))

            for averaged in ("nothing", "shift", "scale"):
                error = film_error_with_mean(model, (test_x, test_condition), test_y, means, averaged)
                results.setdefault(averaged, []).append(error)

    return results


def measure_gating(seeds: range = range(3)) -> dict[str, list[tuple[float, ...]]]:
    """combine2's table, and how far each trained gated model's effective weights sit from A and B."""
    x, control, y = switch_data(4000, torch.Generator().manual_seed(0))
    test = switch_data(2000, torch.Generator().manual_seed(1))
    larger = switch_data(2000, torch.Generator().manual_seed(2), 4.0)
    results: dict[str, list[tuple[float, ...]]] = {"guess": [(test[2].var(dim=0).mean().item(), larger[2].var(dim=0).mean().item())]}

    with pinned_threads():
        for name, make in (("added", AddedLinear), ("added, hidden", ReferenceAdded), ("gated", ReferenceGated)):
            results[name] = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = train_conditioned(make(), (x, control), y, 1000, seed, 1e-2)
                results[name].append((conditioned_error(model, test[:2], test[2]), conditioned_error(model, larger[:2], larger[2])))

                if name == "gated":
                    weights = reference_effective_weights(model, torch.eye(2))
                    results.setdefault("weights off A and B", []).append(
                        ((weights[0] - MAP_A).abs().max().item(), (weights[1] - MAP_B).abs().max().item()),
                    )

    return results


def measure_summing(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """combine2's second table: the same models on the sum task, error on new inputs and on inputs 4 times larger."""
    x, second, y = sum_data(4000, torch.Generator().manual_seed(0))
    test = sum_data(2000, torch.Generator().manual_seed(1))
    larger = sum_data(2000, torch.Generator().manual_seed(2), 4.0)
    results: dict[str, list[tuple[float, float]]] = {"guess": [(test[2].var(dim=0).mean().item(), larger[2].var(dim=0).mean().item())]}

    with pinned_threads():
        for name, make in (("added", AddedLinear), ("added, hidden", ReferenceAdded), ("gated", ReferenceGated)):
            results[name] = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = train_conditioned(make(), (x, second), y, 1000, seed, 1e-2)
                results[name].append((conditioned_error(model, test[:2], test[2]), conditioned_error(model, larger[:2], larger[2])))

    return results


def trained_base(useful: bool) -> tuple[nn.Module, float]:
    x, _, y = side_data(4000, torch.Generator().manual_seed(0), useful)
    test_x, _, test_y = side_data(2000, torch.Generator().manual_seed(1), useful)
    torch.manual_seed(0)
    base = Base()
    optimizer = torch.optim.Adam(base.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(3)

    for _ in range(2000):
        batch = torch.randint(0, 4000, (128,), generator=generator)
        loss = F.mse_loss(base(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return base, F.mse_loss(base(test_x), test_y).item()


def started_branch(
    start: str,
    hidden: int,
) -> nn.Module:
    branch = make_branch(hidden)

    if start == "kaiming":
        for layer in (branch[0], branch[-1]):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    if start == "last zero":
        reference_zero_last_(branch)

    if start == "all zero":
        for layer in (branch[0], branch[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    return branch


BRANCH_RATE = 1e-2
TRAINED_RATE = 1e-3


def measure_branch_starts(seeds: range = range(3)) -> dict[str, object]:
    """combine3's table, with the branch at its own rate of 0.01. With signal the branch is wide, 512 hidden units, as
    ControlNet's branch is a full copy of its model, and the trained model trains too, at 0.001. Without signal the
    branch has 64 units, since 64 examples overfit a wide one from any start, and the trained model stays frozen, as
    ControlNet locks its model, since fine tuning it on 64 examples would raise its error with no branch at all. For
    each start, the error at step 0, the worst error checked every 50 steps, how many of those checks sat above the
    trained model alone, and the error at step 1,500, with signal on 4,000 examples and without on 64."""
    results: dict[str, object] = {}

    with pinned_threads():
        for useful, examples, hidden in ((True, 4000, 512), (False, 64, 64)):
            x, side, y = side_data(4000, torch.Generator().manual_seed(0), useful)
            test_x, test_side, test_y = side_data(2000, torch.Generator().manual_seed(1), useful)
            base, base_error = trained_base(useful)
            results[f"{'signal' if useful else 'no signal'}, trained model alone"] = base_error

            for start in ("last zero", "default", "kaiming", "all zero"):
                runs = []

                for seed in seeds:
                    torch.manual_seed(10 + seed)
                    model = ReferenceWithBranch(copy.deepcopy(base), started_branch(start, hidden), 1)
                    groups = [{"params": model.branch.parameters(), "lr": BRANCH_RATE}]

                    if useful:
                        groups.append({"params": model.base.parameters(), "lr": TRAINED_RATE})

                    optimizer = torch.optim.Adam(groups)
                    generator = torch.Generator().manual_seed(20 + seed)
                    marks = []

                    for step in range(1501):
                        if step % 50 == 0:
                            with torch.no_grad():
                                marks.append(F.mse_loss(model(test_x, test_side), test_y).item())

                        batch = torch.randint(0, examples, (min(128, examples),), generator=generator)
                        loss = F.mse_loss(model(x[batch], side[batch]), y[batch])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    worse = sum(mark > base_error * 1.0001 for mark in marks)
                    runs.append((marks[0], max(marks), worse, marks[-1]))

                results[f"{'signal' if useful else 'no signal'}, {start}"] = runs

    return results


def measure_mixing(seeds: range = range(3)) -> dict[str, object]:
    """combine4's table: the error after 1,500 steps for each order of blocks."""
    x, y = neighbour_data(4000, torch.Generator().manual_seed(0))
    test_x, test_y = neighbour_data(2000, torch.Generator().manual_seed(1))
    results: dict[str, object] = {"guess": test_y.var().item()}

    with pinned_threads():
        for name, kinds in (
            ("within only", ["within"] * 4),
            ("across only", ["across"] * 4),
            ("across first", ["across", "within", "across", "within"]),
            ("within first", ["within", "across", "within", "across"]),
        ):
            errors = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = Mixer(kinds)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
                generator = torch.Generator().manual_seed(seed + 5)

                for _ in range(1500):
                    batch = torch.randint(0, 4000, (128,), generator=generator)
                    loss = F.mse_loss(model(x[batch]), y[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    errors.append(F.mse_loss(model(test_x), test_y).item())

            results[name] = (sum(parameter.numel() for parameter in model.parameters()), errors)

    return results


def measure_across(seeds: range = range(3)) -> dict[str, object]:
    """combine4's second table: the across block swapped, each alternated with within blocks as across first is,
    trained on 8 positions for 1,500 steps, and its error on 8 positions and on 16, or None where it can't run."""
    x, y = neighbour_data(4000, torch.Generator().manual_seed(0))
    test_x, test_y = neighbour_data(2000, torch.Generator().manual_seed(1))
    long_x, long_y = neighbour_data(2000, torch.Generator().manual_seed(3), 16)
    results: dict[str, object] = {"guess": (test_y.var().item(), long_y.var().item())}

    with pinned_threads():
        for name, token_mix in (
            ("token MLP", ReferenceTokenMix),
            ("attention", AttentionMix),
            ("convolution", lambda width, positions, hidden: ConvMix(width)),
        ):
            errors = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = Mixer(["across", "within", "across", "within"], token_mix=token_mix)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
                generator = torch.Generator().manual_seed(seed + 5)

                for _ in range(1500):
                    batch = torch.randint(0, 4000, (128,), generator=generator)
                    loss = F.mse_loss(model(x[batch]), y[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    error = F.mse_loss(model(test_x), test_y).item()

                    try:
                        long_error = F.mse_loss(model(long_x), long_y).item()
                    except (RuntimeError, IndexError):
                        long_error = None

                errors.append((error, long_error))

            results[name] = (sum(parameter.numel() for parameter in model.parameters()), errors)

    return results


class PlainSumCounter(nn.Module):
    """A count read off the sum of the signal, with no detector, which a level of its own throws off."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, signals: Tensor) -> Tensor:
        return self.scale * signals.sum(dim=1)


class LinearBumpCounter(ReferenceBumpCounter):
    """The detector with no ReLU, so it adds up as one fixed weighting of the signal."""

    def forward(self, signals: Tensor) -> Tensor:
        return self.score(self.detect(signals.unsqueeze(1))).squeeze(1).sum(dim=1)


class RuleBasketTotal(nn.Module):
    """The total written by hand, with only its scale learned."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, items: Tensor) -> Tensor:
        return self.scale * (items[..., 0] * items[..., 1]).sum(dim=1)


class RuleKeyLookup(nn.Module):
    """The match written by hand, keys equal to the query, with only the values' table and head learned."""

    def __init__(self) -> None:
        super().__init__()
        self.value_table = nn.Embedding(IDS, 16)
        self.head = nn.Linear(16, IDS)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        weights = (keys == query.unsqueeze(1)).float()

        return self.head((weights.unsqueeze(-1) * self.value_table(values)).sum(dim=1))


def measure_problems(seeds: range = range(3)) -> dict[str, list[tuple[float, float, float | None]]]:
    """The right structure and wrong ones for each of combine5's problems, scored as the checks score them: the
    score on the bigger inputs, on the twin task, and the move under the change the answer ignores."""
    cases = (
        ("bumps, sum", ReferenceBumpCounter, train_bumps, bump_miss, bump_shift_gap),
        ("bumps, mean", MeanBumpCounter, train_bumps, bump_miss, bump_shift_gap),
        ("bumps, plain sum of the signal", PlainSumCounter, train_bumps, bump_miss, None),
        ("bumps, no ReLU", LinearBumpCounter, train_bumps, bump_miss, None),
        ("baskets, sum", ReferenceBasketTotal, train_baskets, basket_miss, basket_shuffle_gap),
        ("baskets, mean", MeanBasketTotal, train_baskets, basket_miss, basket_shuffle_gap),
        ("baskets, rule by hand", RuleBasketTotal, train_baskets, basket_miss, None),
        ("lookups, attention", ReferenceKeyLookup, train_lookups, lookup_share, lookup_shuffle_gap),
        ("lookups, GRU", GRULookup, train_lookups, lookup_share, lookup_shuffle_gap),
        ("lookups, match by hand", RuleKeyLookup, train_lookups, lookup_share, None),
    )
    results: dict[str, list[tuple[float, float, float | None]]] = {}

    with pinned_threads():
        for name, make, train, measure, gap in cases:
            results[name] = []

            for seed in seeds:
                torch.manual_seed(seed)
                model = train(make(), seed)
                torch.manual_seed(seed)
                twin = train(make(), seed, True)
                results[name].append((measure(model), measure(twin, True), None if gap is None else gap(model)))

    return results
