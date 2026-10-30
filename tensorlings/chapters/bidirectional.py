"""42_bidirectional, one RNN forwards and one backwards, matched with PyTorch's, and when reading ahead is fair."""

from __future__ import annotations

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
    expect_tensor,
    small_tensor,
)
from ..text import prose, source_file

BIDIRECTIONAL = Chapter(
    folder="42_bidirectional",
    title="Bidirectional RNNs",
    readme=source_file(
        """
        # Bidirectional RNNs

        An RNN's state at a position knows only what came before it. A
        bidirectional RNN adds a second RNN that reads from the end, so every
        position sees the whole sequence. This chapter builds one, matches it
        with PyTorch's, and shows when reading ahead is fair.

        ## Further reading

        - [Schuster and Paliwal, bidirectional recurrent neural networks](https://ieeexplore.ieee.org/document/650093)
        - [Dive into Deep Learning, bidirectional RNNs](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html)
        """,
    ),
)


def run_watching_autograd(
    modules: list[nn.Module],
    run: Callable[[], object],
) -> tuple[object, bool]:
    """Runs run while hooks note whether any of modules was called with autograd recording. A function told
    to record nothing gives itself away here, even when its answer is right."""
    recorded: list[bool] = []
    handles = [
        module.register_forward_pre_hook(lambda _module, _inputs: recorded.append(torch.is_grad_enabled()))
        for module in modules
    ]

    try:
        result = run()
    finally:
        for handle in handles:
            handle.remove()

    return result, any(recorded)


def expect_nothing_recorded(
    function: str,
    modules: list[nn.Module],
    run: Callable[[], object],
) -> None:
    _, recorded = run_watching_autograd(modules, run)

    if recorded:
        raise Mismatch(f"{function} recorded for autograd. Run the model inside torch.no_grad().")


def reference_run_rnn(
    x: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
) -> Tensor:
    h = torch.zeros(len(x), weight_hh.shape[0])
    outputs = []

    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh)
        outputs.append(h)

    return torch.stack(outputs, dim=1)


def seeded_bidirectional(
    input_size: int,
    hidden_size: int,
    seed: int,
) -> nn.RNN:
    torch.manual_seed(seed)

    return nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)


# ---------------------------------------------------------------------------
# bidir1
# ---------------------------------------------------------------------------

BIDIR1_LESSON = prose(
    """
    An RNN reads from left to right, so its state at a position knows only
    what came before. A bidirectional RNN, from Schuster and Paliwal in 1997,
    runs a second RNN from right to left and joins the two states at every
    position. Then each position sees the whole sequence, before and after.

    The backward RNN reads the sequence flipped along its steps. Its states
    come out in flipped order too, so flip them back, and each one lines up
    with its own position.

    ```python
    torch.tensor([[1, 2, 3]]).flip(1)    # tensor([[3, 2, 1]])
    ```

    `nn.RNN(input, hidden, batch_first=True, bidirectional=True)` holds two
    sets of weights. The backward set has the same names with `_reverse` on
    the end, like `weight_ih_l0_reverse`. Its output has 2 * hidden features
    at each step, the forward state first, then the backward one.

    The last state it returns has shape (2, batch, hidden). Its second row is
    the backward RNN's state after the whole sequence. That state belongs to
    position 0 of the output, not to the last position.
    """,
)

BIDIR1_SOURCE = source_file(
    '''
    # bidir1.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The RNN of rnn1, its state at every step, starting from zeros.
    def run_rnn(
        x: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        h = torch.zeros(len(x), weight_hh.shape[0])
        outputs = []

        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


    flipped = torch.tensor([[1, 2, 3]]).flip(1)    # tensor([[3, 2, 1]])


    # TODO: Return the backward RNN's state at every position of x, of shape
    #   (batch, steps, input), as one tensor of shape (batch, steps, hidden).
    #   Run run_rnn on x flipped along its steps, then flip the states back.
    def run_backward(
        x: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        ...


    joined = torch.cat([torch.zeros(1, 2), torch.ones(1, 2)], dim=1)    # tensor([[0., 0., 1., 1.]])


    # TODO: Return what rnn, an nn.RNN with bidirectional=True and
    #   batch_first=True, gives at every position of x. Join its forward
    #   states and its backward states along the features.
    def run_bidirectional(
        x: Tensor,
        rnn: nn.RNN,
    ) -> Tensor:
        ...
    ''',
)

BIDIR1_SOLUTION = source_file(
    '''
    # bidir1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def run_rnn(
        x: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        h = torch.zeros(len(x), weight_hh.shape[0])
        outputs = []

        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


    flipped = torch.tensor([[1, 2, 3]]).flip(1)


    def run_backward(
        x: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        return run_rnn(x.flip(1), weight_ih, weight_hh, bias_ih, bias_hh).flip(1)


    joined = torch.cat([torch.zeros(1, 2), torch.ones(1, 2)], dim=1)


    def run_bidirectional(
        x: Tensor,
        rnn: nn.RNN,
    ) -> Tensor:
        forward = run_rnn(x, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0)
        backward = run_backward(x, rnn.weight_ih_l0_reverse, rnn.weight_hh_l0_reverse, rnn.bias_ih_l0_reverse, rnn.bias_hh_l0_reverse)

        return torch.cat([forward, backward], dim=2)
    ''',
)


def backward_weights(seed: int) -> tuple[Described, ...]:
    rnn = seeded_bidirectional(3, 4, seed)
    names = ("weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse")

    return tuple(Described(getattr(rnn, name).detach(), f"{name} of shape {tuple(getattr(rnn, name).shape)}") for name in names)


def bidirectional_outputs(
    x: Tensor,
    rnn: nn.RNN,
) -> Tensor:
    with torch.no_grad():
        return rnn(x)[0]


def check_bidir1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="run_backward",
        description="reads from the end and lines each state up with its position",
        inputs=(
            (small_tensor(460, 2, 5, 3) / 2, *backward_weights(461)),
            (small_tensor(462, 1, 4, 3) / 2, *backward_weights(463)),
        ),
        reference=lambda x, *weights: reference_run_rnn(x.flip(1), *weights).flip(1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="run_bidirectional",
        description="matches nn.RNN with bidirectional=True",
        inputs=(
            (small_tensor(464, 2, 5, 3) / 2, Described(seeded_bidirectional(3, 4, 465), "nn.RNN(3, 4, batch_first=True, bidirectional=True)")),
            (small_tensor(466, 3, 4, 2) / 2, Described(seeded_bidirectional(2, 6, 467), "nn.RNN(2, 6, batch_first=True, bidirectional=True)")),
        ),
        reference=bidirectional_outputs,
        expect=expect_tensor,
    )


BIDIR1 = Exercise(
    name="bidir1",
    chapter=BIDIRECTIONAL,
    title="Reading both ways",
    lesson=BIDIR1_LESSON,
    source=BIDIR1_SOURCE,
    solution=BIDIR1_SOLUTION,
    hints={
        "run_backward": "Call run_rnn on x.flip(1) with the weights, then .flip(1) the result.",
        "run_bidirectional": "run_rnn with rnn.weight_ih_l0 and the other _l0 weights gives the forward states, and "
        "run_backward with the _l0_reverse ones the backward states. Join them with torch.cat along dim=2.",
    },
    introduces=(),
    check=check_bidir1,
)


# ---------------------------------------------------------------------------
# bidir2
# ---------------------------------------------------------------------------


def reference_peak_tags(x: Tensor) -> Tensor:
    lowest = torch.full_like(x[:, :1], float("-inf"))
    left = torch.cat([lowest, x[:, :-1]], dim=1)
    right = torch.cat([x[:, 1:], lowest], dim=1)

    return ((x > left) & (x > right)).float()


class ReferenceTagger(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.layer = nn.GRU(1, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs, _ = self.layer(x.unsqueeze(2))

        return self.head(outputs).squeeze(2)


def reference_generate(
    model: nn.Module,
    prefix: Tensor,
    count: int,
) -> Tensor:
    sequence = prefix

    with torch.no_grad():
        for _ in range(count):
            sequence = torch.cat([sequence, model(sequence)[:, -1:]], dim=1)

    return sequence[:, prefix.shape[1] :]


def seeded_tagger(
    make: type,
    hidden_size: int,
    bidirectional: bool,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(hidden_size, bidirectional)


def tagging_accuracy(
    model: nn.Module,
    seed: int,
    train_steps: int,
) -> float:
    """The run behind the lesson's first table, Adam at 0.01 on batches of 64 sequences of 20 numbers."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(train_steps):
        x = torch.randn(64, 20, generator=generator)
        loss = F.binary_cross_entropy_with_logits(model(x), reference_peak_tags(x))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x = torch.randn(1000, 20, generator=torch.Generator().manual_seed(77))

    with torch.no_grad():
        return ((model(x) > 0).float() == reference_peak_tags(x)).float().mean().item()


def noisy_waves(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sine waves of random speed and start, with a little noise on top, and the waves without it."""
    speed = 0.2 + 0.4 * torch.rand(count, 1, generator=generator)
    start = 2 * torch.pi * torch.rand(count, 1, generator=generator)
    clean = torch.sin(speed * torch.arange(steps) + start)

    return clean + 0.1 * torch.randn(count, steps, generator=generator), clean


def leak_errors(
    model: nn.Module,
    seed: int,
    train_steps: int,
) -> tuple[float, float]:
    """The run behind the lesson's second table. It trains on the next value of noisy waves, then measures
    that error and the error of 20 steps the model writes itself after 20 real ones."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(train_steps):
        x, _ = noisy_waves(64, 30, generator)
        loss = F.mse_loss(model(x[:, :-1]), x[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x, clean = noisy_waves(500, 40, torch.Generator().manual_seed(77))

    with torch.no_grad():
        teacher = F.mse_loss(model(x[:, :-1]), x[:, 1:]).item()

    written = reference_generate(model, x[:, :20], 20)

    return teacher, F.mse_loss(written, clean[:, 20:]).item()


BIDIR2_LESSON = prose(
    """
    A bidirectional RNN sees the whole sequence at every position. Whether
    that is fair depends on the job.

    Tagging labels every position of a sequence that already exists in full,
    like marking each word of a sentence as a noun or a verb. Reading ahead is
    fair there, since the end is already written. Here each of 20 random
    numbers is tagged 1 when it is larger than both its neighbours, a peak. A
    GRU of 16 units trains for 300 steps.

    | Reads | Tags right |
    |---|---|
    | forward only | 86.3% |
    | both ways | 98.2% |

    Tagging nothing as a peak already gets 64.9%. Reading forward only, a
    step can't see whether the next number is smaller, so it guesses from its
    own size.

    Generating writes a sequence one step at a time, each new step from the
    steps before it. The future doesn't exist yet, so a model that needs it
    can't run. Worse, it looks better in training. Train each model to give
    the next value of a noisy wave, the target being the input moved one step.

    ```python
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x[:, :-1], x[:, 1:]    # (tensor([[1., 2., 3.]]), tensor([[2., 3., 4.]]))
    ```

    | Reads | Error on the next value in training | Error on 20 steps it wrote itself |
    |---|---|---|
    | forward only | 0.021 | 0.052 |
    | both ways | 0.0025 | 0.555 |

    The backward RNN reads the very value it is asked to give, so its error
    in training is 8 times lower. When it writes its own steps that value
    isn't there yet, and it does worse than a flat line at 0, which scores
    0.500.

    So read both ways when the whole input exists before any output is
    needed. Tagging is one case, and so is an encoder that reads a whole
    sentence before 43_seq2seq's decoder writes. Read forward only when the
    model writes its own next step. Section 7 meets this same leak again.
    """,
)

BIDIR2_SOURCE = source_file(
    '''
    # bidir2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 3.0, 2.0]])
    rises = x[:, 1:] > x[:, :-1]    # tensor([[ True, False]]), each step against the one before


    # TODO: Return 1.0 at every position of x, of shape (batch, steps), whose
    #   number is larger than both its neighbours, and 0.0 elsewhere. The
    #   first and last steps have one neighbour each.
    def peak_tags(x: Tensor) -> Tensor:
        ...


    both = nn.GRU(1, 16, batch_first=True, bidirectional=True)
    width = both(torch.zeros(2, 20, 1))[0].shape    # torch.Size([2, 20, 32]), both directions joined


    # TODO: Finish Tagger. self.head turns each step's output of self.layer
    #   into one score, so its input size depends on bidirectional. forward
    #   takes x of shape (batch, steps) and returns one score per step, of
    #   shape (batch, steps).
    class Tagger(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            bidirectional: bool,
        ) -> None:
            super().__init__()
            self.layer = nn.GRU(1, hidden_size, batch_first=True, bidirectional=bidirectional)
            self.head = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    longer = torch.cat([torch.zeros(2, 3), torch.ones(2, 1)], dim=1)    # shape (2, 4), one step longer


    # TODO: Return count steps that model writes after prefix, of shape
    #   (batch, steps), as one tensor of shape (batch, count). At each step,
    #   run model on the sequence so far, take its score at the last position
    #   as the next value, and add it on. Record nothing for autograd.
    def generate(
        model: nn.Module,
        prefix: Tensor,
        count: int,
    ) -> Tensor:
        ...
    ''',
)

BIDIR2_SOLUTION = source_file(
    '''
    # bidir2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 3.0, 2.0]])
    rises = x[:, 1:] > x[:, :-1]


    # Each step's neighbours, with -inf past the ends so the ends compare with one neighbour only.
    def peak_tags(x: Tensor) -> Tensor:
        lowest = torch.full_like(x[:, :1], float("-inf"))
        left = torch.cat([lowest, x[:, :-1]], dim=1)
        right = torch.cat([x[:, 1:], lowest], dim=1)

        return ((x > left) & (x > right)).float()


    both = nn.GRU(1, 16, batch_first=True, bidirectional=True)
    width = both(torch.zeros(2, 20, 1))[0].shape


    class Tagger(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            bidirectional: bool,
        ) -> None:
            super().__init__()
            self.layer = nn.GRU(1, hidden_size, batch_first=True, bidirectional=bidirectional)
            self.head = nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1)

        def forward(self, x: Tensor) -> Tensor:
            outputs, _ = self.layer(x.unsqueeze(2))

            return self.head(outputs).squeeze(2)


    longer = torch.cat([torch.zeros(2, 3), torch.ones(2, 1)], dim=1)


    def generate(
        model: nn.Module,
        prefix: Tensor,
        count: int,
    ) -> Tensor:
        sequence = prefix

        with torch.no_grad():
            for _ in range(count):
                sequence = torch.cat([sequence, model(sequence)[:, -1:]], dim=1)

        return sequence[:, prefix.shape[1] :]
    ''',
)


def check_tagger_forward(found: type) -> None:
    for bidirectional in (False, True):
        call = f"Tagger(4, {bidirectional})"
        model = seeded_tagger(found, 4, bidirectional, 470)
        reference = seeded_tagger(ReferenceTagger, 4, bidirectional, 470)
        x = small_tensor(471, 3, 6) / 2

        with torch.no_grad():
            expected = reference(x)

        expect_tensor(
            call=f"{call} on 3 sequences of 6 steps",
            actual=model(x),
            expected=expected,
        )


def check_tagger_learns(found: type) -> None:
    accuracy = tagging_accuracy(
        model=seeded_tagger(found, 16, True, 0),
        seed=0,
        train_steps=300,
    )

    if accuracy < 0.95:
        raise Mismatch(
            f"Tagger(16, True) trained for 300 steps tags {accuracy:.1%} of new steps right, but it should pass 95%.",
        )


def check_generate(found: object) -> None:
    prefix = small_tensor(472, 2, 5) / 3
    expected = reference_generate(seeded_tagger(ReferenceTagger, 4, False, 473), prefix, 4)
    model = seeded_tagger(ReferenceTagger, 4, False, 473)
    expect_tensor(
        call="generate with Tagger(4, False), a prefix of shape (2, 5) and count 4",
        actual=found(model, prefix.clone(), 4),
        expected=expected,
    )
    expect_nothing_recorded(
        function="generate",
        modules=[model],
        run=lambda: found(model, prefix.clone(), 4),
    )


def check_bidir2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="peak_tags",
        description="tags each number larger than both neighbours",
        inputs=(
            (torch.tensor([[1.0, 3.0, 2.0, 5.0], [4.0, 1.0, 2.0, 0.0]]),),
            (torch.tensor([[2.0, 2.0, 1.0, 3.0, 3.0]]),),
            (small_tensor(474, 3, 7) + torch.linspace(0.0, 0.1, 7),),
        ),
        reference=reference_peak_tags,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="Tagger",
        method="forward",
        description="gives one score per step, reading one way or both",
        run=check_tagger_forward,
    )
    checker.test_method(
        class_name="Tagger",
        method="forward",
        description="reading both ways, learns to tag the peaks",
        run=check_tagger_learns,
    )
    checker.test(
        function="generate",
        description="writes each next step from the steps before it",
        run=check_generate,
    )


BIDIR2 = Exercise(
    name="bidir2",
    chapter=BIDIRECTIONAL,
    title="When reading ahead is fair",
    lesson=BIDIR2_LESSON,
    source=BIDIR2_SOURCE,
    solution=BIDIR2_SOLUTION,
    hints={
        "peak_tags": "Build each step's left and right neighbour by shifting x one step with torch.cat, putting "
        "float('-inf') past the ends. A peak is larger than both, as floats.",
        "Tagger.forward": "In __init__, the head is nn.Linear(2 * hidden_size, 1) when bidirectional and "
        "nn.Linear(hidden_size, 1) when not. forward runs self.layer on x.unsqueeze(2), then the head, then "
        ".squeeze(2).",
        "generate": "Inside torch.no_grad(), repeat count times: model(sequence)[:, -1:] is the next value, so "
        "torch.cat it onto sequence along dim=1. Return the part after the prefix.",
    },
    introduces=(),
    check=check_bidir2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
