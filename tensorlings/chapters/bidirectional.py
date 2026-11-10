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
from .tensors import pinned_threads

BIDIRECTIONAL = Chapter(
    folder="42_bidirectional",
    title="Bidirectional RNNs",
    readme=source_file(
        """
        # Bidirectional RNNs

        The hidden state of an RNN at a position depends only on the steps up
        to that position. A bidirectional RNN adds a second RNN that reads the
        sequence from the end. So its output at every position depends on the
        whole sequence. This chapter builds one and checks it against
        PyTorch's version. Then it shows when it is fair for a model to read
        the steps after a position, and when it is not.

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
    An RNN reads from left to right. So its hidden state at a position
    depends only on that step and the steps before it. A bidirectional RNN,
    from Schuster and Paliwal in 1997, also runs a second RNN from right to
    left. At every position it joins the two hidden states into one vector.
    So the output at each position depends on the whole sequence, before and
    after it.

    The backward RNN reads the sequence with its steps in reverse order,
    which `flip` gives. Its hidden states also come out in reverse order.
    Flip them back, and the state at each index belongs to the step at that
    same index.

    ```python
    torch.tensor([[1, 2, 3]]).flip(1)    # tensor([[3, 2, 1]])
    ```

    `nn.RNN(input, hidden, batch_first=True, bidirectional=True)` holds two
    sets of weights. The backward set has the same names with `_reverse`
    added at the end, like `weight_ih_l0_reverse`. Its output has 2 * hidden
    features at each step. The first `hidden` of them are the forward state,
    and the rest are the backward state.

    The last state it returns has shape (2, batch, hidden). Its second row is
    the backward RNN's state after it has read the whole sequence, from the
    end back to the start. So that state matches position 0 of the output,
    and not the last position.
    """,
)

BIDIR1_SOURCE = source_file(
    '''
    # bidir1.py

    import torch
    import torch.nn as nn
    from torch import Tensor


    # The RNN from rnn1, run over a batch. It returns the hidden state at every step, starting from zeros.
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


    # TODO: Return the backward RNN's hidden state at every position of x, of
    #   shape (batch, steps, input), as one tensor of shape
    #   (batch, steps, hidden). Run run_rnn on x with its steps reversed. Then
    #   reverse the order of the states, so the state at index t belongs to
    #   step t.
    def run_backward(
        x: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        ...


    joined = torch.cat([torch.zeros(1, 2), torch.ones(1, 2)], dim=1)    # tensor([[0., 0., 1., 1.]])


    # TODO: rnn is an nn.RNN with bidirectional=True and batch_first=True.
    #   Return the same output that rnn gives for x, built with run_rnn and
    #   run_backward from rnn's weights. Join the forward states and the
    #   backward states along the feature dimension, forward first.
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
        description="reads from the end and puts each state back at its own position",
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
        "run_bidirectional": "run_rnn with rnn.weight_ih_l0 and the other _l0 tensors gives the forward states. "
        "run_backward with the _l0_reverse tensors gives the backward states. Join them with torch.cat along "
        "dim=2.",
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
    At every position, a bidirectional RNN sees the whole sequence. Whether
    it is fair to use the steps after a position depends on the task.

    Tagging gives a label to every position of a sequence that already exists
    in full. An example is marking each word of a sentence as a noun or a
    verb. Reading ahead is fair there, because the whole sequence is known before
    any label is needed. In this example, each of 20 random numbers gets the
    tag 1 when it is larger than both its neighbours, which makes it a peak.
    All the other numbers get 0. A GRU with 16 units trains for 300 steps.

    | Reads | Correct tags |
    |---|---|
    | forward only | 86.3% |
    | both ways | 98.2% |

    Tagging no number as a peak already gets 64.9% right. A model that reads
    forward only cannot see whether the next number is smaller. So it has to
    guess from the size of the number itself. Over five seeds, reading forward
    only scored 86.2% to 86.5%, and reading both ways scored 98.2% to 98.5%.

    Generating writes a sequence one step at a time, and computes each new
    step from the steps before it. The future steps do not exist yet, so a
    model that reads them cannot run. Worse, such a model looks better during
    training. To see this, train each model for 500 steps to predict the next
    value of a noisy sine wave. The target is the input shifted by one step.

    ```python
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x[:, :-1], x[:, 1:]    # (tensor([[1., 2., 3.]]), tensor([[2., 3., 4.]]))
    ```

    | Reads | Error on the next value in training | Error on 20 steps it wrote itself |
    |---|---|---|
    | forward only | 0.021 | 0.052 |
    | both ways | 0.0025 | 0.555 |

    In training, the backward RNN reads the exact value it is asked to
    predict, so its error is 8 times lower. When the model writes its own
    steps, that value does not exist yet. Then it does worse than always
    predicting 0, which has an error of 0.500. Over five seeds, the model that
    reads both ways scored 0.526 to 0.591 on the steps it wrote itself. The
    model that reads forward only scored 0.034 to 0.052.

    So read both ways when the whole input exists before any output is
    needed. Tagging is one case. Another is the encoder in 43_seq2seq, which
    reads a whole sentence before the decoder starts writing. Read forward
    only when the model writes its own next step. Letting a model see steps it
    will not have when it runs is called a leak. Section 7 meets the same
    leak again.
    """,
)

BIDIR2_SOURCE = source_file(
    '''
    # bidir2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.tensor([[1.0, 3.0, 2.0]])
    rises = x[:, 1:] > x[:, :-1]    # tensor([[ True, False]]), whether each step is larger than the one before


    # TODO: x has shape (batch, steps). Return a float tensor of the same
    #   shape, with 1.0 where a number is larger than both of its neighbours
    #   and 0.0 everywhere else. The first and last steps have only one
    #   neighbour each, so they only need to be larger than that one.
    #   Example: peak_tags(torch.tensor([[1.0, 3.0, 2.0, 5.0]])) returns
    #   tensor([[0., 1., 0., 1.]]).
    def peak_tags(x: Tensor) -> Tensor:
        ...


    both = nn.GRU(1, 16, batch_first=True, bidirectional=True)
    width = both(torch.zeros(2, 20, 1))[0].shape    # torch.Size([2, 20, 32]), both directions joined


    # TODO: Finish Tagger. self.head is a Linear layer that turns the output of
    #   self.layer at each step into one score. Its input size depends on
    #   bidirectional, because a bidirectional GRU outputs twice as many
    #   features. forward takes x of shape (batch, steps) and returns one score
    #   per step, of shape (batch, steps).
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


    # TODO: Generate count new steps after prefix, of shape (batch, steps), and
    #   return them as one tensor of shape (batch, count). At each step, run
    #   model on the sequence so far and take its output at the last position
    #   as the next value. Append that value to the sequence. Do not record
    #   anything for autograd.
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


    # The neighbours of each step, with -inf past both ends, so each end is compared with its one real neighbour.
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


def expect_float_tags(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """Tags are training targets, and binary_cross_entropy_with_logits takes floats, not True and False."""
    if isinstance(actual, Tensor) and not actual.is_floating_point():
        raise Mismatch(
            f"{call} holds entries of type {actual.dtype}, but it should hold floats, 1.0 for a peak and 0.0 "
            "elsewhere. .float() turns True and False into them.",
        )

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
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
        expect=expect_float_tags,
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
        "peak_tags": "Build tensors of each step's left and right neighbour by shifting x one step with torch.cat, "
        "and put float('-inf') past the ends. A peak is larger than both neighbours. Convert the result to floats.",
        "Tagger.forward": "In __init__, the head is nn.Linear(2 * hidden_size, 1) when bidirectional and "
        "nn.Linear(hidden_size, 1) when not. forward runs self.layer on x.unsqueeze(2), then the head, then "
        ".squeeze(2).",
        "generate": "Inside torch.no_grad(), repeat count times. model(sequence)[:, -1:] is the next value, so "
        "torch.cat it onto sequence along dim=1. Return the part after the prefix.",
    },
    introduces=(),
    check=check_bidir2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_reading(seeds: range = range(5)) -> dict[tuple[str, bool, int], float | tuple[float, float]]:
    """bidir2's two tables, keyed by table, whether the GRU of 16 reads both ways, and seed. Each model trains
    from torch.manual_seed(seed) on data drawn with seed + 1. "tagging" holds the share of steps tagged right after
    300 Adam steps, and "leak" the error on the next value in training and on 20 written steps, after 500. Seed 0
    is the tables' run."""
    results = {}

    with pinned_threads():
        for seed in seeds:
            for bidirectional in (False, True):
                results[("tagging", bidirectional, seed)] = tagging_accuracy(
                    model=seeded_tagger(ReferenceTagger, 16, bidirectional, seed),
                    seed=seed,
                    train_steps=300,
                )
                results[("leak", bidirectional, seed)] = leak_errors(
                    model=seeded_tagger(ReferenceTagger, 16, bidirectional, seed),
                    seed=seed,
                    train_steps=500,
                )

    return results


def measure_baselines() -> dict[str, float]:
    """bidir2's two baselines on its test data, tagging no step as a peak and writing a flat line at 0."""
    x = torch.randn(1000, 20, generator=torch.Generator().manual_seed(77))
    _, clean = noisy_waves(500, 40, torch.Generator().manual_seed(77))

    return {
        "tag nothing": (reference_peak_tags(x) == 0).float().mean().item(),
        "flat line": F.mse_loss(torch.zeros_like(clean[:, 20:]), clean[:, 20:]).item(),
    }
