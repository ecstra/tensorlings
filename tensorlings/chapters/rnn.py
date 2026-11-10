"""39_rnn, one layer reused at every step with a hidden state, batches of sequences and their shapes, and why a hidden state beats a fixed window."""

from __future__ import annotations

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
    expect_parts,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import pinned_threads

RNN = Chapter(
    folder="39_rnn",
    title="Recurrent networks",
    readme=source_file(
        """
        # Recurrent networks

        A recurrent network reads a sequence one step at a time. As it reads,
        it keeps a vector that summarizes the steps so far, called its hidden
        state. This chapter builds one step by hand and checks it against
        `nn.RNN`. Then it runs batches of sequences. It ends with a task where
        the hidden state beats a model that sees only a fixed window of recent
        steps, once the sequence is longer than the window.

        ## Further reading

        - [Karpathy, the unreasonable effectiveness of recurrent neural networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
        - [Dive into Deep Learning, recurrent neural networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
        - [PyTorch, nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        """,
    ),
)


def rnn_weights(
    seed: int,
    inputs: int,
    hidden: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    return (
        torch.randn(hidden, inputs, generator=generator) / 2,
        torch.randn(hidden, hidden, generator=generator) / 2,
        torch.randn(hidden, generator=generator) / 2,
        torch.randn(hidden, generator=generator) / 2,
    )


def reference_run_rnn(
    xs: Tensor,
    W_ih: Tensor,
    W_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tensor:
    """Any number of leading dimensions before the steps, so one reference serves one sequence and a batch."""
    h = torch.zeros(*xs.shape[:-2], W_hh.shape[0])
    states = []

    for t in range(xs.shape[-2]):
        h = torch.tanh(xs[..., t, :] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        states.append(h)

    return torch.stack(states, dim=-2)


# ---------------------------------------------------------------------------
# rnn1
# ---------------------------------------------------------------------------

RNN1_LESSON = prose(
    """
    Section 6 is about sequences, like the letters of a name, the samples of a
    sound recording, or a price recorded day by day. A sequence model reads a
    sequence one step at a time. As it reads, it keeps a summary of every step
    so far.

    That summary is the **hidden state**, a vector `h`. At each step, one
    layer combines the input of that step, `x`, with the previous hidden
    state. Then tanh squashes every entry of the result into the range -1 to
    1. The result is the new hidden state.

    ```text
    h = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
    ```

    The hidden state before the first step is all zeros. The same weights are
    used at every step. So a model that reads 10 steps has exactly the same
    parameters as one that reads 1000. Convolution reused its weights at every
    position of its input, and this model reuses its weights at every step in
    time. A model built this way is a **recurrent neural network**, or RNN.

    The two matrix multiplications can also be done by one Linear layer. Join
    `x` and `h` into one longer vector with `torch.cat`. Then make a weight
    that has the columns of `W_ih` followed by the columns of `W_hh`, and use
    the bias `b_ih + b_hh`. A Linear layer with that weight and bias, applied
    to the joined vector, gives the same sum as the formula.

    ```python
    torch.cat([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])    # tensor([1., 2., 3.])
    ```

    `nn.RNN(input_size, hidden_size)` stores the four tensors of the formula
    as `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0` and `bias_hh_l0`. The
    `l0` in each name means layer 0. With `batch_first=True`, it takes an
    input of shape (batch, steps, input_size) and returns two tensors. The
    first holds the hidden state of every step, with shape (batch, steps,
    hidden_size). The second holds only the last hidden state, with shape (1,
    batch, hidden_size).
    """,
)

RNN1_SOURCE = source_file(
    '''
    # rnn1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    squashed = torch.tanh(torch.tensor([0.0, 5.0]))    # tensor([0.0000, 0.9999])


    # TODO: Return the next hidden state, computed with the lesson's formula.
    #   x is the input of this step, of shape (input_size,), and h is the
    #   previous hidden state, of shape (hidden_size,). The result has shape
    #   (hidden_size,).
    def rnn_step(
        x: Tensor,
        h: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        ...


    rows = torch.stack([torch.ones(2), torch.zeros(2)])    # tensor([[1., 1.], [0., 0.]])


    # TODO: Run the RNN over the sequence xs, of shape (steps, input_size).
    #   Start from a hidden state of zeros and call rnn_step once for each row
    #   of xs. Return the hidden state after every step, stacked into a tensor
    #   of shape (steps, hidden_size).
    def run_rnn(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        ...


    side_by_side = torch.cat([torch.ones(2, 1), torch.zeros(2, 2)], dim=1)    # shape (2, 3)


    # TODO: Return the weight and the bias of one Linear layer that, applied to
    #   torch.cat([x, h]), gives the same sum as the lesson's formula. W_ih has
    #   shape (hidden_size, input_size) and W_hh has shape
    #   (hidden_size, hidden_size), so the weight you return has shape
    #   (hidden_size, input_size + hidden_size).
    def as_one_linear(
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    example = nn.RNN(3, 4)
    example.weight_hh_l0.shape    # torch.Size([4, 4])


    # TODO: rnn is an nn.RNN with one layer. Return the hidden states that
    #   run_rnn gives for xs, of shape (steps, input_size), using the four
    #   weight and bias tensors stored in rnn.
    def run_module(
        rnn: nn.RNN,
        xs: Tensor,
    ) -> Tensor:
        ...
    ''',
)

RNN1_SOLUTION = source_file(
    '''
    # rnn1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    squashed = torch.tanh(torch.tensor([0.0, 5.0]))


    def rnn_step(
        x: Tensor,
        h: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        return torch.tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)


    rows = torch.stack([torch.ones(2), torch.zeros(2)])


    def run_rnn(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        h = torch.zeros(W_hh.shape[0])
        states = []

        for x in xs:
            h = rnn_step(x, h, W_ih, W_hh, b_ih, b_hh)
            states.append(h)

        return torch.stack(states)


    side_by_side = torch.cat([torch.ones(2, 1), torch.zeros(2, 2)], dim=1)


    def as_one_linear(
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return torch.cat([W_ih, W_hh], dim=1), b_ih + b_hh


    example = nn.RNN(3, 4)
    example.weight_hh_l0.shape


    def run_module(
        rnn: nn.RNN,
        xs: Tensor,
    ) -> Tensor:
        return run_rnn(xs, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0)
    ''',
)


def seeded_rnn(
    seed: int,
    inputs: int,
    hidden: int,
) -> nn.RNN:
    torch.manual_seed(seed)

    return nn.RNN(inputs, hidden)


def check_rnn1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="rnn_step",
        description="gives the next hidden state",
        inputs=(
            (small_tensor(401, 3), small_tensor(402, 4) / 3, *rnn_weights(403, 3, 4)),
            (small_tensor(404, 2), small_tensor(405, 5) / 3, *rnn_weights(406, 2, 5)),
        ),
        reference=lambda x, h, W_ih, W_hh, b_ih, b_hh: torch.tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="run_rnn",
        description="gives every step's hidden state",
        inputs=(
            (small_tensor(407, 5, 3), *rnn_weights(408, 3, 4)),
            (small_tensor(409, 2, 2), *rnn_weights(410, 2, 3)),
        ),
        reference=reference_run_rnn,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="as_one_linear",
        description="joins the weights side by side and adds the biases",
        inputs=(
            rnn_weights(411, 3, 4),
            rnn_weights(412, 2, 5),
        ),
        reference=lambda W_ih, W_hh, b_ih, b_hh: (torch.cat([W_ih, W_hh], dim=1), b_ih + b_hh),
        expect=expect_parts("the weight", "the bias"),
    )
    check_on_inputs(
        checker=checker,
        function="run_module",
        description="matches nn.RNN",
        inputs=(
            (Described(seeded_rnn(413, 3, 4), "nn.RNN(3, 4)"), small_tensor(414, 6, 3)),
            (Described(seeded_rnn(415, 2, 6), "nn.RNN(2, 6)"), small_tensor(416, 4, 2)),
        ),
        reference=lambda rnn, xs: rnn(xs.unsqueeze(1))[0].squeeze(1).detach(),
        expect=expect_tensor,
    )


RNN1 = Exercise(
    name="rnn1",
    chapter=RNN,
    title="One layer, every step",
    lesson=RNN1_LESSON,
    source=RNN1_SOURCE,
    solution=RNN1_SOLUTION,
    hints={
        "rnn_step": "Write the formula as it appears in the lesson, with torch.tanh around the whole sum.",
        "run_rnn": "Start h at torch.zeros(W_hh.shape[0]). Loop over the rows of xs, update h with rnn_step, keep "
        "each h in a list, and torch.stack the list.",
        "as_one_linear": "torch.cat([W_ih, W_hh], dim=1) is the weight, and b_ih + b_hh the bias.",
        "run_module": "Call run_rnn with rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0 and rnn.bias_hh_l0.",
    },
    introduces=(),
    check=check_rnn1,
)


# ---------------------------------------------------------------------------
# rnn2
# ---------------------------------------------------------------------------

RNN2_LESSON = prose(
    """
    Training processes many sequences at once, as a batch. With
    `batch_first=True`, the input has shape (batch, steps, features) and the
    hidden state has shape (batch, hidden). Each step applies the formula from
    rnn1 to every sequence of the batch at the same time, one row per
    sequence.

    PyTorch's default is `batch_first=False`, which puts the steps first, as
    (steps, batch, features). A lot of code uses this layout, so check which
    one a model expects. `x.transpose(0, 1)` swaps the first two dimensions,
    which converts one layout into the other.

    `num_layers=2` stacks a second RNN on top of the first. The second RNN
    takes the hidden states of the first as its input sequence. The outputs
    are then the hidden states of the top layer. The last state holds one row
    per layer, so its first dimension is the number of layers.

    | Setting | Input | Outputs | Last state |
    |---|---|---|---|
    | `batch_first=True` | (batch, steps, features) | (batch, steps, hidden) | (layers, batch, hidden) |
    | `batch_first=False` | (steps, batch, features) | (steps, batch, hidden) | (layers, batch, hidden) |

    The last state never puts the batch first, whatever `batch_first` says.

    The sequences in one batch often have different lengths. The short ones
    are padded with zeros at the end, up to the length of the longest one. So
    the output you want from each sequence is its output at its own last real
    step. For a short sequence that is not the last step of the outputs,
    because the last step there comes from padding.

    To pick a different step for each sequence, index with two tensors. The
    first tensor lists the rows and the second lists the steps. Each pair of
    values, one from each tensor, picks one entry. Here row 0 takes step 2 and
    row 1 takes step 0.

    ```python
    outputs = torch.arange(12.0).view(2, 3, 2)
    outputs[torch.tensor([0, 1]), torch.tensor([2, 0])]    # tensor([[4., 5.], [6., 7.]])
    ```
    """,
)

RNN2_SOURCE = source_file(
    '''
    # rnn2.py

    import torch
    from torch import Tensor

    batch = torch.zeros(8, 5, 3)          # 8 sequences of 5 steps, 3 features each
    first_steps = batch[:, 0]             # shape (8, 3), step 0 of every sequence


    # TODO: Run the RNN over xs, a batch of sequences of shape
    #   (batch, steps, input_size), starting from hidden states of zeros. At
    #   each step t, process step t of every sequence at once, as one matrix of
    #   shape (batch, input_size). Return the hidden states of every step, as a
    #   tensor of shape (batch, steps, hidden_size).
    def run_rnn_batch(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the shape of the outputs and the shape of the last state that
    #   nn.RNN returns for these sizes, as two tuples. Use the lesson's table.
    #   Example: rnn_shapes(8, 5, 3, 16, 1, True) returns
    #   ((8, 5, 16), (1, 8, 16)).
    def rnn_shapes(
        batch: int,
        steps: int,
        features: int,
        hidden: int,
        layers: int,
        batch_first: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        ...


    # TODO: Return each sequence's output at its last real step, the step just
    #   before its padding starts. outputs has shape (batch, steps, hidden).
    #   lengths has shape (batch,) and holds the real length of each sequence.
    #   The result has shape (batch, hidden).
    def last_real_outputs(
        outputs: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        ...
    ''',
)

RNN2_SOLUTION = source_file(
    '''
    # rnn2.py, a solution for comparison

    import torch
    from torch import Tensor

    batch = torch.zeros(8, 5, 3)
    first_steps = batch[:, 0]


    def run_rnn_batch(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        h = torch.zeros(len(xs), W_hh.shape[0])
        states = []

        for t in range(xs.shape[1]):
            h = torch.tanh(xs[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
            states.append(h)

        return torch.stack(states, dim=1)


    def rnn_shapes(
        batch: int,
        steps: int,
        features: int,
        hidden: int,
        layers: int,
        batch_first: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        last = (layers, batch, hidden)

        if batch_first:
            return (batch, steps, hidden), last

        return (steps, batch, hidden), last


    def last_real_outputs(
        outputs: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        return outputs[torch.arange(len(outputs)), lengths - 1]
    ''',
)


def reference_rnn_shapes(
    batch: int,
    steps: int,
    features: int,
    hidden: int,
    layers: int,
    batch_first: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rnn = nn.RNN(features, hidden, num_layers=layers, batch_first=batch_first)
    x = torch.zeros(batch, steps, features) if batch_first else torch.zeros(steps, batch, features)
    outputs, last = rnn(x)

    return tuple(outputs.shape), tuple(last.shape)


def check_rnn2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="run_rnn_batch",
        description="runs every sequence of the batch at once",
        inputs=(
            (small_tensor(421, 3, 4, 2), *rnn_weights(422, 2, 5)),
            (small_tensor(423, 2, 6, 3), *rnn_weights(424, 3, 4)),
        ),
        reference=reference_run_rnn,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rnn_shapes",
        description="gives the shapes nn.RNN returns",
        inputs=(
            (8, 5, 3, 16, 1, True),
            (4, 7, 2, 10, 2, False),
            (6, 9, 5, 12, 3, True),
        ),
        reference=reference_rnn_shapes,
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=tuple(tuple(part) for part in actual) if isinstance(actual, (tuple, list)) else actual,
            expected=expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="last_real_outputs",
        description="picks each sequence's last real step",
        inputs=(
            (small_tensor(425, 3, 4, 2), torch.tensor([4, 1, 3])),
            (small_tensor(426, 2, 5, 3), torch.tensor([2, 5])),
        ),
        reference=lambda outputs, lengths: outputs[torch.arange(len(outputs)), lengths - 1],
        expect=expect_tensor,
    )


RNN2 = Exercise(
    name="rnn2",
    chapter=RNN,
    title="Batches of sequences",
    lesson=RNN2_LESSON,
    source=RNN2_SOURCE,
    solution=RNN2_SOLUTION,
    hints={
        "run_rnn_batch": "Start h at torch.zeros(len(xs), W_hh.shape[0]). For each t, xs[:, t] holds step t of "
        "every sequence. Stack the states with dim=1.",
        "rnn_shapes": "The last state is (layers, batch, hidden) either way. The outputs swap batch and steps when "
        "batch_first is False.",
        "last_real_outputs": "Index outputs with torch.arange(len(outputs)) for the rows and lengths - 1 for the "
        "steps.",
    },
    introduces=(),
    check=check_rnn2,
)


# ---------------------------------------------------------------------------
# rnn3
# ---------------------------------------------------------------------------


def vote_sequences(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Steps of +1 or -1, and whether the +1s won, as 1.0 or 0.0. An odd length means no ties."""
    x = torch.randint(0, 2, (count, steps), generator=generator).float() * 2 - 1

    return x, (x.sum(dim=1) > 0).float()


RNN3_LESSON = prose(
    """
    Here is a task that needs every step of the past. Each sequence is a vote
    of 21 steps, and each step is +1 or -1. The label is 1 when there are more
    +1s than -1s, and 0 otherwise. Every step counts toward the result, so a
    model cannot be sure of the answer without reading the whole sequence.

    A fixed window model reads only the last few steps and passes them through
    a small MLP. An RNN reads every step and keeps a running summary in its
    hidden state. Each model below was trained for 500 steps of Adam, on new
    random votes at every step.

    | Model | Accuracy on votes of 21 |
    |---|---|
    | MLP on the last 5 steps | 67.5% |
    | MLP on the last 11 steps | 77.9% |
    | MLP on all 21 steps | 100% |
    | RNN with a hidden state of 16 | 98.9% |

    The MLP on all 21 steps wins, but only because it was built for exactly 21
    steps. Given a longer vote, it can still read only the last 21 steps. To
    read more steps it needs a bigger model, which has to be trained again.

    The RNN has the same 321 parameters for any length of vote. The next table
    shows both models trained on votes of 21 steps, then tested on longer
    votes they never saw in training.

    | Model | Votes of 21 | Votes of 41 | Votes of 101 |
    |---|---|---|---|
    | MLP on the last 21 steps | 100% | 76.3% | 65.2% |
    | RNN with a hidden state of 16 | 98.9% | 93.0% | 77.2% |

    On the longer votes, the window model scores as well as any rule that
    reads only the last 21 steps can score. The RNN beats it at every longer
    length, and does so for each of five random seeds. The RNN learned to keep
    a running count of the vote in its hidden state. That count includes steps
    from before the last 21.

    The 40_bptt chapter shows what training through many steps does to the
    gradient. The 41_gated_rnns chapter shows how gates let a network keep a
    value for even longer.
    """,
)

RNN3_SOURCE = source_file(
    '''
    # rnn3.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    votes = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    last_two = votes[:, -2:]    # tensor([[1., 1.]])


    # TODO: Finish forward. x has shape (batch, steps). Take only the last
    #   self.window steps of each sequence and pass them through self.hidden,
    #   a ReLU and self.out. Return one score per sequence, with shape
    #   (batch,).
    class WindowModel(nn.Module):
        def __init__(
            self,
            window: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.window = window
            self.hidden = nn.Linear(window, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. x has shape (batch, steps), but self.rnn expects
    #   (batch, steps, 1), with one feature per step. Add that last dimension,
    #   run self.rnn, and pass its last hidden state through self.out. Return
    #   one score per sequence, with shape (batch,).
    class Reader(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

RNN3_SOLUTION = source_file(
    '''
    # rnn3.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    votes = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    last_two = votes[:, -2:]


    class WindowModel(nn.Module):
        def __init__(
            self,
            window: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.window = window
            self.hidden = nn.Linear(window, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> Tensor:
            recent = x[:, -self.window :]

            return self.out(torch.relu(self.hidden(recent))).squeeze(1)


    class Reader(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        # The last state has shape (1, batch, hidden), with one row because there is one layer.
        def forward(self, x: Tensor) -> Tensor:
            _, last = self.rnn(x.unsqueeze(2))

            return self.out(last[0]).squeeze(1)
    ''',
)


class ReferenceWindowModel(nn.Module):
    def __init__(
        self,
        window: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.window = window
        self.hidden = nn.Linear(window, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(torch.relu(self.hidden(x[:, -self.window :]))).squeeze(1)


class ReferenceReader(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.rnn(x.unsqueeze(2))[1][0]).squeeze(1)


def seeded_model(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def train_on_votes(
    model: nn.Module,
    steps: int,
    data_seed: int = 1,
) -> nn.Module:
    """The lesson's runs are 500 steps of this, and the check runs 300."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(data_seed)

    for _ in range(steps):
        x, y = vote_sequences(64, 21, generator)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def vote_accuracy(
    model: nn.Module,
    steps: int,
) -> float:
    x, y = vote_sequences(2000, steps, torch.Generator().manual_seed(99))

    with torch.no_grad():
        return ((model(x) > 0).float() == y).float().mean().item()


def check_window_forward(found: type) -> None:
    model = expect_module("WindowModel(5, 16)", seeded_model(found, 5, 16))
    x, _ = vote_sequences(4, 9, torch.Generator().manual_seed(431))

    with torch.no_grad():
        expected = seeded_model(ReferenceWindowModel, 5, 16)(x)

    expect_tensor(
        call="WindowModel(5, 16) on 4 votes of 9 steps",
        actual=model(x),
        expected=expected,
    )


def check_reader_forward(found: type) -> None:
    model = expect_module("Reader(16)", seeded_model(found, 16))
    x, _ = vote_sequences(4, 7, torch.Generator().manual_seed(432))

    with torch.no_grad():
        expected = seeded_model(ReferenceReader, 16)(x)

    expect_tensor(
        call="Reader(16) on 4 votes of 7 steps",
        actual=model(x),
        expected=expected,
    )


def check_reader_learns(found: type) -> None:
    model = train_on_votes(seeded_model(found, 16), 300)
    accuracy = vote_accuracy(model, 21)

    if accuracy < 0.9:
        raise Mismatch(
            f"After 300 steps of training, Reader(16) gets {accuracy:.1%} of new votes of 21 right, but it should get "
            "over 90%.",
        )


def check_rnn3(checker: Checker) -> None:
    checker.test_method(
        class_name="WindowModel",
        method="forward",
        description="scores the last few steps",
        run=check_window_forward,
    )
    checker.test_method(
        class_name="Reader",
        method="forward",
        description="scores the last hidden state",
        run=check_reader_forward,
    )
    checker.test_method(
        class_name="Reader",
        method="forward",
        description="learns who won a vote of 21",
        run=check_reader_learns,
    )


RNN3 = Exercise(
    name="rnn3",
    chapter=RNN,
    title="Why a hidden state",
    lesson=RNN3_LESSON,
    source=RNN3_SOURCE,
    solution=RNN3_SOLUTION,
    hints={
        "WindowModel.forward": "x[:, -self.window :] keeps the last steps. Pass them through self.hidden, "
        "torch.relu and self.out, then .squeeze(1).",
        "Reader.forward": "x.unsqueeze(2) has shape (batch, steps, 1). self.rnn returns two tensors, the outputs and "
        "the last state. If the last state is called last, last[0] is the state of the one layer.",
    },
    introduces=(),
    check=check_rnn3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_votes(seeds: range = range(5)) -> dict[tuple[str, int, int], float]:
    """rnn3's two tables, keyed by model, vote length and seed: the accuracy on 2,000 new votes after 500 Adam
    steps on votes of 21. Seed s starts the model with torch.manual_seed(s) and draws its training votes with seed
    s + 1, so seed 0 is train_on_votes's own run and the tables' run. The windows are MLPs of 16 hidden units. All
    seeds take about 20 seconds on 4 threads."""
    results = {}

    with pinned_threads():
        for seed in seeds:
            for window in (5, 11, 21):
                torch.manual_seed(seed)
                model = train_on_votes(ReferenceWindowModel(window, 16), 500, seed + 1)

                for length in (21, 41, 101):
                    results[(f"window {window}", length, seed)] = vote_accuracy(model, length)

            torch.manual_seed(seed)
            reader = train_on_votes(ReferenceReader(16), 500, seed + 1)

            for length in (21, 41, 101):
                results[("rnn", length, seed)] = vote_accuracy(reader, length)

    return results
