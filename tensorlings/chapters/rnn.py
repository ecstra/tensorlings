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

RNN = Chapter(
    folder="39_rnn",
    title="Recurrent networks",
    readme=source_file(
        """
        # Recurrent networks

        A recurrent network reads a sequence one step at a time and carries a
        summary of the past, its hidden state. This chapter builds one step by
        hand, matches `nn.RNN`, runs batches of sequences, and shows a task
        where the hidden state beats any fixed window.

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
    sound or a price day by day. A sequence model reads one step at a time and
    carries a summary of everything so far.

    That summary is the **hidden state**, a vector `h`. At each step one layer
    combines the step's input `x` with the previous hidden state, and a tanh
    keeps the result between -1 and 1.

    ```text
    h = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
    ```

    The first state is all zeros. The same weights run at every step, so a
    model of 10 steps and one of 1000 have the same parameters. Convolution
    shared its weights across positions, and this shares them across time. A
    model built this way is a **recurrent neural network**, an RNN.

    The two matrices are one Linear in disguise. Put `x` and `h` side by side
    with `torch.cat`. A Linear whose weight is `W_ih` and `W_hh` side by side,
    with the bias `b_ih + b_hh`, then gives the same sum.

    ```python
    torch.cat([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])    # tensor([1., 2., 3.])
    ```

    `nn.RNN(input_size, hidden_size)` keeps the four tensors as `weight_ih_l0`,
    `weight_hh_l0`, `bias_ih_l0` and `bias_hh_l0`, with l0 for layer 0. With
    `batch_first=True` it reads a tensor of shape (batch, steps, input_size).
    It returns every step's hidden state, of shape (batch, steps,
    hidden_size), and the last state, of shape (1, batch, hidden_size).
    """,
)

RNN1_SOURCE = source_file(
    '''
    # rnn1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    squashed = torch.tanh(torch.tensor([0.0, 5.0]))    # tensor([0.0000, 0.9999])


    # TODO: Return the next hidden state from input x, of shape (input_size,),
    #   and hidden state h, of shape (hidden_size,), by the lesson's formula.
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


    # TODO: Run rnn_step along xs, of shape (steps, input_size), from a hidden
    #   state of zeros. Return every step's hidden state, of shape
    #   (steps, hidden_size).
    def run_rnn(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        ...


    side_by_side = torch.cat([torch.ones(2, 1), torch.zeros(2, 2)], dim=1)    # shape (2, 3)


    # TODO: Return the weight and the bias of one Linear that, applied to x and
    #   h side by side, gives the same sum as the two matrices and two biases.
    def as_one_linear(
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    example = nn.RNN(3, 4)
    example.weight_hh_l0.shape    # torch.Size([4, 4])


    # TODO: Return run_rnn of xs, of shape (steps, input_size), with the four
    #   tensors of rnn, an nn.RNN with one layer.
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
        "rnn_step": "Write the formula as it is, with torch.tanh around the whole sum.",
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
    Training reads many sequences at once. With `batch_first=True` the input
    has shape (batch, steps, features) and the hidden state (batch, hidden).
    Each step is rnn1's formula, run on every row together.

    PyTorch's default, `batch_first=False`, puts the steps first, as (steps,
    batch, features). Plenty of code uses it, so check which one a model
    expects. `x.transpose(0, 1)` swaps the two.

    `num_layers=2` stacks a second RNN that reads the first one's hidden
    states as its input. The outputs are the top layer's states. The last
    state holds one row per layer.

    | Setting | Input | Outputs | Last state |
    |---|---|---|---|
    | `batch_first=True` | (batch, steps, features) | (batch, steps, hidden) | (layers, batch, hidden) |
    | `batch_first=False` | (steps, batch, features) | (steps, batch, hidden) | (layers, batch, hidden) |

    The last state never puts the batch first, whatever the setting.

    The sequences of one batch often differ in length. The short ones are
    padded with zeros at the end, up to the longest. Then each sequence's
    answer is its output at its own last real step, which the last column
    isn't for a short one. Indexing with two tensors picks one row and one
    step for each pair.

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


    # TODO: Return every step's hidden state for xs, of shape
    #   (batch, steps, input_size), from hidden states of zeros, as a tensor of
    #   shape (batch, steps, hidden_size). Run all the rows of each step at once.
    def run_rnn_batch(
        xs: Tensor,
        W_ih: Tensor,
        W_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the shapes of the outputs and the last state that nn.RNN gives
    #   for these sizes, as two tuples, following the lesson's table.
    def rnn_shapes(
        batch: int,
        steps: int,
        features: int,
        hidden: int,
        layers: int,
        batch_first: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        ...


    # TODO: Return each sequence's output at its last real step. outputs has
    #   shape (batch, steps, hidden) and lengths holds each sequence's length.
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
    Here is a task that needs the whole past. Each sequence is a vote of 21
    steps, each +1 or -1, and the label says whether the +1s won. Every step
    counts, so nothing short of the whole sequence settles it.

    A fixed window model reads only the last few steps, through a small MLP.
    An RNN reads every step and carries a running summary. Each model trained
    for 500 steps of Adam on fresh random votes.

    | Model | Accuracy on votes of 21 |
    |---|---|
    | MLP on the last 5 steps | 67.5% |
    | MLP on the last 11 steps | 77.9% |
    | MLP on all 21 steps | 100% |
    | RNN with a hidden state of 16 | 98.9% |

    The window of 21 wins, but only because it was built for exactly 21 steps.
    Given a vote of 41 it can still only read the last 21, and it scores
    76.3%. To read 41 it needs a bigger model, trained again.

    The RNN has the same 321 parameters for any length. Trained on 21 steps, it
    scored 93.0% on votes of 41 and 77.2% on votes of 101, lengths it never
    saw. It learned to keep a running count in its hidden state, which carries
    what matters forward whatever the length.

    40_bptt shows what training through many steps does to the gradient, and
    41_gated_rnns how gates hold a value longer still.
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


    # TODO: Finish forward. x has shape (batch, steps). Read only the last
    #   self.window steps, through self.hidden, a ReLU and self.out, and return
    #   one score per sequence, shape (batch,).
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


    # TODO: Finish forward. x has shape (batch, steps). Give each step a feature
    #   dimension of size 1, run self.rnn, pass its last state through self.out,
    #   and return one score per sequence, shape (batch,).
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

        # The last state has shape (1, batch, hidden), one row for the one layer.
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
) -> nn.Module:
    """The lesson's runs are 500 steps of this, and the check runs 300."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)

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
        "Reader.forward": "x.unsqueeze(2) has shape (batch, steps, 1). self.rnn returns the outputs and the last "
        "state, and last[0] is the one layer's state.",
    },
    introduces=(),
    check=check_rnn3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
