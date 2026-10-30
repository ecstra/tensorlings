"""41_gated_rnns, the LSTM and the GRU built and matched with PyTorch's, and the gate as a weight computed from the input."""

from __future__ import annotations

import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    NOT_STARTED,
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file

GATED_RNNS = Chapter(
    folder="41_gated_rnns",
    title="Gated RNNs",
    readme=source_file(
        """
        # Gated RNNs

        A plain RNN rewrites its whole state at every step, and its gradient
        fades over long spans. Gated RNNs let the input decide, step by step,
        what to keep and what to write. This chapter builds the LSTM and the
        GRU, matches them with PyTorch's, measures when each one beats a plain
        RNN, and names the idea under both, a weight computed from the input.

        ## Further reading

        - [Olah, understanding LSTM networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
        - [Dive into Deep Learning, the LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)
        - [Dive into Deep Learning, the GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)
        - [Cho and colleagues, the paper that introduced the GRU](https://arxiv.org/abs/1406.1078)
        - [Greff and colleagues, LSTM: a search space odyssey](https://arxiv.org/abs/1503.04069)
        - [Jozefowicz, Zaremba and Sutskever, an empirical exploration of recurrent network architectures](https://proceedings.mlr.press/v37/jozefowicz15.pdf)
        - [Dauphin and colleagues, gated linear units](https://arxiv.org/abs/1612.08083)
        """,
    ),
)


def seeded_layer(
    make: type,
    input_size: int,
    hidden_size: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(input_size, hidden_size, batch_first=True)


def half_tensor(
    seed: int,
    *shape: int,
) -> Tensor:
    """Small whole numbers halved, so the gates sit away from where the sigmoid is flat."""
    return small_tensor(seed, *shape) / 2


# ---------------------------------------------------------------------------
# lstm1
# ---------------------------------------------------------------------------

LSTM1_LESSON = prose(
    """
    An RNN rewrites its whole hidden state at every step, through one tanh.
    bptt3 showed what that does to a gradient over a long span. The **LSTM**,
    from Hochreiter and Schmidhuber in 1997, keeps a second state, the cell,
    which a step changes only where gates let it.

    Each step takes the input `x` and the last hidden state `h`, and computes
    four things, each with its own weights on `x` and `h`.

    ```text
    i = sigmoid(x W_i + h U_i + b_i)    how much new content goes in
    f = sigmoid(x W_f + h U_f + b_f)    how much of the old cell stays
    g = tanh(x W_g + h U_g + b_g)       the new content
    o = sigmoid(x W_o + h U_o + b_o)    how much of the cell shows
    c = f * c + i * g
    h = o * tanh(c)
    ```

    The cell's update is a sum with no squash. Where `f` is near 1 and `i`
    near 0, the cell carries its value forward unchanged, and its gradient
    flows back unchanged too. The forget gate `f` came later, from Gers,
    Schmidhuber and Cummins in 2000. Without it, a cell could only add.

    `nn.LSTM` stacks the four sets of weights in one matrix, in the order i,
    f, g, o. `weight_ih_l0` has shape (4 * hidden, input), `weight_hh_l0` has
    shape (4 * hidden, hidden), and both biases, `bias_ih_l0` and
    `bias_hh_l0`, are added. `.chunk(4, dim=1)` cuts the stacked result back
    into four parts.

    ```python
    torch.arange(8.0).view(1, 8).chunk(4, dim=1)
    # (tensor([[0., 1.]]), tensor([[2., 3.]]), tensor([[4., 5.]]), tensor([[6., 7.]]))
    ```

    `nn.LSTM(input, hidden, batch_first=True)` takes `x` of shape (batch,
    steps, input). It returns the hidden state at every step first. Then
    comes the pair of the last `h` and `c`, each with an extra first dimension
    of size 1. Both states start at zeros.
    """,
)

LSTM1_SOURCE = source_file(
    '''
    # lstm1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    parts = torch.arange(8.0).view(1, 8).chunk(4, dim=1)    # four pieces of shape (1, 2)


    # TODO: Return the new h and c after one LSTM step, from x of shape (batch,
    #   input) and h and c of shape (batch, hidden), with weights stacked in
    #   nn.LSTM's order, i, f, g, o.
    def lstm_cell(
        x: Tensor,
        h: Tensor,
        c: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ...


    stacked = nn.LSTM(3, 4).weight_ih_l0.shape    # torch.Size([16, 3])


    # TODO: Return the hidden state at every step of x, of shape (batch, steps,
    #   input), as one tensor of shape (batch, steps, hidden). Run lstm_cell
    #   with lstm's own weights, starting h and c at zeros.
    def run_lstm(
        x: Tensor,
        lstm: nn.LSTM,
    ) -> Tensor:
        ...


    count = sum(p.numel() for p in nn.RNN(1, 16).parameters())    # 304


    # TODO: Return how many numbers nn.LSTM(input_size, hidden_size) learns,
    #   from its weight shapes, without building one.
    def lstm_size(
        input_size: int,
        hidden_size: int,
    ) -> int:
        ...
    ''',
)

LSTM1_SOLUTION = source_file(
    '''
    # lstm1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    parts = torch.arange(8.0).view(1, 8).chunk(4, dim=1)


    def lstm_cell(
        x: Tensor,
        h: Tensor,
        c: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        i, f, g, o = (x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        return torch.sigmoid(o) * torch.tanh(c), c


    stacked = nn.LSTM(3, 4).weight_ih_l0.shape


    def run_lstm(
        x: Tensor,
        lstm: nn.LSTM,
    ) -> Tensor:
        hidden_size = lstm.hidden_size
        h = torch.zeros(len(x), hidden_size)
        c = torch.zeros(len(x), hidden_size)
        outputs = []

        for t in range(x.shape[1]):
            h, c = lstm_cell(x[:, t], h, c, lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


    count = sum(p.numel() for p in nn.RNN(1, 16).parameters())


    def lstm_size(
        input_size: int,
        hidden_size: int,
    ) -> int:
        return 4 * (hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size)
    ''',
)


def reference_lstm_cell(
    x: Tensor,
    h: Tensor,
    c: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    i, f, g, o = (x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh).chunk(4, dim=1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

    return torch.sigmoid(o) * torch.tanh(c), c


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        expect_value(
            call=call,
            actual=actual,
            expected=expected,
        )
        return

    expect_tensor(
        call=f"the new h of {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_tensor(
        call=f"the new c of {call}",
        actual=actual[1],
        expected=expected[1],
    )


def lstm_weights(seed: int) -> tuple[Described, ...]:
    lstm = seeded_layer(nn.LSTM, 3, 4, seed)

    return tuple(
        Described(parameter.detach(), f"{name} of shape {tuple(parameter.shape)}")
        for name, parameter in lstm.named_parameters()
    )


def layer_outputs(
    x: Tensor,
    layer: nn.Module,
) -> Tensor:
    with torch.no_grad():
        return layer(x)[0]


def check_lstm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="lstm_cell",
        description="gives the new h and c after one step",
        inputs=(
            (half_tensor(410, 2, 3), half_tensor(411, 2, 4), half_tensor(412, 2, 4), *lstm_weights(413)),
            (half_tensor(414, 1, 3), torch.zeros(1, 4), torch.zeros(1, 4), *lstm_weights(415)),
        ),
        reference=reference_lstm_cell,
        expect=expect_pair,
    )
    check_on_inputs(
        checker=checker,
        function="run_lstm",
        description="matches nn.LSTM's outputs at every step",
        inputs=(
            (half_tensor(416, 2, 5, 3), Described(seeded_layer(nn.LSTM, 3, 4, 417), "nn.LSTM(3, 4, batch_first=True)")),
            (half_tensor(418, 3, 4, 2), Described(seeded_layer(nn.LSTM, 2, 6, 419), "nn.LSTM(2, 6, batch_first=True)")),
        ),
        reference=layer_outputs,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="lstm_size",
        description="counts nn.LSTM's parameters",
        inputs=(
            (3, 4),
            (10, 32),
            (1, 16),
        ),
        reference=lambda input_size, hidden_size: sum(parameter.numel() for parameter in nn.LSTM(input_size, hidden_size).parameters()),
        expect=expect_value,
    )


LSTM1 = Exercise(
    name="lstm1",
    chapter=GATED_RNNS,
    title="The LSTM cell",
    lesson=LSTM1_LESSON,
    source=LSTM1_SOURCE,
    solution=LSTM1_SOLUTION,
    hints={
        "lstm_cell": "Add x @ weight_ih.T, bias_ih, h @ weight_hh.T and bias_hh, then .chunk(4, dim=1) into i, f, "
        "g and o. Put sigmoid on i, f and o and tanh on g, then follow the lesson's last two lines.",
        "run_lstm": "Start h and c as torch.zeros(len(x), lstm.hidden_size). For each step t, call lstm_cell on "
        "x[:, t] with lstm.weight_ih_l0 and the rest, then torch.stack the h's along dim=1.",
        "lstm_size": "Each of the 4 parts has a weight on the input, a weight on h and two biases of hidden_size.",
    },
    introduces=(),
    check=check_lstm1,
)


# ---------------------------------------------------------------------------
# lstm2
# ---------------------------------------------------------------------------

RECURRENT_LAYERS: dict[str, type] = {
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}


def marked_values(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """derive1's marked value stretched out. Each step holds a value from -1 to 1 and a mark, one of the
    first 5 steps is marked, and the answer is its value."""
    values = 2 * torch.rand(count, steps, generator=generator) - 1
    where = torch.randint(0, 5, (count,), generator=generator)
    rows = torch.arange(count)
    marks = torch.zeros(count, steps)
    marks[rows, where] = 1.0

    return torch.stack([values, marks], dim=2), values[rows, where]


def reference_set_forget_bias_(
    lstm: nn.LSTM,
    value: float,
) -> None:
    hidden_size = lstm.hidden_size

    with torch.no_grad():
        lstm.bias_ih_l0[hidden_size : 2 * hidden_size] = value
        lstm.bias_hh_l0[hidden_size : 2 * hidden_size] = 0.0


class ReferenceRecaller(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs, _ = self.layer(x)

        return self.head(outputs[:, -1]).squeeze(1)


def seeded_recaller(
    kind: str,
    hidden_size: int,
    seed: int,
    forget_bias: float | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    model = ReferenceRecaller(RECURRENT_LAYERS[kind](2, hidden_size, batch_first=True), hidden_size)

    if forget_bias is not None:
        reference_set_forget_bias_(model.layer, forget_bias)

    return model


def recall_error(
    model: nn.Module,
    steps: int,
    seed: int,
    train_steps: int,
) -> float:
    """The run the lessons' tables come from, Adam at 0.01 on batches of 64, then the mean squared error on
    1,000 new sequences."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(train_steps):
        x, y = marked_values(64, steps, generator)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x, y = marked_values(1000, steps, torch.Generator().manual_seed(77))

    with torch.no_grad():
        return F.mse_loss(model(x), y).item()


LSTM2_LESSON = prose(
    """
    bptt3 showed a plain RNN's gradient fading over a long span, and lstm1
    built a cell whose update has no squash. Here is what that buys, measured,
    and the catch that comes with it.

    The task is derive1's marked value, stretched out. Each step holds a value
    from -1 to 1 and a mark of 0 or 1. One of the first 5 steps is marked, and
    the model gives back its value at the very end. Always guessing 0 scores a
    mean squared error of 0.333.

    Every model has 16 hidden units and trains for 1,500 Adam steps, three
    times with different seeds. A run counts as solved below an error of 0.01.

    | Model | Length 10 | Length 20 | Length 40 |
    |---|---|---|---|
    | RNN | 3 of 3 | 0 of 3 | 0 of 3 |
    | LSTM, forget bias 0 | 3 of 3 | 0 of 3 | 0 of 3 |
    | LSTM, forget bias 1 | 3 of 3 | 3 of 3 | 0 of 3 |
    | LSTM, forget bias 3 | 3 of 3 | 3 of 3 | 3 of 3 |
    | GRU | 3 of 3 | 3 of 3 | 1 of 3 |

    At length 10 every model solves it. A plain RNN is fine for short gaps, and
    for answers that build up over the whole sequence, like rnn3's vote.

    The LSTM as PyTorch builds it fails exactly where the RNN does. PyTorch
    starts every bias near 0, so the forget gate starts near sigmoid(0), 0.5.
    The cell then halves at every step, and so does its gradient.

    | Forget bias | Forget gate at the start | Left after 20 steps | Left after 40 steps |
    |---|---|---|---|
    | 0 | 0.50 | 1e-6 | 9e-13 |
    | 1 | 0.73 | 0.0019 | 3.6e-6 |
    | 3 | 0.95 | 0.38 | 0.14 |

    Start the forget gate's bias higher, and the cell carries the value and
    its gradient across the gap from the first step of training. Training can
    still close the gate later. Gers, Schmidhuber and Cummins proposed this in
    2000. In 2015 Jozefowicz, Zaremba and Sutskever found the GRU beating the
    LSTM on nearly every task without it. With a bias of 1, the LSTM nearly
    matched the GRU.

    So the LSTM's edge is real, and it lives in two places, a cell with no
    squash and a forget gate that starts open.
    """,
)

LSTM2_SOURCE = source_file(
    '''
    # lstm2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    gate = torch.sigmoid(torch.tensor(1.0))    # tensor(0.7311), a forget gate's start at a bias of 1


    # TODO: Return the share of a cell's value left after steps steps, when its
    #   forget gate stays at its start, the sigmoid of bias, and nothing new
    #   goes in. Return a plain number.
    def forget_fraction(
        bias: float,
        steps: int,
    ) -> float:
        ...


    second_quarter = torch.arange(8)[2:4]    # tensor([2, 3]), the second quarter of 8 rows


    # TODO: Set the forget gate's part of lstm.bias_ih_l0 to value and its part
    #   of lstm.bias_hh_l0 to 0, in place, without recording for autograd. In
    #   the order i, f, g, o, the forget gate's part is the second quarter.
    def set_forget_bias_(
        lstm: nn.LSTM,
        value: float,
    ) -> None:
        ...


    last = torch.zeros(2, 5, 16)[:, -1]    # shape (2, 16), each sequence's last step


    # TODO: Finish forward. x has shape (batch, steps, 2). Run self.layer, then
    #   self.head on the last step's output, and return one guess per sequence,
    #   of shape (batch,).
    class Recaller(nn.Module):
        def __init__(
            self,
            layer: nn.Module,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.layer = layer
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

LSTM2_SOLUTION = source_file(
    '''
    # lstm2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    gate = torch.sigmoid(torch.tensor(1.0))


    def forget_fraction(
        bias: float,
        steps: int,
    ) -> float:
        return torch.sigmoid(torch.tensor(bias)).item() ** steps


    second_quarter = torch.arange(8)[2:4]


    def set_forget_bias_(
        lstm: nn.LSTM,
        value: float,
    ) -> None:
        hidden_size = lstm.hidden_size

        with torch.no_grad():
            lstm.bias_ih_l0[hidden_size : 2 * hidden_size] = value
            lstm.bias_hh_l0[hidden_size : 2 * hidden_size] = 0.0


    last = torch.zeros(2, 5, 16)[:, -1]


    class Recaller(nn.Module):
        def __init__(
            self,
            layer: nn.Module,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.layer = layer
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Tensor) -> Tensor:
            outputs, _ = self.layer(x)

            return self.head(outputs[:, -1]).squeeze(1)
    ''',
)


def check_set_forget_bias(found: object) -> None:
    for hidden_size, value in ((4, 3.0), (6, 1.0)):
        call = f"set_forget_bias_ on nn.LSTM(2, {hidden_size}) with {value}"
        lstm = seeded_layer(nn.LSTM, 2, hidden_size, 450)
        expected = seeded_layer(nn.LSTM, 2, hidden_size, 450)
        reference_set_forget_bias_(expected, value)
        before = lstm.bias_ih_l0.detach().clone()
        found(lstm, value)

        if torch.equal(lstm.bias_ih_l0.detach(), before):
            raise Mismatch(f"After {call}, bias_ih_l0 is unchanged, so set_forget_bias_ is {NOT_STARTED}.")

        for name in ("bias_ih_l0", "bias_hh_l0"):
            expect_tensor(
                call=f"{name} after {call}",
                actual=getattr(lstm, name).detach(),
                expected=getattr(expected, name).detach(),
            )


def check_recaller_forward(found: type) -> None:
    torch.manual_seed(451)
    model = found(nn.LSTM(2, 4, batch_first=True), 4)
    torch.manual_seed(451)
    reference = ReferenceRecaller(nn.LSTM(2, 4, batch_first=True), 4)
    x, _ = marked_values(3, 6, torch.Generator().manual_seed(452))

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="Recaller(nn.LSTM(2, 4, batch_first=True), 4) on 3 sequences of 6 steps",
        actual=model(x),
        expected=expected,
    )


def check_recall_across_40(found: object) -> None:
    model = seeded_recaller("lstm", 16, 0)
    found(model.layer, 3.0)
    error = recall_error(
        model=model,
        steps=40,
        seed=0,
        train_steps=500,
    )

    if error > 0.05:
        raise Mismatch(
            f"An LSTM given set_forget_bias_(lstm, 3.0) and trained for 500 steps scores an error of {error:.3f} "
            "at length 40, but it should be below 0.05.",
        )


def check_lstm2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="forget_fraction",
        description="gives the share left after many steps",
        inputs=(
            (2.0, 10),
            (-1.0, 3),
            (0.5, 25),
        ),
        reference=lambda bias, steps: torch.sigmoid(torch.tensor(bias)).item() ** steps,
        expect=expect_close,
    )
    checker.test(
        function="set_forget_bias_",
        description="sets the forget gate's bias and nothing else",
        run=check_set_forget_bias,
    )
    checker.test_method(
        class_name="Recaller",
        method="forward",
        description="gives one guess per sequence from the last step",
        run=check_recaller_forward,
    )
    checker.test(
        function="set_forget_bias_",
        description="lets an LSTM recall a value 40 steps back",
        run=check_recall_across_40,
    )


LSTM2 = Exercise(
    name="lstm2",
    chapter=GATED_RNNS,
    title="Why the LSTM, and its forget bias",
    lesson=LSTM2_LESSON,
    source=LSTM2_SOURCE,
    solution=LSTM2_SOLUTION,
    hints={
        "forget_fraction": "torch.sigmoid(torch.tensor(bias)).item() is the gate. Raise it to the power steps.",
        "set_forget_bias_": "With h = lstm.hidden_size, the forget part is [h : 2 * h]. Inside torch.no_grad(), "
        "assign value to that slice of lstm.bias_ih_l0 and 0.0 to that slice of lstm.bias_hh_l0.",
        "Recaller.forward": "self.layer(x) returns the outputs at every step first. Take outputs[:, -1], run "
        "self.head, then .squeeze(1).",
    },
    introduces=(),
    check=check_lstm2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# gru1
# ---------------------------------------------------------------------------

GRU1_LESSON = prose(
    """
    The **GRU**, from Cho and colleagues in 2014, keeps one state and two
    gates. Each step computes this.

    ```text
    r = sigmoid(x W_r + b_r + h U_r + c_r)        how much of h the new content reads
    z = sigmoid(x W_z + b_z + h U_z + c_z)        how much of h stays
    n = tanh(x W_n + b_n + r * (h U_n + c_n))     the new content
    h = (1 - z) * n + z * h
    ```

    The last line is derive1's gated memory, with `1 - z` as the gate that
    blends new content in. The reset gate `r` lets a step ignore the old state
    when it writes new content.

    `nn.GRU` stacks its weights in the order r, z, n. One detail needs care.
    `r` multiplies the hidden part of `n` with its bias `c_n` included, so the
    parts from `x` and from `h` stay apart until then. Cut each into three
    with `.chunk(3, dim=1)`.

    It has three sets of weights where the LSTM has four. With an input of 1
    and a hidden size of 16, that is 912 numbers against the LSTM's 1,216.
    Chung and colleagues compared the two in 2014 on music and speech, and
    neither won across the board.
    """,
)

GRU1_SOURCE = source_file(
    '''
    # gru1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    z = torch.tensor([0.9])
    kept = (1 - z) * torch.tensor([5.0]) + z * torch.tensor([2.0])    # tensor([2.3000]), mostly the old 2


    # TODO: Return the new h after one GRU step, from x of shape (batch, input)
    #   and h of shape (batch, hidden), with weights stacked in nn.GRU's order,
    #   r, z, n.
    def gru_cell(
        x: Tensor,
        h: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the hidden state at every step of x, of shape (batch, steps,
    #   input), as one tensor of shape (batch, steps, hidden). Run gru_cell with
    #   gru's own weights, starting h at zeros.
    def run_gru(
        x: Tensor,
        gru: nn.GRU,
    ) -> Tensor:
        ...
    ''',
)

GRU1_SOLUTION = source_file(
    '''
    # gru1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    z = torch.tensor([0.9])
    kept = (1 - z) * torch.tensor([5.0]) + z * torch.tensor([2.0])


    def gru_cell(
        x: Tensor,
        h: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        x_r, x_z, x_n = (x @ weight_ih.T + bias_ih).chunk(3, dim=1)
        h_r, h_z, h_n = (h @ weight_hh.T + bias_hh).chunk(3, dim=1)
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)

        return (1 - z) * n + z * h


    def run_gru(
        x: Tensor,
        gru: nn.GRU,
    ) -> Tensor:
        h = torch.zeros(len(x), gru.hidden_size)
        outputs = []

        for t in range(x.shape[1]):
            h = gru_cell(x[:, t], h, gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)
            outputs.append(h)

        return torch.stack(outputs, dim=1)
    ''',
)


def reference_gru_cell(
    x: Tensor,
    h: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
) -> Tensor:
    x_r, x_z, x_n = (x @ weight_ih.T + bias_ih).chunk(3, dim=1)
    h_r, h_z, h_n = (h @ weight_hh.T + bias_hh).chunk(3, dim=1)
    r = torch.sigmoid(x_r + h_r)
    z = torch.sigmoid(x_z + h_z)

    return (1 - z) * torch.tanh(x_n + r * h_n) + z * h


def gru_weights(seed: int) -> tuple[Described, ...]:
    gru = seeded_layer(nn.GRU, 3, 4, seed)

    return tuple(
        Described(parameter.detach(), f"{name} of shape {tuple(parameter.shape)}")
        for name, parameter in gru.named_parameters()
    )


def check_gru1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gru_cell",
        description="gives the new h after one step",
        inputs=(
            (half_tensor(420, 2, 3), half_tensor(421, 2, 4), *gru_weights(422)),
            (half_tensor(423, 1, 3), torch.zeros(1, 4), *gru_weights(424)),
        ),
        reference=reference_gru_cell,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="run_gru",
        description="matches nn.GRU's outputs at every step",
        inputs=(
            (half_tensor(425, 2, 5, 3), Described(seeded_layer(nn.GRU, 3, 4, 426), "nn.GRU(3, 4, batch_first=True)")),
            (half_tensor(427, 3, 4, 2), Described(seeded_layer(nn.GRU, 2, 6, 428), "nn.GRU(2, 6, batch_first=True)")),
        ),
        reference=layer_outputs,
        expect=expect_tensor,
    )


GRU1 = Exercise(
    name="gru1",
    chapter=GATED_RNNS,
    title="The GRU",
    lesson=GRU1_LESSON,
    source=GRU1_SOURCE,
    solution=GRU1_SOLUTION,
    hints={
        "gru_cell": "Chunk x @ weight_ih.T + bias_ih and h @ weight_hh.T + bias_hh into three parts each. Add the "
        "r parts and the z parts before their sigmoids, and multiply only the h part of n by r.",
        "run_gru": "Start h as torch.zeros(len(x), gru.hidden_size). For each step t, call gru_cell on x[:, t] "
        "with gru.weight_ih_l0 and the rest, then torch.stack the h's along dim=1.",
    },
    introduces=(),
    check=check_gru1,
)


# ---------------------------------------------------------------------------
# gating1
# ---------------------------------------------------------------------------

GATING1_LESSON = prose(
    """
    Every weight so far was fixed by training. Once trained, a Linear
    multiplies every input by the same numbers. A gate is the first weight
    here that is computed from the input itself.

    ```text
    gate = sigmoid(x W + b)
    out = gate * signal
    ```

    Each entry of `signal` gets scaled by a number from 0 to 1 that this input
    chose. Another input chooses another scale. derive1's gated memory, the
    LSTM's i, f and o, and the GRU's r and z are all this one move.

    A gate can do what no fixed weight can, and multiply two inputs. "Pass the
    value through when the switch is on" is the value times the switch. A
    Linear can only add the two, so it can't learn that, while a gate on the
    switch can.

    The gated linear unit of Dauphin and colleagues, from 2016, makes a whole
    layer of gates. One Linear gives twice the width, and its second half
    gates its first, `a * sigmoid(b)`. `F.glu` does this split for you.

    ```python
    F.glu(torch.tensor([1.0, 2.0, 0.0, -3.0]), dim=-1)    # tensor([0.5000, 0.0949])
    ```

    Here `a` is `[1, 2]` and `b` is `[0, -3]`, so the gates are 0.5 and
    0.047. Section 7 comes back to weights computed from the input several
    times.
    """,
)

GATING1_SOURCE = source_file(
    '''
    # gating1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    scales = torch.sigmoid(torch.tensor([0.0, 2.0]))    # tensor([0.5000, 0.8808])


    # TODO: Return the gate that x gives, sigmoid of a Linear with weight and
    #   bias, one number from 0 to 1 per output.
    def gate(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    scaled = scales * torch.tensor([4.0, 4.0])    # tensor([2.0000, 3.5232])


    # TODO: Return signal scaled by the gate that x gives with weight and bias.
    def gated(
        signal: Tensor,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    halves = torch.tensor([1.0, 2.0, 0.0, -3.0]).chunk(2, dim=-1)    # (tensor([1., 2.]), tensor([ 0., -3.]))


    # TODO: Return the gated linear unit of x without F.glu. A Linear with
    #   weight and bias gives twice the width, and its second half, through a
    #   sigmoid, scales its first half.
    def glu(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...
    ''',
)

GATING1_SOLUTION = source_file(
    '''
    # gating1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    scales = torch.sigmoid(torch.tensor([0.0, 2.0]))


    def gate(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        return torch.sigmoid(x @ weight.T + bias)


    scaled = scales * torch.tensor([4.0, 4.0])


    def gated(
        signal: Tensor,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        return gate(x, weight, bias) * signal


    halves = torch.tensor([1.0, 2.0, 0.0, -3.0]).chunk(2, dim=-1)


    def glu(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        a, b = (x @ weight.T + bias).chunk(2, dim=-1)

        return a * torch.sigmoid(b)
    ''',
)


def check_gating1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gate",
        description="gives sigmoid of a Linear",
        inputs=(
            (half_tensor(430, 2, 3), half_tensor(431, 4, 3), half_tensor(432, 4)),
            (half_tensor(433, 1, 2), half_tensor(434, 2, 2), half_tensor(435, 2)),
        ),
        reference=lambda x, weight, bias: torch.sigmoid(F.linear(x, weight, bias)),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="gated",
        description="scales the signal by the gate x gives",
        inputs=(
            (small_tensor(436, 2, 4), half_tensor(437, 2, 3), half_tensor(438, 4, 3), half_tensor(439, 4)),
            (small_tensor(440, 3, 2), half_tensor(441, 3, 5), half_tensor(442, 2, 5), half_tensor(443, 2)),
        ),
        reference=lambda signal, x, weight, bias: torch.sigmoid(F.linear(x, weight, bias)) * signal,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="glu",
        description="matches F.glu after the Linear",
        inputs=(
            (half_tensor(444, 2, 3), half_tensor(445, 8, 3), half_tensor(446, 8)),
            (half_tensor(447, 3, 4), half_tensor(448, 6, 4), half_tensor(449, 6)),
        ),
        reference=lambda x, weight, bias: F.glu(F.linear(x, weight, bias), dim=-1),
        expect=expect_tensor,
    )


GATING1 = Exercise(
    name="gating1",
    chapter=GATED_RNNS,
    title="A weight computed from the input",
    lesson=GATING1_LESSON,
    source=GATING1_SOURCE,
    solution=GATING1_SOLUTION,
    hints={
        "gate": "torch.sigmoid of x @ weight.T + bias.",
        "gated": "Multiply signal by gate(x, weight, bias).",
        "glu": "Split x @ weight.T + bias into a and b with .chunk(2, dim=-1), then return a * torch.sigmoid(b).",
    },
    introduces=(),
    check=check_gating1,
)


# ---------------------------------------------------------------------------
# lstm3
# ---------------------------------------------------------------------------

WEIGHT_SETS: dict[str, int] = {
    "rnn": 1,
    "gru": 3,
    "lstm": 4,
}

LSTM3_LESSON = prose(
    """
    Which recurrent layer to reach for, from what this chapter measured.

    First, is the LSTM winning only because it has four times the weights?
    Give each model about 1,150 weights instead, which buys the RNN 32 hidden
    units, the GRU 18 and the LSTM 15. Here is lstm2's task again, three seeds
    each.

    | Model | Weights | Length 20 | Length 40 |
    |---|---|---|---|
    | RNN, 32 units | 1,152 | 0 of 3 | 0 of 3 |
    | GRU, 18 units | 1,188 | 3 of 3 | 1 of 3 |
    | LSTM, 15 units, forget bias 1 | 1,140 | 3 of 3 | 0 of 3 |
    | LSTM, 15 units, forget bias 3 | 1,140 | 3 of 3 | 3 of 3 |

    More units don't save the RNN. Its trouble is the squash at every step,
    not its size. Speed goes the other way. A training step at length 40 took
    about 1 ms for the RNN here, and about 2.5 ms or more for the gated ones.

    - A plain RNN suits short gaps, and answers that build up over the whole sequence. It is the smallest and the fastest.
    - A GRU is a good first choice with nothing to tune, at three quarters of an LSTM's weights.
    - An LSTM reached the longest gaps here, once its forget gate started open. Set that bias every time you make one.

    All three run one step after another, which a GPU can't spread out.
    44_causal_conv and 45_state_space train on every step at once, and
    Section 7's models replaced RNNs for most text. An RNN still fits where
    its fixed size state matters, like reading a stream one step at a time on
    a small device.
    """,
)

SET_FORGET_BIAS = '''
# set_forget_bias_ from lstm2.
def set_forget_bias_(
    lstm: nn.LSTM,
    value: float,
) -> None:
    hidden_size = lstm.hidden_size

    with torch.no_grad():
        lstm.bias_ih_l0[hidden_size : 2 * hidden_size] = value
        lstm.bias_hh_l0[hidden_size : 2 * hidden_size] = 0.0
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_SET_FORGET_BIAS = textwrap.indent(SET_FORGET_BIAS.strip("\n"), "    ")

LSTM3_SOURCE = source_file(
    '''
    # lstm3.py

    import torch
    import torch.nn as nn


''' + INDENTED_SET_FORGET_BIAS + '''


    # How many sets of weights each layer learns, one per gate or content.
    WEIGHT_SETS = {"rnn": 1, "gru": 3, "lstm": 4}

    sets = WEIGHT_SETS["gru"]    # 3, the GRU's r, z and n


    # TODO: Return how many numbers a one layer nn.RNN, nn.GRU or nn.LSTM
    #   learns, with kind "rnn", "gru" or "lstm", from its sets of weights,
    #   without building one.
    def recurrent_size(
        kind: str,
        input_size: int,
        hidden_size: int,
    ) -> int:
        ...


    fits = [h for h in range(1, 6) if h * h <= 10]    # [1, 2, 3], so 3 is the largest that fits


    # TODO: Return the largest hidden size whose recurrent_size, for kind and
    #   input_size, is at most budget.
    def hidden_for_budget(
        kind: str,
        input_size: int,
        budget: int,
    ) -> int:
        ...


    layer = nn.GRU(2, 18, batch_first=True)    # batch first, like every layer in this chapter


    # TODO: Return a one layer nn.RNN, nn.GRU or nn.LSTM for kind, with
    #   batch_first=True. Start an LSTM's forget bias at forget_bias with
    #   set_forget_bias_.
    def make_recurrent(
        kind: str,
        input_size: int,
        hidden_size: int,
        forget_bias: float = 1.0,
    ) -> nn.Module:
        ...
    ''',
)

LSTM3_SOLUTION = source_file(
    '''
    # lstm3.py, a solution for comparison

    import torch
    import torch.nn as nn


''' + INDENTED_SET_FORGET_BIAS + '''


    WEIGHT_SETS = {"rnn": 1, "gru": 3, "lstm": 4}

    sets = WEIGHT_SETS["gru"]


    # Each set has a weight on the input, a weight on h and two biases.
    def recurrent_size(
        kind: str,
        input_size: int,
        hidden_size: int,
    ) -> int:
        return WEIGHT_SETS[kind] * (hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size)


    fits = [h for h in range(1, 6) if h * h <= 10]


    def hidden_for_budget(
        kind: str,
        input_size: int,
        budget: int,
    ) -> int:
        hidden_size = 1

        while recurrent_size(kind, input_size, hidden_size + 1) <= budget:
            hidden_size += 1

        return hidden_size


    layer = nn.GRU(2, 18, batch_first=True)


    def make_recurrent(
        kind: str,
        input_size: int,
        hidden_size: int,
        forget_bias: float = 1.0,
    ) -> nn.Module:
        layer = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](input_size, hidden_size, batch_first=True)

        if kind == "lstm":
            set_forget_bias_(layer, forget_bias)

        return layer
    ''',
)


def reference_hidden_for_budget(
    kind: str,
    input_size: int,
    budget: int,
) -> int:
    hidden_size = 1

    while WEIGHT_SETS[kind] * ((hidden_size + 1) * input_size + (hidden_size + 1) ** 2 + 2 * (hidden_size + 1)) <= budget:
        hidden_size += 1

    return hidden_size


def check_make_recurrent(found: object) -> None:
    for kind, input_size, hidden_size in (("rnn", 3, 5), ("gru", 2, 4), ("lstm", 2, 6)):
        call = f'make_recurrent("{kind}", {input_size}, {hidden_size}, forget_bias=2.0)'
        layer = found(kind, input_size, hidden_size, forget_bias=2.0)

        if type(layer) is not RECURRENT_LAYERS[kind]:
            raise Mismatch(f"{call} is a {type(layer).__name__}, but it should be an nn.{RECURRENT_LAYERS[kind].__name__}.")

        shape = (layer.input_size, layer.hidden_size, layer.num_layers, layer.batch_first)

        if shape != (input_size, hidden_size, 1, True):
            raise Mismatch(f"{call} should take {input_size} inputs to {hidden_size} hidden units, in one layer, with batch_first=True.")

        if kind != "lstm":
            continue

        forget = (layer.bias_ih_l0 + layer.bias_hh_l0)[hidden_size : 2 * hidden_size].detach()
        expect_tensor(
            call=f"the forget gate's two biases added, after {call}",
            actual=forget,
            expected=torch.full((hidden_size,), 2.0),
        )

    layer = found("lstm", 2, 4)
    expect_tensor(
        call='the forget gate\'s two biases added, after make_recurrent("lstm", 2, 4)',
        actual=(layer.bias_ih_l0 + layer.bias_hh_l0)[4:8].detach(),
        expected=torch.ones(4),
    )


def check_lstm3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="recurrent_size",
        description="counts each layer's weights",
        inputs=(
            ("rnn", 2, 32),
            ("gru", 2, 18),
            ("lstm", 2, 15),
            ("lstm", 10, 7),
            ("gru", 1, 16),
        ),
        reference=lambda kind, input_size, hidden_size: sum(
            parameter.numel() for parameter in RECURRENT_LAYERS[kind](input_size, hidden_size).parameters()
        ),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="hidden_for_budget",
        description="gives the largest hidden size that fits the budget",
        inputs=(
            ("rnn", 2, 1200),
            ("gru", 2, 1200),
            ("lstm", 2, 1200),
            ("lstm", 10, 5000),
            ("gru", 3, 100),
        ),
        reference=reference_hidden_for_budget,
        expect=expect_value,
    )
    checker.test(
        function="make_recurrent",
        description="builds the layer, an LSTM with its forget gate open",
        run=check_make_recurrent,
    )


LSTM3 = Exercise(
    name="lstm3",
    chapter=GATED_RNNS,
    title="An RNN, an LSTM or a GRU",
    lesson=LSTM3_LESSON,
    source=LSTM3_SOURCE,
    solution=LSTM3_SOLUTION,
    hints={
        "recurrent_size": "Each set holds hidden_size * input_size, hidden_size * hidden_size and two biases of "
        "hidden_size. Multiply by WEIGHT_SETS[kind].",
        "hidden_for_budget": "Start at 1 and step up while recurrent_size of one more unit still fits the budget.",
        "make_recurrent": "Pick nn.RNN, nn.GRU or nn.LSTM by kind and build it with batch_first=True. When kind is "
        'lstm, call set_forget_bias_(layer, forget_bias) before returning it.',
    },
    introduces=(),
    check=check_lstm3,
)
