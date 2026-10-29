"""41_gated_rnns, the LSTM and the GRU built and matched with PyTorch's, and the gate as a weight computed from the input."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    check_on_inputs,
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
        GRU, matches them with PyTorch's, and names the idea under both, a
        weight computed from the input.

        ## Further reading

        - [Olah, understanding LSTM networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
        - [Dive into Deep Learning, the LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)
        - [Dive into Deep Learning, the GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)
        - [Cho and colleagues, the paper that introduced the GRU](https://arxiv.org/abs/1406.1078)
        - [Greff and colleagues, LSTM: a search space odyssey](https://arxiv.org/abs/1503.04069)
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
