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
from .bptt import marked_values
from .tensors import fastest_seconds, pinned_threads

GATED_RNNS = Chapter(
    folder="41_gated_rnns",
    title="Gated RNNs",
    readme=source_file(
        """
        # Gated RNNs

        A plain RNN replaces its whole hidden state at every step, and its
        gradient vanishes over many steps. A gated RNN computes gates from each
        input. At every step, the gates decide how much of the old state to
        keep and how much new content to write. This chapter builds the LSTM
        and the GRU and checks them against PyTorch's versions. It measures
        when each one beats a plain RNN. Last, it looks at the idea behind
        both, the gate, which is a weight computed from the input.

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
    A plain RNN computes its whole new hidden state through one tanh at every
    step. bptt3 showed that this makes the gradient vanish over many steps.
    The **LSTM**, from Hochreiter and Schmidhuber in 1997, keeps a second
    state vector, called the cell. At each step, gates control how much of
    the cell changes.

    Each step takes the input `x` and the previous hidden state `h`, and
    computes four vectors. Each of the four has its own weights and biases
    for `x` and for `h`. The names below are the ones PyTorch uses.

    ```text
    i = sigmoid(x @ W_ii.T + b_ii + h @ W_hi.T + b_hi)    how much of g goes in
    f = sigmoid(x @ W_if.T + b_if + h @ W_hf.T + b_hf)    how much of the old c is kept
    g = tanh(x @ W_ig.T + b_ig + h @ W_hg.T + b_hg)       the new content
    o = sigmoid(x @ W_io.T + b_io + h @ W_ho.T + b_ho)    how much of c reaches h
    c = f * c + i * g
    h = o * tanh(c)
    ```

    `i`, `f` and `o` are **gates**, called the input, forget and output gate.
    Each is a vector of numbers between 0 and 1, from the sigmoid. A gate
    multiplies another vector entry by entry, so each of its entries sets how
    much of the matching entry passes through.

    The cell update `c = f * c + i * g` has no tanh around it. Where `f` is
    near 1 and `i` is near 0, the cell keeps its value unchanged from one step
    to the next. Its gradient also flows back through those steps unchanged.
    The forget gate `f` was added later, by Gers, Schmidhuber and Cummins in
    2000. Without it the update was `c = c + i * g`, so a cell could only add
    to its value.

    `nn.LSTM` stacks the four sets of weights into one matrix, in the order i,
    f, g, o. So `weight_ih_l0` has shape (4 * hidden, input) and
    `weight_hh_l0` has shape (4 * hidden, hidden). It has two bias vectors,
    `bias_ih_l0` and `bias_hh_l0`, and both are added. After one matrix
    multiplication with the stacked weights, `.chunk(4, dim=1)` cuts the
    result back into four equal parts.

    ```python
    torch.arange(8.0).view(1, 8).chunk(4, dim=1)
    # (tensor([[0., 1.]]), tensor([[2., 3.]]), tensor([[4., 5.]]), tensor([[6., 7.]]))
    ```

    `nn.LSTM(input, hidden, batch_first=True)` takes `x` of shape (batch,
    steps, input) and returns two things. The first is the hidden state at
    every step. The second is a pair, the last `h` and the last `c`, each
    with an extra first dimension of size 1. Both states start at zeros.
    """,
)

LSTM1_SOURCE = source_file(
    '''
    # lstm1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    parts = torch.arange(8.0).view(1, 8).chunk(4, dim=1)    # four pieces of shape (1, 2)


    # TODO: Run one LSTM step and return the new h and the new c, in that
    #   order. x has shape (batch, input), and h and c have shape
    #   (batch, hidden). The weights and biases are stacked in nn.LSTM's
    #   order, i, f, g, o, as the lesson describes.
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


    # TODO: Run lstm_cell over every step of x, of shape (batch, steps, input),
    #   with the weights and biases stored in lstm. Start h and c at zeros.
    #   Return the hidden state h of every step, as one tensor of shape
    #   (batch, steps, hidden).
    def run_lstm(
        x: Tensor,
        lstm: nn.LSTM,
    ) -> Tensor:
        ...


    count = sum(p.numel() for p in nn.RNN(1, 16).parameters())    # 304


    # TODO: Return the number of parameters of nn.LSTM(input_size, hidden_size),
    #   which is the count of all the numbers in its weights and biases. Work
    #   it out from their shapes, without building the layer.
    #   Example: lstm_size(1, 16) returns 1216.
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
        "lstm_cell": "Add x @ weight_ih.T, bias_ih, h @ weight_hh.T and bias_hh, then split the sum into i, f, g "
        "and o with .chunk(4, dim=1). Apply sigmoid to i, f and o and tanh to g. Then compute c and h with the "
        "last two lines of the lesson's formulas.",
        "run_lstm": "Start h and c as torch.zeros(len(x), lstm.hidden_size). For each step t, call lstm_cell on "
        "x[:, t] with lstm.weight_ih_l0 and the other three tensors. Collect each h in a list, then torch.stack "
        "the list along dim=1.",
        "lstm_size": "Each of the 4 parts has a weight of shape (hidden_size, input_size) for x and a weight of "
        "shape (hidden_size, hidden_size) for h. It also has two biases of size hidden_size.",
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
    bptt3 showed that a plain RNN's gradient vanishes over many steps. lstm1
    built a cell whose update has no tanh around it. This lesson measures how
    much that helps, and shows a catch that comes with it.

    The task is the one from the end of bptt3. Each step holds a value from -1
    to 1 and a mark of 0 or 1. One of the first 5 steps is marked. After the
    last step, the model has to output the value of the marked step. Always
    guessing 0 gives a mean squared error of 0.333.

    Every model has 16 hidden units and trains for 1,500 steps of Adam. Each
    one trains three times, with different seeds. A run counts as solved if
    its error ends below 0.01. The length is the number of steps in each
    sequence.

    | Model | Length 10 | Length 20 | Length 40 |
    |---|---|---|---|
    | RNN | 3 of 3 | 0 of 3 | 0 of 3 |
    | LSTM, PyTorch's start | 3 of 3 | 0 of 3 | 0 of 3 |
    | LSTM, forget bias 1 | 3 of 3 | 3 of 3 | 0 of 3 |
    | LSTM, forget bias 3 | 3 of 3 | 3 of 3 | 3 of 3 |

    At length 10 every model solves the task. A plain RNN is fine when the
    value it needs is only a few steps back. It is also fine when the answer
    builds up over the whole sequence, like the vote in rnn3.

    The LSTM with PyTorch's starting weights fails at exactly the same lengths
    as the RNN. PyTorch starts every bias near 0, so the forget gate starts
    near `sigmoid(0)`, which is 0.5. So the cell's value is halved at every
    step, and so is its gradient. The table shows how much of the cell's
    value is left after 20 and after 40 steps, for three starting biases.

    | Forget bias | Forget gate at the start | Left after 20 steps | Left after 40 steps |
    |---|---|---|---|
    | 0 | 0.50 | 1e-6 | 9e-13 |
    | 1 | 0.73 | 0.0019 | 3.6e-6 |
    | 3 | 0.95 | 0.38 | 0.14 |

    With a higher starting bias for the forget gate, the cell keeps the value
    and its gradient across all the steps. This works from the very first
    step of training. Training can still lower the bias later, which closes
    the gate. Gers, Schmidhuber and Cummins proposed this in 2000. In 2015,
    Jozefowicz, Zaremba and Sutskever found that a bias of 1 improved the
    LSTM on almost every task. They recommended it for every LSTM. The
    identity start in bptt3 did the same thing for a ReLU RNN. In both cases
    the state is carried forward unchanged at the start of training.

    So the LSTM really does beat the plain RNN, for two reasons. Its cell
    update has no tanh around it, and its forget gate can start open.
    """,
)

LSTM2_SOURCE = source_file(
    '''
    # lstm2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    gate = torch.sigmoid(torch.tensor(1.0))    # tensor(0.7311), the forget gate at the start when its bias is 1


    # TODO: Return the fraction of a cell's value that is left after the given
    #   number of steps. The forget gate stays at sigmoid(bias) at every step,
    #   and nothing new is written to the cell. Return a Python float.
    #   Example: forget_fraction(1.0, 20) returns about 0.0019.
    def forget_fraction(
        bias: float,
        steps: int,
    ) -> float:
        ...


    second_quarter = torch.arange(8)[2:4]    # tensor([2, 3]), the second quarter of 8 rows


    # TODO: Set the forget gate's entries of lstm.bias_ih_l0 to value, and its
    #   entries of lstm.bias_hh_l0 to 0. Change the tensors in place, and make
    #   sure autograd does not record the change. The order is i, f, g, o, so
    #   the forget gate's entries are the second quarter of each bias.
    def set_forget_bias_(
        lstm: nn.LSTM,
        value: float,
    ) -> None:
        ...


    last = torch.zeros(2, 5, 16)[:, -1]    # shape (2, 16), each sequence's last step


    # TODO: Finish forward. x has shape (batch, steps, 2). Run self.layer on x,
    #   then apply self.head to the output of the last step. Return one guess
    #   per sequence, of shape (batch,). self.layer can be an nn.RNN, nn.GRU
    #   or nn.LSTM.
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
    """The lesson's table runs an nn.RNN through the same Recaller, whose last state is no (h, c) pair."""
    x, _ = marked_values(3, 6, torch.Generator().manual_seed(452))

    for layer in (nn.LSTM, nn.RNN):
        torch.manual_seed(451)
        model = found(layer(2, 4, batch_first=True), 4)
        torch.manual_seed(451)
        reference = ReferenceRecaller(layer(2, 4, batch_first=True), 4)

        with torch.no_grad():
            expected = reference(x)

        expect_tensor(
            call=f"Recaller(nn.{layer.__name__}(2, 4, batch_first=True), 4) on 3 sequences of 6 steps",
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
        description="gives the fraction of the cell left after many steps",
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
        "forget_fraction": "torch.sigmoid(torch.tensor(bias)).item() is the gate as a Python float. Raise it to the "
        "power steps.",
        "set_forget_bias_": "With h = lstm.hidden_size, the forget part is [h : 2 * h]. Inside torch.no_grad(), "
        "assign value to that slice of lstm.bias_ih_l0 and 0.0 to that slice of lstm.bias_hh_l0.",
        "Recaller.forward": "The first thing self.layer(x) returns is the outputs at every step. Take "
        "outputs[:, -1], apply self.head, then .squeeze(1).",
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
    The **GRU**, or gated recurrent unit, from Cho and colleagues in 2014, has
    one state and two gates. Each step computes the following, with PyTorch's
    names.

    ```text
    r = sigmoid(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)       how much of h goes into n
    z = sigmoid(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)       how much of the old h is kept
    n = tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))    the new content
    h = (1 - z) * n + z * h
    ```

    The last line is the gated memory from derive1. The update gate `z` sets
    how much of the old `h` is kept, and `1 - z` sets how much of the new
    content `n` is written. The reset gate `r` lets a step ignore the old
    state when it computes the new content.

    Cho's paper multiplies `h` by `r` before the matrix multiplication, as
    `(r * h) @ W_hn.T`. PyTorch multiplies after, as
    `r * (h @ W_hn.T + b_hn)`, and its documentation says this is on purpose,
    for speed. So in `nn.GRU`, the parts of `n` computed from `x` and from `h`
    must stay separate. Only the part from `h` is multiplied by `r`.

    `nn.GRU` stacks its weights in the order r, z, n. You can cut each stacked
    result into three parts with `.chunk(3, dim=1)`.

    A GRU has three sets of weights where the LSTM has four. With an input
    size of 1 and a hidden size of 16, the GRU has 912 parameters and the LSTM
    has 1,216.

    The update gate `z` also starts near 0.5. So the fraction of the old state
    that is kept halves at every step, as with the forget gate in lstm2. `z`
    is the second of the GRU's three parts, as `f` is the second of the
    LSTM's four. So `set_forget_bias_` from lstm2 works on a GRU as it is,
    and sets the bias of `z`. Here is the task from lstm2, with 16 units and
    three seeds.

    | Model | Length 10 | Length 20 | Length 40 |
    |---|---|---|---|
    | GRU, PyTorch's start | 3 of 3 | 3 of 3 | 1 of 3 |
    | GRU, update bias 3 | 3 of 3 | 3 of 3 | 3 of 3 |
    | LSTM, forget bias 3 | 3 of 3 | 3 of 3 | 3 of 3 |

    Chung and colleagues compared the GRU and the LSTM in 2014 on music and
    speech, and neither was better on every task. In 2015, Jozefowicz and
    colleagues found the GRU better than the LSTM on all their tasks except
    language modelling. But an LSTM with a forget bias of 1 beat both on
    almost all of them.
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


    # TODO: Run one GRU step with PyTorch's formulas from the lesson, and return
    #   the new h. x has shape (batch, input) and h has shape (batch, hidden).
    #   The weights and biases are stacked in nn.GRU's order, r, z, n.
    def gru_cell(
        x: Tensor,
        h: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Tensor:
        ...


    # TODO: Run gru_cell over every step of x, of shape (batch, steps, input),
    #   with the weights and biases stored in gru. Start h at zeros. Return the
    #   hidden state h of every step, as one tensor of shape
    #   (batch, steps, hidden).
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
        "gru_cell": "Split x @ weight_ih.T + bias_ih and h @ weight_hh.T + bias_hh into three parts each with "
        ".chunk(3, dim=1). For r and z, add the part from x and the part from h before the sigmoid. For n, "
        "multiply only the part from h by r.",
        "run_gru": "Start h as torch.zeros(len(x), gru.hidden_size). For each step t, call gru_cell on x[:, t] "
        "with gru.weight_ih_l0 and the other three tensors. Collect each h in a list, then torch.stack the list "
        "along dim=1.",
    },
    introduces=(),
    check=check_gru1,
)


# ---------------------------------------------------------------------------
# gating1
# ---------------------------------------------------------------------------

GATING1_LESSON = prose(
    """
    Once a Linear layer is trained, it multiplies every input by the same
    weights. A gate is different, because it is a weight computed from the
    input itself. The gated memory of derive1, the LSTM's i, f and o, and the
    GRU's r and z are all gates. Here is a gate on its own.

    ```text
    gate = sigmoid(x @ W.T + b)
    out = gate * signal
    ```

    Each entry of `signal` is multiplied by a number from 0 to 1 that was
    computed from `x`. A different `x` gives different numbers.

    A gate can multiply two inputs together, which no fixed weight can do.
    Take the rule "output the value when the switch is 1, and 0 when it is
    0". That rule is the value times the switch. Each model below learns this
    rule from a random value and a switch of 0 or 1. Each one trains for 2,000
    steps of Adam with a learning rate of 0.05, on five seeds.

    | Model | Weights | Mean squared error |
    |---|---|---|
    | Linear on the value and the switch | 3 | 0.24 to 0.26 |
    | MLP with 2 ReLU units | 9 | 0.19 to 0.51 |
    | MLP with 4 ReLU units | 17 | 0.00002 to 0.0002 |
    | gate on the switch, times the value | 2 | 0.0001 to 0.0002 |

    Always guessing 0 gives an error of 0.5. A Linear layer can only add the
    value and the switch, each times a weight. An MLP has to build the product
    out of the straight pieces its ReLU units make. The gate multiplies
    directly, so 2 weights are enough.

    The gated linear unit, or GLU, from Dauphin and colleagues in 2016, makes
    a whole layer of gates. One Linear layer produces twice as many outputs as
    the result needs. The first half is `a`, the second half is `b`, and the
    result is `a * sigmoid(b)`. `F.glu` does this split and product for you.

    ```python
    F.glu(torch.tensor([1.0, 2.0, 0.0, -3.0]), dim=-1)    # tensor([0.5000, 0.0949])
    ```

    Here `a` is `[1, 2]` and `b` is `[0, -3]`, so the gates `sigmoid(b)` are
    0.5 and 0.047. Section 7 comes back to weights computed from the input
    several times.
    """,
)

GATING1_SOURCE = source_file(
    '''
    # gating1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    scales = torch.sigmoid(torch.tensor([0.0, 2.0]))    # tensor([0.5000, 0.8808])


    # TODO: Return the gate computed from x, which is the sigmoid of a Linear
    #   layer with weight and bias applied to x. It holds one number from 0 to
    #   1 for each output of the Linear layer.
    def gate(
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    scaled = scales * torch.tensor([4.0, 4.0])    # tensor([2.0000, 3.5232])


    # TODO: Return signal multiplied, entry by entry, by the gate computed from
    #   x with weight and bias.
    def gated(
        signal: Tensor,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Tensor:
        ...


    halves = torch.tensor([1.0, 2.0, 0.0, -3.0]).chunk(2, dim=-1)    # (tensor([1., 2.]), tensor([ 0., -3.]))


    # TODO: Return the gated linear unit of x, without F.glu. Apply a Linear
    #   layer with weight and bias to x, which gives twice as many outputs as
    #   the result needs. Split them into a first half a and a second half b,
    #   and return a * sigmoid(b).
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
        description="gives the sigmoid of a Linear",
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
        description="multiplies the signal by the gate computed from x",
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
        "gate": "Apply torch.sigmoid to x @ weight.T + bias.",
        "gated": "Multiply signal by gate(x, weight, bias).",
        "glu": "Split x @ weight.T + bias into a and b with .chunk(2, dim=-1), then multiply a by torch.sigmoid of b.",
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
    This lesson uses what the chapter measured to decide which recurrent layer
    to use.

    First, does the LSTM win only because it has four times as many weights?
    To test that, give each model about 1,150 weights. With that budget the
    RNN gets 32 hidden units, the GRU gets 18 and the LSTM gets 15. Here is
    the task from lstm2 again, with three seeds for each model.

    | Model | Weights | Length 20 | Length 40 |
    |---|---|---|---|
    | RNN, 32 units | 1,152 | 0 of 3 | 0 of 3 |
    | GRU, 18 units, PyTorch's start | 1,188 | 3 of 3 | 1 of 3 |
    | GRU, 18 units, update bias 3 | 1,188 | 3 of 3 | 3 of 3 |
    | LSTM, 15 units, forget bias 1 | 1,140 | 3 of 3 | 0 of 3 |
    | LSTM, 15 units, forget bias 3 | 1,140 | 3 of 3 | 3 of 3 |

    More units do not help the RNN. Its problem is the tanh at every step, and
    a bigger state does not remove it. On speed, the RNN wins. On 4 threads of
    this Mac's CPU, an Apple M5, one training step at length 40 took 0.94 to
    0.96 ms for the RNN. The gated layers took 2.0 to 2.1 ms, more than twice
    as long.

    - A plain RNN suits answers that need only recent steps or build up over the whole sequence. It is the smallest and fastest.
    - A GRU or an LSTM handles long gaps when the gate that keeps its state starts open. `make_recurrent` below sets its bias to 3.
    - Jozefowicz and colleagues used a bias of 1, which handled 20 steps here but not 40.
    - A GRU does the same with three quarters of an LSTM's weights.

    All three layers must run one step after another, because each step needs
    the hidden state from the step before. So a GPU cannot compute the steps
    in parallel. 44_causal_conv and 45_state_space show models that train on
    every step at once. Section 7's models replaced RNNs for most text. An RNN
    still fits well where its fixed size state matters, like reading a stream
    one step at a time on a small device.
    """,
)

SET_FORGET_BIAS = '''
# set_forget_bias_ from lstm2. On a GRU it sets the bias of the update gate z, the GRU's second part.
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


    # How many sets of weights each layer has, one for each gate and one for the new content.
    WEIGHT_SETS = {"rnn": 1, "gru": 3, "lstm": 4}

    sets = WEIGHT_SETS["gru"]    # 3, the GRU's r, z and n


    # TODO: Return the number of parameters of a one layer nn.RNN, nn.GRU or
    #   nn.LSTM, chosen by kind, which is "rnn", "gru" or "lstm". Work it out
    #   from WEIGHT_SETS and the shapes in one set of weights, without
    #   building the layer.
    #   Example: recurrent_size("gru", 1, 16) returns 912.
    def recurrent_size(
        kind: str,
        input_size: int,
        hidden_size: int,
    ) -> int:
        ...


    fits = [h for h in range(1, 6) if h * h <= 10]    # [1, 2, 3], so 3 is the largest that fits


    # TODO: Return the largest hidden size whose recurrent_size, for kind and
    #   input_size, is at most budget.
    #   Example: hidden_for_budget("lstm", 2, 1200) returns 15.
    def hidden_for_budget(
        kind: str,
        input_size: int,
        budget: int,
    ) -> int:
        ...


    layer = nn.GRU(2, 18, batch_first=True)    # batch first, like every layer in this chapter


    # TODO: Build and return a one layer nn.RNN, nn.GRU or nn.LSTM, chosen by
    #   kind, with batch_first=True. For a GRU or an LSTM, call
    #   set_forget_bias_(layer, open_bias), so that the gate that keeps the
    #   state starts open.
    def make_recurrent(
        kind: str,
        input_size: int,
        hidden_size: int,
        open_bias: float = 3.0,
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


    # Each set has a weight for the input, a weight for h and two biases.
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
        open_bias: float = 3.0,
    ) -> nn.Module:
        layer = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](input_size, hidden_size, batch_first=True)

        if kind != "rnn":
            set_forget_bias_(layer, open_bias)

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


KEEPING_GATES: dict[str, str] = {
    "gru": "update gate's",
    "lstm": "forget gate's",
}


def check_make_recurrent(found: object) -> None:
    for kind, input_size, hidden_size in (("rnn", 3, 5), ("gru", 2, 4), ("lstm", 2, 6)):
        call = f'make_recurrent("{kind}", {input_size}, {hidden_size}, open_bias=2.0)'
        layer = found(kind, input_size, hidden_size, open_bias=2.0)

        if type(layer) is not RECURRENT_LAYERS[kind]:
            raise Mismatch(f"{call} is a {type(layer).__name__}, but it should be an nn.{RECURRENT_LAYERS[kind].__name__}.")

        shape = (layer.input_size, layer.hidden_size, layer.num_layers, layer.batch_first)

        if shape != (input_size, hidden_size, 1, True):
            raise Mismatch(f"{call} should take {input_size} inputs to {hidden_size} hidden units, in one layer, with batch_first=True.")

        if kind == "rnn":
            continue

        expect_tensor(
            call=f"the {KEEPING_GATES[kind]} two biases added, after {call}",
            actual=(layer.bias_ih_l0 + layer.bias_hh_l0)[hidden_size : 2 * hidden_size].detach(),
            expected=torch.full((hidden_size,), 2.0),
        )

    for kind in ("gru", "lstm"):
        layer = found(kind, 2, 4)
        expect_tensor(
            call=f'the {KEEPING_GATES[kind]} two biases added, after make_recurrent("{kind}", 2, 4)',
            actual=(layer.bias_ih_l0 + layer.bias_hh_l0)[4:8].detach(),
            expected=torch.full((4,), 3.0),
        )


def check_lstm3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="recurrent_size",
        description="counts each layer's parameters",
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
        description="builds the layer, a GRU or an LSTM with the gate that keeps its state open",
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
        "hidden_for_budget": "Start at a hidden size of 1, and add 1 while recurrent_size with one more unit is still "
        "at most budget.",
        "make_recurrent": "Pick nn.RNN, nn.GRU or nn.LSTM by kind and build it with batch_first=True. When kind is "
        "gru or lstm, call set_forget_bias_(layer, open_bias) before returning it.",
    },
    introduces=(),
    check=check_lstm3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

# Each run is a kind, a hidden size and the bias its keeping gate starts at, None for PyTorch's start.
LSTM2_RUNS: tuple[tuple[str, int, float | None], ...] = (
    ("rnn", 16, None),
    ("lstm", 16, None),
    ("lstm", 16, 1.0),
    ("lstm", 16, 3.0),
)
GRU1_RUNS: tuple[tuple[str, int, float | None], ...] = (
    ("gru", 16, None),
    ("gru", 16, 3.0),
)
LSTM3_RUNS: tuple[tuple[str, int, float | None], ...] = (
    ("rnn", 32, None),
    ("gru", 18, None),
    ("gru", 18, 3.0),
    ("lstm", 15, 1.0),
    ("lstm", 15, 3.0),
)


def measure_recall(
    runs: tuple[tuple[str, int, float | None], ...] = LSTM2_RUNS + GRU1_RUNS,
    lengths: tuple[int, ...] = (10, 20, 40),
    seeds: range = range(3),
) -> dict[tuple[str, int, float | None, int, int], float]:
    """lstm2's, gru1's and lstm3's tables, keyed by kind, hidden size, keeping bias, length and seed: the mean
    squared error after 1,500 Adam steps, where a run below 0.01 counts as solved. lstm3 passes LSTM3_RUNS and
    lengths of 20 and 40. A run at length 40 takes 5 to 20 seconds on 4 threads."""
    with pinned_threads():
        return {
            (kind, hidden, bias, length, seed): recall_error(seeded_recaller(kind, hidden, seed, bias), length, seed, 1500)
            for kind, hidden, bias in runs
            for length in lengths
            for seed in seeds
        }


def measure_step_times(repeats: int = 5) -> dict[str, float]:
    """lstm3's speeds: milliseconds per Adam step at length 40 on batches of 64, for its equal size RNN, GRU and
    LSTM, the fastest of several runs of 20 steps on 4 threads of the CPU."""
    results = {}
    x, y = marked_values(64, 40, torch.Generator().manual_seed(1))

    with pinned_threads():
        for kind, hidden, bias in (("rnn", 32, None), ("gru", 18, 3.0), ("lstm", 15, 3.0)):
            model = seeded_recaller(kind, hidden, 0, bias)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            def twenty_steps() -> None:
                for _ in range(20):
                    loss = F.mse_loss(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            results[kind] = fastest_seconds(twenty_steps, repeats) / 20 * 1000

    return results


def switch_batch(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """A random value, a switch of 0 or 1, and the value when the switch is on, else 0."""
    value = torch.randn(count, generator=generator)
    switch = torch.randint(0, 2, (count,), generator=generator).float()

    return value, switch, value * switch


class SwitchModel(nn.Module):
    """gating1's comparison. A Linear or a ReLU MLP reads the value and the switch side by side, and the gate
    scales the value by a sigmoid of a Linear on the switch."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

        if kind == "gate":
            self.net = nn.Linear(1, 1)
        elif kind == "linear":
            self.net = nn.Linear(2, 1)
        else:
            units = int(kind.split()[1])
            self.net = nn.Sequential(nn.Linear(2, units), nn.ReLU(), nn.Linear(units, 1))

    def forward(
        self,
        value: Tensor,
        switch: Tensor,
    ) -> Tensor:
        if self.kind == "gate":
            return torch.sigmoid(self.net(switch.unsqueeze(1))).squeeze(1) * value

        return self.net(torch.stack([value, switch], dim=1)).squeeze(1)


def measure_switch(seeds: range = range(5)) -> dict[tuple[str, int], float]:
    """gating1's table, keyed by model and seed: the mean squared error on 2,000 new pairs after 2,000 Adam steps
    at 0.05 on batches of 64. The weights each model has are in the lesson. A seed takes about a second."""
    results = {}

    with pinned_threads():
        for kind in ("linear", "mlp 2", "mlp 4", "gate"):
            for seed in seeds:
                torch.manual_seed(seed)
                model = SwitchModel(kind)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                generator = torch.Generator().manual_seed(seed + 1)

                for _ in range(2000):
                    value, switch, target = switch_batch(64, generator)
                    loss = F.mse_loss(model(value, switch), target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                value, switch, target = switch_batch(2000, torch.Generator().manual_seed(99))

                with torch.no_grad():
                    results[(kind, seed)] = F.mse_loss(model(value, switch), target).item()

    return results
