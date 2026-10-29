"""42_bidirectional, one RNN forwards and one backwards, matched with PyTorch's, and when reading ahead is fair."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
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
    #   (batch, steps, input). Run run_rnn on x flipped along its steps, then
    #   flip the states back.
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
