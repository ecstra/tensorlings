"""40_bptt, backprop through time by hand, truncated backprop in chunks, and the gradients that vanish or explode over long spans."""

from __future__ import annotations

import copy
import textwrap

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
    expect_close,
    expect_parts,
    expect_tensor,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .rnn import vote_sequences
from .tensors import pinned_threads

BPTT = Chapter(
    folder="40_bptt",
    title="Backprop through time",
    readme=source_file(
        """
        # Backprop through time

        Training an RNN means backprop through every step it ran. This chapter
        does that by hand on a one number RNN, splits long sequences into
        chunks the way language models trained on them did, and measures the
        gradient fading over long spans and blowing up, with clipping as the
        fix for the second.

        ## Further reading

        - [Dive into Deep Learning, backpropagation through time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)
        - [Pascanu, Mikolov and Bengio, on the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)
        - [Le, Jaitly and Hinton, a simple way to initialize recurrent networks of rectified linear units](https://arxiv.org/abs/1504.00941)
        """,
    ),
)


def reference_scalar_rnn(
    xs: Tensor,
    w: Tensor,
    u: Tensor,
) -> Tensor:
    h = torch.zeros(())
    states = []

    for x in xs:
        h = torch.tanh(w * h + u * x)
        states.append(h)

    return torch.stack(states)


def autograd_rnn_gradients(
    xs: Tensor,
    w: Tensor,
    u: Tensor,
) -> tuple[Tensor, Tensor]:
    w = w.clone().requires_grad_()
    u = u.clone().requires_grad_()
    reference_scalar_rnn(xs, w, u)[-1].backward()

    return w.grad, u.grad


def reference_state_gradients(
    xs: Tensor,
    w: Tensor,
    u: Tensor,
) -> Tensor:
    factors = (1 - reference_scalar_rnn(xs, w, u) ** 2) * w

    return torch.cumprod(factors.flip(0), dim=0)


# ---------------------------------------------------------------------------
# bptt1
# ---------------------------------------------------------------------------

BPTT1_LESSON = prose(
    """
    An RNN uses the same weights at every step. Unrolled, its loop is one long
    chain of layers that share their weights, and backprop runs through it like
    through any chain. Running it backwards through the steps is called
    **backprop through time**.

    Take an RNN whose state is one number, `h = tanh(w * h + u * x)` at each
    step. Walking back from the last step, the chain rule gives three lines
    per step.

    - The gradient inside the tanh is the gradient of `h` times `1 - h ** 2`, tanh's derivative.
    - That times the previous step's `h` adds to `w`'s gradient, and that times `x` adds to `u`'s.
    - That times `w` is the gradient of the previous state, where the next step back starts.

    Every step adds to the same `w` and `u`, since they are the same numbers at
    every step.

    The last line is where trouble starts. The gradient reaching a state k
    steps back is k factors multiplied together, `(1 - h ** 2) * w` from each
    step. A number multiplied by itself often enough shrinks towards 0 or
    grows without end.

    | Factor | 10 times | 50 times |
    |---|---|---|
    | 0.9 | 0.349 | 0.005 |
    | 1.1 | 2.59 | 117 |

    With one number for a state, a bigger `w` doesn't explode for long. The
    state saturates near -1 or 1, `1 - h ** 2` falls towards 0, and the
    factors shrink. Here is `state_gradients` below on 50 random inputs, with
    `u` at 0.5.

    | w | 1 step back | 10 steps back | 50 steps back |
    |---|---|---|---|
    | 0.5 | 0.481 | 1e-5 | 5e-22 |
    | 0.9 | 0.796 | 0.004 | 6e-11 |
    | 1.5 | 0.846 | 7e-5 | 2e-15 |
    | 3 | 0.036 | 1e-14 | 0 |

    Only with inputs of 0, where every state stays at 0, does a `w` of 1.1
    give the 2.59 and 117 of the first table. So this one number RNN's
    gradient fades, and its far past barely trains. bptt3 shows a vector
    state, where a large `W_hh` makes the gradient explode instead.
    """,
)

BPTT1_SOURCE = source_file(
    '''
    # bptt1.py

    import torch
    from torch import Tensor

    factors = torch.tensor([0.5, 0.5, 0.5])
    so_far = torch.cumprod(factors, dim=0)    # tensor([0.5000, 0.2500, 0.1250]), the running products


    # TODO: Run a one number RNN along xs, of shape (steps,), from a state of 0,
    #   with h = tanh(w * h + u * x) at each step. Return every step's state,
    #   of shape (steps,).
    def scalar_rnn(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the loop, which walks back from the last step to the first by
    #   the lesson's three lines. It returns the gradient of the last state with
    #   respect to w and to u, without autograd.
    def backprop_through_time(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> tuple[Tensor, Tensor]:
        states = scalar_rnn(xs, w, u)
        grad_h = torch.tensor(1.0)
        grad_w = torch.tensor(0.0)
        grad_u = torch.tensor(0.0)

        for t in reversed(range(len(xs))):
            before = states[t - 1] if t > 0 else torch.tensor(0.0)
            grad_inside = ...
            grad_w = ...
            grad_u = ...
            grad_h = ...

        return grad_w, grad_u


    # TODO: Return the gradient of the last state with respect to every state
    #   before it, the state one step back first and the starting state last,
    #   of shape (steps,). Each is a running product of the factors
    #   (1 - h ** 2) * w, starting from the last step's.
    def state_gradients(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> Tensor:
        ...
    ''',
)

BPTT1_SOLUTION = source_file(
    '''
    # bptt1.py, a solution for comparison

    import torch
    from torch import Tensor

    factors = torch.tensor([0.5, 0.5, 0.5])
    so_far = torch.cumprod(factors, dim=0)


    def scalar_rnn(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> Tensor:
        h = torch.tensor(0.0)
        states = []

        for x in xs:
            h = torch.tanh(w * h + u * x)
            states.append(h)

        return torch.stack(states)


    def backprop_through_time(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> tuple[Tensor, Tensor]:
        states = scalar_rnn(xs, w, u)
        grad_h = torch.tensor(1.0)
        grad_w = torch.tensor(0.0)
        grad_u = torch.tensor(0.0)

        for t in reversed(range(len(xs))):
            before = states[t - 1] if t > 0 else torch.tensor(0.0)
            grad_inside = grad_h * (1 - states[t] ** 2)
            grad_w = grad_w + grad_inside * before
            grad_u = grad_u + grad_inside * xs[t]
            grad_h = grad_inside * w

        return grad_w, grad_u


    # Step t's factor is how much its state moves per nudge to the state before it.
    def state_gradients(
        xs: Tensor,
        w: Tensor,
        u: Tensor,
    ) -> Tensor:
        factors = (1 - scalar_rnn(xs, w, u) ** 2) * w

        return torch.cumprod(factors.flip(0), dim=0)
    ''',
)


def scalar_inputs(
    seed: int,
    steps: int,
) -> Tensor:
    return torch.randn(steps, generator=torch.Generator().manual_seed(seed))


BPTT1_CASES = (
    (scalar_inputs(441, 6), torch.tensor(0.8), torch.tensor(0.5)),
    (scalar_inputs(442, 9), torch.tensor(1.3), torch.tensor(-0.7)),
    (scalar_inputs(443, 4), torch.tensor(-0.6), torch.tensor(1.2)),
)


def check_bptt1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="scalar_rnn",
        description="gives every step's state",
        inputs=BPTT1_CASES,
        reference=reference_scalar_rnn,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="backprop_through_time",
        description="matches autograd's gradients for w and u",
        inputs=BPTT1_CASES,
        reference=autograd_rnn_gradients,
        expect=expect_parts("the gradient for w", "the gradient for u"),
    )
    check_on_inputs(
        checker=checker,
        function="state_gradients",
        description="gives the gradient reaching each earlier state",
        inputs=BPTT1_CASES,
        reference=reference_state_gradients,
        expect=expect_tensor,
    )


BPTT1 = Exercise(
    name="bptt1",
    chapter=BPTT,
    title="Backprop through time by hand",
    lesson=BPTT1_LESSON,
    source=BPTT1_SOURCE,
    solution=BPTT1_SOLUTION,
    hints={
        "scalar_rnn": "Start h at torch.tensor(0.0). For each x in xs, set h to torch.tanh(w * h + u * x) and keep it "
        "in a list. torch.stack the list.",
        "backprop_through_time": "grad_inside is grad_h * (1 - states[t] ** 2). Add grad_inside * before to grad_w "
        "and grad_inside * xs[t] to grad_u, then set grad_h to grad_inside * w.",
        "state_gradients": "The factors are (1 - scalar_rnn(xs, w, u) ** 2) * w. Flip them so the last step's comes "
        "first, then take torch.cumprod with dim=0.",
    },
    introduces=(),
    check=check_bptt1,
)


# ---------------------------------------------------------------------------
# bptt2
# ---------------------------------------------------------------------------


def delayed_sequences(
    count: int,
    steps: int,
    delay: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Steps of +1 or -1, and as targets the same steps delay steps later, with 0 before there is one to copy."""
    x = torch.randint(0, 2, (count, steps), generator=generator).float() * 2 - 1
    y = torch.cat([torch.zeros(count, delay), x[:, :-delay]], dim=1)

    return x.unsqueeze(2), y.unsqueeze(2)


BPTT2_LESSON = prose(
    """
    Backprop through time keeps every step's numbers until `backward` runs, and
    the model learns only once per pass. On a sequence of a million steps,
    like a long text, that is far too much memory and far too few updates.

    Truncated backprop through time cuts the sequence into chunks of a few
    dozen steps and updates after each chunk. Each chunk starts from the state
    the last one ended in, so the model still reads the whole sequence in
    order. The gradient, though, stops at the chunk's start.

    `.detach()` makes that cut. It gives a tensor with the same numbers and no
    history, so `backward` stops there.

    ```python
    h = torch.ones(2, requires_grad=True) * 3
    h.detach().requires_grad    # False, the same numbers cut from how they were made
    ```

    Without it, the second chunk's `backward` would run back into the first
    chunk's steps, whose saved numbers the first `backward` already freed.
    PyTorch stops with an error that says it is trying to backward through the
    graph a second time.

    `nn.RNN` takes a starting state as its second argument, of shape (layers,
    batch, hidden), and returns the state it ended in.

    The cost is the longest span the model can learn. Here an RNN learns to
    give back each input 3 steps later, on 32 sequences of 1,000 steps, in 10
    passes of Adam.

    | Chunk length | Updates per pass | Mean squared error |
    |---|---|---|
    | 2 | 500 | 0.957 |
    | 4 | 250 | 0.00008 |
    | 10 | 100 | 0.00007 |
    | 20 | 50 | 0.00012 |
    | 100 | 10 | 0.001 |
    | 1,000, the whole sequence | 1 | 0.503 |

    Chunks of 2 never learn it, since no gradient ever spans the 3 steps it
    needs. Their error stays near 1, what always guessing 0 gives. Chunks of 4
    to 20 learn it well. The whole sequence learns it worst after chunks of 2,
    with only 10 updates in all. Over five seeds, chunks of 4 to 20 stayed
    below 0.0002, while chunks of 2 and the whole sequence stayed at 0.48 or
    more. So a chunk must be longer than the span to learn, and short enough
    to give many updates.
    """,
)

BPTT2_SOURCE = source_file(
    '''
    # bptt2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    h = torch.ones(2, requires_grad=True) * 3
    cut = h.detach()              # the same numbers, with no history behind them
    cut_has_history = cut.requires_grad    # False


    # TODO: Finish forward. x has shape (batch, steps, 1), and h is the state to
    #   start from, or None for zeros. Return the prediction at every step, of
    #   shape (batch, steps, 1), and the state the RNN ended in.
    class Predictor(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            h: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]:
            ...


    # TODO: Finish the loop, one pass of truncated backprop through time over x
    #   and y, both of shape (batch, steps, 1), in chunks of `length` steps. Each
    #   chunk starts from the state the chunk before ended in, cut from its
    #   history, and takes one step of optimizer on its mean squared error.
    #   Return the last chunk's loss.
    def train_truncated(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        length: int,
    ) -> float:
        h = None

        for x_chunk, y_chunk in zip(x.split(length, dim=1), y.split(length, dim=1)):
            ...

        return loss.item()
    ''',
)

BPTT2_SOLUTION = source_file(
    '''
    # bptt2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    h = torch.ones(2, requires_grad=True) * 3
    cut = h.detach()
    cut_has_history = cut.requires_grad


    class Predictor(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(
            self,
            x: Tensor,
            h: Tensor | None = None,
        ) -> tuple[Tensor, Tensor]:
            outputs, last = self.rnn(x, h)

            return self.out(outputs), last


    def train_truncated(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        length: int,
    ) -> float:
        h = None

        for x_chunk, y_chunk in zip(x.split(length, dim=1), y.split(length, dim=1)):
            predictions, h = model(x_chunk, h)
            loss = F.mse_loss(predictions, y_chunk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            h = h.detach()

        return loss.item()
    ''',
)


class ReferencePredictor(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(
        self,
        x: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        outputs, last = self.rnn(x, h)

        return self.out(outputs), last


def seeded(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def reference_train_truncated(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    length: int,
) -> float:
    h = None

    for x_chunk, y_chunk in zip(x.split(length, dim=1), y.split(length, dim=1)):
        predictions, h = model(x_chunk, h)
        loss = F.mse_loss(predictions, y_chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h = h.detach()

    return loss.item()


def delay_error(
    model: nn.Module,
    delay: int,
) -> float:
    x, y = delayed_sequences(32, 1000, delay, torch.Generator().manual_seed(2))

    with torch.no_grad():
        return F.mse_loss(model(x)[0], y).item()


def check_predictor_forward(found: type) -> None:
    model = expect_module("Predictor(8)", seeded(found, 8))
    reference = seeded(ReferencePredictor, 8)
    x, _ = delayed_sequences(3, 7, 2, torch.Generator().manual_seed(451))
    start = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(452))

    for call, h in (("Predictor(8) on 3 sequences of 7 steps", None), ("Predictor(8) from a given state", start)):
        with torch.no_grad():
            expected = reference(x, h)

        result = model(x, h)

        if not isinstance(result, tuple) or len(result) != 2:
            raise Mismatch(f"{call} should give the predictions and the last state, as a tuple of two.")

        expect_tensor(
            call=f"the predictions of {call}",
            actual=result[0],
            expected=expected[0],
        )
        expect_tensor(
            call=f"the last state of {call}",
            actual=result[1],
            expected=expected[1],
        )


def check_one_pass(found: object) -> None:
    x, y = delayed_sequences(4, 30, 3, torch.Generator().manual_seed(453))
    runs = []

    for train in (found, reference_train_truncated):
        model = seeded(ReferencePredictor, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = train(model, optimizer, x, y, 7)
        runs.append((loss, model.out.weight.detach().clone()))

    (loss, weight), (expected_loss, expected_weight) = runs
    expect_close(
        call="train_truncated in chunks of 7 over 4 sequences of 30 steps",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call="the output layer's weights after that pass",
        actual=weight,
        expected=expected_weight,
    )


def check_truncated_learns(found: object) -> None:
    model = seeded(ReferencePredictor, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    x, y = delayed_sequences(32, 1000, 3, torch.Generator().manual_seed(1))

    for _ in range(3):
        found(model, optimizer, x, y, 10)

    error = delay_error(model, 3)

    if error > 0.05:
        raise Mismatch(
            f"After 3 passes in chunks of 10, the model gives back inputs from 3 steps earlier with a mean squared "
            f"error of {error:.3f}, but it should be below 0.05.",
        )


def check_bptt2(checker: Checker) -> None:
    checker.test_method(
        class_name="Predictor",
        method="forward",
        description="predicts every step and gives back the last state",
        run=check_predictor_forward,
    )
    checker.test(
        function="train_truncated",
        description="updates once per chunk, carrying the state across",
        run=check_one_pass,
    )
    checker.test(
        function="train_truncated",
        description="learns to give back each input 3 steps later",
        run=check_truncated_learns,
    )


BPTT2 = Exercise(
    name="bptt2",
    chapter=BPTT,
    title="Truncated backprop through time",
    lesson=BPTT2_LESSON,
    source=BPTT2_SOURCE,
    solution=BPTT2_SOLUTION,
    hints={
        "Predictor.forward": "self.rnn(x, h) gives every step's state and the last state. Pass the first through "
        "self.out and return both.",
        "train_truncated": "Call model on x_chunk and the carried h, and keep both the predictions and the new h. "
        "Take F.mse_loss against y_chunk, zero the gradients, backward and step. Then set h = h.detach() for the "
        "next chunk.",
    },
    introduces=(),
    check=check_bptt2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# bptt3
# ---------------------------------------------------------------------------

BPTT3_LESSON = prose(
    """
    bptt1 multiplied factors on a one number state. A real RNN has a vector
    state and a matrix `W_hh`, and the same thing happens. Here is
    `gradient_by_distance` below on a tanh `nn.RNN` with a state of 32 and
    `batch_first=True`, over 16 sequences of 60 random steps.

    | Steps back | 0 | 1 | 5 | 10 | 20 | 30 | 50 |
    |---|---|---|---|---|---|---|---|
    | PyTorch's start | 1.04 | 1.41 | 0.046 | 0.021 | 0.0001 | 8e-7 | 3e-11 |
    | `W_hh` times 5 | 0.80 | 1.44 | 4.5 | 24 | 377 | 5,650 | 559,000 |

    From PyTorch's start the gradient fades, the **vanishing gradient**. A
    step 50 back gets a gradient 30 billion times weaker than the last step's,
    so the model can't learn that it matters. A bigger learning rate doesn't
    help, since the signal never arrives. Gates, in 41_gated_rnns, are the fix.

    The second row is the opposite, the **exploding gradient**. With a vector
    state, a large `W_hh` multiplies the gradient up faster than the tanh
    shrinks it. A ReLU never shrinks it at all.

    Le, Jaitly and Hinton started a ReLU RNN's `W_hh` at the **identity
    matrix**, 1 on the diagonal and 0 elsewhere. Then `h @ W_hh.T` is `h`, so
    the state carries forward unchanged. A little above the identity, the
    state and its gradient grow at every step. Here are such RNNs, with a
    state of 16, trained on votes of 51 steps by 300 steps of SGD at 0.01.

    | Start | Clipping | Largest gradient norm | Accuracy |
    |---|---|---|---|
    | the identity | none | 2,211 | 61.8% |
    | the identity | at 1.0 | 94 | 81.6% |
    | 1.05 times the identity | none | 141 million | 50.6% |
    | 1.05 times the identity | at 1.0 | 590 | 79.5% |

    Unclipped, one huge gradient throws the weights far off in a single step.
    Clipped at 1.0, the norms before clipping still spike, but no step moves
    the weights by more than 0.01 in all. Over five seeds the clipped runs
    reached 79.0% to 84.9%, and the unclipped ones 49.5% to 61.8%.

    Clipping, from clipping1, caps the size of a step and keeps its direction.
    Pascanu, Mikolov and Bengio proposed it for RNNs whose gradients explode,
    and the RNNs in their paper were tanh and sigmoid ones. It does nothing
    for a gradient that fades.

    The identity start also carries a value across a long gap. Each step holds
    a value from -1 to 1 and a mark. One of the first 5 steps is marked, and
    the RNN gives back its value after 40 steps. Each start trained for 1,500
    steps of Adam at 0.01, clipped at 1.0, on five seeds.

    | Start | Mean squared error |
    |---|---|
    | PyTorch's, tanh | 0.333 to 0.340 |
    | PyTorch's, ReLU | 0.333 to 0.334 |
    | the identity, ReLU | 0.003 to 0.026 |

    Guessing 0 scores 0.333, so from PyTorch's start no seed learns it. From
    the identity, every seed does. lstm2 comes back to this task.
    """,
)

GIVEN_VOTES = '''
# The votes of rnn3, each step +1 or -1, labelled by whether the +1s won.
def vote_sequences(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    x = torch.randint(0, 2, (count, steps), generator=generator).float() * 2 - 1

    return x, (x.sum(dim=1) > 0).float()


class ReluReader(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True, nonlinearity="relu")
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, last = self.rnn(x.unsqueeze(2))

        return self.out(last[0]).squeeze(1)
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_VOTES = textwrap.indent(GIVEN_VOTES.strip("\n"), "    ")

BPTT3_SOURCE = source_file(
    '''
    # bptt3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.ones(2, 3, 1, requires_grad=True)
    (x * 2).sum().backward()
    per_step = x.grad.norm(dim=(0, 2))    # tensor([2.8284, 2.8284, 2.8284]), one norm per step


''' + INDENTED_VOTES + '''


    # TODO: Return how strongly the last state of rnn depends on each step of x.
    #   x has shape (batch, steps, 1). Take the gradient of the sum of the last
    #   state with respect to x, and its norm over the batch for each step.
    #   Return one norm per step, shape (steps,), the last step first.
    def gradient_by_distance(
        rnn: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        ...


    eye = torch.eye(3)    # tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), the identity matrix
    size = nn.RNN(1, 3).hidden_size    # 3, the size of its state


    # TODO: Set rnn's weight_hh_l0 to scale times the identity matrix, in place,
    #   without recording for autograd.
    def identity_start_(
        rnn: nn.RNN,
        scale: float,
    ) -> None:
        ...


    # TODO: Finish the loop. After backward, clip the gradient's norm to
    #   max_norm, or only measure it when max_norm is None, as clipping2 did.
    #   Keep the largest norm seen and return it.
    def train_votes(
        model: nn.Module,
        steps: int,
        max_norm: float | None,
    ) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(1)
        largest = 0.0

        for _ in range(steps):
            x, y = vote_sequences(64, 51, generator)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            ...
            optimizer.step()

        return largest
    ''',
)

BPTT3_SOLUTION = source_file(
    '''
    # bptt3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.ones(2, 3, 1, requires_grad=True)
    (x * 2).sum().backward()
    per_step = x.grad.norm(dim=(0, 2))


''' + INDENTED_VOTES + '''


    def gradient_by_distance(
        rnn: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        x = x.clone().requires_grad_()
        _, last = rnn(x)
        last.sum().backward()

        return x.grad.norm(dim=(0, 2)).flip(0)


    eye = torch.eye(3)
    size = nn.RNN(1, 3).hidden_size


    def identity_start_(
        rnn: nn.RNN,
        scale: float,
    ) -> None:
        with torch.no_grad():
            rnn.weight_hh_l0.copy_(torch.eye(rnn.hidden_size) * scale)


    def train_votes(
        model: nn.Module,
        steps: int,
        max_norm: float | None,
    ) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(1)
        largest = 0.0

        for _ in range(steps):
            x, y = vote_sequences(64, 51, generator)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            cap = float("inf") if max_norm is None else max_norm
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cap).item()
            largest = max(largest, norm)
            optimizer.step()

        return largest
    ''',
)


class ReferenceReluReader(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True, nonlinearity="relu")
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.rnn(x.unsqueeze(2))[1][0]).squeeze(1)


def reference_gradient_by_distance(
    rnn: nn.RNN,
    x: Tensor,
) -> Tensor:
    x = x.clone().requires_grad_()
    rnn(x)[1].sum().backward()

    return x.grad.norm(dim=(0, 2)).flip(0)


def reference_identity_start_(
    rnn: nn.RNN,
    scale: float,
) -> None:
    with torch.no_grad():
        rnn.weight_hh_l0.copy_(torch.eye(rnn.hidden_size) * scale)


def reference_train_votes(
    model: nn.Module,
    steps: int,
    max_norm: float | None,
) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)
    largest = 0.0

    for _ in range(steps):
        x, y = vote_sequences(64, 51, generator)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        cap = float("inf") if max_norm is None else max_norm
        largest = max(largest, torch.nn.utils.clip_grad_norm_(model.parameters(), cap).item())
        optimizer.step()

    return largest


def identity_reader(scale: float) -> nn.Module:
    """A ReLU reader with a state of 16 whose hidden to hidden weight starts at scale times the identity."""
    model = seeded(ReferenceReluReader, 16)
    reference_identity_start_(model.rnn, scale)

    return model


def votes_of_51_accuracy(model: nn.Module) -> float:
    x, y = vote_sequences(2000, 51, torch.Generator().manual_seed(99))

    with torch.no_grad():
        return ((model(x) > 0).float() == y).float().mean().item()


def seeded_rnn(
    hidden: int,
    nonlinearity: str,
) -> nn.RNN:
    torch.manual_seed(461)

    return nn.RNN(1, hidden, batch_first=True, nonlinearity=nonlinearity)


def check_identity_start(found: object) -> None:
    for scale in (1.0, 1.05):
        rnn = seeded_rnn(5, "relu")
        before = rnn.weight_ih_l0.detach().clone()
        found(rnn, scale)

        expect_tensor(
            call=f"weight_hh_l0 after identity_start_ with a scale of {scale}",
            actual=rnn.weight_hh_l0.detach(),
            expected=torch.eye(5) * scale,
        )

        if not torch.equal(rnn.weight_ih_l0.detach(), before):
            raise Mismatch("identity_start_ changed weight_ih_l0 too. Only weight_hh_l0 should change.")


def check_train_votes_steps(found: object) -> None:
    for max_norm in (None, 0.5):
        model = identity_reader(1.0)
        largest = found(model, 5, max_norm)
        reference = identity_reader(1.0)
        expected = reference_train_votes(reference, 5, max_norm)
        call = f"train_votes for 5 steps with max_norm {max_norm}"

        # The norm is about 2,200, where float32 rounding alone moves the 7th digit between ways of computing it.
        expect_close(
            call=call,
            actual=largest,
            expected=expected,
            tolerance=1e-5 * expected,
        )
        expect_tensor(
            call=f"the output layer's weights after {call}",
            actual=model.out.weight.detach(),
            expected=reference.out.weight.detach(),
        )


def check_clipping_rescues(found: object) -> None:
    model = identity_reader(1.05)
    found(model, 300, 1.0)
    accuracy = votes_of_51_accuracy(model)

    if accuracy < 0.7:
        raise Mismatch(
            f"Started at 1.05 times the identity and clipped at 1.0, the reader gets {accuracy:.1%} of new votes of 51 "
            "right after 300 steps, but it should pass 70%.",
        )


def check_bptt3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gradient_by_distance",
        description="gives the gradient's norm for each step, the last step first",
        inputs=(
            (Described(seeded_rnn(8, "tanh"), "an nn.RNN(1, 8)"), small_tensor(462, 3, 6, 1) / 3),
            (Described(seeded_rnn(4, "relu"), "a ReLU nn.RNN(1, 4)"), small_tensor(463, 2, 9, 1) / 3),
        ),
        reference=reference_gradient_by_distance,
        expect=expect_tensor,
    )
    checker.test(
        function="identity_start_",
        description="sets the hidden to hidden weight to a scaled identity",
        run=check_identity_start,
    )
    checker.test(
        function="train_votes",
        description="clips, or only measures, and keeps the largest norm",
        run=check_train_votes_steps,
    )
    checker.test(
        function="train_votes",
        description="clipping lets an exploding RNN learn",
        run=check_clipping_rescues,
    )


BPTT3 = Exercise(
    name="bptt3",
    chapter=BPTT,
    title="Gradients that vanish and explode",
    lesson=BPTT3_LESSON,
    source=BPTT3_SOURCE,
    solution=BPTT3_SOLUTION,
    hints={
        "gradient_by_distance": "Clone x and call .requires_grad_(). Run rnn on it, call .sum().backward() on the "
        "last state, take x.grad.norm(dim=(0, 2)), and .flip(0) it.",
        "identity_start_": "Inside torch.no_grad(), build torch.eye of rnn.hidden_size times scale and .copy_ it into "
        "rnn.weight_hh_l0.",
        "train_votes": "cap is float('inf') when max_norm is None, else max_norm. clip_grad_norm_ returns the norm "
        "from before clipping, so keep the larger of it and largest.",
    },
    introduces=(),
    check=check_bptt3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_state_gradients() -> dict[float, tuple[float, float, float]]:
    """bptt1's second table: state_gradients on 50 inputs drawn with seed 0, u at 0.5, for each w, the gradient 1,
    10 and 50 steps back."""
    xs = scalar_inputs(0, 50)
    results = {}

    for w in (0.5, 0.9, 1.5, 3.0):
        gradients = reference_state_gradients(xs, torch.tensor(w), torch.tensor(0.5))
        results[w] = (gradients[0].item(), gradients[9].item(), gradients[49].item())

    return results


def measure_chunks(seeds: range = range(5)) -> dict[tuple[int, int], float]:
    """bptt2's table, keyed by seed and chunk length: the mean squared error after 10 passes of truncated backprop
    with Adam at 0.003, an RNN of 32 and 32 sequences of 1,000 steps that give back each input 3 steps later. Seed
    0 is the table's run. Each seed takes about 5 seconds."""
    results = {}

    with pinned_threads():
        for seed in seeds:
            x, y = delayed_sequences(32, 1000, 3, torch.Generator().manual_seed(seed + 1))

            for length in (2, 4, 10, 20, 100, 1000):
                torch.manual_seed(seed)
                model = ReferencePredictor(32)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

                for _ in range(10):
                    reference_train_truncated(model, optimizer, x, y, length)

                results[(seed, length)] = delay_error(model, 3)

    return results


GRADIENT_DISTANCES = (0, 1, 5, 10, 20, 30, 50)


def measure_gradient_distances() -> dict[str, list[float]]:
    """bptt3's first table: gradient_by_distance on a tanh nn.RNN(1, 32) made after torch.manual_seed(0), over 16
    sequences of 60 random steps, from PyTorch's start and with its W_hh times 5."""
    torch.manual_seed(0)
    rnn = nn.RNN(1, 32, batch_first=True)
    x = torch.randn(16, 60, 1)
    larger = copy.deepcopy(rnn)

    with torch.no_grad():
        larger.weight_hh_l0.mul_(5)

    return {
        name: [reference_gradient_by_distance(layer, x)[steps].item() for steps in GRADIENT_DISTANCES]
        for name, layer in (("pytorch", rnn), ("times 5", larger))
    }


def measure_starts(seeds: range = range(5)) -> dict[tuple[int, float, float | None], tuple[float, float]]:
    """bptt3's clipping table, keyed by seed, the scale of the identity start and the clipping cap: the largest
    gradient norm before clipping and the accuracy on 2,000 new votes of 51, after 300 steps of SGD at 0.01. Seed 0
    is the table's run."""
    results = {}

    with pinned_threads():
        for seed in seeds:
            for scale in (1.0, 1.05):
                for max_norm in (None, 1.0):
                    torch.manual_seed(seed)
                    model = ReferenceReluReader(16)
                    reference_identity_start_(model.rnn, scale)
                    largest = reference_train_votes(model, 300, max_norm)
                    results[(seed, scale, max_norm)] = (largest, votes_of_51_accuracy(model))

    return results


def marked_values(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Each step holds a value from -1 to 1 and a mark, one of the first 5 steps is marked, and the answer is its
    value. bptt3 and lstm2 train on it."""
    values = 2 * torch.rand(count, steps, generator=generator) - 1
    where = torch.randint(0, 5, (count,), generator=generator)
    rows = torch.arange(count)
    marks = torch.zeros(count, steps)
    marks[rows, where] = 1.0

    return torch.stack([values, marks], dim=2), values[rows, where]


class MarkedReader(nn.Module):
    """An RNN of 16 that reads (value, mark) steps and guesses the marked value from its last state."""

    def __init__(self, start: str) -> None:
        super().__init__()
        self.rnn = nn.RNN(2, 16, batch_first=True, nonlinearity="tanh" if start == "tanh" else "relu")
        self.head = nn.Linear(16, 1)

        if start == "identity":
            reference_identity_start_(self.rnn, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.rnn(x)[1][0]).squeeze(1)


def marked_error(
    start: str,
    seed: int,
) -> float:
    """One run of bptt3's long gap: 1,500 steps of Adam at 0.01 on batches of 64 sequences of 40, clipped at 1.0,
    then the mean squared error on 1,000 new sequences. About 4 seconds on 4 threads."""
    torch.manual_seed(seed)
    model = MarkedReader(start)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(1500):
        x, y = marked_values(64, 40, generator)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    x, y = marked_values(1000, 40, torch.Generator().manual_seed(77))

    with torch.no_grad():
        return F.mse_loss(model(x), y).item()


def measure_marked(seeds: range = range(5)) -> dict[tuple[str, int], float]:
    """bptt3's long gap table, keyed by start, tanh or relu from PyTorch's start or the identity start, and seed."""
    with pinned_threads():
        return {(start, seed): marked_error(start, seed) for start in ("tanh", "relu", "identity") for seed in seeds}
