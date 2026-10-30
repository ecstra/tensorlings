"""45_state_space, a linear recurrence run as a loop and as a convolution, a diagonal state kept stable by construction, and Mamba's selective step."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

STATE_SPACE = Chapter(
    folder="45_state_space",
    title="State space models",
    readme=source_file(
        """
        # State space models

        An RNN with its tanh taken out is a linear recurrence, and a linear
        recurrence can run two ways, as a loop for generating and as a
        convolution for training. This chapter builds that model, the
        diagonal form that real models use and the rule that keeps it stable,
        then Mamba's selective step, which lets the input decide what the
        state keeps.

        ## Further reading

        - [Gu, Goel and Ré, efficiently modeling long sequences with structured state spaces (S4)](https://arxiv.org/abs/2111.00396)
        - [Gu, Gupta, Goel and Ré, diagonal state space models (S4D)](https://arxiv.org/abs/2206.11893)
        - [Gu and Dao, Mamba, linear time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752)
        - [Rush and Karamcheti, the annotated S4](https://srush.github.io/annotated-s4/)
        """,
    ),
)


def expect_tensor_near(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """A loop and a convolution add the same terms in a different order, so their roundings differ a little."""
    if isinstance(actual, Tensor) and actual.shape == expected.shape:
        if torch.allclose(actual.double(), expected.double(), atol=1e-4, rtol=1e-4):
            return

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def reference_ssm_loop(
    x: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
) -> Tensor:
    h = torch.zeros_like(B)
    outputs = []

    for value in x:
        h = A @ h + B * value
        outputs.append(C @ h)

    return torch.stack(outputs)


def reference_ssm_kernel(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    length: int,
) -> Tensor:
    entries = []
    power_times_b = B

    for _ in range(length):
        entries.append(C @ power_times_b)
        power_times_b = A @ power_times_b

    return torch.stack(entries)


def stable_matrix(
    seed: int,
    n: int,
) -> Tensor:
    """A random state matrix scaled so its powers fade, the kind training would keep."""
    A = small_tensor(seed, n, n) / 3

    return A / max(1.0, torch.linalg.matrix_norm(A, ord=2).item() * 1.25)


# ---------------------------------------------------------------------------
# ssm1
# ---------------------------------------------------------------------------

SSM1_LESSON = prose(
    """
    An RNN's step runs its state through a tanh. Take the tanh out and the
    step is linear. This is a state space model, the block behind S4 and
    Mamba.

    ```text
    h[t] = A @ h[t - 1] + B * x[t]
    y[t] = C @ h[t]
    ```

    Here each input `x[t]` is one number, and the state `h` holds n numbers,
    starting at zeros. `A` is n by n, and `B` and `C` hold n numbers each.

    Linear steps unroll. Write out the first few, and every output turns out
    to be a weighted sum of the inputs so far.

    ```text
    y[t] = (C @ B) * x[t] + (C @ A @ B) * x[t - 1] + (C @ A @ A @ B) * x[t - 2] + ...
    ```

    The weights `C @ A^j @ B`, for j = 0, 1, 2 and on, form a kernel. So `y`
    is `x` convolved with that kernel, and each output sees only the inputs
    up to its own step. With a state of one number, `A = 0.5` and `B = C = 1`,
    the kernel is 1, 0.5, 0.25, 0.125 and on, an average that fades into the
    past.

    The same model runs two ways.

    | Way | Fits |
    |---|---|
    | the loop, one step at a time | generating, where each input arrives after the last output |
    | the convolution, every output at once | training, where the whole sequence is known already |

    The loop is a chain of steps, each waiting on the one before. The
    convolution has no chain, so a GPU computes every output at once, and
    fft1 makes a long kernel cheap. An RNN can't do this, since its tanh stops
    the unrolling. For 64 sequences of 2,000 steps, the two ways here agreed
    to within 0.00001.

    To get the convolution from `F.conv1d`, flip the kernel, as fft1 did. Then
    put `len(x) - 1` zeros in front of `x`, so the first output sees `x[0]`
    alone.

    ```python
    F.pad(torch.tensor([[1.0, 2.0]]), (2, 0))    # tensor([[0., 0., 1., 2.]])
    ```

    `F.pad` takes how many zeros go on the left, then on the right.
    """,
)

SSM1_SOURCE = source_file(
    '''
    # ssm1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    A = torch.tensor([[0.5]])
    once = A @ torch.tensor([2.0])    # tensor([1.])
    left = F.pad(torch.tensor([[1.0, 2.0]]), (2, 0))    # tensor([[0., 0., 1., 2.]])


    # TODO: Run the state space model along x, of shape (length,), one step at a
    #   time from a state of zeros, and return every output, shape (length,).
    #   A has shape (n, n), and B and C have shape (n,).
    def ssm_loop(
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the first length entries of the kernel, C @ A^j @ B for
    #   j = 0 up to length - 1, without computing a matrix power from scratch
    #   each time.
    def ssm_kernel(
        A: Tensor,
        B: Tensor,
        C: Tensor,
        length: int,
    ) -> Tensor:
        ...


    # TODO: Return the same outputs as ssm_loop, as x convolved with the kernel
    #   of len(x) entries, using F.conv1d.
    def ssm_convolve(
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        ...
    ''',
)

SSM1_SOLUTION = source_file(
    '''
    # ssm1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    A = torch.tensor([[0.5]])
    once = A @ torch.tensor([2.0])
    left = F.pad(torch.tensor([[1.0, 2.0]]), (2, 0))


    def ssm_loop(
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        h = torch.zeros_like(B)
        outputs = []

        for value in x:
            h = A @ h + B * value
            outputs.append(C @ h)

        return torch.stack(outputs)


    # Each entry multiplies one more A into the last one, so no power is computed twice.
    def ssm_kernel(
        A: Tensor,
        B: Tensor,
        C: Tensor,
        length: int,
    ) -> Tensor:
        entries = []
        power_times_b = B

        for _ in range(length):
            entries.append(C @ power_times_b)
            power_times_b = A @ power_times_b

        return torch.stack(entries)


    def ssm_convolve(
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        length = len(x)
        kernel = ssm_kernel(A, B, C, length)
        padded = F.pad(x.view(1, 1, -1), (length - 1, 0))

        return F.conv1d(padded, kernel.flip(0).view(1, 1, -1)).flatten()
    ''',
)

SSM_INPUTS = (
    (small_tensor(450, 6), stable_matrix(451, 2), small_tensor(452, 2), small_tensor(453, 2)),
    (small_tensor(454, 9), stable_matrix(455, 3), small_tensor(456, 3), small_tensor(457, 3)),
)


def check_ssm1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="ssm_loop",
        description="runs the steps one at a time",
        inputs=SSM_INPUTS,
        reference=reference_ssm_loop,
        expect=expect_tensor_near,
    )
    check_on_inputs(
        checker=checker,
        function="ssm_kernel",
        description="gives C @ A^j @ B for each j",
        inputs=(
            (stable_matrix(458, 2), small_tensor(459, 2), small_tensor(460, 2), 5),
            (stable_matrix(461, 3), small_tensor(462, 3), small_tensor(463, 3), 7),
        ),
        reference=reference_ssm_kernel,
        expect=expect_tensor_near,
    )
    check_on_inputs(
        checker=checker,
        function="ssm_convolve",
        description="gives the loop's outputs as a convolution",
        inputs=SSM_INPUTS,
        reference=reference_ssm_loop,
        expect=expect_tensor_near,
    )


SSM1 = Exercise(
    name="ssm1",
    chapter=STATE_SPACE,
    title="A loop and a convolution",
    lesson=SSM1_LESSON,
    source=SSM1_SOURCE,
    solution=SSM1_SOLUTION,
    hints={
        "ssm_loop": "Start h at torch.zeros_like(B). For each value of x, set h to A @ h + B * value and keep "
        "C @ h. Return torch.stack of what you kept.",
        "ssm_kernel": "Keep a running product that starts as B. Each step, keep C @ it, then multiply A into it.",
        "ssm_convolve": "Get ssm_kernel for len(x) entries. Give x the shape (1, 1, length) and pad length - 1 "
        "zeros on the left. Then F.conv1d it with the flipped kernel as (1, 1, length), and .flatten().",
    },
    introduces=(),
    check=check_ssm1,
)


# ---------------------------------------------------------------------------
# ssm2
# ---------------------------------------------------------------------------

SSM2_LESSON = prose(
    """
    A full `A` costs n times n multiplications at every step. S4D and Mamba
    make `A` diagonal instead, so each entry of the state runs on its own.

    ```text
    h[t][i] = a[i] * h[t - 1][i] + b[i] * x[t]
    ```

    The kernel then needs no matrix at all. Its entry j is the sum over i of
    `c[i] * b[i] * a[i] ** j`.

    After j steps, each `a[i]` has been multiplied in j times. If its size is
    above 1, that grows without end and the state blows up. Below 1 it fades.
    So the model is stable when every `a[i]` lies between -1 and 1.

    | `a` | `a ** 10` | `a ** 50` |
    |---|---|---|
    | 1.1 | 2.59 | 117.39 |
    | 0.9 | 0.349 | 0.005 |

    These models don't hope training keeps every `a` below 1. They build it
    in. Each state entry has a rate of decay through time, `rate = -exp(log_rate)`
    for a learned `log_rate`, so the rate is always negative. Turning that
    rate into a step of size `step` is called discretization.

    ```text
    a     = exp(step * rate)
    b_bar = (a - 1) / rate * b
    ```

    The `b_bar` formula assumes the input holds still through the step, which
    S4D calls the zero order hold. With the rate below 0 and the step above 0,
    `a` lands between 0 and 1 whatever training does. Stable by construction.

    The step sets the memory. Here is a rate of -1.

    | `step` | `a` | Steps for the state to fade to half |
    |---|---|---|
    | 0.01 | 0.990 | 69.3 |
    | 0.1 | 0.905 | 6.9 |
    | 1 | 0.368 | 0.7 |

    A small step remembers for a long time, and a large one forgets almost at
    once. The steps to fade to half solve `a ** steps = 0.5`, which gives
    `log(0.5) / log(a)`. ssm3 lets the input choose the step.
    """,
)

SSM2_SOURCE = source_file(
    '''
    # ssm2.py

    import math

    import torch
    from torch import Tensor

    a = torch.tensor([0.5, 0.9])
    powers = a ** torch.arange(3).unsqueeze(1)    # tensor([[1.0000, 1.0000], [0.5000, 0.9000], [0.2500, 0.8100]])


    # TODO: Return the first length entries of the kernel of a diagonal state,
    #   entry j being the sum over i of c[i] * b[i] * a[i] ** j, without a loop.
    #   a, b and c have shape (n,).
    def diagonal_kernel(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        length: int,
    ) -> Tensor:
        ...


    # TODO: Return True when every entry of a is between -1 and 1, so the state
    #   fades, as a plain bool.
    def is_stable(a: Tensor) -> bool:
        ...


    # TODO: Return a and b_bar from the lesson's formulas, with the rate taken
    #   as -exp(log_rate).
    def discretize(
        log_rate: Tensor,
        b: Tensor,
        step: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return how many steps each entry of a, between 0 and 1, takes to
    #   fade to half.
    def half_life(a: Tensor) -> Tensor:
        ...
    ''',
)

SSM2_SOLUTION = source_file(
    '''
    # ssm2.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    a = torch.tensor([0.5, 0.9])
    powers = a ** torch.arange(3).unsqueeze(1)


    def diagonal_kernel(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        length: int,
    ) -> Tensor:
        powers = a ** torch.arange(length).unsqueeze(1)

        return powers @ (c * b)


    def is_stable(a: Tensor) -> bool:
        return bool((a.abs() < 1).all())


    def discretize(
        log_rate: Tensor,
        b: Tensor,
        step: float,
    ) -> tuple[Tensor, Tensor]:
        rate = -torch.exp(log_rate)
        a = torch.exp(step * rate)

        return a, (a - 1) / rate * b


    def half_life(a: Tensor) -> Tensor:
        return math.log(0.5) / torch.log(a)
    ''',
)


def reference_diagonal_kernel(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    length: int,
) -> Tensor:
    return torch.stack([(c * b * a**j).sum() for j in range(length)])


def reference_discretize(
    log_rate: Tensor,
    b: Tensor,
    step: float,
) -> tuple[Tensor, Tensor]:
    rate = -torch.exp(log_rate)
    a = torch.exp(step * rate)

    return a, (a - 1) / rate * b


def expect_pair_near(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should return two tensors, a and b_bar, as a tuple.")

    for name, part, wanted in zip(("a", "b_bar"), actual, expected):
        expect_tensor_near(
            call=f"{name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_ssm2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="diagonal_kernel",
        description="sums c * b * a ** j over the state",
        inputs=(
            (torch.tensor([0.5, -0.8, 0.9]), small_tensor(470, 3), small_tensor(471, 3), 6),
            (torch.tensor([0.95, 0.2]), small_tensor(472, 2), small_tensor(473, 2), 4),
        ),
        reference=reference_diagonal_kernel,
        expect=expect_tensor_near,
    )
    check_on_inputs(
        checker=checker,
        function="is_stable",
        description="says whether every entry fades",
        inputs=(
            (torch.tensor([0.5, -0.99, 0.0]),),
            (torch.tensor([0.5, 1.01]),),
            (torch.tensor([-1.2, 0.3]),),
            (torch.tensor([1.0]),),
        ),
        reference=lambda a: bool((a.abs() < 1).all()),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="discretize",
        description="turns a rate and a step into a and b_bar",
        inputs=(
            (torch.tensor([0.0, 1.0, -2.0]), small_tensor(474, 3), 0.1),
            (small_tensor(475, 4) / 2, small_tensor(476, 4), 1.5),
        ),
        reference=reference_discretize,
        expect=expect_pair_near,
    )
    check_on_inputs(
        checker=checker,
        function="half_life",
        description="gives the steps to fade to half",
        inputs=(
            (torch.tensor([0.99, 0.905, 0.368]),),
            (torch.tensor([0.5, 0.75]),),
        ),
        reference=lambda a: math.log(0.5) / torch.log(a),
        expect=expect_tensor_near,
    )


SSM2 = Exercise(
    name="ssm2",
    chapter=STATE_SPACE,
    title="A diagonal state that stays stable",
    lesson=SSM2_LESSON,
    source=SSM2_SOURCE,
    solution=SSM2_SOLUTION,
    hints={
        "diagonal_kernel": "a ** torch.arange(length).unsqueeze(1) has one row of powers per j. Multiply that "
        "matrix by c * b.",
        "is_stable": "a.abs() < 1 checks every entry, .all() asks whether all are True, and bool(...) makes it a "
        "plain bool.",
        "discretize": "rate = -torch.exp(log_rate), a = torch.exp(step * rate), and b_bar = (a - 1) / rate * b.",
        "half_life": "math.log(0.5) divided by torch.log(a).",
    },
    introduces=(),
    check=check_ssm2,
)


# ---------------------------------------------------------------------------
# ssm3
# ---------------------------------------------------------------------------


def marked_sequences(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sequences of (value, mark) pairs with one step marked at random, and the marked value of each."""
    values = torch.randn(count, steps, generator=generator)
    marked = torch.randint(0, steps, (count,), generator=generator)
    marks = F.one_hot(marked, steps).float()

    return torch.stack([values, marks], dim=2), values[torch.arange(count), marked]


SSM3_LESSON = prose(
    """
    ssm2's model is the same at every step. Its step, `b` and `c` are fixed
    once training ends. So it weighs each input by how long ago it came, never
    by what it is. It can't pick out one input by its content.

    Mamba fixes this with a selective step. It computes the step, `B` and `C`
    from each input, so the update itself depends on the input.

    ```text
    delta[t] = softplus(Linear(x[t]))
    a[t]     = exp(delta[t] * rate)
    h[t]     = a[t] * h[t - 1] + delta[t] * B[t] * u[t]
    y[t]     = C[t] @ h[t]
    ```

    Here `u[t]` is the value the state takes in. `softplus(z)` is
    `log(1 + exp(z))`, a smooth ReLU that is always above 0, so `delta` is a
    valid step. Mamba writes in `delta * B`, a simpler form than ssm2's
    `b_bar`.

    A large `delta` makes `a` near 0, which clears the state and writes the
    new value in. A `delta` near 0 makes `a` near 1 and writes almost nothing,
    so the state is kept. The model chooses, step by step, what to keep.

    The test is selective copying at its smallest. Each sequence holds 16
    values with one of them marked, and the model has to give back the marked
    value at the end. Here are both models after 200 steps of training.

    | Model | Mean squared error on new sequences |
    |---|---|
    | always 0 | 1.11 |
    | fixed, ssm2's kind | 1.02 |
    | selective | 0.002 |

    The fixed model ends barely better than guessing 0, since the mark can sit
    anywhere. The selective one learned to open at the mark and hold
    everywhere else, which is derive1's gate again.

    It costs ssm1's convolution. With a step that changes at every input there
    is no single kernel. So Mamba runs the loop, as a scan written to run fast
    on a GPU.
    """,
)

SSM3_SOURCE = source_file(
    '''
    # ssm3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    smooth = F.softplus(torch.tensor([-5.0, 0.0, 5.0]))    # tensor([0.0067, 0.6931, 5.0067])


    # TODO: Run the selective step along the sequences and return y, of shape
    #   (batch, steps). u and delta have shape (batch, steps), rate has shape
    #   (n,), and B and C have shape (batch, steps, n). h starts at zeros of
    #   shape (batch, n).
    def selective_scan(
        u: Tensor,
        delta: Tensor,
        rate: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. x has shape (batch, steps, 2), each step a value and
    #   its mark. Compute delta with to_delta and a softplus, and B and C with
    #   to_b and to_c, all from x. Run selective_scan on the values with the
    #   rate -exp(log_rate), and return y at the last step, shape (batch,).
    class SelectiveSSM(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.log_rate = nn.Parameter(torch.zeros(n))
            self.to_delta = nn.Linear(2, 1)
            self.to_b = nn.Linear(2, n)
            self.to_c = nn.Linear(2, n)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

SSM3_SOLUTION = source_file(
    '''
    # ssm3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    smooth = F.softplus(torch.tensor([-5.0, 0.0, 5.0]))


    def selective_scan(
        u: Tensor,
        delta: Tensor,
        rate: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        h = torch.zeros(u.shape[0], rate.shape[0])
        outputs = []

        for t in range(u.shape[1]):
            step = delta[:, t].unsqueeze(1)
            a = torch.exp(step * rate)
            h = a * h + step * B[:, t] * u[:, t].unsqueeze(1)
            outputs.append((C[:, t] * h).sum(dim=1))

        return torch.stack(outputs, dim=1)


    class SelectiveSSM(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.log_rate = nn.Parameter(torch.zeros(n))
            self.to_delta = nn.Linear(2, 1)
            self.to_b = nn.Linear(2, n)
            self.to_c = nn.Linear(2, n)

        def forward(self, x: Tensor) -> Tensor:
            delta = F.softplus(self.to_delta(x)).squeeze(2)
            y = selective_scan(x[:, :, 0], delta, -torch.exp(self.log_rate), self.to_b(x), self.to_c(x))

            return y[:, -1]
    ''',
)


def reference_selective_scan(
    u: Tensor,
    delta: Tensor,
    rate: Tensor,
    B: Tensor,
    C: Tensor,
) -> Tensor:
    h = torch.zeros(u.shape[0], rate.shape[0])
    outputs = []

    for t in range(u.shape[1]):
        step = delta[:, t].unsqueeze(1)
        h = torch.exp(step * rate) * h + step * B[:, t] * u[:, t].unsqueeze(1)
        outputs.append((C[:, t] * h).sum(dim=1))

    return torch.stack(outputs, dim=1)


class ReferenceSelectiveSSM(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.log_rate = nn.Parameter(torch.zeros(n))
        self.to_delta = nn.Linear(2, 1)
        self.to_b = nn.Linear(2, n)
        self.to_c = nn.Linear(2, n)

    def forward(self, x: Tensor) -> Tensor:
        delta = F.softplus(self.to_delta(x)).squeeze(2)
        y = reference_selective_scan(x[:, :, 0], delta, -torch.exp(self.log_rate), self.to_b(x), self.to_c(x))

        return y[:, -1]


class FixedSSM(nn.Module):
    """ssm2's kind for the lesson's table, one learned step and fixed b and c, on the same inputs."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.log_rate = nn.Parameter(torch.zeros(n))
        self.log_step = nn.Parameter(torch.zeros(1))
        self.to_b = nn.Linear(2, n, bias=False)
        self.c = nn.Parameter(torch.randn(n) / 2)

    def forward(self, x: Tensor) -> Tensor:
        step = torch.exp(self.log_step)
        a = torch.exp(step * -torch.exp(self.log_rate))
        h = torch.zeros(len(x), len(a))

        for t in range(x.shape[1]):
            h = a * h + step * self.to_b(x[:, t])

        return h @ self.c


def seeded_ssm(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make(4)


def train_ssm(model: nn.Module) -> nn.Module:
    """The run the lesson's table comes from, 200 Adam steps on batches of 64 sequences of 16."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(1)

    for _ in range(200):
        x, y = marked_sequences(64, 16, generator)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def scan_inputs(
    seed: int,
    batch: int,
    steps: int,
    n: int,
) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)

    return (
        torch.randn(batch, steps, generator=generator),
        torch.rand(batch, steps, generator=generator),
        -torch.rand(n, generator=generator) - 0.1,
        torch.randn(batch, steps, n, generator=generator),
        torch.randn(batch, steps, n, generator=generator),
    )


def check_selective_forward(found: type) -> None:
    model = expect_module("SelectiveSSM(4)", seeded_ssm(found))
    x, _ = marked_sequences(3, 5, torch.Generator().manual_seed(480))

    with torch.no_grad():
        expected = seeded_ssm(ReferenceSelectiveSSM)(x)

    expect_tensor_near(
        call="SelectiveSSM(4) on 3 sequences of 5",
        actual=model(x),
        expected=expected,
    )


def check_selective_learns(found: type) -> None:
    model = train_ssm(seeded_ssm(found))
    x, y = marked_sequences(1000, 16, torch.Generator().manual_seed(99))

    with torch.no_grad():
        error = F.mse_loss(model(x), y).item()

    if error > 0.1:
        raise Mismatch(
            f"After 200 steps of training, SelectiveSSM's mean squared error on new sequences is {error:.3f}, "
            "but it should be below 0.1.",
        )


def check_ssm3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="selective_scan",
        description="runs the selective step along the sequences",
        inputs=(
            scan_inputs(481, 2, 4, 3),
            scan_inputs(482, 3, 6, 2),
        ),
        reference=reference_selective_scan,
        expect=expect_tensor_near,
    )
    checker.test_method(
        class_name="SelectiveSSM",
        method="forward",
        description="computes the step, B and C from the input",
        run=check_selective_forward,
    )
    checker.test_method(
        class_name="SelectiveSSM",
        method="forward",
        description="learns to copy the marked value",
        run=check_selective_learns,
    )


SSM3 = Exercise(
    name="ssm3",
    chapter=STATE_SPACE,
    title="A step chosen by the input",
    lesson=SSM3_LESSON,
    source=SSM3_SOURCE,
    solution=SSM3_SOLUTION,
    hints={
        "selective_scan": "Loop over t with step = delta[:, t].unsqueeze(1). Set h to torch.exp(step * rate) * h "
        "plus step * B[:, t] * u[:, t].unsqueeze(1). Keep (C[:, t] * h).sum(dim=1), and stack along dim=1.",
        "SelectiveSSM.forward": "F.softplus(self.to_delta(x)).squeeze(2) is delta. The values are x[:, :, 0], and "
        "the last step of y is y[:, -1].",
    },
    introduces=(),
    check=check_ssm3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz8
# ---------------------------------------------------------------------------

QUIZ8_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 6. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `gru_step` | the reset gate, the update gate and the candidate state of one GRU step | gru1, gating1 |
    | `causal_dilated_conv` | zeros on the left only, then a dilated convolution | causal1, dilated1 |
    | `both_directions` | one RNN over the sequence, one over it flipped, flipped back and joined | bidir1 |
    | `diagonal_scan` | a diagonal linear recurrence run as a loop, read out at every step | ssm1, ssm2 |
    """,
)

QUIZ8_SOURCE = source_file(
    '''
    # quiz8.py
    #
    # No examples and no new tools. Each function combines moves from Section 6.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the next hidden state of a GRU without biases, for x of shape
    #   (batch, inputs) and h of shape (batch, hidden). W and U stack the reset,
    #   update and candidate weights in that order, of shapes (3 * hidden,
    #   inputs) and (3 * hidden, hidden). r = sigmoid(x W_r^T + h U_r^T), z the
    #   same with W_z and U_z, n = tanh(x W_n^T + r * (h U_n^T)), and the result
    #   is (1 - z) * n + z * h.
    def gru_step(
        x: Tensor,
        h: Tensor,
        W: Tensor,
        U: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the causal convolution of x, of shape (batch, channels,
    #   length), with weight, of shape (out channels, channels, k), at the given
    #   dilation. The output keeps the length, and no output sees a later input.
    def causal_dilated_conv(
        x: Tensor,
        weight: Tensor,
        dilation: int,
    ) -> Tensor:
        ...


    # TODO: Return both directions' outputs for x, of shape (batch, steps,
    #   features), joined along the last dimension, forward first. forward and
    #   backward are nn.RNNs with batch_first=True, and backward reads x flipped
    #   in time. Its outputs are flipped back so step t lines up with step t.
    def both_directions(
        forward: nn.RNN,
        backward: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the outputs of the recurrence h_t = a * h_(t-1) + b * x_t,
    #   y_t = the sum of c * h_t, for x of shape (batch, steps). a, b and c hold
    #   one number per state. The state starts at 0, and the result has the
    #   shape of x.
    def diagonal_scan(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

QUIZ8_SOLUTION = source_file(
    '''
    # quiz8.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def gru_step(
        x: Tensor,
        h: Tensor,
        W: Tensor,
        U: Tensor,
    ) -> Tensor:
        W_r, W_z, W_n = W.chunk(3)
        U_r, U_z, U_n = U.chunk(3)
        r = torch.sigmoid(x @ W_r.T + h @ U_r.T)
        z = torch.sigmoid(x @ W_z.T + h @ U_z.T)
        n = torch.tanh(x @ W_n.T + r * (h @ U_n.T))

        return (1 - z) * n + z * h


    def causal_dilated_conv(
        x: Tensor,
        weight: Tensor,
        dilation: int,
    ) -> Tensor:
        k = weight.shape[2]

        return F.conv1d(F.pad(x, (dilation * (k - 1), 0)), weight, dilation=dilation)


    def both_directions(
        forward: nn.RNN,
        backward: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        ahead, _ = forward(x)
        behind, _ = backward(x.flip(1))

        return torch.cat([ahead, behind.flip(1)], dim=2)


    def diagonal_scan(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        x: Tensor,
    ) -> Tensor:
        h = torch.zeros(len(x), len(a))
        outputs = []

        for t in range(x.shape[1]):
            h = a * h + b * x[:, t : t + 1]
            outputs.append((c * h).sum(dim=1))

        return torch.stack(outputs, dim=1)
    ''',
)


def reference_gru_step(
    x: Tensor,
    h: Tensor,
    W: Tensor,
    U: Tensor,
) -> Tensor:
    """nn.GRUCell with its biases at zero is the same step, so the quiz is held to PyTorch's own GRU."""
    cell = nn.GRUCell(x.shape[1], h.shape[1])

    with torch.no_grad():
        cell.weight_ih.copy_(W)
        cell.weight_hh.copy_(U)
        cell.bias_ih.zero_()
        cell.bias_hh.zero_()

        return cell(x, h)


def reference_causal_dilated_conv(
    x: Tensor,
    weight: Tensor,
    dilation: int,
) -> Tensor:
    k = weight.shape[2]

    return F.conv1d(F.pad(x, (dilation * (k - 1), 0)), weight, dilation=dilation)


def reference_both_directions(
    forward: nn.RNN,
    backward: nn.RNN,
    x: Tensor,
) -> Tensor:
    with torch.no_grad():
        ahead, _ = forward(x)
        behind, _ = backward(x.flip(1))

    return torch.cat([ahead, behind.flip(1)], dim=2)


def reference_diagonal_scan(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    x: Tensor,
) -> Tensor:
    h = torch.zeros(len(x), len(a))
    outputs = []

    for t in range(x.shape[1]):
        h = a * h + b * x[:, t : t + 1]
        outputs.append((c * h).sum(dim=1))

    return torch.stack(outputs, dim=1)


def seeded_rnn(seed: int) -> nn.RNN:
    torch.manual_seed(seed)

    return nn.RNN(3, 4, batch_first=True)


def check_quiz8(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gru_step",
        description="gives the GRU's next hidden state",
        inputs=(
            (small_tensor(880, 2, 3) / 3, small_tensor(881, 2, 4) / 3, small_tensor(882, 12, 3) / 3, small_tensor(883, 12, 4) / 3),
        ),
        reference=reference_gru_step,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="causal_dilated_conv",
        description="keeps the length and never reads ahead",
        inputs=(
            (small_tensor(884, 2, 3, 9), small_tensor(885, 4, 3, 2), 3),
            (small_tensor(886, 1, 2, 7), small_tensor(887, 2, 2, 3), 1),
        ),
        reference=reference_causal_dilated_conv,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="both_directions",
        description="joins the forward outputs with the backward ones flipped back",
        inputs=(
            (Described(seeded_rnn(888), "an nn.RNN from 3 to 4"), Described(seeded_rnn(889), "another"), small_tensor(890, 2, 5, 3) / 3),
        ),
        reference=reference_both_directions,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="diagonal_scan",
        description="runs the recurrence and reads it out at every step",
        inputs=(
            (torch.tensor([0.9, 0.5]), torch.tensor([1.0, -0.5]), torch.tensor([0.3, 2.0]), small_tensor(891, 2, 6)),
            (torch.tensor([0.2, 0.99, -0.6]), torch.tensor([0.5, 0.1, 1.0]), torch.tensor([1.0, 1.0, -1.0]), small_tensor(892, 3, 4)),
        ),
        reference=reference_diagonal_scan,
        expect=expect_tensor,
    )


QUIZ8 = Exercise(
    name="quiz8",
    chapter=QUIZZES,
    title="Sequences, start to finish",
    lesson=QUIZ8_LESSON,
    source=QUIZ8_SOURCE,
    solution=QUIZ8_SOLUTION,
    hints={
        "gru_step": "W.chunk(3) and U.chunk(3) give the reset, update and candidate weights. Then follow the "
        "formulas in the TODO in order.",
        "causal_dilated_conv": "F.pad(x, (dilation * (k - 1), 0)) adds the zeros on the left only, then F.conv1d "
        "with dilation=dilation.",
        "both_directions": "Run backward on x.flip(1), flip its outputs back with .flip(1), and torch.cat both "
        "outputs on dim=2.",
        "diagonal_scan": "Start h at torch.zeros(len(x), len(a)). Each step, h = a * h + b * x[:, t : t + 1], and "
        "keep (c * h).sum(dim=1). torch.stack the outputs on dim=1.",
    },
    introduces=(),
    check=check_quiz8,
)
