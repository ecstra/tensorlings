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
from .causal_conv import median_seconds
from .linear_layers import expect_module
from .tensors import pinned_threads

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
    up to its own step. Take a state of one number, `A = 0.5` and `B = C = 1`.
    The kernel is 1, 0.5, 0.25, 0.125 and on, a weighted sum that fades into
    the past.

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

    On the CPU, the loop run on all 64 at once took 21 ms, and `F.conv1d`
    with the 2,000 entry kernel took 38 ms. The same convolution through the
    FFT, as fft1 did it, took 1.5 ms, 14 times quicker than the loop.

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
    #   j = 0 up to length - 1. Each A^j @ B is A @ the one before it, so
    #   keep it from one entry to the next.
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


def check_convolve_without_loop(found: object) -> None:
    """The file's ssm_loop is swapped for one that notes it was called, since an ssm_convolve that just runs the
    loop matches it without convolving at all."""
    namespace = found.__globals__
    original = namespace.get("ssm_loop")
    calls: list[int] = []

    def noted_loop(
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        calls.append(len(x))

        return reference_ssm_loop(x, A, B, C)

    namespace["ssm_loop"] = noted_loop

    try:
        x, A, B, C = SSM_INPUTS[0]
        found(x.clone(), A.clone(), B.clone(), C.clone())
    finally:
        namespace["ssm_loop"] = original

    if calls:
        raise Mismatch(
            "ssm_convolve called ssm_loop. Build its outputs from ssm_kernel with F.conv1d instead, so the two ways "
            "are computed apart and can be compared.",
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
    checker.test(
        function="ssm_convolve",
        description="convolves, without running the loop",
        run=check_convolve_without_loop,
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
    for a learned `log_rate`, so the rate is always negative. Decaying at that
    rate for a time `step` multiplies the state by `exp(step * rate)`. Turning
    the rate into a step like this is called discretization.

    ```text
    a     = exp(step * rate)
    b_bar = (a - 1) / rate * b
    ```

    The `b_bar` formula assumes the input holds still through the step, which
    S4D calls the zero order hold. With the rate below 0 and the step above 0,
    `a` lands between 0 and 1 whatever training does. Stable by construction.

    A state of 4 learns a running sum here, which an `a` of exactly 1 keeps,
    so training pushes every `a` toward 1. Each kind trains 400 steps on
    sequences of 1,000, then runs 20,000.

    | `a` learned | Largest `a`, seeds 0, 1, 2 | Loss after training | Largest output |
    |---|---|---|---|
    | as it is | 1.032, 0.752, 1.037 | 3.6e25, 0.24, 9.8e28 | inf, 0.92, inf |
    | as `exp(rate)`, a step of 1 | 0.9998, 0.9995, 0.9997 | 0.0014, 0.0002, 0.0016 | 3.3, 2.9, 3.3 |

    Two plain runs passed 1 and blew up, and one stalled. The built in form
    neared 1 every time.

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
    #   entry j being the sum over i of c[i] * b[i] * a[i] ** j. a, b and c
    #   have shape (n,). The powers line above gives every a[i] ** j at once.
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
    | `input_gated_scan` | a sigmoid gate and a tanh value from each step's input, blended into the state, every state kept | gating1, derive1, rnn2 |
    | `gated_causal_conv` | zeros on the left only, a dilated convolution, then half the channels gating the other half | causal1, dilated1, gating1 |
    | `bidirectional_summary` | the forward output at the last step joined with the backward output at the first | bidir1, rnn2 |
    | `diagonal_ssm_conv` | a diagonal state discretized and its kernel built, then a causal convolution with the kernel flipped | ssm2, ssm1 |
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


    # TODO: Return the hidden state after every step of x, of shape (batch,
    #   steps, inputs), as one tensor of shape (batch, steps, hidden). W_g and
    #   W_v have shape (hidden, inputs), and there are no biases. The state h
    #   starts at zeros. At each step, with x_t that step's inputs, the gate is
    #   g = sigmoid(x_t W_g^T) and the new value is v = tanh(x_t W_v^T). Then
    #   h becomes g * v + (1 - g) * h.
    def input_gated_scan(
        x: Tensor,
        W_g: Tensor,
        W_v: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the gated causal convolution of x, of shape (batch, channels,
    #   length), with weight, of shape (2 * out, channels, k), at the given
    #   dilation. Causal means no output sees a later input. Add dilation *
    #   (k - 1) zeros on the left of x only, then convolve at that dilation.
    #   Split the result's channels into two halves, and return the first half
    #   times the sigmoid of the second, of shape (batch, out, length).
    def gated_causal_conv(
        x: Tensor,
        weight: Tensor,
        dilation: int,
    ) -> Tensor:
        ...


    # TODO: Return one summary row per sequence of x, of shape (batch, steps,
    #   features), from rnn, an nn.RNN with bidirectional=True and
    #   batch_first=True. Join the forward direction's output at the last step
    #   with the backward direction's output at the first step, forward first.
    #   The result has shape (batch, 2 * hidden).
    def bidirectional_summary(
        rnn: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return what a diagonal state space model gives for x, of shape
    #   (length,), worked out as a convolution. Discretize first, with rate =
    #   -exp(log_rate), a = exp(step * rate), and b becoming (a - 1) / rate * b.
    #   Entry j of the kernel is the sum of c * b * a ** j, for j from 0 to
    #   len(x) - 1. Convolve causally, with len(x) - 1 zeros on the left of x
    #   and the kernel flipped. The result has shape (length,).
    def diagonal_ssm_conv(
        x: Tensor,
        log_rate: Tensor,
        b: Tensor,
        c: Tensor,
        step: float,
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


    def input_gated_scan(
        x: Tensor,
        W_g: Tensor,
        W_v: Tensor,
    ) -> Tensor:
        h = torch.zeros(len(x), len(W_g))
        states = []

        for t in range(x.shape[1]):
            g = torch.sigmoid(x[:, t] @ W_g.T)
            v = torch.tanh(x[:, t] @ W_v.T)
            h = g * v + (1 - g) * h
            states.append(h)

        return torch.stack(states, dim=1)


    def gated_causal_conv(
        x: Tensor,
        weight: Tensor,
        dilation: int,
    ) -> Tensor:
        k = weight.shape[2]
        both = F.conv1d(F.pad(x, (dilation * (k - 1), 0)), weight, dilation=dilation)
        first, second = both.chunk(2, dim=1)

        return first * torch.sigmoid(second)


    def bidirectional_summary(
        rnn: nn.RNN,
        x: Tensor,
    ) -> Tensor:
        outputs, _ = rnn(x)
        hidden = outputs.shape[2] // 2

        return torch.cat([outputs[:, -1, :hidden], outputs[:, 0, hidden:]], dim=1)


    def diagonal_ssm_conv(
        x: Tensor,
        log_rate: Tensor,
        b: Tensor,
        c: Tensor,
        step: float,
    ) -> Tensor:
        rate = -torch.exp(log_rate)
        a = torch.exp(step * rate)
        b_bar = (a - 1) / rate * b
        kernel = (a ** torch.arange(len(x)).unsqueeze(1)) @ (c * b_bar)
        padded = F.pad(x.view(1, 1, -1), (len(x) - 1, 0))

        return F.conv1d(padded, kernel.flip(0).view(1, 1, -1)).flatten()
    ''',
)


def reference_input_gated_scan(
    x: Tensor,
    W_g: Tensor,
    W_v: Tensor,
) -> Tensor:
    gates = torch.sigmoid(x @ W_g.T)
    values = torch.tanh(x @ W_v.T)
    h = torch.zeros(x.shape[0], W_g.shape[0])
    states = []

    for gate, value in zip(gates.unbind(1), values.unbind(1)):
        h = h + gate * (value - h)
        states.append(h)

    return torch.stack(states, dim=1)


def reference_gated_causal_conv(
    x: Tensor,
    weight: Tensor,
    dilation: int,
) -> Tensor:
    """Padding both ends and keeping the first length outputs is the same as padding the left alone."""
    reach = dilation * (weight.shape[2] - 1)
    out = weight.shape[0] // 2
    both = F.conv1d(x, weight, padding=reach, dilation=dilation)[..., : x.shape[-1]]

    return both[:, :out] * torch.sigmoid(both[:, out:])


def reference_bidirectional_summary(
    rnn: nn.RNN,
    x: Tensor,
) -> Tensor:
    """With one layer, the last states nn.RNN returns are the forward state after the last step and the backward
    state after reading back to the first."""
    with torch.no_grad():
        _, last = rnn(x)

    return torch.cat([last[0], last[1]], dim=1)


def reference_diagonal_ssm_conv(
    x: Tensor,
    log_rate: Tensor,
    b: Tensor,
    c: Tensor,
    step: float,
) -> Tensor:
    a, b_bar = reference_discretize(log_rate, b, step)
    h = torch.zeros_like(a)
    outputs = []

    for value in x:
        h = a * h + b_bar * value
        outputs.append((c * h).sum())

    return torch.stack(outputs)


def seeded_bidirectional(
    seed: int,
    features: int,
    hidden: int,
) -> Described:
    torch.manual_seed(seed)

    return Described(
        value=nn.RNN(features, hidden, batch_first=True, bidirectional=True),
        text=f"nn.RNN({features}, {hidden}, batch_first=True, bidirectional=True)",
    )


def check_quiz8(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="input_gated_scan",
        description="blends each step's gated value into the state and keeps every state",
        inputs=(
            (small_tensor(880, 2, 5, 3) / 3, small_tensor(881, 4, 3) / 2, small_tensor(882, 4, 3) / 2),
            (small_tensor(883, 3, 4, 2) / 3, small_tensor(884, 5, 2) / 2, small_tensor(885, 5, 2) / 2),
        ),
        reference=reference_input_gated_scan,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="gated_causal_conv",
        description="never reads ahead, and gates the first half of the channels with the second",
        inputs=(
            (small_tensor(886, 2, 3, 9), small_tensor(887, 4, 3, 2) / 2, 3),
            (small_tensor(888, 1, 2, 7), small_tensor(889, 6, 2, 3) / 2, 1),
            (small_tensor(890, 2, 2, 10), small_tensor(891, 2, 2, 3) / 2, 2),
        ),
        reference=reference_gated_causal_conv,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="bidirectional_summary",
        description="joins the forward output at the last step with the backward output at the first",
        inputs=(
            (seeded_bidirectional(892, 3, 4), small_tensor(893, 2, 5, 3) / 3),
            (seeded_bidirectional(894, 2, 3), small_tensor(895, 3, 4, 2) / 3),
        ),
        reference=reference_bidirectional_summary,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="diagonal_ssm_conv",
        description="discretizes, builds the kernel and convolves causally",
        inputs=(
            (small_tensor(896, 6), torch.tensor([0.0, -1.0]), torch.tensor([1.0, -0.5]), torch.tensor([0.3, 2.0]), 0.5),
            (small_tensor(897, 8), torch.tensor([-0.5, 0.5, -2.0]), torch.tensor([0.5, 1.0, -1.0]), torch.tensor([1.0, 1.0, -1.0]), 0.1),
        ),
        reference=reference_diagonal_ssm_conv,
        expect=expect_tensor_near,
    )


QUIZ8 = Exercise(
    name="quiz8",
    chapter=QUIZZES,
    title="Sequences, start to finish",
    lesson=QUIZ8_LESSON,
    source=QUIZ8_SOURCE,
    solution=QUIZ8_SOLUTION,
    hints={
        "input_gated_scan": "Start h as zeros of shape (batch, hidden). Step t's inputs are x[:, t]. Keep each new "
        "h in a list and stack the list on dim 1, as rnn2's loop does.",
        "gated_causal_conv": "k is weight.shape[2]. F.pad takes the zeros for the left and the right as a pair, and "
        "F.conv1d takes dilation= as nn.Conv1d does. Splitting the channels is .chunk(2, dim=1), as gating1's glu "
        "does.",
        "bidirectional_summary": "rnn(x) gives the outputs and the last states. Each step's output holds the "
        "forward state in its first hidden features and the backward state in the rest. Take step -1 of one and "
        "step 0 of the other.",
        "diagonal_ssm_conv": "Discretize as ssm2's discretize does, and build the kernel as its diagonal_kernel "
        "does. Then convolve as ssm1's ssm_convolve does, with x and the flipped kernel each shaped (1, 1, length).",
    },
    introduces=(),
    check=check_quiz8,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_two_ways(repeats: int = 5) -> dict[str, float]:
    """ssm1's numbers: 64 sequences of 2,000 steps through a state of 4. The largest gap between ssm1's loop,
    one sequence at a time, and the convolution, then the median seconds of the loop run on all 64 sequences at
    once, of F.conv1d, and of the same convolution through the FFT as fft1 did it."""
    generator = torch.Generator().manual_seed(0)
    A = stable_matrix(7, 4)
    B = torch.randn(4, generator=generator)
    C = torch.randn(4, generator=generator)
    xs = torch.randn(64, 2000, generator=generator)
    kernel = reference_ssm_kernel(A, B, C, 2000)
    size = 2 ** (2 * 2000 - 2).bit_length()

    def loop_all() -> Tensor:
        h = torch.zeros(64, 4)
        outputs: list[Tensor] = []

        for t in range(xs.shape[1]):
            h = h @ A.T + B * xs[:, t : t + 1]
            outputs.append(h @ C)

        return torch.stack(outputs, dim=1)

    def conv_all() -> Tensor:
        return F.conv1d(F.pad(xs.unsqueeze(1), (1999, 0)), kernel.flip(0).view(1, 1, -1)).squeeze(1)

    def fft_all() -> Tensor:
        spectrum = torch.fft.rfft(xs, n=size) * torch.fft.rfft(kernel, n=size)

        return torch.fft.irfft(spectrum, n=size)[:, :2000]

    with pinned_threads():
        one_at_a_time = torch.stack([reference_ssm_loop(x, A, B, C) for x in xs])

        return {
            "largest gap to F.conv1d": (one_at_a_time - conv_all()).abs().max().item(),
            "largest gap to the FFT": (one_at_a_time - fft_all()).abs().max().item(),
            "loop seconds": median_seconds(loop_all, repeats),
            "F.conv1d seconds": median_seconds(conv_all, repeats),
            "FFT seconds": median_seconds(fft_all, repeats),
        }


def running_sum_outputs(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    x: Tensor,
) -> Tensor:
    """A diagonal state run over x as a convolution through the FFT, with the kernel of ssm2."""
    steps = x.shape[1]
    kernel = (c * b * a ** torch.arange(steps, dtype=x.dtype).unsqueeze(1)).sum(dim=1)
    size = 2 ** (2 * steps - 1).bit_length()

    return torch.fft.irfft(torch.fft.rfft(x, n=size) * torch.fft.rfft(kernel, n=size), n=size)[:, :steps]


def measure_stability(
    seeds: range = range(3),
    lr: float = 0.05,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    """ssm2's second table: a diagonal state of 4 learns 0.03 times the running sum of its input, which an a of
    exactly 1 keeps, so training pushes every a toward 1. Each a starts between 0.90 and 0.99 and is learned
    either as it is or as exp(step * rate) with a step of 1, for 400 Adam steps on 16 sequences of 1,000. Then
    the trained state runs one step at a time for 20,000 steps. Per kind and seed, the loss after training, the
    largest a, and the largest output it wrote."""
    results: dict[tuple[str, int], tuple[float, float, float]] = {}

    with pinned_threads():
        for kind in ("as it is", "built in"):
            for seed in seeds:
                torch.manual_seed(seed)
                start = 0.9 + 0.09 * torch.rand(4)
                learned = nn.Parameter(start.clone() if kind == "as it is" else torch.log(-torch.log(start)))
                b = nn.Parameter(torch.randn(4) / 2)
                c = nn.Parameter(torch.randn(4) / 2)
                optimizer = torch.optim.Adam([learned, b, c], lr=lr)
                generator = torch.Generator().manual_seed(seed + 1)

                def current_a() -> Tensor:
                    return learned if kind == "as it is" else torch.exp(-torch.exp(learned))

                for _ in range(400):
                    x = torch.randn(16, 1000, generator=generator)
                    loss = F.mse_loss(running_sum_outputs(current_a(), b, c, x), 0.03 * x.cumsum(dim=1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    a = current_a()
                    h = torch.zeros(4)
                    largest = 0.0

                    for value in torch.randn(20000, generator=generator).tolist():
                        h = a * h + b * value
                        largest = max(largest, abs((c * h).sum().item()))

                        if not math.isfinite(largest):
                            break

                results[(kind, seed)] = (loss.item(), a.max().item(), largest)

    return results


def measure_selective(seeds: range = range(5)) -> dict[int, dict[str, float]]:
    """ssm3's table, seed 0 in the lesson: each model trained for 200 Adam steps at 0.05 on batches of 64
    sequences of 16, then its mean squared error on 1,000 new sequences, next to always guessing 0."""
    x, y = marked_sequences(1000, 16, torch.Generator().manual_seed(99))
    results: dict[int, dict[str, float]] = {}

    with pinned_threads():
        for seed in seeds:
            results[seed] = {"always 0": F.mse_loss(torch.zeros_like(y), y).item()}

            for name, make in (("fixed", FixedSSM), ("selective", ReferenceSelectiveSSM)):
                torch.manual_seed(seed)
                model = make(4)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                generator = torch.Generator().manual_seed(seed + 1)

                for _ in range(200):
                    batch, targets = marked_sequences(64, 16, generator)
                    loss = F.mse_loss(model(batch), targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    results[seed][name] = F.mse_loss(model(x), y).item()

    return results
