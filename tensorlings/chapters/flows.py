"""68_flows, the change of variables, RealNVP's affine coupling layer that inverts exactly, and a flow trained on two moons by maximum likelihood."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

FLOWS = Chapter(
    folder="68_flows",
    title="Normalizing flows",
    readme=source_file(
        """
        # Normalizing flows

        A flow turns noise into samples through layers that can be run
        backwards exactly, so it gives the exact likelihood of any point. This
        chapter derives the change of variables, builds RealNVP's affine
        coupling layer, and trains a flow on two moons by maximum likelihood.

        ## Further reading

        - [Dinh, Sohl-Dickstein and Bengio, density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
        - [Rezende and Mohamed, variational inference with normalizing flows](https://arxiv.org/abs/1505.05770)
        - [Papamakarios and colleagues, normalizing flows for probabilistic modeling and inference](https://arxiv.org/abs/1912.02762)
        """,
    ),
)

# The pieces later exercises are handed, written once so the files never drift apart.
NORMAL_PART = '''
def normal_log_prob(z: Tensor) -> Tensor:
    return -0.5 * (z**2).sum(dim=-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)
'''

FLOW1_PARTS = '''
def affine_log_prob(
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> Tensor:
    z = (x - shift) / scale

    return normal_log_prob(z) - torch.log(scale.abs()).sum()


def log_abs_det(
    g: Callable[[Tensor], Tensor],
    x: Tensor,
) -> Tensor:
    jacobian = torch.autograd.functional.jacobian(g, x)

    return torch.linalg.slogdet(jacobian).logabsdet


def flow_log_prob(
    g: Callable[[Tensor], Tensor],
    x: Tensor,
) -> Tensor:
    return normal_log_prob(g(x)) + log_abs_det(g, x)
'''

COUPLING_INIT = '''
class Coupling(nn.Module):
    def __init__(
        self,
        flip: bool,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.flip = flip
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    # The kept number and the moved one, the first or the second of each point by flip.
    def halves(self, x: Tensor) -> tuple[Tensor, Tensor]:
        first, second = x[:, :1], x[:, 1:]

        return (second, first) if self.flip else (first, second)

    def join(
        self,
        kept: Tensor,
        moved: Tensor,
    ) -> Tensor:
        return torch.cat([moved, kept] if self.flip else [kept, moved], dim=1)

    def scale_and_shift(self, kept: Tensor) -> tuple[Tensor, Tensor]:
        s, t = self.net(kept).chunk(2, dim=1)

        return torch.tanh(s), t
'''

COUPLING_SOLUTION = '''
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        kept, moved = self.halves(x)
        s, t = self.scale_and_shift(kept)

        return self.join(kept, moved * torch.exp(s) + t), s.sum(dim=1)

    def inverse(self, y: Tensor) -> Tensor:
        kept, moved = self.halves(y)
        s, t = self.scale_and_shift(kept)

        return self.join(kept, (moved - t) * torch.exp(-s))
'''

MOONS_PART = '''
def two_moons(
    count: int,
    noise: float,
    generator: torch.Generator,
) -> Tensor:
    upper = torch.arange(count) < count // 2
    t = math.pi * torch.rand(count, generator=generator)
    x = torch.where(upper, torch.cos(t), 1 - torch.cos(t))
    y = torch.where(upper, torch.sin(t), 0.5 - torch.sin(t))

    return torch.stack([x, y], dim=1) + noise * torch.randn(count, 2, generator=generator)
'''

FLOW_INIT = '''
class Flow(nn.Module):
    def __init__(
        self,
        layers: int = 8,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Coupling(flip=index % 2 == 1) for index in range(layers)])
'''

FLOW_SOLUTION = '''
    def log_prob(self, x: Tensor) -> Tensor:
        total = torch.zeros(len(x))

        for layer in self.layers:
            x, log_det = layer(x)
            total = total + log_det

        return normal_log_prob(x) + total

    def sample(
        self,
        count: int,
        rng: torch.Generator,
    ) -> Tensor:
        z = torch.randn(count, 2, generator=rng)

        with torch.no_grad():
            for layer in reversed(self.layers):
                z = layer.inverse(z)

        return z
'''

TRAIN_FLOW_SOLUTION = '''
def train_flow(
    flow: nn.Module,
    steps: int,
    rng: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(flow.parameters(), lr=3e-3)

    for _ in range(steps):
        batch = two_moons(256, 0.05, rng)
        loss = -flow.log_prob(batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace(*parts: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    exec("\n\n".join(parts), namespace)

    return namespace


REFERENCE = reference_namespace(
    NORMAL_PART,
    FLOW1_PARTS,
    COUPLING_INIT + COUPLING_SOLUTION,
    MOONS_PART,
    FLOW_INIT + FLOW_SOLUTION,
    TRAIN_FLOW_SOLUTION,
)


# ---------------------------------------------------------------------------
# flow1
# ---------------------------------------------------------------------------

FLOW1_LESSON = prose(
    """
    A flow makes samples the way a GAN does, by pushing noise through a
    network. Its network can also run backwards exactly, and that buys
    something a GAN lacks, the exact likelihood of any point.

    Run it backwards. A point `x` goes in, the backward map `g` gives its
    noise `z = g(x)`, and `z` is standard normal. The density of `x` follows
    from `z`'s by the change of variables.

    ```text
    log p(x) = log N(g(x)) + log |det J_g(x)|
    ```

    `J_g` is the Jacobian of `g`, the matrix of how each output moves with each
    input. Its determinant says how much `g` stretches a small area around `x`.
    Where `g` squeezes a wide region of `x` into a small one of `z`, the
    density of `x` is lower than `z`'s by that factor.

    In one dimension take `x = 3z`. The values of `x` spread 3 times as wide,
    so their density is 3 times lower. Here `g(x) = x / 3` has derivative 1/3,
    so `log p(x) = log N(x / 3) - log 3`.

    ```python
    torch.distributions.Normal(0.0, 3.0).log_prob(torch.tensor(0.0))    # tensor(-2.0176), log N(0) - log 3
    ```

    The standard normal in `d` dimensions has the log density
    `-||z||^2 / 2 - (d / 2) log(2 pi)`, summed over each point's numbers.

    For any `g`, `torch.autograd.functional.jacobian(g, x)` builds the
    Jacobian at one point, and `torch.linalg.slogdet` gives the log of its
    determinant's size as `.logabsdet`. That costs a backward pass per
    dimension and a determinant that grows with the cube of the dimension.
    flow2's layer is built so its determinant comes nearly free.
    """,
)

FLOW1_SOURCE = source_file(
    '''
    # flow1.py

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    jacobian = torch.autograd.functional.jacobian(lambda v: 2 * v, torch.ones(2))    # tensor([[2., 0.], [0., 2.]])
    size = torch.linalg.slogdet(jacobian).logabsdet                                # tensor(1.3863), log 4


    # TODO: Return the log density of a standard normal at each row of z, of
    #   shape (batch, d), as one number per row.
    def normal_log_prob(z: Tensor) -> Tensor:
        ...


    # TODO: Return the log density of each row of x, where x = scale * z + shift
    #   for each number, with z standard normal. scale and shift hold one number
    #   per dimension. Use normal_log_prob and the change of variables.
    def affine_log_prob(
        x: Tensor,
        scale: Tensor,
        shift: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the log of the size of the determinant of g's Jacobian at x,
    #   one point of shape (d,).
    def log_abs_det(
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the log density of one point x, of shape (d,), when g maps it
    #   to standard normal noise, by the change of variables.
    def flow_log_prob(
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

FLOW1_SOLUTION = source_file(
    '''
    # flow1.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    jacobian = torch.autograd.functional.jacobian(lambda v: 2 * v, torch.ones(2))
    size = torch.linalg.slogdet(jacobian).logabsdet


''' + indented(NORMAL_PART) + '''


''' + indented(FLOW1_PARTS) + '''
    ''',
)


def squeeze(v: Tensor) -> Tensor:
    return torch.stack([v[0] / 2, v[1] * 3 + v[0]])


def bend(v: Tensor) -> Tensor:
    return torch.stack([v[0] + 0.5 * torch.sin(v[1]), v[1] * torch.exp(0.3 * v[0])])


# A determinant of -2, so only the log of its size exists.
def swap(v: Tensor) -> Tensor:
    return torch.stack([v[1], 2 * v[0]])


def check_flow1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="normal_log_prob",
        description="gives one log density per row",
        inputs=(
            (small_tensor(680, 4, 2) / 2,),
            (small_tensor(681, 3, 5) / 3,),
        ),
        reference=lambda z: torch.distributions.Normal(0.0, 1.0).log_prob(z).sum(dim=-1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="affine_log_prob",
        description="matches a normal with that mean and spread",
        inputs=(
            (small_tensor(682, 4, 2), torch.tensor([3.0, 0.5]), torch.tensor([1.0, -2.0])),
            (small_tensor(683, 3, 3), torch.tensor([-2.0, 1.5, 4.0]), torch.tensor([0.0, 0.5, -1.0])),
        ),
        reference=lambda x, scale, shift: torch.distributions.Normal(shift, scale.abs()).log_prob(x).sum(dim=-1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="log_abs_det",
        description="gives the log size of the Jacobian's determinant at a point",
        inputs=(
            (Described(squeeze, "g(v) = (v0 / 2, 3 v1 + v0)"), torch.tensor([0.4, -1.0])),
            (Described(bend, "g(v) = (v0 + sin(v1) / 2, v1 exp(0.3 v0))"), torch.tensor([1.0, 2.0])),
            (Described(swap, "g(v) = (v1, 2 v0)"), torch.tensor([0.5, -1.5])),
        ),
        reference=REFERENCE["log_abs_det"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="flow_log_prob",
        description="adds the log determinant to the noise's log density",
        inputs=(
            (Described(squeeze, "g(v) = (v0 / 2, 3 v1 + v0)"), torch.tensor([0.4, -1.0])),
            (Described(bend, "g(v) = (v0 + sin(v1) / 2, v1 exp(0.3 v0))"), torch.tensor([1.0, 2.0])),
            (Described(swap, "g(v) = (v1, 2 v0)"), torch.tensor([0.5, -1.5])),
        ),
        reference=REFERENCE["flow_log_prob"],
        expect=expect_tensor,
    )


FLOW1 = Exercise(
    name="flow1",
    chapter=FLOWS,
    title="The change of variables",
    lesson=FLOW1_LESSON,
    source=FLOW1_SOURCE,
    solution=FLOW1_SOLUTION,
    hints={
        "normal_log_prob": "Sum z ** 2 over the last dimension and halve it with a minus sign. Then take away "
        "z.shape[-1] / 2 times math.log(2 * math.pi).",
        "affine_log_prob": "z = (x - shift) / scale. Then normal_log_prob(z), minus the sum of the logs of the "
        "scale's sizes, torch.log(scale.abs()).sum().",
        "log_abs_det": "torch.autograd.functional.jacobian(g, x) gives the matrix, and torch.linalg.slogdet of it has "
        ".logabsdet.",
        "flow_log_prob": "normal_log_prob of g(x), plus log_abs_det(g, x).",
    },
    introduces=(),
    check=check_flow1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# flow2
# ---------------------------------------------------------------------------

FLOW2_LESSON = prose(
    """
    A determinant through `slogdet` is slow and a general network has no
    inverse. RealNVP, from Dinh, Sohl-Dickstein and Bengio, makes a layer with
    both for free. Split each point in two. One part passes through unchanged,
    and a small network reads it to scale and shift the other.

    ```text
    y1 = x1
    y2 = x2 * exp(s(x1)) + t(x1)
    ```

    `y1` ignores `x2`, so the Jacobian is triangular, and its determinant is
    the product of its diagonal. That is 1 for the kept part and `exp(s)` for
    each scaled number, so the log determinant is the sum of `s`.

    Inverting never inverts the network. `x1 = y1`, so the network reads the
    same input and gives the same `s` and `t`, and `x2 = (y2 - t) * exp(-s)`.
    The network can be anything, since it only ever runs forwards.

    The points here have 2 numbers, so each half is one number. The network
    turns the kept number into `s` and `t`. A tanh keeps `s` between -1 and 1,
    so one layer scales by at most e. A layer never changes its kept half, so
    a flow stacks layers that take turns, by `flip`.

    `.chunk(2, dim=1)` splits a tensor into two along its columns.

    On 1,000 points here, a layer's inverse gave back its input within
    0.000002, the rounding of 32 bit numbers. For one point, the sum of `s`
    gave a log determinant of 0.9089, and flow1's `slogdet` of the full
    Jacobian gave 0.9089 too.
    """,
)

FLOW2_SOURCE = source_file(
    '''
    # flow2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    s, t = torch.tensor([[0.5, 2.0]]).chunk(2, dim=1)    # tensor([[0.5]]) and tensor([[2.]])


    # TODO: Finish forward and inverse. forward takes points x of shape
    #   (batch, 2) and returns y and the log determinant of each point, of shape
    #   (batch,). The kept half passes through, and the moved half becomes
    #   moved * exp(s) + t, with s and t from scale_and_shift of the kept half.
    #   inverse undoes forward exactly.
''' + indented(COUPLING_INIT) + '''

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            ...

        def inverse(self, y: Tensor) -> Tensor:
            ...
    ''',
)

FLOW2_SOLUTION = source_file(
    '''
    # flow2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    s, t = torch.tensor([[0.5, 2.0]]).chunk(2, dim=1)


''' + indented(COUPLING_INIT) + '''

''' + indented(COUPLING_SOLUTION) + '''
    ''',
)


def seeded_coupling(
    make: type,
    flip: bool,
) -> nn.Module:
    torch.manual_seed(690)
    layer = make(flip)

    # A fresh network gives s and t near 0, which hides a missing exp or a swapped half, so they start larger.
    with torch.no_grad():
        for parameter in layer.net.parameters():
            parameter.mul_(3)

    return layer


def points_for_check(seed: int) -> Tensor:
    return torch.randn(6, 2, generator=torch.Generator().manual_seed(seed))


def check_coupling_forward(found: type) -> None:
    for flip in (False, True):
        layer = expect_module(f"Coupling({flip})", seeded_coupling(found, flip))
        reference = seeded_coupling(REFERENCE["Coupling"], flip)
        x = points_for_check(691)
        result = layer(x)

        if not isinstance(result, tuple) or len(result) != 2:
            raise Mismatch(f"Coupling({flip}).forward should return y and the log determinants, as a tuple of two.")

        with torch.no_grad():
            expected_y, expected_log_det = reference(x)

        expect_tensor(
            call=f"y from Coupling({flip}) on 6 points",
            actual=result[0],
            expected=expected_y,
        )
        expect_tensor(
            call=f"the log determinants from Coupling({flip}) on 6 points",
            actual=result[1],
            expected=expected_log_det,
        )


def check_coupling_inverse(found: type) -> None:
    for flip in (False, True):
        layer = expect_module(f"Coupling({flip})", seeded_coupling(found, flip))
        reference = seeded_coupling(REFERENCE["Coupling"], flip)
        y = points_for_check(692) * 2

        with torch.no_grad():
            expected = reference.inverse(y)

        expect_tensor(
            call=f"Coupling({flip}).inverse on 6 points",
            actual=layer.inverse(y),
            expected=expected,
        )


def check_flow2(checker: Checker) -> None:
    checker.test_method(
        class_name="Coupling",
        method="forward",
        description="scales and shifts the moved half, with the log determinant",
        run=check_coupling_forward,
    )
    checker.test_method(
        class_name="Coupling",
        method="inverse",
        description="undoes forward exactly",
        run=check_coupling_inverse,
    )


FLOW2 = Exercise(
    name="flow2",
    chapter=FLOWS,
    title="A layer that inverts exactly",
    lesson=FLOW2_LESSON,
    source=FLOW2_SOURCE,
    solution=FLOW2_SOLUTION,
    hints={
        "Coupling.forward": "kept, moved = self.halves(x). s, t = self.scale_and_shift(kept). Join kept with "
        "moved * torch.exp(s) + t, and return that with s summed over dim=1.",
        "Coupling.inverse": "Split y the same way. The kept half gives the same s and t, so the moved half is "
        "(moved - t) * torch.exp(-s). Join them back.",
    },
    introduces=(),
    check=check_flow2,
)


# ---------------------------------------------------------------------------
# flow3
# ---------------------------------------------------------------------------

FLOW3_LESSON = prose(
    """
    A flow of 8 coupling layers, taking turns, learns two moons, two curved
    arcs of points. Its training is maximum likelihood, as in likelihood3.
    The loss is the mean of `-log p(x)` over a batch, and the change of
    variables gives `log p(x)` exactly.

    `log_prob` runs a batch forwards through every layer and adds up their
    log determinants. The points come out as noise, and the standard normal's
    log density of that noise joins the sum. `sample` goes the other way. It
    draws noise and runs it backwards, through the layers in reverse order,
    each one's `inverse`.

    Here it is against the best single normal, fitted with the data's own
    mean and covariance. The covariance holds each number's spread and how
    the two numbers vary together. A sample counts as on the moons within
    0.15 of the arcs.

    | Model | Mean log density of new points | Samples on the moons |
    |---|---|---|
    | The best single normal | -1.89 | 35.9% |
    | Flow, 250 steps | -0.59 | 89.4% |
    | Flow, 1,000 steps | -0.38 | 97.7% |
    | Flow, 2,000 steps | -0.38 | 98.5% |

    The normal puts nearly two thirds of its samples off the arcs. The flow
    bends space so that the round cloud of noise lands on both of them. Its
    log density stops improving after about 1,000 steps here, while its
    samples keep sharpening a little.

    `reversed(layers)` walks a list from its end.

    ```python
    list(reversed([1, 2, 3]))    # [3, 2, 1]
    ```
    """,
)

FLOW3_SOURCE = source_file(
    '''
    # flow3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    backwards = list(reversed([1, 2, 3]))    # [3, 2, 1]


    # normal_log_prob from flow1, Coupling from flow2, and the two moons.
''' + indented(NORMAL_PART) + '''


''' + indented(COUPLING_INIT) + '''

''' + indented(COUPLING_SOLUTION) + '''


''' + indented(MOONS_PART) + '''


    # TODO: Finish log_prob and sample. log_prob gives the log density of each
    #   row of x, shape (batch,), running x through every layer in order. sample
    #   draws torch.randn(count, 2, generator=rng) and runs it through every
    #   layer's inverse in reverse order, without recording for autograd.
''' + indented(FLOW_INIT) + '''

        def log_prob(self, x: Tensor) -> Tensor:
            ...

        def sample(
            self,
            count: int,
            rng: torch.Generator,
        ) -> Tensor:
            ...


    # TODO: Train flow for steps steps with Adam at a rate of 3e-3, each on
    #   two_moons(256, 0.05, rng), minimizing the mean of -log p. Return the last
    #   loss as a plain number.
    def train_flow(
        flow: nn.Module,
        steps: int,
        rng: torch.Generator,
    ) -> float:
        ...
    ''',
)

FLOW3_SOLUTION = source_file(
    '''
    # flow3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    backwards = list(reversed([1, 2, 3]))


''' + indented(NORMAL_PART) + '''


''' + indented(COUPLING_INIT) + '''

''' + indented(COUPLING_SOLUTION) + '''


''' + indented(MOONS_PART) + '''


''' + indented(FLOW_INIT) + '''

''' + indented(FLOW_SOLUTION) + '''


''' + indented(TRAIN_FLOW_SOLUTION) + '''
    ''',
)


def seeded_flow(
    make: type,
    layers: int = 8,
) -> nn.Module:
    torch.manual_seed(693)
    flow = make(layers)

    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.mul_(2)

    return flow


def held_out_log_prob(flow: nn.Module) -> float:
    test = REFERENCE["two_moons"](2000, 0.05, torch.Generator().manual_seed(1))

    with torch.no_grad():
        return flow.log_prob(test).mean().item()


def share_on_moons(samples: Tensor) -> float:
    curve = REFERENCE["two_moons"](4000, 0.0, torch.Generator().manual_seed(3))

    return (torch.cdist(samples, curve).min(dim=1).values < 0.15).float().mean().item()


def check_flow_log_prob(found: type) -> None:
    flow = expect_module("Flow(4)", seeded_flow(found, 4))
    x = REFERENCE["two_moons"](6, 0.05, torch.Generator().manual_seed(694))

    with torch.no_grad():
        expected = seeded_flow(REFERENCE["Flow"], 4).log_prob(x)

    expect_tensor(
        call="Flow(4).log_prob of 6 points",
        actual=flow.log_prob(x),
        expected=expected,
    )


def check_flow_sample(found: type) -> None:
    flow = expect_module("Flow(4)", seeded_flow(found, 4))
    actual = flow.sample(5, torch.Generator().manual_seed(695))
    expected = seeded_flow(REFERENCE["Flow"], 4).sample(5, torch.Generator().manual_seed(695))
    expect_tensor(
        call="Flow(4).sample(5, a generator seeded with 695)",
        actual=actual,
        expected=expected,
    )

    if actual.requires_grad:
        raise Mismatch("sample's points are still recorded for autograd. Run the layers inside torch.no_grad().")


def check_train_flow(found: object) -> None:
    last_losses: list[object] = []

    for train in (found, REFERENCE["train_flow"]):
        torch.manual_seed(0)
        last_losses.append(train(REFERENCE["Flow"](8), 30, torch.Generator().manual_seed(0)))

    expect_close(
        call="the last loss from train_flow(Flow(8), 30, a generator seeded with 0)",
        actual=last_losses[0],
        expected=last_losses[1],
        tolerance=1e-4,
    )

    torch.manual_seed(0)
    flow = REFERENCE["Flow"](8)
    found(flow, 250, torch.Generator().manual_seed(0))
    log_prob = held_out_log_prob(flow)

    if log_prob < -1.3:
        raise Mismatch(
            f"After 250 steps of train_flow, the mean log density of new moon points is {log_prob:.3f}, but it should "
            "pass -1.3. The best single normal gets -1.89.",
        )


def check_flow3(checker: Checker) -> None:
    checker.test_method(
        class_name="Flow",
        method="log_prob",
        description="adds every layer's log determinant to the noise's log density",
        run=check_flow_log_prob,
    )
    checker.test_method(
        class_name="Flow",
        method="sample",
        description="runs noise backwards through the layers in reverse",
        run=check_flow_sample,
    )
    checker.test(
        function="train_flow",
        description="learns the moons by maximum likelihood",
        run=check_train_flow,
    )


FLOW3 = Exercise(
    name="flow3",
    chapter=FLOWS,
    title="A flow on two moons",
    lesson=FLOW3_LESSON,
    source=FLOW3_SOURCE,
    solution=FLOW3_SOLUTION,
    hints={
        "Flow.log_prob": "Start a total at torch.zeros(len(x)). Each layer gives x and its log determinants, which "
        "join the total. Return normal_log_prob of the last x plus the total.",
        "Flow.sample": "Draw the noise, then inside torch.no_grad() loop over reversed(self.layers), setting z to "
        "layer.inverse(z).",
        "train_flow": "Make the optimizer once. Each step, the loss is minus flow.log_prob(batch).mean(), then "
        "zero_grad, backward and step.",
    },
    introduces=(),
    check=check_flow3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
