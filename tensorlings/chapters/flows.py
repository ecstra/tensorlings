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
from .classic_nets import pinned_threads
from .gan import REFERENCE as GAN_PARTS
from .linear_layers import expect_module

FLOWS = Chapter(
    folder="68_flows",
    title="Normalizing flows",
    readme=source_file(
        """
        # Normalizing flows

        A normalizing flow, or flow, turns noise into samples through layers
        that can be run backwards exactly. Because of that, it can compute the
        exact likelihood of any point. This chapter derives the change of
        variables formula, builds RealNVP's affine coupling layer, and trains
        a flow on a dataset called two moons by maximum likelihood.

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

    # Splits each point into the kept and the moved number. flip says which is kept.
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
    A flow makes samples the same way a GAN does, by passing noise through a
    network. But a flow's network can also be run backwards exactly. That
    gives a flow something a GAN does not have, the exact likelihood of any
    point.

    Think of the network run backwards as a function `g`. A data point `x`
    goes in, and `g` returns the noise that produces it, `z = g(x)`. That
    noise follows a standard normal distribution. The **change of variables**
    formula then gives the density of `x` from the density of `z`.

    ```text
    log p(x) = log N(g(x)) + log |det J_g(x)|
    ```

    `N` is the density of the standard normal. `J_g` is the Jacobian of `g`,
    the matrix of partial derivatives of each output of `g` with respect to
    each input. The absolute value of its determinant says how much `g`
    stretches or shrinks a small area around `x`. Suppose `g` squeezes a wide
    region of `x` into a small region of `z`. The same probability is then
    spread over the wider region, so the density of `x` is lower than the
    density of `z` by that factor.

    For `g(v) = 2 * v` in 2 dimensions, the Jacobian is `[[2, 0], [0, 2]]`.
    When a matrix has zeros off the diagonal, its determinant is the product
    of the diagonal entries, here 4. So `g` turns every small square into one
    with 4 times the area.

    In one dimension, take `x = 3 * z`. The values of `x` spread over a range
    3 times as wide as `z`, so their density is 3 times lower. Here
    `g(x) = x / 3`, whose derivative is 1/3. So
    `log p(x) = log N(x / 3) + log(1/3)`, which is `log N(x / 3) - log 3`.

    ```python
    torch.distributions.Normal(0.0, 3.0).log_prob(torch.tensor(0.0))    # tensor(-2.0176), log N(0) - log 3
    ```

    In `d` dimensions, the log density of the standard normal at a point `z`
    is `-||z||^2 / 2 - (d / 2) * log(2 * pi)`. Here `||z||^2` is the sum of
    the squares of the point's `d` numbers.

    For any `g`, `torch.autograd.functional.jacobian(g, x)` computes the
    Jacobian at one point `x`. `torch.linalg.slogdet` of that matrix returns
    the log of the absolute value of its determinant as `.logabsdet`. This is
    expensive. It takes one backward pass per dimension, and the time to
    compute the determinant grows with the cube of the dimension. flow2's
    layer is designed so that its determinant costs almost nothing.
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


    # TODO: Return the log density of the standard normal at each row of z,
    #   which has shape (batch, d). Return one number per row, so the result
    #   has shape (batch,).
    #   Example: normal_log_prob(torch.zeros(1, 2)) returns about
    #   tensor([-1.8379]).
    def normal_log_prob(z: Tensor) -> Tensor:
        ...


    # TODO: Return the log density of each row of x, where x = scale * z + shift
    #   and z is standard normal. scale and shift hold one number per
    #   dimension, and the formula applies to each number of a row. Use
    #   normal_log_prob and the change of variables.
    #   Example: affine_log_prob(torch.zeros(1, 1), torch.tensor([3.0]),
    #   torch.tensor([0.0])) returns about tensor([-2.0176]).
    def affine_log_prob(
        x: Tensor,
        scale: Tensor,
        shift: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the log of the absolute value of the determinant of g's
    #   Jacobian at x. x is one point of shape (d,).
    #   Example: log_abs_det(lambda v: 2 * v, torch.ones(2)) returns about
    #   tensor(1.3863).
    def log_abs_det(
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the log density of one point x, of shape (d,), using the
    #   change of variables. g maps x to noise that is standard normal.
    #   Example: flow_log_prob(lambda v: v / 3, torch.zeros(1)) returns about
    #   tensor(-2.0176).
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
        description="matches a normal with that mean and standard deviation",
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
        description="gives the log of the absolute determinant of the Jacobian at a point",
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
        "normal_log_prob": "Sum z ** 2 over the last dimension and multiply by -0.5. Then subtract z.shape[-1] / 2 "
        "times math.log(2 * math.pi).",
        "affine_log_prob": "Undo the map to get z, by subtracting the shift from x and dividing by the scale. Return "
        "normal_log_prob(z) minus the sum of the logs of the absolute scales, torch.log(scale.abs()).sum().",
        "log_abs_det": "torch.autograd.functional.jacobian(g, x) returns the Jacobian matrix. torch.linalg.slogdet of "
        "it has the field .logabsdet.",
        "flow_log_prob": "Return normal_log_prob of g(x) plus log_abs_det(g, x).",
    },
    introduces=(),
    check=check_flow1,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# flow2
# ---------------------------------------------------------------------------


def measure_coupling(seed: int = 0) -> dict[str, float]:
    """flow2's numbers for one seed, on a layer whose network starts 3 times larger, as the checks make it, and 1,000
    points. The largest gap between the points and the inverse of the layer's output, and the first point's log
    determinant as the sum of s and as slogdet of the full Jacobian. The lesson gives seeds 0 to 2. Not run by any
    check."""
    torch.manual_seed(seed)
    layer = REFERENCE["Coupling"](False)
    x = torch.randn(1000, 2, generator=torch.Generator().manual_seed(seed))

    with torch.no_grad():
        for parameter in layer.net.parameters():
            parameter.mul_(3)

        y, log_det = layer(x)
        gap = (layer.inverse(y) - x).abs().max().item()

    full = REFERENCE["log_abs_det"](lambda point: layer(point.unsqueeze(0))[0].squeeze(0), x[0])

    return {
        "inverse gap": gap,
        "sum of s": log_det[0].item(),
        "slogdet": full.item(),
    }


FLOW2_LESSON = prose(
    """
    Computing a determinant with `slogdet` is slow, and a general network
    cannot be inverted. RealNVP, from Dinh, Sohl-Dickstein and Bengio, is a
    layer whose inverse and determinant are both cheap. It splits each point
    into two parts. The first part passes through unchanged. A small network
    reads the first part and outputs `s` and `t`, which scale and shift the
    second part.

    ```text
    y1 = x1
    y2 = x2 * exp(s(x1)) + t(x1)
    ```

    `y1` does not depend on `x2`, so the Jacobian is triangular, which means
    every entry above its diagonal is 0. The determinant of a triangular
    matrix is the product of its diagonal entries. Those entries are 1 for
    the kept part and `exp(s)` for each scaled number. So the log determinant
    is the sum of `s`.

    Inverting the layer never requires inverting the network. Since
    `x1 = y1`, the network gets the same input during the inverse and outputs
    the same `s` and `t`. Then `x2 = (y2 - t) * exp(-s)`. So the network can
    be anything, because it only ever runs forwards.

    The points in this chapter have 2 numbers, so each part is one number.
    The network turns the kept number into `s` and `t`. A tanh keeps `s`
    between -1 and 1, so one layer multiplies the moved number by a factor
    between `1 / e` and `e`. A layer never changes its kept number. So a flow
    stacks layers that alternate which number they keep, as set by `flip`.

    `.chunk(2, dim=1)` splits a tensor into two equal parts along its
    columns.

    On 1,000 points and 3 seeds, running a layer and then its inverse gave
    back the input to within 0.000001. That difference is the rounding error
    of 32 bit floats. For the first point of each seed, the sum of `s` gave
    log determinants between -0.997 and 0.692. flow1's `slogdet` of the full
    Jacobian matched each of them to within 0.000001.
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
    #   (batch, 2). It returns y, of the same shape, and the log determinant of
    #   each point, of shape (batch,). Split x with self.halves. The kept half
    #   passes through unchanged. The moved half becomes moved * exp(s) + t,
    #   with s and t from self.scale_and_shift of the kept half. Put the two
    #   halves back together with self.join. inverse takes y and undoes forward
    #   exactly.
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
        "Coupling.forward": "self.halves splits x into the kept and the moved halves. self.scale_and_shift of the "
        "kept half gives s and t. Join kept with "
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
    This exercise trains a flow of 8 coupling layers, which alternate the
    number they keep. The data is two moons, two curved arcs of points. The
    flow is trained by maximum likelihood, as in likelihood3. The loss is the
    mean of `-log p(x)` over a batch, and the change of variables gives
    `log p(x)` exactly.

    `log_prob` runs a batch forwards through every layer and adds up the
    layers' log determinants. The points come out of the last layer as noise.
    The standard normal's log density of that noise is added to the sum.
    `sample` goes the other way. It draws noise and runs it backwards,
    calling each layer's `inverse`, with the layers in reverse order.

    The table compares the flow with two other models. One is the best single
    normal distribution, fitted with the data's own mean and covariance. The
    covariance holds each number's variance and how the two numbers vary
    together. The other is gan3's VAE. A sample counts as on the moons when
    it is within 0.15 of the arcs. Each range covers 3 seeds.

    | Model | Mean log density of new points | Samples on the moons |
    |---|---|---|
    | The best single normal | -1.89 | 34.7% to 34.8% |
    | VAE, 8,000 steps, its ELBO | -1.03 to -1.15 | 88% to 92% |
    | Flow, 250 steps | -0.52 to -0.77 | 81.4% to 90.7% |
    | Flow, 1,000 steps | -0.38 to -0.39 | 97.7% to 98.1% |
    | Flow, 2,000 steps | -0.35 to -0.38 | 98.5% to 99.1% |

    The normal puts nearly two thirds of its samples off the arcs. The flow
    bends space so that the round cloud of noise is mapped onto both arcs.
    After 1,000 steps its log density improves only a little, but the share
    of its samples on the moons keeps growing.

    The VAE gives only its ELBO, which is a lower bound on its log density. A
    more accurate estimate from 500 codes per point still reaches only -0.77
    to -0.81. A GAN gives no density at all. When you need the likelihood of
    each point, a flow gives it exactly.

    `reversed(layers)` goes through a list from its last item to its first.

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


    # TODO: Finish log_prob and sample. log_prob returns the log density of
    #   each row of x, as a tensor of shape (batch,). It runs x through every
    #   layer in order, adds up the layers' log determinants, and adds
    #   normal_log_prob of the final output. sample draws
    #   torch.randn(count, 2, generator=rng) and runs it through every layer's
    #   inverse, in reverse order, with autograd recording turned off. It
    #   returns the points, of shape (count, 2).
''' + indented(FLOW_INIT) + '''

        def log_prob(self, x: Tensor) -> Tensor:
            ...

        def sample(
            self,
            count: int,
            rng: torch.Generator,
        ) -> Tensor:
            ...


    # TODO: Train flow for steps steps with Adam at a learning rate of 3e-3.
    #   Each step draws a new batch with two_moons(256, 0.05, rng) and minimizes
    #   the mean of -log p over that batch. Return the last loss as a Python
    #   float.
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


def train_moons_vae(
    seed: int,
    steps: int,
    sigma: float = 0.1,
) -> tuple[nn.Module, nn.Module]:
    """gan3's VAE with its layers, trained on the moons with Adam at 0.001 on batches of 256. Its reconstruction is
    the squared error over 2 * sigma ** 2, the negative log density of a normal of spread sigma, constants dropped."""
    torch.manual_seed(seed)
    encoder = GAN_PARTS["mlp"](2, 4)
    decoder = GAN_PARTS["mlp"](2, 2)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=1e-3)
    rng = torch.Generator().manual_seed(seed + 100)

    for _ in range(steps):
        real = REFERENCE["two_moons"](256, 0.05, rng)
        mean, log_variance = encoder(real).chunk(2, dim=1)
        code = mean + torch.exp(0.5 * log_variance) * torch.randn(256, 2, generator=rng)

        reconstruction = ((real - decoder(code)) ** 2).sum(dim=1) / (2 * sigma**2)
        kl = 0.5 * (mean**2 + log_variance.exp() - 1 - log_variance).sum(dim=1)
        loss = (reconstruction + kl).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return encoder, decoder


def vae_log_densities(
    encoder: nn.Module,
    decoder: nn.Module,
    sigma: float,
    x: Tensor,
    draws: int = 500,
) -> tuple[float, float]:
    """The VAE's ELBO averaged over the rows of x, and a closer estimate of their log density. The closer one weights
    each code drawn from the encoder by how much likelier the model makes it than the encoder does, which the ELBO
    averages away. Both read the same 500 codes per point."""
    with torch.no_grad():
        mean, log_variance = encoder(x).chunk(2, dim=1)
        noise = torch.randn(draws, *mean.shape, generator=torch.Generator().manual_seed(11))
        z = mean + torch.exp(0.5 * log_variance) * noise
        rebuilt = decoder(z.reshape(-1, 2)).reshape(draws, len(x), 2)

    log_likelihood = -((x - rebuilt) ** 2).sum(dim=-1) / (2 * sigma**2) - 2 * math.log(sigma) - math.log(2 * math.pi)
    log_prior = REFERENCE["normal_log_prob"](z)
    log_encoder = REFERENCE["normal_log_prob"](noise) - 0.5 * log_variance.sum(dim=-1)
    weights = log_likelihood + log_prior - log_encoder

    return weights.mean(dim=0).mean().item(), (torch.logsumexp(weights, dim=0) - math.log(draws)).mean().item()


def measure_moons_models(seed: int = 0) -> dict[str, float]:
    """Every number of flow3's table for one seed, the mean log density of new points and the share of samples on
    the moons for each model. The normal fits 20,000 points, the flows train from scratch for each length, and the
    VAE, with sigma 0.1, trains 8,000 steps. A few minutes on a laptop CPU. The lesson gives seeds 0 to 2. Not run by
    any check."""
    test = REFERENCE["two_moons"](2000, 0.05, torch.Generator().manual_seed(1))
    results: dict[str, float] = {}

    with pinned_threads():
        data = REFERENCE["two_moons"](20000, 0.05, torch.Generator().manual_seed(100 + seed))
        centre = data.mean(dim=0)
        covariance = torch.cov(data.T)
        drawn = centre + torch.randn(2000, 2, generator=torch.Generator().manual_seed(4)) @ torch.linalg.cholesky(covariance).T
        results["normal, log density"] = torch.distributions.MultivariateNormal(centre, covariance).log_prob(test).mean().item()
        results["normal, on the moons"] = share_on_moons(drawn)

        for steps in (250, 1000, 2000):
            torch.manual_seed(seed)
            flow = REFERENCE["Flow"](8)
            REFERENCE["train_flow"](flow, steps, torch.Generator().manual_seed(seed))
            results[f"flow {steps}, log density"] = held_out_log_prob(flow)
            results[f"flow {steps}, on the moons"] = share_on_moons(flow.sample(2000, torch.Generator().manual_seed(4)))

        encoder, decoder = train_moons_vae(seed, 8000)
        elbo, closer = vae_log_densities(encoder, decoder, 0.1, test)

        with torch.no_grad():
            samples = decoder(torch.randn(2000, 2, generator=torch.Generator().manual_seed(4)))

        results["vae, ELBO"] = elbo
        results["vae, closer log density"] = closer
        results["vae, on the moons"] = share_on_moons(samples)

    return results


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
            "be above -1.3. The best single normal gets -1.89.",
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
        "Flow.log_prob": "Start a total at torch.zeros(len(x)). Each layer returns the new x and its log "
        "determinants, and the log determinants go into the total. Return normal_log_prob of the final x plus the "
        "total.",
        "Flow.sample": "Draw the noise, then inside torch.no_grad() loop over reversed(self.layers), setting z to "
        "layer.inverse(z).",
        "train_flow": "Create the optimizer once, before the loop. In each step the loss is minus "
        "flow.log_prob(batch).mean(). Then call zero_grad, backward and step.",
    },
    introduces=(),
    check=check_flow3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
