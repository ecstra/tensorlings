"""74_equivariance, symmetries tested, rotation invariant features from distances after EGNN, when to build a symmetry in, and the Geometric Deep Learning blueprint."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

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
    expect_close,
    expect_tensor,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .sets import AttentionLayer, ReferenceEquivariantLayer
from .tensors import pinned_threads

EQUIVARIANCE = Chapter(
    folder="74_equivariance",
    title="Equivariance",
    readme=source_file(
        """
        # Equivariance

        A symmetry of a task is a change to the input that a model should
        ignore, or follow. This chapter tests models for symmetries, builds a
        layer for points in space that respects rotations by construction,
        measures when building a symmetry in beats teaching it with
        augmentation, and ends with the recipe behind convolutions, Deep Sets
        and graph networks.

        ## Further reading

        - [Satorras, Hoogeboom and Welling, E(n) equivariant graph neural networks](https://arxiv.org/abs/2102.09844)
        - [Bronstein, Bruna, Cohen and Veličković, Geometric Deep Learning](https://arxiv.org/abs/2104.13478)
        - [Zhang, making convolutional networks shift invariant again](https://arxiv.org/abs/1904.11486)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def no_grad_recorder(
    module: nn.Module,
    calls: list[bool],
) -> torch.utils.hooks.RemovableHandle:
    """Records, on every call of module, whether autograd was recording."""
    return module.register_forward_pre_hook(lambda _, __: calls.append(torch.is_grad_enabled()))


# ---------------------------------------------------------------------------
# equivariance1
# ---------------------------------------------------------------------------


def reference_shift(
    x: Tensor,
    steps: int,
) -> Tensor:
    return torch.roll(x, steps, dims=-1)


def reference_rotate(
    points: Tensor,
    angle: float,
) -> Tensor:
    turn = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    return points @ turn.T


def reference_equivariance_gap(
    f: Callable[[Tensor], Tensor],
    g: Callable[[Tensor], Tensor],
    x: Tensor,
) -> float:
    return (f(g(x)) - g(f(x))).abs().max().item()


def reference_invariance_gap(
    f: Callable[[Tensor], Tensor],
    g: Callable[[Tensor], Tensor],
    x: Tensor,
) -> float:
    return (f(g(x)) - f(x)).abs().max().item()


EQUIVARIANCE1_LESSON = prose(
    """
    A symmetry of a task is a change to the input that shouldn't change the
    answer, or should change it the same way. Call the change g, like
    shifting a signal, shuffling a set or rotating some points.

    A function f is **invariant** to g when f(g x) = f(x). It is
    **equivariant** when f(g x) = g f(x), so changing the input changes the
    output the same way.

    ```python
    torch.roll(torch.tensor([1.0, 2.0, 3.0, 4.0]), 1)    # tensor([4., 1., 2., 3.])
    ```

    `torch.roll` moves every entry along and wraps the last ones round to
    the start, a shift with nothing lost.

    Both properties are tested the same way. Run both sides and compare. When
    the property holds, floats leave a gap the size of rounding, around 1e-7.

    | f | g | Gap |
    |---|---|---|
    | conv1d, circular padding | shift by 3 | 0, equivariant |
    | conv1d, zero padding | shift by 3 | 5.11 |
    | a Linear on the whole signal | shift by 3 | 2.01 |
    | the sum of the rows | shuffle the rows | 6e-8, invariant |
    | a Linear on each row | shuffle the rows | 6e-8, equivariant |
    | a Linear on the rows flattened | shuffle the rows | 0.26 |
    | all distances between points | rotate the points | 2.4e-7, invariant |
    | a 2 by 2 Linear on the points | rotate the points | 1.21 |

    Zero padding breaks shift equivariance at the edges. A shifted signal
    pushes values off one end and zeros in at the other, so a convolution is
    equivariant only away from its borders. Circular padding wraps round,
    which makes a shift an exact symmetry of it.

    Turned by an angle t around the origin, a point (a, b) lands here.

    ```text
    (a cos t - b sin t,  a sin t + b cos t)
    ```

    As a matrix that is `R = [[cos t, -sin t], [sin t, cos t]]`, and rows of
    points turn as `points @ R.T`.
    """,
)

EQUIVARIANCE1_SOURCE = source_file(
    '''
    # equivariance1.py

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    rolled = torch.roll(torch.tensor([1.0, 2.0, 3.0, 4.0]), 1)    # tensor([4., 1., 2., 3.])
    quarter = torch.tensor([[0.0, -1.0], [1.0, 0.0]])             # R for a quarter turn
    turned = torch.tensor([[1.0, 0.0]]) @ quarter.T               # tensor([[0., 1.]])


    # TODO: Return x shifted steps places along its last dimension, with the
    #   entries that fall off the end wrapped round to the start.
    def shift(
        x: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return points, of shape (..., 2), each turned angle radians around
    #   the origin, counterclockwise.
    def rotate(
        points: Tensor,
        angle: float,
    ) -> Tensor:
        ...


    # TODO: Return the largest gap between f(g(x)) and g(f(x)), entry by entry,
    #   as a plain number.
    def equivariance_gap(
        f: Callable[[Tensor], Tensor],
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> float:
        ...


    # TODO: Return the largest gap between f(g(x)) and f(x), entry by entry, as
    #   a plain number.
    def invariance_gap(
        f: Callable[[Tensor], Tensor],
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> float:
        ...
    ''',
)

EQUIVARIANCE1_SOLUTION = source_file(
    '''
    # equivariance1.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    rolled = torch.roll(torch.tensor([1.0, 2.0, 3.0, 4.0]), 1)
    quarter = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    turned = torch.tensor([[1.0, 0.0]]) @ quarter.T


    def shift(
        x: Tensor,
        steps: int,
    ) -> Tensor:
        return torch.roll(x, steps, dims=-1)


    def rotate(
        points: Tensor,
        angle: float,
    ) -> Tensor:
        turn = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

        return points @ turn.T


    def equivariance_gap(
        f: Callable[[Tensor], Tensor],
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> float:
        return (f(g(x)) - g(f(x))).abs().max().item()


    def invariance_gap(
        f: Callable[[Tensor], Tensor],
        g: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> float:
        return (f(g(x)) - f(x)).abs().max().item()
    ''',
)

CHECK_KERNEL = torch.tensor([[[0.5, -1.0, 2.0]]])
CHECK_ORDER = torch.tensor([3, 0, 4, 1, 2])


def circular_conv(x: Tensor) -> Tensor:
    return F.conv1d(F.pad(x, (1, 1), mode="circular"), CHECK_KERNEL)


def zero_conv(x: Tensor) -> Tensor:
    return F.conv1d(x, CHECK_KERNEL, padding=1)


def seeded_linear(
    seed: int,
    inputs: int,
    outputs: int,
) -> nn.Linear:
    torch.manual_seed(seed)

    return nn.Linear(inputs, outputs)


CHECK_ROW_LINEAR = seeded_linear(740, 3, 4)
CHECK_FLAT_LINEAR = seeded_linear(741, 15, 4)
CHECK_POINT_LINEAR = seeded_linear(742, 2, 2)

SHIFT_BY_3 = Described(lambda x: torch.roll(x, 3, dims=-1), "a shift by 3")
SHUFFLE = Described(lambda x: x[CHECK_ORDER], "a shuffle of the 5 rows")
TURN = Described(lambda points: reference_rotate(points, 0.7), "a rotation by 0.7")


def check_equivariance1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="shift",
        description="moves every entry along and wraps the end round",
        inputs=(
            (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
            (small_tensor(743, 2, 6), -1),
            (small_tensor(744, 3, 4), 6),
        ),
        reference=reference_shift,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rotate",
        description="turns each point counterclockwise around the origin",
        inputs=(
            (torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]]), 0.5),
            (small_tensor(745, 2, 4, 2), -1.2),
        ),
        reference=reference_rotate,
        expect=expect_tensor,
    )

    signals = Described(small_tensor(746, 4, 1, 16), "4 signals of 16")
    rows = Described(small_tensor(747, 5, 3), "5 rows of 3")
    points = Described(small_tensor(748, 6, 2), "6 points")
    gap = lambda call, actual, expected: expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-5,
    )
    check_on_inputs(
        checker=checker,
        function="equivariance_gap",
        description="compares f after g with g after f",
        inputs=(
            (Described(circular_conv, "a conv1d with circular padding"), SHIFT_BY_3, signals),
            (Described(zero_conv, "a conv1d with zero padding"), SHIFT_BY_3, signals),
            (Described(CHECK_ROW_LINEAR, "a Linear on each row"), SHUFFLE, rows),
            (Described(CHECK_POINT_LINEAR, "a 2 by 2 Linear on the points"), TURN, points),
        ),
        reference=reference_equivariance_gap,
        expect=gap,
    )
    check_on_inputs(
        checker=checker,
        function="invariance_gap",
        description="compares f after g with f alone",
        inputs=(
            (Described(lambda x: x.sum(dim=0), "the sum of the rows"), SHUFFLE, rows),
            (Described(lambda x: CHECK_FLAT_LINEAR(x.flatten()), "a Linear on the rows flattened"), SHUFFLE, rows),
            (Described(lambda p: torch.cdist(p, p), "all distances between points"), TURN, points),
            (Described(lambda p: p.sum(dim=0), "the sum of the points"), TURN, points),
        ),
        reference=reference_invariance_gap,
        expect=gap,
    )


EQUIVARIANCE1 = Exercise(
    name="equivariance1",
    chapter=EQUIVARIANCE,
    title="Symmetries as tests",
    lesson=EQUIVARIANCE1_LESSON,
    source=EQUIVARIANCE1_SOURCE,
    solution=EQUIVARIANCE1_SOLUTION,
    hints={
        "shift": "torch.roll with dims=-1 rolls the last dimension.",
        "rotate": "Build R from math.cos and math.sin of angle as the lesson writes it, then multiply points by its "
        "transpose.",
        "equivariance_gap": "Take f(g(x)) minus g(f(x)), then .abs().max().item().",
        "invariance_gap": "Take f(g(x)) minus f(x), then .abs().max().item().",
    },
    introduces=(),
    check=check_equivariance1,
)


# ---------------------------------------------------------------------------
# equivariance2
# ---------------------------------------------------------------------------

EGNN_INIT = '''
class EGNNLayer(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * features + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.coord = nn.Linear(hidden, 1)
        self.node = nn.Sequential(nn.Linear(features + hidden, hidden), nn.SiLU(), nn.Linear(hidden, features))
'''

EGNN_FORWARD = '''
    def forward(
        self,
        h: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n = len(h)
        differences = x[:, None, :] - x[None, :, :]
        squared = (differences**2).sum(dim=-1, keepdim=True)
        pairs = torch.cat([h[:, None, :].expand(n, n, -1), h[None, :, :].expand(n, n, -1), squared], dim=-1)
        others = 1 - torch.eye(n).unsqueeze(-1)
        messages = self.edge(pairs) * others
        moved = x + (differences * self.coord(messages)).sum(dim=1) / (n - 1)
        updated = self.node(torch.cat([h, messages.sum(dim=1)], dim=-1))

        return updated, moved
'''

E_N_GAPS_SOLUTION = '''
def e_n_gaps(
    layer: nn.Module,
    h: Tensor,
    x: Tensor,
    rotation: Tensor,
    offset: Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        features, points = layer(h, x)
        moved_features, moved_points = layer(h, x @ rotation.T + offset)

    feature_gap = (moved_features - features).abs().max().item()
    point_gap = (moved_points - (points @ rotation.T + offset)).abs().max().item()

    return feature_gap, point_gap
'''


def egnn_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec(EGNN_INIT + EGNN_FORWARD + E_N_GAPS_SOLUTION, namespace)

    return namespace


EGNN_REFERENCE = egnn_namespace()


def reference_squared_distances(x: Tensor) -> Tensor:
    return ((x[:, None, :] - x[None, :, :]) ** 2).sum(dim=-1)


class PointMLP(nn.Module):
    """A layer that reads the coordinates as plain numbers, for the lesson's comparison."""

    def __init__(
        self,
        features: int,
        hidden: int,
        dims: int,
    ) -> None:
        super().__init__()
        self.features = features
        self.mlp = nn.Sequential(nn.Linear(features + dims, hidden), nn.SiLU(), nn.Linear(hidden, features + dims))

    def forward(
        self,
        h: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        out = self.mlp(torch.cat([h, x], dim=-1))

        return out[:, : self.features], x + out[:, self.features :]


def random_rotation_3d(generator: torch.Generator) -> Tensor:
    """A rotation in 3D, from the QR of a random matrix, with signs fixed so it turns and never mirrors."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))

    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]

    return q


def seeded_egnn(
    make: type,
    seed: int = 750,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(4, 16)


EQUIVARIANCE2_LESSON = prose(
    """
    A molecule or a cloud of points lives in space. Turned, moved or mirrored,
    it is the same object. A model of its energy should be invariant to all
    three. The forces on each atom should turn and mirror with it, and stay
    the same when it moves.

    A layer that reads the coordinates as plain numbers is neither. Satorras,
    Hoogeboom and Welling's E(n) equivariant graph network, EGNN, gets both
    from one idea. E(n) names every turn, move and mirror in n dimensions.
    Distances between points never change under any of them. So the layer
    reads only distances, and moves points only along the differences
    between them.

    One layer, for points `x_i` with features `h_i`, as the paper writes it:

    ```text
    m_ij  = phi_e(h_i, h_j, |x_i - x_j|^2)
    x_i'  = x_i + C * sum over j != i of (x_i - x_j) * phi_x(m_ij),  C = 1 / (n - 1)
    h_i'  = phi_h(h_i, sum over j != i of m_ij)
    ```

    `phi_e` and `phi_h` are small MLPs. `phi_x` is one Linear on the message
    here, where the paper uses a small MLP too. The messages see only
    distances, so the new features are invariant. Each point moves along
    differences, which turn with the input, so the new points are
    equivariant.

    Here is a random layer of each kind on 8 points in 3D, turned and moved,
    with the largest gap of each output.

    | Layer | Feature gap | Point gap |
    |---|---|---|
    | EGNN | 7.5e-8 | 2.4e-7 |
    | an MLP on features and coordinates | 0.64 | 0.68 |

    The same EGNN layer, with the points mirrored by flipping one coordinate,
    gives gaps of exactly 0.
    With the points shuffled, its outputs shuffle the same way, to within
    3e-8. The paper points out that it needs none of the spherical harmonics,
    special functions on a sphere, that earlier equivariant networks were
    built on.

    `.expand` repeats a tensor along a dimension of size 1 without copying
    it, and -1 keeps a size as it is.
    """,
)

EQUIVARIANCE2_SOURCE = source_file(
    '''
    # equivariance2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    points = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    gaps = points[:, None, :] - points[None, :, :]    # shape (2, 2, 2), gaps[i, j] is point i minus point j
    first = gaps[0, 1]                                 # tensor([-3., -4.])

    h = torch.tensor([[1.0], [2.0]])
    rows = h[:, None, :].expand(2, 2, -1)              # shape (2, 2, 1), rows[i, j] is h[i] for every j


    # TODO: Return the squared distance between every pair of points of x, of
    #   shape (n, dims), as a tensor of shape (n, n).
    def squared_distances(x: Tensor) -> Tensor:
        ...


    # TODO: Finish forward, one EGNN layer from the lesson, for features h of
    #   shape (n, features) and points x of shape (n, dims).
    #   - Each pair (i, j) joins h_i, h_j and |x_i - x_j|^2, in that order, and
    #     self.edge turns it into a message.
    #   - The pairs with j equal to i get a message of 0.
    #   - Each point moves by the sum over j of (x_i - x_j) times
    #     self.coord(message), divided by n - 1.
    #   - The new features are self.node of h_i joined with the sum of its
    #     messages.
    #   Return the new features, then the new points.
''' + indented(EGNN_INIT) + '''

        def forward(
            self,
            h: Tensor,
            x: Tensor,
        ) -> tuple[Tensor, Tensor]:
            ...


    # TODO: Return the largest gap in layer's features and the largest gap in
    #   its points, as two plain numbers, when x is turned by rotation and
    #   moved by offset. The features should not change at all, and the points
    #   should turn and move with x. Record nothing for autograd.
    def e_n_gaps(
        layer: nn.Module,
        h: Tensor,
        x: Tensor,
        rotation: Tensor,
        offset: Tensor,
    ) -> tuple[float, float]:
        ...
    ''',
)

EQUIVARIANCE2_SOLUTION = source_file(
    '''
    # equivariance2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    points = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    gaps = points[:, None, :] - points[None, :, :]
    first = gaps[0, 1]

    h = torch.tensor([[1.0], [2.0]])
    rows = h[:, None, :].expand(2, 2, -1)


    def squared_distances(x: Tensor) -> Tensor:
        return ((x[:, None, :] - x[None, :, :]) ** 2).sum(dim=-1)


''' + indented(EGNN_INIT) + '''

''' + indented(EGNN_FORWARD) + '''


''' + indented(E_N_GAPS_SOLUTION) + '''
    ''',
)

CHECK_TURN = random_rotation_3d(torch.Generator().manual_seed(751))
CHECK_OFFSET = torch.tensor([0.5, -2.0, 1.5])


def check_egnn_forward(found: type) -> None:
    layer = expect_module("EGNNLayer(4, 16)", seeded_egnn(found))
    reference = seeded_egnn(EGNN_REFERENCE["EGNNLayer"])
    generator = torch.Generator().manual_seed(752)
    h = torch.randn(5, 4, generator=generator)
    x = torch.randn(5, 3, generator=generator)

    with torch.no_grad():
        expected_h, expected_x = reference(h, x)
        result = layer(h, x)

    if not isinstance(result, tuple) or len(result) != 2:
        raise Mismatch("EGNNLayer's forward should return the new features and the new points, in a tuple.")

    expect_tensor(
        call="the new features of EGNNLayer(4, 16) on 5 points in 3D",
        actual=result[0],
        expected=expected_h,
    )
    expect_tensor(
        call="the new points of EGNNLayer(4, 16) on 5 points in 3D",
        actual=result[1],
        expected=expected_x,
    )


def check_e_n_gaps(found: object) -> None:
    generator = torch.Generator().manual_seed(753)
    h = torch.randn(6, 4, generator=generator)
    x = torch.randn(6, 3, generator=generator)

    for name, layer in (
        ("an EGNN layer", seeded_egnn(EGNN_REFERENCE["EGNNLayer"], 754)),
        ("an MLP on features and coordinates", seeded_egnn(lambda features, hidden: PointMLP(features, hidden, 3), 755)),
    ):
        expected = EGNN_REFERENCE["e_n_gaps"](layer, h, x, CHECK_TURN, CHECK_OFFSET)
        recording: list[bool] = []
        hook = no_grad_recorder(layer, recording)

        try:
            actual = found(layer, h, x, CHECK_TURN, CHECK_OFFSET)
        finally:
            hook.remove()

        if any(recording):
            raise Mismatch("e_n_gaps ran the layer while recording for autograd. Run it inside torch.no_grad().")

        if not isinstance(actual, tuple) or len(actual) != 2:
            raise Mismatch("e_n_gaps should return the feature gap and the point gap as two numbers in a tuple.")

        for part, value, wanted in zip(("feature gap", "point gap"), actual, expected):
            expect_close(
                call=f"the {part} of e_n_gaps for {name}",
                actual=value,
                expected=wanted,
                tolerance=1e-5,
            )


def check_equivariance2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="squared_distances",
        description="gives the squared distance of every pair",
        inputs=(
            (torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]),),
            (small_tensor(756, 5, 3),),
        ),
        reference=reference_squared_distances,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="EGNNLayer",
        method="forward",
        description="passes messages on distances and moves points along differences",
        run=check_egnn_forward,
    )
    checker.test(
        function="e_n_gaps",
        description="measures how far features and points stray under a turn and a move",
        run=check_e_n_gaps,
    )


EQUIVARIANCE2 = Exercise(
    name="equivariance2",
    chapter=EQUIVARIANCE,
    title="Invariant features from distances",
    lesson=EQUIVARIANCE2_LESSON,
    source=EQUIVARIANCE2_SOURCE,
    solution=EQUIVARIANCE2_SOLUTION,
    hints={
        "squared_distances": "x[:, None, :] - x[None, :, :] gives every difference, as in the example. Square it and "
        "sum over the last dimension.",
        "EGNNLayer.forward": "Build the differences and their squared lengths with keepdim=True. torch.cat the expanded "
        "h rows, the expanded h columns and the squared lengths on dim=-1. Multiply the messages by "
        "1 - torch.eye(n).unsqueeze(-1).",
        "e_n_gaps": "Inside torch.no_grad(), run layer on x and on x @ rotation.T + offset. Compare the features "
        "directly, and the second points with the first turned and moved the same way.",
    },
    introduces=(),
    check=check_equivariance2,
)


# ---------------------------------------------------------------------------
# equivariance3
# ---------------------------------------------------------------------------

SET_POINTS = 6


def reference_random_rotations(
    count: int,
    generator: torch.Generator,
) -> Tensor:
    angles = torch.rand(count, generator=generator) * 2 * math.pi
    cos, sin = torch.cos(angles), torch.sin(angles)

    return torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)


def reference_scramble(
    points: Tensor,
    generator: torch.Generator,
) -> Tensor:
    turned = points @ reference_random_rotations(len(points), generator).transpose(1, 2)
    order = torch.argsort(torch.rand(points.shape[0], points.shape[1], generator=generator), dim=1)

    return turned[torch.arange(len(points)).unsqueeze(1), order]


def reference_sorted_distances(points: Tensor) -> Tensor:
    i, j = torch.triu_indices(points.shape[1], points.shape[1], 1)

    return (points[:, i] - points[:, j]).norm(dim=-1).sort(dim=1).values


def reference_agreement(
    predict: Callable[[Tensor], Tensor],
    points: Tensor,
    generator: torch.Generator,
) -> float:
    return (predict(points) == predict(reference_scramble(points, generator))).float().mean().item()


def shape_templates(
    classes: int,
    seed: int,
) -> Tensor:
    shapes = torch.randn(classes, SET_POINTS, 2, generator=torch.Generator().manual_seed(seed))

    return shapes - shapes.mean(dim=1, keepdim=True)


def shape_sets(
    templates: Tensor,
    count: int,
    generator: torch.Generator,
    noise: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Noisy copies of the templates, each point in its template's order, neither turned nor shuffled."""
    labels = torch.randint(0, len(templates), (count,), generator=generator)

    return templates[labels] + noise * torch.randn(count, SET_POINTS, 2, generator=generator), labels


def train_shapes(
    kind: str,
    x: Tensor,
    y: Tensor,
    classes: int,
    steps: int,
    generator: torch.Generator,
    features: Callable[[Tensor], Tensor] = reference_sorted_distances,
    scramble: Callable[[Tensor, torch.Generator], Tensor] = reference_scramble,
) -> Callable[[Tensor], Tensor]:
    """An MLP trained on sets as they come, which a "built in" model reads as sorted distances, and an
    "augmented" one as raw coordinates scrambled afresh every step. Returns the class it predicts."""
    torch.manual_seed(0)
    pairs = SET_POINTS * (SET_POINTS - 1) // 2
    model = nn.Sequential(nn.Linear(pairs if kind == "built in" else 2 * SET_POINTS, 64), nn.ReLU(), nn.Linear(64, classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def inputs_of(points: Tensor) -> Tensor:
        return features(points) if kind == "built in" else points.flatten(1)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (64,), generator=generator)
        points = x[batch] if kind == "built in" else scramble(x[batch], generator)
        loss = F.cross_entropy(model(inputs_of(points)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(points: Tensor) -> Tensor:
        with torch.no_grad():
            return model(inputs_of(points)).argmax(dim=1)

    return predict


EQUIVARIANCE3_LESSON = prose(
    """
    There are two ways to give a model a symmetry. Build it in, so the model
    can't break it, as equivariance2 did. Or teach it, by training on inputs
    the symmetry changes, the data augmentation of augment1. Here are both,
    on telling apart 10 shapes of 6 points that come turned at any angle and
    with their points in any order.

    - Built in: the sorted distances between the points, which no turn or shuffle can change, read by an MLP.
    - Taught: an MLP on the raw coordinates, trained on sets turned and shuffled afresh every step.

    Both train for 1,500 steps on noisy copies of the shapes and are tested on
    2,000 new ones, turned and shuffled.

    | Training sets | 20 | 100 | 1,000 |
    |---|---|---|---|
    | Built in | 70.3% | 92.8% | 97.0% |
    | Taught | 39.7% | 46.3% | 47.9% |

    Over 3 seeds the built in model led at every size. Turn and shuffle the
    test sets once more, and it gives the same answer every time. The taught
    one keeps only 52% to 72% of its answers. It has every angle times 720
    orders of the points to learn. Trained 4 times as long it reached 47% to
    60%, and 62% to 82% with 256 units. The built in model passed 90% at a
    quarter of those steps.

    Teaching does as well when the symmetry is small. Keep each set's points
    in their template's order, so only the turn is left to learn. At 1,000
    sets the taught model then got 100% in every seed, and so did an MLP on
    the same distances unsorted. The sorted distances stayed at 90% to 97%,
    since sorting throws away which pair each distance belongs to.

    Building in more symmetry than the task has loses information for good.
    Distances can't tell a shape from its mirror image. Here the task is one
    shape against its mirror, with only the turn to learn. Both distance
    models got 48% to 50%, a coin flip, and the taught one 100%.

    So build in the symmetries the data has for sure, and no more. Teach the
    ones that only hold roughly, or that the structure can't express.
    """,
)

EQUIVARIANCE3_SOURCE = source_file(
    '''
    # equivariance3.py

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    pairs = torch.triu_indices(3, 3, 1)                 # tensor([[0, 0, 1], [1, 2, 2]]), every pair i < j
    picked = torch.tensor([[10, 20, 30]])[torch.tensor([[0]]), torch.tensor([[2, 0, 1]])]    # tensor([[30, 10, 20]])
    ordered = torch.tensor([3.0, 1.0, 2.0]).sort().values                                   # tensor([1., 2., 3.])


    # TODO: Return count rotation matrices, of shape (count, 2, 2), each the R
    #   of equivariance1 for an angle drawn with
    #   torch.rand(count, generator=generator) * 2 * math.pi.
    def random_rotations(
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return points, of shape (sets, n, 2), with each set turned by its
    #   own matrix from random_rotations, drawn first. Then shuffle the points
    #   of each set in an order drawn with torch.argsort of
    #   torch.rand(sets, n, generator=generator) along dim=1.
    def scramble(
        points: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the distances between every pair of points i < j of each set
    #   of points, of shape (sets, n, 2), sorted from smallest to largest, as a
    #   tensor of shape (sets, n * (n - 1) / 2).
    def sorted_distances(points: Tensor) -> Tensor:
        ...


    # TODO: Return the share of sets whose class predict gives stays the same
    #   after scramble with generator, as a plain number. predict takes sets of
    #   points and returns one class per set.
    def agreement(
        predict: Callable[[Tensor], Tensor],
        points: Tensor,
        generator: torch.Generator,
    ) -> float:
        ...
    ''',
)

EQUIVARIANCE3_SOLUTION = source_file(
    '''
    # equivariance3.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    pairs = torch.triu_indices(3, 3, 1)
    picked = torch.tensor([[10, 20, 30]])[torch.tensor([[0]]), torch.tensor([[2, 0, 1]])]
    ordered = torch.tensor([3.0, 1.0, 2.0]).sort().values


    def random_rotations(
        count: int,
        generator: torch.Generator,
    ) -> Tensor:
        angles = torch.rand(count, generator=generator) * 2 * math.pi
        cos, sin = torch.cos(angles), torch.sin(angles)

        return torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)


    def scramble(
        points: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        turned = points @ random_rotations(len(points), generator).transpose(1, 2)
        order = torch.argsort(torch.rand(points.shape[0], points.shape[1], generator=generator), dim=1)

        return turned[torch.arange(len(points)).unsqueeze(1), order]


    def sorted_distances(points: Tensor) -> Tensor:
        i, j = torch.triu_indices(points.shape[1], points.shape[1], 1)

        return (points[:, i] - points[:, j]).norm(dim=-1).sort(dim=1).values


    def agreement(
        predict: Callable[[Tensor], Tensor],
        points: Tensor,
        generator: torch.Generator,
    ) -> float:
        return (predict(points) == predict(scramble(points, generator))).float().mean().item()
    ''',
)


def nearest_template(templates: Tensor) -> Callable[[Tensor], Tensor]:
    """A predictor that reads raw coordinates, so its answers change when the sets are turned or shuffled."""

    def predict(points: Tensor) -> Tensor:
        return ((points[:, None] - templates[None]) ** 2).sum(dim=(2, 3)).argmin(dim=1)

    return predict


def check_learns_with_distances(found: object) -> None:
    templates = shape_templates(10, 760)
    x, y = shape_sets(templates, 100, torch.Generator().manual_seed(761))
    test_x, test_y = shape_sets(templates, 500, torch.Generator().manual_seed(762))
    test_x = reference_scramble(test_x, torch.Generator().manual_seed(763))
    predict = train_shapes("built in", x, y, 10, 300, torch.Generator().manual_seed(764), features=found)
    right = (predict(test_x) == test_y).float().mean().item()

    if right < 0.7:
        raise Mismatch(
            f"An MLP trained for 300 steps on sorted_distances of 100 sets gets {right:.1%} of turned and shuffled "
            "test sets right, but it should pass 70%.",
        )


def check_equivariance3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="random_rotations",
        description="draws one rotation matrix per angle",
        inputs=(
            (3, seeded(765)),
            (1, seeded(766)),
        ),
        reference=reference_random_rotations,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="scramble",
        description="turns each set, then shuffles its points",
        inputs=(
            (small_tensor(767, 2, 4, 2), seeded(768)),
            (small_tensor(769, 3, 5, 2), seeded(770)),
        ),
        reference=reference_scramble,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="sorted_distances",
        description="gives every pair's distance, sorted",
        inputs=(
            (torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]]]),),
            (small_tensor(771, 2, 5, 2),),
        ),
        reference=reference_sorted_distances,
        expect=expect_tensor,
    )
    checker.test(
        function="sorted_distances",
        description="lets an MLP tell turned and shuffled shapes apart",
        run=check_learns_with_distances,
    )
    templates = shape_templates(4, 772)
    sets, _ = shape_sets(templates, 40, torch.Generator().manual_seed(773))
    check_on_inputs(
        checker=checker,
        function="agreement",
        description="gives the share of answers a scramble leaves alone",
        inputs=(
            (Described(nearest_template(templates), "a predictor that compares raw coordinates"), Described(sets, "40 sets of 6 points"), seeded(774)),
            (
                Described(lambda points: reference_sorted_distances(points).sum(dim=1).round().long(), "a predictor on distances"),
                Described(sets, "40 sets of 6 points"),
                seeded(775),
            ),
        ),
        reference=reference_agreement,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
        ),
    )


EQUIVARIANCE3 = Exercise(
    name="equivariance3",
    chapter=EQUIVARIANCE,
    title="Built in or taught",
    lesson=EQUIVARIANCE3_LESSON,
    source=EQUIVARIANCE3_SOURCE,
    solution=EQUIVARIANCE3_SOLUTION,
    hints={
        "random_rotations": "torch.cos and torch.sin of the angles give two tensors of shape (count,). torch.stack "
        "[cos, -sin] and [sin, cos] on dim=-1 for the rows, then stack the two rows on dim=-2.",
        "scramble": "points @ random_rotations(len(points), generator).transpose(1, 2) turns each set. Then index the "
        "turned sets with torch.arange(len(points)).unsqueeze(1) and the order, as the example picks.",
        "sorted_distances": "torch.triu_indices(n, n, 1) gives i and j. points[:, i] - points[:, j], then .norm(dim=-1) "
        "and .sort(dim=1).values.",
        "agreement": "Compare predict(points) with predict of the scrambled points, then .float().mean().item().",
    },
    introduces=(),
    check=check_equivariance3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# equivariance4
# ---------------------------------------------------------------------------

SHIFT_NET_INIT = '''
class ShiftNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, padding_mode="circular")
        self.head = nn.Linear(8, classes)
'''

SHIFT_NET_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.conv1(x.unsqueeze(1)))
        y = F.max_pool1d(y, 2)
        y = torch.relu(self.conv2(y))

        return self.head(y.mean(dim=2))
'''

SHIFT_GAPS_SOLUTION = '''
def shift_gaps(
    model: nn.Module,
    x: Tensor,
) -> list[float]:
    with torch.no_grad():
        start = model(x)

        return [(model(torch.roll(x, steps, dims=-1)) - start).abs().max().item() for steps in range(x.shape[-1])]
'''


def shift_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(SHIFT_NET_INIT + SHIFT_NET_FORWARD + SHIFT_GAPS_SOLUTION, namespace)

    return namespace


SHIFT_REFERENCE = shift_namespace()

BLUR_POOL_PART = '''
# Zhang's fix for the max pool, given to try in ShiftNet. It takes the larger of each pair at every
# position, blurs with the weights 1, 4, 6, 4, 1 over 16, and only then keeps every second position.
def blur_pool(y: Tensor) -> Tensor:
    channels = y.shape[1]
    y = F.max_pool1d(F.pad(y, (0, 1), mode="circular"), 2, stride=1)
    blur = (torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16).repeat(channels, 1, 1)
    y = F.conv1d(F.pad(y, (2, 2), mode="circular"), blur, groups=channels)

    return y[:, :, ::2]
'''


def blur_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "F": F,
        "Tensor": Tensor,
    }
    exec(BLUR_POOL_PART, namespace)

    return namespace


BLUR_REFERENCE = blur_namespace()


class UnpooledShiftNet(nn.Module):
    """ShiftNet without its local pooling, for the lesson's comparison."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, padding_mode="circular")
        self.head = nn.Linear(8, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.conv2(torch.relu(self.conv1(x.unsqueeze(1))))).mean(dim=2))


class FlatShiftNet(nn.Module):
    """The same convolutions with a Linear on every position in place of the global pooling."""

    def __init__(
        self,
        classes: int,
        length: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, padding_mode="circular")
        self.head = nn.Linear(8 * length, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.conv2(torch.relu(self.conv1(x.unsqueeze(1))))).flatten(1))


def seeded_shift_net(
    make: type,
    seed: int = 780,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(3)


EQUIVARIANCE4_LESSON = prose(
    """
    Most models this course built for structured data follow one recipe,
    which Bronstein, Bruna, Cohen and Veličković called the Geometric Deep
    Learning blueprint. Find the symmetry of the data. Then stack these.

    - Layers equivariant to that symmetry, each with a nonlinearity after it.
    - Local pooling now and then, which coarsens the data while keeping its structure.
    - One invariant pooling at the end, when the answer shouldn't change, then a head.

    Equivariant layers stack, since each passes the change on to the next. The
    invariant pooling goes last, because nothing after it can see the change.

    | Data | Symmetry | Equivariant layer | Built in |
    |---|---|---|---|
    | signals and images | shifts | convolution | conv1, 30_convolution |
    | sets | shuffles | a layer on each element, plus a sum | sets1, sets3 |
    | graphs | renumbering the nodes | message passing | message1 |
    | points in space | turns, moves and mirrors | EGNN | equivariance2 |
    | tokens with no positions | shuffles | attention | position1 |

    Here is the recipe for shifts that wrap round, as `torch.roll` makes them.
    A circular convolution and a ReLU, a max pool of 2, another convolution
    and ReLU, then the mean over the positions. Each gap below is the largest
    change in any output, over 32 random signals of length 16 and every even
    or every odd shift.

    | Model | Even shifts | Odd shifts |
    |---|---|---|
    | the recipe | 4.5e-8 | 0.052 |
    | the recipe with `blur_pool` for the max pool | 3.7e-8 | 0.0048 |
    | the recipe without the max pool | 3e-8 | 3e-8 |
    | the convolutions, then a Linear on every position | 0.29 | 0.35 |

    The recipe's outputs are about 0.21 in size, so an odd shift moves them by
    a quarter. A Linear on every position throws the symmetry away at the
    last step. The max pool keeps shifts by 2 exact but breaks odd ones,
    since it pairs different neighbours.

    Zhang found the same in image classifiers, where small shifts changed
    their answers. His fix takes the larger of each pair at every position,
    blurs, and only then keeps every second one. The file gives it as
    `blur_pool`. Over 5 seeds it shrank the odd shift gap 7 to 13 times. It
    can't close it, since keeping every second position can't follow a shift
    by one exactly.
    """,
)

EQUIVARIANCE4_SOURCE = source_file(
    '''
    # equivariance4.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    signal = torch.tensor([[[1.0, 5.0, 2.0, 0.0]]])
    pooled = F.max_pool1d(signal, 2)        # tensor([[[5., 2.]]]), the larger of each pair
    averaged = pooled.mean(dim=2)          # tensor([[3.5000]]), one number per channel


''' + indented(BLUR_POOL_PART) + '''


    # TODO: Finish forward, the blueprint for shifts that wrap round. x has shape
    #   (batch, length). Give it a channel dimension, then run conv1 and a ReLU,
    #   a max pool of 2 with F.max_pool1d, conv2 and a ReLU, the mean over the
    #   positions and the head. The result has shape (batch, classes).
''' + indented(SHIFT_NET_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return how far model's outputs on x, of shape (batch, length),
    #   move under each shift from 0 up to length - 1, as a list of plain
    #   numbers. Each is the largest gap between the outputs on x shifted with
    #   torch.roll along its last dimension and the outputs on x. Record nothing
    #   for autograd.
    def shift_gaps(
        model: nn.Module,
        x: Tensor,
    ) -> list[float]:
        ...
    ''',
)

EQUIVARIANCE4_SOLUTION = source_file(
    '''
    # equivariance4.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    signal = torch.tensor([[[1.0, 5.0, 2.0, 0.0]]])
    pooled = F.max_pool1d(signal, 2)
    averaged = pooled.mean(dim=2)


''' + indented(BLUR_POOL_PART) + '''


''' + indented(SHIFT_NET_INIT) + '''

''' + indented(SHIFT_NET_FORWARD) + '''


''' + indented(SHIFT_GAPS_SOLUTION) + '''
    ''',
)


def check_shift_net(found: type) -> None:
    model = expect_module("ShiftNet(3)", seeded_shift_net(found))
    reference = seeded_shift_net(SHIFT_REFERENCE["ShiftNet"])
    signals = torch.randn(4, 16, generator=torch.Generator().manual_seed(781))

    with torch.no_grad():
        expected = reference(signals)

    expect_tensor(
        call="ShiftNet(3) on 4 signals of 16",
        actual=model(signals),
        expected=expected,
    )


def check_shift_gaps(found: object) -> None:
    signals = torch.randn(5, 12, generator=torch.Generator().manual_seed(782))

    for name, model in (
        ("ShiftNet(3)", seeded_shift_net(SHIFT_REFERENCE["ShiftNet"])),
        ("the convolutions with a Linear on every position", seeded_shift_net(lambda classes: FlatShiftNet(classes, 12), 783)),
    ):
        expected = SHIFT_REFERENCE["shift_gaps"](model, signals)
        recording: list[bool] = []
        hook = no_grad_recorder(model, recording)

        try:
            actual = found(model, signals)
        finally:
            hook.remove()

        if any(recording):
            raise Mismatch("shift_gaps ran the model while recording for autograd. Run it inside torch.no_grad().")

        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Mismatch(f"shift_gaps for {name} on signals of 12 should give a list of 12 numbers, one per shift.")

        for steps, (value, wanted) in enumerate(zip(actual, expected)):
            expect_close(
                call=f"the gap after a shift by {steps} for {name}",
                actual=value,
                expected=wanted,
                tolerance=1e-5,
            )


def check_equivariance4(checker: Checker) -> None:
    checker.test_method(
        class_name="ShiftNet",
        method="forward",
        description="convolves, pools locally, then averages over every position",
        run=check_shift_net,
    )
    checker.test(
        function="shift_gaps",
        description="measures the change after every shift",
        run=check_shift_gaps,
    )


EQUIVARIANCE4 = Exercise(
    name="equivariance4",
    chapter=EQUIVARIANCE,
    title="The blueprint",
    lesson=EQUIVARIANCE4_LESSON,
    source=EQUIVARIANCE4_SOURCE,
    solution=EQUIVARIANCE4_SOLUTION,
    hints={
        "ShiftNet.forward": "x.unsqueeze(1) gives the channel. torch.relu after each convolution, F.max_pool1d(y, 2) "
        "between them, then .mean(dim=2) before self.head.",
        "shift_gaps": "Inside torch.no_grad(), keep model(x) once. For each steps in range(x.shape[-1]), compare "
        "model(torch.roll(x, steps, dims=-1)) with it, through .abs().max().item().",
    },
    introduces=(),
    check=check_equivariance4,
)


# ---------------------------------------------------------------------------
# quiz13
# ---------------------------------------------------------------------------

QUIZ13_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 10. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `distance_set` | every pair's distance, as sorted_distances took them, then a Deep Set over those distances | equivariance3, sets1 |
    | `mean_messages` | the messages summed where they arrive with `index_add_`, divided by each node's count of arriving edges | message1, graph1 |
    | `gcn_step` | self loops, the GCN's two sided normalization, then the features mixed and weighted | message2 |
    | `pairwise_distances` | every difference of two points, as squared_distances made them, then each one's length with `.norm` | equivariance2, equivariance3 |
    | `shuffle_gap` | each set's own random order, as shuffle_each drew it, then the largest gap after reordering | sets2, sets3 |
    """,
)

QUIZ13_SOURCE = source_file(
    '''
    # quiz13.py
    #
    # No examples and no new tools. Each function combines moves from Section 10.

    import torch
    import torch.nn as nn
    from torch import Tensor


    # TODO: Return a Deep Set over the distances between the points of each
    #   set, for points of shape (sets, n, dims). Take the distance of every
    #   pair i < j as a column of shape (sets, pairs, 1), run phi on every
    #   distance, sum over the pairs, then run rho.
    def distance_set(
        points: Tensor,
        phi: nn.Module,
        rho: nn.Module,
    ) -> Tensor:
        ...


    # TODO: Return the mean of the messages that reach each node, of shape
    #   (nodes, features). messages has one row per edge, and edges, of shape
    #   (2, edges), holds each edge's source in row 0 and target in row 1. A
    #   node no edge reaches gets zeros.
    def mean_messages(
        messages: Tensor,
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return one GCN step. Add self loops to the adjacency, then scale it
    #   on both sides by one over the square root of each node's degree, with
    #   the self loop counted in the degree. Then multiply by h and by weight.
    #   adjacency is (nodes, nodes), h (nodes, features) and weight
    #   (features, out).
    def gcn_step(
        adjacency: Tensor,
        h: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the distance between every two of points, of shape
    #   (n, dims), as an (n, n) tensor.
    def pairwise_distances(points: Tensor) -> Tensor:
        ...


    # TODO: Return the largest change, as a plain number, between layer's output
    #   for x with each set's elements in its own random order, and layer's
    #   output for x put in that same order. x has shape (sets, elements,
    #   features). Draw the orders as shuffle_each did, from
    #   torch.rand(sets, elements, generator=generator).
    def shuffle_gap(
        layer: nn.Module,
        x: Tensor,
        generator: torch.Generator,
    ) -> float:
        ...
    ''',
)

QUIZ13_SOLUTION = source_file(
    '''
    # quiz13.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor


    def distance_set(
        points: Tensor,
        phi: nn.Module,
        rho: nn.Module,
    ) -> Tensor:
        i, j = torch.triu_indices(points.shape[1], points.shape[1], 1)
        distances = (points[:, i] - points[:, j]).norm(dim=-1)

        return rho(phi(distances.unsqueeze(-1)).sum(dim=1))


    def mean_messages(
        messages: Tensor,
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        totals = torch.zeros(nodes, messages.shape[1]).index_add_(0, edges[1], messages)
        counts = torch.bincount(edges[1], minlength=nodes).clamp(min=1)

        return totals / counts.unsqueeze(1)


    def gcn_step(
        adjacency: Tensor,
        h: Tensor,
        weight: Tensor,
    ) -> Tensor:
        looped = adjacency + torch.eye(len(adjacency))
        scale = looped.sum(dim=1) ** -0.5

        return scale.unsqueeze(1) * looped * scale.unsqueeze(0) @ h @ weight


    def pairwise_distances(points: Tensor) -> Tensor:
        return (points.unsqueeze(1) - points.unsqueeze(0)).norm(dim=-1)


    def shuffle_gap(
        layer: nn.Module,
        x: Tensor,
        generator: torch.Generator,
    ) -> float:
        order = torch.rand(x.shape[0], x.shape[1], generator=generator).argsort(dim=1)
        rows = torch.arange(len(x)).unsqueeze(1)

        return (layer(x[rows, order]) - layer(x)[rows, order]).abs().max().item()
    ''',
)


def quiz13_module(
    seed: int,
    module: nn.Module,
) -> nn.Module:
    torch.manual_seed(seed)

    return module


def reference_gcn_step(
    adjacency: Tensor,
    h: Tensor,
    weight: Tensor,
) -> Tensor:
    looped = adjacency + torch.eye(len(adjacency))
    scale = looped.sum(dim=1) ** -0.5

    return scale.unsqueeze(1) * looped * scale.unsqueeze(0) @ h @ weight


QUIZ13_EDGES = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 1]])
QUIZ13_ADJACENCY = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])


def reference_distance_set(
    points: Tensor,
    phi: nn.Module,
    rho: nn.Module,
) -> Tensor:
    i, j = torch.triu_indices(points.shape[1], points.shape[1], 1)

    return rho(phi((points[:, i] - points[:, j]).norm(dim=-1).unsqueeze(-1)).sum(dim=1))


def reference_mean_messages(
    messages: Tensor,
    edges: Tensor,
    nodes: int,
) -> Tensor:
    totals = torch.zeros(nodes, messages.shape[1]).index_add_(0, edges[1], messages)

    return totals / torch.bincount(edges[1], minlength=nodes).clamp(min=1).unsqueeze(1)


def reference_shuffle_gap(
    layer: nn.Module,
    x: Tensor,
    generator: torch.Generator,
) -> float:
    order = torch.rand(x.shape[0], x.shape[1], generator=generator).argsort(dim=1)
    rows = torch.arange(len(x)).unsqueeze(1)

    return (layer(x[rows, order]) - layer(x)[rows, order]).abs().max().item()


def check_quiz13(checker: Checker) -> None:
    phi = quiz13_module(960, nn.Sequential(nn.Linear(1, 8), nn.ReLU()))
    rho = quiz13_module(961, nn.Linear(8, 2))
    check_on_inputs(
        checker=checker,
        function="distance_set",
        description="runs a Deep Set over every pair's distance",
        inputs=(
            (small_tensor(962, 2, 5, 2) / 3, Described(phi, "a Linear from 1 to 8 and a ReLU"), Described(rho, "a Linear from 8 to 2")),
            (small_tensor(976, 3, 4, 3) / 3, Described(phi, "a Linear from 1 to 8 and a ReLU"), Described(rho, "a Linear from 8 to 2")),
        ),
        reference=reference_distance_set,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mean_messages",
        description="averages the messages arriving at each node",
        inputs=(
            (small_tensor(963, 5, 3), QUIZ13_EDGES, 4),
            # Nodes 4 and 5 get no message, so dividing by a count of 0 shows.
            (small_tensor(964, 5, 2), QUIZ13_EDGES, 6),
        ),
        reference=reference_mean_messages,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="gcn_step",
        description="normalizes with self loops on both sides, then mixes",
        inputs=(
            (QUIZ13_ADJACENCY, small_tensor(965, 4, 3), small_tensor(966, 3, 2)),
            # Some answers here are below 0, so a ReLU left over from message2's gcn_layer shows.
            (QUIZ13_ADJACENCY, small_tensor(965, 4, 3), small_tensor(975, 3, 2)),
        ),
        reference=reference_gcn_step,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="pairwise_distances",
        description="gives the distance between every two points",
        inputs=(
            (small_tensor(967, 4, 2),),
            (small_tensor(968, 3, 3),),
        ),
        reference=lambda points: torch.cdist(points, points),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="shuffle_gap",
        description="reorders each set its own way and compares the outputs",
        inputs=(
            (Described(quiz13_module(969, ReferenceEquivariantLayer(3, 3)), "sets3's equivariant Linear"), small_tensor(977, 4, 5, 3) / 3, seeded(978)),
            (Described(quiz13_module(970, AttentionLayer(3, 5, True)), "attention with positions added"), small_tensor(979, 4, 5, 3) / 3, seeded(980)),
        ),
        reference=reference_shuffle_gap,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


QUIZ13 = Exercise(
    name="quiz13",
    chapter=QUIZZES,
    title="Symmetry and graphs, start to finish",
    lesson=QUIZ13_LESSON,
    source=QUIZ13_SOURCE,
    solution=QUIZ13_SOLUTION,
    hints={
        "distance_set": "torch.triu_indices(n, n, 1) gives the pairs, as in sorted_distances, without the sort. "
        ".unsqueeze(-1) makes the column. phi on it, .sum(dim=1) over the pairs, then rho.",
        "mean_messages": "Sum the messages at their targets as message1's aggregate does. torch.bincount of edges[1] "
        "with minlength=nodes counts them, .clamp(min=1) keeps a count of 0 from dividing, and .unsqueeze(1) lines "
        "the counts up with the rows.",
        "gcn_step": "Add torch.eye to the adjacency and take each row's sum to the power -0.5. Scale the rows and the "
        "columns by it, then @ h @ weight.",
        "pairwise_distances": "points.unsqueeze(1) - points.unsqueeze(0) gives every difference. Take .norm over the "
        "last dimension.",
        "shuffle_gap": "Argsort the random numbers along dim=1 and index with torch.arange(len(x)).unsqueeze(1), as "
        "shuffle_each did. Reorder the input before layer and the output after, then .abs().max().item().",
    },
    introduces=(),
    check=check_quiz13,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons' tables
# ---------------------------------------------------------------------------


def measure_symmetry_gaps() -> dict[str, float]:
    """equivariance1's table. Random signals, rows and points, a random kernel and freshly made Linears, with a
    shift by 3, one fixed shuffle of 5 rows and a turn by 0.7 radians."""
    generator = torch.Generator().manual_seed(0)
    signals = torch.randn(4, 1, 16, generator=generator)
    kernel = torch.randn(1, 1, 3, generator=generator)
    rows = torch.randn(5, 3, generator=generator)
    points = torch.randn(6, 2, generator=generator)
    order = torch.randperm(5, generator=generator)
    torch.manual_seed(0)
    signal_linear = nn.Linear(16, 16)
    row_linear = nn.Linear(3, 4)
    flat_linear = nn.Linear(15, 4)
    point_linear = nn.Linear(2, 2)

    def shift(x: Tensor) -> Tensor:
        return torch.roll(x, 3, dims=-1)

    def shuffle(x: Tensor) -> Tensor:
        return x[order]

    def turn(x: Tensor) -> Tensor:
        return reference_rotate(x, 0.7)

    with torch.no_grad():
        return {
            "conv1d, circular padding": reference_equivariance_gap(lambda x: F.conv1d(F.pad(x, (1, 1), mode="circular"), kernel), shift, signals),
            "conv1d, zero padding": reference_equivariance_gap(lambda x: F.conv1d(x, kernel, padding=1), shift, signals),
            "Linear on the whole signal": reference_equivariance_gap(signal_linear, shift, signals),
            "sum of the rows": reference_invariance_gap(lambda x: x.sum(dim=0), shuffle, rows),
            "Linear on each row": reference_equivariance_gap(row_linear, shuffle, rows),
            "Linear on the rows flattened": reference_invariance_gap(lambda x: flat_linear(x.flatten()), shuffle, rows),
            "all distances": reference_invariance_gap(lambda p: torch.cdist(p, p), turn, points),
            "2 by 2 Linear on the points": reference_equivariance_gap(point_linear, turn, points),
        }


def train_taught(
    x: Tensor,
    y: Tensor,
    hidden: int,
    steps: int,
    rate: float,
    generator: torch.Generator,
) -> Callable[[Tensor], Tensor]:
    """train_shapes' taught model with a chosen width, length and rate, for the lesson's check on its size."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2 * SET_POINTS, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (64,), generator=generator)
        loss = F.cross_entropy(model(reference_scramble(x[batch], generator).flatten(1)), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(points: Tensor) -> Tensor:
        with torch.no_grad():
            return model(points.flatten(1)).argmax(dim=1)

    return predict


def turn_only(
    points: Tensor,
    generator: torch.Generator,
) -> Tensor:
    return points @ reference_random_rotations(len(points), generator).transpose(1, 2)


def unsorted_distances(points: Tensor) -> Tensor:
    i, j = torch.triu_indices(points.shape[1], points.shape[1], 1)

    return (points[:, i] - points[:, j]).norm(dim=-1)


def measure_built_in_or_taught(seeds: range = range(3)) -> dict[int, dict[str, float]]:
    """equivariance3's numbers by seed. Accuracy on 2,000 new noisy sets, and how many answers survive one more
    scramble, for 10 shapes turned and shuffled, the same shapes only turned, and one shape against its mirror."""
    results = {}

    def accuracy(
        predict: Callable[[Tensor], Tensor],
        x: Tensor,
        y: Tensor,
    ) -> float:
        return (predict(x) == y).float().mean().item()

    with pinned_threads():
        for seed in seeds:
            row = {}
            templates = shape_templates(10, seed)
            test_x, test_y = shape_sets(templates, 2000, torch.Generator().manual_seed(seed + 1000))
            scrambled = reference_scramble(test_x, torch.Generator().manual_seed(seed + 2000))
            turned = turn_only(test_x, torch.Generator().manual_seed(seed + 2000))

            for count in (20, 100, 1000):
                x, y = shape_sets(templates, count, torch.Generator().manual_seed(seed + 3000 + count))

                for kind in ("built in", "augmented"):
                    predict = train_shapes(kind, x, y, 10, 1500, torch.Generator().manual_seed(seed + 4000))
                    row[f"{kind} {count}"] = accuracy(predict, scrambled, test_y)
                    row[f"{kind} {count} agreement"] = reference_agreement(predict, scrambled, torch.Generator().manual_seed(seed + 5000))

            x, y = shape_sets(templates, 1000, torch.Generator().manual_seed(seed + 4000))
            generator = torch.Generator().manual_seed(seed + 4000)
            row["taught, 64 units, 6,000 steps"] = accuracy(train_taught(x, y, 64, 6000, 0.01, generator), scrambled, test_y)
            row["taught, 256 units, 6,000 steps"] = accuracy(train_taught(x, y, 256, 6000, 3e-3, generator), scrambled, test_y)

            x, y = shape_sets(templates, 1000, torch.Generator().manual_seed(seed + 3000 + 1000))
            row["turn only, taught"] = accuracy(train_shapes("augmented", x, y, 10, 1500, torch.Generator().manual_seed(seed + 4000), scramble=turn_only), turned, test_y)
            row["turn only, unsorted distances"] = accuracy(train_shapes("built in", x, y, 10, 1500, torch.Generator().manual_seed(seed + 4000), features=unsorted_distances), turned, test_y)
            row["turn only, sorted distances"] = accuracy(train_shapes("built in", x, y, 10, 1500, torch.Generator().manual_seed(seed + 4000)), turned, test_y)

            shape = shape_templates(1, seed + 7)
            pair = torch.cat([shape, shape * torch.tensor([-1.0, 1.0])])
            x, y = shape_sets(pair, 1000, torch.Generator().manual_seed(seed + 6000))
            mirror_x, mirror_y = shape_sets(pair, 2000, torch.Generator().manual_seed(seed + 6001))
            mirror_x = turn_only(mirror_x, torch.Generator().manual_seed(seed + 6002))
            row["mirror, taught"] = accuracy(train_shapes("augmented", x, y, 2, 1500, torch.Generator().manual_seed(seed + 4000), scramble=turn_only), mirror_x, mirror_y)
            row["mirror, unsorted distances"] = accuracy(train_shapes("built in", x, y, 2, 1500, torch.Generator().manual_seed(seed + 4000), features=unsorted_distances), mirror_x, mirror_y)
            row["mirror, sorted distances"] = accuracy(train_shapes("built in", x, y, 2, 1500, torch.Generator().manual_seed(seed + 4000)), mirror_x, mirror_y)
            results[seed] = row

    return results


class BlurShiftNet(nn.Module):
    """ShiftNet with Zhang's blur_pool in place of its max pool, for the lesson's comparison."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, padding_mode="circular")
        self.head = nn.Linear(8, classes)

    def forward(self, x: Tensor) -> Tensor:
        y = BLUR_REFERENCE["blur_pool"](torch.relu(self.conv1(x.unsqueeze(1))))

        return self.head(torch.relu(self.conv2(y)).mean(dim=2))


def measure_shift_gaps(seeds: range = range(5)) -> dict[int, dict[str, tuple[float, float]]]:
    """equivariance4's table by seed: for each untrained model, the largest change in any output over every even
    shift and over every odd one, on 32 random signals of length 16, and the outputs' average size."""
    makers = {
        "recipe": SHIFT_REFERENCE["ShiftNet"],
        "recipe with blur_pool": BlurShiftNet,
        "no max pool": UnpooledShiftNet,
        "Linear on every position": FlatShiftNet,
    }
    results = {}

    with pinned_threads(), torch.no_grad():
        for seed in seeds:
            signals = torch.randn(32, 16, generator=torch.Generator().manual_seed(seed))
            row = {}

            for name, make in makers.items():
                torch.manual_seed(seed)
                model = make(3)
                start = model(signals)
                gaps = [(model(torch.roll(signals, steps, dims=-1)) - start).abs().max().item() for steps in range(16)]
                row[name] = (max(gaps[0::2]), max(gaps[1::2]))

            torch.manual_seed(seed)
            row["recipe's output size"] = (SHIFT_REFERENCE["ShiftNet"](3)(signals).abs().mean().item(), 0.0)
            results[seed] = row

    return results
