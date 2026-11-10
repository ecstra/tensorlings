"""52_positions, why attention needs positions, learned and sinusoidal ones added to the input, and RoPE, which turns queries and keys."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

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
from .linear_layers import expect_module

POSITIONS = Chapter(
    folder="52_positions",
    title="Positions",
    readme=source_file(
        """
        # Positions

        Attention compares every position with every other position, but
        nothing in it knows which position came first. This chapter shows
        that. Then it adds position information to the input in the two
        classic ways, learned and sinusoidal. Last, it builds RoPE, which
        rotates the queries and keys by angles that depend on their position,
        and shows why most recent models use it.

        ## Further reading

        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Su and colleagues, RoFormer and rotary position embedding](https://arxiv.org/abs/2104.09864)
        - [EleutherAI, rotary embeddings explained](https://blog.eleuther.ai/rotary-embeddings/)
        - [Press, Smith and Lewis, train short, test long](https://arxiv.org/abs/2108.12409)
        """,
    ),
)


def projections(
    seed: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)

    return tuple(torch.randn(size, size, generator=generator) / math.sqrt(size) for _ in range(3))


def attention_layer(
    seed: int,
    size: int,
    causal: bool,
) -> Callable[[Tensor], Tensor]:
    """One head of attention with fixed random weights, the layer the lessons shuffle inputs through."""
    query_weights, key_weights, value_weights = projections(seed, size)

    def attend(x: Tensor) -> Tensor:
        scores = (x @ query_weights) @ (x @ key_weights).transpose(-2, -1) / math.sqrt(size)

        if causal:
            steps = x.shape[-2]
            scores = scores.masked_fill(torch.triu(torch.ones(steps, steps), diagonal=1).bool(), float("-inf"))

        return torch.softmax(scores, dim=-1) @ (x @ value_weights)

    return attend


# ---------------------------------------------------------------------------
# position1
# ---------------------------------------------------------------------------

POSITION1_LESSON = prose(
    """
    attention3 to attention5 built attention from queries, keys and values.
    Nothing in that computation uses the position of a row. Each output row
    depends on its own input row and on the other input rows as a set, not on
    their order.

    So if you shuffle the input rows, the output rows come out shuffled the
    same way, and nothing else changes. Take an attention layer without the
    causal mask, an input of 5 positions with 8 features, and the order 3, 0,
    4, 1, 2. Compare the layer's output on the shuffled input with the
    shuffled output of the original input. They differ by at most 1.2e-7,
    which is rounding error.

    To shuffle the positions, index the steps dimension with a tensor of
    position numbers. The result has the positions in that order.

    ```python
    x = torch.tensor([[[1.0], [2.0], [3.0]]])    # one sequence of 3 positions
    x[:, torch.tensor([2, 0, 1])]                # tensor([[[3.], [1.], [2.]]])
    ```

    So "dog bites man" and "man bites dog" give the same three output rows,
    only in a different order. A classifier that averages the outputs over
    the positions gets the same result for both sentences, up to rounding.

    A layer that works on each position separately, like a Linear, LayerNorm
    or an MLP, also ignores order. So a whole stack of such layers and
    attention ignores order too.

    The causal mask changes this. After a shuffle, a position has different
    rows before it, so it sees a different past. Here the outputs then differ
    by up to 2.51. Haviv and others found that language models with the
    causal mask and no position information at all learn to work out
    positions from the mask. Most models still give every position a signal
    of its own, which makes this easier. position2 adds such a signal.

    `torch.allclose(a, b, atol=1e-5)` returns a plain True or False, saying
    whether two tensors match within 1e-5 everywhere.
    """,
)

POSITION1_SOURCE = source_file(
    '''
    # position1.py

    from collections.abc import Callable

    import torch
    from torch import Tensor

    x = torch.tensor([[[1.0], [2.0], [3.0]]])
    reordered = x[:, torch.tensor([2, 0, 1])]    # tensor([[[3.], [1.], [2.]]])
    close = torch.allclose(torch.tensor([1.0]), torch.tensor([1.000001]), atol=1e-5)    # True


    # TODO: x has shape (batch, steps, features), and order is a tensor of
    #   position numbers. Return x with its positions rearranged, so that
    #   position i of the result is position order[i] of x.
    #   Example: for the x above, permute(x, torch.tensor([2, 0, 1])) returns
    #   tensor([[[3.], [1.], [2.]]]).
    def permute(
        x: Tensor,
        order: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return True if layer ignores order on x, and False otherwise. It
    #   ignores order when running layer on x shuffled by order gives the same
    #   result, within 1e-5, as shuffling layer(x) by order.
    def order_blind(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        order: Tensor,
    ) -> bool:
        ...


    # TODO: Return layer(x) averaged over the positions, which gives one row of
    #   features per sequence, of shape (batch, features). A classifier might
    #   summarize a sentence this way.
    def pooled(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

POSITION1_SOLUTION = source_file(
    '''
    # position1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    from torch import Tensor

    x = torch.tensor([[[1.0], [2.0], [3.0]]])
    reordered = x[:, torch.tensor([2, 0, 1])]
    close = torch.allclose(torch.tensor([1.0]), torch.tensor([1.000001]), atol=1e-5)


    def permute(
        x: Tensor,
        order: Tensor,
    ) -> Tensor:
        return x[:, order]


    def order_blind(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        order: Tensor,
    ) -> bool:
        return torch.allclose(layer(permute(x, order)), permute(layer(x), order), atol=1e-5)


    def pooled(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        return layer(x).mean(dim=1)
    ''',
)


def linear_layer(seed: int) -> Callable[[Tensor], Tensor]:
    torch.manual_seed(seed)
    linear = nn.Linear(4, 4)

    def run(x: Tensor) -> Tensor:
        with torch.no_grad():
            return linear(x)

    return run


def measure_shuffle_gaps() -> dict[str, float]:
    """position1's two numbers, the largest gap between the output of the shuffled input and the shuffled output,
    for 5 positions of 8 features in the order 3, 0, 4, 1, 2, without and with the causal mask."""
    x = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(1))
    order = torch.tensor([3, 0, 4, 1, 2])

    return {
        name: (layer(x[:, order]) - layer(x)[:, order]).abs().max().item()
        for name, layer in (
            ("without the mask", attention_layer(0, 8, causal=False)),
            ("with the causal mask", attention_layer(0, 8, causal=True)),
        )
    }


def check_position1(checker: Checker) -> None:
    order = torch.tensor([3, 0, 4, 1, 2])
    x = small_tensor(520, 2, 5, 4) / 3
    check_on_inputs(
        checker=checker,
        function="permute",
        description="puts the positions in the given order",
        inputs=(
            (small_tensor(521, 2, 4, 3), torch.tensor([1, 3, 0, 2])),
            (x, order),
        ),
        reference=lambda x, order: x[:, order],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="order_blind",
        description="tells apart the layers that ignore order and the ones that don't",
        inputs=(
            (Described(attention_layer(522, 4, causal=False), "an attention layer without the mask"), x, order),
            (Described(attention_layer(522, 4, causal=True), "an attention layer with the causal mask"), x, order),
            (Described(linear_layer(523), "an nn.Linear(4, 4)"), x, order),
            (Described(lambda x: x.cumsum(dim=1), "a running sum along the positions"), x, order),
        ),
        reference=lambda layer, x, order: torch.allclose(layer(x[:, order]), layer(x)[:, order], atol=1e-5),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="pooled",
        description="averages the layer's output over the positions",
        inputs=(
            (Described(attention_layer(524, 4, causal=False), "an attention layer without the mask"), x),
            (Described(attention_layer(525, 3, causal=True), "an attention layer with the causal mask"), small_tensor(526, 3, 4, 3) / 3),
        ),
        reference=lambda layer, x: layer(x).mean(dim=1),
        expect=expect_tensor,
    )


POSITION1 = Exercise(
    name="position1",
    chapter=POSITIONS,
    title="Attention ignores order",
    lesson=POSITION1_LESSON,
    source=POSITION1_SOURCE,
    solution=POSITION1_SOLUTION,
    hints={
        "permute": "Index the steps dimension, the second one, with order, as the lesson's example does.",
        "order_blind": "Run layer on the input shuffled with permute, and also shuffle layer(x) with permute. "
        "Then compare the two results with torch.allclose and atol=1e-5.",
        "pooled": "Run layer on x, then take the mean over dim=1, the positions.",
    },
    introduces=(),
    check=check_position1,
)


# ---------------------------------------------------------------------------
# position2
# ---------------------------------------------------------------------------

POSITION2_LESSON = prose(
    """
    The fix is to give each position its own vector of numbers and add it to
    the input row at that position. Then the same token at two different
    positions gives two different rows. There are two classic ways to make
    these position vectors.

    **Learned positions** are an `nn.Embedding` with one row per position,
    trained like any other weight. GPT-2 learns 1,024 of them. A position past
    1,023 has no row to look up, so the model cannot read a longer sequence.

    **Sinusoidal positions**, from the 2017 paper by Vaswani and colleagues,
    are fixed waves that are computed, not learned. The table has one row per
    position. Columns 2i and 2i + 1 form pair i, which holds a sine and a
    cosine at the pair's own frequency.

    ```text
    table[p, 2i]     = sin(p * 10000^(-2i / size))
    table[p, 2i + 1] = cos(p * 10000^(-2i / size))
    ```

    For pair 0 the angle grows fastest, by one radian per position. For each
    later pair the angle grows more slowly. So every position gets its own
    pattern of values, and the formula gives a row for any position, however
    long the sequence.

    ```python
    p = torch.arange(4)       # tensor([0, 1, 2, 3])
    torch.sin(p / 2.0)        # tensor([0.0000, 0.4794, 0.8415, 0.9975])
    torch.arange(0, 8, 2)     # tensor([0, 2, 4, 6]), from 0 up to 8 in steps of 2
    ```

    The dot product of two rows of this table depends only on how far apart
    the two positions are. Here the table has size 16, and each cell is the
    dot product of a row with a row a few positions later.

    | Row | Itself | 1 later | 2 later | 5 later | 20 later |
    |---|---|---|---|---|---|
    | 0 | 8.000 | 7.485 | 6.368 | 6.137 | 5.776 |
    | 1000 | 8.000 | 7.485 | 6.368 | 6.137 | 5.776 |

    The rows of a learned table start out random, and they have no such
    pattern unless training gives them one. RoPE, in rope1, builds on this
    property of the waves.

    Vaswani's team found that the two kinds gave nearly identical results.
    They chose the waves in the hope that the model could read longer texts
    than it was trained on. Press and colleagues later found that a
    sinusoidal model handles only a few dozen tokens past its training length
    before it gets worse. So with either kind, train on the lengths the model
    will need to read.
    """,
)

POSITION2_SOURCE = source_file(
    '''
    # position2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    p = torch.arange(4)       # tensor([0, 1, 2, 3])
    waves = torch.sin(p / 2.0)    # tensor([0.0000, 0.4794, 0.8415, 0.9975])
    evens = torch.arange(0, 8, 2)    # tensor([0, 2, 4, 6])


    # TODO: Return the table of sinusoidal positions from the lesson, of shape
    #   (steps, size). Use the lesson's formula, with the sines in the even
    #   columns and the cosines in the odd columns. size is even.
    #   Example: sinusoidal_positions(2, 2) returns
    #   tensor([[0.0000, 1.0000], [0.8415, 0.5403]]).
    def sinusoidal_positions(
        steps: int,
        size: int,
    ) -> Tensor:
        ...


    # TODO: Finish forward. ids has shape (batch, steps). For each entry, return
    #   the row of self.ids for its id plus the row of self.positions for its
    #   position. The result has shape (batch, steps, size).
    class PositionalEmbedding(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            max_steps: int,
            size: int,
        ) -> None:
            super().__init__()
            self.ids = nn.Embedding(vocabulary, size)
            self.positions = nn.Embedding(max_steps, size)

        def forward(self, ids: Tensor) -> Tensor:
            ...


    # TODO: table has shape (rows, size). For each row i that has a row
    #   i + distance, return the dot product of row i with row i + distance.
    #   The result has shape (rows - distance,).
    #   Example: for sinusoidal_positions(30, 8) and distance 5, distance_scores
    #   returns 25 numbers, each about 3.16.
    def distance_scores(
        table: Tensor,
        distance: int,
    ) -> Tensor:
        ...
    ''',
)

POSITION2_SOLUTION = source_file(
    '''
    # position2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    p = torch.arange(4)
    waves = torch.sin(p / 2.0)
    evens = torch.arange(0, 8, 2)


    def sinusoidal_positions(
        steps: int,
        size: int,
    ) -> Tensor:
        frequencies = 10000 ** (-torch.arange(0, size, 2) / size)
        angles = torch.arange(steps).unsqueeze(1) * frequencies
        table = torch.zeros(steps, size)
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)

        return table


    class PositionalEmbedding(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            max_steps: int,
            size: int,
        ) -> None:
            super().__init__()
            self.ids = nn.Embedding(vocabulary, size)
            self.positions = nn.Embedding(max_steps, size)

        def forward(self, ids: Tensor) -> Tensor:
            steps = torch.arange(ids.shape[1])

            return self.ids(ids) + self.positions(steps)


    def distance_scores(
        table: Tensor,
        distance: int,
    ) -> Tensor:
        return (table[: len(table) - distance] * table[distance:]).sum(dim=1)
    ''',
)


def reference_sinusoidal(
    steps: int,
    size: int,
) -> Tensor:
    angles = torch.arange(steps).unsqueeze(1) * 10000 ** (-torch.arange(0, size, 2) / size)

    return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)


def measure_distance_table() -> dict[int, list[float]]:
    """position2's table, rows 0 and 1000 of a size 16 sinusoidal table against the rows 0, 1, 2, 5 and 20 on."""
    table = reference_sinusoidal(1021, 16)

    return {row: [(table[row] @ table[row + distance]).item() for distance in (0, 1, 2, 5, 20)] for row in (0, 1000)}


class ReferencePositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        max_steps: int,
        size: int,
    ) -> None:
        super().__init__()
        self.ids = nn.Embedding(vocabulary, size)
        self.positions = nn.Embedding(max_steps, size)

    def forward(self, ids: Tensor) -> Tensor:
        return self.ids(ids) + self.positions.weight[: ids.shape[1]]


def seeded_embedding(make: type) -> nn.Module:
    torch.manual_seed(527)

    return make(10, 8, 4)


def check_positional_embedding(found: type) -> None:
    model = expect_module("PositionalEmbedding(10, 8, 4)", seeded_embedding(found))
    ids = torch.randint(0, 10, (2, 5), generator=torch.Generator().manual_seed(528))

    with torch.no_grad():
        expected = seeded_embedding(ReferencePositionalEmbedding)(ids)
        actual = model(ids)

    expect_tensor(
        call=f"PositionalEmbedding(10, 8, 4) on ids {ids.tolist()}",
        actual=actual,
        expected=expected,
    )


def check_position2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="sinusoidal_positions",
        description="gives sines in the even columns and cosines in the odd ones",
        inputs=(
            (5, 4),
            (7, 8),
            (3, 2),
        ),
        reference=reference_sinusoidal,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="PositionalEmbedding",
        method="forward",
        description="adds each position's row to each id's row",
        run=check_positional_embedding,
    )
    check_on_inputs(
        checker=checker,
        function="distance_scores",
        description="gives each row's dot product with the row distance further down",
        inputs=(
            (small_tensor(529, 5, 3), 1),
            (small_tensor(530, 6, 4), 2),
            (Described(reference_sinusoidal(30, 8), "sinusoidal_positions(30, 8)"), 5),
            (small_tensor(531, 4, 3), 0),
        ),
        reference=lambda table, distance: (table[: len(table) - distance] * table[distance:]).sum(dim=1),
        expect=expect_tensor,
    )


POSITION2 = Exercise(
    name="position2",
    chapter=POSITIONS,
    title="Learned and sinusoidal positions",
    lesson=POSITION2_LESSON,
    source=POSITION2_SOURCE,
    solution=POSITION2_SOLUTION,
    hints={
        "sinusoidal_positions": "torch.arange(0, size, 2) / size gives 2i / size for every pair. Raise 10000 to minus "
        "that to get the frequencies, and multiply them by torch.arange(steps).unsqueeze(1) to get the angles. Then "
        "fill table[:, 0::2] with the sines and table[:, 1::2] with the cosines.",
        "PositionalEmbedding.forward": "torch.arange(ids.shape[1]) gives the position numbers. Look up the ids in "
        "self.ids and the position numbers in self.positions, and add the two results.",
        "distance_scores": "table[: len(table) - distance] and table[distance:] pair each row with the row distance "
        "further down. This also works for distance 0. Multiply the two slices and sum over dim=1.",
    },
    introduces=(),
    check=check_position2,
)


# ---------------------------------------------------------------------------
# rope1
# ---------------------------------------------------------------------------

ROPE1_LESSON = prose(
    """
    **RoPE**, rotary position embedding, comes from Su and colleagues. It adds
    nothing to the input. Instead it rotates each query and key by angles
    that depend on its position.

    The features are grouped in pairs, (x0, x1), (x2, x3) and so on, and each
    pair is treated as a point in the plane. At position m, pair i is rotated
    by the angle `m * theta_i`, where `theta_i = 10000^(-2i / size)`. These
    are the same frequencies as in position2. Rotating a pair (a, b) by an
    angle phi gives a new pair (a', b').

    ```text
    a' = a cos(phi) - b sin(phi)
    b' = a sin(phi) + b cos(phi)
    ```

    ```python
    phi = torch.tensor(torch.pi / 2)    # a quarter turn
    1.0 * torch.cos(phi) - 0.0 * torch.sin(phi), 1.0 * torch.sin(phi) + 0.0 * torch.cos(phi)
    # (tensor(-4.3711e-08), tensor(1.)), so (1, 0) became (0, 1)
    ```

    In an index, `...` stands for all the dimensions before the ones you
    name. So `x[..., 0]` is the first feature of every row, whatever the shape
    of x. A shape written `(..., size)` means any number of dimensions
    followed by a last dimension of length `size`.

    `x[..., 0::2]` takes every other feature, starting from the first, which
    gives the a of each pair. `x[..., 1::2]` gives the b of each pair. To put
    the new values back in their original order, stack them on a new last
    dimension and flatten the last two dimensions.

    ```python
    a, b = torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])
    torch.stack([a, b], dim=-1).flatten(-2)    # tensor([1., 2., 3., 4.])
    ```

    Only the queries and keys are rotated. The values carry the content of a
    position, and that content should not depend on where the position is.
    RoPE also has no weights to learn. Llama, Mistral and most models since
    use it. Hugging Face's Llama code pairs feature i with feature
    i + size / 2 instead of with its neighbour. That is the same idea with
    the features in another order.
    """,
)

ROPE1_SOURCE = source_file(
    '''
    # rope1.py

    import torch
    from torch import Tensor

    phi = torch.tensor(torch.pi / 2)
    turned = (1.0 * torch.cos(phi) - 0.0 * torch.sin(phi), 1.0 * torch.sin(phi) + 0.0 * torch.cos(phi))    # about (0, 1)
    zipped = torch.stack([torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])], dim=-1).flatten(-2)    # tensor([1., 2., 3., 4.])


    # TODO: Return the angles of RoPE, of shape (steps, size // 2). Entry [m, i]
    #   is m * theta_i, the angle of pair i at position m, where
    #   theta_i = base ** (-2i / size).
    #   Example: rope_angles(3, 4) returns
    #   tensor([[0.0000, 0.0000], [1.0000, 0.0100], [2.0000, 0.0200]]).
    def rope_angles(
        steps: int,
        size: int,
        base: float = 10000.0,
    ) -> Tensor:
        ...


    # TODO: x has shape (..., size) and angles has shape (..., size // 2).
    #   Return x with each pair of neighbouring features, x[..., 2i] and
    #   x[..., 2i + 1], rotated by angles[..., i] with the lesson's formula.
    #   Example: a quarter turn of the pair (1, 0) gives about (0, 1), so
    #   rotate_pairs(torch.tensor([1.0, 0.0]), torch.tensor([torch.pi / 2]))
    #   returns about tensor([0., 1.]).
    def rotate_pairs(
        x: Tensor,
        angles: Tensor,
    ) -> Tensor:
        ...


    # TODO: x has shape (..., steps, size). Return x with RoPE applied, so the
    #   features at position m are rotated by row m of rope_angles.
    def rope(
        x: Tensor,
        base: float = 10000.0,
    ) -> Tensor:
        ...
    ''',
)

ROPE_FUNCTIONS = '''
def rope_angles(
    steps: int,
    size: int,
    base: float = 10000.0,
) -> Tensor:
    frequencies = base ** (-torch.arange(0, size, 2) / size)

    return torch.arange(steps).unsqueeze(1) * frequencies


def rotate_pairs(
    x: Tensor,
    angles: Tensor,
) -> Tensor:
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = torch.cos(angles), torch.sin(angles)

    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_ROPE_FUNCTIONS = textwrap.indent(ROPE_FUNCTIONS.strip("\n"), "    ")

ROPE1_SOLUTION = source_file(
    '''
    # rope1.py, a solution for comparison

    import torch
    from torch import Tensor

    phi = torch.tensor(torch.pi / 2)
    turned = (1.0 * torch.cos(phi) - 0.0 * torch.sin(phi), 1.0 * torch.sin(phi) + 0.0 * torch.cos(phi))
    zipped = torch.stack([torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])], dim=-1).flatten(-2)


''' + INDENTED_ROPE_FUNCTIONS + '''


    def rope(
        x: Tensor,
        base: float = 10000.0,
    ) -> Tensor:
        return rotate_pairs(x, rope_angles(x.shape[-2], x.shape[-1], base))
    ''',
)


def complex_rope(
    x: Tensor,
    angles: Tensor,
) -> Tensor:
    """RoPE the way Meta's first Llama code wrote it, each pair as one complex number times a unit turn."""
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    turned = pairs * torch.polar(torch.ones_like(angles), angles)

    return torch.view_as_real(turned).flatten(-2)


def reference_angles(
    steps: int,
    size: int,
    base: float = 10000.0,
) -> Tensor:
    return torch.outer(torch.arange(steps).float(), 1.0 / base ** (torch.arange(0, size, 2).float() / size))


def check_rope1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="rope_angles",
        description="gives position times each pair's frequency",
        inputs=(
            (4, 4),
            (6, 8, 100.0),
        ),
        reference=reference_angles,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rotate_pairs",
        description="rotates each pair of neighbouring features by its angle",
        inputs=(
            (torch.tensor([1.0, 0.0, 0.0, 2.0]), torch.tensor([math.pi / 2, math.pi])),
            (small_tensor(531, 3, 6), small_tensor(532, 3, 3) / 2),
        ),
        reference=complex_rope,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rope",
        description="rotates each position by its own angles",
        inputs=(
            (small_tensor(533, 5, 4),),
            (small_tensor(534, 2, 6, 8), 100.0),
        ),
        reference=lambda x, base=10000.0: complex_rope(x, reference_angles(x.shape[-2], x.shape[-1], base)),
        expect=expect_tensor,
    )


ROPE1 = Exercise(
    name="rope1",
    chapter=POSITIONS,
    title="RoPE rotates pairs of features",
    lesson=ROPE1_LESSON,
    source=ROPE1_SOURCE,
    solution=ROPE1_SOLUTION,
    hints={
        "rope_angles": "base ** (-torch.arange(0, size, 2) / size) gives one frequency per pair. Multiply it by "
        "torch.arange(steps).unsqueeze(1), so that each row belongs to one position.",
        "rotate_pairs": "a is x[..., 0::2] and b is x[..., 1::2]. Compute the new a and b with the lesson's formula, "
        "using torch.cos and torch.sin of angles. Then stack them on dim=-1 and call flatten(-2).",
        "rope": "x.shape[-2] is the number of positions and x.shape[-1] is the size. Compute rope_angles from them "
        "and pass the result to rotate_pairs together with x.",
    },
    introduces=(),
    check=check_rope1,
)


# ---------------------------------------------------------------------------
# rope2
# ---------------------------------------------------------------------------

ROPE2_LESSON = prose(
    """
    An attention score is the dot product of a query and a key. With RoPE,
    pair i of the query at position m is rotated by `m * theta_i`, and pair i
    of the key at position n by `n * theta_i`. Rotating both vectors changes
    their dot product the same way as rotating only the key, by the
    difference `(n - m) * theta_i`. So the score depends on `n - m`, how far
    apart the two positions are, and never on where they are in the sequence.

    The table below uses one query and one key of size 8, three positions
    apart, at four places in a sequence. The first row applies RoPE. The
    second row instead adds the sinusoidal rows of position2 to the query and
    the key, the way GPT-2 adds its learned rows.

    | Positions | 0 and 3 | 10 and 13 | 100 and 103 | 1000 and 1003 |
    |---|---|---|---|---|
    | RoPE | -0.334 | -0.334 | -0.334 | -0.334 |
    | Added positions | 1.342 | 1.168 | 0.589 | 1.175 |

    With added positions, the score mixes the content with the positions
    themselves, so the same pair of words gets a different score at every
    place. With RoPE, a pattern learned at one place, like "look at the
    previous token", works the same at position 5 and at position 500.

    A trained model shows this. One attention layer was trained to output the
    previous token, on sequences at positions 0 to 31. Then it was tested on
    new sequences at positions 0 to 31 and at positions 500 to 531. Here is
    the fraction it got right, over three seeds.

    | Positions | At 0 to 31 | At 500 to 531 |
    |---|---|---|
    | learned | 100% | 15.7% to 19.0% |
    | sinusoidal, added | 100% | 40.1% to 63.6% |
    | RoPE | 100% | 100% |

    The learned rows for positions past 31 were never trained. The added
    waves take values at those positions that the layer never saw. RoPE's
    scores depend only on the distance, so the layer behaves the same at
    every place.

    The angles still grow with the position, so a model trained on 2,000
    positions meets angles it never saw beyond that length. Position
    interpolation, by Chen and colleagues, scales every position down so it
    fits in the trained range. With under 1,000 steps of fine tuning, it
    extended LLaMA from 2,048 to 32,768 positions.
    """,
)

ROPE2_SOURCE = source_file(
    '''
    # rope2.py

    import torch
    from torch import Tensor

    query, key = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    score = query @ key    # tensor(0.), a dot product


    # rope_angles and rotate_pairs from rope1.
''' + INDENTED_ROPE_FUNCTIONS + '''


    # TODO: q and k are vectors of the same even size. Rotate q by the RoPE
    #   angles of position m, and k by the RoPE angles of position n. Return
    #   the dot product of the two rotated vectors.
    def rope_score(
        q: Tensor,
        k: Tensor,
        m: int,
        n: int,
    ) -> Tensor:
        ...


    # TODO: For each position m in starts, compute rope_score with q at
    #   position m and k at position m + distance. Return the scores stacked
    #   into one tensor of shape (len(starts),).
    def rope_scores_at_distance(
        q: Tensor,
        k: Tensor,
        distance: int,
        starts: list[int],
    ) -> Tensor:
        ...


    # TODO: For each m in starts, add row m of table to q and row m + distance
    #   of table to k, then take the dot product of the two sums. Return the
    #   scores stacked into one tensor of shape (len(starts),).
    def added_scores_at_distance(
        q: Tensor,
        k: Tensor,
        distance: int,
        starts: list[int],
        table: Tensor,
    ) -> Tensor:
        ...
    ''',
)

ROPE2_SOLUTION = source_file(
    '''
    # rope2.py, a solution for comparison

    import torch
    from torch import Tensor

    query, key = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    score = query @ key


''' + INDENTED_ROPE_FUNCTIONS + '''


    def rope_score(
        q: Tensor,
        k: Tensor,
        m: int,
        n: int,
    ) -> Tensor:
        angles = rope_angles(max(m, n) + 1, len(q))

        return rotate_pairs(q, angles[m]) @ rotate_pairs(k, angles[n])


    def rope_scores_at_distance(
        q: Tensor,
        k: Tensor,
        distance: int,
        starts: list[int],
    ) -> Tensor:
        return torch.stack([rope_score(q, k, m, m + distance) for m in starts])


    def added_scores_at_distance(
        q: Tensor,
        k: Tensor,
        distance: int,
        starts: list[int],
        table: Tensor,
    ) -> Tensor:
        return torch.stack([(q + table[m]) @ (k + table[m + distance]) for m in starts])
    ''',
)


def reference_rope_score(
    q: Tensor,
    k: Tensor,
    m: int,
    n: int,
) -> Tensor:
    angles = reference_angles(max(m, n) + 1, len(q))

    return complex_rope(q, angles[m]) @ complex_rope(k, angles[n])


def measure_rope_scores() -> dict[str, list[float]]:
    """rope2's first table, one query and one key of size 8, three positions apart, at four places."""
    generator = torch.Generator().manual_seed(3)
    q, k = torch.randn(8, generator=generator), torch.randn(8, generator=generator)
    table = reference_sinusoidal(1003 + 1, 8)
    starts = (0, 10, 100, 1000)

    return {
        "RoPE": [reference_rope_score(q, k, m, m + 3).item() for m in starts],
        "Added positions": [((q + table[m]) @ (k + table[m + 3])).item() for m in starts],
    }


LOOK_BACK_TOKENS = 16
LOOK_BACK_SIZE = 32
LOOK_BACK_STEPS = 32


class LookBack(nn.Module):
    """One head of causal attention and a head, trained to give the token one place back, with learned positions,
    added sinusoidal ones or RoPE. start says where the sequence sits, so it can be read at positions it never
    trained on."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(LOOK_BACK_TOKENS, LOOK_BACK_SIZE)

        if kind == "learned":
            self.positions = nn.Embedding(1024, LOOK_BACK_SIZE)

        self.query = nn.Linear(LOOK_BACK_SIZE, LOOK_BACK_SIZE, bias=False)
        self.key = nn.Linear(LOOK_BACK_SIZE, LOOK_BACK_SIZE, bias=False)
        self.value = nn.Linear(LOOK_BACK_SIZE, LOOK_BACK_SIZE, bias=False)
        self.head = nn.Linear(LOOK_BACK_SIZE, LOOK_BACK_TOKENS)

    def forward(
        self,
        tokens: Tensor,
        start: int = 0,
    ) -> Tensor:
        steps = tokens.shape[1]
        x = self.embed(tokens)

        if self.kind == "learned":
            x = x + self.positions(torch.arange(start, start + steps))
        elif self.kind == "sinusoidal":
            x = x + reference_sinusoidal(start + steps, LOOK_BACK_SIZE)[start:]

        q, k = self.query(x), self.key(x)

        if self.kind == "rope":
            angles = reference_angles(start + steps, LOOK_BACK_SIZE)[start:]
            q, k = complex_rope(q, angles), complex_rope(k, angles)

        scores = q @ k.transpose(-2, -1) / math.sqrt(LOOK_BACK_SIZE)
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)

        return self.head(torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ self.value(x))


def look_back_accuracy(
    model: nn.Module,
    tokens: Tensor,
    start: int,
) -> float:
    """The share of positions after the first whose guess is the token one place back."""
    with torch.no_grad():
        guesses = model(tokens, start).argmax(dim=-1)

    return (guesses[:, 1:] == tokens[:, :-1]).float().mean().item()


def measure_shifted_reading(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[tuple[float, float]]]:
    """rope2's second table. Each kind of position trains 1,500 steps of Adam at 3e-3 on batches of 64 sequences
    at positions 0 to 31, then reads 500 new sequences there and at positions 500 to 531. Measured at 4 threads."""
    results: dict[str, list[tuple[float, float]]] = {}
    test = torch.randint(0, LOOK_BACK_TOKENS, (500, LOOK_BACK_STEPS), generator=torch.Generator().manual_seed(7))

    for kind in ("learned", "sinusoidal", "rope"):
        for seed in seeds:
            torch.manual_seed(seed)
            model = LookBack(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            generator = torch.Generator().manual_seed(seed + 100)

            for _ in range(1500):
                tokens = torch.randint(0, LOOK_BACK_TOKENS, (64, LOOK_BACK_STEPS), generator=generator)
                logits = model(tokens)[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, LOOK_BACK_TOKENS), tokens[:, :-1].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            results.setdefault(kind, []).append((look_back_accuracy(model, test, 0), look_back_accuracy(model, test, 500)))

    return results


def expect_close_scores(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """Angles of a hundred radians or more lose digits to float32 rounding, and a learner may compute them in
    another order, so these scores are compared within 1e-3."""
    if isinstance(actual, Tensor) and actual.shape == expected.shape and torch.allclose(actual.double(), expected.double(), atol=1e-3):
        return

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_rope2(checker: Checker) -> None:
    q, k = small_tensor(535, 8) / 2, small_tensor(536, 8) / 2
    check_on_inputs(
        checker=checker,
        function="rope_score",
        description="takes the dot product of the rotated query and the rotated key",
        inputs=(
            (q, k, 2, 5),
            (small_tensor(537, 4), small_tensor(538, 4), 7, 1),
        ),
        reference=reference_rope_score,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rope_scores_at_distance",
        description="gives the same score at every place for the same distance",
        inputs=(
            (q, k, 3, [0, 10, 100]),
            (small_tensor(539, 6), small_tensor(540, 6), 1, [4, 9]),
        ),
        reference=lambda q, k, distance, starts: torch.stack([reference_rope_score(q, k, m, m + distance) for m in starts]),
        expect=expect_close_scores,
    )
    table = Described(reference_sinusoidal(120, 8), "sinusoidal_positions(120, 8)")
    check_on_inputs(
        checker=checker,
        function="added_scores_at_distance",
        description="scores the query and key with their positions added",
        inputs=(
            (q, k, 3, [0, 10, 100], table),
        ),
        reference=lambda q, k, distance, starts, table: torch.stack([(q + table[m]) @ (k + table[m + distance]) for m in starts]),
        expect=expect_tensor,
    )


ROPE2 = Exercise(
    name="rope2",
    chapter=POSITIONS,
    title="Why RoPE",
    lesson=ROPE2_LESSON,
    source=ROPE2_SOURCE,
    solution=ROPE2_SOLUTION,
    hints={
        "rope_score": "rope_angles(max(m, n) + 1, len(q)) has a row for both positions. Rotate q by row m and k by "
        "row n with rotate_pairs, then take their dot product with @.",
        "rope_scores_at_distance": "Call rope_score with m and m + distance for each m in starts, and join the "
        "scores with torch.stack.",
        "added_scores_at_distance": "For each m, add table[m] to q and table[m + distance] to k, and take the dot "
        "product. Join the scores with torch.stack.",
    },
    introduces=(),
    check=check_rope2,
)
