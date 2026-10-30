"""52_positions, why attention needs positions, learned and sinusoidal ones added to the input, and RoPE, which turns queries and keys."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
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

        Attention compares every position with every other, and nothing in it
        knows which came first. This chapter shows that, adds positions to the
        input the two classic ways, learned and sinusoidal, then builds RoPE,
        which turns queries and keys by their position instead, and shows why
        most recent models use it.

        ## Further reading

        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Su and colleagues, RoFormer and rotary position embedding](https://arxiv.org/abs/2104.09864)
        - [EleutherAI, rotary embeddings explained](https://blog.eleuther.ai/rotary-embeddings/)
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
    Nothing in that knows where a position sits. Each output row depends on
    its own input row and on the set of the others, not their order.

    So shuffling the input only shuffles the output. Take an attention layer
    without the mask, 5 positions of 8 features, and the order 3, 0, 4, 1, 2.
    The output of the shuffled input and the shuffled output differ by at most
    1.2e-7, which is rounding.

    Indexing the steps dimension with a tensor of position numbers puts them
    in that order.

    ```python
    x = torch.tensor([[[1.0], [2.0], [3.0]]])    # one sequence of 3 positions
    x[:, torch.tensor([2, 0, 1])]                # tensor([[[3.], [1.], [2.]]])
    ```

    So "dog bites man" and "man bites dog" give the same three rows, in other
    places. A classifier that averages over the positions sees the same
    summary for both, up to rounding.

    A layer that works on each position alone is blind to order too, like a
    Linear, LayerNorm or an MLP. So a whole stack of them with attention is.

    The causal mask breaks this. A shuffled position sees other neighbours in
    its past, and here the outputs then differ by up to 2.51. Haviv and
    others found that language models with the mask and no positions at all
    learn to work out positions from it. Most models still give every
    position a signal of its own, which makes it easy, and position2 adds
    one.

    `torch.allclose(a, b, atol=1e-5)` says whether two tensors match within
    1e-5, as a plain True or False.
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


    # TODO: Return x, of shape (batch, steps, features), with its positions in
    #   the order that order, a tensor of position numbers, gives.
    def permute(
        x: Tensor,
        order: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return True when shuffling layer's input by order only shuffles its
    #   output the same way, within 1e-5, and False otherwise.
    def order_blind(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        order: Tensor,
    ) -> bool:
        ...


    # TODO: Return layer's output for x averaged over the positions, one row of
    #   features per sequence, the way a classifier might sum up a sentence.
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
        description="tells the layers that ignore order from the ones that don't",
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
        "order_blind": "Run layer on the shuffled input, and shuffle layer's output on x with permute. Then "
        "compare the two with torch.allclose and atol=1e-5.",
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
    The fix is to give each position a row of numbers of its own and add it to
    the input row there. Then the same input at two places looks different.
    There are two classic ways to get those rows.

    Learned positions are an `nn.Embedding` with one row per position, trained
    like any other weight. GPT-2 learns 1,024 of them. A longer sequence has
    no row to look up, so the model can't read it.

    Sinusoidal positions, from the 2017 paper, are fixed waves. Column pair i
    of the table holds a sine and a cosine at its own frequency.

    ```text
    table[p, 2i]     = sin(p * 10000^(-2i / size))
    table[p, 2i + 1] = cos(p * 10000^(-2i / size))
    ```

    The first pair turns fastest, one radian per position, and each pair
    after it turns slower. Every position gets its own pattern, and any
    position has one, however long the sequence.

    ```python
    p = torch.arange(4)       # tensor([0, 1, 2, 3])
    torch.sin(p / 2.0)        # tensor([0.0000, 0.4794, 0.8415, 0.9975])
    ```

    Their dot products depend only on how far apart two positions are. With a
    size of 16, here is each row against the rows a few places further on.

    | Row | Itself | 1 on | 2 on | 5 on | 20 on |
    |---|---|---|---|---|---|
    | 0 | 8.000 | 7.485 | 6.368 | 6.137 | 5.776 |
    | 1000 | 8.000 | 7.485 | 6.368 | 6.137 | 5.776 |

    Rows of a learned table start out random and have no such pattern until
    training gives them one. RoPE in rope1 builds on this property of the
    waves.
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


    # TODO: Return the steps by size table of sinusoidal positions from the
    #   lesson, with the sines in the even columns and the cosines in the odd
    #   ones. size is even.
    def sinusoidal_positions(
        steps: int,
        size: int,
    ) -> Tensor:
        ...


    # TODO: Finish forward. ids has shape (batch, steps). Return each id's row of
    #   self.ids plus its position's row of self.positions.
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


    # TODO: Return the dot product of each row of table with the row distance
    #   further down, one number for each row that has one.
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
        "sinusoidal_positions": "torch.arange(0, size, 2) / size gives 2i / size. Raise 10000 to minus that, multiply by "
        "torch.arange(steps).unsqueeze(1), then fill table[:, 0::2] with the sines and table[:, 1::2] with the cosines.",
        "PositionalEmbedding.forward": "torch.arange(ids.shape[1]) numbers the positions. Look up both tables and add.",
        "distance_scores": "table[: len(table) - distance] and table[distance:] line each row up with the one "
        "distance further down, distance 0 included. Multiply them and sum over dim=1.",
    },
    introduces=(),
    check=check_position2,
)


# ---------------------------------------------------------------------------
# rope1
# ---------------------------------------------------------------------------

ROPE1_LESSON = prose(
    """
    RoPE, rotary position embedding from Su and colleagues, adds nothing to
    the input. It turns the queries and keys by their position instead.

    The features go in pairs, (x0, x1), (x2, x3) and so on. At position m,
    pair i turns by the angle `m * theta_i`, with `theta_i = 10000^(-2i / size)`,
    position2's frequencies again. Turning a pair (a, b) by an angle phi gives
    a new pair.

    ```text
    a' = a cos(phi) - b sin(phi)
    b' = a sin(phi) + b cos(phi)
    ```

    ```python
    phi = torch.tensor(torch.pi / 2)    # a quarter turn
    1.0 * torch.cos(phi) - 0.0 * torch.sin(phi), 1.0 * torch.sin(phi) + 0.0 * torch.cos(phi)
    # (tensor(-4.3711e-08), tensor(1.)), so (1, 0) became (0, 1)
    ```

    `x[..., 0::2]` takes every other feature from the first, the a of each
    pair, and `x[..., 1::2]` the b. Stacking the new a and b on a last
    dimension and flattening the last two zips them back in order.

    ```python
    a, b = torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])
    torch.stack([a, b], dim=-1).flatten(-2)    # tensor([1., 2., 3., 4.])
    ```

    Only queries and keys turn. Values carry what a position says, and that
    shouldn't change with where it sits. RoPE has no weights to learn either.
    Llama, Mistral and most models since use it. Hugging Face's Llama code
    pairs feature i with feature i + size / 2 instead of its neighbour, the
    same idea in another order.
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


    # TODO: Return the angles of RoPE, of shape (steps, size // 2), where row m
    #   holds m * theta_i for each pair i.
    def rope_angles(
        steps: int,
        size: int,
        base: float = 10000.0,
    ) -> Tensor:
        ...


    # TODO: Return x, of shape (..., size), with each pair of neighbouring
    #   features turned by its angle in angles, of shape (..., size // 2).
    def rotate_pairs(
        x: Tensor,
        angles: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return x, of shape (..., steps, size), with RoPE applied, each
    #   position turned by its own row of rope_angles.
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
        description="turns each pair of neighbouring features by its angle",
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
        description="turns each position by its own angles",
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
    title="RoPE turns pairs of features",
    lesson=ROPE1_LESSON,
    source=ROPE1_SOURCE,
    solution=ROPE1_SOLUTION,
    hints={
        "rope_angles": "base ** (-torch.arange(0, size, 2) / size) gives one frequency per pair. Multiply by "
        "torch.arange(steps).unsqueeze(1), so each row is one position.",
        "rotate_pairs": "a is x[..., 0::2] and b is x[..., 1::2]. Compute the new a and b from the lesson's formula "
        "with torch.cos and torch.sin of angles, then stack on dim=-1 and flatten(-2).",
        "rope": "x.shape[-2] is the number of positions and x.shape[-1] the size. Pass rope_angles of them to "
        "rotate_pairs.",
    },
    introduces=(),
    check=check_rope1,
)


# ---------------------------------------------------------------------------
# rope2
# ---------------------------------------------------------------------------

ROPE2_LESSON = prose(
    """
    A score compares a query and a key with a dot product. With RoPE the query
    at position m has turned by m times each angle, and the key at n by n
    times. Two turned vectors compare the way the originals would after one
    turn by the difference. So the score depends on n - m, how far apart they
    are, and never on where they sit.

    Here is one query and one key of size 8, three positions apart, at four
    places in a sequence. The second row adds position2's sinusoidal rows to
    them instead, as GPT-2 adds its learned ones.

    | Positions | 0 and 3 | 10 and 13 | 100 and 103 | 1000 and 1003 |
    |---|---|---|---|---|
    | RoPE | -0.334 | -0.334 | -0.334 | -0.334 |
    | Added positions | 1.342 | 1.168 | 0.589 | 1.175 |

    With added positions the score mixes the content with where each one
    sits, so the same pair of words scores differently at every place. With
    RoPE a pattern learned at one place, like "look one back", means the same
    at position 5 and at position 500.

    The angles still grow with the position. A model trained on sequences of
    2,000 meets new angles past that. So reading much longer text needs care,
    like scaling the angles down.
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


    # TODO: Return the score of query q at position m against key k at position
    #   n, the dot product after RoPE turns each by its position.
    def rope_score(
        q: Tensor,
        k: Tensor,
        m: int,
        n: int,
    ) -> Tensor:
        ...


    # TODO: Return rope_score for q at each position m in starts against k at
    #   m + distance, as one tensor.
    def rope_scores_at_distance(
        q: Tensor,
        k: Tensor,
        distance: int,
        starts: list[int],
    ) -> Tensor:
        ...


    # TODO: Return the score of q plus table's row m against k plus table's row
    #   m + distance, for each m in starts, as one tensor.
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
        description="compares the turned query with the turned key",
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
        "rope_score": "rope_angles(max(m, n) + 1, len(q)) has a row for both positions. Turn q by row m and k by "
        "row n with rotate_pairs, then take their dot product with @.",
        "rope_scores_at_distance": "Call rope_score with m and m + distance for each m in starts, and torch.stack "
        "the scores.",
        "added_scores_at_distance": "Add table[m] to q and table[m + distance] to k, take the dot product for each m, "
        "and torch.stack the scores.",
    },
    introduces=(),
    check=check_rope2,
)
