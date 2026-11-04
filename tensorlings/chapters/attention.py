"""47_attention, each position as a weighted average of the ones before it, from plain averages to queries, keys and values, and why it recalls what an RNN can't."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import fastest_seconds, pinned_threads

ATTENTION = Chapter(
    folder="47_attention",
    title="Attention",
    readme=source_file(
        """
        # Attention

        Section 7 is about attention, the way each position of a sequence
        gathers what it needs from the others by a weighted average. This
        chapter starts from plain averages of the past, turns the weights into
        a masked softmax over scores, makes the scores from queries and keys,
        and averages values. It ends with a recall task an RNN fails and
        attention solves.

        ## Further reading

        - [Karpathy, let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Bahdanau, Cho and Bengio, learning to align and translate](https://arxiv.org/abs/1409.0473)
        """,
    ),
)


# ---------------------------------------------------------------------------
# attention1
# ---------------------------------------------------------------------------

ATTENTION1_LESSON = prose(
    """
    An RNN carries the past forward in one fixed set of numbers. Section 7
    takes another road. Each position looks back at every position before it
    and takes a weighted average of them. This is **attention**.

    It starts with the plainest weights, equal ones. Position t becomes the
    average of positions 0 up to t, itself included. A position never looks
    ahead, so a model that predicts the next step can't peek at the answer.

    With a loop, each position slices the rows up to its own and takes their
    mean.

    ```python
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [2.0, 4.0]])
    x[:2].mean(dim=0)    # tensor([2., 1.]), the average for position 1
    ```

    The loop does one position at a time. A matrix does them all at once.
    Row t of a weight matrix holds position t's weights, and the matrix times
    `x` gives every average together. For equal weights over the past, row t
    has `1 / (t + 1)` in its first t + 1 places and 0 after.

    `torch.tril` keeps the lower triangle of a matrix, the diagonal and below,
    and sets the rest to 0.

    ```python
    torch.tril(torch.ones(3, 3))    # tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
    ```

    Dividing each row by its sum makes it average. With a batch of sequences,
    `x` has shape `(batch, steps, features)`, and the same matrix works on
    every sequence through `@`, by broadcasting.
    """,
)

ATTENTION1_SOURCE = source_file(
    '''
    # attention1.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [2.0, 4.0]])
    so_far = x[:2].mean(dim=0)    # tensor([2., 1.])
    lower = torch.tril(torch.ones(3, 3))    # tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])


    # TODO: Return each position of x, of shape (steps, features), replaced by
    #   the average of the positions up to and including it, with a loop.
    def average_so_far_loop(x: Tensor) -> Tensor:
        ...


    # TODO: Return the steps by steps matrix whose row t holds equal weights on
    #   positions 0 up to t, summing to 1, and 0 after.
    def averaging_matrix(steps: int) -> Tensor:
        ...


    # TODO: Return the same averages as average_so_far_loop for x of shape
    #   (batch, steps, features), with averaging_matrix and no loop.
    def average_so_far(x: Tensor) -> Tensor:
        ...
    ''',
)

ATTENTION1_SOLUTION = source_file(
    '''
    # attention1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [2.0, 4.0]])
    so_far = x[:2].mean(dim=0)
    lower = torch.tril(torch.ones(3, 3))


    def average_so_far_loop(x: Tensor) -> Tensor:
        return torch.stack([x[: t + 1].mean(dim=0) for t in range(len(x))])


    def averaging_matrix(steps: int) -> Tensor:
        weights = torch.tril(torch.ones(steps, steps))

        return weights / weights.sum(dim=1, keepdim=True)


    def average_so_far(x: Tensor) -> Tensor:
        return averaging_matrix(x.shape[1]) @ x
    ''',
)


def reference_average_so_far(x: Tensor) -> Tensor:
    return x.cumsum(dim=-2) / torch.arange(1, x.shape[-2] + 1).unsqueeze(1)


def reference_averaging_matrix(steps: int) -> Tensor:
    weights = torch.tril(torch.ones(steps, steps))

    return weights / weights.sum(dim=1, keepdim=True)


def check_attention1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="average_so_far_loop",
        description="averages each position with the ones before it",
        inputs=(
            (small_tensor(470, 4, 2),),
            (small_tensor(471, 6, 3),),
        ),
        reference=reference_average_so_far,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="averaging_matrix",
        description="gives equal weights on the past in each row",
        inputs=(
            (3,),
            (5,),
        ),
        reference=reference_averaging_matrix,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="average_so_far",
        description="averages every sequence of a batch at once",
        inputs=(
            (small_tensor(472, 2, 4, 3),),
            (small_tensor(473, 3, 5, 2),),
        ),
        reference=reference_average_so_far,
        expect=expect_tensor,
    )


ATTENTION1 = Exercise(
    name="attention1",
    chapter=ATTENTION,
    title="Averages of the past",
    lesson=ATTENTION1_LESSON,
    source=ATTENTION1_SOURCE,
    solution=ATTENTION1_SOLUTION,
    hints={
        "average_so_far_loop": "For each t in range(len(x)), take x[: t + 1].mean(dim=0). Stack the results.",
        "averaging_matrix": "torch.tril(torch.ones(steps, steps)), divided by its row sums with keepdim=True.",
        "average_so_far": "Build averaging_matrix for the number of steps, x.shape[1], and multiply x by it from "
        "the left with @.",
    },
    introduces=("attention",),
    check=check_attention1,
)


# ---------------------------------------------------------------------------
# attention2
# ---------------------------------------------------------------------------

ATTENTION2_LESSON = prose(
    """
    Equal weights treat every past position the same. A model needs to weight
    some more than others, so the weights should come from scores, through a
    softmax, as soft1's soft pick did.

    The future still has to stay out. Setting a future weight to 0 after the
    softmax would leave rows that no longer sum to 1. So the scores of future
    positions are set to minus infinity before it, and the softmax turns
    those into weights of exactly 0.

    ```python
    torch.softmax(torch.tensor([1.0, 2.0, float("-inf")]), dim=0)    # tensor([0.2689, 0.7311, 0.0000])
    ```

    `torch.triu` with `diagonal=1` keeps what lies above the diagonal, the
    future of each row. `.bool()` turns its ones into True.

    ```python
    torch.triu(torch.ones(3, 3), diagonal=1).bool()
    # tensor([[False,  True,  True], [False, False,  True], [False, False, False]])
    ```

    `scores.masked_fill(mask, value)` puts `value` wherever `mask` is True.

    With every score 0, the masked softmax gives back attention1's averaging
    matrix exactly. Any other scores give other weights, still on the past
    alone and still summing to 1 in each row. attention3 computes the scores
    from the data.
    """,
)

ATTENTION2_SOURCE = source_file(
    '''
    # attention2.py

    import torch
    from torch import Tensor

    weights = torch.softmax(torch.tensor([1.0, 2.0, float("-inf")]), dim=0)    # tensor([0.2689, 0.7311, 0.0000])
    above = torch.triu(torch.ones(2, 2), diagonal=1).bool()    # tensor([[False, True], [False, False]])


    # TODO: Return the steps by steps mask that is True where a position would
    #   look at the future, above the diagonal, and False elsewhere.
    def future_mask(steps: int) -> Tensor:
        ...


    # TODO: Return the weights from scores, of shape (..., steps, steps), with
    #   the future set to minus infinity and a softmax over each row.
    def masked_weights(scores: Tensor) -> Tensor:
        ...


    # TODO: Return each position of x, of shape (..., steps, features), as the
    #   average of the past weighted by masked_weights of scores.
    def weighted_average(
        scores: Tensor,
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

ATTENTION2_SOLUTION = source_file(
    '''
    # attention2.py, a solution for comparison

    import torch
    from torch import Tensor

    weights = torch.softmax(torch.tensor([1.0, 2.0, float("-inf")]), dim=0)
    above = torch.triu(torch.ones(2, 2), diagonal=1).bool()


    def future_mask(steps: int) -> Tensor:
        return torch.triu(torch.ones(steps, steps), diagonal=1).bool()


    def masked_weights(scores: Tensor) -> Tensor:
        hidden = scores.masked_fill(future_mask(scores.shape[-1]), float("-inf"))

        return torch.softmax(hidden, dim=-1)


    def weighted_average(
        scores: Tensor,
        x: Tensor,
    ) -> Tensor:
        return masked_weights(scores) @ x
    ''',
)


def reference_masked_weights(scores: Tensor) -> Tensor:
    future = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()

    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


def expect_bool_mask(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype != torch.bool:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but masked_fill needs a mask of True and False. Add .bool().")


def check_attention2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="future_mask",
        description="marks the future of each row",
        inputs=(
            (3,),
            (5,),
        ),
        reference=lambda steps: torch.triu(torch.ones(steps, steps), diagonal=1).bool(),
        expect=expect_bool_mask,
    )
    check_on_inputs(
        checker=checker,
        function="masked_weights",
        description="gives softmax weights on the past alone",
        inputs=(
            (small_tensor(474, 4, 4),),
            (small_tensor(475, 2, 3, 3),),
            (torch.zeros(4, 4),),
        ),
        reference=reference_masked_weights,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="weighted_average",
        description="averages the past with those weights",
        inputs=(
            (small_tensor(476, 4, 4), small_tensor(477, 4, 2)),
            (small_tensor(478, 2, 3, 3), small_tensor(479, 2, 3, 4)),
        ),
        reference=lambda scores, x: reference_masked_weights(scores) @ x,
        expect=expect_tensor,
    )


ATTENTION2 = Exercise(
    name="attention2",
    chapter=ATTENTION,
    title="Weights from scores",
    lesson=ATTENTION2_LESSON,
    source=ATTENTION2_SOURCE,
    solution=ATTENTION2_SOLUTION,
    hints={
        "future_mask": "torch.triu(torch.ones(steps, steps), diagonal=1), then .bool().",
        "masked_weights": "scores.masked_fill(future_mask(scores.shape[-1]), float(\"-inf\")), then torch.softmax "
        "with dim=-1.",
        "weighted_average": "Get the weights from masked_weights, then multiply x by them from the left with @.",
    },
    introduces=(),
    check=check_attention2,
)


def seeded_module(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


# ---------------------------------------------------------------------------
# attention3
# ---------------------------------------------------------------------------

ATTENTION3_LESSON = prose(
    """
    attention2 took its scores as given. Now each position makes them from its
    own features, with two Linear layers, as derive2 did with its two tables.

    - A **query** says what a position is looking for.
    - A **key** says what a position holds.

    Position t's score for position s is the dot product of t's query and s's
    key. With one row per position in `q` and in `k`, one matrix product gives
    every score at once, of shape `(steps, steps)`. `.transpose(-2, -1)` swaps
    the last two dimensions, so it works with a batch in front too.

    ```python
    q = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    q @ q.transpose(-2, -1)    # tensor([[1., 0.], [0., 4.]])
    ```

    The scores then get divided by the square root of d, the size of a query.
    A dot product sums d products. With random entries of spread 1, its spread
    grows like the square root of d. Big scores make the softmax nearly a hard
    pick, as soft2's bigger query did, and a hard pick passes almost no
    gradient back.

    Here are random queries and keys, with 8 keys for each query.

    | Size d | Spread of the scores | Largest weight | Spread, divided | Largest weight, divided |
    |---|---|---|---|---|
    | 4 | 1.99 | 0.529 | 0.99 | 0.344 |
    | 16 | 4.02 | 0.748 | 1.01 | 0.359 |
    | 64 | 8.00 | 0.875 | 1.00 | 0.361 |
    | 256 | 15.99 | 0.938 | 1.00 | 0.362 |

    Without the division, a bigger model starts with sharper weights, at 256
    already putting 94% on one key. With it, the spread stays near 1 at every
    size. The model starts soft and learns how sharp to be.
    """,
)

ATTENTION3_SOURCE = source_file(
    '''
    # attention3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    q = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    scores = q @ q.transpose(-2, -1)    # tensor([[1., 0.], [0., 4.]])
    root = math.sqrt(16)                 # 4.0


    # masked_weights from attention2.
    def masked_weights(scores: Tensor) -> Tensor:
        future = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    # TODO: Return the scores of the queries q against the keys k, of shapes
    #   (..., q steps, d) and (..., k steps, d), divided by the square root of
    #   d. The result has shape (..., q steps, k steps).
    def attention_scores(
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the average, over every row of scores, of the largest
    #   softmax weight in that row, as a plain number.
    def mean_largest_weight(scores: Tensor) -> float:
        ...


    # TODO: Finish ScoreHead. __init__ makes self.query, then self.key, each an
    #   nn.Linear from features to size with no bias. forward gives the masked
    #   weights of x, of shape (..., steps, features), from attention_scores of
    #   its queries and keys.
    class ScoreHead(nn.Module):
        def __init__(
            self,
            features: int,
            size: int,
        ) -> None:
            super().__init__()
            self.query = ...
            self.key = ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

ATTENTION3_SOLUTION = source_file(
    '''
    # attention3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    q = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    scores = q @ q.transpose(-2, -1)
    root = math.sqrt(16)


    def masked_weights(scores: Tensor) -> Tensor:
        future = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    def attention_scores(
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


    def mean_largest_weight(scores: Tensor) -> float:
        return torch.softmax(scores, dim=-1).amax(dim=-1).mean().item()


    class ScoreHead(nn.Module):
        def __init__(
            self,
            features: int,
            size: int,
        ) -> None:
            super().__init__()
            self.query = nn.Linear(features, size, bias=False)
            self.key = nn.Linear(features, size, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return masked_weights(attention_scores(self.query(x), self.key(x)))
    ''',
)


def reference_scores(
    q: Tensor,
    k: Tensor,
) -> Tensor:
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


class ReferenceScoreHead(nn.Module):
    def __init__(
        self,
        features: int,
        size: int,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(features, size, bias=False)
        self.key = nn.Linear(features, size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return reference_masked_weights(reference_scores(self.query(x), self.key(x)))


def check_score_head(found: type) -> None:
    for features, size in ((4, 3), (6, 8)):
        call = f"ScoreHead({features}, {size})"
        head = expect_module(call, seeded_module(found, features, size))
        layers = [getattr(head, name, None) for name in ("query", "key")]

        if not all(
            isinstance(layer, nn.Linear) and layer.bias is None and tuple(layer.weight.shape) == (size, features)
            for layer in layers
        ):
            raise Mismatch(f"{call} should hold self.query and self.key, each an nn.Linear from {features} to {size} with bias=False.")

        reference = seeded_module(ReferenceScoreHead, features, size)

        # The check seeds the weights before __init__ runs, so layers made in another order start from other numbers.
        if not (torch.equal(head.query.weight, reference.query.weight) and torch.equal(head.key.weight, reference.key.weight)):
            raise Mismatch(f"{call} starts from other weights than expected. Make self.query first, then self.key, since the check seeds the weights in that order.")

        x = small_tensor(480 + size, 2, 5, features) / 2

        with torch.no_grad():
            expected = reference(x)

        expect_tensor(
            call=f"{call} on a batch of shape (2, 5, {features})",
            actual=head(x),
            expected=expected,
        )


def check_attention3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="attention_scores",
        description="gives every dot product, divided by the square root of d",
        inputs=(
            (small_tensor(481, 3, 4), small_tensor(482, 3, 4)),
            (small_tensor(483, 2, 4), small_tensor(484, 5, 4)),
            (small_tensor(485, 2, 3, 9), small_tensor(486, 2, 3, 9)),
        ),
        reference=reference_scores,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mean_largest_weight",
        description="averages each row's largest softmax weight",
        inputs=(
            (small_tensor(487, 4, 6),),
            (small_tensor(488, 2, 3, 5) / 3,),
        ),
        reference=lambda scores: torch.softmax(scores, dim=-1).amax(dim=-1).mean().item(),
        expect=expect_close,
    )
    checker.test_method(
        class_name="ScoreHead",
        method="forward",
        description="makes queries and keys, then the masked weights",
        run=check_score_head,
    )


ATTENTION3 = Exercise(
    name="attention3",
    chapter=ATTENTION,
    title="Queries and keys",
    lesson=ATTENTION3_LESSON,
    source=ATTENTION3_SOURCE,
    solution=ATTENTION3_SOLUTION,
    hints={
        "attention_scores": "q @ k.transpose(-2, -1), then divide by math.sqrt(q.shape[-1]).",
        "mean_largest_weight": "torch.softmax with dim=-1, then .amax(dim=-1) for each row's largest weight, then "
        ".mean().item().",
        "ScoreHead.forward": "In __init__, nn.Linear(features, size, bias=False) twice. In forward, pass "
        "self.query(x) and self.key(x) to attention_scores, then to masked_weights.",
    },
    introduces=(),
    check=check_attention3,
)


# ---------------------------------------------------------------------------
# attention4
# ---------------------------------------------------------------------------

ATTENTION4_LESSON = prose(
    """
    ScoreHead's weights would average x itself. A full head makes a third
    Linear, the **value** of each position, what it passes on when it is
    picked. The weights average the values, and an output Linear maps the
    result back to the features size.

    Why make queries, keys and values at all, instead of comparing x with
    itself? There are two reasons.

    Compared with itself, every position becomes its own best match. A row's
    dot product with itself is the sum of its squares, always large, while two
    different rows mix signs and partly cancel. Here is the weight each of 8
    positions puts on itself, for random rows with no future mask.

    | Features | Weight on itself |
    |---|---|
    | 4 | 0.418 |
    | 16 | 0.775 |
    | 64 | 0.992 |

    At 64 features each position takes back 99% of itself and learns almost
    nothing from the others. Queries and keys from their own Linears can make
    any position match any other.

    Training shows the difference. Here a model copies the token right before
    each position, out of 16 kinds, in sequences of 12. It adds a learned row
    per position to each token, so it can tell positions apart. One version
    compares x with itself and averages x, and the other is SingleHead. Each
    trained for 600 steps, over three seeds.

    | Attention | Previous token right |
    |---|---|
    | x with itself | 57.3% to 83.0% |
    | SingleHead | 100% on every seed |

    Without the Linears a position keeps finding itself. 19.0% to 44.5% of
    its answers were its own token, against 6.3% by chance.

    The features that make a good match are often not the ones worth passing
    on. A name is found by "this is a name", and passes on which name it is.
    Separate keys and values let a position be found by one thing and pass
    on another.

    The output Linear merges what came back into the size the rest of the
    model uses. In 50_multihead it also mixes several heads together.

    `.diagonal(dim1=-2, dim2=-1)` gives the diagonal of each matrix in a
    batch.

    ```python
    pair = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    pair.diagonal(dim1=-2, dim2=-1)    # tensor([[1., 4.], [5., 8.]])
    ```
    """,
)

ATTENTION4_SOURCE = source_file(
    '''
    # attention4.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    pair = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    diagonals = pair.diagonal(dim1=-2, dim2=-1)    # tensor([[1., 4.], [5., 8.]])


    # masked_weights from attention2.
    def masked_weights(scores: Tensor) -> Tensor:
        future = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    # attention_scores from attention3.
    def attention_scores(
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


    # TODO: Return the average weight each position of x, of shape (..., steps,
    #   features), puts on itself when x is its own queries and keys, with no
    #   future mask, as a plain number.
    def self_weight(x: Tensor) -> float:
        ...


    # TODO: Finish SingleHead. __init__ adds self.value, an nn.Linear from
    #   features to size with no bias, then self.out, an nn.Linear from size
    #   back to features with its bias. forward averages the values with the
    #   masked weights from the queries and keys, then runs self.out.
    class SingleHead(nn.Module):
        def __init__(
            self,
            features: int,
            size: int,
        ) -> None:
            super().__init__()
            self.query = nn.Linear(features, size, bias=False)
            self.key = nn.Linear(features, size, bias=False)
            self.value = ...
            self.out = ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

ATTENTION4_SOLUTION = source_file(
    '''
    # attention4.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    pair = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    diagonals = pair.diagonal(dim1=-2, dim2=-1)


    def masked_weights(scores: Tensor) -> Tensor:
        future = torch.triu(torch.ones(scores.shape[-1], scores.shape[-1]), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    def attention_scores(
        q: Tensor,
        k: Tensor,
    ) -> Tensor:
        return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


    def self_weight(x: Tensor) -> float:
        weights = torch.softmax(attention_scores(x, x), dim=-1)

        return weights.diagonal(dim1=-2, dim2=-1).mean().item()


    class SingleHead(nn.Module):
        def __init__(
            self,
            features: int,
            size: int,
        ) -> None:
            super().__init__()
            self.query = nn.Linear(features, size, bias=False)
            self.key = nn.Linear(features, size, bias=False)
            self.value = nn.Linear(features, size, bias=False)
            self.out = nn.Linear(size, features)

        def forward(self, x: Tensor) -> Tensor:
            weights = masked_weights(attention_scores(self.query(x), self.key(x)))

            return self.out(weights @ self.value(x))
    ''',
)


def reference_self_weight(x: Tensor) -> float:
    return torch.softmax(reference_scores(x, x), dim=-1).diagonal(dim1=-2, dim2=-1).mean().item()


class ReferenceSingleHead(nn.Module):
    def __init__(
        self,
        features: int,
        size: int,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(features, size, bias=False)
        self.key = nn.Linear(features, size, bias=False)
        self.value = nn.Linear(features, size, bias=False)
        self.out = nn.Linear(size, features)

    def forward(self, x: Tensor) -> Tensor:
        weights = reference_masked_weights(reference_scores(self.query(x), self.key(x)))

        return self.out(weights @ self.value(x))


def check_single_head(found: type) -> None:
    for features, size in ((4, 3), (6, 8)):
        call = f"SingleHead({features}, {size})"
        head = expect_module(call, seeded_module(found, features, size))
        value, out = getattr(head, "value", None), getattr(head, "out", None)

        if not isinstance(value, nn.Linear) or value.bias is not None or tuple(value.weight.shape) != (size, features):
            raise Mismatch(f"{call} should hold self.value, an nn.Linear from {features} to {size} with bias=False.")

        if not isinstance(out, nn.Linear) or tuple(out.weight.shape) != (features, size):
            raise Mismatch(f"{call} should hold self.out, an nn.Linear from {size} back to {features}.")

        if out.bias is None:
            raise Mismatch(f"{call} should keep the bias of self.out. Only the query, key and value Linears leave theirs out.")

        reference = seeded_module(ReferenceSingleHead, features, size)

        # The check seeds the weights before __init__ runs, so layers made in another order start from other numbers.
        if not (torch.equal(value.weight, reference.value.weight) and torch.equal(out.weight, reference.out.weight)):
            raise Mismatch(f"{call} starts from other weights than expected. Make self.value, then self.out, after the given layers, since the check seeds the weights in that order.")

        x = small_tensor(490 + size, 2, 5, features) / 2

        with torch.no_grad():
            expected = reference(x)

        expect_tensor(
            call=f"{call} on a batch of shape (2, 5, {features})",
            actual=head(x),
            expected=expected,
        )


def check_attention4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="self_weight",
        description="gives the average weight each position puts on itself",
        inputs=(
            (small_tensor(491, 6, 4) / 2,),
            (small_tensor(492, 3, 5, 8) / 2,),
        ),
        reference=reference_self_weight,
        expect=expect_close,
    )
    checker.test_method(
        class_name="SingleHead",
        method="forward",
        description="averages the values, then runs the output layer",
        run=check_single_head,
    )


ATTENTION4 = Exercise(
    name="attention4",
    chapter=ATTENTION,
    title="Values and the output layer",
    lesson=ATTENTION4_LESSON,
    source=ATTENTION4_SOURCE,
    solution=ATTENTION4_SOLUTION,
    hints={
        "self_weight": "torch.softmax of attention_scores(x, x) with dim=-1, then .diagonal(dim1=-2, dim2=-1), "
        "then .mean().item().",
        "SingleHead.forward": "The weights are masked_weights of attention_scores(self.query(x), self.key(x)). "
        "Multiply self.value(x) by them from the left with @, then run self.out.",
    },
    introduces=(),
    check=check_attention4,
)


# ---------------------------------------------------------------------------
# attention5
# ---------------------------------------------------------------------------

ATTENTION5_LESSON = prose(
    """
    attention3 and attention4 built every piece. Put together, they are
    scaled dot product attention.

    ```text
    attention(q, k, v) = softmax(q kᵀ / √d) v
    ```

    With the future masked, it is the causal kind a model that writes text
    uses. Without the mask, every position sees every other, which suits a
    model that reads a whole sentence at once.

    The keys and values can come from a sequence of another length than the
    queries, as long as the keys and values match each other. Then the scores
    have shape `(q steps, k steps)`, and 51_cross_attention uses that. The
    causal mask only makes sense when all three come from the same sequence.

    PyTorch has the whole computation as one call,
    `F.scaled_dot_product_attention(q, k, v)`, which masks the future when
    given `is_causal=True`. It runs a fused kernel, one piece of code that
    does every step in a single pass. FlashAttention is the best known one.
    It never stores the full weight matrix, which is large. At 2,048 steps and
    4 heads it holds 16,777,216 numbers, 67 MB in float32, for one sequence.

    PyTorch has such a kernel for this CPU too. Here it ran at that size with
    the causal mask and heads of 64. Over three runs the call took 3.6 to 4.2
    milliseconds, and the same work by hand 14 to 17.
    """,
)

ATTENTION5_SOURCE = source_file(
    '''
    # attention5.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    q = torch.ones(1, 3, 4)
    together = F.scaled_dot_product_attention(q, q, q)    # shape (1, 3, 4)


    # TODO: Return scaled dot product attention of q, of shape (..., q steps, d),
    #   over k and v, of shapes (..., k steps, d) and (..., k steps, size), by
    #   hand. When causal is True, mask the future first.
    def attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        ...


    # TODO: Return the same with F.scaled_dot_product_attention.
    def fast_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        ...


    # TODO: Return how many bytes the full weight matrices take for a batch of
    #   sequences with this many steps and heads, at 4 bytes a number.
    def weight_bytes(
        batch: int,
        heads: int,
        steps: int,
    ) -> int:
        ...
    ''',
)

ATTENTION5_SOLUTION = source_file(
    '''
    # attention5.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    q = torch.ones(1, 3, 4)
    together = F.scaled_dot_product_attention(q, q, q)


    def attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

        if causal:
            future = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1]), diagonal=1).bool()
            scores = scores.masked_fill(future, float("-inf"))

        return torch.softmax(scores, dim=-1) @ v


    def fast_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


    def weight_bytes(
        batch: int,
        heads: int,
        steps: int,
    ) -> int:
        return batch * heads * steps * steps * 4
    ''',
)


def calls_to_attention(run: Callable[[], object]) -> list[dict[str, object]]:
    """Runs run while F.scaled_dot_product_attention notes the keyword arguments of each call, so a check can tell
    PyTorch's call from attention worked out by hand. The module's own name is swapped, which the learner's F reads."""
    original = F.scaled_dot_product_attention
    calls: list[dict[str, object]] = []

    def recording(*arguments: object, **keywords: object) -> Tensor:
        calls.append(keywords)

        return original(*arguments, **keywords)

    F.scaled_dot_product_attention = recording

    try:
        run()
    finally:
        F.scaled_dot_product_attention = original

    return calls


def check_attention_by_hand(found: object) -> None:
    q, k, v = small_tensor(509, 2, 4, 3) / 2, small_tensor(510, 2, 4, 3) / 2, small_tensor(511, 2, 4, 2)

    if calls_to_attention(lambda: found(q, k, v, True)):
        raise Mismatch(
            "attention called F.scaled_dot_product_attention. Work it out by hand here, with the scores, the mask and "
            "the softmax, and leave the call to fast_attention.",
        )


def check_fast_attention_calls(found: object) -> None:
    q, k, v = small_tensor(509, 2, 4, 3) / 2, small_tensor(510, 2, 4, 3) / 2, small_tensor(511, 2, 4, 2)

    if not calls_to_attention(lambda: found(q, k, v, True)):
        raise Mismatch("fast_attention never called F.scaled_dot_product_attention. Pass it q, k and v, with causal as its is_causal.")


def check_attention5(checker: Checker) -> None:
    attention_inputs = (
        (small_tensor(500, 4, 3) / 2, small_tensor(501, 4, 3) / 2, small_tensor(502, 4, 2), True),
        (small_tensor(503, 2, 5, 4) / 2, small_tensor(504, 2, 5, 4) / 2, small_tensor(505, 2, 5, 3), True),
        (small_tensor(506, 2, 4) / 2, small_tensor(507, 6, 4) / 2, small_tensor(508, 6, 3), False),
    )
    check_on_inputs(
        checker=checker,
        function="attention",
        description="matches F.scaled_dot_product_attention, causal or not",
        inputs=attention_inputs,
        reference=lambda q, k, v, causal: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
        expect=expect_tensor,
    )
    checker.test(
        function="attention",
        description="works it out by hand",
        run=check_attention_by_hand,
    )
    check_on_inputs(
        checker=checker,
        function="fast_attention",
        description="calls F.scaled_dot_product_attention",
        inputs=attention_inputs,
        reference=lambda q, k, v, causal: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
        expect=expect_tensor,
    )
    checker.test(
        function="fast_attention",
        description="calls F.scaled_dot_product_attention",
        run=check_fast_attention_calls,
    )
    check_on_inputs(
        checker=checker,
        function="weight_bytes",
        description="counts 4 bytes for every weight",
        inputs=(
            (1, 4, 2048),
            (8, 12, 1024),
        ),
        reference=lambda batch, heads, steps: batch * heads * steps * steps * 4,
        expect=expect_value,
    )


ATTENTION5 = Exercise(
    name="attention5",
    chapter=ATTENTION,
    title="Scaled dot product attention",
    lesson=ATTENTION5_LESSON,
    source=ATTENTION5_SOURCE,
    solution=ATTENTION5_SOLUTION,
    hints={
        "attention": "The scores are q @ k.transpose(-2, -1) over math.sqrt(q.shape[-1]). When causal, fill the "
        "future, torch.triu of ones with diagonal=1, with float(\"-inf\"). Then softmax with dim=-1, then @ v.",
        "fast_attention": "Pass q, k and v to F.scaled_dot_product_attention, with causal as its is_causal.",
        "weight_bytes": "Each sequence and head has a steps by steps matrix, and each number takes 4 bytes.",
    },
    introduces=(),
    check=check_attention5,
)


# ---------------------------------------------------------------------------
# attention6
# ---------------------------------------------------------------------------

RECALL_IDS = 16


def recall_sequences(
    count: int,
    pairs: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pairs of a key and a value from 16 ids, the keys all different, a query that is one of the keys, and its value."""
    keys = torch.stack([torch.randperm(RECALL_IDS, generator=generator)[:pairs] for _ in range(count)])
    values = torch.randint(0, RECALL_IDS, (count, pairs), generator=generator)
    picked = torch.randint(0, pairs, (count,), generator=generator)
    rows = torch.arange(count)

    return keys, values, keys[rows, picked], values[rows, picked]


ATTENTION6_LESSON = prose(
    """
    This task shows why attention exists. A sequence holds pairs, each a key
    and a value, both one of 16 ids. Then comes a query, one of the keys, and
    the answer is the value stored with it.

    An RNN reads the pairs one at a time, then the query. It has to keep
    every pair in its fixed hidden state, before it knows which one the query
    will ask for. Attention keeps every pair and compares the query with each
    key when the query comes, as derive2 did.

    `GRURecaller` in the file is the RNN, and its `hidden` sets how many
    numbers its state holds. `Recaller` is the attention, with queries, keys
    and values of 16. Each GRU trained for 3,000 steps and the attention for
    400. Here is the share of new sequences each got right, over three seeds.

    | Pairs | GRU of 64 | GRU of 256 | Attention |
    |---|---|---|---|
    | 2 | 0.878 to 0.965 | 0.994 to 0.999 | 1.000 |
    | 4 | 0.296 to 0.345 | 0.823 to 0.854 | 1.000 |
    | 8 | 0.244 to 0.275 | 0.225 to 0.274 | 1.000 |
    | 16 | 0.163 to 0.169 | 0.164 to 0.169 | 1.000 |

    Guessing would get 1 in 16 right, 0.0625.

    A state of 64 numbers holds 2 pairs, and one of 256 holds 4. Neither
    holds 8. Every pair is squeezed into the same fixed state, so a bigger
    state holds more pairs, but any fixed size runs out. Attention's memory
    grows with the sequence, one key and one value per pair, so no pair is
    squeezed out. The price is that the query is compared with every key, so
    the work grows with the sequence too. 48_linear_attention looks at that
    trade.
    """,
)

GRU_RECALLER = '''
# The RNN to compare with, given. It reads each pair's key and value, then the query's key as one more step,
# and answers from its last state. hidden is the size of that state.
class GRURecaller(nn.Module):
    def __init__(
        self,
        ids: int,
        size: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.keys = nn.Embedding(ids, size)
        self.values = nn.Embedding(ids, size)
        self.gru = nn.GRU(2 * size, hidden, batch_first=True)
        self.out = nn.Linear(hidden, ids)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        pairs = torch.cat([self.keys(keys), self.values(values)], dim=2)
        asked = self.keys(query).unsqueeze(1)
        last_step = torch.cat([asked, torch.zeros_like(asked)], dim=2)
        _, last = self.gru(torch.cat([pairs, last_step], dim=1))

        return self.out(last[0])
'''

ATTENTION6_SOURCE = source_file(
    '''
    # attention6.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    rows = torch.ones(2, 1, 4) @ torch.ones(2, 4, 3)    # shape (2, 1, 3), one row of scores each

''' + textwrap.indent(GRU_RECALLER.strip("\n"), "    ") + '''


    # TODO: Finish forward. keys, values and query hold ids, of shapes (batch,
    #   pairs), (batch, pairs) and (batch,). Compare each query with every key
    #   of its sequence by scaled dot products, average the values with the
    #   softmax weights, and give self.out's scores for every id, of shape
    #   (batch, ids).
    class Recaller(nn.Module):
        def __init__(
            self,
            ids: int,
            size: int,
        ) -> None:
            super().__init__()
            self.keys = nn.Embedding(ids, size)
            self.queries = nn.Embedding(ids, size)
            self.values = nn.Embedding(ids, size)
            self.out = nn.Linear(size, ids)

        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            ...
    ''',
)

ATTENTION6_SOLUTION = source_file(
    '''
    # attention6.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    rows = torch.ones(2, 1, 4) @ torch.ones(2, 4, 3)

''' + textwrap.indent(GRU_RECALLER.strip("\n"), "    ") + '''


    class Recaller(nn.Module):
        def __init__(
            self,
            ids: int,
            size: int,
        ) -> None:
            super().__init__()
            self.keys = nn.Embedding(ids, size)
            self.queries = nn.Embedding(ids, size)
            self.values = nn.Embedding(ids, size)
            self.out = nn.Linear(size, ids)

        # The query is one row per sequence, so its scores are one row against all the keys.
        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            k = self.keys(keys)
            q = self.queries(query).unsqueeze(1)
            weights = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)

            return self.out((weights @ self.values(values)).squeeze(1))
    ''',
)


class ReferenceRecaller(nn.Module):
    def __init__(
        self,
        ids: int,
        size: int,
    ) -> None:
        super().__init__()
        self.keys = nn.Embedding(ids, size)
        self.queries = nn.Embedding(ids, size)
        self.values = nn.Embedding(ids, size)
        self.out = nn.Linear(size, ids)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        k = self.keys(keys)
        weights = torch.softmax(self.queries(query).unsqueeze(1) @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)

        return self.out((weights @ self.values(values)).squeeze(1))


def train_recaller(
    model: nn.Module,
    pairs: int,
    steps: int,
    lr: float,
    seed: int = 0,
) -> nn.Module:
    """The runs the lesson's table comes from, Adam on batches of 64 new sequences drawn from seed + 1."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(steps):
        keys, values, query, answer = recall_sequences(64, pairs, generator)
        loss = F.cross_entropy(model(keys, values, query), answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def recall_accuracy(
    model: nn.Module,
    pairs: int,
) -> float:
    keys, values, query, answer = recall_sequences(1000, pairs, torch.Generator().manual_seed(99))

    with torch.no_grad():
        return (model(keys, values, query).argmax(dim=1) == answer).float().mean().item()


def check_recaller_forward(found: type) -> None:
    model = expect_module("Recaller(16, 8)", seeded_module(found, RECALL_IDS, 8))
    keys, values, query, _ = recall_sequences(3, 5, torch.Generator().manual_seed(510))

    with torch.no_grad():
        expected = seeded_module(ReferenceRecaller, RECALL_IDS, 8)(keys, values, query)

    expect_tensor(
        call="Recaller(16, 8) on 3 sequences of 5 pairs",
        actual=model(keys, values, query),
        expected=expected,
    )


def check_recaller_learns(found: type) -> None:
    model = train_recaller(seeded_module(found, RECALL_IDS, 16), 16, 400, 0.05)
    accuracy = recall_accuracy(model, 16)

    if accuracy < 0.95:
        raise Mismatch(f"After 400 steps on 16 pairs, Recaller gets {accuracy:.1%} of new sequences right, but it should pass 95%.")


def check_attention6(checker: Checker) -> None:
    checker.test_method(
        class_name="Recaller",
        method="forward",
        description="compares the query with every key, then averages the values",
        run=check_recaller_forward,
    )
    checker.test_method(
        class_name="Recaller",
        method="forward",
        description="recalls the value of 16 pairs after training",
        run=check_recaller_learns,
    )


ATTENTION6 = Exercise(
    name="attention6",
    chapter=ATTENTION,
    title="Why attention",
    lesson=ATTENTION6_LESSON,
    source=ATTENTION6_SOURCE,
    solution=ATTENTION6_SOLUTION,
    hints={
        "Recaller.forward": "k = self.keys(keys) has shape (batch, pairs, size). self.queries(query).unsqueeze(1) "
        "has shape (batch, 1, size), so q @ k.transpose(1, 2) gives (batch, 1, pairs). Divide by "
        "math.sqrt(k.shape[-1]), softmax with dim=-1, @ self.values(values), .squeeze(1), then self.out.",
    },
    introduces=(),
    check=check_attention6,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

COPY_IDS = 16
COPY_STEPS = 12
COPY_SIZE = 16


class CopyPreviousModel(nn.Module):
    """attention4's comparison: copy the token right before each position, with SingleHead's layers or with x
    compared with itself and averaged."""

    def __init__(self, project: bool) -> None:
        super().__init__()
        self.project = project
        self.tokens = nn.Embedding(COPY_IDS, COPY_SIZE)
        self.positions = nn.Embedding(COPY_STEPS, COPY_SIZE)

        if project:
            self.head = ReferenceSingleHead(COPY_SIZE, COPY_SIZE)

        self.read = nn.Linear(COPY_SIZE, COPY_IDS)

    def forward(self, x: Tensor) -> Tensor:
        h = self.tokens(x) + self.positions(torch.arange(x.shape[1]))

        if self.project:
            return self.read(self.head(h))

        return self.read(reference_masked_weights(reference_scores(h, h)) @ h)


def measure_projection(seeds: range = range(3)) -> dict[tuple[int, bool], tuple[float, float]]:
    """attention4's table: by seed and whether the model has SingleHead's layers, the share of positions after the
    first that give the previous token, then the share that give their own, on 2,000 new sequences after 600 steps."""
    test = torch.randint(0, COPY_IDS, (2000, COPY_STEPS), generator=torch.Generator().manual_seed(99))
    results = {}

    with pinned_threads():
        for seed in seeds:
            for project in (False, True):
                torch.manual_seed(seed)
                model = CopyPreviousModel(project)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                generator = torch.Generator().manual_seed(seed + 1)

                for _ in range(600):
                    x = torch.randint(0, COPY_IDS, (64, COPY_STEPS), generator=generator)
                    loss = F.cross_entropy(model(x)[:, 1:].reshape(-1, COPY_IDS), x[:, :-1].reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    guesses = model(test)[:, 1:].argmax(dim=-1)

                results[(seed, project)] = (
                    (guesses == test[:, :-1]).float().mean().item(),
                    (guesses == test[:, 1:]).float().mean().item(),
                )

    return results


def measure_attention_time(repeats: int = 10) -> tuple[float, float]:
    """attention5's timing: causal attention over one sequence of 2,048 steps in 4 heads of 64, by hand against
    F.scaled_dot_product_attention, the fastest of repeats runs of each in milliseconds."""
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, 2048, 64, generator=generator) for _ in range(3))

    def by_hand() -> Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        future = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1]), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v

    with pinned_threads(), torch.no_grad():
        hand = fastest_seconds(by_hand, repeats)
        call = fastest_seconds(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), repeats)

    return 1000 * hand, 1000 * call


def given_class(
    code: str,
    name: str,
) -> type:
    """A class the learner's file is given, built from the same code, so a measurement runs what the file holds."""
    namespace: dict[str, object] = {"torch": torch, "nn": nn, "Tensor": Tensor}
    exec(code, namespace)

    return namespace[name]


def measure_recall(seeds: range = range(3)) -> dict[tuple[str, int, int], float]:
    """attention6's table: the share of 1,000 new sequences recalled right, by model, seed and number of pairs.
    Each GRU trains for 3,000 steps at a rate of 0.003, and Recaller for 400 steps at 0.05."""
    gru = given_class(GRU_RECALLER, "GRURecaller")
    models: tuple[tuple[str, Callable[[], nn.Module], int, float], ...] = (
        ("GRU of 64", lambda: gru(RECALL_IDS, 16, 64), 3000, 0.003),
        ("GRU of 256", lambda: gru(RECALL_IDS, 16, 256), 3000, 0.003),
        ("Attention", lambda: ReferenceRecaller(RECALL_IDS, 16), 400, 0.05),
    )
    results = {}

    with pinned_threads():
        for seed in seeds:
            for pairs in (2, 4, 8, 16):
                for name, make, steps, lr in models:
                    torch.manual_seed(seed)
                    model = train_recaller(make(), pairs, steps, lr, seed)
                    results[(name, seed, pairs)] = recall_accuracy(model, pairs)

    return results
