"""47_attention, each position as a weighted average of the ones before it, from plain averages to masked softmax weights."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    check_on_inputs,
    expect_tensor,
    small_tensor,
)
from ..text import prose, source_file

ATTENTION = Chapter(
    folder="47_attention",
    title="Attention",
    readme=source_file(
        """
        # Attention

        Section 7 is about attention, the way each position of a sequence
        gathers what it needs from the others by a weighted average. This
        chapter starts from plain averages of the past, then turns the weights
        into a masked softmax over scores.

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
        expect=expect_tensor,
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
