"""51_cross_attention, queries from one sequence and keys and values from another, and the adder's long sums solved by reading the question at every step."""

from __future__ import annotations

import math

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
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

CROSS_ATTENTION = Chapter(
    folder="51_cross_attention",
    title="Cross attention",
    readme=source_file(
        """
        # Cross attention

        Cross attention takes its queries from one sequence and its keys and
        values from another. A decoder uses it to read the encoder's outputs
        at every step it writes, instead of one vector that has to hold the
        whole input. This chapter builds it and puts it to work on the adder's
        long sums.

        ## Further reading

        - [Bahdanau, Cho and Bengio, learning to align and translate](https://arxiv.org/abs/1409.0473)
        - [Luong, Pham and Manning, effective approaches to attention based translation](https://arxiv.org/abs/1508.04025)
        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        """,
    ),
)


def reference_cross_scores(
    queries: Tensor,
    keys: Tensor,
) -> Tensor:
    return queries @ keys.transpose(1, 2) / math.sqrt(queries.shape[-1])


def reference_cross_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    blocked: Tensor,
) -> Tensor:
    scores = reference_cross_scores(queries, keys).masked_fill(blocked.unsqueeze(1), float("-inf"))

    return torch.softmax(scores, dim=-1) @ values


class ReferenceCrossAttention(nn.Module):
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

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        mixed = reference_cross_attention(self.query(x), self.key(memory), self.value(memory), blocked)

        return self.out(mixed)


# ---------------------------------------------------------------------------
# cross1
# ---------------------------------------------------------------------------

CROSS1_LESSON = prose(
    """
    Attention so far took its queries, keys and values from one sequence.
    Cross attention takes the queries from one sequence and the keys and
    values from another. A decoder writing an answer asks its questions of the
    encoder's outputs, so at each step it reads the part of the input it
    needs.

    The weights have one row per query position and one column per key
    position, so they need not be square.

    ```python
    writing = torch.zeros(2, 3, 8)    # 3 positions writing
    reading = torch.zeros(2, 5, 8)    # 5 positions read
    (writing @ reading.transpose(1, 2)).shape    # torch.Size([2, 3, 5])
    ```

    There is no causal mask here. The whole input is known before the decoder
    starts, so every query may read every key. The padding mask still applies,
    since short inputs are padded, as in mask1. A padding mask of shape
    `(batch, keys)` becomes `(batch, 1, keys)` with `.unsqueeze(1)`, which
    broadcasts it over every query.

    Bahdanau, Cho and Bengio added this to the RNN encoder and decoder in
    2014, so a translating decoder could look back at the source words. The
    encoder and decoder models of 54_model_types put it in every block of the
    decoder.
    """,
)

CROSS1_SOURCE = source_file(
    '''
    # cross1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    writing = torch.zeros(2, 3, 8)
    reading = torch.zeros(2, 5, 8)
    shape = (writing @ reading.transpose(1, 2)).shape    # torch.Size([2, 3, 5])

    blocked = torch.tensor([[False, False, True]])
    spread = blocked.unsqueeze(1)                          # shape (1, 1, 3), one row for every query


    # TODO: Return the scores of every query against every key, divided by the
    #   square root of their size. queries has shape (batch, writing, size) and
    #   keys (batch, reading, size).
    def cross_scores(
        queries: Tensor,
        keys: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the values mixed by the softmax of cross_scores, with the
    #   keys where blocked, of shape (batch, reading), is True left out.
    def cross_attention(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the same as cross_attention, from
    #   F.scaled_dot_product_attention. Its attn_mask is True where a key takes
    #   part, the opposite of blocked.
    def with_torch(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. The queries come from x, the keys and values from
    #   memory, and the mixed values go through self.out.
    class CrossAttention(nn.Module):
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

        def forward(
            self,
            x: Tensor,
            memory: Tensor,
            blocked: Tensor,
        ) -> Tensor:
            ...
    ''',
)

CROSS1_SOLUTION = source_file(
    '''
    # cross1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    writing = torch.zeros(2, 3, 8)
    reading = torch.zeros(2, 5, 8)
    shape = (writing @ reading.transpose(1, 2)).shape

    blocked = torch.tensor([[False, False, True]])
    spread = blocked.unsqueeze(1)


    def cross_scores(
        queries: Tensor,
        keys: Tensor,
    ) -> Tensor:
        return queries @ keys.transpose(1, 2) / math.sqrt(queries.shape[-1])


    def cross_attention(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        scores = cross_scores(queries, keys).masked_fill(blocked.unsqueeze(1), float("-inf"))

        return torch.softmax(scores, dim=-1) @ values


    def with_torch(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        return F.scaled_dot_product_attention(queries, keys, values, attn_mask=~blocked.unsqueeze(1))


    class CrossAttention(nn.Module):
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

        def forward(
            self,
            x: Tensor,
            memory: Tensor,
            blocked: Tensor,
        ) -> Tensor:
            mixed = cross_attention(self.query(x), self.key(memory), self.value(memory), blocked)

            return self.out(mixed)
    ''',
)


def attention_triple(
    seed: int,
    batch: int,
    writing: int,
    reading: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        small_tensor(seed, batch, writing, size) / 3,
        small_tensor(seed + 1, batch, reading, size) / 3,
        small_tensor(seed + 2, batch, reading, size),
    )


BLOCKED_A = torch.tensor([[False, False, False, True], [False, True, True, True]])
BLOCKED_B = torch.tensor([[False, False, False, False, False], [False, False, False, True, True], [False, True, True, True, True]])


def seeded_cross(make: type) -> nn.Module:
    torch.manual_seed(510)

    return make(6, 4)


def check_cross_module(found: type) -> None:
    model = expect_module("CrossAttention(6, 4)", seeded_cross(found))
    x = small_tensor(511, 2, 3, 6) / 3
    memory = small_tensor(512, 2, 4, 6) / 3

    with torch.no_grad():
        expected = seeded_cross(ReferenceCrossAttention)(x, memory, BLOCKED_A)

    expect_tensor(
        call="CrossAttention(6, 4) on 3 positions reading 4, the last keys blocked",
        actual=model(x, memory, BLOCKED_A),
        expected=expected,
    )


def check_cross1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="cross_scores",
        description="scores every query against every key",
        inputs=(
            attention_triple(500, 2, 3, 4, 4)[:2],
            attention_triple(503, 1, 2, 5, 9)[:2],
        ),
        reference=reference_cross_scores,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="cross_attention",
        description="mixes the values, leaving out the blocked keys",
        inputs=(
            (*attention_triple(506, 2, 3, 4, 4), Described(BLOCKED_A, "the last key blocked in one row, three in the other")),
            (*attention_triple(509, 3, 2, 5, 3), Described(BLOCKED_B, "none, two and four keys blocked")),
        ),
        reference=reference_cross_attention,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="with_torch",
        description="matches cross_attention through PyTorch",
        inputs=(
            (*attention_triple(513, 2, 3, 4, 4), Described(BLOCKED_A, "the last key blocked in one row, three in the other")),
        ),
        reference=reference_cross_attention,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="CrossAttention",
        method="forward",
        description="reads memory from the positions of x",
        run=check_cross_module,
    )


CROSS1 = Exercise(
    name="cross1",
    chapter=CROSS_ATTENTION,
    title="Queries from one, keys from another",
    lesson=CROSS1_LESSON,
    source=CROSS1_SOURCE,
    solution=CROSS1_SOLUTION,
    hints={
        "cross_scores": "queries @ keys.transpose(1, 2), divided by math.sqrt of the size, queries.shape[-1].",
        "cross_attention": "Fill the scores with float(\"-inf\") where blocked.unsqueeze(1) is True, take the "
        "softmax over the last dimension, then @ values.",
        "with_torch": "Pass the queries, keys and values to F.scaled_dot_product_attention, with attn_mask set to "
        "~ of blocked, given its query dimension with .unsqueeze(1) as in cross_attention.",
        "CrossAttention.forward": "cross_attention(self.query(x), self.key(memory), self.value(memory), blocked), "
        "then self.out.",
    },
    introduces=(),
    check=check_cross1,
)
