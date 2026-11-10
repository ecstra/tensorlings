"""48_linear_attention, attention without the softmax as an RNN whose state is a matrix, and what that fixed state can hold."""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .attention import RECALL_IDS, recall_accuracy, recall_sequences, seeded_module, train_recaller
from .linear_layers import expect_module

LINEAR_ATTENTION = Chapter(
    folder="48_linear_attention",
    title="Linear attention",
    readme=source_file(
        """
        # Linear attention

        Attention keeps every key and value it has seen, and compares each
        query with all of them. This chapter removes the softmax, which turns
        attention into an RNN whose state is a matrix. Then it measures what
        that fixed size state can hold and what it cannot.

        ## Further reading

        - [Katharopoulos and colleagues, transformers are RNNs](https://arxiv.org/abs/2006.16236)
        - [Schlag, Irie and Schmidhuber, linear transformers are secretly fast weight programmers](https://arxiv.org/abs/2102.11174)
        - [Yang and colleagues, gated linear attention transformers](https://arxiv.org/abs/2312.06635)
        """,
    ),
)


# ---------------------------------------------------------------------------
# linattn1
# ---------------------------------------------------------------------------

LINATTN1_LESSON = prose(
    """
    attention5 computes `softmax(q kᵀ / √d) v`. If you remove the softmax and
    the division by `√d`, you can move the brackets, because matrix
    multiplication is associative.

    ```text
    (q kᵀ) v = q (kᵀ v)
    ```

    Here q, k and v each have one row per step and size columns. The left
    side first builds a steps by steps matrix of scores. The right side first
    builds `kᵀ v`, a size by size matrix, however long the sequence is.

    With the future masked, the output at position t uses only the keys and
    values up to t. So each position can multiply its query by a running
    state that grows by one key and one value per step.

    ```text
    state at t = state at t - 1 + k_t v_tᵀ
    output at t = q_t state at t
    ```

    That is an RNN whose state is a matrix, and each step adds the product of
    one key and one value to it. This is called linear attention. A model
    that writes text with it keeps one size by size matrix as it goes,
    instead of every key and value so far. At 4,096 steps and a size of 64,
    the state is 4,096 numbers, while the keys and values are 524,288.

    Multiplying a column by a row, which is called the outer product, makes
    such a matrix.

    ```python
    torch.tensor([[1.0], [2.0]]) @ torch.tensor([[3.0, 4.0]])    # tensor([[3., 4.], [6., 8.]])
    ```

    The parallel form and the recurrent form give the same outputs, up to
    rounding. On 2 sequences of 50 random steps with a size of 8, they
    differed by at most 0.00002 over three seeds.

    Removing the softmax has a cost. The weights can be negative, and they no
    longer add up to 1. Katharopoulos and colleagues fix both problems. They
    pass queries and keys through `F.elu(x) + 1`, which is always positive.
    Then they divide each output by the sum of its weights. `F.elu(x)` returns
    x when x is above 0, and `exp(x) - 1` otherwise.
    """,
)

LINATTN1_SOURCE = source_file(
    '''
    # linattn1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    outer = torch.tensor([[1.0], [2.0]]) @ torch.tensor([[3.0, 4.0]])    # tensor([[3., 4.], [6., 8.]])
    positive = F.elu(torch.tensor([-1.0, 2.0])) + 1                     # tensor([0.3679, 3.0000])


    # TODO: Return linear attention of q, k and v in the parallel form. Each has
    #   shape (..., steps, size). Compute the scores q kᵀ, set the scores of
    #   future positions to 0, and multiply the scores by v. Use no softmax and
    #   no division by the square root of size.
    #   Example: with q, k and v all torch.ones(2, 1), it returns
    #   tensor([[1.], [2.]]).
    def parallel_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish the loop, so it returns the same outputs as
    #   parallel_linear_attention, computed as a recurrence. At each step t,
    #   add the outer product of key t and value t to state. Then the output
    #   at step t is query t, as a row, times state.
    def recurrent_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        state = torch.zeros(*q.shape[:-2], q.shape[-1], v.shape[-1])
        outputs = []

        for t in range(q.shape[-2]):
            state = ...
            outputs.append(...)

        return torch.stack(outputs, dim=-2)


    # TODO: Return the normalized form of linear attention. Pass q and k through
    #   F.elu(x) + 1, compute the weights as parallel_linear_attention computes
    #   its scores, and set the weights of future positions to 0. Multiply the
    #   weights by v, then divide each output row by the sum of its weights.
    def normalized_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        ...
    ''',
)

LINATTN1_SOLUTION = source_file(
    '''
    # linattn1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    outer = torch.tensor([[1.0], [2.0]]) @ torch.tensor([[3.0, 4.0]])
    positive = F.elu(torch.tensor([-1.0, 2.0])) + 1


    def parallel_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        return torch.tril(q @ k.transpose(-2, -1)) @ v


    def recurrent_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        state = torch.zeros(*q.shape[:-2], q.shape[-1], v.shape[-1])
        outputs = []

        for t in range(q.shape[-2]):
            state = state + k[..., t, :].unsqueeze(-1) @ v[..., t, :].unsqueeze(-2)
            outputs.append((q[..., t, :].unsqueeze(-2) @ state).squeeze(-2))

        return torch.stack(outputs, dim=-2)


    def normalized_linear_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        weights = torch.tril((F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-2, -1))

        return weights @ v / weights.sum(dim=-1, keepdim=True)
    ''',
)


def reference_parallel(
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    return torch.tril(q @ k.transpose(-2, -1)) @ v


def reference_normalized(
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    weights = torch.tril((F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-2, -1))

    return weights @ v / weights.sum(dim=-1, keepdim=True)


def linear_inputs(seed: int) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
    return (
        (small_tensor(seed, 5, 3) / 2, small_tensor(seed + 1, 5, 3) / 2, small_tensor(seed + 2, 5, 2)),
        (small_tensor(seed + 3, 2, 4, 3) / 2, small_tensor(seed + 4, 2, 4, 3) / 2, small_tensor(seed + 5, 2, 4, 4)),
    )


def check_linattn1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="parallel_linear_attention",
        description="sets the future scores to 0, then multiplies the scores by the values",
        inputs=linear_inputs(520),
        reference=reference_parallel,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="recurrent_linear_attention",
        description="gives the same outputs from a running state",
        inputs=linear_inputs(530),
        reference=reference_parallel,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="normalized_linear_attention",
        description="gives positive weights that sum to 1",
        inputs=linear_inputs(540),
        reference=reference_normalized,
        expect=expect_tensor,
    )


LINATTN1 = Exercise(
    name="linattn1",
    chapter=LINEAR_ATTENTION,
    title="Attention as an RNN",
    lesson=LINATTN1_LESSON,
    source=LINATTN1_SOURCE,
    solution=LINATTN1_SOLUTION,
    hints={
        "parallel_linear_attention": "q @ k.transpose(-2, -1) gives the scores. torch.tril sets the future scores to "
        "0. Then multiply by v with @.",
        "recurrent_linear_attention": "k[..., t, :].unsqueeze(-1) @ v[..., t, :].unsqueeze(-2) is the outer product to "
        "add. The output is q[..., t, :].unsqueeze(-2) @ state, with the extra dimension removed by .squeeze(-2).",
        "normalized_linear_attention": "The weights are torch.tril of (F.elu(q) + 1) @ (F.elu(k) + 1) transposed. "
        "Divide weights @ v by weights.sum(dim=-1, keepdim=True).",
    },
    introduces=(),
    check=check_linattn1,
)


# ---------------------------------------------------------------------------
# linattn2
# ---------------------------------------------------------------------------

LINATTN2_LESSON = prose(
    """
    The state of linear attention has a fixed size, size by size, however long
    the sequence is. Softmax attention keeps every key and value, so its
    memory grows by one key and one value per step. Each new token has to
    read all of them.

    Here is the cost of writing one more token, measured over three runs. The
    batch has 16 sequences, the size is 64, and everything runs on one CPU
    thread of this Mac. The softmax cache is the stored keys and values.

    | Earlier tokens | Softmax step | Linear step | Softmax cache | Linear state |
    |---|---|---|---|---|
    | 256 | 0.10 to 0.13 ms | 0.10 to 0.12 ms | 2.1 MB | 0.27 MB |
    | 1,024 | 0.27 to 0.36 ms | 0.09 to 0.12 ms | 8.4 MB | 0.27 MB |
    | 4,096 | 1.1 to 1.4 ms | 0.09 to 0.12 ms | 33.6 MB | 0.27 MB |
    | 16,384 | 5.0 to 9.7 ms | 0.12 to 0.38 ms | 134 MB | 0.27 MB |

    The softmax step takes longer as the text grows, while the linear step
    takes about the same time. Katharopoulos and colleagues found their linear
    attention models up to 4,000 times faster at writing very long sequences.
    But the fixed state limits what the model can remember.

    Here is the recall task from attention6, with queries, keys and values of
    size 8. Each model trained for 400 steps. The table shows the fraction of
    new sequences each model got right.

    | Pairs | Linear attention | Softmax attention |
    |---|---|---|
    | 2 | 1.000 | 1.000 |
    | 4 | 0.969 | 1.000 |
    | 8 | 0.857 | 1.000 |
    | 16 | 0.573 | 1.000 |

    The linear state holds 72 numbers at any number of pairs, an 8 by 8 matrix
    plus the sum of the keys. Softmax attention keeps 16 numbers per pair, a
    key and a value of 8 numbers each, so it keeps 256 numbers at 16 pairs.

    The state is a sum of outer products of keys and values. When there are
    few pairs, their keys can point in different directions. Then multiplying
    a query by the state brings back only the value stored with the matching
    key. With more than about size pairs, 8 here, the keys have to overlap.
    Then reading one key also brings back part of the other values.

    So linear attention suits long sequences where a summary of the past is
    enough. Softmax attention suits exact recall of any earlier step. Some
    recent models mix the two, with a few softmax layers among many linear
    ones. Newer linear attention layers also add a gate that lets the state
    forget, as the selective step of ssm3 did.

    `.sum(dim=1, keepdim=True)` keeps the summed dimension, with size 1. So
    the result has the same number of dimensions as the input.

    ```python
    torch.ones(2, 5, 3).sum(dim=1, keepdim=True).shape    # torch.Size([2, 1, 3])
    ```
    """,
)

LINATTN2_SOURCE = source_file(
    '''
    # linattn2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    totals = torch.ones(2, 5, 3).sum(dim=1, keepdim=True)    # shape (2, 1, 3)


    # TODO: Return how many numbers a model keeps after reading pairs pairs, when
    #   its queries, keys and values have size numbers each. If linear is True,
    #   the model uses linear attention, which keeps its size by size state and
    #   the sum of its keys. Otherwise it uses softmax attention, which keeps
    #   every key and every value.
    #   Example: memory_numbers(16, 8, True) returns 72, and
    #   memory_numbers(16, 8, False) returns 256.
    def memory_numbers(
        pairs: int,
        size: int,
        linear: bool,
    ) -> int:
        ...


    # TODO: Finish forward. It is the Recaller from attention6, with normalized
    #   linear attention in place of softmax attention. Embed keys, values and
    #   query with self.keys, self.values and self.queries. Pass the embedded
    #   keys and the embedded query through F.elu(x) + 1. The state is the keys
    #   transposed times the values, and the total is the sum of the keys over
    #   the pairs. The output is the query times the state, divided by the
    #   query times the total. Run self.out on it to get shape (batch, ids).
    class LinearRecaller(nn.Module):
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

LINATTN2_SOLUTION = source_file(
    '''
    # linattn2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    totals = torch.ones(2, 5, 3).sum(dim=1, keepdim=True)


    def memory_numbers(
        pairs: int,
        size: int,
        linear: bool,
    ) -> int:
        if linear:
            return size * size + size

        return 2 * pairs * size


    class LinearRecaller(nn.Module):
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
            k = F.elu(self.keys(keys)) + 1
            q = F.elu(self.queries(query)).unsqueeze(1) + 1
            state = k.transpose(1, 2) @ self.values(values)
            total = k.sum(dim=1, keepdim=True)
            read = (q @ state) / (q @ total.transpose(1, 2))

            return self.out(read.squeeze(1))
    ''',
)


class ReferenceLinearRecaller(nn.Module):
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
        k = F.elu(self.keys(keys)) + 1
        q = F.elu(self.queries(query)).unsqueeze(1) + 1
        read = (q @ (k.transpose(1, 2) @ self.values(values))) / (q @ k.sum(dim=1, keepdim=True).transpose(1, 2))

        return self.out(read.squeeze(1))


def check_linear_recaller(found: type) -> None:
    model = expect_module("LinearRecaller(16, 8)", seeded_module(found, RECALL_IDS, 8))
    keys, values, query, _ = recall_sequences(3, 5, torch.Generator().manual_seed(550))

    with torch.no_grad():
        expected = seeded_module(ReferenceLinearRecaller, RECALL_IDS, 8)(keys, values, query)

    expect_tensor(
        call="LinearRecaller(16, 8) on 3 sequences of 5 pairs",
        actual=model(keys, values, query),
        expected=expected,
    )


def check_linear_recaller_learns(found: type) -> None:
    model = train_recaller(seeded_module(found, RECALL_IDS, 8), 4, 400, 0.05)
    accuracy = recall_accuracy(model, 4)

    if accuracy < 0.9:
        raise Mismatch(f"After 400 steps on 4 pairs, LinearRecaller gets {accuracy:.1%} of new sequences right, but it should get at least 90%.")


def check_linattn2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="memory_numbers",
        description="counts the fixed state and the growing keys and values",
        inputs=(
            (16, 8, True),
            (16, 8, False),
            (1000, 64, True),
            (1000, 64, False),
        ),
        reference=lambda pairs, size, linear: size * size + size if linear else 2 * pairs * size,
        expect=expect_value,
    )
    checker.test_method(
        class_name="LinearRecaller",
        method="forward",
        description="reads the query against the state of keys times values",
        run=check_linear_recaller,
    )
    checker.test_method(
        class_name="LinearRecaller",
        method="forward",
        description="recalls the value of 4 pairs after training",
        run=check_linear_recaller_learns,
    )


LINATTN2 = Exercise(
    name="linattn2",
    chapter=LINEAR_ATTENTION,
    title="A fixed state against a growing one",
    lesson=LINATTN2_LESSON,
    source=LINATTN2_SOURCE,
    solution=LINATTN2_SOLUTION,
    hints={
        "memory_numbers": "Linear attention keeps size * size for the state plus size for the sum of the keys. "
        "Softmax attention keeps pairs keys and pairs values, each of size numbers.",
        "LinearRecaller.forward": "For k, put keys through self.keys and F.elu, then add 1. Make q the same way from "
        "self.queries(query), with .unsqueeze(1). Divide q @ (k.transpose(1, 2) @ self.values(values)) by "
        "q @ k.sum(dim=1, keepdim=True) transposed. Then call .squeeze(1) and run self.out.",
    },
    introduces=(),
    check=check_linattn2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def step_milliseconds(
    cached: int,
    rounds: int = 5,
    repeats: int = 20,
) -> tuple[float, float]:
    """The time to write one more token, for a batch of 16 at size 64 on one CPU thread: softmax attention
    reading a cache of cached keys and values, against linear attention adding to its state and reading it."""
    batch, size = 16, 64
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(batch, cached, size, generator=generator)
    values = torch.randn(batch, cached, size, generator=generator)
    query = torch.randn(batch, 1, size, generator=generator)
    state = torch.randn(batch, size, size, generator=generator)
    total = torch.rand(batch, 1, size, generator=generator)
    new_key = F.elu(torch.randn(batch, 1, size, generator=generator)) + 1
    new_value = torch.randn(batch, 1, size, generator=generator)
    feature = F.elu(query) + 1

    def softmax_step() -> Tensor:
        return torch.softmax(query @ keys.transpose(1, 2) / size**0.5, dim=-1) @ values

    def linear_step() -> Tensor:
        grown = state + new_key.transpose(1, 2) @ new_value
        summed = total + new_key

        return (feature @ grown) / (feature @ summed.transpose(1, 2))

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    times = []

    try:
        for run in (softmax_step, linear_step):
            run()
            medians = []

            for _ in range(rounds):
                start = time.perf_counter()

                for _ in range(repeats):
                    run()

                medians.append((time.perf_counter() - start) / repeats)

            times.append(1000 * statistics.median(medians))
    finally:
        torch.set_num_threads(threads)

    return times[0], times[1]


def measure_step_cost() -> dict[int, tuple[float, float, int, int]]:
    """linattn2's table: for each number of earlier tokens, the two step times in milliseconds, then the bytes the
    softmax cache holds and the bytes the linear state holds, at a batch of 16, size 64 and 4 bytes a number."""
    batch, size = 16, 64
    state_bytes = batch * (size * size + size) * 4

    return {
        cached: (*step_milliseconds(cached), batch * cached * size * 2 * 4, state_bytes)
        for cached in (256, 1024, 4096, 16384)
    }


def reference_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    state = torch.zeros(*q.shape[:-2], q.shape[-1], v.shape[-1])
    outputs = []

    for t in range(q.shape[-2]):
        state = state + k[..., t, :].unsqueeze(-1) @ v[..., t, :].unsqueeze(-2)
        outputs.append((q[..., t, :].unsqueeze(-2) @ state).squeeze(-2))

    return torch.stack(outputs, dim=-2)


def measure_forms_gap(seeds: range = range(3)) -> dict[int, float]:
    """linattn1's claim: by seed, the largest gap between the parallel and the recurrent form on 2 sequences of 50
    random steps with a size of 8."""
    gaps = {}

    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        q, k, v = (torch.randn(2, 50, 8, generator=generator) for _ in range(3))
        gaps[seed] = (reference_parallel(q, k, v) - reference_recurrent(q, k, v)).abs().max().item()

    return gaps
