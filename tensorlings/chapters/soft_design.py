"""38_soft_design, the design loop: an exact algorithm, its hard steps found, and soft steps in their place, once for a gate and once for a lookup."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

SOFT_DESIGN = Chapter(
    folder="38_soft_design",
    title="Designing with soft steps",
    readme=source_file(
        """
        # Designing with soft steps

        Section 6 is about sequences, and its models, like Section 7's, are
        built from soft steps. This chapter shows the method behind them on
        two small tasks, before the named blocks arrive. You write the exact
        algorithm, find its hard steps, swap each one for a soft step, and
        train the model that results.

        ## Further reading

        - [Graves, Wayne and Danihelka, Neural Turing Machines](https://arxiv.org/abs/1410.5401)
        - [Bahdanau, Cho and Bengio, learning to align and translate](https://arxiv.org/abs/1409.0473)
        - [Olah, understanding LSTM networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
        """,
    ),
)


def marked_sequences(
    count: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sequences of (value, mark) pairs, about 3 marks in 10, and the last marked value of each, 0 when none is."""
    values = torch.randn(count, steps, generator=generator)
    marks = (torch.rand(count, steps, generator=generator) < 0.3).float()
    memory = torch.zeros(count)

    for t in range(steps):
        memory = torch.where(marks[:, t] == 1, values[:, t], memory)

    return torch.stack([values, marks], dim=2), memory


def lookup_sequences(
    count: int,
    pairs: int,
    ids: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sequences of pairs with different key ids, a query that is one of the keys, and the value paired with it."""
    keys = torch.stack([torch.randperm(ids, generator=generator)[:pairs] for _ in range(count)])
    values = torch.randn(count, pairs, generator=generator)
    picked = torch.randint(0, pairs, (count,), generator=generator)
    rows = torch.arange(count)

    return keys, values, keys[rows, picked], values[rows, picked]


# ---------------------------------------------------------------------------
# derive1
# ---------------------------------------------------------------------------

DERIVE1_LESSON = prose(
    """
    Section 6 is about sequences, and its blocks are built from soft steps.
    This chapter shows the method behind them, on two small tasks, before the
    named blocks arrive. The method has four moves.

    1. Write the exact algorithm by hand, in plain Python.
    2. Find its hard steps, the comparisons, `if`s and picks that pass no gradient, as soft1 showed.
    3. Swap each one for a soft step.
    4. Let the model compute what the soft step needs, and train it.

    The first task reads a sequence of pairs, each a value and a mark of 0 or
    1. It gives back the last value that was marked. By hand it is a loop with
    one `if`, which you write first.

    The `if` is the hard step. Without it, the same update reads

    ```text
    memory = mark * value + (1 - mark) * memory
    ```

    With a mark of exactly 0 or 1, this is the same algorithm. Put any number
    from 0 to 1 in the mark's place and it blends the new value into the
    memory instead. That number is a **gate**. A gate of 0.9 keeps mostly the
    new value, and 0.1 keeps mostly the memory.

    The model has to decide each gate itself, from that step's input. Here an
    `nn.Linear(2, 1)` scores the value and the mark, and a sigmoid turns the
    score into the gate. Trained for 300 steps on random sequences of 10, it
    gives this.

    | Model | Mean squared error on new sequences |
    |---|---|
    | always 0 | 0.900 |
    | the gated memory, untrained | 0.699 |
    | the gated memory, trained | 0.012 |

    The trained Linear puts a weight of 6.8 on the mark, about 0 on the value,
    and a bias of -3.8. So a marked step gets a gate of 0.95 and an unmarked
    one 0.02. It found the `if`, as a soft one.

    With `(score > 0).float()` as the gate instead, `loss.backward()` fails
    with "element 0 of tensors does not require grad". Nothing in the model
    gets any gradient at all.

    A gate computed from the input is the heart of the LSTM and the GRU in
    41_gated_rnns.
    """,
)

DERIVE1_SOURCE = source_file(
    '''
    # derive1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    gate = torch.tensor(0.9)
    blended = gate * 5.0 + (1 - gate) * 2.0    # tensor(4.7000), mostly the new 5


    # TODO: Return the last value whose mark is 1, or 0.0 when none is, with a
    #   plain loop and an if. values and marks are lists of the same length.
    def keep_last_marked(
        values: list[float],
        marks: list[int],
    ) -> float:
        ...


    # TODO: Run the memory along values, of shape (batch, steps), starting from
    #   0. At step t, gates[:, t] blends values[:, t] into the memory, as in the
    #   lesson. Return the memory at the end, of shape (batch,).
    def gated_memory(
        values: Tensor,
        gates: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. x has shape (batch, steps, 2), each step a value and
    #   its mark. Compute every step's gate with self.gate and a sigmoid, then
    #   return gated_memory of the values with those gates.
    class GatedMemory(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(2, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

DERIVE1_SOLUTION = source_file(
    '''
    # derive1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    gate = torch.tensor(0.9)
    blended = gate * 5.0 + (1 - gate) * 2.0


    def keep_last_marked(
        values: list[float],
        marks: list[int],
    ) -> float:
        memory = 0.0

        for value, mark in zip(values, marks):
            if mark == 1:
                memory = value

        return memory


    def gated_memory(
        values: Tensor,
        gates: Tensor,
    ) -> Tensor:
        memory = torch.zeros(len(values))

        for t in range(values.shape[1]):
            memory = gates[:, t] * values[:, t] + (1 - gates[:, t]) * memory

        return memory


    class GatedMemory(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(2, 1)

        # The Linear works on the last dimension, so one call scores every step of every sequence.
        def forward(self, x: Tensor) -> Tensor:
            gates = torch.sigmoid(self.gate(x)).squeeze(2)

            return gated_memory(x[:, :, 0], gates)
    ''',
)


def reference_gated_memory(
    values: Tensor,
    gates: Tensor,
) -> Tensor:
    memory = torch.zeros(len(values))

    for t in range(values.shape[1]):
        memory = gates[:, t] * values[:, t] + (1 - gates[:, t]) * memory

    return memory


class ReferenceGatedMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return reference_gated_memory(x[:, :, 0], torch.sigmoid(self.gate(x)).squeeze(2))


def seeded_model(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


def train_memory(model: nn.Module) -> nn.Module:
    """The run the lesson's table comes from, 300 Adam steps on batches of 64 sequences of 10."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(1)

    for _ in range(300):
        x, y = marked_sequences(64, 10, generator)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def check_gated_forward(found: type) -> None:
    model = expect_module("GatedMemory()", seeded_model(found))
    x, _ = marked_sequences(4, 6, torch.Generator().manual_seed(380))

    with torch.no_grad():
        expected = seeded_model(ReferenceGatedMemory)(x)

    expect_tensor(
        call="GatedMemory() on 4 sequences of 6",
        actual=model(x),
        expected=expected,
    )


def check_memory_learns(found: type) -> None:
    model = train_memory(seeded_model(found))
    x, y = marked_sequences(1000, 10, torch.Generator().manual_seed(99))

    with torch.no_grad():
        error = F.mse_loss(model(x), y).item()

    if error > 0.05:
        raise Mismatch(
            f"After 300 steps of training, GatedMemory's mean squared error on new sequences is {error:.3f}, "
            "but it should be below 0.05.",
        )


def check_derive1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="keep_last_marked",
        description="keeps the last marked value",
        inputs=(
            ([0.5, -1.0, 2.0, 3.0], [1, 0, 1, 0]),
            ([1.5, 2.5, -0.5], [0, 0, 0]),
            ([4.0, -2.0, 7.0, 1.0, 0.25], [0, 1, 0, 0, 1]),
        ),
        reference=lambda values, marks: ([0.0] + [value for value, mark in zip(values, marks) if mark == 1])[-1],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="gated_memory",
        description="blends each value into the memory by its gate",
        inputs=(
            (small_tensor(381, 2, 4), torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            (small_tensor(382, 3, 5), torch.rand(3, 5, generator=torch.Generator().manual_seed(383))),
        ),
        reference=reference_gated_memory,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="GatedMemory",
        method="forward",
        description="computes each gate from its step, then runs the memory",
        run=check_gated_forward,
    )
    checker.test_method(
        class_name="GatedMemory",
        method="forward",
        description="learns to keep the last marked value",
        run=check_memory_learns,
    )


DERIVE1 = Exercise(
    name="derive1",
    chapter=SOFT_DESIGN,
    title="A gate from an if",
    lesson=DERIVE1_LESSON,
    source=DERIVE1_SOURCE,
    solution=DERIVE1_SOLUTION,
    hints={
        "keep_last_marked": "Start memory at 0.0. Loop over zip(values, marks), and when the mark is 1, set memory "
        "to the value.",
        "gated_memory": "Start from torch.zeros(len(values)). For each t in range(values.shape[1]), set memory to "
        "gates[:, t] * values[:, t] plus (1 - gates[:, t]) * memory.",
        "GatedMemory.forward": "torch.sigmoid(self.gate(x)) has shape (batch, steps, 1), so .squeeze(2) it. The "
        "values are x[:, :, 0].",
    },
    introduces=(),
    check=check_derive1,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# derive2
# ---------------------------------------------------------------------------

DERIVE2_LESSON = prose(
    """
    The second task is a lookup. A sequence holds pairs of a key and a value,
    then a query, and the answer is the value paired with the query's key. By
    hand it is a loop that compares each key with the query, which you write
    first.

    It has two hard steps, the comparison `key == query` and returning the one
    value it picked. soft2's soft lookup swaps both. A score says how well each
    key matches the query, and a softmax turns the scores into weights. The
    answer is the weighted sum of the values.

    The model has to learn what "matches" means. Here each key id gets a row
    of 4 numbers from one `nn.Embedding`, and each query id a row from a
    second, separate one. The score is the dot product of the two rows. Both
    tables start out random, so at first a query matches its own key no better
    than any other.

    One shared table would match equal ids from the start, since a row's dot
    product with itself is large. Two tables leave "match" for training to
    define, which is what later lets a query match something other than an
    equal id.

    Trained for 300 steps on sequences of 6 pairs, with ids from 0 to 9, it
    gives this.

    | | Untrained | Trained |
    |---|---|---|
    | Mean squared error on new sequences | 1.133 | 0.0002 |
    | Weight on the right key | 0.139 | 0.989 |

    Training made each query row score high with its own key's row and low
    with every other. The `==` became a learned match, sharp enough to act
    almost like the exact one.

    Notice what the model is. It scores every pair against the query, and
    takes a weighted sum of the values. Section 7's main block does exactly
    this, with the rows computed from the input itself.
    """,
)

DERIVE2_SOURCE = source_file(
    '''
    # derive2.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    rows = torch.ones(2, 6, 4)       # (batch, pairs, size)
    query = torch.ones(2, 4)         # (batch, size)
    scores = rows @ query.unsqueeze(2)    # shape (2, 6, 1), one dot product per pair


    # TODO: Return the value paired with query, with a plain loop and an if.
    #   keys and values are lists of the same length, and query is one of keys.
    def exact_lookup(
        keys: list[int],
        values: list[float],
        query: int,
    ) -> float:
        ...


    # TODO: Return the weights of a soft lookup, a softmax over the pairs of how
    #   well each key row matches the query row by the dot product. key_rows
    #   has shape (batch, pairs, size) and query_rows (batch, size). The result
    #   has shape (batch, pairs).
    def match_weights(
        key_rows: Tensor,
        query_rows: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. keys and values have shape (batch, pairs) and query
    #   (batch,), with the keys and the query as ids. Look the key rows up in
    #   self.keys and the query rows in self.queries, then return the values
    #   summed with match_weights, of shape (batch,).
    class SoftLookup(nn.Module):
        def __init__(
            self,
            ids: int,
            size: int,
        ) -> None:
            super().__init__()
            self.keys = nn.Embedding(ids, size)
            self.queries = nn.Embedding(ids, size)

        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            ...
    ''',
)

DERIVE2_SOLUTION = source_file(
    '''
    # derive2.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    rows = torch.ones(2, 6, 4)
    query = torch.ones(2, 4)
    scores = rows @ query.unsqueeze(2)


    def exact_lookup(
        keys: list[int],
        values: list[float],
        query: int,
    ) -> float:
        for key, value in zip(keys, values):
            if key == query:
                return value


    def match_weights(
        key_rows: Tensor,
        query_rows: Tensor,
    ) -> Tensor:
        return torch.softmax((key_rows @ query_rows.unsqueeze(2)).squeeze(2), dim=1)


    class SoftLookup(nn.Module):
        def __init__(
            self,
            ids: int,
            size: int,
        ) -> None:
            super().__init__()
            self.keys = nn.Embedding(ids, size)
            self.queries = nn.Embedding(ids, size)

        def forward(
            self,
            keys: Tensor,
            values: Tensor,
            query: Tensor,
        ) -> Tensor:
            weights = match_weights(self.keys(keys), self.queries(query))

            return (weights * values).sum(dim=1)
    ''',
)


def reference_match_weights(
    key_rows: Tensor,
    query_rows: Tensor,
) -> Tensor:
    return torch.softmax((key_rows @ query_rows.unsqueeze(2)).squeeze(2), dim=1)


class ReferenceSoftLookup(nn.Module):
    def __init__(
        self,
        ids: int,
        size: int,
    ) -> None:
        super().__init__()
        self.keys = nn.Embedding(ids, size)
        self.queries = nn.Embedding(ids, size)

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        query: Tensor,
    ) -> Tensor:
        return (reference_match_weights(self.keys(keys), self.queries(query)) * values).sum(dim=1)


def train_lookup(model: nn.Module) -> nn.Module:
    """The run the lesson's table comes from, 300 Adam steps on batches of 64 sequences of 6 pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(1)

    for _ in range(300):
        keys, values, query, answer = lookup_sequences(64, 6, 10, generator)
        loss = F.mse_loss(model(keys, values, query), answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def check_lookup_forward(found: type) -> None:
    model = expect_module("SoftLookup(10, 4)", seeded_model(found, 10, 4))
    keys, values, query, _ = lookup_sequences(3, 5, 10, torch.Generator().manual_seed(390))

    with torch.no_grad():
        expected = seeded_model(ReferenceSoftLookup, 10, 4)(keys, values, query)

    expect_tensor(
        call="SoftLookup(10, 4) on 3 sequences of 5 pairs",
        actual=model(keys, values, query),
        expected=expected,
    )


def check_lookup_learns(found: type) -> None:
    model = train_lookup(seeded_model(found, 10, 4))
    keys, values, query, answer = lookup_sequences(1000, 6, 10, torch.Generator().manual_seed(98))

    with torch.no_grad():
        error = F.mse_loss(model(keys, values, query), answer).item()

    if error > 0.01:
        raise Mismatch(
            f"After 300 steps of training, SoftLookup's mean squared error on new sequences is {error:.4f}, "
            "but it should be below 0.01.",
        )


def check_derive2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="exact_lookup",
        description="gives the value paired with the query",
        inputs=(
            ([3, 7, 1], [0.5, -2.0, 4.0], 7),
            ([9, 2, 5, 0], [1.0, 3.0, -1.5, 2.5], 0),
            ([4, 6], [-3.0, 8.0], 4),
        ),
        reference=lambda keys, values, query: values[keys.index(query)],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="match_weights",
        description="gives a softmax over the pairs of each dot product",
        inputs=(
            (Described(small_tensor(391, 2, 3, 4) / 2, "key rows of shape (2, 3, 4)"), small_tensor(392, 2, 4) / 2),
            (Described(small_tensor(393, 3, 5, 2) / 2, "key rows of shape (3, 5, 2)"), small_tensor(394, 3, 2) / 2),
        ),
        reference=reference_match_weights,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="SoftLookup",
        method="forward",
        description="weights the values by how well each key matches",
        run=check_lookup_forward,
    )
    checker.test_method(
        class_name="SoftLookup",
        method="forward",
        description="learns which key each query matches",
        run=check_lookup_learns,
    )


DERIVE2 = Exercise(
    name="derive2",
    chapter=SOFT_DESIGN,
    title="A lookup from an ==",
    lesson=DERIVE2_LESSON,
    source=DERIVE2_SOURCE,
    solution=DERIVE2_SOLUTION,
    hints={
        "exact_lookup": "Loop over zip(keys, values), and return the value when the key equals query.",
        "match_weights": "query_rows.unsqueeze(2) has shape (batch, size, 1), so key_rows @ it gives (batch, pairs, "
        "1). Take .squeeze(2), then torch.softmax with dim=1.",
        "SoftLookup.forward": "self.keys(keys) gives the key rows and self.queries(query) the query rows. Multiply "
        "match_weights of them by values and sum over dim=1.",
    },
    introduces=(),
    check=check_derive2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
