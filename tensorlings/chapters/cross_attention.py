"""51_cross_attention, queries from one sequence and keys and values from another, and long copies solved by reading the input at every step."""

from __future__ import annotations

import math
import textwrap

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
        values from another. A decoder uses it to read all of the encoder's
        outputs at every step it writes. Without it, the decoder gets the
        whole input packed into one vector. This chapter builds cross
        attention and uses it to copy long rows of digits, a task where a
        single vector fails.

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
    So far, attention took its queries, keys and values from the same
    sequence. **Cross attention** takes the queries from one sequence and the
    keys and values from another. In an encoder and decoder model, the
    decoder's queries are compared with keys made from the encoder's outputs.
    So at each step of writing, the decoder can read the part of the input it
    needs.

    The attention weights have one row per query position and one column per
    key position. The two sequences can have different lengths, so the
    weights need not be square.

    ```python
    writing = torch.zeros(2, 3, 8)    # 3 query positions, in the sequence being written
    reading = torch.zeros(2, 5, 8)    # 5 key positions, in the sequence being read
    (writing @ reading.transpose(1, 2)).shape    # torch.Size([2, 3, 5])
    ```

    Cross attention needs no causal mask. The whole input is known before the
    decoder starts, so every query may read every key. The padding mask still
    applies, because short inputs are padded, as in mask1. Calling
    `.unsqueeze(1)` turns a padding mask of shape `(batch, keys)` into shape
    `(batch, 1, keys)`. Broadcasting then applies it to every query.

    Bahdanau, Cho and Bengio added cross attention to the RNN encoder and
    decoder in 2014. With it, a decoder translating a sentence could look back
    at the words of the source sentence. The encoder and decoder models of
    54_model_types put cross attention in every block of the decoder.
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


    # TODO: queries has shape (batch, writing, size) and keys has shape
    #   (batch, reading, size). Return the dot product of every query with
    #   every key, divided by the square root of size. The result has shape
    #   (batch, writing, reading).
    def cross_scores(
        queries: Tensor,
        keys: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the attention output. Take the softmax of cross_scores over
    #   the keys, and use the weights to average values, of shape
    #   (batch, reading, size). Leave out every key where blocked, of shape
    #   (batch, reading), is True, by setting its scores to minus infinity
    #   before the softmax.
    def cross_attention(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the same result as cross_attention, computed with
    #   F.scaled_dot_product_attention. Its attn_mask is True where the query
    #   may look at the key, the opposite of blocked. The mask must broadcast
    #   to shape (batch, writing, reading).
    def with_torch(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Make the queries from x with self.query, and the
    #   keys and values from memory with self.key and self.value. Run
    #   cross_attention on them with blocked, and return the result passed
    #   through self.out.
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
        description="gives the same result as cross_attention using PyTorch",
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
        "cross_scores": "Compute queries @ keys.transpose(1, 2) and divide it by math.sqrt of the size, which is "
        "queries.shape[-1].",
        "cross_attention": "Use masked_fill to set the scores to float(\"-inf\") where blocked.unsqueeze(1) is "
        "True. Take the softmax over the last dimension, then multiply by values with @.",
        "with_torch": "Pass the queries, keys and values to F.scaled_dot_product_attention. Set attn_mask to the "
        "flipped mask, ~blocked, after adding the query dimension with .unsqueeze(1) as in cross_attention.",
        "CrossAttention.forward": "Call cross_attention(self.query(x), self.key(memory), self.value(memory), "
        "blocked), then pass the result through self.out.",
    },
    introduces=(),
    check=check_cross1,
)


# ---------------------------------------------------------------------------
# cross2
# ---------------------------------------------------------------------------

START = 10
END = 11

CROSS_PART = '''
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
        scores = self.query(x) @ self.key(memory).transpose(1, 2) / math.sqrt(self.query.out_features)
        weights = torch.softmax(scores.masked_fill(blocked.unsqueeze(1), float("-inf")), dim=-1)

        return self.out(weights @ self.value(memory))
'''

COPIER_INIT = '''
class Copier(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 16)
        self.encoder = nn.GRU(16, hidden, batch_first=True)
        self.decoder = nn.GRU(16, hidden, batch_first=True)
        self.cross = CrossAttention(hidden, hidden)
        self.head = nn.Linear(2 * hidden, 12)
'''

COPIER_SOLUTION = '''
    def read(
        self,
        outputs: Tensor,
        memory: Tensor,
    ) -> Tensor:
        nothing_blocked = torch.zeros(memory.shape[:2], dtype=torch.bool)
        found = self.cross(outputs, memory, nothing_blocked)

        return self.head(torch.cat([outputs, found], dim=-1))

    def forward(
        self,
        source: Tensor,
        so_far: Tensor,
    ) -> Tensor:
        memory, state = self.encoder(self.embedding(source))
        outputs, _ = self.decoder(self.embedding(so_far), state)

        return self.read(outputs, memory)
'''

GREEDY_COPY_SOLUTION = '''
def greedy_copy(
    model: nn.Module,
    source: Tensor,
) -> Tensor:
    memory, state = model.encoder(model.embedding(source))
    symbol = torch.full((len(source), 1), START)
    written = []

    for _ in range(source.shape[1]):
        outputs, state = model.decoder(model.embedding(symbol), state)
        symbol = model.read(outputs, memory).argmax(dim=2)
        written.append(symbol)

    return torch.cat(written, dim=1)
'''

TRAIN_COPIER_PART = '''
def train_copier(
    model: nn.Module,
    length: int,
    steps: int,
    generator: torch.Generator,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        source = torch.randint(0, 10, (64, length), generator=generator)
        so_far = torch.cat([torch.full((64, 1), START), source], dim=1)
        target = torch.cat([source, torch.full((64, 1), END)], dim=1)
        loss = F.cross_entropy(model(source, so_far).reshape(-1, 12), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model
'''


def indented_part(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


CROSS2_LESSON = prose(
    """
    encdec2 showed that a single vector fails to copy long rows of digits.
    The encoder has to pack the whole row into its last state, and the
    decoder has to recover every digit from that one state. Cross attention
    instead lets the decoder read the encoder's output at every position. So
    step 7 of the copy can look directly at position 7 of the input.

    The model here keeps the GRU encoder and decoder from encdec1. At each
    step, the decoder's output is the query, and cross1's `CrossAttention`
    compares it with every encoder output. The head then reads two vectors
    joined side by side, the decoder's output and the result of the cross
    attention.

    ```python
    torch.cat([torch.ones(2, 3, 4), torch.zeros(2, 3, 4)], dim=-1).shape    # torch.Size([2, 3, 8])
    ```

    Here are the results after 1,500 steps of training with hidden size 64,
    measured on 500 rows the models never saw. Each copy is written greedily,
    so a wrong digit is fed back in and affects the steps after it.

    | Length | One vector, digits | Whole rows | Cross attention, digits | Whole rows |
    |---|---|---|---|---|
    | 20 | 27.0% | 0% | 99.9% | 99.4% |
    | 40 | 12.2% | 0% | 98.4% | 94.4% |

    encdec2 measured digits differently, by feeding in the true earlier
    digits before each guess. Measured that way, the single vector here gets
    56.4% at length 20 and 23.2% at length 40. So cross attention does far
    better under either measure.

    Copying is the cleanest test of reading long inputs, because every digit
    of the row has to come back out in order.

    This is the fix Bahdanau, Cho and Bengio proposed in 2014 for translating
    long sentences. The layout here is the version by Luong and colleagues
    from a year later. It reads the input after each decoder step and joins
    the result to the decoder's output. Three years later, attention like
    this became the whole model, with no GRU at all.
    """,
)

CROSS2_SOURCE = source_file(
    '''
    # cross2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    START = 10
    END = 11

    side_by_side = torch.cat([torch.ones(2, 3, 4), torch.zeros(2, 3, 4)], dim=-1)    # shape (2, 3, 8)


    # CrossAttention from cross1.
''' + indented_part(CROSS_PART) + '''


    # TODO: Finish read and forward. read runs self.cross with the decoder's
    #   outputs as queries and the encoder's outputs, memory, as keys and
    #   values, with nothing blocked. Then it joins outputs and the result of
    #   self.cross along the last dimension and returns self.head of that.
    #   forward embeds source and runs self.encoder on it. Then it embeds
    #   so_far and runs self.decoder on it, starting from the encoder's final
    #   state. It returns read of the decoder's outputs and memory.
''' + indented_part(COPIER_INIT) + '''

        def read(
            self,
            outputs: Tensor,
            memory: Tensor,
        ) -> Tensor:
            ...

        def forward(
            self,
            source: Tensor,
            so_far: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return model's copy of source, of shape (batch, steps), written one
    #   symbol at a time. First run model.encoder on the embedded source. Its
    #   outputs are memory, and its final state is the decoder's first state.
    #   Start every row with the symbol START. At each step, run model.decoder
    #   for one step on the embedded symbol and the current state, and pass
    #   its outputs and memory to model.read. The most likely symbol is the
    #   next symbol, and it is fed back in at the next step. Return all the
    #   written symbols joined along dimension 1.
    def greedy_copy(
        model: nn.Module,
        source: Tensor,
    ) -> Tensor:
        ...


    # Training as in encdec2, with a learning rate of 3e-3 and the gradients clipped.
''' + indented_part(TRAIN_COPIER_PART) + '''
    ''',
)

CROSS2_SOLUTION = source_file(
    '''
    # cross2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    START = 10
    END = 11

    side_by_side = torch.cat([torch.ones(2, 3, 4), torch.zeros(2, 3, 4)], dim=-1)


''' + indented_part(CROSS_PART) + '''


''' + indented_part(COPIER_INIT) + '''

''' + indented_part(COPIER_SOLUTION) + '''


''' + indented_part(GREEDY_COPY_SOLUTION) + '''


''' + indented_part(TRAIN_COPIER_PART) + '''
    ''',
)


def copier_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "START": START,
        "END": END,
    }
    exec(CROSS_PART + COPIER_INIT + COPIER_SOLUTION + GREEDY_COPY_SOLUTION + TRAIN_COPIER_PART, namespace)

    return namespace


COPIER_REFERENCE = copier_namespace()


class OneVectorCopier(nn.Module):
    """encdec1's model, the head reading the decoder's output alone, for the lesson's comparison."""

    def __init__(
        self,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 16)
        self.encoder = nn.GRU(16, hidden, batch_first=True)
        self.decoder = nn.GRU(16, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 12)

    def read(
        self,
        outputs: Tensor,
        memory: Tensor,
    ) -> Tensor:
        return self.head(outputs)

    def forward(
        self,
        source: Tensor,
        so_far: Tensor,
    ) -> Tensor:
        memory, state = self.encoder(self.embedding(source))
        outputs, _ = self.decoder(self.embedding(so_far), state)

        return self.read(outputs, memory)


def copy_scores(
    model: nn.Module,
    length: int,
    rows: int = 500,
) -> tuple[float, float]:
    """The share of digits and of whole rows model copies right, on rows it never saw."""
    source = torch.randint(0, 10, (rows, length), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        written = COPIER_REFERENCE["greedy_copy"](model, source)

    right = written == source

    return right.float().mean().item(), right.all(dim=1).float().mean().item()


def forced_digits(
    model: nn.Module,
    length: int,
    rows: int = 500,
) -> float:
    """The share of digits right with the true ones fed in before each guess, as encdec2 measured."""
    source = torch.randint(0, 10, (rows, length), generator=torch.Generator().manual_seed(1))
    so_far = torch.cat([torch.full((rows, 1), START), source], dim=1)

    with torch.no_grad():
        guesses = model(source, so_far)[:, :-1].argmax(dim=2)

    return (guesses == source).float().mean().item()


def measure_copying(
    lengths: tuple[int, ...] = (20, 40),
    seeds: tuple[int, ...] = (0,),
) -> dict[tuple[int, str], list[tuple[float, float, float]]]:
    """cross2's table, each model trained 1,500 steps by train_copier from seed s, with training rows drawn from
    generator seed s. Gives the greedy digits and rows right, and the digits right with the true ones fed in."""
    results: dict[tuple[int, str], list[tuple[float, float, float]]] = {}

    for length in lengths:
        for name, make in (("one vector", OneVectorCopier), ("cross attention", COPIER_REFERENCE["Copier"])):
            for seed in seeds:
                torch.manual_seed(seed)
                model = COPIER_REFERENCE["train_copier"](make(), length, 1500, torch.Generator().manual_seed(seed))
                results.setdefault((length, name), []).append((*copy_scores(model, length), forced_digits(model, length)))

    return results


def seeded_copier(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make()


def check_copier_forward(found: type) -> None:
    model = expect_module("Copier()", seeded_copier(found))
    source = torch.randint(0, 10, (3, 6), generator=torch.Generator().manual_seed(520))
    so_far = torch.cat([torch.full((3, 1), START), source], dim=1)

    with torch.no_grad():
        expected = seeded_copier(COPIER_REFERENCE["Copier"])(source, so_far)

    expect_tensor(
        call="Copier() on 3 rows of 6 digits",
        actual=model(source, so_far),
        expected=expected,
    )


def check_greedy_copy(found: object) -> None:
    model = seeded_copier(COPIER_REFERENCE["Copier"])
    source = torch.randint(0, 10, (4, 7), generator=torch.Generator().manual_seed(521))

    with torch.no_grad():
        expected = COPIER_REFERENCE["greedy_copy"](model, source)
        actual = found(model, source)

    expect_tensor(
        call="greedy_copy of an untrained Copier() on 4 rows of 7",
        actual=actual,
        expected=expected,
    )


def check_copier_learns(found: type) -> None:
    model = COPIER_REFERENCE["train_copier"](seeded_copier(found), 12, 300, torch.Generator().manual_seed(0))
    _, rows = copy_scores(model, 12, 200)

    if rows < 0.5:
        raise Mismatch(f"After 300 steps on rows of 12, Copier copies {rows:.1%} of new rows exactly, but it should pass 50%.")


def check_cross2(checker: Checker) -> None:
    checker.test_method(
        class_name="Copier",
        method="forward",
        description="reads the encoder's outputs at every step",
        run=check_copier_forward,
    )
    checker.test(
        function="greedy_copy",
        description="writes one symbol at a time, reading memory at each step",
        run=check_greedy_copy,
    )
    checker.test_method(
        class_name="Copier",
        method="forward",
        description="learns to copy rows of 12 digits exactly",
        run=check_copier_learns,
    )


CROSS2 = Exercise(
    name="cross2",
    chapter=CROSS_ATTENTION,
    title="Why read the input at every step",
    lesson=CROSS2_LESSON,
    source=CROSS2_SOURCE,
    solution=CROSS2_SOLUTION,
    hints={
        "Copier.read": "torch.zeros(memory.shape[:2], dtype=torch.bool) is a mask that blocks nothing. Pass "
        "outputs, memory and that mask to self.cross. Join outputs and the result with torch.cat on dim=-1, then "
        "run self.head on it.",
        "Copier.forward": "Run self.encoder on the embedded source. It returns the outputs of every step, which "
        "are memory, and the last state. Run self.decoder on self.embedding(so_far) starting from that state. "
        "Return self.read of the decoder's outputs and memory.",
        "greedy_copy": "torch.full((len(source), 1), START) gives the first symbol of every row. At each step, "
        "run model.decoder on the embedded symbol and the state. Then model.read(outputs, memory).argmax(dim=2) "
        "is the next symbol. Keep every symbol in a list and join them with torch.cat at the end.",
    },
    introduces=(),
    check=check_cross2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
