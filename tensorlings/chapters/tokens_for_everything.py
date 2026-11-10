"""98_tokens_for_everything, every kind of input as a matrix of tokens: a map of each kind's structure, the layer that
fits it and the tokens it becomes, then what attention needs added to read a set, a sequence, a grid or a graph, and
one sequence holding two kinds told apart by type embeddings."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Callable, Iterator

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
    reject_missing_return,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .attention import expect_bool_mask

TOKENS_FOR_EVERYTHING = Chapter(
    folder="98_tokens_for_everything",
    title="Tokens for everything",
    readme=source_file(
        """
        # Tokens for everything

        Tables, text, grids, time series, sound, video, sets and graphs all
        end up as a matrix of tokens. This chapter lists each kind of input
        with its structure and the tokens it becomes. It shows what you have
        to add so attention can read a set, a sequence, a grid or a graph.
        Then it puts two kinds of input into one sequence.

        ## Further reading

        - [Devlin and colleagues, BERT, with its class token, separator and segment embeddings](https://arxiv.org/abs/1810.04805)
        - [Gorishniy and colleagues, FT-Transformer, a token per column](https://arxiv.org/abs/2106.11959)
        - [Nie and colleagues, PatchTST, a series as patches](https://arxiv.org/abs/2211.14730)
        - [Radford and colleagues, Whisper, speech as log mel frames](https://arxiv.org/abs/2212.04356)
        """,
    ),
)

MEASURE_THREADS = 4

ATTEND_PART = '''
def attend(
    tokens: Tensor,
    allowed: Tensor,
) -> Tensor:
    """Attention with no weights of its own. allowed[i, j] says whether token i may look at token j."""
    scores = tokens @ tokens.transpose(-2, -1) / math.sqrt(tokens.shape[-1])
    weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)

    return weights @ tokens
'''

STRUCTURE_SOLUTION = '''
def with_positions(
    tokens: Tensor,
    table: Tensor,
) -> Tensor:
    return tokens + table[: tokens.shape[-2]]


def with_grid_positions(
    tokens: Tensor,
    rows: Tensor,
    columns: Tensor,
    width: int,
) -> Tensor:
    places = torch.arange(tokens.shape[-2])

    return tokens + rows[places // width] + columns[places % width]


def graph_allowed(adjacency: Tensor) -> Tensor:
    return adjacency.bool() | torch.eye(len(adjacency), dtype=torch.bool)
'''

PROPERTY_SOLUTION = '''
def shuffle_gap(
    make_tokens: Callable[[Tensor], Tensor],
    elements: Tensor,
    order: Tensor,
) -> float:
    everyone = torch.ones(len(elements), len(elements), dtype=torch.bool)
    outputs = attend(make_tokens(elements), everyone)
    shuffled = attend(make_tokens(elements[order]), everyone)

    return (shuffled - outputs[order]).abs().max().item()


def reach(
    tokens: Tensor,
    allowed: Tensor,
    node: int,
) -> Tensor:
    changed = tokens.clone()
    changed[node] += 1.0

    return (attend(changed, allowed) - attend(tokens, allowed)).abs().amax(dim=1) > 1e-6
'''

PAIRS_PART = '''
PATCH = 4
PATCHES = 4
SIZE = 32


def pairs(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Two series per example, A and B, each with its own mean. The label is 1 when A's mean is the larger."""
    means = torch.rand(count, 2, generator=generator) * 2 - 1
    a = means[:, :1] + torch.randn(count, PATCH * PATCHES, generator=generator)
    b = means[:, 1:] + torch.randn(count, PATCH * PATCHES, generator=generator)

    return a, b, (a.mean(dim=1) > b.mean(dim=1)).long()
'''

SEQUENCE_SOLUTION = '''
def build_sequence(
    cls: Tensor,
    a_tokens: Tensor,
    b_tokens: Tensor,
    types: nn.Embedding | None,
) -> Tensor:
    batch = len(a_tokens)
    sequence = torch.cat([cls.expand(batch, 1, -1), a_tokens, b_tokens], dim=1)

    if types is None:
        return sequence

    kinds = torch.tensor([0] + [1] * a_tokens.shape[1] + [2] * b_tokens.shape[1])

    return sequence + types(kinds)
'''

MIXED_INIT = '''
class MixedModel(nn.Module):
    def __init__(self, use_types: bool) -> None:
        super().__init__()
        self.patch = nn.Linear(PATCH, SIZE)
        self.cls = nn.Parameter(torch.randn(SIZE) * 0.02)
        self.types = nn.Embedding(3, SIZE) if use_types else None
        self.block = nn.TransformerEncoderLayer(SIZE, 4, 2 * SIZE, dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.Linear(SIZE, 2)
'''

MIXED_FORWARD = '''
    def forward(
        self,
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        a_tokens = self.patch(a.reshape(len(a), -1, PATCH))
        b_tokens = self.patch(b.reshape(len(b), -1, PATCH))
        sequence = build_sequence(self.cls, a_tokens, b_tokens, self.types)

        return self.head(self.block(sequence)[:, 0])
'''

SWAP_SOLUTION = '''
def swap_gap(
    model: nn.Module,
    a: Tensor,
    b: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        return (model(a, b) - model(b, a)).abs().max().item()
'''

TRAIN_PART = '''
def train_mixed(
    model: nn.Module,
    steps: int,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        a, b, labels = pairs(64, generator)
        loss = F.cross_entropy(model(a, b), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def accuracy(
    model: nn.Module,
    a: Tensor,
    b: Tensor,
    labels: Tensor,
) -> float:
    model.eval()

    with torch.no_grad():
        return (model(a, b).argmax(dim=1) == labels).float().mean().item()
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    exec(code, namespace)

    return namespace


REFERENCE = namespace_of(
    ATTEND_PART + STRUCTURE_SOLUTION + PROPERTY_SOLUTION + PAIRS_PART + SEQUENCE_SOLUTION + MIXED_INIT + MIXED_FORWARD + SWAP_SOLUTION + TRAIN_PART,
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# tokens1
# ---------------------------------------------------------------------------

TOKENS1_LESSON = prose(
    """
    Every kind of input in this course ends up the same way, as a matrix
    with one row per token. Here is a map of them, with the exercise that
    covered each kind.

    | Input | Its structure | The layer that suits it | The tokens it becomes |
    |---|---|---|---|
    | a table's numbers | a meaning per column, no order | a token per column, table1 | one per column |
    | text | order, a fixed vocabulary | embeddings of bpe1's tokens, embedding1 | one per piece of text |
    | a grid or an image | nearby cells are related | convolution, conv3, or patches, vit1 | 196 patches of 16 by 16 from 224 by 224 |
    | a time series | order and local shape | patches, series1 | 42 patches for 336 steps in PatchTST |
    | audio | thousands of samples a second | log mel frames, sound1 | 1,500 for 30 seconds in Whisper |
    | video | frames over time | tubelets, video1 | 4,900 for 2 seconds at 224 by 224 |
    | a set | no order at all | Deep Sets, sets1 | one per element |
    | a graph | edges between nodes | message passing, message1 | one per node |

    Whisper makes a frame every 10 ms, so 30 seconds give 3,000 frames. Its
    second convolution has a stride of 2, which halves that to 1,500 tokens.

    Attention treats every token the same way, and position1 showed that it
    ignores order. So whatever you add, to the tokens or as a mask, is what
    tells attention how the tokens relate.

    - Add nothing, and attention sees a set. Shuffling the inputs shuffles the outputs the same way.
    - Add a position table, one row per place, and attention sees a sequence, as in position2.
    - Add one table for the row plus one for the column, and attention sees a grid.
    - Give attention a mask that lets each token look only at its neighbours, and it sees a graph, as with mask1's masks.

    The file gives you `attend`, which is attention with no learned weights
    of its own. So only what you add can change its output. Its mask means
    the opposite of mask1's. It holds True where a token may look, which is
    how `F.scaled_dot_product_attention` reads a boolean mask.

    ```python
    torch.arange(6) // 3    # tensor([0, 0, 0, 1, 1, 1]), the row of each place in a grid 3 wide
    torch.arange(6) % 3     # tensor([0, 1, 2, 0, 1, 2]), its column
    ```

    equivariance1 measured a symmetry directly. `shuffle_gap` does the same
    for shuffling. It compares attention on shuffled inputs with attention's
    outputs shuffled the same way. `reach` finds which outputs change when
    one token changes. Later, Section 14's combine5 turns this map into a
    guide for choosing a structure for your own problem.
    """,
)

TOKENS1_SOURCE = source_file(
    '''
    # tokens1.py

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    rows_of = torch.arange(6) // 3          # tensor([0, 0, 0, 1, 1, 1])
    columns_of = torch.arange(6) % 3        # tensor([0, 1, 2, 0, 1, 2])
    first = torch.ones(5, 2)[:3].shape      # torch.Size([3, 2])
    either = torch.tensor([1, 0]).bool() | torch.tensor([False, True])    # tensor([True, True])
    diagonal = torch.eye(2, dtype=torch.bool)    # tensor([[ True, False], [False,  True]])


    # Attention with no weights, from attention5. allowed[i, j] says whether token i may look at token j.
''' + indented(ATTEND_PART) + '''


    # TODO: Return tokens with a position added to each token. Add row 0 of
    #   table to the first token, row 1 to the second, and so on. tokens has
    #   shape (count, size) or (batch, count, size), and table has shape
    #   (rows, size) with at least count rows.
    def with_positions(
        tokens: Tensor,
        table: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return tokens, of shape (count, size), with a grid position added.
    #   The tokens fill a grid `width` wide, row by row, so token k sits in row
    #   k // width and column k % width. Add rows[k // width] and
    #   columns[k % width] to token k.
    def with_grid_positions(
        tokens: Tensor,
        rows: Tensor,
        columns: Tensor,
        width: int,
    ) -> Tensor:
        ...


    # TODO: adjacency is a (nodes, nodes) matrix of 0s and 1s, with a 1 at
    #   [i, j] when node i has an edge to node j. Return a boolean matrix of
    #   the same shape that is True where node i may look at node j. A node may
    #   look at the nodes it has an edge to, and at itself.
    def graph_allowed(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return the largest absolute difference between two results, as a
    #   plain Python float. The first is attend on the shuffled elements,
    #   elements[order]. The second is attend on the original elements, with
    #   its outputs then shuffled by order. Make the tokens with make_tokens in
    #   both cases, and let every token see every other token.
    def shuffle_gap(
        make_tokens: Callable[[Tensor], Tensor],
        elements: Tensor,
        order: Tensor,
    ) -> float:
        ...


    # TODO: Add 1.0 to every number in the token at index node, and run attend
    #   before and after the change. Return a boolean tensor with one entry
    #   per token, True where that token's output changed by more than 1e-6.
    #   Do not change tokens itself.
    def reach(
        tokens: Tensor,
        allowed: Tensor,
        node: int,
    ) -> Tensor:
        ...


    # Six elements of size 8 and a shuffled order. The three gaps treat them as a set, a sequence and a grid 3 wide.
    generator = torch.Generator().manual_seed(0)
    elements = torch.randn(6, 8, generator=generator)
    order = torch.randperm(6, generator=generator)
    table = torch.randn(16, 8, generator=generator)
    grid_rows = torch.randn(4, 8, generator=generator)
    grid_columns = torch.randn(4, 8, generator=generator)
    set_gap = shuffle_gap(lambda kept: kept, elements, order)
    sequence_gap = shuffle_gap(lambda kept: with_positions(kept, table), elements, order)
    grid_gap = shuffle_gap(lambda kept: with_grid_positions(kept, grid_rows, grid_columns, 3), elements, order)

    # A ring of 6 nodes, each joined to the one before and the one after. Which outputs does node 0 reach?
    ring = torch.roll(torch.eye(6), 1, dims=1) + torch.roll(torch.eye(6), -1, dims=1)
    reached = reach(elements, graph_allowed(ring), 0).tolist()
    ''',
)

TOKENS1_SOLUTION = source_file(
    '''
    # tokens1.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    rows_of = torch.arange(6) // 3
    columns_of = torch.arange(6) % 3
    first = torch.ones(5, 2)[:3].shape
    either = torch.tensor([1, 0]).bool() | torch.tensor([False, True])
    diagonal = torch.eye(2, dtype=torch.bool)


''' + indented(ATTEND_PART) + '''


''' + indented(STRUCTURE_SOLUTION) + '''


''' + indented(PROPERTY_SOLUTION) + '''


    generator = torch.Generator().manual_seed(0)
    elements = torch.randn(6, 8, generator=generator)
    order = torch.randperm(6, generator=generator)
    table = torch.randn(16, 8, generator=generator)
    grid_rows = torch.randn(4, 8, generator=generator)
    grid_columns = torch.randn(4, 8, generator=generator)
    set_gap = shuffle_gap(lambda kept: kept, elements, order)
    sequence_gap = shuffle_gap(lambda kept: with_positions(kept, table), elements, order)
    grid_gap = shuffle_gap(lambda kept: with_grid_positions(kept, grid_rows, grid_columns, 3), elements, order)

    ring = torch.roll(torch.eye(6), 1, dims=1) + torch.roll(torch.eye(6), -1, dims=1)
    reached = reach(elements, graph_allowed(ring), 0).tolist()
    ''',
)


def lesson_inputs() -> dict[str, Tensor]:
    """The tensors tokens1's file makes, drawn in the same order."""
    generator = torch.Generator().manual_seed(0)

    return {
        "elements": torch.randn(6, 8, generator=generator),
        "order": torch.randperm(6, generator=generator),
        "table": torch.randn(16, 8, generator=generator),
        "rows": torch.randn(4, 8, generator=generator),
        "columns": torch.randn(4, 8, generator=generator),
    }


def lesson_gaps() -> dict[str, float]:
    made = lesson_inputs()
    gap = REFERENCE["shuffle_gap"]

    return {
        "set_gap": gap(lambda kept: kept, made["elements"], made["order"]),
        "sequence_gap": gap(lambda kept: REFERENCE["with_positions"](kept, made["table"]), made["elements"], made["order"]),
        "grid_gap": gap(
            lambda kept: REFERENCE["with_grid_positions"](kept, made["rows"], made["columns"], 3),
            made["elements"],
            made["order"],
        ),
    }


def ring_of(count: int) -> Tensor:
    return torch.roll(torch.eye(count), 1, dims=1) + torch.roll(torch.eye(count), -1, dims=1)


def check_with_positions(found: object) -> None:
    for tokens, table in (
        (small_tensor(1100, 3, 4), small_tensor(1101, 6, 4)),
        (small_tensor(1102, 2, 5, 3), small_tensor(1103, 5, 3)),
    ):
        call = f"with_positions({value_text(tokens)}, {value_text(table)})"
        kept = tokens.clone()
        expect_tensor(
            call=call,
            actual=found(tokens, table),
            expected=REFERENCE["with_positions"](kept, table),
        )

        if not torch.equal(tokens, kept):
            raise Mismatch(f"{call} changed tokens itself. Return a new tensor, with + rather than +=.")


def expect_gap(
    call: str,
    actual: object,
    expected: float,
) -> None:
    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-5,
    )


def check_shuffle_gap(found: object) -> None:
    elements = small_tensor(1110, 5, 4).float()
    order = torch.tensor([3, 0, 4, 1, 2])
    table = small_tensor(1111, 8, 4).float()

    for make_tokens, text in (
        (lambda kept: kept, "the elements as they are"),
        (lambda kept: REFERENCE["with_positions"](kept, table), "the elements with positions added"),
        (lambda kept: kept * 2.0, "the elements doubled"),
    ):
        expect_gap(
            call=f"shuffle_gap({text}, 5 elements of size 4, order {order.tolist()})",
            actual=found(make_tokens, elements.clone(), order),
            expected=REFERENCE["shuffle_gap"](make_tokens, elements.clone(), order),
        )


def check_reach(found: object) -> None:
    for tokens, adjacency, node in (
        (small_tensor(1120, 6, 4).float(), ring_of(6), 0),
        (small_tensor(1121, 5, 3).float(), torch.tensor([[0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]), 4),
        (small_tensor(1122, 4, 3).float(), torch.ones(4, 4), 2),
    ):
        allowed = REFERENCE["graph_allowed"](adjacency)
        call = f"reach({value_text(tokens)}, {value_text(allowed)}, {node})"
        kept = tokens.clone()
        expect_bool_mask(
            call=call,
            actual=found(tokens, allowed, node),
            expected=REFERENCE["reach"](kept, allowed, node),
        )

        if not torch.equal(tokens, kept):
            raise Mismatch(f"{call} changed tokens itself. Change a clone of it.")


def check_gap_name(name: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        expect_gap(
            call=name,
            actual=value,
            expected=lesson_gaps()[name],
        )

    return check


def check_reached(value: object) -> None:
    made = lesson_inputs()
    expected = REFERENCE["reach"](made["elements"], REFERENCE["graph_allowed"](ring_of(6)), 0).tolist()

    if value != expected:
        raise Mismatch(f"reached is {value_text(value)}, but it should be {expected}, node 0 and its two neighbours.")


def check_tokens1(checker: Checker) -> None:
    checker.test(
        function="with_positions",
        description="adds one row of the table per token",
        run=check_with_positions,
    )
    check_on_inputs(
        checker=checker,
        function="with_grid_positions",
        description="adds each token's row and column embeddings",
        inputs=(
            (small_tensor(1104, 6, 3), small_tensor(1105, 2, 3), small_tensor(1106, 3, 3), 3),
            (small_tensor(1107, 8, 2), small_tensor(1108, 4, 2), small_tensor(1109, 2, 2), 2),
            # Tables with more rows than the grid uses, as in the file, so the width must come from `width`.
            (small_tensor(1112, 6, 3), small_tensor(1113, 4, 3), small_tensor(1114, 4, 3), 3),
        ),
        reference=REFERENCE["with_grid_positions"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="graph_allowed",
        description="lets each node see the nodes it has edges to, and itself",
        inputs=(
            (ring_of(5),),
            (torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),),
            (torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),),
        ),
        reference=REFERENCE["graph_allowed"],
        expect=expect_bool_mask,
    )
    checker.test(
        function="shuffle_gap",
        description="compares attend on shuffled inputs with its outputs shuffled",
        run=check_shuffle_gap,
    )
    checker.test(
        function="reach",
        description="marks the outputs one token changes",
        run=check_reach,
    )

    for name, text in (
        ("set_gap", "is the shuffle gap of the elements with nothing added"),
        ("sequence_gap", "is the shuffle gap with positions"),
        ("grid_gap", "is the shuffle gap with grid positions"),
    ):
        checker.check_name(
            name=name,
            description=text,
            run=check_gap_name(name),
        )

    checker.check_name(
        name="reached",
        description="is which outputs change when node 0 changes, on a ring",
        run=check_reached,
    )


TOKENS1 = Exercise(
    name="tokens1",
    chapter=TOKENS_FOR_EVERYTHING,
    title="A map of every kind of input",
    lesson=TOKENS1_LESSON,
    source=TOKENS1_SOURCE,
    solution=TOKENS1_SOLUTION,
    hints={
        "with_positions": "Slice table to its first tokens.shape[-2] rows, one per token, and add the slice to tokens.",
        "with_grid_positions": "Make the places with torch.arange of the token count. Integer division by width "
        "gives each token's row, and the remainder gives its column. Index rows and columns with them and add both "
        "to tokens.",
        "graph_allowed": "Turn adjacency into booleans with .bool(). Then combine it with | and a boolean identity "
        "matrix from torch.eye, which covers the diagonal.",
        "shuffle_gap": "Build an allowed mask that is all True, of shape (len(elements), len(elements)). Run attend "
        "once on make_tokens(elements) and once on make_tokens(elements[order]). Shuffle the first result with "
        "[order], then take the largest absolute difference.",
        "reach": "Clone tokens, add 1.0 to row node of the clone, and run attend on both. A row changed if its "
        "largest absolute difference, over dim 1, is above 1e-6.",
    },
    introduces=(),
    check=check_tokens1,
)


# ---------------------------------------------------------------------------
# tokens2
# ---------------------------------------------------------------------------

TOKENS2_LESSON = prose(
    """
    One sequence can hold several kinds of input at once. BERT set out how to
    do this for two sentences, and the same recipe works for any two kinds of
    input.

    - A class token comes first, as in vit1, and the answer is read from its output.
    - A separator token can mark where one part ends.
    - A learned **type embedding**, BERT's segment embedding, is added to every token and says which part it came from.

    Here the two parts are two time series, A and B. Each is cut into 4
    patches of 4 numbers, and one shared Linear layer turns each patch into a
    token. The given function `pairs` makes the examples. The task is to say
    whether A's mean is larger than B's.

    With no types and no positions, the model sees an unordered set of
    tokens, with A's and B's mixed together. If you swap A and B, the set is
    the same, but the correct answer flips. So no amount of training can
    beat a coin flip.

    An `nn.Embedding(3, size)` holds three rows, one for the class token, one
    for A and one for B. Each token gets the row for its kind added to it.

    ```python
    kinds = torch.tensor([0] + [1] * 2 + [2] * 2)    # tensor([0, 1, 1, 2, 2])
    ```

    The model's block is PyTorch's `nn.TransformerEncoderLayer`, as in vit2.
    Both models trained for 600 steps of Adam on batches of 64. Then each
    was scored on 2,000 new pairs, with 3 seeds.

    | Model | Accuracy on new pairs | Largest change when A and B swap |
    |---|---|---|
    | no types | 48.8% to 51.3% | under 2e-7 |
    | types | 96.4% to 97.8% | 18.7 to 29.4 |

    The model with no types gives a pair and its swap the same answer, so it
    does no better than a coin flip. The type rows make A's tokens different
    from B's, and then the task becomes easy.

    A model that reads several kinds of input needs to know which kind each
    token came from. BERT adds a segment embedding for that. Position
    embeddings can carry this information too, but only when each kind
    always sits in the same positions.
    """,
)


def tokens2_source(solved: bool) -> str:
    header = "# tokens2.py, a solution for comparison" if solved else "# tokens2.py"
    build = indented(SEQUENCE_SOLUTION) if solved else '''
    # TODO: a_tokens has shape (batch, a_count, size) and b_tokens has shape
    #   (batch, b_count, size). Return the sequence of shape
    #   (batch, 1 + a_count + b_count, size). It holds cls, of shape (size,),
    #   first for every example, then a_tokens, then b_tokens. When types is
    #   not None, add row 0 of types to the class token, row 1 to every token
    #   of A, and row 2 to every token of B.
    def build_sequence(
        cls: Tensor,
        a_tokens: Tensor,
        b_tokens: Tensor,
        types: nn.Embedding | None,
    ) -> Tensor:
        ...'''
    forward = indented(MIXED_FORWARD) if solved else '''
        def forward(
            self,
            a: Tensor,
            b: Tensor,
        ) -> Tensor:
            ...'''
    swap = indented(SWAP_SOLUTION) if solved else '''
    # TODO: Return the largest absolute change in model's output when a and b
    #   swap places, as a plain Python float. Put model in eval mode and run it
    #   inside torch.no_grad().
    def swap_gap(
        model: nn.Module,
        a: Tensor,
        b: Tensor,
    ) -> float:
        ...'''
    forward_todo = "" if solved else '''
    # TODO: Finish forward. a and b each have shape (batch, PATCH * PATCHES).
    #   Cut each into patches of PATCH numbers, and turn the patches into
    #   tokens with self.patch. Build the sequence with build_sequence,
    #   self.cls and self.types, and run self.block on it. Return self.head
    #   applied to the class token's output, of shape (batch, 2).'''

    return source_file(
        f'''
    {header}

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    kinds = torch.tensor([0] + [1] * 2 + [2] * 2)                   # tensor([0, 1, 1, 2, 2])
    first = torch.zeros(4).expand(3, 1, -1).shape                   # torch.Size([3, 1, 4])
    joined = torch.cat([torch.ones(3, 1, 4), torch.zeros(3, 2, 4)], dim=1).shape    # torch.Size([3, 3, 4])
    patches = torch.arange(16.0).reshape(2, -1, 4).shape            # torch.Size([2, 2, 4])


    # Pairs of series A and B, labelled 1 when A's mean is larger.
''' + indented(PAIRS_PART) + '''

''' + build + '''

''' + forward_todo + '''
''' + indented(MIXED_INIT) + '''
''' + forward + '''

''' + swap + '''


    # Training and scoring functions, written as in video2.
''' + indented(TRAIN_PART) + f'''


    # Trains both models the same way, then scores them on 2,000 new pairs and measures their swap gaps.
    test_a, test_b, test_labels = pairs(2000, torch.Generator().manual_seed(900))
    torch.manual_seed(0)
    untyped = MixedModel(use_types=False)
    train_mixed(untyped, {TRAIN_STEPS}, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    typed = MixedModel(use_types=True)
    train_mixed(typed, {TRAIN_STEPS}, torch.Generator().manual_seed(0))
    untyped_score = accuracy(untyped, test_a, test_b, test_labels)
    typed_score = accuracy(typed, test_a, test_b, test_labels)
    untyped_gap = swap_gap(untyped, test_a, test_b)
    typed_gap = swap_gap(typed, test_a, test_b)
    ''',
    )


TRAIN_STEPS = 600


def seeded_mixed(
    make: type,
    seed: int,
    use_types: bool,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(use_types=use_types)


def check_build_sequence(found: object) -> None:
    for seed, types_text in ((1130, "nn.Embedding(3, 4)"), (1131, None)):
        torch.manual_seed(seed)
        types = nn.Embedding(3, 4) if types_text else None
        cls = small_tensor(seed, 4).float()
        a_tokens = small_tensor(seed + 10, 2, 3, 4).float()
        b_tokens = small_tensor(seed + 20, 2, 2, 4).float()
        call = f"build_sequence({value_text(cls)}, a_tokens of shape (2, 3, 4), b_tokens of shape (2, 2, 4), {types_text})"
        actual = found(cls, a_tokens, b_tokens, types)
        expect_tensor(
            call=call,
            actual=actual.detach() if isinstance(actual, Tensor) else actual,
            expected=REFERENCE["build_sequence"](cls, a_tokens, b_tokens, types).detach(),
        )


def check_forward(found: type) -> None:
    for seed, use_types in ((1140, True), (1141, False)):
        a, b, _ = REFERENCE["pairs"](5, torch.Generator().manual_seed(seed))
        model = seeded_mixed(found, seed, use_types)
        reference = seeded_mixed(REFERENCE["MixedModel"], seed, use_types)
        model.eval()
        reference.eval()
        scores = model(a, b)
        expect_tensor(
            call=f"MixedModel(use_types={use_types}) seeded with {seed}, on 5 pairs",
            actual=scores.detach() if isinstance(scores, Tensor) else scores,
            expected=reference(a, b).detach(),
        )


def recorded_modes(model: nn.Module) -> list[tuple[bool, bool]]:
    """Every forward pass's training flag and whether autograd was recording, through a forward pre hook."""
    modes: list[tuple[bool, bool]] = []
    model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))

    return modes


def check_swap_gap(found: object) -> None:
    for seed, use_types in ((1150, True), (1151, False), (1152, True)):
        a, b, _ = REFERENCE["pairs"](20, torch.Generator().manual_seed(seed))
        expect_close(
            call=f"swap_gap(MixedModel(use_types={use_types}) seeded with {seed}, 20 pairs)",
            actual=found(seeded_mixed(REFERENCE["MixedModel"], seed, use_types), a, b),
            expected=REFERENCE["swap_gap"](seeded_mixed(REFERENCE["MixedModel"], seed, use_types), a, b),
            tolerance=1e-5,
        )

    model = seeded_mixed(REFERENCE["MixedModel"], 1153, True)
    model.train()
    modes = recorded_modes(model)
    a, b, _ = REFERENCE["pairs"](4, torch.Generator().manual_seed(1153))
    found(model, a, b)

    if any(training for training, _ in modes):
        raise Mismatch("swap_gap ran the model in training mode. Call model.eval() first.")

    if any(recording for _, recording in modes):
        raise Mismatch("swap_gap recorded for autograd. Run the model inside torch.no_grad().")


def check_range(
    name: str,
    low: float,
    high: float,
    advice: str,
) -> Callable[[object], None]:
    def check(value: object) -> None:
        reject_missing_return(
            call=name,
            actual=value,
        )

        if isinstance(value, Tensor) or not isinstance(value, (int, float)):
            raise Mismatch(f"{name} is {value_text(value)}, but it should be a plain number.")

        if not low <= value <= high:
            raise Mismatch(f"{name} is {value:.4g}, but it should be between {low:g} and {high:g}. {advice}")

    return check


def check_tokens2(checker: Checker) -> None:
    checker.test(
        function="build_sequence",
        description="puts the class token, A and B in one sequence, with types when given",
        run=check_build_sequence,
    )
    checker.test_method(
        class_name="MixedModel",
        method="forward",
        description="reads both series and answers from the class token",
        run=check_forward,
    )
    checker.test(
        function="swap_gap",
        description="returns the largest change when A and B swap places",
        run=check_swap_gap,
    )

    for name, low, high, advice in (
        ("untyped_score", 0.4, 0.6, "With no types, a pair and its swap look the same, so it can't beat a coin flip."),
        ("typed_score", 0.9, 1.0, "Check build_sequence's type rows and MixedModel.forward."),
        ("untyped_gap", 0.0, 1e-4, "With no types and no positions, swapping A and B should change nothing."),
        ("typed_gap", 0.5, math.inf, "With types, A's tokens and B's differ, so the swap should change the answer."),
    ):
        checker.check_name(
            name=name,
            description="comes from the two trained models",
            run=check_range(
                name=name,
                low=low,
                high=high,
                advice=advice,
            ),
        )


TOKENS2 = Exercise(
    name="tokens2",
    chapter=TOKENS_FOR_EVERYTHING,
    title="Two kinds in one sequence",
    lesson=TOKENS2_LESSON,
    source=tokens2_source(solved=False),
    solution=tokens2_source(solved=True),
    hints={
        "build_sequence": "Expand cls to (batch, 1, size) and join it with a_tokens and b_tokens along dim 1. For "
        "the types, make a tensor with the kind of every position. That is a 0, then a 1 for each token of A, then a 2 "
        "for each token of B. Add types of that tensor to the sequence.",
        "MixedModel.forward": "reshape(len(a), -1, PATCH) cuts a into patches. Run both through self.patch, build the "
        "sequence, run self.block, and pass position 0 of its output to self.head.",
        "swap_gap": "Take the largest absolute difference between model(a, b) and model(b, a), and turn it into a "
        "plain number.",
    },
    introduces=("type embedding",),
    check=check_tokens2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_map() -> dict[str, object]:
    """tokens1's counts and gaps. The counts follow each source's settings: ViT's 16 by 16 patches of 224 by 224,
    PatchTST's look back of 336 with patches of 16 and a stride of 8, Whisper's 30 seconds of 10 ms frames halved by
    its second convolution, and video1's two seconds at 25 frames a second in tubelets of 2 by 16 by 16."""
    return {
        "vit patches": (224 // 16) ** 2,
        "patchtst patches": (336 - 16) // 8 + 2,
        "whisper frames": 30 * 1000 // 10,
        "whisper tokens": 30 * 1000 // 10 // 2,
        "video tubelets": (50 // 2) * (224 // 16) * (224 // 16),
        **lesson_gaps(),
    }


def measure_types(seeds: range = range(3)) -> dict[str, list[float]]:
    """tokens2's table: each model trained TRAIN_STEPS steps on batches of 64, scored on 2,000 other pairs, with the
    largest change when A and B swap, for each seed, on 4 threads."""
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            a, b, labels = REFERENCE["pairs"](2000, torch.Generator().manual_seed(900 + seed))

            for name, use_types in (("untyped", False), ("typed", True)):
                model = seeded_mixed(REFERENCE["MixedModel"], seed, use_types)
                REFERENCE["train_mixed"](model, TRAIN_STEPS, torch.Generator().manual_seed(seed))
                results.setdefault(f"{name} score", []).append(REFERENCE["accuracy"](model, a, b, labels))
                results.setdefault(f"{name} gap", []).append(REFERENCE["swap_gap"](model, a, b))

    return results
