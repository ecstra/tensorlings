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

        Tables, text, grids, series, sound, video, sets and graphs all end as
        a matrix of tokens. This chapter maps each kind to its structure and
        its tokens, shows what attention needs added to read a set, a
        sequence, a grid or a graph, then puts two kinds in one sequence.

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
    Every input this course has met ends the same way, as a matrix with one
    row per token. Here is the map, with the exercise that showed each kind.

    | Input | Its structure | What fits it | The tokens it becomes |
    |---|---|---|---|
    | a table's numbers | a meaning per column, no order | a token per column, table1 | one per column |
    | text | order, a fixed vocabulary | embeddings, bpe1 | one per piece of text |
    | a grid or an image | nearby cells go together | convolution, conv1, or patches, vit1 | 196 patches of 16 by 16 at 224 by 224 |
    | a time series | order and local shape | patches, series1 | 42 for 336 steps in PatchTST |
    | audio | thousands of samples a second | log mel frames, sound1 | 1,500 for 30 seconds in Whisper |
    | video | frames over time | tubelets, video1 | 4,900 for 2 seconds at 224 by 224 |
    | a set | no order at all | Deep Sets, sets1 | one per element |
    | a graph | edges between nodes | message passing, message1 | one per node |

    Whisper's frames come every 10 ms, so 30 seconds hold 3,000. Its second
    convolution has a stride of two, which halves them to 1,500.

    Attention reads every kind the same way, and position1 showed it ignores
    order. So what you add to the tokens is what says how they relate.

    - Nothing: attention sees a set, and shuffling the inputs shuffles the outputs the same way.
    - A position table added place by place: a sequence, as position2 did.
    - A table for the row plus one for the column: a grid.
    - A mask that lets each token look only at its neighbours: a graph, with mask1's masks.

    `attend`, given in the file, is attention with no weights of its own, so
    only what you add can change its answer.

    ```python
    torch.arange(6) // 3    # tensor([0, 0, 0, 1, 1, 1]), the row of each place in a grid 3 wide
    torch.arange(6) % 3     # tensor([0, 1, 2, 0, 1, 2]), its column
    ```

    equivariance1 tested a symmetry directly. `shuffle_gap` does that for a
    shuffle, and `reach` finds which outputs change when one token does.
    Section 14's combine5 later turns this map into a guide for picking a
    structure for a problem of your own.
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


    # TODO: Return tokens with a position added to each, the first row of
    #   table to the first token and so on. tokens has shape (count, size) or
    #   (batch, count, size), and table has at least count rows.
    def with_positions(
        tokens: Tensor,
        table: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return tokens with a grid position added. The tokens fill a grid
    #   width wide, row by row, so token k sits in row k // width and column
    #   k % width. Add the row's row of rows and the column's row of columns.
    def with_grid_positions(
        tokens: Tensor,
        rows: Tensor,
        columns: Tensor,
        width: int,
    ) -> Tensor:
        ...


    # TODO: Return which tokens of a graph may look at which, of True and
    #   False. A node may look at the nodes it has an edge to in adjacency,
    #   of 0 and 1, and at itself.
    def graph_allowed(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return the largest change, as a plain number, between attend on
    #   the shuffled elements and attend's outputs shuffled the same way. Make
    #   tokens with make_tokens in both cases, and let every token see every
    #   other.
    def shuffle_gap(
        make_tokens: Callable[[Tensor], Tensor],
        elements: Tensor,
        order: Tensor,
    ) -> float:
        ...


    # TODO: Return which of attend's outputs change, as True and False, when
    #   1.0 is added to every number of the token at node. Leave tokens as it
    #   was, and count a change of more than 1e-6.
    def reach(
        tokens: Tensor,
        allowed: Tensor,
        node: int,
    ) -> Tensor:
        ...


    # Six elements of size 8, shuffled. A set, a sequence and a grid 3 wide.
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
        description="adds each token's row and column",
        inputs=(
            (small_tensor(1104, 6, 3), small_tensor(1105, 2, 3), small_tensor(1106, 3, 3), 3),
            (small_tensor(1107, 8, 2), small_tensor(1108, 4, 2), small_tensor(1109, 2, 2), 2),
        ),
        reference=REFERENCE["with_grid_positions"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="graph_allowed",
        description="lets each node see its edges and itself",
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
        ("set_gap", "is the shuffle gap of the bare elements"),
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
        description="is what node 0 reaches on a ring",
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
        "with_positions": "Slice table to as many rows as there are tokens, tokens.shape[-2], and add.",
        "with_grid_positions": "Make the places with torch.arange of the token count. Integer division by width "
        "gives each row, the remainder gives each column. Index rows and columns with them and add both.",
        "graph_allowed": "Turn adjacency into True and False with .bool(), then or it with an identity of booleans "
        "for the diagonal.",
        "shuffle_gap": "Build an all True allowed of size len(elements). Run attend once on make_tokens(elements) and "
        "once on make_tokens(elements[order]). Shuffle the first result with [order], then take the largest "
        "absolute difference.",
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
    One sequence can hold several kinds at once. BERT set out the recipe for
    two sentences, and it carries over to any two kinds.

    - A class token comes first, as in vit1, and the answer is read from its output.
    - A separator token can mark where one part ends.
    - A learned **type embedding**, BERT's segment embedding, is added to every token and says which part it came from.

    Here the two parts are series A and B, each cut into 4 patches of 4 and
    made tokens by one shared Linear. `pairs`, given in the file, makes the
    examples. The task asks whether A's mean is the larger.

    With no types and no positions, the model sees one bag of tokens, A's
    and B's mixed. Swap A and B and the bag is the same, but the answer
    flips. No training can beat a coin flip then.

    An `nn.Embedding(3, size)` holds one row for the class token, one for A
    and one for B. Each token gets its kind's row added.

    ```python
    kinds = torch.tensor([0] + [1] * 2 + [2] * 2)    # tensor([0, 1, 1, 2, 2])
    ```

    The block is PyTorch's `nn.TransformerEncoderLayer`, as vit2 used it.
    Both models trained 600 steps of Adam on batches of 64, then were scored
    on 2,000 new pairs, on 3 seeds.

    | Model | Right on new pairs | Largest change when A and B swap |
    |---|---|---|
    | no types | 48.8% to 51.3% | under 2e-7 |
    | types | 96.4% to 97.8% | 18.7 to 29.4 |

    The model with no types gives a pair and its swap the same answer, so
    it is a coin flip. The type rows make A's tokens and B's differ, and the
    task becomes easy.

    A model that reads several kinds needs to know which token came from
    which. BERT adds a segment embedding for that. Positions can carry it
    too, but only when each kind always sits in the same place.
    """,
)


def tokens2_source(solved: bool) -> str:
    header = "# tokens2.py, a solution for comparison" if solved else "# tokens2.py"
    build = indented(SEQUENCE_SOLUTION) if solved else '''
    # TODO: Return the sequence of shape (batch, 1 + a's count + b's count,
    #   size). cls, of shape (size,), comes first for every example, then
    #   a_tokens, then b_tokens. When types is not None, add types' row 0 to
    #   the class token, row 1 to every token of A and row 2 to every token
    #   of B.
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
    # TODO: Return the largest change in model's output, as a plain number,
    #   when a and b trade places. Put model in eval mode and record nothing
    #   for autograd.
    def swap_gap(
        model: nn.Module,
        a: Tensor,
        b: Tensor,
    ) -> float:
        ...'''
    forward_todo = "" if solved else '''
    # TODO: Finish forward. Cut a and b, each of shape (batch, PATCH *
    #   PATCHES), into patches of PATCH numbers and make tokens of them with
    #   self.patch. Build the sequence with self.cls and self.types, run
    #   self.block, and return self.head of the class token's output.'''

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


    # Pairs of series and whether A's mean is the larger.
''' + indented(PAIRS_PART) + '''

''' + build + '''

''' + forward_todo + '''
''' + indented(MIXED_INIT) + '''
''' + forward + '''

''' + swap + '''


    # Training and scoring, as video2 wrote them.
''' + indented(TRAIN_PART) + f'''


    # Both models trained the same way, then scored on 2,000 new pairs and with A and B swapped.
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
        description="gives the largest change when A and B trade places",
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
        "the types, make the kind of every place, 0 then 1s then 2s, and add types of those kinds.",
        "MixedModel.forward": "reshape(len(a), -1, PATCH) cuts a into patches. Run both through self.patch, build the "
        "sequence, run self.block, and take position 0 of its output into self.head.",
        "swap_gap": "The largest absolute difference between model(a, b) and model(b, a), turned into a plain number.",
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
