"""50_multihead, attention split into heads with reshape and transpose, matched with nn.MultiheadAttention, why heads, and grouped query attention."""

from __future__ import annotations

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
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

MULTIHEAD = Chapter(
    folder="50_multihead",
    title="Multi head attention",
    readme=source_file(
        """
        # Multi head attention

        One attention makes one softmax per position, so it gathers from one
        place or a blend. Multi head attention runs several smaller ones side
        by side on slices of the features. This chapter splits and merges the
        heads, matches PyTorch's layer, measures what heads buy, and shares
        keys and values across heads to save memory.

        ## Further reading

        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Shazeer, one write head is all you need](https://arxiv.org/abs/1911.02150)
        - [Ainslie and colleagues, GQA](https://arxiv.org/abs/2305.13245)
        - [PyTorch, nn.MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)
        """,
    ),
)


def reference_split_heads(
    x: Tensor,
    heads: int,
) -> Tensor:
    batch, steps, size = x.shape

    return x.reshape(batch, steps, heads, size // heads).permute(0, 2, 1, 3)


def reference_merge_heads(x: Tensor) -> Tensor:
    batch, heads, steps, head_size = x.shape

    return x.permute(0, 2, 1, 3).reshape(batch, steps, heads * head_size)


def reference_multi_head_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
) -> Tensor:
    mixed = F.scaled_dot_product_attention(
        reference_split_heads(q, heads),
        reference_split_heads(k, heads),
        reference_split_heads(v, heads),
        is_causal=True,
    )

    return reference_merge_heads(mixed)


# ---------------------------------------------------------------------------
# multihead1
# ---------------------------------------------------------------------------

MULTIHEAD1_LESSON = prose(
    """
    One attention compares queries with keys in one way, and its softmax
    makes one mix per position. Multi head attention runs several smaller
    attentions side by side, each a head with its own slice of the features.
    With size 8 and 2 heads, head 0 reads features 0 to 3 and head 1 reads
    features 4 to 7. That holds for the queries, the keys and the values.

    Splitting takes two moves. `.view` cuts the last dimension, so
    `(batch, steps, size)` becomes `(batch, steps, heads, head_size)`. Then
    `.transpose(1, 2)` swaps steps and heads, which gives
    `(batch, heads, steps, head_size)`. Now each head looks like one more
    batch, and attention works on the last two dimensions for every batch and
    every head at once.

    ```python
    x = torch.arange(16.0).view(1, 2, 8)        # one sequence, 2 steps, size 8
    x.view(1, 2, 2, 4).transpose(1, 2)[0, 1]    # head 1, tensor([[ 4.,  5.,  6.,  7.], [12., 13., 14., 15.]])
    ```

    Merging undoes it. Transpose back to `(batch, steps, heads, head_size)`,
    then join the last two dimensions. Here `.view` fails. A transpose only
    changes the order PyTorch walks through memory, so the numbers no longer
    sit in order, and `.view` needs them in order. `.reshape` copies them when
    it has to.

    ```python
    t = torch.zeros(2, 3, 4).transpose(1, 2)
    t.view(2, 12)       # RuntimeError: view size is not compatible with input tensor's size and stride
    t.reshape(2, 12)    # works
    ```

    `F.scaled_dot_product_attention` from attention5 takes any number of
    leading dimensions, so the split heads go straight in.
    """,
)

MULTIHEAD1_SOURCE = source_file(
    '''
    # multihead1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.arange(16.0).view(1, 2, 8)
    head_one = x.view(1, 2, 2, 4).transpose(1, 2)[0, 1]    # tensor([[ 4.,  5.,  6.,  7.], [12., 13., 14., 15.]])
    rows = torch.zeros(2, 3, 4).transpose(1, 2).reshape(2, 12)    # .view would fail here


    # TODO: Return x, of shape (batch, steps, size), split into heads, of shape
    #   (batch, heads, steps, size // heads).
    def split_heads(
        x: Tensor,
        heads: int,
    ) -> Tensor:
        ...


    # TODO: Return x, of shape (batch, heads, steps, head_size), with its heads
    #   joined back, of shape (batch, steps, heads * head_size).
    def merge_heads(x: Tensor) -> Tensor:
        ...


    # TODO: Return causal attention of q, k and v, each of shape
    #   (batch, steps, size), run separately in each of heads heads with
    #   F.scaled_dot_product_attention, and merged back to (batch, steps, size).
    def multi_head_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        heads: int,
    ) -> Tensor:
        ...
    ''',
)

MULTIHEAD1_SOLUTION = source_file(
    '''
    # multihead1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.arange(16.0).view(1, 2, 8)
    head_one = x.view(1, 2, 2, 4).transpose(1, 2)[0, 1]
    rows = torch.zeros(2, 3, 4).transpose(1, 2).reshape(2, 12)


    def split_heads(
        x: Tensor,
        heads: int,
    ) -> Tensor:
        batch, steps, size = x.shape

        return x.view(batch, steps, heads, size // heads).transpose(1, 2)


    def merge_heads(x: Tensor) -> Tensor:
        batch, heads, steps, head_size = x.shape

        return x.transpose(1, 2).reshape(batch, steps, heads * head_size)


    def multi_head_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        heads: int,
    ) -> Tensor:
        mixed = F.scaled_dot_product_attention(
            split_heads(q, heads),
            split_heads(k, heads),
            split_heads(v, heads),
            is_causal=True,
        )

        return merge_heads(mixed)
    ''',
)


def check_round_trip(found: object) -> None:
    x = small_tensor(500, 2, 3, 8)
    split = found(x, 4)

    if not isinstance(split, Tensor) or tuple(split.shape) != (2, 4, 3, 2):
        return

    if not torch.equal(reference_merge_heads(split), x):
        raise Mismatch("split_heads mixes features between heads. Head h should hold features h * head_size up to (h + 1) * head_size.")


def check_multihead1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="split_heads",
        description="gives each head its own slice of the features",
        inputs=(
            (small_tensor(501, 2, 3, 8), 2),
            (small_tensor(502, 1, 4, 6), 3),
        ),
        reference=reference_split_heads,
        expect=expect_tensor,
    )
    checker.test(
        function="split_heads",
        description="keeps each head's features together",
        run=check_round_trip,
    )
    check_on_inputs(
        checker=checker,
        function="merge_heads",
        description="joins the heads back into the features",
        inputs=(
            (small_tensor(503, 2, 2, 3, 4),),
            (small_tensor(504, 1, 3, 4, 2),),
        ),
        reference=reference_merge_heads,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="multi_head_attention",
        description="runs causal attention in every head, then merges",
        inputs=(
            (small_tensor(505, 2, 4, 8) / 2, small_tensor(506, 2, 4, 8) / 2, small_tensor(507, 2, 4, 8) / 2, 2),
            (small_tensor(508, 1, 5, 6) / 2, small_tensor(509, 1, 5, 6) / 2, small_tensor(510, 1, 5, 6) / 2, 3),
        ),
        reference=reference_multi_head_attention,
        expect=expect_tensor,
    )


MULTIHEAD1 = Exercise(
    name="multihead1",
    chapter=MULTIHEAD,
    title="Splitting into heads",
    lesson=MULTIHEAD1_LESSON,
    source=MULTIHEAD1_SOURCE,
    solution=MULTIHEAD1_SOLUTION,
    hints={
        "split_heads": "Unpack batch, steps, size from x.shape. .view(batch, steps, heads, size // heads), then "
        ".transpose(1, 2).",
        "merge_heads": "Unpack x.shape, .transpose(1, 2) back, then .reshape(batch, steps, heads * head_size), "
        "since .view fails after a transpose.",
        "multi_head_attention": "Split q, k and v with split_heads, pass them to F.scaled_dot_product_attention "
        "with is_causal=True, and merge_heads the result.",
    },
    introduces=(),
    check=check_multihead1,
)


# ---------------------------------------------------------------------------
# multihead2
# ---------------------------------------------------------------------------

# split_heads and merge_heads as multihead2 and multihead3 hand them over.
HEAD_HELPERS = '''
# split_heads and merge_heads from multihead1.
def split_heads(
    x: Tensor,
    heads: int,
) -> Tensor:
    batch, steps, size = x.shape

    return x.view(batch, steps, heads, size // heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    batch, heads, steps, head_size = x.shape

    return x.transpose(1, 2).reshape(batch, steps, heads * head_size)
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_HEAD_HELPERS = textwrap.indent(HEAD_HELPERS.strip("\n"), "    ")

MULTIHEAD2_LESSON = prose(
    """
    A multi head attention layer makes the queries, keys and values with one
    Linear from size to `3 * size`. Then `.chunk(3, dim=-1)` cuts its output
    in three. One big matrix multiplication runs faster than three small
    ones. After the heads, attention4's output Linear, from size to
    size, mixes what the heads found.

    PyTorch's `nn.MultiheadAttention` keeps the same weights under other
    names. Its `in_proj_weight`, of shape `(3 * size, size)`, holds the query,
    key and value weights stacked in that order, `in_proj_bias` their biases,
    and `out_proj` is the output Linear.

    ```python
    nn.MultiheadAttention(8, 2, batch_first=True).in_proj_weight.shape    # torch.Size([24, 8])
    ```

    Copy its weights into your layer and get the same output, and that proves
    the two are the same layer. `batch_first=True` makes it take
    `(batch, steps, size)` like everything here.

    Its `attn_mask` reads True as blocked, like mask1's masks, so no flip is
    needed there. It returns the output and the attention weights averaged
    over the heads, and `need_weights=False` skips the weights.

    The heads add no weights, since splitting only changes shapes. 4 heads of
    16 and 1 head of 64 both use `4 * 64 * 64 + 4 * 64`, 16,640 weights, three
    input Linears and one output Linear with their biases.
    """,
)

MULTIHEAD2_SOURCE = source_file(
    '''
    # multihead2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    packed = torch.arange(6.0)
    q, k, v = packed.chunk(3, dim=-1)    # tensor([0., 1.]), tensor([2., 3.]), tensor([4., 5.])


''' + INDENTED_HEAD_HELPERS + '''


    # TODO: Finish forward. Make q, k and v with self.qkv and .chunk, run
    #   attention in self.heads heads with F.scaled_dot_product_attention, merge
    #   them, and run self.out. blocked, when given, is True where a query may
    #   not look at a key, as in mask1.
    class MultiHeadAttention(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.heads = heads
            self.qkv = nn.Linear(size, 3 * size)
            self.out = nn.Linear(size, size)

        def forward(
            self,
            x: Tensor,
            blocked: Tensor | None = None,
        ) -> Tensor:
            ...


    # TODO: Copy the weights of theirs, an nn.MultiheadAttention, into ours, a
    #   MultiHeadAttention of the same size, without recording for autograd.
    def copy_from_torch(
        ours: nn.Module,
        theirs: nn.MultiheadAttention,
    ) -> None:
        ...


    # TODO: Return the number of weights in a multi head attention layer of
    #   size size, biases included, which doesn't depend on the heads.
    def weight_count(size: int) -> int:
        ...
    ''',
)

MULTIHEAD2_SOLUTION = source_file(
    '''
    # multihead2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    packed = torch.arange(6.0)
    q, k, v = packed.chunk(3, dim=-1)


''' + INDENTED_HEAD_HELPERS + '''


    class MultiHeadAttention(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.heads = heads
            self.qkv = nn.Linear(size, 3 * size)
            self.out = nn.Linear(size, size)

        # F.scaled_dot_product_attention reads True as a pair that takes part, so blocked is flipped.
        def forward(
            self,
            x: Tensor,
            blocked: Tensor | None = None,
        ) -> Tensor:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            allowed = None if blocked is None else ~blocked
            mixed = F.scaled_dot_product_attention(
                split_heads(q, self.heads),
                split_heads(k, self.heads),
                split_heads(v, self.heads),
                attn_mask=allowed,
            )

            return self.out(merge_heads(mixed))


    def copy_from_torch(
        ours: nn.Module,
        theirs: nn.MultiheadAttention,
    ) -> None:
        with torch.no_grad():
            ours.qkv.weight.copy_(theirs.in_proj_weight)
            ours.qkv.bias.copy_(theirs.in_proj_bias)
            ours.out.weight.copy_(theirs.out_proj.weight)
            ours.out.bias.copy_(theirs.out_proj.bias)


    def weight_count(size: int) -> int:
        return 4 * size * size + 4 * size
    ''',
)


class ReferenceMultiHeadAttention(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(size, 3 * size)
        self.out = nn.Linear(size, size)


def seeded_torch_attention(
    size: int,
    heads: int,
    seed: int,
) -> nn.MultiheadAttention:
    torch.manual_seed(seed)

    return nn.MultiheadAttention(size, heads, batch_first=True)


def reference_copy(
    ours: nn.Module,
    theirs: nn.MultiheadAttention,
) -> None:
    with torch.no_grad():
        ours.qkv.weight.copy_(theirs.in_proj_weight)
        ours.qkv.bias.copy_(theirs.in_proj_bias)
        ours.out.weight.copy_(theirs.out_proj.weight)
        ours.out.bias.copy_(theirs.out_proj.bias)


def check_matches_torch(found: type) -> None:
    for size, heads, steps, seed in ((8, 2, 4, 511), (12, 3, 5, 512)):
        theirs = seeded_torch_attention(size, heads, seed)
        ours = expect_module(f"MultiHeadAttention({size}, {heads})", found(size, heads))
        reference_copy(ours, theirs)
        x = small_tensor(seed + 100, 2, steps, size) / 2
        future = torch.ones(steps, steps).triu(diagonal=1).bool()

        for blocked, said in ((None, "no mask"), (future, "the causal mask")):
            with torch.no_grad():
                expected = theirs(x, x, x, attn_mask=blocked, need_weights=False)[0]
                actual = ours(x, blocked)

            expect_tensor(
                call=f"MultiHeadAttention({size}, {heads}) with nn.MultiheadAttention's weights, with {said}",
                actual=actual,
                expected=expected,
            )


def check_copy(found: object) -> None:
    theirs = seeded_torch_attention(8, 2, 513)
    torch.manual_seed(514)
    ours = ReferenceMultiHeadAttention(8, 2)
    found(ours, theirs)
    pairs = (
        ("qkv.weight", ours.qkv.weight, theirs.in_proj_weight),
        ("qkv.bias", ours.qkv.bias, theirs.in_proj_bias),
        ("out.weight", ours.out.weight, theirs.out_proj.weight),
        ("out.bias", ours.out.bias, theirs.out_proj.bias),
    )

    for name, mine, wanted in pairs:
        if not torch.equal(mine, wanted):
            raise Mismatch(f"After copy_from_torch, ours.{name} doesn't hold the matching weights of theirs.")


def check_multihead2(checker: Checker) -> None:
    checker.test_method(
        class_name="MultiHeadAttention",
        method="forward",
        description="matches nn.MultiheadAttention given its weights",
        run=check_matches_torch,
    )
    checker.test(
        function="copy_from_torch",
        description="copies all four weight tensors across",
        run=check_copy,
    )
    check_on_inputs(
        checker=checker,
        function="weight_count",
        description="counts the four Linears' weights and biases",
        inputs=(
            (64,),
            (10,),
        ),
        reference=lambda size: 4 * size * size + 4 * size,
        expect=expect_value,
    )


MULTIHEAD2 = Exercise(
    name="multihead2",
    chapter=MULTIHEAD,
    title="Multi head attention, matched",
    lesson=MULTIHEAD2_LESSON,
    source=MULTIHEAD2_SOURCE,
    solution=MULTIHEAD2_SOLUTION,
    hints={
        "MultiHeadAttention.forward": "q, k, v = self.qkv(x).chunk(3, dim=-1). Pass the split heads to "
        "F.scaled_dot_product_attention with attn_mask=~blocked, or None without a mask. Merge, then self.out.",
        "copy_from_torch": "Inside torch.no_grad(), .copy_ in_proj_weight and in_proj_bias into ours.qkv, and "
        "out_proj's weight and bias into ours.out.",
        "weight_count": "Three input Linears and one output Linear, each size * size weights and size biases.",
    },
    introduces=(),
    check=check_multihead2,
)


# ---------------------------------------------------------------------------
# multihead3
# ---------------------------------------------------------------------------

LOOKUP_VOCABULARY = 16
LOOKUP_SIZE = 16
LOOKUP_STEPS = 12
LOOKUP_OFFSETS = (1, 3)

MULTIHEAD3_LESSON = prose(
    """
    Each head makes one softmax, so it gathers from one place, or a blend.
    Heads earn their keep on a task that needs two places at once.

    Here every position copies two earlier tokens at once, the one right
    before it and the one 3 back, out of 16 kinds. The model is one attention layer
    of size 16 with a Linear readout. 1 head of 16 and 2 heads of 8 have the
    same weights. 600 steps each give this, on 2,000 new sequences.

    | Heads | Average of the two | Token right before | Token 3 back |
    |---|---|---|---|
    | 1 | 57.6% | 98.1% | 17.1% |
    | 2 | 100% | 100% | 100% |

    The single head settles on one place and copies it. The other place it
    barely learns, 17.1% against 6.3% for a blind guess. Two heads each take
    one place. Over three seeds, one head averaged 56.6% to 59.3% and two
    heads 100% every time.

    An MLP after the attention can pull a single head's blend apart, so here
    is a harder test with one. Each position copies four earlier tokens, 1 to
    4 back, after 1,000 steps, over three seeds.

    | Heads | Average of the four |
    |---|---|
    | 1 | 69.2% to 69.5% |
    | 2 | 91.4% to 95.0% |
    | 4 | 100% on every seed |

    Each head reads one place, so four places need four heads. The original
    paper found a single head 0.9 BLEU, its translation score, worse than its
    best number of heads.
    """,
)

LOOKUP_MODEL_INIT = '''
        def __init__(
            self,
            heads: int,
            size: int = 16,
        ) -> None:
            super().__init__()
            self.heads = heads
            self.tokens = nn.Embedding(16, size)
            self.positions = nn.Embedding(12, size)
            self.qkv = nn.Linear(size, 3 * size)
            self.out = nn.Linear(size, size)
            self.read = nn.Linear(size, 2 * 16)
'''

MULTIHEAD3_SOURCE = source_file(
    '''
    # multihead3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[5, 2, 7, 1, 4]])
    one_back = x[:, 2:4]    # tensor([[7, 1]]), the token before positions 3 and 4


''' + INDENTED_HEAD_HELPERS + '''


    # TODO: Return what each position from max(offsets) on should copy, of shape
    #   (batch, positions, len(offsets)). Entry [b, i, j] is the token offsets[j]
    #   back from position max(offsets) + i of sequence b.
    def lookup_targets(
        x: Tensor,
        offsets: tuple[int, ...],
    ) -> Tensor:
        ...


    # TODO: Finish the line that sets mixed, causal attention of q, k and v in
    #   self.heads heads, merged back to shape (batch, steps, size).
    class LookupModel(nn.Module):''' + LOOKUP_MODEL_INIT + '''
        def forward(self, x: Tensor) -> Tensor:
            batch, steps = x.shape
            h = self.tokens(x) + self.positions(torch.arange(steps))
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            mixed = ...

            return self.read(self.out(mixed)).view(batch, steps, 2, 16)


    # TODO: Return the share of lookup_targets that model copies right, from
    #   position max(offsets) on, without recording for autograd.
    def lookup_accuracy(
        model: nn.Module,
        x: Tensor,
        offsets: tuple[int, ...],
    ) -> float:
        ...
    ''',
)

MULTIHEAD3_SOLUTION = source_file(
    '''
    # multihead3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.tensor([[5, 2, 7, 1, 4]])
    one_back = x[:, 2:4]


''' + INDENTED_HEAD_HELPERS + '''


    def lookup_targets(
        x: Tensor,
        offsets: tuple[int, ...],
    ) -> Tensor:
        first = max(offsets)
        steps = x.shape[1]

        return torch.stack([x[:, first - offset : steps - offset] for offset in offsets], dim=2)


    class LookupModel(nn.Module):''' + LOOKUP_MODEL_INIT + '''
        def forward(self, x: Tensor) -> Tensor:
            batch, steps = x.shape
            h = self.tokens(x) + self.positions(torch.arange(steps))
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            mixed = merge_heads(
                F.scaled_dot_product_attention(
                    split_heads(q, self.heads),
                    split_heads(k, self.heads),
                    split_heads(v, self.heads),
                    is_causal=True,
                ),
            )

            return self.read(self.out(mixed)).view(batch, steps, 2, 16)


    def lookup_accuracy(
        model: nn.Module,
        x: Tensor,
        offsets: tuple[int, ...],
    ) -> float:
        with torch.no_grad():
            guesses = model(x)[:, max(offsets) :].argmax(dim=-1)

        return (guesses == lookup_targets(x, offsets)).float().mean().item()
    ''',
)


def reference_lookup_targets(
    x: Tensor,
    offsets: tuple[int, ...],
) -> Tensor:
    first = max(offsets)

    return torch.stack([x[:, first - offset : x.shape[1] - offset] for offset in offsets], dim=2)


class ReferenceLookupModel(nn.Module):
    def __init__(
        self,
        heads: int,
        size: int = LOOKUP_SIZE,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.tokens = nn.Embedding(LOOKUP_VOCABULARY, size)
        self.positions = nn.Embedding(LOOKUP_STEPS, size)
        self.qkv = nn.Linear(size, 3 * size)
        self.out = nn.Linear(size, size)
        self.read = nn.Linear(size, len(LOOKUP_OFFSETS) * LOOKUP_VOCABULARY)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps = x.shape
        h = self.tokens(x) + self.positions(torch.arange(steps))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        mixed = reference_multi_head_attention(q, k, v, self.heads)

        return self.read(self.out(mixed)).view(batch, steps, len(LOOKUP_OFFSETS), LOOKUP_VOCABULARY)


def lookup_sequences(
    count: int,
    generator: torch.Generator,
) -> Tensor:
    return torch.randint(0, LOOKUP_VOCABULARY, (count, LOOKUP_STEPS), generator=generator)


def seeded_lookup(
    make: type,
    heads: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(heads)


def train_lookup(model: nn.Module) -> nn.Module:
    """The run the lesson's table comes from, 600 Adam steps on batches of 64 sequences."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)
    first = max(LOOKUP_OFFSETS)

    for _ in range(600):
        x = lookup_sequences(64, generator)
        logits = model(x)[:, first:]
        loss = F.cross_entropy(logits.reshape(-1, LOOKUP_VOCABULARY), reference_lookup_targets(x, LOOKUP_OFFSETS).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def reference_lookup_accuracy(
    model: nn.Module,
    x: Tensor,
    offsets: tuple[int, ...],
) -> float:
    with torch.no_grad():
        guesses = model(x)[:, max(offsets) :].argmax(dim=-1)

    return (guesses == reference_lookup_targets(x, offsets)).float().mean().item()


def check_lookup_forward(found: type) -> None:
    x = lookup_sequences(3, torch.Generator().manual_seed(515))

    for heads in (1, 2):
        model = expect_module(f"LookupModel({heads})", seeded_lookup(found, heads))

        with torch.no_grad():
            expected = seeded_lookup(ReferenceLookupModel, heads)(x)
            actual = model(x)

        expect_tensor(
            call=f"LookupModel({heads}) on 3 sequences of 12",
            actual=actual,
            expected=expected,
        )


def check_heads_help(found: type) -> None:
    x = lookup_sequences(1000, torch.Generator().manual_seed(99))
    two = reference_lookup_accuracy(train_lookup(seeded_lookup(found, 2)), x, LOOKUP_OFFSETS)

    if two < 0.95:
        raise Mismatch(f"After 600 steps, LookupModel(2) copies {two:.1%} of tokens right, but it should pass 95%.")

    one = reference_lookup_accuracy(train_lookup(seeded_lookup(found, 1)), x, LOOKUP_OFFSETS)

    if one > 0.75:
        raise Mismatch(f"After 600 steps, LookupModel(1) copies {one:.1%} of tokens right, but one head should stay below 75%.")


def check_multihead3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="lookup_targets",
        description="gives the tokens each position should copy",
        inputs=(
            (lookup_sequences(2, torch.Generator().manual_seed(516)), (1, 3)),
            (lookup_sequences(3, torch.Generator().manual_seed(517)), (2, 1, 4)),
        ),
        reference=reference_lookup_targets,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="LookupModel",
        method="forward",
        description="runs causal attention in every head",
        run=check_lookup_forward,
    )
    check_on_inputs(
        checker=checker,
        function="lookup_accuracy",
        description="gives the share of copies right",
        inputs=(
            (
                Described(seeded_lookup(ReferenceLookupModel, 2), "an untrained LookupModel(2)"),
                lookup_sequences(50, torch.Generator().manual_seed(518)),
                LOOKUP_OFFSETS,
            ),
        ),
        reference=reference_lookup_accuracy,
        expect=expect_close,
    )
    checker.test_method(
        class_name="LookupModel",
        method="forward",
        description="trained, two heads copy both tokens and one head can't",
        run=check_heads_help,
    )


MULTIHEAD3 = Exercise(
    name="multihead3",
    chapter=MULTIHEAD,
    title="Why heads",
    lesson=MULTIHEAD3_LESSON,
    source=MULTIHEAD3_SOURCE,
    solution=MULTIHEAD3_SOLUTION,
    hints={
        "lookup_targets": "With first = max(offsets), x[:, first - offset : steps - offset] holds the token offset back "
        "for every position from first on. torch.stack one per offset with dim=2.",
        "LookupModel.forward": "Split q, k and v into self.heads heads, pass them to F.scaled_dot_product_attention "
        "with is_causal=True, and merge_heads the result.",
        "lookup_accuracy": "Inside torch.no_grad(), take model(x)[:, max(offsets):].argmax(dim=-1), compare it with "
        "lookup_targets(x, offsets), and take the mean as floats.",
    },
    introduces=(),
    check=check_multihead3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# gqa1
# ---------------------------------------------------------------------------

GQA1_LESSON = prose(
    """
    While a model writes, it keeps every earlier position's keys and values,
    so each new token computes only its own. That store is the KV cache, and
    on long texts it can outgrow the model's weights in memory. Its size in
    bytes is `2 * layers * key_value_heads * head_size * positions * bytes`,
    the 2 for the keys and the values.

    Grouped query attention shrinks it. It keeps every query head but gives
    the keys and values fewer heads, each shared by a group of query heads.
    Llama 2 70B has 80 layers and heads of size 128. It has 64 query heads and
    8 key value heads, so each key value head serves 8 query heads. Multi
    query attention is the extreme, one key value head for all.

    | Llama 2 70B, 16 bit numbers | KV cache per token | For 4,096 tokens |
    |---|---|---|
    | 64 key value heads, plain multi head | 2.5 MiB | 10 GiB |
    | 8 key value heads, as it shipped | 320 KiB | 1.25 GiB |

    Query head `i` uses key value head `i // groups`. `repeat_interleave`
    copies each key value head `groups` times in a row, which lines them up
    that way. `.repeat` tiles them, the wrong order.

    ```python
    torch.tensor([1, 2]).repeat_interleave(2)    # tensor([1, 1, 2, 2])
    torch.tensor([1, 2]).repeat(2)               # tensor([1, 2, 1, 2])
    ```

    `F.scaled_dot_product_attention` shares them itself with
    `enable_gqa=True`, without making the copies. The GQA paper found quality
    close to plain multi head attention at a speed close to multi query
    attention.
    """,
)

GQA1_SOURCE = source_file(
    '''
    # gqa1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    in_a_row = torch.tensor([1, 2]).repeat_interleave(2)    # tensor([1, 1, 2, 2])


    # TODO: Return x, of shape (batch, kv_heads, steps, head_size), with each
    #   head copied groups times in a row along dimension 1.
    def repeat_kv(
        x: Tensor,
        groups: int,
    ) -> Tensor:
        ...


    # TODO: Return causal attention of q, of shape (batch, heads, steps,
    #   head_size), with k and v of fewer heads, each shared by heads //
    #   kv_heads query heads in a row. Use repeat_kv, not enable_gqa.
    def grouped_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the KV cache's size in bytes, for positions positions.
    def kv_cache_bytes(
        layers: int,
        kv_heads: int,
        head_size: int,
        positions: int,
        bytes_per_number: int,
    ) -> int:
        ...
    ''',
)

GQA1_SOLUTION = source_file(
    '''
    # gqa1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    in_a_row = torch.tensor([1, 2]).repeat_interleave(2)


    def repeat_kv(
        x: Tensor,
        groups: int,
    ) -> Tensor:
        return x.repeat_interleave(groups, dim=1)


    def grouped_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        groups = q.shape[1] // k.shape[1]

        return F.scaled_dot_product_attention(q, repeat_kv(k, groups), repeat_kv(v, groups), is_causal=True)


    def kv_cache_bytes(
        layers: int,
        kv_heads: int,
        head_size: int,
        positions: int,
        bytes_per_number: int,
    ) -> int:
        return 2 * layers * kv_heads * head_size * positions * bytes_per_number
    ''',
)


def check_gqa1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="repeat_kv",
        description="copies each head groups times in a row",
        inputs=(
            (small_tensor(519, 1, 2, 3, 2), 2),
            (small_tensor(520, 2, 3, 2, 2), 4),
        ),
        reference=lambda x, groups: x.repeat_interleave(groups, dim=1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="grouped_attention",
        description="matches F.scaled_dot_product_attention with enable_gqa",
        inputs=(
            (small_tensor(521, 2, 8, 4, 4) / 2, small_tensor(522, 2, 2, 4, 4) / 2, small_tensor(523, 2, 2, 4, 4) / 2),
            (small_tensor(524, 1, 6, 5, 2) / 2, small_tensor(525, 1, 3, 5, 2) / 2, small_tensor(526, 1, 3, 5, 2) / 2),
        ),
        reference=lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kv_cache_bytes",
        description="counts keys and values in every layer",
        inputs=(
            (80, 8, 128, 4096, 2),
            (32, 32, 128, 1000, 4),
        ),
        reference=lambda layers, kv_heads, head_size, positions, size: 2 * layers * kv_heads * head_size * positions * size,
        expect=expect_value,
    )


GQA1 = Exercise(
    name="gqa1",
    chapter=MULTIHEAD,
    title="Grouped query attention",
    lesson=GQA1_LESSON,
    source=GQA1_SOURCE,
    solution=GQA1_SOLUTION,
    hints={
        "repeat_kv": "The lesson's repeat_interleave, with dim=1 so it copies whole heads.",
        "grouped_attention": "groups is q.shape[1] // k.shape[1]. Pass q and the repeated k and v to "
        "F.scaled_dot_product_attention with is_causal=True.",
        "kv_cache_bytes": "Multiply 2, for keys and values, by every argument.",
    },
    introduces=(),
    check=check_gqa1,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


class MlpLookupModel(nn.Module):
    """multihead3's model with an MLP after the attention, reading any number of earlier places at once."""

    def __init__(
        self,
        heads: int,
        offsets: tuple[int, ...],
        size: int = LOOKUP_SIZE,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.offsets = offsets
        self.tokens = nn.Embedding(LOOKUP_VOCABULARY, size)
        self.positions = nn.Embedding(LOOKUP_STEPS, size)
        self.qkv = nn.Linear(size, 3 * size)
        self.out = nn.Linear(size, size)
        self.mlp = nn.Sequential(nn.Linear(size, 4 * size), nn.ReLU(), nn.Linear(4 * size, size))
        self.read = nn.Linear(size, len(offsets) * LOOKUP_VOCABULARY)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps = x.shape
        h = self.tokens(x) + self.positions(torch.arange(steps))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        z = self.out(reference_multi_head_attention(q, k, v, self.heads))
        z = z + self.mlp(z)

        return self.read(z).view(batch, steps, len(self.offsets), LOOKUP_VOCABULARY)


def train_lookup_seeded(
    model: nn.Module,
    offsets: tuple[int, ...],
    steps: int,
    seed: int,
) -> nn.Module:
    """train_lookup's run for any offsets, length and seed. Seed 0 with 600 steps is train_lookup itself."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed + 1)
    first = max(offsets)

    for _ in range(steps):
        x = lookup_sequences(64, generator)
        logits = model(x)[:, first:]
        loss = F.cross_entropy(logits.reshape(-1, LOOKUP_VOCABULARY), reference_lookup_targets(x, offsets).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def measure_heads(
    seeds: range = range(3),
    with_mlp: bool = False,
) -> dict[tuple[int, int], float]:
    """multihead3's tables: the share copied right on 2,000 new sequences, by seed and number of heads. Without
    the MLP the model reads 1 and 3 back for 600 steps, and with it 1 to 4 back for 1,000 steps."""
    offsets = (1, 2, 3, 4) if with_mlp else LOOKUP_OFFSETS
    steps = 1000 if with_mlp else 600
    heads_tried = (1, 2, 4) if with_mlp else (1, 2)
    x = lookup_sequences(2000, torch.Generator().manual_seed(99))
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    results = {}

    try:
        for seed in seeds:
            for heads in heads_tried:
                torch.manual_seed(seed)
                model = MlpLookupModel(heads, offsets) if with_mlp else ReferenceLookupModel(heads)
                train_lookup_seeded(model, offsets, steps, seed)
                results[(seed, heads)] = reference_lookup_accuracy(model, x, offsets)
    finally:
        torch.set_num_threads(threads)

    return results
