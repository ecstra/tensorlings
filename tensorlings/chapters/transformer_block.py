"""53_transformer_block, the pre norm block of attention and an MLP, GPT-2's version of it, what each part buys, and why two layers do what one can't."""

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
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..data import load_shakespeare
from ..text import prose, source_file
from .linear_layers import expect_module

TRANSFORMER_BLOCK = Chapter(
    folder="53_transformer_block",
    title="The transformer block",
    readme=source_file(
        """
        # The transformer block

        A transformer is one block repeated, attention to mix across positions
        and an MLP to work within each, each wrapped in a norm and a residual.
        This chapter builds the block, then GPT-2's exact version of it,
        measures what each part buys by taking it away, and shows why stacking
        blocks lets two attention layers do what one can't.

        ## Further reading

        - [Radford and colleagues, GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        - [Xiong and colleagues, on layer normalization in the transformer](https://arxiv.org/abs/2002.04745)
        - [Dong, Cordonnier and Loukas, pure attention loses rank doubly exponentially with depth](https://arxiv.org/abs/2103.03404)
        - [Olsson and colleagues, in context learning and induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
        - [Karpathy, nanoGPT](https://github.com/karpathy/nanoGPT)
        """,
    ),
)

SELF_ATTENTION = '''
# Multi head attention with the causal mask, as multihead2 built it, with GPT-2's names.
class SelfAttention(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(size, 3 * size)
        self.c_proj = nn.Linear(size, size)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, size = x.shape
        q, k, v = self.c_attn(x).split(size, dim=2)
        q, k, v = (part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2) for part in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.c_proj(out.transpose(1, 2).reshape(batch, steps, size))
'''

# Indented like the files around it, which source_file takes the indentation off.
INDENTED_SELF_ATTENTION = textwrap.indent(SELF_ATTENTION.strip("\n"), "    ")


class SelfAttention(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(size, 3 * size)
        self.c_proj = nn.Linear(size, size)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, size = x.shape
        q, k, v = self.c_attn(x).split(size, dim=2)
        q, k, v = (part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2) for part in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.c_proj(out.transpose(1, 2).reshape(batch, steps, size))


# ---------------------------------------------------------------------------
# block1
# ---------------------------------------------------------------------------

BLOCK1_LESSON = prose(
    """
    A **transformer** is one block repeated many times. Each block has two
    parts, each one residual4's pre norm block.

    ```text
    x = x + attention(norm1(x))
    x = x + mlp(norm2(x))
    ```

    Attention mixes across positions, each one gathering from its past. The
    MLP works within each position alone, with the same weights for all. So
    a transformer mixes, then thinks, then mixes again, block after block.

    To see which positions a layer reaches, nudge one input position and see
    which outputs move. Nudge one feature, not all of them. LayerNorm takes the
    mean away, so adding the same number to every feature changes nothing it
    passes on.

    ```python
    x = torch.ones(1, 3, 2)
    nudged = x.clone()
    nudged[:, 1, 0] += 1.0
    (nudged - x).abs().amax(dim=-1)[0] > 1e-6    # tensor([False,  True, False])
    ```

    Here is which outputs change when position 2 of 5 changes.

    | Part | Outputs that change |
    |---|---|
    | attention | 2, 3 and 4 |
    | MLP | 2 |
    | the whole block | 2, 3 and 4 |

    Do it for every position and stack the rows, and you get a map of what
    reaches what. Row p marks the outputs that input p reaches. For causal
    attention it holds ones on and above the diagonal, for the MLP the
    diagonal alone. One block reaches what its attention reaches, and a stack
    of blocks can't reach any further back than that, only combine more.
    """,
)

BLOCK1_SOURCE = source_file(
    '''
    # block1.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.ones(1, 3, 2)
    nudged = x.clone()
    nudged[:, 1, 0] += 1.0
    changed = (nudged - x).abs().amax(dim=-1)[0] > 1e-6    # tensor([False,  True, False])


''' + INDENTED_SELF_ATTENTION + '''


    # TODO: Finish forward with two pre norm steps, attention with norm1 first,
    #   then mlp with norm2.
    class Block(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(size)
            self.attention = SelfAttention(size, heads)
            self.norm2 = nn.LayerNorm(size)
            self.mlp = nn.Sequential(
                nn.Linear(size, 4 * size),
                nn.ReLU(),
                nn.Linear(4 * size, size),
            )

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return which positions of layer's output change, as one True or
    #   False each, when feature 0 of x at position gets 1 added. x has shape
    #   (1, steps, size), and a change counts when it is above 1e-6.
    def positions_reached(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        position: int,
    ) -> Tensor:
        ...


    # TODO: Return the steps by steps map whose row p is positions_reached for
    #   position p.
    def reach_map(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

BLOCK1_SOLUTION = source_file(
    '''
    # block1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    x = torch.ones(1, 3, 2)
    nudged = x.clone()
    nudged[:, 1, 0] += 1.0
    changed = (nudged - x).abs().amax(dim=-1)[0] > 1e-6


''' + INDENTED_SELF_ATTENTION + '''


    class Block(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(size)
            self.attention = SelfAttention(size, heads)
            self.norm2 = nn.LayerNorm(size)
            self.mlp = nn.Sequential(
                nn.Linear(size, 4 * size),
                nn.ReLU(),
                nn.Linear(4 * size, size),
            )

        def forward(self, x: Tensor) -> Tensor:
            x = x + self.attention(self.norm1(x))

            return x + self.mlp(self.norm2(x))


    def positions_reached(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        position: int,
    ) -> Tensor:
        nudged = x.clone()
        nudged[:, position, 0] += 1.0

        return (layer(nudged) - layer(x)).abs().amax(dim=-1)[0] > 1e-6


    def reach_map(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        return torch.stack([positions_reached(layer, x, position) for position in range(x.shape[1])])
    ''',
)


class ReferenceBlock(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.attention = SelfAttention(size, heads)
        self.norm2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))

        return x + self.mlp(self.norm2(x))


def seeded_module(
    make: Callable[..., nn.Module],
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(530)

    return make(*arguments)


def frozen(module: nn.Module) -> Callable[[Tensor], Tensor]:
    """A module as a plain function that records nothing for autograd, so the learner's nudging compares cleanly."""
    def run(x: Tensor) -> Tensor:
        with torch.no_grad():
            return module(x)

    return run


def reference_positions_reached(
    layer: Callable[[Tensor], Tensor],
    x: Tensor,
    position: int,
) -> Tensor:
    nudged = x.clone()
    nudged[:, position, 0] += 1.0

    return (layer(nudged) - layer(x)).abs().amax(dim=-1)[0] > 1e-6


def check_block_forward(found: type) -> None:
    model = expect_module("Block(8, 2)", seeded_module(found, 8, 2))
    x = small_tensor(531, 2, 5, 8) / 3

    with torch.no_grad():
        expected = seeded_module(ReferenceBlock, 8, 2)(x)
        actual = model(x)

    expect_tensor(
        call="Block(8, 2) on 2 sequences of 5",
        actual=actual,
        expected=expected,
    )


def check_block1(checker: Checker) -> None:
    checker.test_method(
        class_name="Block",
        method="forward",
        description="runs pre norm attention, then a pre norm MLP",
        run=check_block_forward,
    )
    attention = seeded_module(SelfAttention, 8, 2)
    block = seeded_module(ReferenceBlock, 8, 2)
    one = small_tensor(533, 1, 5, 8) / 3
    check_on_inputs(
        checker=checker,
        function="positions_reached",
        description="finds the outputs a change at one position reaches",
        inputs=(
            (Described(frozen(attention), "an attention layer"), one, 2),
            (Described(frozen(block.mlp), "an MLP"), one, 2),
            (Described(frozen(block), "a block"), one, 1),
            (Described(frozen(block), "a block"), one, 4),
        ),
        reference=reference_positions_reached,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="reach_map",
        description="maps which outputs each input position reaches",
        inputs=(
            (Described(frozen(attention), "an attention layer"), one),
            (Described(frozen(block.mlp), "an MLP"), one),
            (Described(frozen(block), "a block"), small_tensor(534, 1, 4, 8) / 3),
        ),
        reference=lambda layer, x: torch.stack([reference_positions_reached(layer, x, position) for position in range(x.shape[1])]),
        expect=expect_tensor,
    )


BLOCK1 = Exercise(
    name="block1",
    chapter=TRANSFORMER_BLOCK,
    title="The block",
    lesson=BLOCK1_LESSON,
    source=BLOCK1_SOURCE,
    solution=BLOCK1_SOLUTION,
    hints={
        "Block.forward": "Add self.attention of self.norm1 of x to x. Then add self.mlp of self.norm2 of the "
        "result to the result.",
        "positions_reached": "Clone x and add 1.0 to nudged[:, position, 0]. Compare layer's outputs on both as the "
        "lesson's example compares the inputs.",
        "reach_map": "Call positions_reached for each position in range(x.shape[1]) and torch.stack the rows.",
    },
    introduces=("transformer",),
    check=check_block1,
)


# ---------------------------------------------------------------------------
# block2
# ---------------------------------------------------------------------------

BLOCK2_LESSON = prose(
    """
    GPT-2's block is block1's, with fixed choices. The norms are LayerNorm.
    The MLP is four times as wide as the model, a Linear up to `4 * size`, a
    GELU, and a Linear back down. GPT-2 uses the GELU paper's cheaper tanh
    approximation, `nn.GELU(approximate="tanh")`. It differs from the exact
    GELU by at most 0.0005.

    The names follow GPT-2's own trained weights, `ln_1`, `attn`, `ln_2` and
    `mlp`, then `c_fc` and `c_proj` inside the MLP. With the same names its
    weights load in, once the Linear ones are transposed, since GPT-2 stored
    them the other way round. nanoGPT does exactly that.

    Count the weights from the shapes. Attention's `c_attn` maps size to
    `3 * size` and `c_proj` maps size to size, both with biases. The MLP's two
    Linears map size to `4 * size` and back. Each LayerNorm holds a scale and
    a shift per feature.

    | Part | Weights |
    |---|---|
    | attention | `4 * size ** 2 + 4 * size` |
    | MLP | `8 * size ** 2 + 5 * size` |
    | two LayerNorms | `4 * size` |

    So two thirds of a block's weights sit in its MLP. At GPT-2 small's size of
    768, a block holds 7,087,872. Twelve blocks, a table of 50,257 token rows,
    a table of 1,024 position rows and a final LayerNorm make 124,439,808. That
    is the 124M of its name. The output head reuses the token table, so it
    adds nothing.
    """,
)

BLOCK2_SOURCE = source_file(
    '''
    # block2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    gelu = nn.GELU(approximate="tanh")
    bent = gelu(torch.tensor([-1.0, 0.0, 1.0]))    # tensor([-0.1588,  0.0000,  0.8412])


''' + INDENTED_SELF_ATTENTION + '''


    # TODO: Finish GPT2MLP. c_fc is a Linear from size to 4 * size, gelu is the
    #   tanh GELU and c_proj is a Linear back to size. forward runs all three.
    class GPT2MLP(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.c_fc = ...
            self.gelu = ...
            self.c_proj = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward, block1's pre norm block with GPT-2's names.
    class GPT2Block(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(size)
            self.attn = SelfAttention(size, heads)
            self.ln_2 = nn.LayerNorm(size)
            self.mlp = GPT2MLP(size)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return how many weights a GPT2Block of this size holds, worked out
    #   from the shapes of its layers, without building one.
    def block_parameters(size: int) -> int:
        ...


    # TODO: Return how many weights a GPT-2 model holds, a table of vocabulary
    #   token rows, a table of max_steps position rows, layers blocks and a
    #   final LayerNorm. Its head reuses the token table.
    def gpt2_parameters(
        vocabulary: int,
        max_steps: int,
        size: int,
        layers: int,
    ) -> int:
        ...
    ''',
)

BLOCK2_SOLUTION = source_file(
    '''
    # block2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    gelu = nn.GELU(approximate="tanh")
    bent = gelu(torch.tensor([-1.0, 0.0, 1.0]))


''' + INDENTED_SELF_ATTENTION + '''


    class GPT2MLP(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.c_fc = nn.Linear(size, 4 * size)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * size, size)

        def forward(self, x: Tensor) -> Tensor:
            return self.c_proj(self.gelu(self.c_fc(x)))


    class GPT2Block(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
        ) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(size)
            self.attn = SelfAttention(size, heads)
            self.ln_2 = nn.LayerNorm(size)
            self.mlp = GPT2MLP(size)

        def forward(self, x: Tensor) -> Tensor:
            x = x + self.attn(self.ln_1(x))

            return x + self.mlp(self.ln_2(x))


    def block_parameters(size: int) -> int:
        attention = 4 * size**2 + 4 * size
        mlp = 8 * size**2 + 5 * size
        norms = 4 * size

        return attention + mlp + norms


    def gpt2_parameters(
        vocabulary: int,
        max_steps: int,
        size: int,
        layers: int,
    ) -> int:
        tables = (vocabulary + max_steps) * size

        return tables + layers * block_parameters(size) + 2 * size
    ''',
)


class ReferenceGPT2MLP(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(size, 4 * size)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * size, size)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ReferenceGPT2Block(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(size)
        self.attn = SelfAttention(size, heads)
        self.ln_2 = nn.LayerNorm(size)
        self.mlp = ReferenceGPT2MLP(size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))

        return x + self.mlp(self.ln_2(x))


def counted_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def expect_same_layers(
    call: str,
    model: nn.Module,
    reference: nn.Module,
) -> None:
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    wanted = {name: tuple(value.shape) for name, value in reference.state_dict().items()}

    if shapes != wanted:
        missing = sorted(set(wanted) - set(shapes))
        raise Mismatch(
            f"{call} should hold layers named and shaped as the TODO says. "
            f"It lacks {', '.join(missing[:3]) or 'nothing'}, or a layer has another shape.",
        )


def check_gpt2_mlp(found: type) -> None:
    model = expect_module("GPT2MLP(8)", seeded_module(found, 8))
    reference = seeded_module(ReferenceGPT2MLP, 8)
    expect_same_layers(
        call="GPT2MLP(8)",
        model=model,
        reference=reference,
    )
    x = small_tensor(534, 2, 3, 8) / 3

    with torch.no_grad():
        expected = reference(x)
        actual = model(x)

    expect_tensor(
        call="GPT2MLP(8) on 2 sequences of 3",
        actual=actual,
        expected=expected,
    )


def check_gpt2_block(found: type) -> None:
    model = expect_module("GPT2Block(8, 2)", seeded_module(found, 8, 2))
    x = small_tensor(535, 2, 5, 8) / 3

    with torch.no_grad():
        expected = seeded_module(ReferenceGPT2Block, 8, 2)(x)
        actual = model(x)

    expect_tensor(
        call="GPT2Block(8, 2) on 2 sequences of 5",
        actual=actual,
        expected=expected,
    )


def counted_gpt2(
    vocabulary: int,
    max_steps: int,
    size: int,
    layers: int,
) -> int:
    """Built and counted, so the check never trusts a formula. A block of 768 takes a moment to build."""
    block = counted_parameters(ReferenceGPT2Block(size, 1))

    return (vocabulary + max_steps) * size + layers * block + 2 * size


def check_block2(checker: Checker) -> None:
    checker.test_method(
        class_name="GPT2MLP",
        method="forward",
        description="goes up to four times the size, through the tanh GELU, and back",
        run=check_gpt2_mlp,
    )
    checker.test_method(
        class_name="GPT2Block",
        method="forward",
        description="runs GPT-2's pre norm block",
        run=check_gpt2_block,
    )
    check_on_inputs(
        checker=checker,
        function="block_parameters",
        description="counts a block's weights from its shapes",
        inputs=(
            (8,),
            (64,),
            (768,),
        ),
        reference=lambda size: counted_parameters(ReferenceGPT2Block(size, 1)),
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="gpt2_parameters",
        description="counts the whole model's weights",
        inputs=(
            (50257, 1024, 768, 12),
            (65, 256, 64, 4),
        ),
        reference=counted_gpt2,
        expect=expect_value,
    )


BLOCK2 = Exercise(
    name="block2",
    chapter=TRANSFORMER_BLOCK,
    title="The GPT-2 block",
    lesson=BLOCK2_LESSON,
    source=BLOCK2_SOURCE,
    solution=BLOCK2_SOLUTION,
    hints={
        "GPT2MLP.forward": "Run self.c_fc, then self.gelu, then self.c_proj.",
        "GPT2Block.forward": "As block1's Block, with self.ln_1 and self.attn, then self.ln_2 and self.mlp.",
        "block_parameters": "Add up the three rows of the lesson's table.",
        "gpt2_parameters": "Both tables have size columns. Add layers blocks of block_parameters(size), and the final "
        "LayerNorm's scale and shift.",
    },
    introduces=(),
    check=check_block2,
)


# ---------------------------------------------------------------------------
# block3
# ---------------------------------------------------------------------------

TOY_LENGTH = 16


class ReferenceToyBlock(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
        attention: bool = True,
        mlp: bool = True,
        residual: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(size)
        self.attention = SelfAttention(size, heads)
        self.norm2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.GELU(),
            nn.Linear(4 * size, size),
        )
        self.use_attention = attention
        self.use_mlp = mlp
        self.use_residual = residual
        self.use_norm = norm

    def branch(
        self,
        x: Tensor,
        norm: nn.Module,
        layer: nn.Module,
    ) -> Tensor:
        out = layer(norm(x) if self.use_norm else x)

        return x + out if self.use_residual else out

    def forward(self, x: Tensor) -> Tensor:
        if self.use_attention:
            x = self.branch(x, self.norm1, self.attention)

        if self.use_mlp:
            x = self.branch(x, self.norm2, self.mlp)

        return x


def product_sequences(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Random numbers, and at each step the product of the number there and the one before it."""
    x = torch.randn(count, TOY_LENGTH, generator=generator)
    previous = torch.cat([torch.zeros(count, 1), x[:, :-1]], dim=1)

    return x, x * previous


class ToyModel(nn.Module):
    """Numbers in, ToyBlocks, then a head, for the product task block3's check trains."""

    def __init__(
        self,
        block: type,
        layers: int,
        **flags: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.positions = nn.Embedding(TOY_LENGTH, 32)
        self.blocks = nn.Sequential(*[block(32, 2, **flags) for _ in range(layers)])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.embed(x.unsqueeze(-1)) + self.positions(torch.arange(x.shape[1]))

        return self.head(self.norm(self.blocks(hidden))).squeeze(-1)


def train_toy(
    block: type,
    layers: int,
    steps: int,
    seed: int,
    **flags: bool,
) -> float:
    """Trains one variant on the product task and gives its error on new sequences."""
    torch.manual_seed(seed)
    model = ToyModel(block, layers, **flags)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed + 10)

    for _ in range(steps):
        x, y = product_sequences(64, generator)
        loss = F.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x, y = product_sequences(2000, torch.Generator().manual_seed(99))

    with torch.no_grad():
        out = model(x)

    # Step 0 has no step before it, so it is left out of the score.
    return F.mse_loss(out[:, 1:], y[:, 1:]).item()


TEXT_SIZE = 64
TEXT_LAYERS = 6
TEXT_HEADS = 4
TEXT_CONTEXT = 64


class TextModel(nn.Module):
    """Characters in, six ToyBlocks of size 64 with causal attention, then a head, for block3's table."""

    def __init__(
        self,
        vocabulary: int,
        **flags: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary, TEXT_SIZE)
        self.positions = nn.Embedding(TEXT_CONTEXT, TEXT_SIZE)
        self.blocks = nn.Sequential(*[ReferenceToyBlock(TEXT_SIZE, TEXT_HEADS, **flags) for _ in range(TEXT_LAYERS)])
        self.norm = nn.LayerNorm(TEXT_SIZE)
        self.head = nn.Linear(TEXT_SIZE, vocabulary)

    def forward(self, ids: Tensor) -> Tensor:
        hidden = self.embed(ids) + self.positions(torch.arange(ids.shape[1]))

        return self.head(self.norm(self.blocks(hidden)))


def text_batch(
    data: Tensor,
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """count stretches of TEXT_CONTEXT characters, and each one moved on by a character as its targets."""
    starts = torch.randint(len(data) - TEXT_CONTEXT - 1, (count,), generator=generator)
    x = torch.stack([data[start : start + TEXT_CONTEXT] for start in starts])
    y = torch.stack([data[start + 1 : start + TEXT_CONTEXT + 1] for start in starts])

    return x, y


def train_text(
    rate: float,
    seed: int,
    steps: int = 1000,
    **flags: bool,
) -> float:
    """One run behind block3's table, on tiny Shakespeare's characters. AdamW with weight decay 0.1 on batches of
    32, warmed up over 100 steps then cosine down to a tenth, clipped at 1. Gives the validation loss on 20 batches
    of 64."""
    text = load_shakespeare()
    characters = sorted(set(text))
    positions = {character: position for position, character in enumerate(characters)}
    data = torch.tensor([positions[character] for character in text])
    cut = int(0.9 * len(data))
    train, validation = data[:cut], data[cut:]

    torch.manual_seed(seed)
    model = TextModel(len(characters), **flags)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.1)
    generator = torch.Generator().manual_seed(seed + 1)

    for step in range(steps):
        progress = step / steps

        for group in optimizer.param_groups:
            group["lr"] = rate * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

        x, y = text_batch(train, 32, generator)
        loss = F.cross_entropy(model(x).reshape(-1, len(characters)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    generator = torch.Generator().manual_seed(999)

    with torch.no_grad():
        losses = [
            F.cross_entropy(model(x).reshape(-1, len(characters)), y.reshape(-1)).item()
            for x, y in (text_batch(validation, 64, generator) for _ in range(20))
        ]

    return sum(losses) / len(losses)


BLOCK3_LESSON = prose(
    """
    A block has four kinds of part, attention, the MLP, the residuals and the
    norms. Taking each one out in turn shows what it buys. The test is real
    text, tiny Shakespeare one character at a time, which the GPT project
    reads later.

    Six blocks of size 64 with 4 heads read 64 characters and guess each next
    one. Each model trained 1,000 steps of AdamW, warmed up then cosine down,
    clipped at 1, at two peak rates. Here is the validation loss from two
    seeds, where lower is better.

    | Taken out | Rate 0.01 | Rate 0.003 |
    |---|---|---|
    | nothing | 1.737 to 1.743 | 1.890 to 1.899 |
    | attention | 2.490 | 2.490 to 2.491 |
    | the MLP | 1.908 to 1.930 | 2.091 to 2.121 |
    | the residuals | 3.353 to 3.355 | 3.353 to 3.355 |
    | the norms | blew up to nan | 1.809 to 1.811 |

    Attention is the only part that moves information between positions.
    Without it each position sees only its own character, and the loss sits at
    2.49. names1's counting bigram model gets 2.48 on this text. The MLP works
    within each position, and taking it out costs about 0.18 at either rate.

    Without residuals, six blocks never got past 3.35, what counting
    characters alone gives. Dong and colleagues showed why. Stacked attention
    alone collapses its output toward one row repeated, and the skips and MLPs
    stop that.

    The norms buy the rate. Without them the stack blew up at 0.01, so it had
    to train at 0.003. There it reached 1.81, a little below the full block's
    1.89. With them 0.01 trained too, to 1.74, the best in the table. Xiong and colleagues found that a norm before
    each part, as here, lets a transformer train without warmup.

    So every part earns its place. Attention and the MLP do the work, and the
    residuals and norms keep a deep stack trainable.

    The check trains your block on a smaller task. At each step the target is
    the number there times the one before it, which needs attention.

    A dict of flags reads well as a list of what is on.

    ```python
    flags = {"attention": True, "mlp": False}
    [name for name, on in flags.items() if on]    # ['attention']
    ```
    """,
)

BLOCK3_SOURCE = source_file(
    '''
    # block3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    flags = {"attention": True, "mlp": False}
    kept = [name for name, on in flags.items() if on]    # ['attention']


''' + INDENTED_SELF_ATTENTION + '''


    # TODO: Finish branch and forward, a block whose parts switch off.
    #   branch returns layer run on norm of x, or on x itself when use_norm is
    #   False, plus x when use_residual is True. forward runs the attention
    #   branch with norm1 when use_attention is True, then the MLP branch with
    #   norm2 when use_mlp is True.
    class ToyBlock(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
            attention: bool = True,
            mlp: bool = True,
            residual: bool = True,
            norm: bool = True,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(size)
            self.attention = SelfAttention(size, heads)
            self.norm2 = nn.LayerNorm(size)
            self.mlp = nn.Sequential(
                nn.Linear(size, 4 * size),
                nn.GELU(),
                nn.Linear(4 * size, size),
            )
            self.use_attention = attention
            self.use_mlp = mlp
            self.use_residual = residual
            self.use_norm = norm

        def branch(
            self,
            x: Tensor,
            norm: nn.Module,
            layer: nn.Module,
        ) -> Tensor:
            ...

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

BLOCK3_SOLUTION = source_file(
    '''
    # block3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    flags = {"attention": True, "mlp": False}
    kept = [name for name, on in flags.items() if on]


''' + INDENTED_SELF_ATTENTION + '''


    class ToyBlock(nn.Module):
        def __init__(
            self,
            size: int,
            heads: int,
            attention: bool = True,
            mlp: bool = True,
            residual: bool = True,
            norm: bool = True,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(size)
            self.attention = SelfAttention(size, heads)
            self.norm2 = nn.LayerNorm(size)
            self.mlp = nn.Sequential(
                nn.Linear(size, 4 * size),
                nn.GELU(),
                nn.Linear(4 * size, size),
            )
            self.use_attention = attention
            self.use_mlp = mlp
            self.use_residual = residual
            self.use_norm = norm

        def branch(
            self,
            x: Tensor,
            norm: nn.Module,
            layer: nn.Module,
        ) -> Tensor:
            inner = layer(norm(x)) if self.use_norm else layer(x)

            if not self.use_residual:
                return inner

            return x + inner

        def forward(self, x: Tensor) -> Tensor:
            if self.use_attention:
                x = self.branch(x, self.norm1, self.attention)

            if self.use_mlp:
                x = self.branch(x, self.norm2, self.mlp)

            return x
    ''',
)

TOY_VARIANTS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("every part", {}),
    ("no attention", {"attention": False}),
    ("no MLP", {"mlp": False}),
    ("no residuals", {"residual": False}),
    ("no norms", {"norm": False}),
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """Two CPU threads while measuring, as block3's table was measured, so the sums add up in the same order and
    every number comes out again exactly."""
    threads = torch.get_num_threads()
    torch.set_num_threads(2)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def measure_text_ablation(
    rates: tuple[float, ...] = (0.01, 0.003),
    seeds: tuple[int, ...] = (0, 1),
    variants: tuple[tuple[str, dict[str, bool]], ...] = TOY_VARIANTS,
) -> dict[tuple[float, str], list[float]]:
    """block3's table, the validation loss of each variant at each peak rate, from each seed."""
    results: dict[tuple[float, str], list[float]] = {}

    with pinned_threads():
        for rate in rates:
            for name, flags in variants:
                for seed in seeds:
                    results.setdefault((rate, name), []).append(train_text(rate, seed, **flags))

    return results


def toy_weights() -> dict[str, Tensor]:
    """A ToyBlock's weights with the two norms given different scales and shifts, so mixing them up shows."""
    block = seeded_module(lambda: ReferenceToyBlock(8, 2))
    generator = torch.Generator().manual_seed(539)

    with torch.no_grad():
        for norm in (block.norm1, block.norm2):
            norm.weight.copy_(torch.rand(8, generator=generator) + 0.5)
            norm.bias.copy_(torch.rand(8, generator=generator) - 0.5)

    return block.state_dict()


def check_toy_block(found: type) -> None:
    x = small_tensor(540, 2, 5, 8) / 3
    weights = toy_weights()

    for name, flags in TOY_VARIANTS:
        call = f"ToyBlock(8, 2) with {name}"
        model = expect_module(call, seeded_module(lambda: found(8, 2, **flags)))
        model.load_state_dict(weights)
        reference = ReferenceToyBlock(8, 2, **flags)
        reference.load_state_dict(weights)

        with torch.no_grad():
            expected = reference(x)
            actual = model(x)

        expect_tensor(
            call=f"{call}, on 2 sequences of 5",
            actual=actual,
            expected=expected,
        )


def check_parts_matter(found: type) -> None:
    full = train_toy(found, layers=2, steps=300, seed=0)
    blind = train_toy(found, layers=2, steps=300, seed=0, attention=False)

    if full > 0.2:
        raise Mismatch(f"After 300 steps on the product task, the full ToyBlock model's error is {full:.3f}, but it should be below 0.2.")

    if blind < 0.8:
        raise Mismatch(f"Without attention, the product task's error after 300 steps is {blind:.3f}, but a model that can't see the step before should stay above 0.8.")


def check_block3(checker: Checker) -> None:
    checker.test_method(
        class_name="ToyBlock",
        method="forward",
        description="leaves out each part its flags switch off",
        run=check_toy_block,
    )
    checker.test_method(
        class_name="ToyBlock",
        method="forward",
        description="trains on the product task, and fails without attention",
        run=check_parts_matter,
    )


BLOCK3 = Exercise(
    name="block3",
    chapter=TRANSFORMER_BLOCK,
    title="Why each part",
    lesson=BLOCK3_LESSON,
    source=BLOCK3_SOURCE,
    solution=BLOCK3_SOLUTION,
    hints={
        "ToyBlock.forward": "In branch, pick layer(norm(x)) or layer(x) by self.use_norm, then add x only when "
        "self.use_residual. In forward, call branch once for each part that is switched on.",
    },
    introduces=(),
    check=check_block3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# induction1
# ---------------------------------------------------------------------------

MATCH_VOCABULARY = 32
MATCH_STEPS = 24


def match_sequences(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Random tokens whose last one appeared exactly once before, at a random place after the first, and for
    each the answer, the token that followed that earlier one. Sequences with a second earlier match are left out."""
    tokens = torch.randint(0, MATCH_VOCABULARY, (count, MATCH_STEPS), generator=generator)
    starts = torch.randint(1, MATCH_STEPS - 2, (count,), generator=generator)
    rows = torch.arange(count)
    tokens[:, -1] = tokens[rows, starts]
    answers = tokens[rows, starts + 1]
    unique = (tokens[:, :-1] == tokens[:, -1:]).sum(dim=1) == 1

    return tokens[unique], answers[unique]


class AttentionOnly(nn.Module):
    """Layers of attention alone, with residuals, learned positions and a head, for the lesson's trained table."""

    def __init__(self, layers: int) -> None:
        super().__init__()
        self.ids = nn.Embedding(MATCH_VOCABULARY, 64)
        self.positions = nn.Embedding(MATCH_STEPS, 64)
        self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(layers)])
        self.layers = nn.ModuleList([SelfAttention(64, 1) for _ in range(layers)])
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, MATCH_VOCABULARY)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.ids(tokens) + self.positions(torch.arange(tokens.shape[1]))

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        return self.head(self.norm(x))[:, -1]


def train_attention_only(
    layers: int,
    steps: int,
) -> float:
    """The run behind the lesson's table, scored on sequences it never saw."""
    torch.manual_seed(0)
    model = AttentionOnly(layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)

    for _ in range(steps):
        tokens, answers = match_sequences(128, generator)
        loss = F.cross_entropy(model(tokens), answers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tokens, answers = match_sequences(4000, torch.Generator().manual_seed(99))

    with torch.no_grad():
        return (model(tokens).argmax(dim=-1) == answers).float().mean().item()


INDUCTION1_LESSON = prose(
    """
    Here is a task one attention layer can't do. A sequence of random tokens
    ends with a token that showed up once before. The answer is the token that
    came right after that earlier one. In `A B C D B` the last B showed up
    at position 1, and C followed it, so the answer is C. Text is full of this.
    A name seen once is likely to be followed again by what followed it then.

    One layer can find the earlier B, with a query that matches keys of the
    same token. But the answer sits one place later, at C. C's key knows only
    C and its own position, nothing about what came before it. So no score in
    one layer can single C out.

    Two layers can. The first copies each position's previous token into it,
    a shift by one, so C's position also holds "after B". The second layer's
    query asks for "after B", matches C, and copies C. This pair of layers is
    an induction head, which Olsson and colleagues found in trained models.

    You build both layers by hand, with one hot tokens and no training. The
    shift is attention whose scores are high right below the diagonal, where
    each position looks one place back. `torch.diag` with `diagonal=-1` puts
    a vector there.

    ```python
    torch.diag(torch.full((2,), 5.0), diagonal=-1)    # tensor([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
    ```

    Trained on the task instead for 3,000 steps, with 32 tokens in sequences
    of 24, layers of attention alone got this far on new sequences.

    | Layers | Right answers |
    |---|---|
    | 1 | 12.2% |
    | 2 | 99.9% |

    Guessing gets 3.1%. One layer gets a little above that from shortcuts,
    like preferring tokens the sequence holds. It can't compose a shift with a
    match, and stacking is what lets a model do it.
    """,
)

INDUCTION1_SOURCE = source_file(
    '''
    # induction1.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    below = torch.diag(torch.full((2,), 5.0), diagonal=-1)    # tensor([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
    one_hot = F.one_hot(torch.tensor([2, 0]), num_classes=3).float()    # tensor([[0., 0., 1.], [1., 0., 0.]])


    # The masked softmax of attention2.
    def masked_weights(scores: Tensor) -> Tensor:
        steps = scores.shape[-1]
        future = torch.triu(torch.ones(steps, steps), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    # TODO: Return the first layer's output, attention that copies each
    #   position's previous row of x into it. Its scores are sharpness right
    #   below the diagonal and 0 elsewhere. x has shape (batch, steps, size).
    def previous_rows(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: Return the second layer's scores, of shape (batch, steps, steps),
    #   sharpness times the dot product of each row of x, the query, with each
    #   row of previous, the key.
    def induction_scores(
        x: Tensor,
        previous: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: Return the second layer's output, the rows of x weighted by
    #   masked_weights of induction_scores, with previous from previous_rows.
    def induction_head(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: Return one answer id for each sequence of tokens, which has shape
    #   (batch, steps), so a result of shape (batch,). Turn the tokens into one
    #   hot rows of size vocabulary, run induction_head, and read its last
    #   position.
    def answers(
        tokens: Tensor,
        vocabulary: int,
        sharpness: float = 20.0,
    ) -> Tensor:
        ...
    ''',
)

INDUCTION1_SOLUTION = source_file(
    '''
    # induction1.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    below = torch.diag(torch.full((2,), 5.0), diagonal=-1)
    one_hot = F.one_hot(torch.tensor([2, 0]), num_classes=3).float()


    def masked_weights(scores: Tensor) -> Tensor:
        steps = scores.shape[-1]
        future = torch.triu(torch.ones(steps, steps), diagonal=1).bool()

        return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


    def previous_rows(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        steps = x.shape[-2]
        scores = torch.diag(torch.full((steps - 1,), sharpness), diagonal=-1)

        return masked_weights(scores) @ x


    def induction_scores(
        x: Tensor,
        previous: Tensor,
        sharpness: float,
    ) -> Tensor:
        return sharpness * x @ previous.transpose(-2, -1)


    def induction_head(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        previous = previous_rows(x, sharpness)

        return masked_weights(induction_scores(x, previous, sharpness)) @ x


    def answers(
        tokens: Tensor,
        vocabulary: int,
        sharpness: float = 20.0,
    ) -> Tensor:
        x = F.one_hot(tokens, num_classes=vocabulary).float()

        return induction_head(x, sharpness)[:, -1].argmax(dim=-1)
    ''',
)


def reference_masked_weights(scores: Tensor) -> Tensor:
    steps = scores.shape[-1]

    return torch.softmax(scores.masked_fill(torch.triu(torch.ones(steps, steps), diagonal=1).bool(), float("-inf")), dim=-1)


def reference_previous_rows(
    x: Tensor,
    sharpness: float,
) -> Tensor:
    steps = x.shape[-2]

    return reference_masked_weights(torch.diag(torch.full((steps - 1,), sharpness), diagonal=-1)) @ x


def reference_induction_head(
    x: Tensor,
    sharpness: float,
) -> Tensor:
    scores = sharpness * x @ reference_previous_rows(x, sharpness).transpose(-2, -1)

    return reference_masked_weights(scores) @ x


def one_hot_rows(
    seed: int,
    batch: int,
    steps: int,
    vocabulary: int,
) -> Tensor:
    tokens = torch.randint(0, vocabulary, (batch, steps), generator=torch.Generator().manual_seed(seed))

    return F.one_hot(tokens, num_classes=vocabulary).float()


def check_finds_answers(found: object) -> None:
    tokens, wanted = match_sequences(400, torch.Generator().manual_seed(541))
    got = found(tokens, MATCH_VOCABULARY)
    expect_tensor(
        call=f"answers on {len(tokens)} sequences with one earlier match each",
        actual=got,
        expected=wanted,
    )


def check_induction1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="previous_rows",
        description="copies each position's previous row into it",
        inputs=(
            (small_tensor(542, 2, 4, 3), 20.0),
            (small_tensor(543, 1, 5, 2) / 2, 2.0),
        ),
        reference=reference_previous_rows,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="induction_scores",
        description="compares each query row with each key row",
        inputs=(
            (one_hot_rows(544, 2, 5, 4), one_hot_rows(545, 2, 5, 4), 20.0),
            (small_tensor(546, 1, 3, 2), small_tensor(547, 1, 3, 2), 0.5),
        ),
        reference=lambda x, previous, sharpness: sharpness * x @ previous.transpose(-2, -1),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="induction_head",
        description="copies the row after the earlier match",
        inputs=(
            (one_hot_rows(548, 2, 6, 4), 20.0),
            (one_hot_rows(549, 1, 5, 3), 3.0),
        ),
        reference=reference_induction_head,
        expect=expect_tensor,
    )
    checker.test(
        function="answers",
        description="answers every sequence right, built by hand",
        run=check_finds_answers,
    )


INDUCTION1 = Exercise(
    name="induction1",
    chapter=TRANSFORMER_BLOCK,
    title="Why stacking",
    lesson=INDUCTION1_LESSON,
    source=INDUCTION1_SOURCE,
    solution=INDUCTION1_SOLUTION,
    hints={
        "previous_rows": "x.shape[-2] is the number of steps. Build the scores with torch.diag as the lesson does, "
        "with steps - 1 entries of sharpness, then weight x by masked_weights of them.",
        "induction_scores": "previous.transpose(-2, -1) turns each key row into a column, so x @ it gives every "
        "dot product. Multiply by sharpness.",
        "induction_head": "Get previous from previous_rows, the scores from induction_scores, and weight x by "
        "masked_weights of the scores.",
        "answers": "F.one_hot(tokens, num_classes=vocabulary).float() gives the rows. Take position -1 of "
        "induction_head's output and its argmax over dim=-1.",
    },
    introduces=(),
    check=check_induction1,
)
