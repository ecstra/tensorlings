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

        A transformer is one block repeated many times. Each block has an
        attention layer, which mixes information across positions, and an
        MLP, which works on each position separately. Each of the two has a
        norm before it and a residual connection around it. This chapter
        builds the block and then GPT-2's exact version of it. It measures
        what each part contributes by taking it out. Last, it shows how
        stacking blocks lets two attention layers do what one cannot.

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
    steps, and each step has the pre norm form from residual4,
    `x = x + f(norm(x))`.

    ```text
    x = x + attention(norm1(x))
    x = x + mlp(norm2(x))
    ```

    Attention mixes information across positions, so each position gathers
    information from the positions before it. The MLP works on each position
    separately, with the same weights at every position. So a transformer
    alternates between mixing across positions and working within each
    position, block after block.

    To find out which output positions an input position can affect, change
    that input position a little and see which outputs change. Change only
    one feature, not all of them. LayerNorm subtracts the mean of the
    features, so adding the same number to every feature would change nothing
    after the norm. In the example below, position 1 changes, and
    `.amax(dim=-1)` takes the largest change over the features of each
    position.

    ```python
    x = torch.ones(1, 3, 2)
    nudged = x.clone()
    nudged[:, 1, 0] += 1.0
    (nudged - x).abs().amax(dim=-1)[0] > 1e-6    # tensor([False,  True, False])
    ```

    Here are the outputs that change when input position 2 of 5 changes.

    | Part | Outputs that change |
    |---|---|
    | attention | 2, 3 and 4 |
    | MLP | 2 |
    | the whole block | 2, 3 and 4 |

    Do this for every input position and stack the rows, and you get a map of
    which inputs affect which outputs. Row p marks the outputs that input p
    affects. For causal attention the map is True on and above the diagonal.
    For the MLP it is True only on the diagonal. One block affects the same
    outputs as its attention. A stack of blocks reaches no further than one
    block does, but it can combine what it gathers in more ways.
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


    # TODO: Finish forward with the two pre norm steps shown in the lesson. The
    #   first step uses self.norm1 and self.attention, and the second uses
    #   self.norm2 and self.mlp.
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


    # TODO: x has shape (1, steps, size). Add 1 to feature 0 of x at position,
    #   run layer on both the original and the changed x, and return which
    #   output positions changed. The result has shape (steps,), with True
    #   where the largest change over the features is above 1e-6.
    #   Example: for an MLP, only the changed position is True.
    def positions_reached(
        layer: Callable[[Tensor], Tensor],
        x: Tensor,
        position: int,
    ) -> Tensor:
        ...


    # TODO: Return the map of shape (steps, steps) whose row p is
    #   positions_reached for position p.
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


def measure_reach_table() -> dict[str, list[int]]:
    """block1's table, the outputs that change when position 2 of 5 changes, for the check's attention, MLP and block."""
    block = seeded_module(ReferenceBlock, 8, 2)
    parts = {
        "attention": seeded_module(SelfAttention, 8, 2),
        "MLP": block.mlp,
        "the whole block": block,
    }
    x = small_tensor(533, 1, 5, 8) / 3

    return {
        name: reference_positions_reached(frozen(part), x, 2).nonzero().flatten().tolist()
        for name, part in parts.items()
    }


def distinct_norms(
    module: nn.Module,
    names: tuple[str, ...],
) -> dict[str, Tensor]:
    """module's weights with its norms given different scales and shifts. Fresh LayerNorms all start at 1 and 0, so
    a forward that runs one norm twice would otherwise pass."""
    generator = torch.Generator().manual_seed(539)

    with torch.no_grad():
        for name in names:
            norm = getattr(module, name)
            norm.weight.copy_(torch.rand(norm.weight.shape, generator=generator) + 0.5)
            norm.bias.copy_(torch.rand(norm.bias.shape, generator=generator) - 0.5)

    return module.state_dict()


def check_block_forward(found: type) -> None:
    model = expect_module("Block(8, 2)", seeded_module(found, 8, 2))
    reference = seeded_module(ReferenceBlock, 8, 2)
    expect_same_layers(
        call="Block(8, 2)",
        model=model,
        reference=reference,
    )
    model.load_state_dict(distinct_norms(reference, ("norm1", "norm2")))
    x = small_tensor(531, 2, 5, 8) / 3

    with torch.no_grad():
        expected = reference(x)
        actual = model(x)

    expect_tensor(
        call="Block(8, 2) with norm1 and norm2 given different weights, on 2 sequences of 5",
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
        description="finds the outputs that change when one input position changes",
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
        description="maps which outputs each input position affects",
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
        "positions_reached": "Clone x and add 1.0 to nudged[:, position, 0]. Then compare layer's outputs on nudged "
        "and on x the same way the lesson's example compares the two inputs.",
        "reach_map": "Call positions_reached for each position in range(x.shape[1]) and join the rows with "
        "torch.stack.",
    },
    introduces=("transformer",),
    check=check_block1,
)


# ---------------------------------------------------------------------------
# block2
# ---------------------------------------------------------------------------

BLOCK2_LESSON = prose(
    """
    GPT-2's block is the block from block1, with some fixed choices. The
    norms are LayerNorm. The MLP's hidden layer is four times as wide as the
    model. It is a Linear from size up to `4 * size`, then a GELU, then a
    Linear back down to size. GPT-2 uses a cheaper approximation of GELU
    based on tanh, from the GELU paper. PyTorch provides it as
    `nn.GELU(approximate="tanh")`, and it differs from the exact GELU by at
    most 0.0005.

    The layer names match the names in GPT-2's released weights. They are
    `ln_1`, `attn`, `ln_2` and `mlp`, with `c_fc` and `c_proj` inside the
    MLP. With the same names, GPT-2's weights can be loaded into this block.
    The Linear weights have to be transposed first, because GPT-2 stored them
    the other way round. nanoGPT loads them exactly this way.

    You can count the weights from the shapes of the layers. In attention,
    `c_attn` maps size to `3 * size` and `c_proj` maps size to size, both
    with biases. The MLP's two Linears map size to `4 * size` and back. Each
    LayerNorm holds a scale and a shift for every feature.

    | Part | Weights |
    |---|---|
    | attention | `4 * (size ** 2) + 4 * size` |
    | MLP | `8 * (size ** 2) + 5 * size` |
    | two LayerNorms | `4 * size` |

    So about two thirds of a block's weights are in its MLP. GPT-2 small has
    size 768, so each of its blocks holds 7,087,872 weights. The whole model
    has twelve blocks, a token table with 50,257 rows, a position table with
    1,024 rows and a final LayerNorm. Together they hold 124,439,808 weights.
    The output head reuses the token table, so it adds no weights. The GPT-2
    paper listed this model as 117M. Counting the released weights gives the
    number above, which is why code like nanoGPT calls it GPT-2 124M.
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


    # TODO: Finish GPT2MLP. Set c_fc to a Linear from size to 4 * size, gelu to
    #   the tanh approximation of GELU, and c_proj to a Linear from 4 * size
    #   back to size. forward runs x through all three, in that order.
    class GPT2MLP(nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.c_fc = ...
            self.gelu = ...
            self.c_proj = ...

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. It is the pre norm block from block1 with GPT-2's
    #   names, ln_1 and attn in the first step, then ln_2 and mlp in the second.
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


    # TODO: Return the number of weights, biases included, in a GPT2Block of
    #   this size. Work it out from the shapes of its layers, without building
    #   a block.
    #   Example: block_parameters(768) returns 7087872.
    def block_parameters(size: int) -> int:
        ...


    # TODO: Return the number of weights in a GPT-2 model. It has a token table
    #   with vocabulary rows, a position table with max_steps rows, layers
    #   blocks and a final LayerNorm. Its output head reuses the token table,
    #   so the head adds no weights.
    #   Example: gpt2_parameters(50257, 1024, 768, 12) returns 124439808.
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
    reference = seeded_module(ReferenceGPT2Block, 8, 2)
    expect_same_layers(
        call="GPT2Block(8, 2)",
        model=model,
        reference=reference,
    )
    model.load_state_dict(distinct_norms(reference, ("ln_1", "ln_2")))
    x = small_tensor(535, 2, 5, 8) / 3

    with torch.no_grad():
        expected = reference(x)
        actual = model(x)

    expect_tensor(
        call="GPT2Block(8, 2) with ln_1 and ln_2 given different weights, on 2 sequences of 5",
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
        "GPT2Block.forward": "Write it like the Block of block1, with self.ln_1 and self.attn in the first step, "
        "then self.ln_2 and self.mlp.",
        "block_parameters": "Add up the three rows of the lesson's table.",
        "gpt2_parameters": "Both tables have size columns. Add layers times block_parameters(size), and the scale "
        "and shift of the final LayerNorm.",
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
    """Characters in, ToyBlocks of size 64 with causal attention, six unless told otherwise, then a head, for
    block3's tables."""

    def __init__(
        self,
        vocabulary: int,
        layers: int = TEXT_LAYERS,
        **flags: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary, TEXT_SIZE)
        self.positions = nn.Embedding(TEXT_CONTEXT, TEXT_SIZE)
        self.blocks = nn.Sequential(*[ReferenceToyBlock(TEXT_SIZE, TEXT_HEADS, **flags) for _ in range(layers)])
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
    layers: int = TEXT_LAYERS,
    warmup: int = 100,
    **flags: bool,
) -> float:
    """One run behind block3's tables, on tiny Shakespeare's characters. AdamW with weight decay 0.1 on batches of
    32, warmed up over warmup steps, or not at all at 0, then cosine down to a tenth, clipped at 1. Gives the
    validation loss on 20 batches of 64."""
    text = load_shakespeare()
    characters = sorted(set(text))
    positions = {character: position for position, character in enumerate(characters)}
    data = torch.tensor([positions[character] for character in text])
    cut = int(0.9 * len(data))
    train, validation = data[:cut], data[cut:]

    torch.manual_seed(seed)
    model = TextModel(len(characters), layers, **flags)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=0.1)
    generator = torch.Generator().manual_seed(seed + 1)

    for step in range(steps):
        progress = step / steps
        ramp = min(1, (step + 1) / warmup) if warmup else 1

        for group in optimizer.param_groups:
            group["lr"] = rate * ramp * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

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
    A block has four kinds of part, which are attention, the MLP, the
    residuals and the norms. Taking each one out in turn shows what it
    contributes. The test uses real text, the tiny Shakespeare dataset read
    one character at a time. The GPT project later trains on the same text.

    The model has six blocks of size 64 with 4 heads. It reads 64 characters
    and predicts the next character at each position. Each version trained
    for 1,000 steps of AdamW with a peak learning rate of 0.01. The rate
    rose from near 0 over a warmup of 100 steps, then followed a cosine curve
    down. The gradients were clipped at 1. Here is the validation loss from
    two seeds, where lower is better.

    | Taken out | Validation loss |
    |---|---|
    | nothing | 1.737 to 1.743 |
    | attention | 2.490 |
    | the MLP | 1.908 to 1.930 |
    | the residuals | 3.353 to 3.355 |
    | the norms | diverged, the loss became nan |

    Attention is the only part that moves information between positions.
    Without it, each position sees only its own character, and the loss stays
    at 2.49. The counting bigram model of names1 gets 2.48 on this text. The
    MLP works within each position, and taking it out makes the loss worse
    by about 0.18.

    Without residuals, the six blocks never got below 3.35. That is the loss
    you get from counting how often each character appears. Dong and
    colleagues showed why. Without residuals, stacked attention layers push
    their output toward the same row repeated at every position. The
    residuals prevent that. MLPs only slow it down, which is why six blocks
    with MLPs still failed.

    Without norms, training diverged at a learning rate of 0.01, which means
    the numbers grew out of control until the loss became nan. A fair test
    gives each stack the learning rates that suit it. So here is each stack
    at the best of the rates tried.

    | Stack | Rates tried | Best |
    |---|---|---|
    | 6 blocks without norms | 0.003 to 0.01 | 1.726 to 1.770 at 0.007 |
    | 6 blocks with norms | 0.01 and 0.02 | 1.723 to 1.724 at 0.02 |
    | 12 blocks without norms | 0.001 and 0.003 | 1.948 to 1.957 at 0.001 |
    | 12 blocks with norms | 0.02 | 1.709 to 1.733 |

    The norms allow a higher learning rate. With 6 blocks, the stack without
    norms trained only at rates up to 0.007, and it still did worse on both
    seeds. With 12 blocks, it diverged at 0.003 and trained only at 0.001.
    The stack with norms trained at 0.02 and matched its 6 block loss. Xiong
    and colleagues found that a norm before each part, as here, also lets a
    transformer train without warmup.

    So every part is needed. Attention and the MLP do the computing, and the
    residuals and norms keep a deep stack trainable.

    The check trains your block on a smaller task. The input is a sequence of
    numbers, and the target at each step is the number there times the number
    before it. This needs attention, because each position must read the
    position before it.

    Here is how to turn a dict of flags into a list of the names that are
    switched on.

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


    # TODO: Finish branch and forward, for a block whose parts can be switched
    #   off. branch runs layer on norm(x), or on x itself when self.use_norm is
    #   False. It adds x to that result when self.use_residual is True, and
    #   returns it. forward runs the attention branch with norm1 when
    #   self.use_attention is True. Then it runs the MLP branch with norm2 when
    #   self.use_mlp is True, and returns x.
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
    rates: tuple[float, ...] = (0.01,),
    seeds: tuple[int, ...] = (0, 1),
    variants: tuple[tuple[str, dict[str, bool]], ...] = TOY_VARIANTS,
) -> dict[tuple[float, str], list[float]]:
    """block3's first table, the validation loss of each variant at each peak rate, from each seed."""
    results: dict[tuple[float, str], list[float]] = {}

    with pinned_threads():
        for rate in rates:
            for name, flags in variants:
                for seed in seeds:
                    results.setdefault((rate, name), []).append(train_text(rate, seed, **flags))

    return results


# (blocks, peak rate, warmup steps, variant) for block3's second table.
NORM_RUNS: tuple[tuple[int, float, int, str], ...] = (
    (6, 0.003, 100, "no norms"),
    (6, 0.005, 100, "no norms"),
    (6, 0.007, 100, "no norms"),
    (6, 0.01, 100, "no norms"),
    (6, 0.01, 100, "every part"),
    (6, 0.02, 100, "every part"),
    (12, 0.001, 100, "no norms"),
    (12, 0.003, 100, "no norms"),
    (12, 0.02, 100, "every part"),
)

# The same 6 block stacks without warmup. On two seeds they gave a mixed answer, 0.004 and 0.018 worse than with
# warmup for the full block and 0.033 and 0.018 for the one without norms, so block3 leaves Xiong's warmup result as
# a citation and shows no table for it.
WARMUP_RUNS: tuple[tuple[int, float, int, str], ...] = (
    (6, 0.005, 0, "no norms"),
    (6, 0.01, 0, "every part"),
)


def measure_norm_rates(
    runs: tuple[tuple[int, float, int, str], ...] = NORM_RUNS,
    seeds: tuple[int, ...] = (0, 1),
) -> dict[tuple[int, float, int, str], list[float]]:
    """block3's second table, each stack with and without norms at the rates it was tried at, at 6 and 12 blocks,
    from each seed. A run with warmup 0 measures the same stack without warmup."""
    variants = dict(TOY_VARIANTS)
    results: dict[tuple[int, float, int, str], list[float]] = {}

    with pinned_threads():
        for layers, rate, warmup, name in runs:
            for seed in seeds:
                loss = train_text(rate, seed, layers=layers, warmup=warmup, **variants[name])
                results.setdefault((layers, rate, warmup, name), []).append(loss)

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
        description="leaves out each part that its flags switch off",
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
        "ToyBlock.forward": "In branch, choose layer(norm(x)) or layer(x) depending on self.use_norm, then add x "
        "only when self.use_residual is True. In forward, call branch once for each part that is switched on.",
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
    Here is a task that one attention layer cannot do. A sequence of random
    tokens ends with a token that appeared once before. The answer is the
    token that came right after that earlier appearance. In `A B C D B`, the
    last token B also appears at position 1, and C follows it there, so the
    answer is C. Real text is full of this pattern. After a name appears once, the
    word that followed it is likely to follow it again.

    One layer can find the earlier B, with a query that matches keys of the
    same token. But the answer is one position later, at C. The key at C is
    computed from C and its position only, and it holds nothing about the
    token before C. So no score in one layer can pick out C.

    Two layers can do it. The first layer copies each position's previous
    token into that position, which shifts the tokens by one. Now the
    position of C also holds B, the token before it. In the second layer, the
    last B's query looks for a position whose previous token is B. It matches
    the position of C and copies C. The first layer is called a **previous
    token head** and the second an **induction head**. Olsson and colleagues
    found this pair working together in trained models.

    You build both layers by hand, with one hot tokens and no training. The
    shift is an attention whose scores are high right below the diagonal, so
    each position looks one position back. `torch.diag` with `diagonal=-1`
    builds a matrix with a given vector right below the diagonal and zeros
    everywhere else.

    ```python
    torch.diag(torch.full((2,), 5.0), diagonal=-1)    # tensor([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
    ```

    Models made only of attention layers were also trained on this task, for
    3,000 steps, with 32 possible tokens and sequences of 24. Here is how
    often they gave the right answer on new sequences.

    | Layers | Right answers |
    |---|---|
    | 1 | 12.2% |
    | 2 | 99.9% |

    Random guessing gets 3.1%. One layer does a little better than that by
    using shortcuts, like preferring tokens that appear in the sequence. But
    one layer cannot combine a shift with a match. Stacking layers is what
    lets a model do that.
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


    # TODO: Return the first layer's output, an attention that copies each
    #   position's previous row of x into that position. x has shape
    #   (batch, steps, size). The scores are a (steps, steps) matrix that is
    #   sharpness right below the diagonal and 0 elsewhere. Use masked_weights
    #   to turn the scores into weights.
    def previous_rows(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: Return the second layer's scores, of shape (batch, steps, steps).
    #   Entry [b, i, j] is sharpness times the dot product of row i of x, the
    #   query, with row j of previous, the key.
    def induction_scores(
        x: Tensor,
        previous: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: Return the second layer's output. Compute previous with
    #   previous_rows, then the scores with induction_scores. Turn the scores
    #   into weights with masked_weights, and use them to take a weighted sum
    #   of the rows of x.
    def induction_head(
        x: Tensor,
        sharpness: float,
    ) -> Tensor:
        ...


    # TODO: tokens has shape (batch, steps). Return the answer id for each
    #   sequence, as a tensor of shape (batch,). Turn the tokens into one hot
    #   rows of size vocabulary and run induction_head on them. The answer is
    #   the id with the largest value at the last position of its output.
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
        "from steps - 1 entries equal to sharpness. Then multiply masked_weights of the scores by x.",
        "induction_scores": "previous.transpose(-2, -1) turns each key row into a column, so multiplying x by it "
        "with @ gives every dot product. Multiply the result by sharpness.",
        "induction_head": "Get previous from previous_rows and the scores from induction_scores. Then multiply "
        "masked_weights of the scores by x.",
        "answers": "F.one_hot(tokens, num_classes=vocabulary).float() gives the rows. Take position -1 of the "
        "output of induction_head, then its argmax over dim=-1.",
    },
    introduces=(),
    check=check_induction1,
)
