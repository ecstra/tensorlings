"""55_modern_llms, GPT-2 to Llama, RoPE inside attention, the SwiGLU MLP and its width, and a Llama block next to a GPT-2 block with QK norm."""

from __future__ import annotations

import math
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
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

MODERN_LLMS = Chapter(
    folder="55_modern_llms",
    title="From GPT-2 to Llama",
    readme=source_file(
        """
        # From GPT-2 to Llama

        Llama keeps GPT-2's plan and changes a few parts, each one measured in
        an earlier paper. This chapter swaps them in one at a time, RMSNorm
        and RoPE, then the SwiGLU MLP and its width, and counts a Llama block
        against a GPT-2 block weight by weight, with the QK norm of recent
        models.

        ## Further reading

        - [Touvron and colleagues, LLaMA](https://arxiv.org/abs/2302.13971)
        - [Shazeer, GLU variants improve the transformer](https://arxiv.org/abs/2002.05202)
        - [Zhang and Sennrich, root mean square layer normalization](https://arxiv.org/abs/1910.07467)
        - [Su and colleagues, RoFormer and rotary position embedding](https://arxiv.org/abs/2104.09864)
        - [Dehghani and colleagues, scaling vision transformers to 22 billion parameters](https://arxiv.org/abs/2302.05442)
        - [Meta, the LLaMA model code](https://github.com/meta-llama/llama/blob/main/llama/model.py)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# llama1
# ---------------------------------------------------------------------------

ROPE_PART = '''
# rope1's RoPE on adjacent pairs of features, positions 0, 1, 2 and on along dim -2.
def rope_angles(
    steps: int,
    size: int,
    base: float = 10000.0,
) -> Tensor:
    frequencies = base ** (-torch.arange(0, size, 2) / size)

    return torch.arange(steps).unsqueeze(1) * frequencies


def rotate_pairs(
    x: Tensor,
    angles: Tensor,
) -> Tensor:
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = torch.cos(angles), torch.sin(angles)

    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


def rope(
    x: Tensor,
    base: float = 10000.0,
) -> Tensor:
    return rotate_pairs(x, rope_angles(x.shape[-2], x.shape[-1], base))
'''

LLAMA_ATTENTION_INIT = '''
class LlamaAttention(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.wq = nn.Linear(size, size, bias=False)
        self.wk = nn.Linear(size, size, bias=False)
        self.wv = nn.Linear(size, size, bias=False)
        self.wo = nn.Linear(size, size, bias=False)
'''

LLAMA_ATTENTION_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        batch, steps, size = x.shape
        q, k, v = (
            part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2)
            for part in (self.wq(x), self.wk(x), self.wv(x))
        )
        mixed = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)

        return self.wo(mixed.transpose(1, 2).reshape(batch, steps, size))
'''

LLAMA1_LESSON = prose(
    """
    GPT-2 came out in 2019 and Llama in 2023. Llama keeps GPT-2's plan, pre
    norm blocks of attention and an MLP, each with a residual. It changes a
    few parts, and each change comes from an earlier paper that measured it.

    | Part | GPT-2 | Llama |
    |---|---|---|
    | norm | LayerNorm before each sublayer | RMSNorm before each sublayer |
    | positions | a learned table added to the token embeddings | RoPE on the queries and keys of every layer |
    | MLP | GELU, 4 times as wide | SwiGLU, about 8/3 times as wide, which llama2 builds |
    | biases | in every Linear and norm | none |
    | output layer | shares the token embedding's weights | its own weights |

    RMSNorm, from rmsnorm1, skips taking the mean away and has no bias. Its
    paper reports quality close to LayerNorm's, with 7% to 64% less running
    time. A LayerNorm of size d holds 2d weights, a scale and a bias, and an
    RMSNorm holds d.

    GPT-2 small's position table has 1,024 rows of 768 numbers, 786,432
    weights, and a position past 1,023 has no row at all. RoPE has no weights.
    As rope1 did, it turns each pair of features of the queries and the keys
    by an angle that grows with the position. Llama does this inside every
    attention layer, head by head.

    ```python
    q = torch.ones(1, 2, 3, 4)    # (batch, heads, steps, head size)
    rope(q).shape                 # torch.Size([1, 2, 3, 4]), each head turned the same way
    ```

    The values aren't turned. The position should change where a query looks,
    which the scores decide, and not what it brings back. LLaMA's code turns
    adjacent pairs, features 0 and 1, then 2 and 3, with a base of 10,000.
    """,
)

LLAMA1_SOURCE = source_file(
    '''
    # llama1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    heads = torch.zeros(2, 5, 3, 4)                   # (batch, steps, heads, head size)
    by_head = heads.transpose(1, 2).shape             # torch.Size([2, 3, 5, 4])


''' + indented(ROPE_PART) + '''


    # TODO: Return how many weights a model's positions take, for kind
    #   "learned", a table of one row of size numbers per position up to
    #   context, or "rope".
    def position_weights(
        kind: str,
        context: int,
        size: int,
    ) -> int:
        ...


    # TODO: Return how many weights a norm of the given size holds, for kind
    #   "layernorm" or "rmsnorm".
    def norm_weights(
        kind: str,
        size: int,
    ) -> int:
        ...


    # TODO: Finish forward, causal attention as Llama does it. Project x, of
    #   shape (batch, steps, size), with wq, wk and wv, and split each into
    #   heads, shape (batch, heads, steps, size // heads). Turn the queries and
    #   the keys with rope, not the values, then F.scaled_dot_product_attention
    #   with is_causal=True. Join the heads back and project with wo.
''' + indented(LLAMA_ATTENTION_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

LLAMA1_SOLUTION = source_file(
    '''
    # llama1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    heads = torch.zeros(2, 5, 3, 4)
    by_head = heads.transpose(1, 2).shape


''' + indented(ROPE_PART) + '''


    def position_weights(
        kind: str,
        context: int,
        size: int,
    ) -> int:
        if kind == "learned":
            return context * size

        return 0


    def norm_weights(
        kind: str,
        size: int,
    ) -> int:
        if kind == "layernorm":
            return 2 * size

        return size


''' + indented(LLAMA_ATTENTION_INIT) + '''

''' + indented(LLAMA_ATTENTION_FORWARD) + '''
    ''',
)


def llama1_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(ROPE_PART + LLAMA_ATTENTION_INIT + LLAMA_ATTENTION_FORWARD, namespace)

    return namespace


LLAMA1_REFERENCE = llama1_namespace()


def seeded_attention(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make(8, 2)


def check_llama_attention(found: type) -> None:
    model = expect_module("LlamaAttention(8, 2)", seeded_attention(found))
    x = small_tensor(550, 2, 5, 8) / 3

    with torch.no_grad():
        expected = seeded_attention(LLAMA1_REFERENCE["LlamaAttention"])(x)
        actual = model(x)

    expect_tensor(
        call="LlamaAttention(8, 2) on 2 sequences of 5 steps",
        actual=actual,
        expected=expected,
    )


def check_llama1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="position_weights",
        description="counts a learned table's weights, and RoPE's none",
        inputs=(
            ("learned", 1024, 768),
            ("rope", 4096, 4096),
            ("learned", 64, 32),
        ),
        reference=lambda kind, context, size: context * size if kind == "learned" else 0,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="norm_weights",
        description="counts a scale and a bias for LayerNorm, a scale for RMSNorm",
        inputs=(
            ("layernorm", 768),
            ("rmsnorm", 4096),
        ),
        reference=lambda kind, size: 2 * size if kind == "layernorm" else size,
        expect=expect_value,
    )
    checker.test_method(
        class_name="LlamaAttention",
        method="forward",
        description="turns the queries and keys with RoPE, then attends causally",
        run=check_llama_attention,
    )


LLAMA1 = Exercise(
    name="llama1",
    chapter=MODERN_LLMS,
    title="From GPT-2 to Llama",
    lesson=LLAMA1_LESSON,
    source=LLAMA1_SOURCE,
    solution=LLAMA1_SOLUTION,
    hints={
        "position_weights": "A learned table has context rows of size numbers. RoPE computes its angles and has "
        "no weights.",
        "norm_weights": "LayerNorm learns a scale and a bias per feature. RMSNorm learns only the scale.",
        "LlamaAttention.forward": "view each projection as (batch, steps, self.heads, size // self.heads) and "
        "transpose(1, 2). After attention, transpose(1, 2) back and reshape to (batch, steps, size).",
    },
    introduces=(),
    check=check_llama1,
)


# ---------------------------------------------------------------------------
# llama2
# ---------------------------------------------------------------------------

SWIGLU_INIT = '''
class SwiGLU(nn.Module):
    def __init__(
        self,
        size: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, size, bias=False)
        self.w3 = nn.Linear(size, hidden, bias=False)
'''

SWIGLU_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
'''

LLAMA2_LESSON = prose(
    """
    GPT-2's MLP widens each position four times, applies GELU, and narrows it
    back, two weight matrices. Shazeer's 2020 paper tried gated versions with
    three matrices. The one Llama took is SwiGLU.

    ```text
    swiglu(x) = (silu(x W) * (x V)) W2
    ```

    `silu(z)` is `z * sigmoid(z)`, also called Swish. One branch says how much
    of each hidden feature gets through, and the other carries it. This is
    gating1's gate, a weight computed from the input, inside every MLP.

    ```python
    F.silu(torch.tensor([-2.0, 0.0, 2.0]))    # tensor([-0.2384,  0.0000,  1.7616])
    ```

    The paper trained the same T5 model with each MLP. It measured the cross
    entropy loss on held out text, which it calls log perplexity, where lower
    is better.

    | MLP | After 65,536 steps | After 524,288 steps |
    |---|---|---|
    | ReLU | 1.997 | 1.677 |
    | GELU | 1.983 | 1.679 |
    | SwiGLU | 1.944 | 1.636 |

    Three matrices hold more weights than two at the same width, so the
    hidden width shrinks to keep the count equal. Two matrices of `d` by `4d`
    hold `8d²` weights, and three of `d` by `h` hold `3dh`. They match at `h =
    8d / 3`, two thirds of `4d`. At `d = 768` the paper went from 3,072 to
    2,048.

    The LLaMA paper uses the same two thirds of `4d`. Its code then rounds up
    to a multiple of 256 so the matrices suit the hardware. That gives 11,008
    for the 7B model's `d` of 4,096, and 13,824 for the 13B model's 5,120.
    """,
)

LLAMA2_SOURCE = source_file(
    '''
    # llama2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    swished = F.silu(torch.tensor([-2.0, 0.0, 2.0]))    # tensor([-0.2384,  0.0000,  1.7616])
    rounded_up = 256 * ((1000 + 255) // 256)            # 1024, the next multiple of 256


    # TODO: Return SwiGLU of x, of shape (..., size), with W and V of shape
    #   (size, hidden) and W2 of shape (hidden, size).
    def swiglu(
        x: Tensor,
        W: Tensor,
        V: Tensor,
        W2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward, as LLaMA's code names the parts. w1 is the gate's
    #   branch, which goes through silu, w3 is the other branch, and w2 brings
    #   their product back down to size.
''' + indented(SWIGLU_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return LLaMA's hidden width for a model of size, two thirds of four
    #   times size, made a whole number with int(), then rounded up to a
    #   multiple of multiple_of.
    def swiglu_hidden(
        size: int,
        multiple_of: int,
    ) -> int:
        ...


    # TODO: Return how many weights an MLP of size and hidden holds without
    #   biases, for kind "gelu", two matrices, or "swiglu", three.
    def mlp_weights(
        kind: str,
        size: int,
        hidden: int,
    ) -> int:
        ...
    ''',
)

LLAMA2_SOLUTION = source_file(
    '''
    # llama2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    swished = F.silu(torch.tensor([-2.0, 0.0, 2.0]))
    rounded_up = 256 * ((1000 + 255) // 256)


    def swiglu(
        x: Tensor,
        W: Tensor,
        V: Tensor,
        W2: Tensor,
    ) -> Tensor:
        return (F.silu(x @ W) * (x @ V)) @ W2


''' + indented(SWIGLU_INIT) + '''

''' + indented(SWIGLU_FORWARD) + '''


    def swiglu_hidden(
        size: int,
        multiple_of: int,
    ) -> int:
        hidden = int(2 * 4 * size / 3)

        return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


    def mlp_weights(
        kind: str,
        size: int,
        hidden: int,
    ) -> int:
        if kind == "swiglu":
            return 3 * size * hidden

        return 2 * size * hidden
    ''',
)


def reference_swiglu_hidden(
    size: int,
    multiple_of: int,
) -> int:
    hidden = int(2 * 4 * size / 3)

    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def swiglu_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(SWIGLU_INIT + SWIGLU_FORWARD, namespace)

    return namespace


SWIGLU_CLASS = swiglu_namespace()["SwiGLU"]


def seeded_swiglu(make: type) -> nn.Module:
    torch.manual_seed(0)

    return make(6, 16)


def check_swiglu_module(found: type) -> None:
    model = expect_module("SwiGLU(6, 16)", seeded_swiglu(found))
    x = small_tensor(551, 2, 3, 6) / 3

    with torch.no_grad():
        expected = seeded_swiglu(SWIGLU_CLASS)(x)
        actual = model(x)

    expect_tensor(
        call="SwiGLU(6, 16) on 2 sequences of 3 steps",
        actual=actual,
        expected=expected,
    )


def check_llama2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="swiglu",
        description="gates one branch with silu of the other",
        inputs=(
            (small_tensor(552, 2, 4) / 2, small_tensor(553, 4, 6) / 2, small_tensor(554, 4, 6) / 2, small_tensor(555, 6, 4) / 2),
            (small_tensor(556, 3, 2, 3) / 2, small_tensor(557, 3, 5) / 2, small_tensor(558, 3, 5) / 2, small_tensor(559, 5, 3) / 2),
        ),
        reference=lambda x, W, V, W2: (F.silu(x @ W) * (x @ V)) @ W2,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="SwiGLU",
        method="forward",
        description="runs w2 on silu of w1 times w3",
        run=check_swiglu_module,
    )
    check_on_inputs(
        checker=checker,
        function="swiglu_hidden",
        description="gives two thirds of four times the size, rounded up",
        inputs=(
            (4096, 256),
            (6656, 256),
            (768, 64),
            (100, 8),
        ),
        reference=reference_swiglu_hidden,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="mlp_weights",
        description="counts two matrices for GELU and three for SwiGLU",
        inputs=(
            ("gelu", 768, 3072),
            ("swiglu", 768, 2048),
            ("swiglu", 4096, 11008),
        ),
        reference=lambda kind, size, hidden: (3 if kind == "swiglu" else 2) * size * hidden,
        expect=expect_value,
    )


LLAMA2 = Exercise(
    name="llama2",
    chapter=MODERN_LLMS,
    title="The SwiGLU MLP",
    lesson=LLAMA2_LESSON,
    source=LLAMA2_SOURCE,
    solution=LLAMA2_SOLUTION,
    hints={
        "swiglu": "F.silu of x @ W, times x @ V, then @ W2.",
        "SwiGLU.forward": "self.w2 of F.silu(self.w1(x)) times self.w3(x).",
        "swiglu_hidden": "Start from int(2 * 4 * size / 3). Rounding n up to a multiple of m is "
        "m * ((n + m - 1) // m).",
        "mlp_weights": "GELU's two matrices are size by hidden and hidden by size. SwiGLU adds a third of size by "
        "hidden.",
    },
    introduces=(),
    check=check_llama2,
)


# ---------------------------------------------------------------------------
# llama3
# ---------------------------------------------------------------------------

LLAMA3_LESSON = prose(
    """
    Here are the two blocks, weight by weight, for a model of size `d`.

    | Part | GPT-2 block | Llama block |
    |---|---|---|
    | first norm | `2d` | `d` |
    | query, key and value | `3d² + 3d` | `3d²` |
    | attention output | `d² + d` | `d²` |
    | second norm | `2d` | `d` |
    | MLP | `8d² + 5d` | `3dh` |

    That makes `12d² + 13d` for GPT-2, or 7,087,872 at GPT-2 small's 768.
    Twelve blocks, the embeddings and a last norm give its 124,439,808
    weights. A Llama block makes `4d² + 3dh + 2d`, or 202,383,360 for the 7B
    model, with `d` of 4,096 and `h` of 11,008. 32 blocks, two tables of
    32,000 tokens and a last norm give 6,738,415,616.

    In both, the MLP holds two thirds of the block's weights and attention one
    third. The width of llama2 keeps that split as it was.

    Recent models add one more piece, QK norm. Scaling vision transformers
    toward 22 billion weights, Dehghani and colleagues saw a model of 8
    billion diverge in training. Its
    attention scores grew past 50,000, and a softmax of scores that large puts
    all its weight on one key. Normalizing the queries and the keys before the
    dot product stopped it. OLMo 2 and Qwen3 use RMSNorm there for the same
    reason.

    The table below scales 16 random queries and 16 keys of size 64, as
    weights that grow during training would. The largest weight is averaged
    over the 16 queries.

    | Scale | Largest score | Largest weight | With QK norm |
    |---|---|---|---|
    | 1 | 3.27 | 0.248 | 3.15, 0.249 |
    | 4 | 52.37 | 0.917 | 3.15, 0.249 |
    | 16 | 837.9 | 0.998 | 3.15, 0.249 |

    After an RMSNorm every query and key has the same size, so no scale can
    blow the scores up. The norms' own learned weights can still sharpen the
    softmax, but slowly, through training.
    """,
)

QK_PARTS = '''
def qk_norm(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)

    return q * q_weight, k * k_weight


def top_weight(
    q: Tensor,
    k: Tensor,
) -> float:
    weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)

    return weights.max(dim=-1).values.mean().item()
'''

LLAMA3_SOURCE = source_file(
    '''
    # llama3.py

    import math

    import torch
    from torch import Tensor

    block = 12 * 768**2 + 13 * 768                     # 7087872, one GPT-2 small block
    rows = torch.tensor([[3.0, 4.0]])
    rms = rows.pow(2).mean(dim=-1, keepdim=True).sqrt()    # tensor([[3.5355]])


    # TODO: Return how many weights a GPT-2 block of size holds, with the
    #   biases and both LayerNorms, from the table in the lesson.
    def gpt2_block_weights(size: int) -> int:
        ...


    # TODO: Return how many weights a Llama block of size and MLP width hidden
    #   holds, from the table in the lesson.
    def llama_block_weights(
        size: int,
        hidden: int,
    ) -> int:
        ...


    # TODO: Return q and k after QK norm. RMSNorm each row of the last dimension,
    #   with eps inside the root, then scale q by q_weight and k by k_weight.
    def qk_norm(
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the largest softmax weight of each query over the keys,
    #   averaged over the queries, as a plain number. The scores are
    #   q @ k^T over the square root of the size.
    def top_weight(
        q: Tensor,
        k: Tensor,
    ) -> float:
        ...
    ''',
)

LLAMA3_SOLUTION = source_file(
    '''
    # llama3.py, a solution for comparison

    import math

    import torch
    from torch import Tensor

    block = 12 * 768**2 + 13 * 768
    rows = torch.tensor([[3.0, 4.0]])
    rms = rows.pow(2).mean(dim=-1, keepdim=True).sqrt()


    def gpt2_block_weights(size: int) -> int:
        return 12 * size * size + 13 * size


    def llama_block_weights(
        size: int,
        hidden: int,
    ) -> int:
        return 4 * size * size + 3 * size * hidden + 2 * size


''' + indented(QK_PARTS) + '''
    ''',
)


def qk_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "Tensor": Tensor,
    }
    exec(QK_PARTS, namespace)

    return namespace


QK_REFERENCE = qk_namespace()


def measure_qk_table() -> dict[int, tuple[float, float, float, float]]:
    """llama3's table. For each scale of 16 random queries and 16 keys of size 64, from generators 566 and 567, the
    largest score and the largest softmax weight averaged over the queries, without and with QK norm."""
    queries = torch.randn(16, 64, generator=torch.Generator().manual_seed(566))
    keys = torch.randn(16, 64, generator=torch.Generator().manual_seed(567))
    results: dict[int, tuple[float, float, float, float]] = {}

    for scale in (1, 4, 16):
        q, k = queries * scale, keys * scale
        normed_q, normed_k = QK_REFERENCE["qk_norm"](q, k, torch.ones(64), torch.ones(64))
        results[scale] = (
            (q @ k.T / math.sqrt(64)).max().item(),
            QK_REFERENCE["top_weight"](q, k),
            (normed_q @ normed_k.T / math.sqrt(64)).max().item(),
            QK_REFERENCE["top_weight"](normed_q, normed_k),
        )

    return results


def gpt2_block_by_modules(size: int) -> int:
    """GPT-2's block built from PyTorch's own layers, so the count is held to real modules."""
    parts = [
        nn.LayerNorm(size),
        nn.Linear(size, 3 * size),
        nn.Linear(size, size),
        nn.LayerNorm(size),
        nn.Linear(size, 4 * size),
        nn.Linear(4 * size, size),
    ]

    return sum(parameter.numel() for part in parts for parameter in part.parameters())


def llama_block_by_modules(
    size: int,
    hidden: int,
) -> int:
    parts = [
        nn.RMSNorm(size),
        nn.Linear(size, size, bias=False),
        nn.Linear(size, size, bias=False),
        nn.Linear(size, size, bias=False),
        nn.Linear(size, size, bias=False),
        nn.RMSNorm(size),
        nn.Linear(size, hidden, bias=False),
        nn.Linear(hidden, size, bias=False),
        nn.Linear(size, hidden, bias=False),
    ]

    return sum(parameter.numel() for part in parts for parameter in part.parameters())


def expect_qk_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the normed q and k as two tensors in a tuple.")

    for name, part, wanted in zip(("q", "k"), actual, expected):
        expect_tensor(
            call=f"the {name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_llama3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gpt2_block_weights",
        description="counts a GPT-2 block, biases and LayerNorms included",
        inputs=(
            (768,),
            (64,),
            (1600,),
        ),
        reference=gpt2_block_by_modules,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="llama_block_weights",
        description="counts a Llama block, with no biases",
        inputs=(
            (64, 176),
            (96, 256),
        ),
        reference=llama_block_by_modules,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="qk_norm",
        description="norms each query and key row, then scales it",
        inputs=(
            (small_tensor(560, 3, 4), small_tensor(561, 5, 4), torch.tensor([1.0, 0.5, 2.0, 1.0]), torch.tensor([0.5, 1.0, 1.0, 3.0])),
            (small_tensor(562, 2, 2, 6) * 4, small_tensor(563, 2, 3, 6), torch.ones(6), torch.ones(6), 1e-3),
        ),
        reference=QK_REFERENCE["qk_norm"],
        expect=expect_qk_pair,
    )
    check_on_inputs(
        checker=checker,
        function="top_weight",
        description="averages each query's largest softmax weight",
        inputs=(
            (small_tensor(564, 4, 8) / 2, small_tensor(565, 6, 8) / 2),
            (Described(torch.randn(16, 64, generator=torch.Generator().manual_seed(566)) * 4, "16 random queries of size 64, times 4"), torch.randn(16, 64, generator=torch.Generator().manual_seed(567)) * 4),
        ),
        reference=QK_REFERENCE["top_weight"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


LLAMA3 = Exercise(
    name="llama3",
    chapter=MODERN_LLMS,
    title="A Llama block next to a GPT-2 block",
    lesson=LLAMA3_LESSON,
    source=LLAMA3_SOURCE,
    solution=LLAMA3_SOLUTION,
    hints={
        "gpt2_block_weights": "Add up the GPT-2 column of the table, which comes to 12 size squared plus 13 size.",
        "llama_block_weights": "Four size by size matrices for attention, three matrices of size by hidden for "
        "the MLP, and two norms of size each.",
        "qk_norm": "Divide each row by the root of the mean of its squares plus eps, which torch.rsqrt gives "
        "as a factor to multiply by. Then multiply by the weights.",
        "top_weight": "Softmax over the last dimension, take .max(dim=-1).values, average them, and call .item().",
    },
    introduces=(),
    check=check_llama3,
)
