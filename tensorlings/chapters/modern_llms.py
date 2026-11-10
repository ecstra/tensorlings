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

        Llama keeps the overall design of GPT-2 and changes a few parts. Each
        change was tested in an earlier paper. This chapter adds the changes
        one at a time. First come RMSNorm and RoPE, then the SwiGLU MLP and
        its width. Last, it counts the weights of a Llama block and a GPT-2
        block side by side, and adds the QK norm that recent models use.

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
# RoPE from rope1. It rotates adjacent pairs of features, and the positions
# 0, 1, 2 and so on run along dim -2.
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
    GPT-2 came out in 2019 and Llama in 2023. Llama keeps GPT-2's overall
    design, a stack of pre norm blocks. Each block has an attention layer and
    an MLP, and each of those has a residual connection. Llama changes a few
    parts, and each change comes from an earlier paper that tested it.

    | Part | GPT-2 | Llama |
    |---|---|---|
    | norm | LayerNorm before each sublayer | RMSNorm before each sublayer |
    | positions | a learned table added to the token embeddings | RoPE on the queries and keys of every layer |
    | MLP | GELU, 4 times as wide | SwiGLU, about 8/3 times as wide, which llama2 builds |
    | biases | in every Linear and norm | none |
    | output layer | shares the token embedding's weights | its own weights |

    RMSNorm, from rmsnorm1, does not subtract the mean and has no bias. Its
    paper reports quality close to LayerNorm's, with 7% to 64% less running
    time. A LayerNorm of size `d` holds `2d` weights, a scale and a bias for
    each feature. An RMSNorm of size `d` holds only the `d` scales.

    GPT-2 small learns a position table of 1,024 rows of 768 numbers, which
    is 786,432 weights. A position past 1,023 has no row at all. RoPE has no
    weights. As in rope1, it rotates each pair of features in the queries and
    the keys by an angle that grows with the position. Llama applies RoPE
    inside every attention layer, to each head separately.

    ```python
    q = torch.ones(1, 2, 3, 4)    # (batch, heads, steps, head size)
    rope(q).shape                 # torch.Size([1, 2, 3, 4]), each head rotated the same way
    ```

    The values are not rotated. The position should change which keys a
    query attends to, and the scores decide that. The position should not
    change the content that the values carry back. LLaMA's code rotates
    adjacent pairs, features 0 and 1, then features 2 and 3, with a base of
    10,000.
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


    # TODO: Return how many weights a model uses to encode positions. kind is
    #   "learned" or "rope". A learned table has one row of size numbers for
    #   each of the context positions.
    #   Example: position_weights("learned", 1024, 768) returns 786432.
    def position_weights(
        kind: str,
        context: int,
        size: int,
    ) -> int:
        ...


    # TODO: Return how many weights a norm layer of the given size holds. kind
    #   is "layernorm" or "rmsnorm".
    #   Example: norm_weights("layernorm", 768) returns 1536.
    def norm_weights(
        kind: str,
        size: int,
    ) -> int:
        ...


    # TODO: Finish forward so it computes causal attention the way Llama does.
    #   x has shape (batch, steps, size). Project x with wq, wk and wv to get
    #   the queries, keys and values. Split each one into heads, with shape
    #   (batch, heads, steps, size // heads). Apply rope to the queries and the
    #   keys but not to the values. Then call F.scaled_dot_product_attention
    #   with is_causal=True. Join the heads back into shape
    #   (batch, steps, size) and project the result with wo.
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
        description="counts the weights of a learned table, and zero for RoPE",
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
        description="counts a scale and a bias per feature for LayerNorm, and only a scale for RMSNorm",
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
        description="rotates the queries and keys with RoPE, then runs causal attention",
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
        "position_weights": "A learned table has context rows, and each row holds size numbers. RoPE computes its "
        "angles from the position, so it has no weights.",
        "norm_weights": "LayerNorm learns a scale and a bias per feature. RMSNorm learns only the scale.",
        "LlamaAttention.forward": "Use view to give each projection the shape (batch, steps, self.heads, size // "
        "self.heads), then call transpose(1, 2). After attention, transpose(1, 2) back and reshape to "
        "(batch, steps, size).",
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
    GPT-2's MLP uses two weight matrices. The first widens the vector at each
    position to four times its size. Then GELU is applied, and the second
    matrix narrows the vector back. Shazeer's 2020 paper tried gated versions
    that use three matrices. The one Llama adopted is called SwiGLU.

    ```text
    swiglu(x) = (silu(x W) * (x V)) W2
    ```

    `silu(z)` is `z * sigmoid(z)`, and it is also called Swish. The formula
    multiplies two branches entry by entry, `silu(x W)` and `x V`. The first
    branch decides how much of each hidden feature gets through, and the
    second branch carries the feature's value. This is the gate from gating1,
    a weight computed from the input, now placed inside every MLP.

    ```python
    F.silu(torch.tensor([-2.0, 0.0, 2.0]))    # tensor([-0.2384,  0.0000,  1.7616])
    ```

    The paper trained the same T5 model with each kind of MLP. It measured the
    cross entropy loss on held out text, which it calls log perplexity. Lower
    is better.

    | MLP | After 65,536 steps | After 524,288 steps |
    |---|---|---|
    | ReLU | 1.997 | 1.677 |
    | GELU | 1.983 | 1.679 |
    | SwiGLU | 1.944 | 1.636 |

    At the same width, three matrices hold more weights than two. So SwiGLU
    uses a smaller hidden width to keep the total number of weights the same.
    Two matrices of `d` by `4d` hold `8d²` weights, and three matrices of `d`
    by `h` hold `3dh`. These are equal when `h = 8d / 3`, which is two thirds
    of `4d`. For `d = 768`, the paper reduced the hidden width from 3,072 to
    2,048.

    The LLaMA paper also uses two thirds of `4d`. Its code then rounds the
    width up to a multiple of 256, because matrices with sizes like that run
    well on the hardware. That gives a hidden width of 11,008 for the 7B
    model, whose `d` is 4,096. The 13B model has a `d` of 5,120 and a hidden
    width of 13,824.
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


    # TODO: Return swiglu(x) from the formula in the lesson. x has shape
    #   (..., size), W and V have shape (size, hidden), and W2 has shape
    #   (hidden, size). The result has the same shape as x.
    def swiglu(
        x: Tensor,
        W: Tensor,
        V: Tensor,
        W2: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. The layers are named as in LLaMA's code. w1 is the
    #   gate branch, whose output goes through silu. w3 is the other branch.
    #   Multiply the two branches entry by entry, then w2 maps that product
    #   back down to size.
''' + indented(SWIGLU_INIT) + '''

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return LLaMA's hidden width for a model of the given size. Compute
    #   two thirds of 4 * size, turn it into a whole number with int(), then
    #   round it up to the next multiple of multiple_of.
    #   Example: swiglu_hidden(4096, 256) returns 11008.
    def swiglu_hidden(
        size: int,
        multiple_of: int,
    ) -> int:
        ...


    # TODO: Return how many weights an MLP holds, with no biases. size is the
    #   model width and hidden is the hidden width. kind is "gelu", which uses
    #   two matrices, or "swiglu", which uses three.
    #   Example: mlp_weights("gelu", 768, 3072) returns 4718592.
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
        description="multiplies silu(x @ W) by x @ V, then applies W2",
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
        description="applies w2 to silu(w1(x)) * w3(x)",
        run=check_swiglu_module,
    )
    check_on_inputs(
        checker=checker,
        function="swiglu_hidden",
        description="returns two thirds of 4 * size, rounded up to a multiple",
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
        "swiglu": "Apply F.silu to x @ W, multiply that by x @ V, then multiply the product by W2 with @.",
        "SwiGLU.forward": "Multiply F.silu(self.w1(x)) by self.w3(x), then pass the product through self.w2.",
        "swiglu_hidden": "Start from int(2 * 4 * size / 3). Rounding n up to a multiple of m is "
        "m * ((n + m - 1) // m).",
        "mlp_weights": "GELU's two matrices are size by hidden and hidden by size. SwiGLU adds a third matrix of "
        "size by hidden.",
    },
    introduces=(),
    check=check_llama2,
)


# ---------------------------------------------------------------------------
# llama3
# ---------------------------------------------------------------------------

LLAMA3_LESSON = prose(
    """
    Here are the weight counts of the two blocks, part by part, for a model of
    size `d`. In the Llama column, `h` is the hidden width of the SwiGLU MLP.

    | Part | GPT-2 block | Llama block |
    |---|---|---|
    | first norm | `2d` | `d` |
    | query, key and value | `3d² + 3d` | `3d²` |
    | attention output | `d² + d` | `d²` |
    | second norm | `2d` | `d` |
    | MLP | `8d² + 5d` | `3dh` |

    The GPT-2 column adds up to `12d² + 13d`. For GPT-2 small, where `d` is
    768, that is 7,087,872 weights per block. Twelve blocks, plus the
    embeddings and a final norm, give GPT-2 small's total of 124,439,808. The
    Llama column adds up to `4d² + 3dh + 2d`. For the 7B model, with `d` of
    4,096 and `h` of 11,008, that is 202,383,360 per block. 32 blocks, plus a
    final norm and two tables of 32,000 rows for the token embedding and the
    output layer, give 6,738,415,616.

    In both blocks, the MLP holds about two thirds of the weights and
    attention holds about one third. The hidden width from llama2 is what
    keeps this split the same.

    Recent models add one more piece, called QK norm. Dehghani and colleagues
    scaled vision transformers up to 22 billion weights. Along the way, a
    model with 8 billion weights diverged in training, which means its loss
    blew up instead of going down. Its attention scores grew past 50,000. A
    softmax of scores that large puts almost all of its weight on one key.
    Normalizing the queries and the keys before the dot product fixed the
    problem. OLMo 2 and Qwen3 apply RMSNorm to the queries and keys for the
    same reason.

    The table below takes 16 random queries and 16 random keys of size 64 and
    multiplies them by a scale. Weights that grow during training scale the
    queries and keys up in the same way. The largest score is taken over all
    query and key pairs. The largest weight is each query's biggest softmax
    weight, averaged over the 16 queries. The last column repeats both numbers
    with QK norm.

    | Scale | Largest score | Largest weight | With QK norm |
    |---|---|---|---|
    | 1 | 3.27 | 0.248 | 3.15, 0.249 |
    | 4 | 52.37 | 0.917 | 3.15, 0.249 |
    | 16 | 837.9 | 0.998 | 3.15, 0.249 |

    After an RMSNorm, every query and key has a root mean square of 1,
    whatever scale it started with. So scaling cannot make the scores grow.
    The learned weights of the two norms can still make the softmax sharper,
    but only slowly, as training changes them.
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


    # TODO: Return how many weights a GPT-2 block of the given size holds,
    #   counting the biases and both LayerNorms. Use the table in the lesson.
    #   Example: gpt2_block_weights(768) returns 7087872.
    def gpt2_block_weights(size: int) -> int:
        ...


    # TODO: Return how many weights a Llama block holds, for a model of the
    #   given size and an MLP of hidden width hidden. Use the table in the
    #   lesson.
    #   Example: llama_block_weights(4096, 11008) returns 202383360.
    def llama_block_weights(
        size: int,
        hidden: int,
    ) -> int:
        ...


    # TODO: Return the pair (q, k) after QK norm. Apply RMSNorm along the last
    #   dimension of both, which divides each row by the square root of the
    #   mean of its squares plus eps. Then multiply q by q_weight and k by
    #   k_weight.
    def qk_norm(
        q: Tensor,
        k: Tensor,
        q_weight: Tensor,
        k_weight: Tensor,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Compute the attention scores q @ k^T / sqrt(size), where size is
    #   the last dimension of q, and take the softmax of each query's scores
    #   over the keys. Return each query's largest softmax weight, averaged
    #   over the queries, as a Python float.
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
        raise Mismatch(f"{call} should return the normalized q and k as a tuple of two tensors.")

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
        description="applies RMSNorm to each row of q and k, then multiplies by the weights",
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
        "gpt2_block_weights": "Add up the GPT-2 column of the table. The total is 12 times size squared, plus 13 "
        "times size.",
        "llama_block_weights": "Attention has four matrices of size by size. The MLP has three matrices of size by "
        "hidden. The two norms hold size weights each.",
        "qk_norm": "Divide each row by the square root of the mean of its squares plus eps. torch.rsqrt gives one "
        "over the square root, so you can multiply by it instead. Then multiply by the weights.",
        "top_weight": "Softmax over the last dimension, take .max(dim=-1).values, average them, and call .item().",
    },
    introduces=(),
    check=check_llama3,
)
