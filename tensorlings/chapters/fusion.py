"""100_fusion, how two kinds of input meet in one model: early fusion in one sequence against late fusion of separate summaries, gated cross attention that starts closed, and training with inputs dropped so a missing one doesn't break the model."""

from __future__ import annotations

import contextlib
import copy
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
    seeded,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .transformer_block import SELF_ATTENTION

FUSION = Chapter(
    folder="100_fusion",
    title="Fusion",
    readme=source_file(
        """
        # Fusion

        A model that reads two kinds of input has to decide where they meet.
        Early fusion puts both in one sequence, so attention mixes them from
        the first layer. Late fusion sums each up on its own and joins the
        summaries. This chapter measures when each wins, joins a new input to
        a trained model through gated cross attention that starts closed, as
        Flamingo does, and trains a model to keep working when an input is
        missing.

        ## Further reading

        - [Alayrac and colleagues, Flamingo](https://arxiv.org/abs/2204.14198)
        - [Baltrušaitis and colleagues, a survey of multimodal machine learning](https://arxiv.org/abs/1705.09406)
        - [Neverova and colleagues, ModDrop](https://arxiv.org/abs/1501.00102)
        """,
    ),
)


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count moves the last digits of a run, so the measurements pin it."""
    before = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(before)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "copy": copy,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


def seeded_model(
    make: Callable[..., nn.Module],
    seed: int,
    *arguments: object,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(*arguments)


def recording_hook(
    model: nn.Module,
    seen: list[bool],
) -> torch.utils.hooks.RemovableHandle:
    """Notes on every call whether autograd was recording, since a model's outputs can't show it."""
    return model.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))


# ---------------------------------------------------------------------------
# fuse1
# ---------------------------------------------------------------------------

LENGTH = 32
WIDTH = 16

MIX_BLOCK_PART = '''
# block1's block with the causal mask taken off, so every position sees every other.
class MixBlock(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(size)
        self.c_attn = nn.Linear(size, 3 * size)
        self.c_proj = nn.Linear(size, size)
        self.ln_2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(nn.Linear(size, 4 * size), nn.GELU(), nn.Linear(4 * size, size))

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, size = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(size, dim=2)
        q, k, v = (part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2) for part in (q, k, v))
        mixed = F.scaled_dot_product_attention(q, k, v)
        x = x + self.c_proj(mixed.transpose(1, 2).reshape(batch, steps, size))

        return x + self.mlp(self.ln_2(x))
'''

SERIES_TOKENS_SOLUTION = '''
def series_tokens(
    series: Tensor,
    value: nn.Linear,
    positions: nn.Embedding,
) -> Tensor:
    return value(series.unsqueeze(-1)) + positions.weight[: series.shape[1]]
'''

LATE_INIT = '''
class LateFusion(nn.Module):
    def __init__(
        self,
        length: int = 32,
        size: int = 16,
    ) -> None:
        super().__init__()
        self.value = nn.Linear(1, size)
        self.positions = nn.Embedding(length, size)
        self.series_mlp = nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size))
        self.question = nn.Embedding(length, size)
        self.head = nn.Sequential(nn.Linear(2 * size, 4 * size), nn.ReLU(), nn.Linear(4 * size, 1))
'''

LATE_FORWARD = '''
    def forward(
        self,
        series: Tensor,
        question: Tensor,
    ) -> Tensor:
        summary = self.series_mlp(series_tokens(series, self.value, self.positions)).mean(dim=1)
        joined = torch.cat([summary, self.question(question)], dim=-1)

        return self.head(joined).squeeze(-1)
'''

EARLY_INIT = '''
class EarlyFusion(nn.Module):
    def __init__(
        self,
        length: int = 32,
        size: int = 16,
    ) -> None:
        super().__init__()
        self.value = nn.Linear(1, size)
        self.positions = nn.Embedding(length, size)
        self.question = nn.Embedding(length, size)
        self.kinds = nn.Embedding(2, size)
        self.block = MixBlock(size, 2)
        self.ln_f = nn.LayerNorm(size)
        self.head = nn.Linear(size, 1)
'''

EARLY_FORWARD = '''
    def forward(
        self,
        series: Tensor,
        question: Tensor,
    ) -> Tensor:
        tokens = series_tokens(series, self.value, self.positions) + self.kinds.weight[0]
        asked = (self.question(question) + self.kinds.weight[1]).unsqueeze(1)
        mixed = self.block(torch.cat([tokens, asked], dim=1))

        return self.head(self.ln_f(mixed[:, -1])).squeeze(-1)
'''

FUSE1_REFERENCE = namespace_of(MIX_BLOCK_PART + SERIES_TOKENS_SOLUTION + LATE_INIT + LATE_FORWARD + EARLY_INIT + EARLY_FORWARD)


def lookup_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Series of 32 numbers, a question naming one position, and the number at that position."""
    series = torch.rand(count, LENGTH, generator=generator) * 2 - 1
    question = torch.randint(0, LENGTH, (count,), generator=generator)

    return series, question, series.gather(1, question.unsqueeze(1)).squeeze(1)


def sum_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """The same inputs, with an answer of 3 times the series' mean plus an offset the question names."""
    series = torch.rand(count, LENGTH, generator=generator) * 2 - 1
    question = torch.randint(0, LENGTH, (count,), generator=generator)
    offsets = torch.linspace(-1, 1, LENGTH)

    return series, question, series.mean(dim=1) * 3 + offsets[question]


FUSE1_LESSON = prose(
    """
    A model with two kinds of input must decide where they meet. Here one
    input is a series of 32 numbers and the other a question, one token.

    - **late fusion**: each input is summed up on its own, and the two summaries are joined at the end.
    - **early fusion**: both become tokens in one sequence, so attention mixes them from the first layer.

    Late fusion sums up the series before it knows the question. Early fusion
    lets the question pick what to read. In the one sequence, a learned
    **type embedding** per kind of input marks each token's kind. BERT's
    segment embeddings told two sentences apart the same way.

    Two tasks, each model trained 1,500 steps on three seeds, both 16 wide.

    - lookup: the question names a position, and the answer is the series' number there.
    - sum: the answer is 3 times the series' mean, plus an offset the question names.

    | Task | Late fusion, 3,777 weights | Early fusion, 4,417 weights | Guessing the mean |
    |---|---|---|---|
    | lookup | 0.315 to 0.320 | 0.0003 to 0.011 | 0.33 |
    | sum | 0.0001 to 0.0002 | 0.0002 | 0.44 |

    A summary of 16 numbers can't hold 32, so late fusion has lost the asked
    number before the question arrives. In the sum, each input brings its
    own piece of the answer, and the summaries hold everything needed.

    So fuse early when one input decides what to read in the other. Fuse late
    when each brings its own evidence, as CLIP's two encoders do. Late fusion
    runs each encoder once per input, and its summaries can be stored.
    """,
)

FUSE1_SOURCE = source_file(
    '''
    # fuse1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    series = torch.tensor([[0.5, -1.0, 2.0]])
    series.unsqueeze(-1).shape    # torch.Size([1, 3, 1]), one number per position, ready for a Linear(1, size)
    joined = torch.cat([torch.zeros(2, 3, 4), torch.ones(2, 1, 4)], dim=1)    # shape (2, 4, 4), a token added at the end


''' + indented(MIX_BLOCK_PART) + '''


    # TODO: Return one token per number of series, of shape (batch, steps).
    #   Each token is value of the number plus the row of positions for its
    #   place, giving shape (batch, steps, size).
    def series_tokens(
        series: Tensor,
        value: nn.Linear,
        positions: nn.Embedding,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Make series_tokens, run each through series_mlp,
    #   and average over the positions for the series' summary. Join it with
    #   the question's embedding along the last dimension, run head, and
    #   return one answer per row, of shape (batch,).
''' + indented(LATE_INIT) + '''

        def forward(
            self,
            series: Tensor,
            question: Tensor,
        ) -> Tensor:
            ...


    # TODO: Finish forward. Add kinds row 0 to every series token, and make the
    #   question one more token, its embedding plus kinds row 1. Put it after
    #   the series tokens, run block over all of them, and read the answer
    #   from the question's place with ln_f then head, of shape (batch,).
''' + indented(EARLY_INIT) + '''

        def forward(
            self,
            series: Tensor,
            question: Tensor,
        ) -> Tensor:
            ...
    ''',
)

FUSE1_SOLUTION = source_file(
    '''
    # fuse1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    series = torch.tensor([[0.5, -1.0, 2.0]])
    series.unsqueeze(-1).shape
    joined = torch.cat([torch.zeros(2, 3, 4), torch.ones(2, 1, 4)], dim=1)


''' + indented(MIX_BLOCK_PART) + '''


''' + indented(SERIES_TOKENS_SOLUTION) + '''


''' + indented(LATE_INIT + LATE_FORWARD) + '''


''' + indented(EARLY_INIT + EARLY_FORWARD) + '''
    ''',
)


def fusion_inputs(
    seed: int,
    batch: int,
    length: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    series = torch.rand(batch, length, generator=generator) * 2 - 1

    return series, torch.randint(0, length, (batch,), generator=generator)


def check_fusion_forward(
    found: type,
    name: str,
) -> None:
    for length, size, batch, seed in ((6, 8, 3, 1001), (5, 4, 2, 1002)):
        call = f"{name}({length}, {size})"
        model = expect_module(call, seeded_model(found, seed, length, size))
        series, question = fusion_inputs(seed, batch, length)

        with torch.no_grad():
            expected = seeded_model(FUSE1_REFERENCE[name], seed, length, size)(series, question)
            actual = model(series, question)

        expect_tensor(
            call=f"{call} on {batch} series of {length} and their questions",
            actual=actual,
            expected=expected,
        )


def check_fuse1(checker: Checker) -> None:
    for seed, batch, steps, size in ((1003, 2, 3, 4), (1004, 3, 5, 2)):
        value = seeded_model(nn.Linear, seed, 1, size)
        positions = seeded_model(nn.Embedding, seed + 1, steps + 2, size)
        check_on_inputs(
            checker=checker,
            function="series_tokens",
            description="gives each number a token, its value plus its place",
            inputs=((small_tensor(seed, batch, steps) / 3, Described(value, f"nn.Linear(1, {size})"), Described(positions, f"nn.Embedding({steps + 2}, {size})")),),
            reference=FUSE1_REFERENCE["series_tokens"],
        )

    checker.test_method(
        class_name="LateFusion",
        method="forward",
        description="sums up the series alone, then joins the summary with the question",
        run=lambda found: check_fusion_forward(found, "LateFusion"),
    )
    checker.test_method(
        class_name="EarlyFusion",
        method="forward",
        description="puts the question in the series' sequence and reads the answer at its place",
        run=lambda found: check_fusion_forward(found, "EarlyFusion"),
    )


FUSE1 = Exercise(
    name="fuse1",
    chapter=FUSION,
    title="Early and late fusion",
    lesson=FUSE1_LESSON,
    source=FUSE1_SOURCE,
    solution=FUSE1_SOLUTION,
    hints={
        "series_tokens": "Give series a last dimension of 1 with .unsqueeze(-1) and run value on it. Add the first "
        "series.shape[1] rows of positions.weight, which broadcast over the batch.",
        "LateFusion.forward": "series_tokens with self.value and self.positions, then self.series_mlp, then .mean(dim=1). "
        "torch.cat that with self.question(question) on dim=-1, run self.head and .squeeze(-1).",
        "EarlyFusion.forward": "self.kinds.weight[0] and self.kinds.weight[1] are the two type rows. The question token needs "
        "a positions dimension, .unsqueeze(1), before torch.cat on dim=1. The answer sits at index -1 of dim 1.",
    },
    introduces=("late fusion", "early fusion", "type embedding"),
    check=check_fuse1,
)


# ---------------------------------------------------------------------------
# fuse2
# ---------------------------------------------------------------------------

VOCABULARY = 16
TEXT_STEPS = 12
SERIES_LENGTH = 16
LM_WIDTH = 32

LM_PART = '''
''' + SELF_ATTENTION + '''

class Block(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(size)
        self.attn = SelfAttention(size, heads)
        self.ln_2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(nn.Linear(size, 4 * size), nn.GELU(), nn.Linear(4 * size, size))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))

        return x + self.mlp(self.ln_2(x))


# A small GPT, trained on text alone before this exercise begins.
class TinyLM(nn.Module):
    def __init__(
        self,
        vocabulary: int = 16,
        steps: int = 12,
        size: int = 32,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocabulary, size)
        self.wpe = nn.Embedding(steps, size)
        self.h = nn.ModuleList([Block(size, 4) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(size)
        self.head = nn.Linear(size, vocabulary)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.wte(ids) + self.wpe.weight[: ids.shape[1]]

        for block in self.h:
            x = block(x)

        return self.head(self.ln_f(x))


# cross1's cross attention with heads. The queries come from the text, the keys and values from the memory.
class MemoryAttention(nn.Module):
    def __init__(
        self,
        size: int,
        memory_size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(size, size)
        self.key_value = nn.Linear(memory_size, 2 * size)
        self.out = nn.Linear(size, size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
    ) -> Tensor:
        batch, steps, size = x.shape
        q = self.query(x).view(batch, steps, self.heads, -1).transpose(1, 2)
        k, v = self.key_value(memory).split(size, dim=2)
        k, v = (part.view(batch, memory.shape[1], self.heads, -1).transpose(1, 2) for part in (k, v))
        mixed = F.scaled_dot_product_attention(q, k, v)

        return self.out(mixed.transpose(1, 2).reshape(batch, steps, size))


# A token per number of the series, as fuse1's series_tokens made them.
class SeriesEncoder(nn.Module):
    def __init__(
        self,
        length: int = 16,
        size: int = 32,
    ) -> None:
        super().__init__()
        self.value = nn.Linear(1, size)
        self.positions = nn.Embedding(length, size)

    def forward(self, series: Tensor) -> Tensor:
        return self.value(series.unsqueeze(-1)) + self.positions.weight[: series.shape[1]]
'''

GATED_START = '''
class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        size: int,
        memory_size: int,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(size)
        self.cross = MemoryAttention(size, memory_size, 4)
        self.norm_2 = nn.LayerNorm(size)
        self.ff = nn.Sequential(nn.Linear(size, 4 * size), nn.GELU(), nn.Linear(4 * size, size))
'''

GATES_SOLUTION = '''
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
'''

GATED_FORWARD = '''
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
    ) -> Tensor:
        x = x + torch.tanh(self.attn_gate) * self.cross(self.norm_1(x), memory)

        return x + torch.tanh(self.ff_gate) * self.ff(self.norm_2(x))
'''

FLAMINGO_INIT = '''
class Flamingo(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        size: int = 32,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.encoder = SeriesEncoder(16, size)
        self.xattn = nn.ModuleList([GatedCrossAttention(size, size) for _ in lm.h])
'''

FLAMINGO_FORWARD = '''
    def forward(
        self,
        ids: Tensor,
        series: Tensor,
    ) -> Tensor:
        memory = self.encoder(series)
        x = self.lm.wte(ids) + self.lm.wpe.weight[: ids.shape[1]]

        for xattn, block in zip(self.xattn, self.lm.h):
            x = block(xattn(x, memory))

        return self.lm.head(self.lm.ln_f(x))
'''

FREEZE_SOLUTION = '''
def freeze_(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
'''

FUSE2_REFERENCE = namespace_of(
    LM_PART + GATED_START + GATES_SOLUTION + GATED_FORWARD + FLAMINGO_INIT + FLAMINGO_FORWARD + FREEZE_SOLUTION,
)


def climbing_text(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sequences of 12 tokens that climb by a step of 1 to 7 each token, wrapping past 15, and their steps."""
    start = torch.randint(0, VOCABULARY, (count, 1), generator=generator)
    step = torch.randint(1, 8, (count, 1), generator=generator)

    return (start + step * torch.arange(TEXT_STEPS)) % VOCABULARY, step.squeeze(1)


def slope_series(
    step: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """A series of 16 numbers rising with the step as its slope, with noise."""
    t = torch.linspace(0, 1, SERIES_LENGTH)

    return step.float().unsqueeze(1) * t + 0.1 * torch.randn(len(step), SERIES_LENGTH, generator=generator)


FUSE2_LESSON = prose(
    """
    A trained language model is costly to train again. Flamingo joined images
    to a frozen one by adding new layers between its blocks, each reading the
    image through cross attention, as cross1 did.

    New random layers would change what the trained model says from the
    first step. So each new layer's output is multiplied by `tanh(gate)`
    before it joins the stream, with the gate a learned number starting at 0.

    ```text
    x = x + tanh(attn_gate) * cross(norm(x), memory)
    x = x + tanh(ff_gate) * ff(norm(x))
    ```

    At step 0, `tanh(0) = 0`, so the model is exactly the frozen one. It is
    residual3's zero start, made by a gate in the spirit of gating1's. The
    gate still gets a gradient at 0, since tanh's slope there is 1, and it
    opens as the new layers prove useful. Without the gates, Flamingo trained unstably and
    scored 66.5 against 70.7.

    Here a tiny LM learned text that climbs by a step of 1 to 7 each token.
    After two tokens it knows the step. Its first prediction can only guess,
    a loss near ln 7, 1.95. A series whose slope is the step carries what it
    lacks. The LM stays frozen, and only the new layers train, three seeds.

    | New layers | Text loss at step 0 | First token, step 200 | First token, step 1,000 | Text loss, step 1,000 |
    |---|---|---|---|---|
    | gated, starting closed | 0.0004 to 0.0005 | 0.83 to 0.95 | 0.018 to 0.042 | 0.0011 to 0.0012 |
    | open, no gates | 1.26 to 1.87 | 1.61 to 1.76 | 0.054 to 0.134 | 0.0014 to 0.0018 |

    The frozen LM alone scores 2.00 to 2.02 on the first token and 0.0004 to
    0.0005 on the rest. The open layers broke it at step 0. The gated ones
    learned to read the series faster and further, and kept a lower text loss.
    """,
)

FUSE2_SOURCE = source_file(
    '''
    # fuse2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    gate = nn.Parameter(torch.zeros(1))
    closed = torch.tanh(gate) * torch.tensor([3.0, -2.0])    # tensor([0., -0.]), the branch adds nothing yet

''' + indented(LM_PART) + '''


    # TODO: Finish __init__ and forward. Make attn_gate and ff_gate, each a
    #   learned number of shape (1,) that starts at 0. Then add the cross
    #   attention on norm_1(x) and the memory to x, times tanh(attn_gate).
    #   Add ff on norm_2 of the result, times tanh(ff_gate), and return it.
''' + indented(GATED_START) + '''
            ...

        def forward(
            self,
            x: Tensor,
            memory: Tensor,
        ) -> Tensor:
            ...


    # TODO: Finish forward. Encode the series into the memory. Embed ids with
    #   lm's wte and the first rows of its wpe. Before each of lm's blocks, run
    #   its own GatedCrossAttention with the memory, then the block. Return the
    #   scores from lm's ln_f and head.
''' + indented(FLAMINGO_INIT) + '''

        def forward(
            self,
            ids: Tensor,
            series: Tensor,
        ) -> Tensor:
            ...


    # TODO: Stop every parameter of module from training, so no gradient is
    #   kept for it and no optimizer can change it.
    def freeze_(module: nn.Module) -> None:
        ...
    ''',
)

FUSE2_SOLUTION = source_file(
    '''
    # fuse2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    gate = nn.Parameter(torch.zeros(1))
    closed = torch.tanh(gate) * torch.tensor([3.0, -2.0])

''' + indented(LM_PART) + '''


''' + indented(GATED_START + GATES_SOLUTION + GATED_FORWARD) + '''


''' + indented(FLAMINGO_INIT + FLAMINGO_FORWARD) + '''


''' + indented(FREEZE_SOLUTION) + '''
    ''',
)


def made_gated(
    found: type,
    seed: int,
) -> nn.Module:
    return expect_module("GatedCrossAttention(8, 6)", seeded_model(found, seed, 8, 6))


def check_gates(found: type) -> None:
    layer = made_gated(found, 1101)

    for name in ("attn_gate", "ff_gate"):
        gate = getattr(layer, name, None)

        if not isinstance(gate, nn.Parameter):
            raise Mismatch(f"GatedCrossAttention(8, 6).{name} is {value_text(gate)}. It should be an nn.Parameter, so it learns.")

        if gate.shape != (1,):
            raise Mismatch(f"GatedCrossAttention(8, 6).{name} has shape {tuple(gate.shape)}, but it should hold one number, shape (1,).")

        if gate.item() != 0:
            raise Mismatch(f"GatedCrossAttention(8, 6).{name} starts at {gate.item():.4g}, but it should start at 0, so the layer starts closed.")

    x = small_tensor(1102, 2, 3, 8) / 3
    memory = small_tensor(1103, 2, 4, 6) / 3

    with torch.no_grad():
        actual = layer(x, memory)

    expect_tensor(
        call="a new GatedCrossAttention(8, 6) on x of shape (2, 3, 8)",
        actual=actual,
        expected=x,
    )


def check_gated_forward(found: type) -> None:
    for seed, gates in ((1104, (0.5, -0.3)), (1105, (1.2, 0.8))):
        layer = made_gated(found, seed)
        reference = seeded_model(FUSE2_REFERENCE["GatedCrossAttention"], seed, 8, 6)

        # Fresh LayerNorms are all alike, so the second gets its own scale to tell the two apart.
        with torch.no_grad():
            for module in (layer, reference):
                module.attn_gate.fill_(gates[0])
                module.ff_gate.fill_(gates[1])
                module.norm_2.weight.copy_(torch.linspace(0.5, 1.5, 8))

        x = small_tensor(seed, 2, 3, 8) / 3
        memory = small_tensor(seed + 1, 2, 5, 6) / 3

        with torch.no_grad():
            expected = reference(x, memory)
            actual = layer(x, memory)

        expect_tensor(
            call=f"GatedCrossAttention(8, 6) with attn_gate {gates[0]} and ff_gate {gates[1]}",
            actual=actual,
            expected=expected,
        )


def small_lm(seed: int) -> nn.Module:
    torch.manual_seed(seed)

    return FUSE2_REFERENCE["TinyLM"](16, 12, 8, 2)


def check_flamingo_forward(found: type) -> None:
    for seed, batch, steps in ((1106, 2, 5), (1107, 3, 4)):
        torch.manual_seed(seed)
        model = expect_module("Flamingo(lm, 8)", found(small_lm(seed), 8))
        torch.manual_seed(seed)
        reference = FUSE2_REFERENCE["Flamingo"](small_lm(seed), 8)

        with torch.no_grad():
            for module in (model, reference):
                for layer in module.xattn:
                    layer.attn_gate.fill_(0.7)
                    layer.ff_gate.fill_(-0.4)

        generator = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, 16, (batch, steps), generator=generator)
        series = torch.randn(batch, 16, generator=generator)

        with torch.no_grad():
            expected = reference(ids, series)
            actual = model(ids, series)

        expect_tensor(
            call=f"Flamingo(lm, 8) on {batch} texts of {steps} tokens and their series, every gate opened",
            actual=actual,
            expected=expected,
        )


def check_starts_as_lm(found: type) -> None:
    lm = small_lm(1108)
    torch.manual_seed(1108)
    model = expect_module("Flamingo(lm, 8)", found(lm, 8))
    generator = torch.Generator().manual_seed(1108)
    ids = torch.randint(0, 16, (3, 6), generator=generator)
    series = torch.randn(3, 16, generator=generator)

    with torch.no_grad():
        expected = lm(ids)
        actual = model(ids, series)

    expect_tensor(
        call="a new Flamingo(lm, 8), every gate still at 0, against lm(ids) alone",
        actual=actual,
        expected=expected,
    )


def check_freeze(found: object) -> None:
    for seed in (1109, 1110):
        lm = small_lm(seed)
        torch.manual_seed(seed)
        model = FUSE2_REFERENCE["Flamingo"](lm, 8)
        found(lm)
        trainable = [name for name, parameter in lm.named_parameters() if parameter.requires_grad]

        if trainable:
            raise Mismatch(f"After freeze_(lm), these of lm's parameters still train: {', '.join(trainable)}.")

        new = [name for name, parameter in model.named_parameters() if not name.startswith("lm.") and not parameter.requires_grad]

        if new:
            raise Mismatch(f"freeze_(lm) froze layers outside lm too: {', '.join(new)}.")


def check_fuse2(checker: Checker) -> None:
    checker.test_method(
        class_name="GatedCrossAttention",
        method="__init__",
        description="makes two gates, one learned number each, starting at 0",
        run=check_gates,
    )
    checker.test_method(
        class_name="GatedCrossAttention",
        method="forward",
        description="adds cross attention and ff, each times tanh of its gate",
        run=check_gated_forward,
    )
    checker.test_method(
        class_name="Flamingo",
        method="forward",
        description="runs a gated cross attention before each of the LM's blocks",
        run=check_flamingo_forward,
    )
    checker.test_method(
        class_name="Flamingo",
        method="forward",
        description="starts as exactly the frozen LM",
        run=check_starts_as_lm,
    )
    checker.test(
        function="freeze_",
        description="stops every parameter of a module from training",
        run=check_freeze,
    )


FUSE2 = Exercise(
    name="fuse2",
    chapter=FUSION,
    title="Gated cross attention",
    lesson=FUSE2_LESSON,
    source=FUSE2_SOURCE,
    solution=FUSE2_SOLUTION,
    hints={
        "GatedCrossAttention.__init__": "Each gate is nn.Parameter of torch.zeros(1), set as self.attn_gate and self.ff_gate in "
        "place of the ... line.",
        "GatedCrossAttention.forward": "Two residual lines. The first adds torch.tanh(self.attn_gate) times self.cross of "
        "self.norm_1(x) and memory. The second does the same with self.ff_gate, self.ff and self.norm_2.",
        "Flamingo.forward": "self.encoder(series) is the memory. zip(self.xattn, self.lm.h) pairs each new layer with the "
        "block it goes before. Run the new layer on x and memory, then the block on that.",
        "freeze_": "Loop over module.parameters() and call .requires_grad_(False) on each.",
    },
    introduces=(),
    check=check_fuse2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# fuse3
# ---------------------------------------------------------------------------

PAIRED_SERIES = 16
COLUMNS = 4
TABLE_WEIGHTS = torch.tensor([0.8, -0.6, 0.5, 0.3])

DROP_SOLUTION = '''
def dropped_masks(
    count: int,
    rate: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    roll = torch.rand(count, generator=generator)
    has_series = roll >= rate
    has_table = (roll < rate) | (roll >= 2 * rate)

    return has_series, has_table
'''

PAIRED_INIT = '''
class PairedModel(nn.Module):
    def __init__(
        self,
        length: int = 16,
        columns: int = 4,
        size: int = 16,
    ) -> None:
        super().__init__()
        self.series_encoder = nn.Sequential(nn.Linear(length, 32), nn.ReLU(), nn.Linear(32, size))
        self.table_encoder = nn.Sequential(nn.Linear(columns, 32), nn.ReLU(), nn.Linear(32, size))
        self.missing_series = nn.Parameter(torch.randn(size))
        self.missing_table = nn.Parameter(torch.randn(size))
        self.head = nn.Sequential(nn.Linear(2 * size, 32), nn.ReLU(), nn.Linear(32, 1))
'''

PAIRED_FORWARD = '''
    def forward(
        self,
        series: Tensor,
        table: Tensor,
        has_series: Tensor,
        has_table: Tensor,
    ) -> Tensor:
        from_series = torch.where(has_series.unsqueeze(1), self.series_encoder(series), self.missing_series)
        from_table = torch.where(has_table.unsqueeze(1), self.table_encoder(table), self.missing_table)

        return self.head(torch.cat([from_series, from_table], dim=1)).squeeze(1)
'''

TRAIN_PAIRED_SOLUTION = '''
def train_paired(
    model: nn.Module,
    data: tuple[Tensor, Tensor, Tensor],
    rate: float,
    steps: int,
    generator: torch.Generator,
) -> float:
    series, table, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(steps):
        batch = torch.randint(0, len(y), (128,), generator=generator)
        has_series, has_table = dropped_masks(128, rate, generator)
        loss = F.mse_loss(model(series[batch], table[batch], has_series, has_table), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

ERROR_SOLUTION = '''
def error_with(
    model: nn.Module,
    data: tuple[Tensor, Tensor, Tensor],
    has_series: Tensor,
    has_table: Tensor,
) -> float:
    series, table, y = data

    with torch.no_grad():
        return F.mse_loss(model(series, table, has_series, has_table), y).item()
'''

FUSE3_REFERENCE = namespace_of(DROP_SOLUTION + PAIRED_INIT + PAIRED_FORWARD + TRAIN_PAIRED_SOLUTION + ERROR_SOLUTION)


def paired_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """A series of 16 and a table row of 4. The answer is a part from each, and the parts come back too."""
    series = torch.rand(count, PAIRED_SERIES, generator=generator) * 2 - 1
    table = torch.rand(count, COLUMNS, generator=generator) * 2 - 1
    from_series = series[:, PAIRED_SERIES // 2 :].mean(dim=1) * 3
    from_table = table @ TABLE_WEIGHTS

    return series, table, from_series + from_table, from_series, from_table


FUSE3_LESSON = prose(
    """
    In use, an input goes missing. A sensor fails, a form has no table, a
    question comes with no series. A model that always saw both inputs has
    never learned what to do then.

    Give each kind of input a learned stand in, a vector the model gets in
    its place when it is missing. Then train with inputs dropped at random,
    as dropout1 dropped units. Neverova and colleagues called this ModDrop.
    The model learns to answer from what it has.

    Here the answer is one part from a series of 16 and one from a table row
    of 4. With one input missing, the best any model can do is answer the
    part it has and the average of the other. Each model trained 3,000 steps
    on three seeds. A rate of 0.15 drops the series in 15% of rows and the
    table in another 15%, never both.

    | Training | Both present | Series missing | Table missing |
    |---|---|---|---|
    | always both | 0.0001 | 0.42 to 0.72 | 0.55 to 1.14 |
    | inputs dropped at 0.15 | 0.002 to 0.003 | 0.38 to 0.39 | 0.45 to 0.46 |
    | the best any model can do | 0 | 0.38 to 0.39 | 0.45 to 0.46 |

    Guessing the mean answer scores 0.84 to 0.85. The model that always saw
    both had never trained its stand ins. A missing input sent it far past
    the best, at times worse than the guess. The dropped model landed on the
    best, and with both inputs it still reads them nearly perfectly.
    """,
)

FUSE3_SOURCE = source_file(
    '''
    # fuse3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    roll = torch.tensor([0.05, 0.2, 0.5])
    (roll < 0.1) | (roll >= 0.3)    # tensor([ True, False,  True]), true below 0.1 or from 0.3 up
    torch.where(torch.tensor([[True], [False]]), torch.ones(2, 3), torch.zeros(3))    # row 0 ones, row 1 the stand in


    # TODO: Return has_series and has_table, two tensors of count True and
    #   False. Draw one number per row with torch.rand and the generator. A
    #   row drops its series below rate, and its table from rate up to
    #   2 * rate, so it never drops both.
    def dropped_masks(
        count: int,
        rate: float,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish forward. Encode the series and the table. Where has_series
    #   is False, use missing_series in the series' place, and the same for
    #   the table. Join the two along dim 1, run head, return shape (batch,).
''' + indented(PAIRED_INIT) + '''

        def forward(
            self,
            series: Tensor,
            table: Tensor,
            has_series: Tensor,
            has_table: Tensor,
        ) -> Tensor:
            ...


    # TODO: Train model for steps steps with Adam at 1e-3, and return the last
    #   loss as a plain number. Each step, draw 128 rows of data with
    #   torch.randint and the generator, then dropped_masks for 128 rows at
    #   rate. The loss is the mean squared error against y.
    def train_paired(
        model: nn.Module,
        data: tuple[Tensor, Tensor, Tensor],
        rate: float,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        series, table, y = data
        ...


    # TODO: Return the model's mean squared error on all of data, with the
    #   inputs present as has_series and has_table say, as a plain number.
    #   Record nothing for autograd.
    def error_with(
        model: nn.Module,
        data: tuple[Tensor, Tensor, Tensor],
        has_series: Tensor,
        has_table: Tensor,
    ) -> float:
        ...
    ''',
)

FUSE3_SOLUTION = source_file(
    '''
    # fuse3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    roll = torch.tensor([0.05, 0.2, 0.5])
    (roll < 0.1) | (roll >= 0.3)
    torch.where(torch.tensor([[True], [False]]), torch.ones(2, 3), torch.zeros(3))


''' + indented(DROP_SOLUTION) + '''


''' + indented(PAIRED_INIT + PAIRED_FORWARD) + '''


''' + indented(TRAIN_PAIRED_SOLUTION) + '''


''' + indented(ERROR_SOLUTION) + '''
    ''',
)


def expect_mask_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a pair, has_series and has_table.")

    for name, found, wanted in zip(("has_series", "has_table"), actual, expected):
        expect_tensor(
            call=f"{name} from {call}",
            actual=found,
            expected=wanted,
        )

        if found.dtype != torch.bool:
            raise Mismatch(f"{name} from {call} holds {found.dtype} numbers, but it should hold True and False. Return the comparisons themselves.")


def check_paired_forward(found: type) -> None:
    for seed, batch in ((1201, 4), (1202, 5)):
        model = expect_module("PairedModel(6, 3, 4)", seeded_model(found, seed, 6, 3, 4))
        reference = seeded_model(FUSE3_REFERENCE["PairedModel"], seed, 6, 3, 4)
        series = small_tensor(seed, batch, 6) / 3
        table = small_tensor(seed + 1, batch, 3) / 3
        generator = torch.Generator().manual_seed(seed)
        has_series = torch.rand(batch, generator=generator) > 0.4
        has_table = torch.rand(batch, generator=generator) > 0.4

        with torch.no_grad():
            expected = reference(series, table, has_series, has_table)
            actual = model(series, table, has_series, has_table)

        expect_tensor(
            call=f"PairedModel(6, 3, 4) on {batch} rows, has_series {value_text(has_series)} and has_table {value_text(has_table)}",
            actual=actual,
            expected=expected,
        )


def small_paired(seed: int) -> tuple[nn.Module, tuple[Tensor, Tensor, Tensor]]:
    model = seeded_model(FUSE3_REFERENCE["PairedModel"], seed, 6, 3, 4)
    generator = torch.Generator().manual_seed(seed)
    series = torch.rand(300, 6, generator=generator) * 2 - 1
    table = torch.rand(300, 3, generator=generator) * 2 - 1

    return model, (series, table, series.mean(dim=1) + table.sum(dim=1))


def check_train_paired(found: object) -> None:
    for seed, rate in ((1203, 0.2), (1204, 0.35)):
        model, data = small_paired(seed)
        reference = copy.deepcopy(model)
        actual = found(model, data, rate, 2, torch.Generator().manual_seed(seed))
        expected = FUSE3_REFERENCE["train_paired"](reference, data, rate, 2, torch.Generator().manual_seed(seed))
        call = f"train_paired on a PairedModel(6, 3, 4), rate {rate}, 2 steps"

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        )

        for (name, trained), wanted in zip(model.named_parameters(), reference.parameters()):
            expect_tensor(
                call=f"{name} after {call}",
                actual=trained.detach(),
                expected=wanted.detach(),
            )


def check_error_with(found: object) -> None:
    for seed in (1205, 1206):
        model, data = small_paired(seed)
        generator = torch.Generator().manual_seed(seed)
        has_series = torch.rand(300, generator=generator) > 0.3
        has_table = torch.rand(300, generator=generator) > 0.3
        seen: list[bool] = []
        handle = recording_hook(model, seen)

        try:
            actual = found(model, data, has_series, has_table)
        finally:
            handle.remove()

        expect_close(
            call="error_with on a PairedModel(6, 3, 4) and 300 rows",
            actual=actual,
            expected=FUSE3_REFERENCE["error_with"](model, data, has_series, has_table),
            tolerance=1e-5,
        )

        if any(seen):
            raise Mismatch("error_with ran the model while recording for autograd. Score inside torch.no_grad().")


def check_fuse3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="dropped_masks",
        description="drops the series below the rate and the table from the rate to twice it",
        inputs=(
            (8, 0.25, seeded(1207)),
            (10, 0.4, seeded(1208)),
        ),
        reference=FUSE3_REFERENCE["dropped_masks"],
        expect=expect_mask_pair,
    )
    checker.test_method(
        class_name="PairedModel",
        method="forward",
        description="uses the learned stand in wherever an input is missing",
        run=check_paired_forward,
    )
    checker.test(
        function="train_paired",
        description="trains with inputs dropped at the rate",
        run=check_train_paired,
    )
    checker.test(
        function="error_with",
        description="scores the model with the inputs it is given",
        run=check_error_with,
    )


FUSE3 = Exercise(
    name="fuse3",
    chapter=FUSION,
    title="When an input is missing",
    lesson=FUSE3_LESSON,
    source=FUSE3_SOURCE,
    solution=FUSE3_SOLUTION,
    hints={
        "dropped_masks": "One torch.rand(count, generator=generator). has_series is where it is at least rate. has_table "
        "is where it is below rate, or at least 2 * rate, joined with |.",
        "PairedModel.forward": "torch.where takes the mask with .unsqueeze(1), the encoded rows, and the stand in, which "
        "broadcasts to every row. torch.cat the two results on dim=1, then self.head and .squeeze(1).",
        "train_paired": "Like gpt3's train_step in a loop, drawing torch.randint(0, len(y), (128,), generator=generator) "
        "before dropped_masks(128, rate, generator). Zero the gradients, backpropagate, step.",
        "error_with": "Inside torch.no_grad(), F.mse_loss of the model's answers and y, then .item().",
    },
    introduces=(),
    check=check_fuse3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def train_fusion(
    model: nn.Module,
    data: Callable[[int, torch.Generator], tuple[Tensor, Tensor, Tensor]],
    seed: int,
    steps: int = 1500,
) -> float:
    """fuse1's runs, Adam at 3e-3 on fresh batches of 128, scored on 2,000 new inputs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        series, question, y = data(128, generator)
        loss = F.mse_loss(model(series, question), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    series, question, y = data(2000, torch.Generator().manual_seed(seed + 1000))

    with torch.no_grad():
        return F.mse_loss(model(series, question), y).item()


def measure_fusion(seeds: range = range(3)) -> dict[str, object]:
    """fuse1's table: late and early fusion on the lookup and the sum, their weights, and the error of guessing the mean."""
    results: dict[str, object] = {}

    with pinned_threads():
        for task, data in (("lookup", lookup_data), ("sum", sum_data)):
            for name in ("LateFusion", "EarlyFusion"):
                errors = [train_fusion(seeded_model(FUSE1_REFERENCE[name], seed), data, seed) for seed in seeds]
                results[f"{name} {task}"] = errors

            _, _, y = data(2000, torch.Generator().manual_seed(1000))
            results[f"guess {task}"] = y.var().item()

        for name in ("LateFusion", "EarlyFusion"):
            results[f"{name} weights"] = sum(parameter.numel() for parameter in FUSE1_REFERENCE[name]().parameters())

    return results


class OpenCrossAttention(nn.Module):
    """fuse2's comparison, the same layers with no gates, so they add their full output from the start."""

    def __init__(
        self,
        size: int,
        memory_size: int,
    ) -> None:
        super().__init__()
        self.inner = FUSE2_REFERENCE["GatedCrossAttention"](size, memory_size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
    ) -> Tensor:
        inner = self.inner
        x = x + inner.cross(inner.norm_1(x), memory)

        return x + inner.ff(inner.norm_2(x))


def climbing_losses(
    model: Callable[[Tensor], Tensor],
    ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """The loss of the first prediction, which needs the step, and of every later one, which the text gives."""
    scores = model(ids[:, :-1])
    each = F.cross_entropy(scores.flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(ids.shape[0], -1)

    return each[:, 0].mean(), each[:, 1:].mean()


def trained_lm(
    seed: int,
    steps: int = 1500,
) -> nn.Module:
    """fuse2's frozen LM, trained on climbing text alone with Adam at 3e-3."""
    torch.manual_seed(seed)
    lm = FUSE2_REFERENCE["TinyLM"]()
    optimizer = torch.optim.Adam(lm.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        ids, _ = climbing_text(128, generator)
        first, rest = climbing_losses(lm, ids)
        loss = (first + rest * (TEXT_STEPS - 2)) / (TEXT_STEPS - 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    FUSE2_REFERENCE["freeze_"](lm)

    return lm


def flamingo_scores(
    model: nn.Module,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed + 500)
    ids, step = climbing_text(1000, generator)
    series = slope_series(step, generator)

    with torch.no_grad():
        first, rest = climbing_losses(lambda shown: model(shown, series), ids)

    return first.item(), rest.item()


def measure_gates(
    seeds: range = range(3),
    steps: int = 1000,
) -> dict[str, list[dict[int, tuple[float, float]]]]:
    """fuse2's table: the first token's loss and the rest's, at steps 0, 200 and 1,000, gated and open, on frozen LMs."""
    results: dict[str, list[dict[int, tuple[float, float]]]] = {"lm": [], "gated": [], "open": []}

    with pinned_threads():
        for seed in seeds:
            lm = trained_lm(seed)
            generator = torch.Generator().manual_seed(seed + 500)
            ids, _ = climbing_text(1000, generator)

            with torch.no_grad():
                first, rest = climbing_losses(lm, ids)

            results["lm"].append({0: (first.item(), rest.item())})

            for kind in ("gated", "open"):
                torch.manual_seed(100 + seed)
                model = FUSE2_REFERENCE["Flamingo"](lm)

                if kind == "open":
                    model.xattn = nn.ModuleList([OpenCrossAttention(LM_WIDTH, LM_WIDTH) for _ in lm.h])

                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=1e-3)
                generator = torch.Generator().manual_seed(seed + 7)
                marks = {0: flamingo_scores(model, seed)}

                for step in range(1, steps + 1):
                    ids, climb = climbing_text(128, generator)
                    series = slope_series(climb, generator)
                    first, rest = climbing_losses(lambda shown: model(shown, series), ids)
                    loss = (first + rest * (TEXT_STEPS - 2)) / (TEXT_STEPS - 1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step in (200, steps):
                        marks[step] = flamingo_scores(model, seed)

                results[kind].append(marks)

    return results


def measure_dropping(
    seeds: range = range(3),
    steps: int = 3000,
) -> dict[str, list[tuple[float, float, float]]]:
    """fuse3's table: the error with both inputs, without the series and without the table, the best possible,
    and the error of guessing the mean answer."""
    results: dict[str, list[tuple[float, float, float]]] = {"best": [], "guess": [], "0.0": [], "0.15": []}

    with pinned_threads():
        for seed in seeds:
            series, table, y, from_series, from_table = paired_data(8192, torch.Generator().manual_seed(seed))
            test = paired_data(4000, torch.Generator().manual_seed(seed + 99))
            scored = test[:3]
            results["best"].append((0.0, test[3].var().item(), test[4].var().item()))
            results["guess"].append((test[2].var().item(), test[2].var().item(), test[2].var().item()))
            present = torch.ones(4000, dtype=torch.bool)
            absent = torch.zeros(4000, dtype=torch.bool)

            for rate in (0.0, 0.15):
                model = seeded_model(FUSE3_REFERENCE["PairedModel"], seed)
                FUSE3_REFERENCE["train_paired"](model, (series, table, y), rate, steps, torch.Generator().manual_seed(seed))
                results[str(rate)].append(
                    tuple(
                        FUSE3_REFERENCE["error_with"](model, scored, has_series, has_table)
                        for has_series, has_table in ((present, present), (absent, present), (present, absent))
                    ),
                )

    return results
