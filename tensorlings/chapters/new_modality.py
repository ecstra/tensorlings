"""101_reading_a_new_modality, Section 13, LLaVA's recipe on your GPT: a frozen signal encoder, a small projector into
the GPT's embeddings, the projector trained alone and then the GPT with LoRA, and what the projected tokens carry."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    expect_close,
    expect_tensor,
    reject_missing_return,
    value_text,
)
from ..text import prose, source_file
from ..training import load_checkpoint
from .finetuning import GPT_SETTINGS, LOAD_PART, MODEL_PART
from .lora import EVERYWHERE_SOLUTION, LORA_PART

NEW_MODALITY = Chapter(
    folder="101_reading_a_new_modality",
    title="Reading a new modality",
    readme=source_file(
        """
        # Reading a new modality

        Your GPT reads characters. This chapter teaches it to read signals,
        the way LLaVA taught a language model to read images. A frozen encoder
        turns each signal into tokens, a small projector maps them into the
        GPT's embeddings, and two stages of training teach the GPT to use them.
        Then you look at what those projected tokens carry.

        ## Further reading

        - [Liu and colleagues, visual instruction tuning, LLaVA](https://arxiv.org/abs/2304.08485)
        - [Liu and colleagues, improved baselines with visual instruction tuning, LLaVA-1.5](https://arxiv.org/abs/2310.03744)
        - [Radford and colleagues, CLIP](https://arxiv.org/abs/2103.00020)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def joined(*parts: str) -> str:
    """Parts of code one after another, two blank lines apart as in a file."""
    return "\n\n\n".join(part.strip("\n") for part in parts) + "\n"


# lora1's layer without add_lora, which puts it on the attention alone.
LORA_LAYER_PART = LORA_PART[: LORA_PART.index("def add_lora(")]

# The new modality, signals built from three facts, and the caption stage one trains on.
SIGNAL_PART = '''
LENGTH = 64
WAVE_WORDS = ["one", "two", "three", "four"]
THIRDS = ["early", "middle", "late"]


# count signals of LENGTH steps, each built from three facts. It rises or falls, its wave has 1 to 4 cycles, and a
# spike sits in its first, middle or last third. The facts come back too, as whole numbers.
def make_signals(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    steps = torch.linspace(0, 1, LENGTH)
    rises = torch.randint(0, 2, (count,), generator=generator)
    waves = torch.randint(1, 5, (count,), generator=generator)
    thirds = torch.randint(0, 3, (count,), generator=generator)
    slopes = (2 * rises - 1) * (0.8 + 0.6 * torch.rand(count, generator=generator))
    signals = 0.6 * torch.sin(2 * math.pi * (waves[:, None] * steps - 0.25)) + 2 * slopes[:, None] * (steps - 0.5)
    spikes = thirds * (LENGTH // 3) + torch.randint(3, LENGTH // 3 - 3, (count,), generator=generator)
    signals[torch.arange(count), spikes] += 2.5

    return signals + 0.08 * torch.randn(count, LENGTH, generator=generator), rises, waves, thirds


# The three facts in words, as stage one's captions say them.
def caption(
    rise: int,
    waves: int,
    third: int,
) -> str:
    return f"{'rises' if rise else 'falls'}, {WAVE_WORDS[waves - 1]} waves, spike {THIRDS[third]}"
'''

# The frozen encoder, pretrained as an autoencoder in a few seconds, standing in for LLaVA's CLIP.
ENCODER_PART = '''
# Each window of 8 steps becomes one token of 32 numbers, so a signal becomes 8 tokens.
class SignalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 8, stride=8),
        )

    def forward(self, signals: Tensor) -> Tensor:
        return self.layers(signals.unsqueeze(1)).transpose(1, 2)


# Trained as autoencoder1 trained its codes, a decoder rebuilding each signal from its tokens, then frozen.
def pretrained_encoder(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    encoder = SignalEncoder()
    decoder = nn.Sequential(nn.ConvTranspose1d(32, 32, 8, stride=8), nn.GELU(), nn.Conv1d(32, 1, 5, padding=2))
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)
    generator = torch.Generator().manual_seed(seed + 100)

    for _ in range(300):
        signals, _, _, _ = make_signals(32, generator)
        loss = F.mse_loss(decoder(encoder(signals).transpose(1, 2)).squeeze(1), signals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return encoder.requires_grad_(False).eval()
'''

PROJECTOR_SOLUTION = '''
class Projector(nn.Module):
    def __init__(
        self,
        width: int,
        size: int,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(width, size)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(size, size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.proj(self.gelu(self.fc(tokens)))
'''

BRIDGE_SOLUTION = '''
def projected_tokens(
    encoder: nn.Module,
    projector: nn.Module,
    signals: Tensor,
) -> Tensor:
    with torch.no_grad():
        tokens = encoder(signals)

    return projector(tokens)


def prefix_forward(
    gpt: nn.Module,
    prefix: Tensor,
    ids: Tensor,
) -> Tensor:
    x = torch.cat([prefix, gpt.wte(ids)], dim=1)
    x = x + gpt.wpe(torch.arange(x.shape[1], device=ids.device))

    for block in gpt.h:
        x = block(x)

    return gpt.lm_head(gpt.ln_f(x))[:, prefix.shape[1] :]
'''

# sft1's answer only targets, for plain text, and the batches of captions stage one and two train on.
ROWS_PART = '''
# x and y for each prompt, then its answer, then a newline, with y set to -100 before the answer, padded to one batch.
def answer_rows(
    prompts: list[str],
    answers: list[str],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    rows = []

    for prompt, answer in zip(prompts, answers):
        ids = torch.tensor([positions[character] for character in prompt + answer + "\\n"])
        y = ids[1:].clone()
        y[: len(prompt) - 1] = -100
        rows.append((ids[:-1], y))

    longest = max(len(row_x) for row_x, _ in rows)
    x = torch.full((len(rows), longest), positions["\\n"])
    y = torch.full((len(rows), longest), -100)

    for place, (row_x, row_y) in enumerate(rows):
        x[place, : len(row_x)] = row_x
        y[place, : len(row_y)] = row_y

    return x, y


CAPTION_PROMPT = "Describe it: "


# count new signals and their captions, as x and y after the prompt.
def caption_batch(
    count: int,
    positions: dict[str, int],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    signals, rises, waves, thirds = make_signals(count, generator)
    captions = [caption(rise, wave, third) for rise, wave, third in zip(rises.tolist(), waves.tolist(), thirds.tolist())]
    x, y = answer_rows([CAPTION_PROMPT] * count, captions, positions)

    return signals, x, y
'''

STAGES_SOLUTION = '''
def stage_one(
    gpt: nn.Module,
    projector: nn.Module,
) -> torch.optim.Optimizer:
    gpt.requires_grad_(False)

    return torch.optim.AdamW(projector.parameters(), lr=2e-3, weight_decay=0.0)


def stage_two(
    gpt: nn.Module,
    projector: nn.Module,
) -> torch.optim.Optimizer:
    add_lora_everywhere(gpt, 8, 8.0).to(gpt.wte.weight.device)
    lora = [parameter for parameter in gpt.parameters() if parameter.requires_grad]

    return torch.optim.AdamW([*projector.parameters(), *lora], lr=1e-3, weight_decay=0.0)


def bridge_step(
    gpt: nn.Module,
    encoder: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    signals: Tensor,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    scores = prefix_forward(gpt, projected_tokens(encoder, projector, signals.to(device)), x.to(device))
    loss = F.cross_entropy(scores.flatten(0, 1), y.to(device).flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

# Greedy writing after the prefix, as post1's answer wrote, for the captions the lesson scores.
WRITE_PART = '''
# What gpt writes after each signal's tokens and the prompt, the likeliest character each step, until a newline or
# 40 characters.
def write_after(
    gpt: nn.Module,
    encoder: nn.Module,
    projector: nn.Module,
    signals: Tensor,
    prompt: str,
    positions: dict[str, int],
    characters: list[str],
) -> list[str]:
    gpt.eval()
    ids = torch.tensor([[positions[character] for character in prompt]] * len(signals), device=signals.device)
    written = [""] * len(signals)
    done = [False] * len(signals)

    with torch.no_grad():
        prefix = projected_tokens(encoder, projector, signals)

        for _ in range(40):
            tokens = prefix_forward(gpt, prefix, ids)[:, -1].argmax(dim=1)

            for row, token in enumerate(tokens.tolist()):
                if characters[token] == "\\n":
                    done[row] = True

                if not done[row]:
                    written[row] += characters[token]

            if all(done):
                break

            ids = torch.cat([ids, tokens.unsqueeze(1)], dim=1)

    gpt.train()

    return written
'''

NEAREST_SOLUTION = '''
def nearest_characters(
    gpt: nn.Module,
    vectors: Tensor,
    characters: list[str],
) -> tuple[list[str], Tensor]:
    similarity = F.normalize(vectors, dim=-1) @ F.normalize(gpt.wte.weight, dim=-1).T
    best, places = similarity.max(dim=-1)

    return [characters[place] for place in places.tolist()], best
'''

PROBE_SOLUTION = '''
def probe_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    classes: int,
) -> float:
    torch.manual_seed(0)
    probe = nn.Linear(train_features.shape[1], classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)

    for _ in range(300):
        loss = F.cross_entropy(probe(train_features), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (probe(test_features).argmax(dim=1) == test_labels).float().mean().item()
'''


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
        "load_checkpoint": load_checkpoint,
    }
    exec(code, namespace)

    return namespace


BRIDGE_PARTS = joined(
    GPT_SETTINGS,
    MODEL_PART,
    LOAD_PART,
    LORA_LAYER_PART,
    EVERYWHERE_SOLUTION,
    SIGNAL_PART,
    ENCODER_PART,
    PROJECTOR_SOLUTION,
    BRIDGE_SOLUTION,
    ROWS_PART,
    STAGES_SOLUTION,
    WRITE_PART,
    NEAREST_SOLUTION,
    PROBE_SOLUTION,
)

BRIDGE_REFERENCE = namespace_of(BRIDGE_PARTS)
MEASURE_THREADS = 4


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# bridge1
# ---------------------------------------------------------------------------

BRIDGE1_LESSON = prose(
    """
    Your GPT reads characters. This chapter teaches it to read a signal of 64
    numbers, as LLaVA taught a language model to read images.

    LLaVA joined two trained models, CLIP's image encoder and the Vicuna
    language model, with one linear layer between them. That layer maps each
    image feature into the language model's word embeddings, so the image
    arrives as extra tokens beside the text's. LLaVA-1.5 swapped the layer
    for a two layer MLP and scored higher.

    Here the frozen encoder is a small convolution stack. It was trained in a
    few seconds the way autoencoder1 trained its codes, then frozen. It turns
    each window of 8 steps into one token of 32 numbers.

    ```python
    SignalEncoder()(torch.zeros(5, 64)).shape    # torch.Size([5, 8, 32])
    ```

    `Projector` maps each token's 32 numbers to the GPT's 128, as
    LLaVA-1.5's MLP does. `projected_tokens` runs the frozen encoder without
    recording for autograd, then the projector, which trains.

    `prefix_forward` lays the projected tokens before the text's embeddings,
    so the GPT reads 8 vectors it never saw, then characters. Positions count
    across both, and the scores come back for the text's places alone.

    | Part | Weights | Trained |
    |---|---|---|
    | `SignalEncoder` | 13,568 | before, then frozen |
    | `Projector` | 20,736 | in both stages |
    | Your GPT | 818,048 | frozen in stage one, LoRA in stage two |

    The bridge is 2.5% the size of the GPT. bridge2 trains it.
    """,
)

BRIDGE_IMPORTS = '''
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import load_checkpoint
'''

BRIDGE1_GIVEN = '''
''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, and your model from gpt3 read back.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


    # The new modality, and the frozen encoder that reads it.
''' + indented(SIGNAL_PART) + '''


''' + indented(ENCODER_PART) + '''
'''

BRIDGE1_SOURCE = source_file(
    '''
    # bridge1.py
''' + BRIDGE_IMPORTS + '''
    smooth = nn.GELU()(torch.tensor([-1.0, 0.0, 1.0]))    # tensor([-0.1587,  0.0000,  0.8413])
    joined = torch.cat([torch.zeros(2, 3, 4), torch.ones(2, 5, 4)], dim=1).shape    # torch.Size([2, 8, 4])
''' + BRIDGE1_GIVEN + '''

    # TODO: Map each token of width numbers to size numbers, as LLaVA-1.5's
    #   two layer MLP does. Make self.fc, a Linear from width to size, then
    #   self.gelu, an nn.GELU(), then self.proj, a Linear from size to size, in
    #   that order. forward runs a token batch through all three.
    class Projector(nn.Module):
        def __init__(
            self,
            width: int,
            size: int,
        ) -> None:
            super().__init__()
            ...

        def forward(self, tokens: Tensor) -> Tensor:
            ...


    # TODO: Return projector of encoder's tokens for signals. Run encoder
    #   without recording for autograd, since it stays frozen, and projector
    #   with recording on, since it trains.
    def projected_tokens(
        encoder: nn.Module,
        projector: nn.Module,
        signals: Tensor,
    ) -> Tensor:
        ...


    # TODO: Run gpt on prefix, then ids. Join prefix before gpt.wte(ids) along
    #   the positions, add gpt.wpe of every position, made on ids's device, and
    #   run each block of gpt.h, then gpt.ln_f and gpt.lm_head. Return the
    #   scores at the ids' places only, dropping the prefix's.
    def prefix_forward(
        gpt: nn.Module,
        prefix: Tensor,
        ids: Tensor,
    ) -> Tensor:
        ...
''',
)

BRIDGE1_SOLUTION = source_file(
    '''
    # bridge1.py, a solution for comparison
''' + BRIDGE_IMPORTS + '''
    smooth = nn.GELU()(torch.tensor([-1.0, 0.0, 1.0]))
    joined = torch.cat([torch.zeros(2, 3, 4), torch.ones(2, 5, 4)], dim=1).shape
''' + BRIDGE1_GIVEN + '''

''' + indented(PROJECTOR_SOLUTION) + '''


''' + indented(BRIDGE_SOLUTION) + '''
''',
)


def small_gpt(seed: int) -> nn.Module:
    """A GPT of 31 characters, 2 blocks of size 32, the size checks run fast at."""
    torch.manual_seed(seed)

    return BRIDGE_REFERENCE["GPT"](31, 24, 32, 2, 4)


def made_projector(
    found: type,
    width: int,
    size: int,
) -> nn.Module:
    made = found(width, size)
    call = f"Projector({width}, {size})"
    reject_missing_return(
        call=call,
        actual=made,
    )
    layers = dict(made.named_children())

    for name, kind, shape in (("fc", nn.Linear, (size, width)), ("gelu", nn.GELU, None), ("proj", nn.Linear, (size, size))):
        if not isinstance(layers.get(name), kind):
            raise Mismatch(f"{call} should hold self.{name}, an nn.{kind.__name__}, but it holds {', '.join(layers) or 'nothing'}.")

        if shape is not None and tuple(layers[name].weight.shape) != shape:
            raise Mismatch(f"{call}'s self.{name} has a weight of shape {tuple(layers[name].weight.shape)}, but it should be {shape}.")

    return made


def check_projector_init(found: type) -> None:
    for width, size in ((32, 128), (6, 10)):
        made = made_projector(found, width, size)
        expect_close(
            call=f"the weights of Projector({width}, {size})",
            actual=float(sum(parameter.numel() for parameter in made.parameters())),
            expected=float(width * size + 2 * size + size * size),
        )


def check_projector_forward(found: type) -> None:
    for seed, width, size, shape in ((901, 32, 128, (2, 8, 32)), (902, 6, 10, (3, 5, 6))):
        torch.manual_seed(seed)
        reference = BRIDGE_REFERENCE["Projector"](width, size)
        made = made_projector(found, width, size)
        made.load_state_dict(reference.state_dict())
        tokens = torch.randn(shape, generator=torch.Generator().manual_seed(seed))

        with torch.no_grad():
            expect_tensor(
                call=f"Projector({width}, {size}) of tokens of shape {shape}",
                actual=made(tokens),
                expected=reference(tokens),
            )


def check_projected_tokens(found: object) -> None:
    for seed, shape in ((903, (2, 64)), (904, (3, 64))):
        torch.manual_seed(seed)
        encoder = BRIDGE_REFERENCE["SignalEncoder"]().requires_grad_(False)
        projector = BRIDGE_REFERENCE["Projector"](32, 16)
        modes: dict[str, bool] = {}

        def noting(name: str):
            def hook(
                module: nn.Module,
                inputs: tuple[Tensor, ...],
            ) -> None:
                modes.setdefault(name, torch.is_grad_enabled())

            return hook

        encoder.register_forward_pre_hook(noting("encoder"))
        projector.register_forward_pre_hook(noting("projector"))
        signals = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
        actual = found(encoder, projector, signals)
        call = f"projected_tokens of signals of shape {shape}"
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if "encoder" not in modes or "projector" not in modes:
            raise Mismatch(f"{call} should run both the encoder and the projector.")

        if modes["encoder"]:
            raise Mismatch(f"{call} ran the encoder while recording for autograd. Run it inside torch.no_grad(), since it stays frozen.")

        if not modes["projector"]:
            raise Mismatch(f"{call} ran the projector without recording for autograd, so it could never train. Run it outside no_grad.")

        expect_tensor(
            call=call,
            actual=actual.detach(),
            expected=BRIDGE_REFERENCE["projected_tokens"](encoder, projector, signals).detach(),
        )


def check_prefix_forward(found: object) -> None:
    for seed, prefix_length, length in ((905, 3, 5), (906, 0, 7), (907, 8, 4)):
        gpt = small_gpt(seed)
        generator = torch.Generator().manual_seed(seed)
        prefix = torch.randn(2, prefix_length, 32, generator=generator)
        ids = torch.randint(0, 31, (2, length), generator=generator)
        call = f"prefix_forward of a small GPT, a prefix of {prefix_length} vectors and {length} ids"

        with torch.no_grad():
            actual = found(gpt, prefix, ids)
            expected = BRIDGE_REFERENCE["prefix_forward"](gpt, prefix, ids)

        reject_missing_return(
            call=call,
            actual=actual,
        )
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )

    gpt = small_gpt(908).to("meta")
    places: list[str] = []
    gpt.wpe.register_forward_pre_hook(lambda module, inputs: places.append(inputs[0].device.type))

    try:
        found(gpt, torch.zeros(1, 2, 32, device="meta"), torch.zeros(1, 3, dtype=torch.long, device="meta"))
    except (RuntimeError, NotImplementedError):
        pass

    if places and places[0] != "meta":
        raise Mismatch(
            "prefix_forward made its positions on the CPU, so on a GPU the GPT crashes there. Make them with "
            "device=ids.device.",
        )


def check_bridge1(checker: Checker) -> None:
    checker.test_method(
        class_name="Projector",
        method="__init__",
        description="holds fc, gelu and proj, from width to size",
        run=check_projector_init,
    )
    checker.test_method(
        class_name="Projector",
        method="forward",
        description="runs a token batch through fc, gelu and proj",
        run=check_projector_forward,
    )
    checker.test(
        function="projected_tokens",
        description="runs the frozen encoder without autograd, then the projector",
        run=check_projected_tokens,
    )
    checker.test(
        function="prefix_forward",
        description="reads the prefix then the ids, and scores the ids' places",
        run=check_prefix_forward,
    )


BRIDGE1 = Exercise(
    name="bridge1",
    chapter=NEW_MODALITY,
    title="A projector into your GPT",
    lesson=BRIDGE1_LESSON,
    source=BRIDGE1_SOURCE,
    solution=BRIDGE1_SOLUTION,
    hints={
        "Projector.__init__": "self.fc = nn.Linear(width, size), self.gelu = nn.GELU(), self.proj = nn.Linear(size, size).",
        "Projector.forward": "self.proj of self.gelu of self.fc of tokens.",
        "projected_tokens": "Inside torch.no_grad(), tokens = encoder(signals). Outside it, return projector(tokens).",
        "prefix_forward": "x = torch.cat of prefix and gpt.wte(ids) on dim 1. Add gpt.wpe of torch.arange(x.shape[1], "
        "device=ids.device). Loop over gpt.h, then gpt.lm_head(gpt.ln_f(x)), and slice off the first prefix.shape[1] "
        "places with [:, prefix.shape[1] :].",
    },
    introduces=(),
    check=check_bridge1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# bridge2
# ---------------------------------------------------------------------------

BRIDGE2_GIVEN = BRIDGE1_GIVEN + '''

    # bridge1's projector and prefix_forward.
''' + indented(PROJECTOR_SOLUTION) + '''


''' + indented(BRIDGE_SOLUTION) + '''


    # The LoRA layer from lora1, and add_lora_everywhere from lora3.
''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


    # The batches both stages train on.
''' + indented(ROWS_PART) + '''
'''

BRIDGE2_LESSON = prose(
    """
    LLaVA trained its bridge in two stages. Stage one froze the encoder and
    the language model. It trained the projection alone on about 595,000
    image and caption pairs, at a rate of 2e-3. Stage two kept the encoder
    frozen and trained the projection and the language model together, on
    158,000 instruction examples. Both scored the answer's tokens alone, as
    sft1's loss did.

    Here both stages train on captions, the signal's three facts in words.

    ```text
    Describe it: rises, three waves, spike early
    ```

    `stage_one` freezes every weight of the GPT and hands AdamW the projector
    alone. `stage_two` adds rank 8 LoRA on every Linear, as lora3 did, in
    place of training the whole GPT. Then it hands AdamW the projector and
    the LoRA weights. `bridge_step` runs the signals through the bridge and
    the GPT, and trains on the caption's loss.

    Here are captions written for 300 new signals, over 3 seeds.

    | After | Rises | Waves | Spike's third | Whole caption |
    |---|---|---|---|---|
    | Stage one, 300 steps | 59% to 92% | 0% | 0% | 0% |
    | Stage two, 900 more | 100% | 99% to 100% | 98% | 97% to 98% |

    With the GPT frozen, 300 steps teach the projector to steer it to rises
    or falls, and no further. Your GPT never wrote a wave count in
    Shakespeare, and steering it there through 8 tokens alone is slow. Stage
    two lets LoRA change the GPT a little, and every fact comes through. That
    is why LLaVA's second stage trains the language model too.
    """,
)

BRIDGE2_SOURCE = source_file(
    '''
    # bridge2.py
''' + BRIDGE_IMPORTS + '''
    frozen = nn.Linear(2, 2).requires_grad_(False).weight.requires_grad    # False
    picked = torch.optim.AdamW(nn.Linear(2, 3).parameters(), lr=2e-3).param_groups[0]["lr"]    # 0.002
''' + BRIDGE2_GIVEN + '''

    # TODO: Stage one. Freeze every weight of gpt, then return an AdamW over
    #   the projector's weights alone, at a rate of 2e-3, as LLaVA's stage one
    #   used, with a weight_decay of 0.
    def stage_one(
        gpt: nn.Module,
        projector: nn.Module,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Stage two. Put rank 8 LoRA with an alpha of 8 on every Linear of
    #   gpt with add_lora_everywhere, then move gpt to the device its
    #   gpt.wte.weight sits on, so the new A and B join it there. Return an
    #   AdamW over the projector's weights and every weight of gpt that now
    #   trains, at 1e-3, with a weight_decay of 0.
    def stage_two(
        gpt: nn.Module,
        projector: nn.Module,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Take one step on signals, x and y, each moved to device. The scores
    #   are prefix_forward of gpt on projected_tokens of the signals, then x.
    #   The loss is their cross entropy against y, which skips -100. Zero the
    #   old gradients, backpropagate, step, and return the loss as a plain
    #   number.
    def bridge_step(
        gpt: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
        optimizer: torch.optim.Optimizer,
        signals: Tensor,
        x: Tensor,
        y: Tensor,
        device: torch.device,
    ) -> float:
        ...
''',
)

BRIDGE2_SOLUTION = source_file(
    '''
    # bridge2.py, a solution for comparison
''' + BRIDGE_IMPORTS + '''
    frozen = nn.Linear(2, 2).requires_grad_(False).weight.requires_grad
    picked = torch.optim.AdamW(nn.Linear(2, 3).parameters(), lr=2e-3).param_groups[0]["lr"]
''' + BRIDGE2_GIVEN + '''

''' + indented(STAGES_SOLUTION) + '''
''',
)


def optimized(optimizer: object) -> list[Tensor]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def expect_optimizer(
    call: str,
    optimizer: object,
    wanted: list[Tensor],
    lr: float,
) -> None:
    reject_missing_return(
        call=call,
        actual=optimizer,
    )

    if not isinstance(optimizer, torch.optim.AdamW):
        raise Mismatch(f"{call} gives {value_text(optimizer)}, but it should give a torch.optim.AdamW.")

    held = {id(parameter) for parameter in optimized(optimizer)}
    expected = {id(parameter) for parameter in wanted}

    if held - expected:
        raise Mismatch(f"{call}'s AdamW holds {len(held - expected)} weights it shouldn't train. It should hold the trained ones alone.")

    if expected - held:
        raise Mismatch(f"{call}'s AdamW leaves out {len(expected - held)} weights that should train.")

    for group in optimizer.param_groups:
        expect_close(
            call=f"the rate of {call}'s AdamW",
            actual=group["lr"],
            expected=lr,
        )
        expect_close(
            call=f"the weight_decay of {call}'s AdamW",
            actual=group["weight_decay"],
            expected=0.0,
        )


def check_stage_one(found: object) -> None:
    for seed in (911, 912):
        gpt = small_gpt(seed)
        projector = BRIDGE_REFERENCE["Projector"](32, 32)
        optimizer = found(gpt, projector)
        call = "stage_one of a small GPT and a projector"

        if any(parameter.requires_grad for parameter in gpt.parameters()):
            raise Mismatch(f"{call} left some of the GPT's weights training. Freeze all of them with gpt.requires_grad_(False).")

        expect_optimizer(
            call=call,
            optimizer=optimizer,
            wanted=list(projector.parameters()),
            lr=2e-3,
        )


def lora_layers(gpt: nn.Module) -> list[nn.Module]:
    return [module for module in gpt.modules() if type(module).__name__ == "LoRALinear"]


def check_stage_two(found: object) -> None:
    for seed in (913, 914):
        gpt = small_gpt(seed)
        projector = BRIDGE_REFERENCE["Projector"](32, 32)
        call = "stage_two of a small GPT and a projector"
        optimizer = found(gpt, projector)
        layers = lora_layers(gpt)

        if len(layers) != 4 * len(gpt.h):
            raise Mismatch(f"{call} put LoRA on {len(layers)} Linears, but it should wrap all 4 of each block's, {4 * len(gpt.h)} in all.")

        for layer in layers:
            if layer.A.shape[0] != 8 or layer.scale != 1.0:
                raise Mismatch(f"{call} made LoRA of rank {layer.A.shape[0]} with a scale of {layer.scale}, but it should be rank 8 with an alpha of 8, so a scale of 1.")

        expect_optimizer(
            call=call,
            optimizer=optimizer,
            wanted=[*projector.parameters(), *[parameter for parameter in gpt.parameters() if parameter.requires_grad]],
            lr=1e-3,
        )

    gpt = small_gpt(915).to("meta")
    found(gpt, BRIDGE_REFERENCE["Projector"](32, 32).to("meta"))
    stray = [name for name, parameter in gpt.named_parameters() if parameter.device.type != "meta"]

    if stray:
        raise Mismatch(
            f"stage_two left the new LoRA weights on the CPU while the GPT sits elsewhere, such as {stray[0]}. On a "
            "GPU that crashes. Move gpt to gpt.wte.weight.device after adding LoRA.",
        )


def bridge_setup(seed: int) -> tuple[nn.Module, nn.Module, nn.Module, list[tuple[Tensor, Tensor, Tensor]]]:
    """A small GPT with LoRA, an encoder and a projector, and two batches of signals with their x and y."""
    gpt = BRIDGE_REFERENCE["add_lora_everywhere"](small_gpt(seed), 2, 2.0)
    torch.manual_seed(seed)
    encoder = BRIDGE_REFERENCE["SignalEncoder"]().requires_grad_(False)
    projector = BRIDGE_REFERENCE["Projector"](32, 32)
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for layer in lora_layers(gpt):
            layer.B.copy_(torch.randn(layer.B.shape, generator=generator) * 0.1)

    batches = []

    for size in (3, 2):
        signals = torch.randn(size, 64, generator=generator)
        x = torch.randint(0, 31, (size, 6), generator=generator)
        y = torch.randint(0, 31, (size, 6), generator=generator)
        y[:, :2] = -100
        batches.append((signals, x, y))

    return gpt, encoder, projector, batches


def check_bridge_step(found: object) -> None:
    for seed in (916, 917):
        results = []

        for step in (found, BRIDGE_REFERENCE["bridge_step"]):
            gpt, encoder, projector, batches = bridge_setup(seed)
            trained = [*projector.parameters(), *[parameter for parameter in gpt.parameters() if parameter.requires_grad]]
            optimizer = torch.optim.AdamW(trained, lr=0.01, weight_decay=0.0)
            # Two steps, since gradients left from a first step only show in a second.
            losses = [step(gpt, encoder, projector, optimizer, signals, x, y, torch.device("cpu")) for signals, x, y in batches]
            results.append((losses, projector.fc.weight.detach().clone()))

        (losses, weight), (expected_losses, expected_weight) = results

        for place, (loss, expected) in enumerate(zip(losses, expected_losses)):
            expect_close(
                call=f"bridge_step's loss on a small GPT, step {place + 1}",
                actual=loss,
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="the projector's fc weight after two calls of bridge_step",
            actual=weight,
            expected=expected_weight,
        )

    gpt, encoder, projector, batches = bridge_setup(918)
    gpt, encoder, projector = gpt.to("meta"), encoder.to("meta"), projector.to("meta")
    signals, x, y = batches[0]
    optimizer = torch.optim.SGD([*projector.parameters()], lr=0.1)

    try:
        found(gpt, encoder, projector, optimizer, signals, x, y, torch.device("meta"))
    except (RuntimeError, NotImplementedError) as error:
        message = str(error)

        if "meta tensor" not in message and "device" in message:
            raise Mismatch(
                "bridge_step crashed with the models on another device than the CPU, as main runs them on a GPU. Move "
                f"signals, x and y to device before they meet the models.\n\nThe original message, {message}",
            ) from None


def check_bridge2(checker: Checker) -> None:
    checker.test(
        function="stage_one",
        description="freezes the GPT and trains the projector alone at 2e-3",
        run=check_stage_one,
    )
    checker.test(
        function="stage_two",
        description="adds LoRA everywhere and trains it with the projector at 1e-3",
        run=check_stage_two,
    )
    checker.test(
        function="bridge_step",
        description="trains on the answer's loss after the projected signal, on device",
        run=check_bridge_step,
    )


# ---------------------------------------------------------------------------
# bridge3
# ---------------------------------------------------------------------------

BRIDGE3_GIVEN = BRIDGE2_GIVEN + '''

    # bridge2's stages and step.
''' + indented(STAGES_SOLUTION) + '''
'''

BRIDGE3_SOURCE = source_file(
    '''
    # bridge3.py
''' + BRIDGE_IMPORTS + '''
    alike = F.normalize(torch.tensor([[3.0, 4.0]]), dim=-1)    # tensor([[0.6000, 0.8000]])
    best, place = torch.tensor([[0.2, 0.9, 0.5]]).max(dim=-1)    # tensor([0.9000]), tensor([1])
''' + BRIDGE3_GIVEN + '''

    # TODO: For each vector of vectors, of shape (n, size), find the character
    #   whose row of gpt.wte.weight points the most the same way, by cosine
    #   similarity, the dot product of the two rows scaled to length 1. Return
    #   those characters as a list, and their similarities as a tensor of n.
    def nearest_characters(
        gpt: nn.Module,
        vectors: Tensor,
        characters: list[str],
    ) -> tuple[list[str], Tensor]:
        ...


    # TODO: Fit a linear probe, as masked2 described one, then score it. After
    #   torch.manual_seed(0), make an nn.Linear from the features' width to
    #   classes. Train it 300 steps with Adam at 0.01 on the cross entropy of
    #   the training features against their labels, the whole set each step.
    #   Return the share of the test features it labels right, as a plain
    #   number, scored without recording for autograd.
    def probe_accuracy(
        train_features: Tensor,
        train_labels: Tensor,
        test_features: Tensor,
        test_labels: Tensor,
        classes: int,
    ) -> float:
        ...
''',
)

BRIDGE3_SOLUTION = source_file(
    '''
    # bridge3.py, a solution for comparison
''' + BRIDGE_IMPORTS + '''
    alike = F.normalize(torch.tensor([[3.0, 4.0]]), dim=-1)
    best, place = torch.tensor([[0.2, 0.9, 0.5]]).max(dim=-1)
''' + BRIDGE3_GIVEN + '''

''' + indented(NEAREST_SOLUTION) + '''


''' + indented(PROBE_SOLUTION) + '''
''',
)

NEAREST_NAMES = Described(
    [chr(ord("a") + place) for place in range(31)],
    "31 characters, a onward",
)


def check_nearest_characters(found: object) -> None:
    for seed, count in ((921, 5), (922, 3)):
        gpt = small_gpt(seed)
        vectors = torch.randn(count, 32, generator=torch.Generator().manual_seed(seed)) * 5
        # One vector is a character's own row, so its nearest must be that character at a cosine of 1.
        vectors[0] = gpt.wte.weight[seed % 31].detach() * 3
        call = f"nearest_characters of {count} vectors for a small GPT"

        with torch.no_grad():
            actual = found(gpt, vectors, NEAREST_NAMES.value)
            names, cosines = BRIDGE_REFERENCE["nearest_characters"](gpt, vectors, NEAREST_NAMES.value)

        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should give two things, the characters and their similarities, but it gives {value_text(actual)}.")

        if list(actual[0]) != names:
            raise Mismatch(f"{call} gives the characters {list(actual[0])}, but they should be {names}.")

        expect_tensor(
            call=f"the similarities from {call}",
            actual=actual[1].detach() if isinstance(actual[1], Tensor) else actual[1],
            expected=cosines,
        )


def probe_data(
    seed: int,
    width: int,
    classes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, width, generator=generator) * 2
    labels = torch.randint(0, classes, (90,), generator=generator)
    features = centers[labels] + torch.randn(90, width, generator=generator) * 1.5

    return features[:60], labels[:60], features[60:], labels[60:]


def check_probe_accuracy(found: object) -> None:
    for seed, width, classes in ((923, 6, 3), (924, 10, 4)):
        data = probe_data(seed, width, classes)
        recording: list[bool] = []
        original = nn.Linear.forward

        def noting(
            module: nn.Linear,
            x: Tensor,
        ) -> Tensor:
            if x is data[2]:
                recording.append(torch.is_grad_enabled())

            return original(module, x)

        nn.Linear.forward = noting

        try:
            actual = found(*data, classes)
        finally:
            nn.Linear.forward = original

        call = f"probe_accuracy on {len(data[0])} training rows of width {width}, {classes} classes"

        if any(recording):
            raise Mismatch(f"{call} scored the test features while recording for autograd. Score inside torch.no_grad().")

        expect_close(
            call=call,
            actual=actual,
            expected=BRIDGE_REFERENCE["probe_accuracy"](*data, classes),
        )


def check_bridge3(checker: Checker) -> None:
    checker.test(
        function="nearest_characters",
        description="finds each vector's nearest character by cosine",
        run=check_nearest_characters,
    )
    checker.test(
        function="probe_accuracy",
        description="trains a linear probe 300 steps and scores it",
        run=check_probe_accuracy,
    )


BRIDGE2 = Exercise(
    name="bridge2",
    chapter=NEW_MODALITY,
    title="LLaVA's two stages",
    lesson=BRIDGE2_LESSON,
    source=BRIDGE2_SOURCE,
    solution=BRIDGE2_SOLUTION,
    hints={
        "stage_one": "gpt.requires_grad_(False), then torch.optim.AdamW(projector.parameters(), lr=2e-3, "
        "weight_decay=0.0).",
        "stage_two": "add_lora_everywhere(gpt, 8, 8.0).to(gpt.wte.weight.device). The LoRA weights are the gpt "
        "parameters whose requires_grad is True. Hand AdamW the projector's parameters and those, at lr=1e-3.",
        "bridge_step": "scores = prefix_forward(gpt, projected_tokens(encoder, projector, signals.to(device)), "
        "x.to(device)). Then the cross entropy of the flattened scores and y.to(device), zero_grad, backward, step "
        "and loss.item().",
    },
    introduces=(),
    check=check_bridge2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)

BRIDGE3_LESSON = prose(
    """
    After stage one, each signal reaches your GPT as 8 projected tokens. What
    are they, and what do they carry?

    `nearest_characters` compares each token with every row of the GPT's
    embedding table by cosine similarity, as word2vec3 compared word rows.

    ```python
    F.normalize(torch.tensor([[3.0, 4.0]]), dim=-1)    # tensor([[0.6000, 0.8000]])
    ```

    Here are the tokens of 1,500 new signals after 300 steps of stage one,
    over 3 seeds, next to the GPT's own character rows.

    | Vectors | Cosine with the nearest character | Length |
    |---|---|---|
    | Projected tokens | 0.22 to 0.25 | 7.5 to 8.8 |
    | Character rows | 0.43, to the nearest other character | 1.10 |

    The tokens point further from every character than the characters sit
    from each other, and run about 7 times longer. They are no characters in
    disguise. The GPT learns to read a new kind of input, outside its
    vocabulary.

    `probe_accuracy` fits a linear probe, as masked2 described one, and scores
    it on signals it never saw. Here are probes for each fact, trained on
    1,000 signals and scored on 500 others.

    | Read from | Rises | Waves | Spike's third |
    |---|---|---|---|
    | The 8 projected tokens | 100% | 100% | 100% |
    | The signal as 64 letters | 100% | 100% | 75% to 79% |

    The letters are the signal as text, each step one of 26 letters by its
    level, read here as one hot rows. Both carry the trend and the wave
    count. The projected tokens also lay the spike's third out for one
    linear map, in an eighth of the positions.
    """,
)

BRIDGE3 = Exercise(
    name="bridge3",
    chapter=NEW_MODALITY,
    title="What the projected tokens carry",
    lesson=BRIDGE3_LESSON,
    source=BRIDGE3_SOURCE,
    solution=BRIDGE3_SOLUTION,
    hints={
        "nearest_characters": "F.normalize both vectors and gpt.wte.weight along dim -1, then multiply the first by "
        "the second's .T. .max(dim=-1) gives the best similarity and its place. Look each place up in characters.",
        "probe_accuracy": "After torch.manual_seed(0), probe = nn.Linear(train_features.shape[1], classes) and Adam at "
        "0.01. 300 steps of cross entropy on the whole training set. Then, inside torch.no_grad(), the mean of "
        "argmax(dim=1) == test_labels as floats, with .item().",
    },
    introduces=(),
    check=check_bridge3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def weights_of(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def measure_sizes() -> dict[str, int]:
    """bridge1's table: the weights of the encoder, the projector, the GPT, and the LoRA bridge2 adds to it."""
    gpt = BRIDGE_REFERENCE["GPT"](65, 128, 128, 4, 4)
    sizes = {
        "encoder": weights_of(BRIDGE_REFERENCE["SignalEncoder"]()),
        "projector": weights_of(BRIDGE_REFERENCE["Projector"](32, 128)),
        "gpt": weights_of(gpt),
    }
    BRIDGE_REFERENCE["add_lora_everywhere"](gpt, 8, 8.0)
    sizes["lora"] = sum(parameter.numel() for parameter in gpt.parameters() if parameter.requires_grad)

    return sizes


def shakespeare_positions() -> tuple[list[str], dict[str, int]]:
    characters = sorted(set(load_shakespeare()))

    return characters, {character: place for place, character in enumerate(characters)}


def caption_scores(
    gpt: nn.Module,
    encoder: nn.Module,
    projector: nn.Module,
    count: int,
    seed: int,
) -> dict[str, float]:
    """The share of count new signals whose caption is written whole and right, and the share right on each fact."""
    characters, positions = shakespeare_positions()
    signals, rises, waves, thirds = BRIDGE_REFERENCE["make_signals"](count, torch.Generator().manual_seed(seed))
    device = next(projector.parameters()).device
    written = BRIDGE_REFERENCE["write_after"](
        gpt,
        encoder,
        projector,
        signals.to(device),
        BRIDGE_REFERENCE["CAPTION_PROMPT"],
        positions,
        characters,
    )
    truths = [BRIDGE_REFERENCE["caption"](*facts) for facts in zip(rises.tolist(), waves.tolist(), thirds.tolist())]
    facts = {"rises": [], "waves": [], "spike": []}

    for text, truth in zip(written, truths):
        said = (text.split(", ") + ["", "", ""])[:3]
        right = truth.split(", ")

        for name, guess, wanted in zip(facts, said, right):
            facts[name].append(guess == wanted)

    scores = {name: sum(marks) / count for name, marks in facts.items()}
    scores["whole"] = sum(text == truth for text, truth in zip(written, truths)) / count

    return scores


def bridge_models(
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    encoder = BRIDGE_REFERENCE["pretrained_encoder"](seed).to(device)
    gpt = BRIDGE_REFERENCE["trained_gpt"]().to(device)
    torch.manual_seed(seed)
    projector = BRIDGE_REFERENCE["Projector"](32, 128).to(device)

    return encoder, gpt, projector


def train_captions(
    gpt: nn.Module,
    encoder: nn.Module,
    projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
    generator: torch.Generator,
) -> None:
    _, positions = shakespeare_positions()
    device = next(projector.parameters()).device

    for _ in range(steps):
        signals, x, y = BRIDGE_REFERENCE["caption_batch"](32, positions, generator)
        BRIDGE_REFERENCE["bridge_step"](gpt, encoder, projector, optimizer, signals, x, y, device)


def measure_stages(
    seeds: range = range(3),
    stage_one_steps: int = 300,
    stage_two_steps: int = 900,
    device: str = "mps",
) -> dict[str, list[dict[str, float]]]:
    """bridge2's table: the captions written after stage one, the projector alone, and after stage two, LoRA and the
    projector, scored on 300 new signals, for each seed."""
    results: dict[str, list[dict[str, float]]] = {}
    place = torch.device(device)

    with pinned_threads():
        for seed in seeds:
            encoder, gpt, projector = bridge_models(seed, place)
            generator = torch.Generator().manual_seed(seed + 7)
            optimizer = BRIDGE_REFERENCE["stage_one"](gpt, projector)
            train_captions(gpt, encoder, projector, optimizer, stage_one_steps, generator)
            results.setdefault("stage one", []).append(caption_scores(gpt, encoder, projector, 300, 999))
            optimizer = BRIDGE_REFERENCE["stage_two"](gpt, projector)
            train_captions(gpt, encoder, projector, optimizer, stage_two_steps, generator)
            results.setdefault("stage two", []).append(caption_scores(gpt, encoder, projector, 300, 999))

    return results


def letter_features(signals: Tensor) -> Tensor:
    """Each signal as the text a GPT would read, every step one of 26 letters by its level between the signal's
    lowest and highest, flattened as one hot rows."""
    lowest = signals.min(dim=1, keepdim=True).values
    highest = signals.max(dim=1, keepdim=True).values
    levels = ((signals - lowest) / (highest - lowest) * 26).long().clamp(0, 25)

    return F.one_hot(levels, 26).flatten(1).float()


def measure_tokens(seeds: range = range(3)) -> dict[str, list[float]]:
    """bridge3's tables, after 300 steps of stage one on the CPU: how near each projected token sits to its nearest
    character and how long it is, the same for the characters themselves, and linear probes for each fact from the
    projected tokens and from the letters, trained on 1,000 signals and scored on 500 others."""
    characters, _ = shakespeare_positions()
    results: dict[str, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            encoder, gpt, projector = bridge_models(seed, torch.device("cpu"))
            generator = torch.Generator().manual_seed(seed + 7)
            train_captions(gpt, encoder, projector, BRIDGE_REFERENCE["stage_one"](gpt, projector), 300, generator)
            signals, rises, waves, thirds = BRIDGE_REFERENCE["make_signals"](1500, torch.Generator().manual_seed(seed + 500))

            with torch.no_grad():
                tokens = BRIDGE_REFERENCE["projected_tokens"](encoder, projector, signals)
                _, best = BRIDGE_REFERENCE["nearest_characters"](gpt, tokens.reshape(-1, 128), characters)
                table = F.normalize(gpt.wte.weight, dim=-1)
                others = (table @ table.T).fill_diagonal_(-1.0).max(dim=1).values

            results.setdefault("token cosine", []).append(best.mean().item())
            results.setdefault("token length", []).append(tokens.norm(dim=-1).mean().item())
            results.setdefault("character cosine", []).append(others.mean().item())
            results.setdefault("character length", []).append(gpt.wte.weight.norm(dim=-1).mean().item())

            for name, features in (("tokens", tokens.flatten(1)), ("letters", letter_features(signals))):
                for fact, labels, classes in (("rises", rises, 2), ("waves", waves - 1, 4), ("third", thirds, 3)):
                    score = BRIDGE_REFERENCE["probe_accuracy"](features[:1000], labels[:1000], features[1000:], labels[1000:], classes)
                    results.setdefault(f"{name} {fact}", []).append(score)

    return results
