"""54_model_types, encoder only, decoder only and encoder decoder models and the masks that set them apart, then a decoder only language model end to end."""

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
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

MODEL_TYPES = Chapter(
    folder="54_model_types",
    title="Encoders, decoders and both",
    readme=source_file(
        """
        # Encoders, decoders and both

        The same block builds three kinds of model, and their masks set them
        apart. This chapter names the three, shows what each mask lets every
        output reach, and then builds the kind behind GPT and Llama, a decoder
        only language model, from tokens to generated text.

        ## Further reading

        - [Vaswani and colleagues, attention is all you need](https://arxiv.org/abs/1706.03762)
        - [Devlin and colleagues, BERT](https://arxiv.org/abs/1810.04805)
        - [Raffel and colleagues, T5, a unified text to text model](https://arxiv.org/abs/1910.10683)
        - [Radford and colleagues, GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# ---------------------------------------------------------------------------
# encoder1
# ---------------------------------------------------------------------------

MASKED_ATTENTION_PART = '''
# One head of attention that looks only where mask is True, as in mask1.
class MaskedAttention(nn.Module):
    def __init__(
        self,
        size: int,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(size, size, bias=False)
        self.key = nn.Linear(size, size, bias=False)
        self.value = nn.Linear(size, size, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
        scores = self.query(x) @ self.key(x).transpose(-2, -1) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)

        return weights @ self.value(x)
'''

ENCODER1_LESSON = prose(
    """
    The block of block1 builds three kinds of model. What sets them apart is
    where each attention layer may look, its mask. A mask is a table of True
    and False, one row per query and one column per key, True where that query
    may look.

    | Kind | Masks | What it does | Examples |
    |---|---|---|---|
    | encoder only | none, every position sees every other | reads a whole input, then labels it or each token | BERT |
    | decoder only | causal, each position sees itself and earlier ones | writes one token after another | GPT-2, Llama |
    | encoder decoder | none in the encoder, causal in the decoder, full in the cross | reads an input, then writes an output | the first transformer, T5 |

    ```python
    torch.ones(3, 3, dtype=torch.bool)                # nothing hidden, an encoder's mask
    torch.tril(torch.ones(3, 3, dtype=torch.bool))    # True on and below the diagonal
    ```

    Each kind fits its job. An encoder can't write text a token at a time,
    since every position reads the ones after it. A decoder can't use the end
    of a sentence to read its start. BERT reads the left and the right at
    once, which suits labelling. GPT reads only the left, which suits writing.

    A labelling task shows the gap. Two blocks tag each of 16 random digits
    with whether the digit after it is bigger. Here is the share of tags right
    on new rows after 800 steps, from three seeds.

    | Mask | Tags right |
    |---|---|
    | an encoder's | 100% |
    | a decoder's | 75.3% to 75.5% |

    The decoder never sees the next digit, so it can only guess from the
    current one. A 0 is followed by a bigger digit 9 times in 10, a 9 never,
    and such guesses get 75% at best. When the whole input is there before
    any output, read it with an encoder.

    In an encoder decoder the decoder's queries look at the encoder's keys, as
    cross1 showed. Its cross mask is all True, one row per output position
    and one column per input position.

    The mask shows in what each output can reach. Nudge input j, and output t
    moves only if position t may look at j. Autograd measures that, as causal1
    did. Here is a one layer decoder over 4 positions. Row t is output t, the
    other way round from block1's map, and a 1 marks each input j it reaches.

    ```text
    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1
    ```

    The same layer with an encoder's mask reaches everything. Stacking more
    decoder layers never lets the future in, since no layer looks ahead.
    """,
)

ENCODER1_SOURCE = source_file(
    '''
    # encoder1.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    lower = torch.tril(torch.ones(3, 3, dtype=torch.bool))    # True on and below the diagonal
    everything = torch.ones(2, 4, dtype=torch.bool)           # 2 queries that see 4 keys


''' + indented(MASKED_ATTENTION_PART) + '''


    # TODO: Return the self attention mask of shape (steps, steps) for kind,
    #   "encoder" or "decoder".
    def self_mask(
        kind: str,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return the three masks of an encoder decoder model, in this order:
    #   the encoder's own, of shape (source_steps, source_steps), the decoder's
    #   own, of shape (target_steps, target_steps), and the decoder's cross
    #   mask onto the encoder, of shape (target_steps, source_steps).
    def encoder_decoder_masks(
        source_steps: int,
        target_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...


    # TODO: Return a table of shape (steps, steps), True at [t, j] when output t
    #   of model depends on input j. x has shape (1, steps, features), and model
    #   maps it to the same shape. Find it with autograd, one output at a time.
    def reach(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        ...
    ''',
)

ENCODER1_SOLUTION = source_file(
    '''
    # encoder1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    lower = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    everything = torch.ones(2, 4, dtype=torch.bool)


''' + indented(MASKED_ATTENTION_PART) + '''


    def self_mask(
        kind: str,
        steps: int,
    ) -> Tensor:
        everything = torch.ones(steps, steps, dtype=torch.bool)

        if kind == "decoder":
            return torch.tril(everything)

        return everything


    def encoder_decoder_masks(
        source_steps: int,
        target_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self_mask("encoder", source_steps),
            self_mask("decoder", target_steps),
            torch.ones(target_steps, source_steps, dtype=torch.bool),
        )


    def reach(
        model: nn.Module,
        x: Tensor,
    ) -> Tensor:
        steps = x.shape[1]
        rows = []

        for t in range(steps):
            x = x.detach().requires_grad_()
            model(x)[0, t].sum().backward()
            rows.append(x.grad[0].abs().sum(dim=-1) > 0)

        return torch.stack(rows)
    ''',
)


def reference_self_mask(
    kind: str,
    steps: int,
) -> Tensor:
    everything = torch.ones(steps, steps, dtype=torch.bool)

    if kind == "decoder":
        return torch.tril(everything)

    return everything


def reference_encoder_decoder_masks(
    source_steps: int,
    target_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        reference_self_mask("encoder", source_steps),
        reference_self_mask("decoder", target_steps),
        torch.ones(target_steps, source_steps, dtype=torch.bool),
    )


def reference_reach(
    model: nn.Module,
    x: Tensor,
) -> Tensor:
    rows = []

    for t in range(x.shape[1]):
        x = x.detach().requires_grad_()
        model(x)[0, t].sum().backward()
        rows.append(x.grad[0].abs().sum(dim=-1) > 0)

    return torch.stack(rows)


def masked_attention_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec(MASKED_ATTENTION_PART, namespace)

    return namespace


MASKED_ATTENTION = masked_attention_namespace()["MaskedAttention"]


class MaskedLayers(nn.Module):
    """Attention layers stacked with residuals, each looking only where mask is True."""

    def __init__(
        self,
        size: int,
        layers: int,
        mask: Tensor,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MASKED_ATTENTION(size) for _ in range(layers)])
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(x, self.mask)

        return x


def seeded_layers(
    seed: int,
    size: int,
    layers: int,
    mask: Tensor,
) -> nn.Module:
    torch.manual_seed(seed)

    return MaskedLayers(size, layers, mask)


TAG_DIGITS = 10
TAG_STEPS = 16


class MaskTagger(nn.Module):
    """Two pre norm blocks of MaskedAttention and an MLP, size 32, with an encoder's or a decoder's mask, and a head
    that tags each position, for encoder1's table."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.mask = reference_self_mask(kind, TAG_STEPS)
        self.embed = nn.Embedding(TAG_DIGITS, 32)
        self.positions = nn.Embedding(TAG_STEPS, 32)
        self.layers = nn.ModuleList([MASKED_ATTENTION(32) for _ in range(2)])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32)) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(32) for _ in range(4)])
        self.head = nn.Linear(32, 2)

    def forward(self, digits: Tensor) -> Tensor:
        x = self.embed(digits) + self.positions(torch.arange(TAG_STEPS))

        for index, (attention, mlp) in enumerate(zip(self.layers, self.mlps)):
            x = x + attention(self.norms[2 * index](x), self.mask)
            x = x + mlp(self.norms[2 * index + 1](x))

        return self.head(x)


def bigger_next(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Rows of random digits, and for every digit but the last, 1 when the digit after it is bigger."""
    digits = torch.randint(0, TAG_DIGITS, (count, TAG_STEPS), generator=generator)

    return digits, (digits[:, 1:] > digits[:, :-1]).long()


def measure_mask_tagging(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[float]]:
    """encoder1's table, the share of tags right on 2,000 new rows after 800 steps of Adam at 3e-3 on batches of 64,
    with each mask. Measured at 4 threads."""
    results: dict[str, list[float]] = {}
    digits, tags = bigger_next(2000, torch.Generator().manual_seed(9))

    for kind in ("encoder", "decoder"):
        for seed in seeds:
            torch.manual_seed(seed)
            model = MaskTagger(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            generator = torch.Generator().manual_seed(seed + 50)

            for _ in range(800):
                batch, wanted = bigger_next(64, generator)
                loss = F.cross_entropy(model(batch)[:, :-1].reshape(-1, 2), wanted.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                right = (model(digits)[:, :-1].argmax(dim=-1) == tags).float().mean().item()

            results.setdefault(kind, []).append(right)

    return results


def leaky_mask(steps: int) -> Tensor:
    """A causal mask with one future position let in, so reach has to read the gradients and not the mask's shape."""
    mask = reference_self_mask("decoder", steps)
    mask[0, steps - 1] = True

    return mask


def expect_mask(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    """A mask of 1.0 and 0.0 matches True and False as numbers, but ~ and masked_fill need True and False."""
    if isinstance(actual, Tensor) and actual.dtype != torch.bool:
        raise Mismatch(
            f"{call} holds numbers of dtype {actual.dtype}, but a mask should hold True and False. "
            "Give torch.ones dtype=torch.bool, as the lesson's example does.",
        )

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def expect_three_masks(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 3:
        raise Mismatch(f"{call} should give three masks in a tuple, the encoder's, the decoder's and the cross mask.")

    for name, part, wanted in zip(("the encoder's mask", "the decoder's mask", "the cross mask"), actual, expected):
        expect_mask(
            call=f"{name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_encoder1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="self_mask",
        description="gives an encoder's full mask and a decoder's causal one",
        inputs=(
            ("encoder", 4),
            ("decoder", 5),
            ("decoder", 1),
        ),
        reference=reference_self_mask,
        expect=expect_mask,
    )
    check_on_inputs(
        checker=checker,
        function="encoder_decoder_masks",
        description="gives the encoder's, the decoder's and the cross mask",
        inputs=(
            (5, 3),
            (2, 4),
        ),
        reference=reference_encoder_decoder_masks,
        expect=expect_three_masks,
    )
    x = small_tensor(540, 1, 5, 4) / 3
    check_on_inputs(
        checker=checker,
        function="reach",
        description="finds which inputs each output depends on",
        inputs=(
            (Described(seeded_layers(541, 4, 1, reference_self_mask("decoder", 5)), "a one layer decoder"), x),
            (Described(seeded_layers(542, 4, 2, reference_self_mask("encoder", 5)), "a two layer encoder"), x),
            (Described(seeded_layers(543, 4, 1, leaky_mask(5)), "a decoder whose first position also sees the last"), x),
        ),
        reference=reference_reach,
        expect=expect_tensor,
    )


ENCODER1 = Exercise(
    name="encoder1",
    chapter=MODEL_TYPES,
    title="Encoders, decoders and their masks",
    lesson=ENCODER1_LESSON,
    source=ENCODER1_SOURCE,
    solution=ENCODER1_SOLUTION,
    hints={
        "self_mask": "torch.ones(steps, steps, dtype=torch.bool) for an encoder, and torch.tril of it for a decoder.",
        "encoder_decoder_masks": "Use self_mask for the first two. The cross mask is all True, one row per target "
        "step and one column per source step.",
        "reach": "For each t, make x a fresh leaf with x.detach().requires_grad_(). Call .backward() on the sum of "
        "output t, and keep x.grad[0].abs().sum(dim=-1) > 0 as row t.",
    },
    introduces=(),
    check=check_encoder1,
)


# ---------------------------------------------------------------------------
# decoder1
# ---------------------------------------------------------------------------

CORPUS_PART = '''
SUBJECTS = ["the cat", "the dog", "a bird", "my friend", "the teacher"]
VERBS = ["sees", "likes", "chases", "finds", "draws"]
OBJECTS = ["the ball", "a fish", "the moon", "a red hat", "the garden"]


# A small made up language, three words picked at random in every sentence.
def corpus(
    count: int,
    generator: torch.Generator,
) -> str:
    picks = torch.randint(0, 5, (count, 3), generator=generator).tolist()

    return " ".join(f"{SUBJECTS[s]} {VERBS[v]} {OBJECTS[o]}." for s, v, o in picks)


# count windows of context + 1 tokens each, from random places in data.
def random_windows(
    data: Tensor,
    count: int,
    context: int,
    generator: torch.Generator,
) -> Tensor:
    starts = torch.randint(0, len(data) - context - 1, (count,), generator=generator).tolist()

    return torch.stack([data[start : start + context + 1] for start in starts])
'''

BLOCK_PART = '''
# A pre norm block as in block1, causal attention over heads, then the MLP.
class Block(nn.Module):
    def __init__(
        self,
        size: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(size)
        self.qkv = nn.Linear(size, 3 * size)
        self.out = nn.Linear(size, size)
        self.norm2 = nn.LayerNorm(size)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.GELU(),
            nn.Linear(4 * size, size),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, size = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(size, dim=2)
        q, k, v = (part.view(batch, steps, self.heads, size // self.heads).transpose(1, 2) for part in (q, k, v))
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(mixed.transpose(1, 2).reshape(batch, steps, size))

        return x + self.mlp(self.norm2(x))
'''

DECODER_INIT = '''
class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        size: int = 32,
        context: int = 32,
        layers: int = 2,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocabulary, size)
        self.positions = nn.Embedding(context, size)
        self.blocks = nn.Sequential(*[Block(size, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(size)
        self.head = nn.Linear(size, vocabulary)
'''

DECODER_FORWARD = '''
    def forward(self, ids: Tensor) -> Tensor:
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1]))

        return self.head(self.norm(self.blocks(x)))
'''

LM_SOLUTION = '''
def shift(tokens: Tensor) -> tuple[Tensor, Tensor]:
    return tokens[:, :-1], tokens[:, 1:]


def lm_loss(
    model: nn.Module,
    tokens: Tensor,
) -> Tensor:
    inputs, targets = shift(tokens)
    logits = model(inputs)

    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def generate(
    model: nn.Module,
    prompt: list[int],
    steps: int,
    context: int,
) -> list[int]:
    ids = list(prompt)

    with torch.no_grad():
        for _ in range(steps):
            logits = model(torch.tensor([ids[-context:]]))
            ids.append(logits[0, -1].argmax().item())

    return ids
'''

DECODER1_LESSON = prose(
    """
    This exercise builds the kind behind GPT and Llama, a decoder only
    language model, end to end. It reads characters of a small made up
    language, like `the dog chases a fish.`, and learns to write more of it.

    A language model predicts the next token. The trick that makes training
    fast is shifted targets. A window of tokens gives the inputs, every token
    but the last, and the targets, every token but the first.

    ```python
    window = torch.tensor([[5, 7, 2, 9]])
    window[:, :-1], window[:, 1:]    # (tensor([[5, 7, 2]]), tensor([[7, 2, 9]]))
    ```

    Position t reads tokens 0 to t and is scored on token t + 1. The causal
    mask keeps it from reading that answer. So one window of 33 tokens trains
    32 predictions at once, and a batch of 32 windows trains 1,024 per step.

    The model adds a learned position embedding to a token embedding, then
    runs a stack of the blocks of block1 and a last LayerNorm. A Linear head
    gives a logit for every token of the vocabulary.

    Writing goes one token at a time. Feed the text so far, take the logits at
    the last position, pick the likeliest token, add it and repeat. The model
    has position embeddings for 32 steps only, so it reads the last 32 tokens
    at most.

    Here is the run behind this exercise, 2 blocks and 27,861 weights, with
    Adam at a rate of 0.01 and batches of 32 windows. The validation text is
    2,000 more sentences of the language that training never saw.

    | Steps | 0 | 100 | 200 |
    |---|---|---|---|
    | Validation loss, per character | 3.203 | 0.379 | 0.268 |

    The loss starts near `log(21) = 3.04`, a guess among the 21 characters.
    No model can get below about 0.19 here, since three words in each
    sentence are picked at random. After 200 steps it writes
    `the dog draws the moon. the dog draws the garden.` from `the dog `.
    """,
)

DECODER1_SOURCE = source_file(
    '''
    # decoder1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    window = torch.tensor([[5, 7, 2, 9]])
    last = window[:, -1:]                    # tensor([[9]])
    likeliest = torch.tensor([0.1, 2.0, 0.3]).argmax().item()    # 1


''' + indented(CORPUS_PART) + '''


''' + indented(BLOCK_PART) + '''


    # TODO: Return the inputs and the targets of a batch of windows of tokens,
    #   every token but the last of each window, and every token but the first.
    def shift(tokens: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish forward. Add the token embeddings of ids, of shape
    #   (batch, steps), to the position embeddings of positions 0 to steps - 1.
    #   Run the blocks, the last norm and the head. Return the logits, of shape
    #   (batch, steps, vocabulary).
''' + indented(DECODER_INIT) + '''

        def forward(self, ids: Tensor) -> Tensor:
            ...


    # TODO: Return the cross entropy of model's logits on the inputs of tokens
    #   against their targets, from shift, over every position of every window.
    def lm_loss(
        model: nn.Module,
        tokens: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return prompt with steps more tokens added, each the likeliest next
    #   token after the last context tokens so far. Record nothing for autograd.
    def generate(
        model: nn.Module,
        prompt: list[int],
        steps: int,
        context: int,
    ) -> list[int]:
        ...
    ''',
)

DECODER1_SOLUTION = source_file(
    '''
    # decoder1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    window = torch.tensor([[5, 7, 2, 9]])
    last = window[:, -1:]
    likeliest = torch.tensor([0.1, 2.0, 0.3]).argmax().item()


''' + indented(CORPUS_PART) + '''


''' + indented(BLOCK_PART) + '''


''' + indented(DECODER_INIT) + '''

''' + indented(DECODER_FORWARD) + '''


''' + indented(LM_SOLUTION) + '''
    ''',
)


def decoder_namespace() -> dict[str, object]:
    """The given parts and the solution, run once, so the checks compare against exactly the file's code."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(CORPUS_PART + BLOCK_PART + DECODER_INIT + DECODER_FORWARD + LM_SOLUTION, namespace)

    return namespace


DECODER_REFERENCE = decoder_namespace()
CORPUS = DECODER_REFERENCE["corpus"](2000, torch.Generator().manual_seed(0))
CHARACTERS = sorted(set(CORPUS))
CORPUS_IDS = torch.tensor([CHARACTERS.index(character) for character in CORPUS])

# 2,000 more sentences from another seed, which training never reads, for the validation loss.
HELD_OUT_IDS = torch.tensor([CHARACTERS.index(character) for character in DECODER_REFERENCE["corpus"](2000, torch.Generator().manual_seed(1))])


def seeded_decoder(
    make: type,
    context: int = 32,
) -> nn.Module:
    torch.manual_seed(0)

    return make(len(CHARACTERS), 32, context)


def windows(
    seed: int,
    count: int,
    context: int = 32,
) -> Tensor:
    return DECODER_REFERENCE["random_windows"](CORPUS_IDS, count, context, torch.Generator().manual_seed(seed))


def train_decoder(
    model: nn.Module,
    loss_of: object,
    steps: int,
) -> nn.Module:
    """The run the lesson's table comes from, Adam at 0.01 on batches of 32 windows."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)

    for _ in range(steps):
        tokens = DECODER_REFERENCE["random_windows"](CORPUS_IDS, 32, 32, generator)
        loss = loss_of(model, tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def held_out_loss(model: nn.Module) -> float:
    """lm_loss on 256 windows of the held out sentences."""
    tokens = DECODER_REFERENCE["random_windows"](HELD_OUT_IDS, 256, 32, torch.Generator().manual_seed(2))

    with torch.no_grad():
        return DECODER_REFERENCE["lm_loss"](model, tokens).item()


def measure_decoder_run() -> dict[str, object]:
    """decoder1's numbers, the weights, the characters, the validation loss after 0, 100 and 200 steps of the run
    train_decoder makes, and what the trained model writes from "the dog "."""
    model = seeded_decoder(DECODER_REFERENCE["TinyDecoder"])
    losses = {0: held_out_loss(model)}

    for steps in (100, 200):
        losses[steps] = held_out_loss(train_decoder(seeded_decoder(DECODER_REFERENCE["TinyDecoder"]), DECODER_REFERENCE["lm_loss"], steps))

    trained = train_decoder(seeded_decoder(DECODER_REFERENCE["TinyDecoder"]), DECODER_REFERENCE["lm_loss"], 200)
    prompt = [CHARACTERS.index(character) for character in "the dog "]
    written = DECODER_REFERENCE["generate"](trained, prompt, 42, 32)

    return {
        "weights": sum(parameter.numel() for parameter in model.parameters()),
        "characters": len(CHARACTERS),
        "losses": losses,
        "written": "".join(CHARACTERS[index] for index in written),
    }


def expect_shifted(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the inputs and the targets as two tensors in a tuple.")

    for name, part, wanted in zip(("inputs", "targets"), actual, expected):
        expect_tensor(
            call=f"the {name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_decoder_forward(found: type) -> None:
    model = expect_module(f"TinyDecoder({len(CHARACTERS)})", seeded_decoder(found))
    ids = windows(544, 3)[:, :-1]

    with torch.no_grad():
        expected = seeded_decoder(DECODER_REFERENCE["TinyDecoder"])(ids)
        actual = model(ids)

    expect_tensor(
        call=f"TinyDecoder({len(CHARACTERS)}) on 3 windows of 32 tokens",
        actual=actual,
        expected=expected,
    )


def check_lm_loss(found: object) -> None:
    model = seeded_decoder(DECODER_REFERENCE["TinyDecoder"])
    tokens = windows(545, 4)

    with torch.no_grad():
        expected = DECODER_REFERENCE["lm_loss"](model, tokens).item()
        actual = found(model, tokens)

    expect_close(
        call="lm_loss on 4 windows of 33 tokens",
        actual=actual.item() if isinstance(actual, Tensor) else actual,
        expected=expected,
        tolerance=1e-5,
    )


class GradRecorder(nn.Module):
    """A model that notes whether any call to it ran with autograd recording."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.recorded = False

    def forward(self, ids: Tensor) -> Tensor:
        self.recorded = self.recorded or torch.is_grad_enabled()

        return self.model(ids)


def check_generate(found: object) -> None:
    model = seeded_decoder(DECODER_REFERENCE["TinyDecoder"], context=8)
    watched = GradRecorder(model)
    prompt = CORPUS_IDS[:12].tolist()
    call = "generate(an untrained TinyDecoder with a context of 8, a prompt of 12 tokens, 6, 8)"
    actual = found(watched, prompt, 6, 8)

    if isinstance(actual, list) and any(isinstance(entry, Tensor) for entry in actual):
        raise Mismatch(f"{call} holds tensors among its ids, but every id should be a plain int. .item() gives one.")

    expect_value(
        call=call,
        actual=actual,
        expected=DECODER_REFERENCE["generate"](model, prompt, 6, 8),
    )

    if watched.recorded:
        raise Mismatch(f"{call} ran the model while autograd was recording. Put the loop inside with torch.no_grad():.")


def check_decoder_learns(found: object) -> None:
    model_class = found.__globals__.get("TinyDecoder")

    if not isinstance(model_class, type):
        raise Mismatch("decoder1.py has no TinyDecoder class to train.")

    model = train_decoder(seeded_decoder(model_class), found, 150)
    loss = held_out_loss(model)

    if loss > 0.6:
        raise Mismatch(f"After 150 steps of training with lm_loss, the validation loss is {loss:.3f}, but it should be below 0.6.")


def check_decoder1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="shift",
        description="splits each window into inputs and targets one step apart",
        inputs=(
            (torch.tensor([[5, 7, 2, 9], [1, 0, 3, 3]]),),
            (windows(546, 3, 6),),
        ),
        reference=lambda tokens: (tokens[:, :-1], tokens[:, 1:]),
        expect=expect_shifted,
    )
    checker.test_method(
        class_name="TinyDecoder",
        method="forward",
        description="adds positions to tokens and runs the blocks, the norm and the head",
        run=check_decoder_forward,
    )
    checker.test(
        function="lm_loss",
        description="scores every position on the next token",
        run=check_lm_loss,
    )
    checker.test(
        function="generate",
        description="adds the likeliest token, reading only the last context tokens",
        run=check_generate,
    )
    checker.test(
        function="lm_loss",
        description="trains the model to write the made up language",
        run=check_decoder_learns,
    )


DECODER1 = Exercise(
    name="decoder1",
    chapter=MODEL_TYPES,
    title="A decoder only language model",
    lesson=DECODER1_LESSON,
    source=DECODER1_SOURCE,
    solution=DECODER1_SOLUTION,
    hints={
        "shift": "tokens[:, :-1] drops the last token of each window, and tokens[:, 1:] the first.",
        "TinyDecoder.forward": "self.positions(torch.arange(ids.shape[1])) broadcasts over the batch when added to "
        "self.tokens(ids). Then self.blocks, self.norm and self.head in turn.",
        "lm_loss": "Run the model on the inputs from shift. Reshape the logits to (-1, vocabulary) and the targets "
        "to (-1) before F.cross_entropy.",
        "generate": "Copy prompt with list(prompt). Inside torch.no_grad(), run the model on torch.tensor([ids[-context:]]) "
        "and append the argmax of the logits at the last position, as a plain int.",
    },
    introduces=(),
    check=check_decoder1,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
