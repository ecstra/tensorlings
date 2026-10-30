"""49_masks, causal and padding masks and both at once, and a model without the causal mask that cheats in training and fails at generation."""

from __future__ import annotations

import math

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
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

MASKS = Chapter(
    folder="49_masks",
    title="Masks",
    readme=source_file(
        """
        # Masks

        A mask decides which positions each position may look at. This
        chapter joins the causal mask, which hides the future, with the
        padding mask, which hides the filler at the end of short sequences.
        Then it shows what a model does without the causal mask, which looks
        perfect in training and falls apart when it writes.

        ## Further reading

        - [Rush and colleagues, the annotated transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
        - [PyTorch, F.scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
        - [Karpathy, nanoGPT's model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
        """,
    ),
)


# ---------------------------------------------------------------------------
# mask1
# ---------------------------------------------------------------------------

MASK1_LESSON = prose(
    """
    A batch needs every sequence at the same length, so the short ones are
    filled out at the end with a padding token. Attention must never read
    that filler. It isn't part of the sequence, and its values would leak
    into every average.

    A padding mask marks it, True at every position past a sequence's length.

    ```python
    lengths = torch.tensor([3, 1])
    torch.arange(4) >= lengths.unsqueeze(1)
    # tensor([[False, False, False,  True], [False,  True,  True,  True]])
    ```

    It blocks keys, the columns of the scores. `.unsqueeze(1)` turns each
    sequence's mask into one row of shape `(1, steps)`, which broadcasts down
    every query's row. attention2's causal mask has shape `(steps, steps)`
    and is the same for every sequence. `|` joins the two, True where either
    one blocks, and broadcasting gives one `(steps, steps)` mask per sequence.

    ```python
    torch.tensor([[True, False]]) | torch.tensor([[False], [True]])
    # tensor([[ True, False], [ True,  True]])
    ```

    A padded position still gets an output, from the real positions before
    it. That output is thrown away, and the loss leaves it out too.

    PyTorch's `F.scaled_dot_product_attention` reads a True in a boolean
    `attn_mask` the other way round, as a pair that takes part. `~` flips
    every entry of a mask, so pass it `~blocked`. `nn.MultiheadAttention`
    reads True as blocked, like the masks here. Read the documentation for
    every new function that takes a mask.

    A row with every key blocked takes a softmax over minus infinity alone,
    which is NaN. Padding at the start, which some generation code does,
    makes such rows. PyTorch's function gives zeros for them, and your own
    attention gives NaN, which spreads through every later step.
    """,
)

MASK1_SOURCE = source_file(
    '''
    # mask1.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    past_two = torch.arange(4) >= 2                                   # tensor([False, False,  True,  True])
    either = torch.tensor([[True, False]]) | torch.tensor([[False], [True]])    # tensor([[ True, False], [ True,  True]])
    flipped = ~torch.tensor([True, False])                            # tensor([False,  True])


    # TODO: Return the padding mask of a batch, of shape (batch, steps), True
    #   at every position at or past each sequence's length.
    def padding_mask(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return the mask of shape (batch, steps, steps) that is True where a
    #   query may not look at a key, because the key is in its future or is
    #   padding.
    def blocked_keys(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return attention of q, k and v, each of shape (batch, steps, size),
    #   with scores scaled by the square root of size, and minus infinity
    #   wherever blocked is True before the softmax.
    def masked_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the same attention from F.scaled_dot_product_attention, with
    #   blocked turned into the mask it expects.
    def torch_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...
    ''',
)

MASK1_SOLUTION = source_file(
    '''
    # mask1.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    past_two = torch.arange(4) >= 2
    either = torch.tensor([[True, False]]) | torch.tensor([[False], [True]])
    flipped = ~torch.tensor([True, False])


    def padding_mask(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        return torch.arange(steps) >= lengths.unsqueeze(1)


    def blocked_keys(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        future = torch.triu(torch.ones(steps, steps), diagonal=1).bool()

        return future | padding_mask(lengths, steps).unsqueeze(1)


    def masked_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

        return torch.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1) @ v


    # This function reads True as a pair that takes part, so the mask is flipped.
    def torch_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)
    ''',
)


def reference_padding_mask(
    lengths: Tensor,
    steps: int,
) -> Tensor:
    return torch.arange(steps).unsqueeze(0) >= lengths.unsqueeze(1)


def reference_blocked_keys(
    lengths: Tensor,
    steps: int,
) -> Tensor:
    future = torch.ones(steps, steps).triu(diagonal=1).bool()

    return future.unsqueeze(0) | reference_padding_mask(lengths, steps).unsqueeze(1)


def reference_masked_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    blocked: Tensor,
) -> Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

    return torch.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1) @ v


def attention_inputs(
    seed: int,
    batch: int,
    steps: int,
    size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        small_tensor(seed, batch, steps, size) / 2,
        small_tensor(seed + 1, batch, steps, size) / 2,
        small_tensor(seed + 2, batch, steps, size) / 2,
    )


def check_mask1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="padding_mask",
        description="marks every position past each length",
        inputs=(
            (torch.tensor([3, 1]), 4),
            (torch.tensor([5, 2, 4]), 5),
        ),
        reference=reference_padding_mask,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="blocked_keys",
        description="blocks the future and the padding together",
        inputs=(
            (torch.tensor([3, 1]), 4),
            (torch.tensor([2, 5, 4]), 5),
        ),
        reference=reference_blocked_keys,
        expect=expect_tensor,
    )
    blocked = reference_blocked_keys(torch.tensor([4, 2]), 4)
    check_on_inputs(
        checker=checker,
        function="masked_attention",
        description="leaves out every blocked key",
        inputs=(
            (*attention_inputs(490, 2, 4, 3), Described(blocked, "the blocked keys of lengths 4 and 2")),
            (*attention_inputs(493, 1, 3, 2), torch.ones(3, 3).triu(diagonal=1).bool()),
        ),
        reference=reference_masked_attention,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="torch_attention",
        description="matches masked_attention through PyTorch's function",
        inputs=(
            (*attention_inputs(496, 2, 4, 3), Described(blocked, "the blocked keys of lengths 4 and 2")),
        ),
        reference=reference_masked_attention,
        expect=expect_tensor,
    )


MASK1 = Exercise(
    name="mask1",
    chapter=MASKS,
    title="Causal and padding masks",
    lesson=MASK1_LESSON,
    source=MASK1_SOURCE,
    solution=MASK1_SOLUTION,
    hints={
        "padding_mask": "Compare torch.arange(steps) with lengths.unsqueeze(1) using >=, so every row gets its "
        "own length.",
        "blocked_keys": "Build the future mask with torch.triu as attention2 does, then | it with "
        "padding_mask(lengths, steps).unsqueeze(1).",
        "masked_attention": "Scores are q @ k.transpose(-2, -1) over math.sqrt(q.shape[-1]). masked_fill them with "
        "float(\"-inf\") where blocked is True, softmax over the last dimension, then @ v.",
        "torch_attention": "F.scaled_dot_product_attention(q, k, v, attn_mask=...) with ~blocked, since True means "
        "takes part there.",
    },
    introduces=(),
    check=check_mask1,
)


# ---------------------------------------------------------------------------
# mask2
# ---------------------------------------------------------------------------

HALF = 8
VOCABULARY = 16

MASK2_LESSON = prose(
    """
    This trains the same tiny model twice, with the causal mask and without
    it. Each training sequence holds 8 random tokens, then the same 8 again.
    The model predicts every next token, as a language model does. Here are
    300 steps each, measured on 1,000 new sequences.

    | | Causal mask | No mask |
    |---|---|---|
    | Loss | 1.580 | 0.000 |
    | Next tokens right, first half | 6.3% | 100% |
    | Next tokens right, second half | 100% | 100% |
    | Tokens right when it writes the copy | 100% | 81.7% |
    | Whole copies right when it writes | 100% | 5.5% |

    The first 8 tokens are random draws from 16, so nothing can predict them
    better than 1 time in 16, 6.3%. The unmasked model gets all of them right.
    During training the next token sits in its own input, and it learned to
    read it from there. A loss below what the data allows is the sign of a
    leak.

    Writing takes that away. A model writes one token at a time from what it
    has so far, and the next token doesn't exist yet. Given the first 8
    tokens and asked to write the copy, the unmasked model gets only 5.5% of
    whole copies right. The masked one learned to read the token 8 places
    before the one it writes, which still works while it writes.

    The model adds a learned row per position to each token, so it can tell
    positions apart. 52_positions explains why it needs that. Writing here is
    greedy, the most likely token at every step.
    """,
)

# Indented as it sits inside the class in the files below, which source_file takes the indentation off.
TINY_MODEL_INIT = '''
        def __init__(
            self,
            causal: bool,
            size: int = 32,
        ) -> None:
            super().__init__()
            self.causal = causal
            self.tokens = nn.Embedding(16, size)
            self.positions = nn.Embedding(16, size)
            self.query = nn.Linear(size, size)
            self.key = nn.Linear(size, size)
            self.value = nn.Linear(size, size)
            self.head = nn.Linear(size, 16)
'''

MASK2_SOURCE = source_file(
    '''
    # mask2.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    so_far = torch.tensor([[3, 7]])
    grown = torch.cat([so_far, torch.tensor([[1]])], dim=1)    # tensor([[3, 7, 1]])


    # The future mask from attention2.
    def future_mask(steps: int) -> Tensor:
        return torch.triu(torch.ones(steps, steps), diagonal=1).bool()


    # TODO: Finish the line that sets scores in forward. With self.causal True
    #   the future gets minus infinity, and with it False the scores stay as
    #   they are.
    class TinyModel(nn.Module):''' + TINY_MODEL_INIT + '''
        def forward(self, x: Tensor) -> Tensor:
            steps = x.shape[1]
            h = self.tokens(x) + self.positions(torch.arange(steps))
            q, k, v = self.query(h), self.key(h), self.value(h)
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            scores = ...

            return self.head(torch.softmax(scores, dim=-1) @ v)


    # TODO: Return prompt, of shape (batch, length), with steps more tokens
    #   written after it. Each new token is model's most likely next token at
    #   the last position, given everything so far. Record nothing for autograd.
    def generate(
        model: nn.Module,
        prompt: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of sequences, each of 2 * half tokens, whose second
    #   half generate writes exactly right from the first half.
    def copies_right(
        model: nn.Module,
        sequences: Tensor,
        half: int,
    ) -> float:
        ...
    ''',
)

MASK2_SOLUTION = source_file(
    '''
    # mask2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    so_far = torch.tensor([[3, 7]])
    grown = torch.cat([so_far, torch.tensor([[1]])], dim=1)


    def future_mask(steps: int) -> Tensor:
        return torch.triu(torch.ones(steps, steps), diagonal=1).bool()


    class TinyModel(nn.Module):''' + TINY_MODEL_INIT + '''
        def forward(self, x: Tensor) -> Tensor:
            steps = x.shape[1]
            h = self.tokens(x) + self.positions(torch.arange(steps))
            q, k, v = self.query(h), self.key(h), self.value(h)
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(future_mask(steps), float("-inf")) if self.causal else scores

            return self.head(torch.softmax(scores, dim=-1) @ v)


    def generate(
        model: nn.Module,
        prompt: Tensor,
        steps: int,
    ) -> Tensor:
        with torch.no_grad():
            for _ in range(steps):
                next_tokens = model(prompt)[:, -1].argmax(dim=-1, keepdim=True)
                prompt = torch.cat([prompt, next_tokens], dim=1)

        return prompt


    def copies_right(
        model: nn.Module,
        sequences: Tensor,
        half: int,
    ) -> float:
        written = generate(model, sequences[:, :half], half)

        return (written[:, half:] == sequences[:, half:]).all(dim=1).float().mean().item()
    ''',
)


class ReferenceTinyModel(nn.Module):
    def __init__(
        self,
        causal: bool,
        size: int = 32,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.tokens = nn.Embedding(VOCABULARY, size)
        self.positions = nn.Embedding(2 * HALF, size)
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)
        self.head = nn.Linear(size, VOCABULARY)

    def forward(self, x: Tensor) -> Tensor:
        steps = x.shape[1]
        h = self.tokens(x) + self.positions(torch.arange(steps))
        q, k, v = self.query(h), self.key(h), self.value(h)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

        if self.causal:
            scores = scores.masked_fill(torch.ones(steps, steps).triu(diagonal=1).bool(), float("-inf"))

        return self.head(torch.softmax(scores, dim=-1) @ v)


def copy_sequences(
    count: int,
    generator: torch.Generator,
) -> Tensor:
    first = torch.randint(0, VOCABULARY, (count, HALF), generator=generator)

    return torch.cat([first, first], dim=1)


def seeded_tiny(
    make: type,
    causal: bool,
) -> nn.Module:
    torch.manual_seed(0)

    return make(causal)


def train_tiny(model: nn.Module) -> nn.Module:
    """The run the lesson's table comes from, 300 Adam steps on batches of 64 sequences."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)

    for _ in range(300):
        x = copy_sequences(64, generator)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCABULARY), x[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def reference_generate(
    model: nn.Module,
    prompt: Tensor,
    steps: int,
) -> Tensor:
    with torch.no_grad():
        for _ in range(steps):
            prompt = torch.cat([prompt, model(prompt)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)

    return prompt


def reference_copies_right(
    model: nn.Module,
    sequences: Tensor,
    half: int,
) -> float:
    written = reference_generate(model, sequences[:, :half], half)

    return (written[:, half:] == sequences[:, half:]).all(dim=1).float().mean().item()


def check_copies_right(found: object) -> None:
    """An untrained model copies nothing, so the file's generate is swapped for one that writes 12 of 20
    copies right and the last token of the rest wrong, and the share should come out at 0.6."""
    sequences = copy_sequences(20, torch.Generator().manual_seed(499))
    calls: list[tuple[Tensor, int]] = []

    def scripted_generate(
        model: nn.Module,
        prompt: Tensor,
        steps: int,
    ) -> Tensor:
        calls.append((prompt, steps))
        written = sequences.clone()
        written[12:, -1] = (written[12:, -1] + 1) % 16

        return written

    namespace = found.__globals__
    original = namespace.get("generate")
    namespace["generate"] = scripted_generate

    try:
        actual = found(None, sequences, HALF)
    finally:
        namespace["generate"] = original

    if not calls:
        raise Mismatch("copies_right never called generate to write the copies.")

    prompt, steps = calls[0]

    if not torch.equal(prompt, sequences[:, :HALF]) or steps != HALF:
        raise Mismatch("copies_right should pass generate the first half of each sequence and ask for half more tokens.")

    expect_close(
        call="copies_right on 20 sequences, 12 of them copied right",
        actual=actual,
        expected=0.6,
    )


def first_half_right(
    model: nn.Module,
    sequences: Tensor,
) -> float:
    with torch.no_grad():
        guesses = model(sequences[:, :-1]).argmax(dim=-1)

    return (guesses[:, : HALF - 1] == sequences[:, 1:HALF]).float().mean().item()


def trained_reference(causal: bool) -> nn.Module:
    return train_tiny(seeded_tiny(ReferenceTinyModel, causal))


def check_tiny_forward(found: type) -> None:
    x = copy_sequences(3, torch.Generator().manual_seed(497))[:, :-1]

    for causal in (True, False):
        model = expect_module(f"TinyModel({causal})", seeded_tiny(found, causal))

        with torch.no_grad():
            expected = seeded_tiny(ReferenceTinyModel, causal)(x)
            actual = model(x)

        expect_tensor(
            call=f"TinyModel({causal}) on 3 sequences of 15 tokens",
            actual=actual,
            expected=expected,
        )


def check_cheats(found: type) -> None:
    test = copy_sequences(500, torch.Generator().manual_seed(99))
    masked = train_tiny(seeded_tiny(found, True))
    unmasked = train_tiny(seeded_tiny(found, False))
    masked_copies = reference_copies_right(masked, test, HALF)

    if masked_copies < 0.95:
        raise Mismatch(
            f"With the causal mask, the trained TinyModel writes {masked_copies:.1%} of copies right, but it should "
            "pass 95%.",
        )

    if first_half_right(unmasked, test) < 0.9:
        raise Mismatch("Without the mask, the trained TinyModel should read the next token in training and get the random first half right.")

    unmasked_copies = reference_copies_right(unmasked, test, HALF)

    if unmasked_copies > 0.3:
        raise Mismatch(
            f"Without the mask, the trained TinyModel writes {unmasked_copies:.1%} of copies right, but cut off "
            "from the future it should fall below 30%.",
        )


def check_mask2(checker: Checker) -> None:
    checker.test_method(
        class_name="TinyModel",
        method="forward",
        description="masks the future only when causal",
        run=check_tiny_forward,
    )
    model = seeded_tiny(ReferenceTinyModel, True)
    prompts = copy_sequences(4, torch.Generator().manual_seed(498))[:, :HALF]
    check_on_inputs(
        checker=checker,
        function="generate",
        description="writes one most likely token at a time",
        inputs=(
            (Described(model, "an untrained TinyModel(True)"), prompts, 5),
        ),
        reference=reference_generate,
        expect=expect_tensor,
    )
    checker.test(
        function="copies_right",
        description="gives the share of copies written exactly right",
        run=check_copies_right,
    )
    checker.test_method(
        class_name="TinyModel",
        method="forward",
        description="trained, copies with the mask and cheats without it",
        run=check_cheats,
    )


MASK2 = Exercise(
    name="mask2",
    chapter=MASKS,
    title="Why the causal mask",
    lesson=MASK2_LESSON,
    source=MASK2_SOURCE,
    solution=MASK2_SOLUTION,
    hints={
        "TinyModel.forward": "scores.masked_fill(future_mask(steps), float(\"-inf\")) when self.causal is True, and "
        "scores itself when it is False. A conditional expression fits on one line.",
        "generate": "Inside torch.no_grad(), loop steps times. Take model(prompt)[:, -1].argmax(dim=-1, "
        "keepdim=True) and torch.cat it onto prompt along dim=1.",
        "copies_right": "Generate half tokens from sequences[:, :half]. Compare the written second half with the "
        "true one, .all(dim=1) per sequence, then the mean as floats.",
    },
    introduces=(),
    check=check_mask2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
