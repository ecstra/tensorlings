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
from .attention import calls_to_attention
from .bidirectional import expect_nothing_recorded
from .linear_layers import expect_module

MASKS = Chapter(
    folder="49_masks",
    title="Masks",
    readme=source_file(
        """
        # Masks

        A mask decides which positions each position may look at. This
        chapter combines the causal mask, which hides future positions, with
        the padding mask, which hides the filler added to the end of short
        sequences. Then it shows what a model does without the causal mask.
        Such a model looks perfect in training and fails when it generates
        text.

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
    In a batch, every sequence must have the same length. So the shorter
    sequences are filled out at the end with a padding token. Attention must
    never read these padding positions. They are not part of the real
    sequence, and their values would get mixed into every average that
    attention computes.

    A **padding mask** marks the padding. It is True at every position whose
    index is at least the sequence's length, and False at the real positions.
    Here are two sequences of lengths 3 and 1, padded to 4 steps.

    ```python
    lengths = torch.tensor([3, 1])
    torch.arange(4) >= lengths.unsqueeze(1)
    # tensor([[False, False, False,  True], [False,  True,  True,  True]])
    ```

    The padding mask blocks keys. In the scores, each row is a query and each
    column is a key, so padding blocks whole columns. Calling `.unsqueeze(1)`
    on a padding mask of shape `(batch, steps)` gives shape
    `(batch, 1, steps)`. Each sequence's mask is then a single row, and
    broadcasting copies that row down to every query.

    The causal mask from attention2 has shape `(steps, steps)` and is the
    same for every sequence. The `|` operator combines two masks into one
    that is True wherever either mask is True. With broadcasting, the result
    holds one `(steps, steps)` mask per sequence. Here a row of shape
    `(1, 2)` and a column of shape `(2, 1)` combine into shape `(2, 2)`.

    ```python
    torch.tensor([[True, False]]) | torch.tensor([[False], [True]])
    # tensor([[ True, False], [ True,  True]])
    ```

    A padding position is still a query, so it still gets an output,
    computed from the real positions before it. That output is never used,
    and the loss leaves it out too.

    PyTorch's `F.scaled_dot_product_attention` computes attention in one
    call. In its boolean `attn_mask`, True has the opposite meaning. True
    means the query may look at that key. The `~` operator flips every entry
    of a boolean tensor, so you pass it `~blocked`. PyTorch's attention layer
    `nn.MultiheadAttention`, which 50_multihead builds, uses True for blocked,
    like the masks here. So read the documentation of every new function
    that takes a mask.

    If every key in a row is blocked, every score in that row is minus
    infinity, and the softmax of the row is NaN. This happens when the
    padding is at the start of a sequence, which some generation code does.
    The first query then sees only padding. PyTorch's function returns zeros
    for such rows. Your own attention returns NaN, and NaN spreads into every
    number computed from it in later steps.
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


    # TODO: Return the padding mask of a batch, of shape (batch, steps). lengths
    #   has shape (batch,) and holds each sequence's real length. The mask is
    #   True at every position whose index is at least that sequence's length.
    #   Example: padding_mask(torch.tensor([3, 1]), 4) returns
    #   tensor([[False, False, False,  True], [False,  True,  True,  True]]).
    def padding_mask(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return a mask of shape (batch, steps, steps). Entry [b, i, j] is True
    #   when query i of sequence b may not look at key j, either because j is
    #   after i or because position j is padding in sequence b.
    #   Example: blocked_keys(torch.tensor([2]), 3) returns
    #   tensor([[[False,  True,  True], [False, False,  True],
    #   [False, False,  True]]]).
    def blocked_keys(
        lengths: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return the attention output for q, k and v, each of shape
    #   (batch, steps, size). Compute the scores of every query with every key
    #   and divide them by the square root of size. Set the scores to minus
    #   infinity wherever blocked, of shape (batch, steps, steps), is True.
    #   Then take the softmax over the keys and multiply the weights by v.
    def masked_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        blocked: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the same result as masked_attention, computed with
    #   F.scaled_dot_product_attention instead. Pass blocked as its attn_mask,
    #   converted to the meaning of True that this function expects.
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


    # This function reads True as "the query may look at this key", so the mask is flipped.
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
        description="is True at every position at or past each length",
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
        description="blocks both future keys and padding keys",
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
        description="gives the same result as masked_attention using PyTorch's function",
        inputs=(
            (*attention_inputs(496, 2, 4, 3), Described(blocked, "the blocked keys of lengths 4 and 2")),
        ),
        reference=reference_masked_attention,
        expect=expect_tensor,
    )
    checker.test(
        function="torch_attention",
        description="calls PyTorch's function",
        run=lambda found: check_torch_attention_calls(found, blocked),
    )


def check_torch_attention_calls(
    found: object,
    blocked: Tensor,
) -> None:
    q, k, v = attention_inputs(496, 2, 4, 3)

    if not calls_to_attention(lambda: found(q, k, v, blocked)):
        raise Mismatch(
            "torch_attention never called F.scaled_dot_product_attention. Pass it q, k and v, with blocked "
            "turned into its attn_mask.",
        )


MASK1 = Exercise(
    name="mask1",
    chapter=MASKS,
    title="Causal and padding masks",
    lesson=MASK1_LESSON,
    source=MASK1_SOURCE,
    solution=MASK1_SOLUTION,
    hints={
        "padding_mask": "Compare torch.arange(steps) with lengths.unsqueeze(1) using >=. Broadcasting then "
        "compares every position with each sequence's own length.",
        "blocked_keys": "Build the causal mask with torch.triu as attention2 does. Then combine it with "
        "padding_mask(lengths, steps).unsqueeze(1) using the | operator.",
        "masked_attention": "The scores are q @ k.transpose(-2, -1) divided by math.sqrt(q.shape[-1]). Call "
        "masked_fill on them with float(\"-inf\") where blocked is True. Then take the softmax over the last "
        "dimension and multiply the result by v with @.",
        "torch_attention": "Call F.scaled_dot_product_attention(q, k, v, attn_mask=...) with ~blocked as the mask. "
        "In that function True means the query may look at the key.",
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
    This exercise trains the same tiny model twice, once with the causal mask
    and once without it. Each training sequence has 8 random tokens followed
    by the same 8 tokens again. At every position the model predicts the next
    token, as a language model does. Each model trains for 300 steps, and the
    table measures both on 1,000 new sequences.

    | | Causal mask | No mask |
    |---|---|---|
    | Loss | 1.580 | 0.000 |
    | Next token right, first half | 6.3% | 100% |
    | Next token right, second half | 100% | 100% |
    | Tokens right when it writes the copy | 100% | 81.7% |
    | Whole copies right when it writes | 100% | 5.5% |

    Each of the first 8 tokens is drawn at random from 16 possible tokens. So
    no model can predict them correctly more than 1 time in 16, which is
    6.3%. Yet the unmasked model gets all of them right. Without the mask,
    each position can see the whole input, and the next token is part of
    that input. So the model learned to copy the next token from its input.
    A loss lower than the data allows is a sign of a leak, meaning the answer
    reaches the model through its input.

    When the model writes text on its own, which is called generating, it
    can no longer copy the answer. It writes one token at a time from the tokens so far, and the
    next token does not exist yet. Given the first 8 tokens and asked to
    write the copy, the unmasked model writes only 5.5% of copies fully
    right. The masked model learned to read the token 8 positions before the
    one it predicts. That token is always there while it writes, so the
    masked model writes every copy correctly.

    The model adds a learned embedding for each position to each token's
    embedding, so it can tell positions apart. 52_positions explains why it
    needs that. The writing here is greedy, which means it picks the most
    likely token at every step.
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


    # TODO: Finish the second line that sets scores in forward. When self.causal
    #   is True, set the scores of future keys to minus infinity, using
    #   future_mask. When self.causal is False, leave the scores unchanged.
    class TinyModel(nn.Module):''' + TINY_MODEL_INIT + '''
        def forward(self, x: Tensor) -> Tensor:
            steps = x.shape[1]
            h = self.tokens(x) + self.positions(torch.arange(steps))
            q, k, v = self.query(h), self.key(h), self.value(h)
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            scores = ...

            return self.head(torch.softmax(scores, dim=-1) @ v)


    # TODO: Return prompt, of shape (batch, length), with steps new tokens added
    #   to its end, so the result has shape (batch, length + steps). To pick
    #   each new token, run model on all the tokens so far and take the most
    #   likely token at the last position. Make sure autograd records nothing.
    def generate(
        model: nn.Module,
        prompt: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: sequences has shape (batch, 2 * half). Give generate the first half
    #   of each sequence and let it write half more tokens. Return the fraction
    #   of sequences whose written second half matches the true second half
    #   exactly, as a float.
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
        raise Mismatch("Without the mask, the trained TinyModel should copy the next token from its input and so predict the random first half right.")

    unmasked_copies = reference_copies_right(unmasked, test, HALF)

    if unmasked_copies > 0.3:
        raise Mismatch(
            f"Without the mask, the trained TinyModel writes {unmasked_copies:.1%} of copies right, but without "
            "the future tokens to copy from it should score below 30%.",
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
        description="adds the most likely next token, one token at a time",
        inputs=(
            (Described(model, "an untrained TinyModel(True)"), prompts, 5),
        ),
        reference=reference_generate,
        expect=expect_tensor,
    )
    checker.test(
        function="generate",
        description="records nothing for autograd",
        run=lambda found: expect_nothing_recorded(
            function="generate",
            modules=[model],
            run=lambda: found(model, prompts.clone(), 3),
        ),
    )
    checker.test(
        function="copies_right",
        description="gives the share of copies written exactly right",
        run=check_copies_right,
    )
    checker.test_method(
        class_name="TinyModel",
        method="forward",
        description="after training, copies with the mask and cheats without it",
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
        "TinyModel.forward": "Use scores.masked_fill(future_mask(steps), float(\"-inf\")) when self.causal is "
        "True, and the unchanged scores when it is False. A conditional expression, a if condition else b, fits "
        "this on one line.",
        "generate": "Inside torch.no_grad(), loop steps times. Each time, model(prompt)[:, -1].argmax(dim=-1, "
        "keepdim=True) gives the most likely next token of each sequence. Add it to the end of prompt with "
        "torch.cat along dim=1.",
        "copies_right": "Call generate on sequences[:, :half] for half steps. Compare the written second half with "
        "the true second half, and use .all(dim=1) to get one True or False per sequence. Then take the mean of "
        "those values as floats.",
    },
    introduces=(),
    check=check_mask2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
