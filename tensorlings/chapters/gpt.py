"""59_gpt, Section 7's project, a GPT built from your blocks, trained on tiny Shakespeare for minutes on the best device, and sampled."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_tensor,
    expect_value,
    seeded,
)
from ..text import prose, source_file

GPT = Chapter(
    folder="59_gpt",
    title="Project: a GPT that writes Shakespeare",
    readme=source_file(
        """
        # Project: a GPT that writes Shakespeare

        Section 7's project. You assemble your transformer blocks into a GPT,
        train it on a million characters of Shakespeare for minutes on the
        fastest device the machine has, and sample new text from it, as in
        Karpathy's GPT lecture and nanoGPT. Download the data once from the
        project folder.

        ```bash
        .venv/bin/python run.py data shakespeare
        ```

        ## Further reading

        - [Karpathy, let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
        - [Karpathy, nanoGPT](https://github.com/karpathy/nanoGPT)
        - [Radford and colleagues, language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        """,
    ),
)


def reference_characters_of(text: str) -> list[str]:
    return sorted(set(text))


def reference_encode(
    text: str,
    characters: list[str],
) -> Tensor:
    index = {character: position for position, character in enumerate(characters)}

    return torch.tensor([index[character] for character in text])


def reference_decode(
    ids: Tensor,
    characters: list[str],
) -> str:
    return "".join(characters[index] for index in ids.tolist())


def reference_split(
    data: Tensor,
    fraction: float,
) -> tuple[Tensor, Tensor]:
    cut = int(fraction * len(data))

    return data[:cut], data[cut:]


def reference_get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])

    return x, y


# ---------------------------------------------------------------------------
# gpt1
# ---------------------------------------------------------------------------

GPT1_LESSON = prose(
    """
    This is Section 7's project. You build a GPT and train it on tiny
    Shakespeare, a million characters of the plays, as Karpathy does in his
    GPT lecture. Download it once, about 1 MB, from the project folder with
    `.venv/bin/python run.py data shakespeare`.

    ```python
    from tensorlings.data import load_shakespeare

    text = load_shakespeare()
    len(text)        # 1115394
    text[:14]        # 'First Citizen:'
    ```

    The tokens here are characters, 65 of them, from the newline to `z`.
    58_bpe's tokens would make sequences shorter, but on a million characters
    a small vocabulary keeps the model small and quick to train. nanoGPT's
    Shakespeare example makes the same choice.

    The first 90% of the text is for training and the last 10% for validation.
    The split is one cut, not a shuffle, because overlapping stretches of text
    would put the same lines on both sides.

    A batch takes random starting points. From each, `x` is a block of
    `block_size` tokens, and `y` is the same block moved one token along. So
    every position is one example, predicting the next character from all the
    ones before it, and a block of 128 holds 128 of them.

    ```python
    data = torch.tensor([5, 6, 7, 8, 9])
    data[1:4], data[2:5]    # (tensor([6, 7, 8]), tensor([7, 8, 9]))
    ```

    A model that knows nothing gives every character the same chance, a loss
    of `log(65)`, which is 4.17. That is the number training starts from.
    """,
)

GPT1_SOURCE = source_file(
    '''
    # gpt1.py
    #
    # Section 7's project starts with the data. Download it once from the project
    # folder with
    #
    #     .venv/bin/python run.py data shakespeare

    import torch
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    text = load_shakespeare()                       # 1,115,394 characters
    index = {letter: position for position, letter in enumerate("abc")}    # {'a': 0, 'b': 1, 'c': 2}


    # TODO: Return every character that appears in text, once each, sorted.
    def characters_of(text: str) -> list[str]:
        ...


    # TODO: Return the id of every character of text, its position in
    #   characters, as a tensor. Build a dict from character to position first,
    #   since characters.index would search the list a million times.
    def encode(
        text: str,
        characters: list[str],
    ) -> Tensor:
        ...


    # TODO: Return the text that ids spell.
    def decode(
        ids: Tensor,
        characters: list[str],
    ) -> str:
        ...


    # TODO: Return the first fraction of data for training and the rest for
    #   validation, cut at int(fraction * len(data)).
    def split(
        data: Tensor,
        fraction: float,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return x and y, each of shape (batch_size, block_size). Draw the
    #   starts with torch.randint(0, len(data) - block_size, (batch_size,),
    #   generator=generator). x holds the block at each start, and y the block
    #   one token further along.
    def get_batch(
        data: Tensor,
        block_size: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...
    ''',
)

GPT1_SOLUTION = source_file(
    '''
    # gpt1.py, a solution for comparison

    import torch
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    text = load_shakespeare()
    index = {letter: position for position, letter in enumerate("abc")}


    def characters_of(text: str) -> list[str]:
        return sorted(set(text))


    def encode(
        text: str,
        characters: list[str],
    ) -> Tensor:
        positions = {character: position for position, character in enumerate(characters)}

        return torch.tensor([positions[character] for character in text])


    def decode(
        ids: Tensor,
        characters: list[str],
    ) -> str:
        return "".join(characters[index] for index in ids.tolist())


    def split(
        data: Tensor,
        fraction: float,
    ) -> tuple[Tensor, Tensor]:
        cut = int(fraction * len(data))

        return data[:cut], data[cut:]


    def get_batch(
        data: Tensor,
        block_size: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).tolist()
        x = torch.stack([data[start : start + block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])

        return x, y
    ''',
)


def expect_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give two tensors in a tuple.")

    for name, part, wanted in zip(("first", "second"), actual, expected):
        expect_tensor(
            call=f"the {name} tensor of {call}",
            actual=part,
            expected=wanted,
        )


SAMPLE_TEXT = "To be, or not to be: that is the question.\nWhether 'tis nobler in the mind to suffer"
SAMPLE_CHARACTERS = reference_characters_of(SAMPLE_TEXT)


def check_gpt1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="characters_of",
        description="gives each character once, sorted",
        inputs=(
            (SAMPLE_TEXT,),
            ("banana bread\n",),
        ),
        reference=reference_characters_of,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="encode",
        description="gives each character's position",
        inputs=(
            ("not to be", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
            ("tis nobler\n", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=reference_encode,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="decode",
        description="spells the ids back out",
        inputs=(
            (reference_encode("the question", SAMPLE_CHARACTERS), Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=reference_decode,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="split",
        description="cuts once, training first",
        inputs=(
            (torch.arange(20), 0.9),
            (torch.arange(7), 0.5),
        ),
        reference=reference_split,
        expect=expect_pair,
    )
    check_on_inputs(
        checker=checker,
        function="get_batch",
        description="gives blocks and the same blocks one token along",
        inputs=(
            (torch.arange(100, 140), 5, 3, seeded(590)),
            (torch.arange(12), 4, 6, seeded(591)),
        ),
        reference=reference_get_batch,
        expect=expect_pair,
    )


GPT1 = Exercise(
    name="gpt1",
    chapter=GPT,
    title="Project: Shakespeare as tokens",
    lesson=GPT1_LESSON,
    source=GPT1_SOURCE,
    solution=GPT1_SOLUTION,
    hints={
        "characters_of": "set(text) keeps each character once, and sorted turns it into a sorted list.",
        "encode": "Make the dict with enumerate(characters), then look every character of text up in it and pass "
        "the list to torch.tensor.",
        "decode": "characters[index] for each index of ids.tolist(), joined with \"\".join.",
        "split": "Cut at int(fraction * len(data)), and slice data before and after the cut.",
        "get_batch": "Turn the starts into a list with .tolist(). torch.stack the slices data[start : start + "
        "block_size] for x, and the slices one further along for y.",
    },
    introduces=(),
    check=check_gpt1,
    time_limit=30,
)
