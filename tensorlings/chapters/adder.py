"""46_adder, Section 6's project, sums written as text, a GRU encoder and decoder that adds them, and where it breaks."""

from __future__ import annotations

import torch
from torch import Tensor

from ..course import Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_value,
    seeded,
)
from ..text import prose, source_file

ADDER = Chapter(
    folder="46_adder",
    title="Project: an adder that reads",
    readme=source_file(
        """
        # Project: an adder that reads

        Section 6's project. A model reads a sum written as text, like
        `123+456`, and writes the answer one character at a time. It brings
        together an encoder and a decoder, gated RNNs, and greedy and beam
        search, and it shows where a model that learned from examples stops
        working. The training runs for minutes outside the app.

        ## Further reading

        - [Zaremba and Sutskever, learning to execute](https://arxiv.org/abs/1410.4615)
        - [Sutskever, Vinyals and Le, sequence to sequence learning](https://arxiv.org/abs/1409.3215)
        - [Keras, sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn/)
        """,
    ),
)

VOCABULARY = "_^$+0123456789"


def reference_encode(text: str) -> list[int]:
    return [VOCABULARY.index(character) for character in text]


def reference_decode(ids: list[int]) -> str:
    text = "".join(VOCABULARY[index] for index in ids).split("$")[0]

    return text.replace("_", "").replace("^", "")


def is_held_out(
    a: int,
    b: int,
) -> bool:
    return (a * 7 + b * 13) % 10 == 0


def reference_random_sums(
    count: int,
    digits: int,
    generator: torch.Generator,
    held_out: bool,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    while len(pairs) < count:
        a, b = torch.randint(0, 10**digits, (2,), generator=generator).tolist()

        if is_held_out(a, b) == held_out:
            pairs.append((f"{a}+{b}", str(a + b)))

    return pairs


def reference_to_tensors(
    pairs: list[tuple[str, str]],
    digits: int,
) -> tuple[Tensor, Tensor]:
    width = 2 * digits + 1
    questions = torch.tensor([reference_encode(question.rjust(width, "_")) for question, _ in pairs])
    answers = torch.tensor([reference_encode(("^" + answer + "$").ljust(digits + 3, "_")) for _, answer in pairs])

    return questions, answers


# ---------------------------------------------------------------------------
# adder1
# ---------------------------------------------------------------------------

ADDER1_LESSON = prose(
    """
    This is Section 6's project. The model reads a sum written as text, like
    `123+456`, and writes the answer, `579`, one character at a time.
    Zaremba and Sutskever trained networks to run small programs this way.

    A model reads text as a sequence of **tokens**, the units it takes one at
    a time, each turned into an id. Here every character is one token, from a
    vocabulary of 14.

    ```python
    VOCABULARY = "_^$+0123456789"
    VOCABULARY.index("7")    # 11
    ```

    Three of them never appear in a sum. `_` pads a short sum out to the full
    width, `^` tells the decoder to start writing, and `$` marks the end of the
    answer.

    A question goes in padded on the left, like `__12+45`. rnn2 picked each
    sequence's last real step when padding sat on the right. Padding on the
    left makes the last step real for every sum, so the encoder's final state
    has read the whole question.

    The answer comes out as `^57$`, padded on the right to the longest answer.
    The decoder reads `^57` and learns to write `57$`, the same tokens moved
    one step along, which is encdec1's teacher forcing.

    Numbers of up to 3 digits make a million different sums. One sum in ten
    is held out by a fixed rule, `is_held_out`, and never trained on. So the
    test asks for sums the model has never seen, which it can't answer from
    memory.
    """,
)

ADDER1_SOURCE = source_file(
    '''
    # adder1.py

    import torch
    from torch import Tensor

    VOCABULARY = "_^$+0123456789"
    seven = VOCABULARY.index("7")       # 11
    padded = "12+45".rjust(7, "_")      # '__12+45'


    # The fixed rule that keeps one sum in ten out of training.
    def is_held_out(
        a: int,
        b: int,
    ) -> bool:
        return (a * 7 + b * 13) % 10 == 0


    # TODO: Return the id of every character of text.
    def encode(text: str) -> list[int]:
        ...


    # TODO: Return the text that ids spell up to the first $, without any _ or
    #   ^, so a model's output reads as the answer alone.
    def decode(ids: list[int]) -> str:
        ...


    # TODO: Return count sums as (question, answer) pairs of text, like
    #   ("123+45", "168"). Draw both numbers at once with
    #   torch.randint(0, 10 ** digits, (2,), generator=generator), and keep a
    #   pair only when is_held_out gives held_out for it.
    def random_sums(
        count: int,
        digits: int,
        generator: torch.Generator,
        held_out: bool,
    ) -> list[tuple[str, str]]:
        ...


    # TODO: Return the questions and the answers as two tensors of ids. Each
    #   question is padded with _ on the left to 2 * digits + 1 characters. Each
    #   answer gets ^ before it and $ after it, then _ on the right up to
    #   digits + 3 characters.
    def to_tensors(
        pairs: list[tuple[str, str]],
        digits: int,
    ) -> tuple[Tensor, Tensor]:
        ...
    ''',
)

ADDER1_SOLUTION = source_file(
    '''
    # adder1.py, a solution for comparison

    import torch
    from torch import Tensor

    VOCABULARY = "_^$+0123456789"
    seven = VOCABULARY.index("7")
    padded = "12+45".rjust(7, "_")


    def is_held_out(
        a: int,
        b: int,
    ) -> bool:
        return (a * 7 + b * 13) % 10 == 0


    def encode(text: str) -> list[int]:
        return [VOCABULARY.index(character) for character in text]


    def decode(ids: list[int]) -> str:
        text = "".join(VOCABULARY[index] for index in ids).split("$")[0]

        return text.replace("_", "").replace("^", "")


    def random_sums(
        count: int,
        digits: int,
        generator: torch.Generator,
        held_out: bool,
    ) -> list[tuple[str, str]]:
        pairs = []

        while len(pairs) < count:
            a, b = torch.randint(0, 10**digits, (2,), generator=generator).tolist()

            if is_held_out(a, b) == held_out:
                pairs.append((f"{a}+{b}", str(a + b)))

        return pairs


    def to_tensors(
        pairs: list[tuple[str, str]],
        digits: int,
    ) -> tuple[Tensor, Tensor]:
        width = 2 * digits + 1
        questions = torch.tensor([encode(question.rjust(width, "_")) for question, _ in pairs])
        answers = torch.tensor([encode(("^" + answer + "$").ljust(digits + 3, "_")) for _, answer in pairs])

        return questions, answers
    ''',
)


def expect_two_tensors(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the questions and the answers as two tensors in a tuple.")

    for name, part, wanted in zip(("questions", "answers"), actual, expected):
        expect_value(
            call=f"the {name} of {call}",
            actual=part.tolist() if isinstance(part, Tensor) else part,
            expected=wanted.tolist(),
        )


def check_adder1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="encode",
        description="gives each character's id",
        inputs=(
            ("__12+45",),
            ("^901$__",),
        ),
        reference=reference_encode,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="decode",
        description="reads the answer up to the first $",
        inputs=(
            ([1, 9, 11, 2, 0, 0],),
            ([1, 5, 4, 4, 13, 2, 7, 2],),
            ([1, 6, 8],),
        ),
        reference=reference_decode,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="random_sums",
        description="draws sums on the right side of the held out rule",
        inputs=(
            (4, 3, seeded(460), False),
            (3, 2, seeded(461), True),
        ),
        reference=reference_random_sums,
        expect=lambda call, actual, expected: expect_value(
            call=call,
            actual=[tuple(pair) for pair in actual] if isinstance(actual, list) else actual,
            expected=expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="to_tensors",
        description="pads the questions on the left and the answers on the right",
        inputs=(
            ([("12+45", "57"), ("999+1", "1000"), ("0+0", "0")], 3),
            ([("7+8", "15"), ("10+90", "100")], 2),
        ),
        reference=reference_to_tensors,
        expect=expect_two_tensors,
    )


ADDER1 = Exercise(
    name="adder1",
    chapter=ADDER,
    title="Project: sums as tokens",
    lesson=ADDER1_LESSON,
    source=ADDER1_SOURCE,
    solution=ADDER1_SOLUTION,
    hints={
        "encode": "VOCABULARY.index(character) for each character of text, in a list.",
        "decode": "Join VOCABULARY[index] for each id, keep the part before the first $ with .split(\"$\")[0], then "
        "take out _ and ^ with .replace.",
        "random_sums": "Loop while the list is shorter than count. Turn the two numbers into text with an f-string, "
        "and the answer with str(a + b).",
        "to_tensors": "question.rjust(width, \"_\") pads on the left, and (\"^\" + answer + \"$\").ljust(digits + 3, "
        "\"_\") on the right. Encode each one and pass the lists of ids to torch.tensor.",
    },
    introduces=("token",),
    check=check_adder1,
)
