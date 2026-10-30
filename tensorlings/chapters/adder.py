"""46_adder, Section 6's project, sums written as text, a GRU encoder and decoder that adds them, and where it breaks."""

from __future__ import annotations

import textwrap
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    seeded,
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, load_checkpoint
from .linear_layers import expect_module
from .seq2seq import BEAM_SEARCH_GIVEN


def indented_part(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")

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
    return (a + 3 * b) % 11 == 0


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
    answers = torch.tensor([reference_encode(("^" + answer[::-1] + "$").ljust(digits + 3, "_")) for _, answer in pairs])

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

    The answer is written backwards, units first, so 57 comes out as `^75$`,
    padded on the right to the longest answer. That is the order you add in by
    hand, where each carry moves from a digit to the next one written. adder2
    measures how much that order matters.

    The decoder reads `^75` and learns to write `75$`, the same tokens moved
    one step along, which is encdec1's teacher forcing.

    Numbers of up to 3 digits make a million different sums. One sum in 11
    is held out by a fixed rule, `is_held_out`, and never trained on. So the
    test asks for sums the model has never seen, which it can't answer from
    memory.

    The rule has to mix every digit. A first try held out sums by their last
    digits alone, so last digits like 3 and 9 never met in training. The
    model then got only about half of the held out sums right, which tested a
    different skill from adding.
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
    backwards = "57"[::-1]              # '75'


    # The fixed rule that keeps one sum in 11 out of training.
    def is_held_out(
        a: int,
        b: int,
    ) -> bool:
        return (a + 3 * b) % 11 == 0


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
    #   answer is reversed, gets ^ before it and $ after it, then _ on the
    #   right up to digits + 3 characters.
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
    backwards = "57"[::-1]


    def is_held_out(
        a: int,
        b: int,
    ) -> bool:
        return (a + 3 * b) % 11 == 0


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
        answers = torch.tensor([encode(("^" + answer[::-1] + "$").ljust(digits + 3, "_")) for _, answer in pairs])

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
        "to_tensors": "question.rjust(width, \"_\") pads on the left, and (\"^\" + answer[::-1] + \"$\").ljust(digits + 3, "
        "\"_\") on the right. Encode each one and pass the lists of ids to torch.tensor.",
    },
    introduces=("token",),
    check=check_adder1,
)


# ---------------------------------------------------------------------------
# adder2
# ---------------------------------------------------------------------------

# adder1's pieces, which adder2 and adder3 are handed, written once so the files never drift apart.
ADDER_PARTS = '''
VOCABULARY = "_^$+0123456789"


def is_held_out(
    a: int,
    b: int,
) -> bool:
    return (a + 3 * b) % 11 == 0


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
    answers = torch.tensor([encode(("^" + answer[::-1] + "$").ljust(digits + 3, "_")) for _, answer in pairs])

    return questions, answers
'''

ADDER_INIT = '''
class Adder(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCABULARY), 32)
        self.encoder = nn.GRU(32, hidden, batch_first=True)
        self.decoder = nn.GRU(32, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(VOCABULARY))
'''

ADDER_FORWARD = '''
    def forward(
        self,
        questions: Tensor,
        answers_so_far: Tensor,
    ) -> Tensor:
        _, state = self.encoder(self.embedding(questions))
        outputs, _ = self.decoder(self.embedding(answers_so_far), state)

        return self.head(outputs)
'''

GREEDY_SOLUTION = '''
def greedy(
    model: nn.Module,
    questions: Tensor,
    length: int,
) -> Tensor:
    _, state = model.encoder(model.embedding(questions))
    token = torch.full((len(questions), 1), VOCABULARY.index("^"))
    written = []

    for _ in range(length):
        outputs, state = model.decoder(model.embedding(token), state)
        token = model.head(outputs).argmax(dim=2)
        written.append(token)

    return torch.cat(written, dim=1)
'''

EXACT_MATCH_SOLUTION = '''
def exact_match(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    digits: int,
) -> float:
    questions, _ = to_tensors(pairs, digits)
    model.eval()

    with torch.no_grad():
        written = greedy(model, questions, digits + 2)

    model.train()
    right = [decode(ids)[::-1] == answer for ids, (_, answer) in zip(written.tolist(), pairs)]

    return sum(right) / len(pairs)
'''

INDENTED_ADDER_PARTS = textwrap.indent(ADDER_PARTS.strip("\n"), "    ")
INDENTED_ADDER_INIT = textwrap.indent(ADDER_INIT.strip("\n"), "    ")
INDENTED_ADDER_FORWARD = textwrap.indent(ADDER_FORWARD.strip("\n"), "    ")
INDENTED_GREEDY = textwrap.indent(GREEDY_SOLUTION.strip("\n"), "    ")
INDENTED_EXACT_MATCH = textwrap.indent(EXACT_MATCH_SOLUTION.strip("\n"), "    ")

ADDER2_MAIN = '''
    # Run by .venv/bin/python run.py train adder2, never by the app.
    def main() -> None:
        device = best_device()
        torch.manual_seed(0)
        model = Adder().to(device)
        steps = 4000
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        generator = torch.Generator().manual_seed(0)
        held_out = random_sums(2000, 3, torch.Generator().manual_seed(1), True)
        print(f"Training on {device} for {steps} steps.")

        for step in range(steps):
            questions, answers = to_tensors(random_sums(256, 3, generator, False), 3)
            loss = answer_loss(model, questions.to(device), answers.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % 1000 == 0:
                model.cpu()
                print(f"Step {step + 1}, loss {loss.item():.4f}, held out sums right {exact_match(model, held_out, 3):.1%}")
                model.to(device)

        save_checkpoint(model, "adder2")
        print("Saved the model in checkpoints/adder2.pt. Save adder2.py in the app to check it.")
'''

ANSWER_LOSS_SOLUTION = '''
def answer_loss(
    model: nn.Module,
    questions: Tensor,
    answers: Tensor,
) -> Tensor:
    scores = model(questions, answers[:, :-1])

    return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), answers[:, 1:].reshape(-1), ignore_index=0)
'''

INDENTED_ANSWER_LOSS = textwrap.indent(ANSWER_LOSS_SOLUTION.strip("\n"), "    ")

ADDER2_LESSON = prose(
    """
    This step trains for minutes outside the app, as cifar2 did. Once the
    checks above the last one pass, train from the project folder with
    `.venv/bin/python run.py train adder2`.

    The model is encdec1's encoder and decoder, with a GRU for each. One
    embedding of the 14 tokens serves both. The encoder reads the padded
    question, and its final state starts the decoder.

    Training uses teacher forcing. The decoder reads the right answer so far,
    `^57`, and is scored on every next token at once, `57$`. The padding after
    `$` says nothing about the answer, so it takes no part in the loss.

    ```python
    F.cross_entropy(scores, targets, ignore_index=0)    # targets of 0 count for nothing
    ```

    `F.cross_entropy` wants one row of scores per prediction. So the scores
    of shape `(batch, steps, 14)` become `(batch * steps, 14)` with `.reshape`,
    and the targets become one long row the same way.

    Writing an answer goes one token at a time, as greedy1 did. The decoder
    starts from `^`, takes its most likely token, and feeds it back in.

    Exact match is the honest score. An answer with one wrong digit is wrong,
    however low the loss looks. Here is the run behind this project on 2,000
    held out sums. Next to it is the same model, trained the same way, on
    answers written in the usual order.

    | Steps | 1,000 | 2,000 | 3,000 | 4,000 |
    |---|---|---|---|---|
    | Units first | 88.8% | 99.7% | 100.0% | 100.0% |
    | Usual order | 4.7% | 9.9% | 21.8% | 29.8% |

    In the usual order the first digit written depends on every carry below
    it, before any of them is written. Even after 20,000 steps, five times as
    long, the usual order gets 69.8% right. Choosing the order of the output
    is a design decision, and here it beat a lot of training.

    The run took about a minute here on MPS.
    """,
)

ADDER2_SOURCE = source_file(
    '''
    # adder2.py
    #
    # Train once the checks above the last one pass, from the project folder.
    #
    #     .venv/bin/python run.py train adder2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import best_device, save_checkpoint

    scores = torch.zeros(2, 5, 14)
    rows = scores.reshape(-1, 14)    # shape (10, 14), one row per prediction


    # adder1's pieces.
''' + INDENTED_ADDER_PARTS + '''


    # TODO: Finish forward. The encoder reads the embedded questions, and its
    #   final state starts the decoder, which reads the embedded answers so
    #   far. Return the head's scores for every step, shape (batch, steps, 14).
''' + INDENTED_ADDER_INIT + '''

        def forward(
            self,
            questions: Tensor,
            answers_so_far: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the cross entropy of model's scores against answers moved one
    #   step along. The decoder reads answers[:, :-1] and should give
    #   answers[:, 1:]. Padding, id 0, takes no part.
    def answer_loss(
        model: nn.Module,
        questions: Tensor,
        answers: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the ids model writes for questions, shape (batch, length).
    #   Start every row from the ^ token, run the decoder one step at a time,
    #   and feed each step's most likely token back in.
    def greedy(
        model: nn.Module,
        questions: Tensor,
        length: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of pairs whose answer model writes exactly right,
    #   with greedy for digits + 2 steps and decode, in eval mode and without
    #   recording for autograd. Then put model back in training mode.
    def exact_match(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
    ) -> float:
        ...

''' + ADDER2_MAIN,
)

ADDER2_SOLUTION = source_file(
    '''
    # adder2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.training import best_device, save_checkpoint

    scores = torch.zeros(2, 5, 14)
    rows = scores.reshape(-1, 14)


''' + INDENTED_ADDER_PARTS + '''


''' + INDENTED_ADDER_INIT + '''

''' + INDENTED_ADDER_FORWARD + '''


''' + INDENTED_ANSWER_LOSS + '''


''' + INDENTED_GREEDY + '''


''' + INDENTED_EXACT_MATCH + '''

''' + ADDER2_MAIN,
)


def adder_namespace() -> dict[str, object]:
    """adder1's pieces and the solution's model and functions, run once, so the checks compare against
    exactly the code the learner's file holds."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(ADDER_PARTS + ADDER_INIT + ADDER_FORWARD + ANSWER_LOSS_SOLUTION + GREEDY_SOLUTION + EXACT_MATCH_SOLUTION, namespace)

    return namespace


ADDER_REFERENCE = adder_namespace()


def seeded_adder(
    make: type,
    hidden: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(hidden)


def small_sums(seed: int) -> list[tuple[str, str]]:
    return reference_random_sums(6, 3, torch.Generator().manual_seed(seed), False)


def check_adder_forward(found: type) -> None:
    model = expect_module("Adder(16)", seeded_adder(found, 16))
    questions, answers = reference_to_tensors(small_sums(462), 3)

    with torch.no_grad():
        expected = seeded_adder(ADDER_REFERENCE["Adder"], 16)(questions, answers[:, :-1])

    expect_tensor(
        call="Adder(16) on 6 sums",
        actual=model(questions, answers[:, :-1]),
        expected=expected,
    )


def check_answer_loss(found: object) -> None:
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    questions, answers = reference_to_tensors(small_sums(463), 3)

    with torch.no_grad():
        expected = ADDER_REFERENCE["answer_loss"](model, questions, answers).item()
        actual = found(model, questions, answers)

    expect_close(
        call="answer_loss on 6 sums",
        actual=actual.item() if isinstance(actual, Tensor) else actual,
        expected=expected,
        tolerance=1e-5,
    )


def check_greedy(found: object) -> None:
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    questions, _ = reference_to_tensors(small_sums(464), 3)

    with torch.no_grad():
        expected = ADDER_REFERENCE["greedy"](model, questions, 5)
        actual = found(model, questions, 5)

    expect_tensor(
        call="greedy for 5 steps on 6 sums",
        actual=actual,
        expected=expected,
    )


def check_exact_match(found: object) -> None:
    """The file's greedy is swapped for one that writes 12 of 20 answers right, backwards as the model
    would, and the first digit of the rest wrong, so exact_match should come out at 0.6."""
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    pairs = reference_random_sums(20, 3, torch.Generator().manual_seed(465), True)
    calls: list[tuple[bool, bool, int]] = []

    def scripted_greedy(
        model: nn.Module,
        questions: Tensor,
        length: int,
    ) -> Tensor:
        calls.append((model.training, torch.is_grad_enabled(), length))
        rows = []

        for index, (_, answer) in enumerate(pairs):
            written = answer[::-1]

            if index >= 12:
                written = str((int(written[0]) + 1) % 10) + written[1:]

            rows.append(reference_encode((written + "$").ljust(length, "_")))

        return torch.tensor(rows)

    namespace = found.__globals__
    original = namespace.get("greedy")
    namespace["greedy"] = scripted_greedy

    try:
        actual = found(model, pairs, 3)
    finally:
        namespace["greedy"] = original

    if not calls:
        raise Mismatch("exact_match never called greedy to write the answers.")

    training, recording, length = calls[0]

    if training:
        raise Mismatch("exact_match ran greedy with the model in training mode. Call model.eval() first.")

    if recording:
        raise Mismatch("exact_match ran greedy while recording for autograd. Run it inside torch.no_grad().")

    if length != 5:
        raise Mismatch(f"exact_match asked greedy for {length} steps, but it should ask for digits + 2, here 5.")

    if not model.training:
        raise Mismatch("exact_match left the model in eval mode. Call model.train() before returning.")

    expect_close(
        call="exact_match on 20 sums, 12 of them written right",
        actual=actual,
        expected=0.6,
    )


def check_trained_adder(found: type) -> None:
    model = seeded_adder(found, 128)

    try:
        load_checkpoint(model, "adder2")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit Adder as it is now. Train it again from the project folder.\n\n"
            "    .venv/bin/python run.py train adder2",
        ) from None

    pairs = reference_random_sums(2000, 3, torch.Generator().manual_seed(1), True)
    right = ADDER_REFERENCE["exact_match"](model, pairs, 3)

    if right < 0.95:
        raise Mismatch(f"The trained model gets {right:.1%} of 2,000 held out sums exactly right, but it should pass 95%.")


def check_adder2(checker: Checker) -> None:
    checker.test_method(
        class_name="Adder",
        method="forward",
        description="encodes the question and decodes the answer so far",
        run=check_adder_forward,
    )
    checker.test(
        function="answer_loss",
        description="scores every next token with teacher forcing, padding left out",
        run=check_answer_loss,
    )
    checker.test(
        function="greedy",
        description="writes one token at a time from ^",
        run=check_greedy,
    )
    checker.test(
        function="exact_match",
        description="gives the share of answers written exactly right",
        run=check_exact_match,
    )
    checker.test_method(
        class_name="Adder",
        method="forward",
        description="trained, gets 95 percent of held out sums exactly right",
        run=check_trained_adder,
    )


ADDER2 = Exercise(
    name="adder2",
    chapter=ADDER,
    title="Project: an encoder and decoder that add",
    lesson=ADDER2_LESSON,
    source=ADDER2_SOURCE,
    solution=ADDER2_SOLUTION,
    hints={
        "Adder.forward": "_, state = self.encoder(self.embedding(questions)). Pass self.embedding(answers_so_far) and "
        "state to self.decoder, and run self.head on its outputs.",
        "answer_loss": "scores = model(questions, answers[:, :-1]). Reshape them to (-1, 14) and answers[:, 1:] to "
        "(-1), then F.cross_entropy with ignore_index=0.",
        "greedy": "Start token as torch.full((len(questions), 1), VOCABULARY.index(\"^\")). Each step, run the decoder "
        "on its embedding with the state, take the head's argmax over dim=2, keep it, and torch.cat at the end.",
        "exact_match": "to_tensors gives the questions. Compare decode of each row of greedy's ids, as a list, with the "
        "answer of its pair.",
    },
    introduces=(),
    check=check_adder2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# adder3
# ---------------------------------------------------------------------------

# beam1's search, with the adder's own start and end tokens and the score it ranks finished answers by.
BEAM_PARTS = '''
START = VOCABULARY.index("^")
END = VOCABULARY.index("$")


def sequence_score(
    total: float,
    length: int,
    alpha: float,
) -> float:
    return total / length ** alpha
''' + "\n\n" + BEAM_SEARCH_GIVEN

ADDER_STEP_SOLUTION = '''
def adder_step(model: nn.Module) -> Callable[[int, Tensor], tuple[Tensor, Tensor]]:
    @torch.no_grad()
    def step(
        symbol: int,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        outputs, state = model.decoder(model.embedding(torch.tensor([[symbol]])), state)

        return F.log_softmax(model.head(outputs)[0, -1], dim=0), state

    return step
'''

BEAM_ANSWER_SOLUTION = '''
def beam_answer(
    model: nn.Module,
    question: str,
    width: int,
    digits: int,
) -> str:
    questions = torch.tensor([encode(question.rjust(2 * digits + 1, "_"))])

    with torch.no_grad():
        _, state = model.encoder(model.embedding(questions))

    ids = beam_search(adder_step(model), state, width, digits + 2, 0.0)

    return decode(ids)[::-1]
'''

ADDER3_LESSON = prose(
    """
    Two questions about the adder from adder2. Does beam search help it, and
    does it add numbers longer than any it trained on?

    beam1's beam search keeps the few likeliest answers so far, where greedy
    keeps only the likeliest token. It helps a model that is unsure, as on
    beam1's copying task. Here greedy decoding already gets all 2,000 held out
    sums right, and a beam of 3 gets the same 100%. Beam search can't fix what
    isn't wrong, and it costs a decoder run per beam per step.

    Longer numbers are another story.

    | Sums | Exactly right |
    |---|---|
    | up to 3 digits, as in training | 100% |
    | the same sums with two more `_` in front | 0.15% |
    | up to 4 digits | 0% |
    | up to 5 digits | 0% |

    Two more padding tokens are enough to break it. The model never saw a
    question of 9 tokens, so its encoder ends in a state the decoder never
    learned to read. With 4 digit numbers it also writes at most 4 digits, the
    longest answer it ever wrote, so `4426+7860` comes out as `1205`.

    The model learned the sums it was shown, and not the method of adding. A
    person who adds 3 digit numbers adds 30 digit numbers the same way.
    Getting a network to do that, to work past the lengths it trained on, is
    still studied today. Zaremba and Sutskever trained with a curriculum that
    grew the lengths step by step.
    """,
)

ADDER3_SOURCE = source_file(
    '''
    # adder3.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    chances = F.log_softmax(torch.tensor([0.0, 1.0]), dim=0)    # tensor([-1.3133, -0.3133])


    # adder1's pieces.
''' + INDENTED_ADDER_PARTS + '''


    # Adder, greedy and exact_match from adder2.
''' + INDENTED_ADDER_INIT + '''

''' + INDENTED_ADDER_FORWARD + '''


''' + INDENTED_GREEDY + '''


''' + INDENTED_EXACT_MATCH + '''


    # beam1's search, with the adder's start and end tokens.
''' + indented_part(BEAM_PARTS) + '''


    # TODO: Return a step function for beam_search from model, as beam1's
    #   decoder_step does. It runs the decoder on one token, a tensor of shape
    #   (1, 1), from state, and returns the log_softmax of the head's scores,
    #   of shape (14,), and the new state. Record nothing for autograd.
    def adder_step(model: nn.Module) -> Callable[[int, Tensor], tuple[Tensor, Tensor]]:
        ...


    # TODO: Return model's answer to question, a text like "123+45", from
    #   beam_search with width, up to digits + 2 steps and an alpha of 0. Pad
    #   and encode the question as to_tensors does, start the search from the
    #   encoder's final state, and read the ids back as a forward answer.
    def beam_answer(
        model: nn.Module,
        question: str,
        width: int,
        digits: int,
    ) -> str:
        ...


    # TODO: Return the share of pairs whose answer beam_answer gets exactly
    #   right with width.
    def beam_exact_match(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
        width: int,
    ) -> float:
        ...


    # TODO: Return exact_match for count held out sums of each number of digits
    #   in digit_counts, in order, each drawn with random_sums and generator.
    def accuracy_by_digits(
        model: nn.Module,
        digit_counts: list[int],
        count: int,
        generator: torch.Generator,
    ) -> list[float]:
        ...
    ''',
)

ADDER3_SOLUTION = source_file(
    '''
    # adder3.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    chances = F.log_softmax(torch.tensor([0.0, 1.0]), dim=0)


''' + INDENTED_ADDER_PARTS + '''


''' + INDENTED_ADDER_INIT + '''

''' + INDENTED_ADDER_FORWARD + '''


''' + INDENTED_GREEDY + '''


''' + INDENTED_EXACT_MATCH + '''


''' + indented_part(BEAM_PARTS) + '''


''' + indented_part(ADDER_STEP_SOLUTION) + '''


''' + indented_part(BEAM_ANSWER_SOLUTION) + '''


    def beam_exact_match(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
        width: int,
    ) -> float:
        right = [beam_answer(model, question, width, digits) == answer for question, answer in pairs]

        return sum(right) / len(pairs)


    def accuracy_by_digits(
        model: nn.Module,
        digit_counts: list[int],
        count: int,
        generator: torch.Generator,
    ) -> list[float]:
        return [exact_match(model, random_sums(count, digits, generator, True), digits) for digits in digit_counts]
    ''',
)


def beam_namespace() -> dict[str, object]:
    namespace = dict(ADDER_REFERENCE)
    namespace["Callable"] = Callable
    exec(BEAM_PARTS + ADDER_STEP_SOLUTION + BEAM_ANSWER_SOLUTION, namespace)

    return namespace


BEAM_REFERENCE = beam_namespace()


def check_adder_step(found: object) -> None:
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    questions, _ = reference_to_tensors(small_sums(466), 3)

    with torch.no_grad():
        _, state = model.encoder(model.embedding(questions[:1]))

    step = found(model)

    if not callable(step):
        raise Mismatch("adder_step(model) should return a function, the step that beam_search calls.")

    log_probs, next_state = step(VOCABULARY.index("^"), state)
    expected_log_probs, expected_state = BEAM_REFERENCE["adder_step"](model)(VOCABULARY.index("^"), state)
    expect_tensor(
        call="the log probabilities of adder_step(model)(the ^ token, the encoder's state)",
        actual=log_probs,
        expected=expected_log_probs,
    )
    expect_tensor(
        call="the state it returns",
        actual=next_state,
        expected=expected_state,
    )


def check_beam_answer(found: object) -> None:
    """The file's beam_search is swapped for one that records what it is given and writes 2 then 1,
    so beam_answer should read that back as 12."""
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    calls: list[tuple[Callable[[int, Tensor], tuple[Tensor, Tensor]], Tensor, int, int, float]] = []

    def scripted_beam_search(
        step: Callable[[int, Tensor], tuple[Tensor, Tensor]],
        state: Tensor,
        width: int,
        max_length: int,
        alpha: float,
    ) -> list[int]:
        calls.append((step, state, width, max_length, alpha))

        return reference_encode("21")

    namespace = found.__globals__
    original = namespace.get("beam_search")
    namespace["beam_search"] = scripted_beam_search

    try:
        actual = found(model, "907+8", 2, 3)
    finally:
        namespace["beam_search"] = original

    if not calls:
        raise Mismatch("beam_answer never called beam_search.")

    step, state, width, max_length, alpha = calls[0]

    if (width, max_length, alpha) != (2, 5, 0.0):
        raise Mismatch(
            f"beam_answer passed beam_search a width of {width}, {max_length} steps and an alpha of {alpha}, "
            "but for a width of 2 and 3 digits it should pass 2, 5 and 0.0.",
        )

    with torch.no_grad():
        _, expected_state = model.encoder(model.embedding(torch.tensor([reference_encode("__907+8")])))

    expect_tensor(
        call='the state beam_answer starts the search from, for "907+8" at 3 digits',
        actual=state,
        expected=expected_state,
    )
    expect_tensor(
        call="the log probabilities of the step function it passes, from ^",
        actual=step(VOCABULARY.index("^"), expected_state)[0],
        expected=BEAM_REFERENCE["adder_step"](model)(VOCABULARY.index("^"), expected_state)[0],
    )
    expect_value(
        call='beam_answer when the search writes the tokens 2 then 1',
        actual=actual,
        expected="12",
    )


def check_beam_exact_match(found: object) -> None:
    """The file's beam_answer is swapped for one that gets 12 of 20 answers right, so the share is 0.6."""
    pairs = reference_random_sums(20, 3, torch.Generator().manual_seed(467), True)
    answers = dict(pairs)
    calls: list[tuple[int, int]] = []

    def scripted_beam_answer(
        model: nn.Module,
        question: str,
        width: int,
        digits: int,
    ) -> str:
        calls.append((width, digits))
        position = [asked for asked, _ in pairs].index(question)

        return answers[question] if position < 12 else answers[question] + "0"

    namespace = found.__globals__
    original = namespace.get("beam_answer")
    namespace["beam_answer"] = scripted_beam_answer

    try:
        actual = found(None, pairs, 3, 4)
    finally:
        namespace["beam_answer"] = original

    if any(call != (4, 3) for call in calls):
        raise Mismatch("beam_exact_match should pass its width and digits on to beam_answer.")

    expect_close(
        call="beam_exact_match on 20 sums, 12 of them answered right",
        actual=actual,
        expected=0.6,
    )


def check_accuracy_by_digits(found: object) -> None:
    """The file's exact_match is swapped for one that records the sums it gets and answers digits / 10."""
    seen: list[tuple[list[tuple[str, str]], int]] = []

    def scripted_exact_match(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
    ) -> float:
        seen.append((pairs, digits))

        return digits / 10

    namespace = found.__globals__
    original = namespace.get("exact_match")
    namespace["exact_match"] = scripted_exact_match

    try:
        actual = found(None, [3, 5, 4], 6, torch.Generator().manual_seed(468))
    finally:
        namespace["exact_match"] = original

    expect_value(
        call="accuracy_by_digits(model, [3, 5, 4], 6, a generator seeded with 468)",
        actual=actual,
        expected=[0.3, 0.5, 0.4],
    )

    generator = torch.Generator().manual_seed(468)
    expected_pairs = [(reference_random_sums(6, digits, generator, True), digits) for digits in (3, 5, 4)]

    if seen != expected_pairs:
        raise Mismatch(
            "accuracy_by_digits should draw count held out sums with random_sums for each number of digits in "
            "turn, from the one generator, and pass each batch to exact_match with its digits.",
        )


def check_adder3(checker: Checker) -> None:
    checker.test(
        function="adder_step",
        description="gives the next token's log probabilities and the new state",
        run=check_adder_step,
    )
    checker.test(
        function="beam_answer",
        description="answers a question with beam search, read forwards",
        run=check_beam_answer,
    )
    checker.test(
        function="beam_exact_match",
        description="gives the share beam_answer gets right",
        run=check_beam_exact_match,
    )
    checker.test(
        function="accuracy_by_digits",
        description="measures exact_match at each number of digits",
        run=check_accuracy_by_digits,
    )


ADDER3 = Exercise(
    name="adder3",
    chapter=ADDER,
    title="Project: beam search, and where it breaks",
    lesson=ADDER3_LESSON,
    source=ADDER3_SOURCE,
    solution=ADDER3_SOLUTION,
    hints={
        "adder_step": "Inside it, define step(symbol, state) under @torch.no_grad(). Run model.decoder on "
        "model.embedding(torch.tensor([[symbol]])) and state, and return F.log_softmax(model.head(outputs)[0, -1], "
        "dim=0) with the new state.",
        "beam_answer": "encode(question.rjust(2 * digits + 1, \"_\")) in a list gives a batch of one. The encoder's "
        "final state goes to beam_search(adder_step(model), state, width, digits + 2, 0.0). Reverse decode of the ids.",
        "beam_exact_match": "Compare beam_answer(model, question, width, digits) with each answer, and divide the "
        "count of matches by len(pairs).",
        "accuracy_by_digits": "For each digits in digit_counts, exact_match(model, random_sums(count, digits, "
        "generator, True), digits).",
    },
    introduces=(),
    check=check_adder3,
    time_limit=60,
)
