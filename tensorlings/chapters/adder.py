"""46_adder, Section 6's project, sums written as text, a GRU encoder and decoder that adds them, and where it breaks."""

from __future__ import annotations

import textwrap

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
    answers = torch.tensor([encode(("^" + answer + "$").ljust(digits + 3, "_")) for _, answer in pairs])

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
    right = [decode(ids) == answer for ids, (_, answer) in zip(written.tolist(), pairs)]

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
        steps = ADDER_STEPS
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

            if (step + 1) % 2000 == 0:
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
    however low the loss looks. Here is the run behind this project, on the
    held out sums.

    ADDER2_TABLE

    ADDER2_TIME
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
    """An untrained model gets no sum right, so 12 of the 20 pairs are given the answers it writes itself,
    and exact_match should come out at 0.6."""
    model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
    pairs = reference_random_sums(20, 3, torch.Generator().manual_seed(465), True)
    questions, _ = reference_to_tensors(pairs, 3)

    with torch.no_grad():
        written = ADDER_REFERENCE["greedy"](model, questions, 5)

    own = [reference_decode(ids) for ids in written.tolist()]
    pairs = [(question, own[index] if index < 12 else own[index] + "1") for index, (question, _) in enumerate(pairs)]
    actual = found(model, pairs, 3)

    if not model.training:
        raise Mismatch("exact_match left the model in eval mode. Call model.train() before returning.")

    expect_close(
        call="exact_match on 20 sums, 12 of whose answers the model writes",
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
