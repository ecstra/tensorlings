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
from ..training import MissingCheckpoint, best_device, load_checkpoint
from .linear_layers import expect_module
from .tensors import pinned_threads


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
        together an encoder and a decoder, gated RNNs, and greedy decoding,
        and it shows where a model that learned from examples stops working. The training runs for minutes outside the app.

        ## Further reading

        - [Zaremba and Sutskever, learning to execute](https://arxiv.org/abs/1410.4615)
        - [Sutskever, Vinyals and Le, sequence to sequence learning](https://arxiv.org/abs/1409.3215)
        - [Keras, sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn/)
        - [Kaiser and Sutskever, neural GPUs learn algorithms](https://arxiv.org/abs/1511.08228)
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
    rule: Callable[[int, int], bool] = is_held_out,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    while len(pairs) < count:
        a, b = torch.randint(0, 10**digits, (2,), generator=generator).tolist()

        if rule(a, b) == held_out:
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
    digits alone, so sums whose last digits matched, like 3 and 3, never
    appeared in training. The model then got under 10 percent of the held out
    sums right, and every new sum of the kinds it trained on. That tested a
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
    This step trains for minutes outside the app, as dna2 did. Once the
    checks above the last one pass, train from the project folder with
    `.venv/bin/python run.py train adder2`.

    The model is encdec1's encoder and decoder, with a GRU for each. One
    embedding of the 14 tokens serves both. The encoder reads the padded
    question, and its final state starts the decoder.

    Training uses teacher forcing. The decoder reads the right answer so far,
    `^75` for an answer of 57, and is scored on every next token at once,
    `75$`. The padding after
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
    | Units first | 88.9% | 99.8% | 99.9% | 99.95% |
    | Usual order | 5.7% | 9.4% | 21.9% | 30.3% |

    In the usual order the first digit written depends on every carry below
    it, before any of them is written. Even after 20,000 steps, five times as
    long, the usual order gets 68.8% right. Choosing the order of the output
    is a design decision, and here it beat a lot of training.

    The run took 1.5 to 3 minutes here on the Mac's GPU.
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

PADDED_SOLUTION = '''
def padded_questions(
    pairs: list[tuple[str, str]],
    digits: int,
    extra: int,
) -> Tensor:
    width = 2 * digits + 1 + extra

    return torch.tensor([encode(question.rjust(width, "_")) for question, _ in pairs])
'''

PADDED_MATCH_SOLUTION = '''
def padded_exact_match(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    digits: int,
    extra: int,
) -> float:
    questions = padded_questions(pairs, digits, extra)
    model.eval()

    with torch.no_grad():
        written = greedy(model, questions, digits + 2)

    model.train()
    right = [decode(ids)[::-1] == answer for ids, (_, answer) in zip(written.tolist(), pairs)]

    return sum(right) / len(pairs)
'''

LONGEST_SOLUTION = '''
def longest_answer(
    model: nn.Module,
    pairs: list[tuple[str, str]],
    digits: int,
) -> int:
    questions, _ = to_tensors(pairs, digits)
    model.eval()

    with torch.no_grad():
        written = greedy(model, questions, digits + 2)

    model.train()

    return max(len(decode(ids)) for ids in written.tolist())
'''

ADDER3_LESSON = prose(
    """
    adder2's model gets every held out sum of up to 3 digits right. This
    project asks where that skill ends, by feeding it questions longer than
    any it trained on.

    Two things make a question longer. The same sum can have more `_` in
    front, which changes nothing about the answer. Or the numbers can have
    more digits. Here are 2,000 held out sums of each kind.

    | Sums | Exactly right | Longest answer written |
    |---|---|---|
    | up to 3 digits, as in training | 100% | 4 digits |
    | the same sums with one more `_` in front | 0.15% | 4 digits |
    | the same sums with two more `_` in front | 0.15% | 4 digits |
    | up to 4 digits | 0% | 4 digits |
    | up to 5 digits | 0% | 4 digits |

    One more padding token is enough to break it. The model never saw a
    question of 8 tokens, so its encoder ends in a state the decoder never
    learned to read. With 4 digit numbers it still writes at most 4 digits,
    the longest answer it ever wrote, so `5674+6652` comes out as `1236`.

    The model learned the sums it was shown, and not the method of adding. A
    person who adds 3 digit numbers adds 30 digit numbers the same way.
    Zaremba and Sutskever reached 99 percent on 9 digit sums by mixing easy
    and hard sums in training. Their tests had the same lengths as training.
    Kaiser and Sutskever's Neural GPU, built from convolutions, trained on
    binary sums of up to 20 bits. It then made no mistakes on far longer ones.
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

    padded = "907+8".rjust(9, "_")    # '____907+8'


    # adder1's pieces.
''' + INDENTED_ADDER_PARTS + '''


    # Adder, greedy and exact_match from adder2.
''' + INDENTED_ADDER_INIT + '''

''' + INDENTED_ADDER_FORWARD + '''


''' + INDENTED_GREEDY + '''


''' + INDENTED_EXACT_MATCH + '''


    # TODO: Return the questions of pairs encoded as to_tensors does, each
    #   padded on the left with "_" to 2 * digits + 1 + extra tokens, as a
    #   tensor of whole numbers of shape (len(pairs), 2 * digits + 1 + extra).
    #   to_tensors is the case extra = 0.
    def padded_questions(
        pairs: list[tuple[str, str]],
        digits: int,
        extra: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of pairs answered exactly right, as exact_match
    #   does, but with the questions from padded_questions. Write the answers
    #   with greedy for digits + 2 steps in eval mode without recording for
    #   autograd, and put the model back in training mode before returning.
    def padded_exact_match(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
        extra: int,
    ) -> float:
        ...


    # TODO: Return the number of digits in the longest answer model writes for
    #   pairs, as a plain whole number. Encode with to_tensors and write with
    #   greedy for digits + 2 steps, in eval mode without recording for
    #   autograd, then put the model back in training mode. decode drops what
    #   comes after the end token.
    def longest_answer(
        model: nn.Module,
        pairs: list[tuple[str, str]],
        digits: int,
    ) -> int:
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

    padded = "907+8".rjust(9, "_")


''' + INDENTED_ADDER_PARTS + '''


''' + INDENTED_ADDER_INIT + '''

''' + INDENTED_ADDER_FORWARD + '''


''' + INDENTED_GREEDY + '''


''' + INDENTED_EXACT_MATCH + '''


''' + indented_part(PADDED_SOLUTION) + '''


''' + indented_part(PADDED_MATCH_SOLUTION) + '''


''' + indented_part(LONGEST_SOLUTION) + '''


    def accuracy_by_digits(
        model: nn.Module,
        digit_counts: list[int],
        count: int,
        generator: torch.Generator,
    ) -> list[float]:
        return [exact_match(model, random_sums(count, digits, generator, True), digits) for digits in digit_counts]
    ''',
)


def length_namespace() -> dict[str, object]:
    namespace = dict(ADDER_REFERENCE)
    exec(PADDED_SOLUTION + PADDED_MATCH_SOLUTION + LONGEST_SOLUTION, namespace)

    return namespace


LENGTH_REFERENCE = length_namespace()


def check_padded_questions(found: object) -> None:
    for seed, digits, extra in ((469, 3, 2), (470, 2, 0), (471, 3, 5)):
        pairs = reference_random_sums(6, digits, torch.Generator().manual_seed(seed), True)
        actual = found(pairs, digits, extra)

        if isinstance(actual, Tensor) and actual.is_floating_point():
            raise Mismatch("padded_questions should give whole numbers, the token ids, not floats.")

        expect_tensor(
            call=f"padded_questions(6 sums of up to {digits} digits, {digits}, {extra})",
            actual=actual,
            expected=LENGTH_REFERENCE["padded_questions"](pairs, digits, extra),
        )


GreedyCall = tuple[bool, bool, Tensor, int]


def recording_greedy(
    written_rows: Callable[[int], list[str]],
    calls: list[GreedyCall],
) -> Callable[[nn.Module, Tensor, int], Tensor]:
    """A greedy that notes how it was run and writes the given answers, each ended by $ and padded."""

    def scripted_greedy(
        model: nn.Module,
        questions: Tensor,
        length: int,
    ) -> Tensor:
        calls.append((model.training, torch.is_grad_enabled(), questions, length))

        return torch.tensor([reference_encode((row + "$").ljust(length, "_")) for row in written_rows(length)])

    return scripted_greedy


def expect_greedy_run(
    function: str,
    calls: list[GreedyCall],
    questions: Tensor,
    digits: int,
    model: nn.Module,
) -> None:
    if not calls:
        raise Mismatch(f"{function} never called greedy to write the answers.")

    training, recording, handed, length = calls[0]

    if training:
        raise Mismatch(f"{function} ran greedy with the model in training mode. Call model.eval() first.")

    if recording:
        raise Mismatch(f"{function} ran greedy while recording for autograd. Run it inside torch.no_grad().")

    if length != digits + 2:
        raise Mismatch(f"{function} asked greedy for {length} steps, but it should ask for digits + 2, here {digits + 2}.")

    expect_tensor(
        call=f"the questions {function} handed greedy",
        actual=handed,
        expected=questions,
    )

    if not model.training:
        raise Mismatch(f"{function} left the model in eval mode. Call model.train() before returning.")


def check_padded_exact_match(found: object) -> None:
    """The file's greedy is swapped for one that writes the first answers right, backwards as the model
    would, and the first digit of the rest wrong."""
    for seed, count, digits, extra, right in ((472, 20, 3, 2, 12), (473, 13, 2, 0, 7)):
        model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
        pairs = reference_random_sums(count, digits, torch.Generator().manual_seed(seed), True)
        written = [answer[::-1] for _, answer in pairs]
        written = written[:right] + [str((int(row[0]) + 1) % 10) + row[1:] for row in written[right:]]
        calls: list[GreedyCall] = []
        namespace = found.__globals__
        original = namespace.get("greedy")
        namespace["greedy"] = recording_greedy(lambda length: written, calls)

        try:
            actual = found(model, pairs, digits, extra)
        finally:
            namespace["greedy"] = original

        expect_greedy_run(
            function="padded_exact_match",
            calls=calls,
            questions=LENGTH_REFERENCE["padded_questions"](pairs, digits, extra),
            digits=digits,
            model=model,
        )
        expect_close(
            call=f"padded_exact_match on {count} sums with {extra} extra, {right} of them written right",
            actual=actual,
            expected=right / count,
        )


def check_longest_answer(found: object) -> None:
    """The file's greedy is swapped for one that writes answers of chosen lengths."""
    for seed, digits, lengths in ((474, 3, [3, 1, 4, 2]), (475, 2, [2, 2, 1])):
        model = seeded_adder(ADDER_REFERENCE["Adder"], 16)
        pairs = reference_random_sums(len(lengths), digits, torch.Generator().manual_seed(seed), True)
        written = ["7" * size for size in lengths]
        calls: list[GreedyCall] = []
        namespace = found.__globals__
        original = namespace.get("greedy")
        namespace["greedy"] = recording_greedy(lambda length: written, calls)

        try:
            actual = found(model, pairs, digits)
        finally:
            namespace["greedy"] = original

        expect_greedy_run(
            function="longest_answer",
            calls=calls,
            questions=reference_to_tensors(pairs, digits)[0],
            digits=digits,
            model=model,
        )
        expect_value(
            call=f"longest_answer when the answers written have {lengths} digits",
            actual=actual,
            expected=max(lengths),
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
        function="padded_questions",
        description="pads each question on the left with extra more _",
        run=check_padded_questions,
    )
    checker.test(
        function="padded_exact_match",
        description="gives the share answered right with the padded questions",
        run=check_padded_exact_match,
    )
    checker.test(
        function="longest_answer",
        description="gives the length of the longest answer written",
        run=check_longest_answer,
    )
    checker.test(
        function="accuracy_by_digits",
        description="measures exact_match at each number of digits",
        run=check_accuracy_by_digits,
    )


ADDER3 = Exercise(
    name="adder3",
    chapter=ADDER,
    title="Project: where it breaks",
    lesson=ADDER3_LESSON,
    source=ADDER3_SOURCE,
    solution=ADDER3_SOLUTION,
    hints={
        "padded_questions": "to_tensors pads to a width of 2 * digits + 1. Add extra to that width, rjust each "
        "question to it, encode it, and make one tensor of the rows.",
        "padded_exact_match": "Follow exact_match line by line, with padded_questions in place of to_tensors.",
        "longest_answer": "Write the answers as exact_match does, then take the largest len(decode(ids)) over "
        "the rows of written.tolist().",
        "accuracy_by_digits": "For each digits in digit_counts, exact_match(model, random_sums(count, digits, "
        "generator, True), digits).",
    },
    introduces=(),
    check=check_adder3,
    time_limit=60,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def written_answers(
    model: nn.Module,
    questions: Tensor,
    digits: int,
) -> list[str]:
    """The answers greedy decoding writes for questions, read back in the usual order."""
    model.eval()

    with torch.no_grad():
        written = ADDER_REFERENCE["greedy"](model, questions, digits + 2)

    return [reference_decode(ids)[::-1] for ids in written.tolist()]


def measure_lengths() -> dict[str, tuple[float, int]]:
    """adder3's table, on the trained model in checkpoints/adder2.pt: for 2,000 held out sums of each kind,
    the share answered exactly right and the longest answer written, both from the questions as they were
    padded."""
    results: dict[str, tuple[float, int]] = {}

    with pinned_threads():
        model = seeded_adder(ADDER_REFERENCE["Adder"], 128)
        load_checkpoint(model, "adder2")
        kinds = (
            ("3 digits", 3, 0),
            ("3 digits, 1 more _", 3, 1),
            ("3 digits, 2 more _", 3, 2),
            ("4 digits", 4, 0),
            ("5 digits", 5, 0),
        )

        for name, digits, extra in kinds:
            pairs = reference_random_sums(2000, digits, torch.Generator().manual_seed(1), True)
            questions = LENGTH_REFERENCE["padded_questions"](pairs, digits, extra)
            answers = written_answers(model, questions, digits)
            right = sum(answer == truth for answer, (_, truth) in zip(answers, pairs)) / len(pairs)
            results[name] = (right, max(len(answer) for answer in answers))

    return results


def usual_order_tensors(
    pairs: list[tuple[str, str]],
    digits: int,
) -> tuple[Tensor, Tensor]:
    """adder1's to_tensors with each answer left in the usual order, highest digit first."""
    questions, _ = reference_to_tensors(pairs, digits)
    answers = torch.tensor([reference_encode(("^" + answer + "$").ljust(digits + 3, "_")) for _, answer in pairs])

    return questions, answers


def train_adder(
    seed: int,
    steps: int,
    units_first: bool,
    rule: Callable[[int, int], bool],
    measured_at: tuple[int, ...],
) -> tuple[nn.Module, list[float]]:
    """adder2's main on the best device, with the seed, the order of the answers and the held out rule left
    open. Gives the trained model on the CPU and its share of 2,000 held out sums answered exactly right after
    each count of steps in measured_at. On MPS two runs of the same seed differ by a few points, since MPS
    doesn't add numbers up in a fixed order."""
    device = best_device()
    torch.manual_seed(seed)
    model = ADDER_REFERENCE["Adder"]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)
    held_out = reference_random_sums(2000, 3, torch.Generator().manual_seed(1), True, rule)
    held_out_questions, _ = reference_to_tensors(held_out, 3)
    encoded = reference_to_tensors if units_first else usual_order_tensors
    shares: list[float] = []

    for step in range(steps):
        questions, answers = encoded(reference_random_sums(256, 3, generator, False, rule), 3)
        loss = ADDER_REFERENCE["answer_loss"](model, questions.to(device), answers.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step + 1 in measured_at:
            answers_written = written_answers(model.cpu(), held_out_questions, 3)
            model.train().to(device)

            if not units_first:
                answers_written = [answer[::-1] for answer in answers_written]

            shares.append(sum(answer == truth for answer, (_, truth) in zip(answers_written, held_out)) / len(held_out))

    return model.cpu(), shares


def measure_answer_order(long_steps: int = 20000) -> dict[str, list[float]]:
    """adder2's table: the share of 2,000 held out sums answered exactly right after 1,000 to 4,000 steps, with
    answers written units first and in the usual order, then the usual order trained for long_steps."""
    with pinned_threads():
        _, units_first = train_adder(0, 4000, True, is_held_out, (1000, 2000, 3000, 4000))
        _, usual = train_adder(0, 4000, False, is_held_out, (1000, 2000, 3000, 4000))
        _, usual_long = train_adder(0, long_steps, False, is_held_out, (long_steps,))

    return {
        "units first": units_first,
        "usual order": usual,
        f"usual order, {long_steps} steps": usual_long,
    }


def first_held_out_rule(
    a: int,
    b: int,
) -> bool:
    """The rule adder1's lesson says came first. 7a + 13b ends in 0 exactly when a and b end in the same digit."""
    return (a * 7 + b * 13) % 10 == 0


def measure_first_rule() -> dict[str, float]:
    """adder1's last paragraph: adder2's run with sums held out by first_held_out_rule, then the share of 2,000
    held out sums and of 2,000 new sums from the training side that it answers exactly right."""
    with pinned_threads():
        model, (held_out_share,) = train_adder(0, 4000, True, first_held_out_rule, (4000,))
        seen_kind = reference_random_sums(2000, 3, torch.Generator().manual_seed(2), False, first_held_out_rule)
        questions, _ = reference_to_tensors(seen_kind, 3)
        answers = written_answers(model, questions, 3)

    return {
        "held out": held_out_share,
        "new sums from the training side": sum(answer == truth for answer, (_, truth) in zip(answers, seen_kind)) / 2000,
    }
