"""43_seq2seq, an encoder into one vector and a decoder that writes step by step, why one vector struggles with long inputs, and greedy and beam search decoding."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable

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
)
from ..text import prose, source_file
from .bidirectional import expect_nothing_recorded
from .linear_layers import expect_module
from .tensors import pinned_threads

SEQ2SEQ = Chapter(
    folder="43_seq2seq",
    title="Encoders and decoders",
    readme=source_file(
        """
        # Encoders and decoders

        Some tasks turn one sequence into another sequence of a different
        length, like a sentence into its translation or a sum into its answer.
        An encoder RNN reads the whole input and summarizes it in one vector. A
        decoder RNN starts from that vector and writes the output one symbol
        at a time. This chapter builds both and trains them with teacher
        forcing. It shows how one vector stops being enough as the input gets
        longer. Last, it writes outputs with greedy decoding and with beam
        search.

        ## Further reading

        - [Sutskever, Vinyals and Le, sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215)
        - [Cho and colleagues, on the properties of neural machine translation](https://arxiv.org/abs/1409.1259)
        - [Dive into Deep Learning, sequence to sequence learning](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)
        - [Dive into Deep Learning, beam search](https://d2l.ai/chapter_recurrent-modern/beam-search.html)
        - [Wu and colleagues, Google's neural machine translation system](https://arxiv.org/abs/1609.08144)
        - [Koehn and Knowles, six challenges for neural machine translation](https://arxiv.org/abs/1706.03872)
        """,
    ),
)

START = 10
END = 11
VOCABULARY = 12

# What the files after encdec1 are handed, written once so they never drift apart.
PAIRS_PART = '''
START = 10
END = 11


def decoder_pairs(target: Tensor) -> tuple[Tensor, Tensor]:
    column = torch.full((len(target), 1), START)
    end_column = torch.full((len(target), 1), END)

    return torch.cat([column, target], dim=1), torch.cat([target, end_column], dim=1)
'''

ENCODER_DECODER_PART = '''
class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 16)
        self.gru = nn.GRU(16, hidden, batch_first=True)

    def forward(self, symbols: Tensor) -> Tensor:
        _, state = self.gru(self.embedding(symbols))

        return state


class Decoder(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 16)
        self.gru = nn.GRU(16, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocabulary)

    def forward(
        self,
        symbols: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        outputs, state = self.gru(self.embedding(symbols), state)

        return self.out(outputs), state
'''

LOSS_PART = '''
def sequence_loss(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
    target: Tensor,
) -> Tensor:
    inputs, outputs = decoder_pairs(target)
    scores, _ = decoder(inputs, encoder(source))

    return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), outputs.reshape(-1))
'''

TRAIN_PART = '''
def train_copier(
    encoder: nn.Module,
    decoder: nn.Module,
    length: int,
    steps: int,
    generator: torch.Generator,
) -> float:
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.01)

    for _ in range(steps):
        source = torch.randint(0, 10, (64, length), generator=generator)
        loss = sequence_loss(encoder, decoder, source, source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

GREEDY_PART = '''
def greedy_decode(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
    max_length: int,
) -> list[int]:
    state = encoder(source.unsqueeze(0))
    symbol = START
    written = []

    for _ in range(max_length):
        scores, state = decoder(torch.tensor([[symbol]]), state)
        symbol = scores[0, -1].argmax().item()

        if symbol == END:
            break

        written.append(symbol)

    return written
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace() -> dict[str, object]:
    """The parts run once, so the checks compare against exactly the code the learner's files hold."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }

    exec(PAIRS_PART + ENCODER_DECODER_PART + LOSS_PART + TRAIN_PART + GREEDY_PART, namespace)

    return namespace


REFERENCE = reference_namespace()


def seeded_pair(
    hidden: int,
    seed: int,
    encoder_class: type | None = None,
    decoder_class: type | None = None,
) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    encoder = (encoder_class or REFERENCE["Encoder"])(VOCABULARY, hidden)
    decoder = (decoder_class or REFERENCE["Decoder"])(VOCABULARY, hidden)

    return encoder, decoder


def digit_rows(
    seed: int,
    count: int,
    length: int,
) -> Tensor:
    return torch.randint(0, 10, (count, length), generator=torch.Generator().manual_seed(seed))


def reference_predictions(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
) -> Tensor:
    """The decoder's best guess at each digit of a copy, reading the true digits before it."""
    inputs, _ = REFERENCE["decoder_pairs"](source)

    with torch.no_grad():
        scores, _ = decoder(inputs, encoder(source))

    return scores[:, :-1].argmax(dim=2)


def reference_position_accuracy(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
) -> Tensor:
    return (reference_predictions(encoder, decoder, source) == source).float().mean(dim=0)


def reference_exact_match(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
) -> float:
    return (reference_predictions(encoder, decoder, source) == source).all(dim=1).float().mean().item()


def trained_copier(
    length: int,
    hidden: int,
    steps: int,
) -> tuple[nn.Module, nn.Module]:
    """The runs the lessons quote, Adam at 0.01 on batches of 64 random rows of digits."""
    encoder, decoder = seeded_pair(hidden, 0)
    REFERENCE["train_copier"](encoder, decoder, length, steps, torch.Generator().manual_seed(1))

    return encoder, decoder


# ---------------------------------------------------------------------------
# encdec1
# ---------------------------------------------------------------------------

ENCDEC1_LESSON = prose(
    """
    Some tasks turn one sequence into another sequence of a different length.
    A sentence becomes its translation, and in 46_adder a sum like `123+456`
    becomes `579`. In 2014, Sutskever, Vinyals and Le, and separately Cho and
    colleagues, split the model for these tasks into two parts.

    The **encoder** is an RNN that reads the input and keeps only its last
    hidden state. That one vector has to hold everything about the input.
    The **decoder** is a second RNN that starts from that vector and writes
    the output one symbol at a time. A symbol here is one item of a sequence,
    like a digit or a word. At each step the decoder reads the previous
    symbol and gives a score to every symbol that could come next.

    Two extra symbols mark the start and the end of the output. START is the
    first symbol of the decoder's input, because its first step has no
    previous symbol to read. END is the last symbol of the target, so the
    decoder learns when to stop. With the digits 0 to 9 as symbols, START is
    10 and END is 11.

    ```text
    target          4      7  2
    decoder reads   START  4  7  2
    decoder writes  4      7  2  END
    ```

    In training, the decoder reads the true previous symbol at each step,
    never its own guess. This is called **teacher forcing**. Because all the
    inputs are known in advance, every step runs in one call of the GRU. Also,
    an early mistake cannot cause mistakes in the later steps. When the model
    writes on its own, in greedy1, it has to read its own guesses instead.

    The loss is the cross entropy at every step the decoder writes. To compute
    it in one call, reshape the scores so there is one row per step.

    ```python
    scores = torch.zeros(2, 4, 12)    # (batch, steps, symbols)
    scores.reshape(-1, 12).shape      # torch.Size([8, 12])
    ```

    Both the encoder and the decoder use `nn.GRU` with `batch_first=True`. The
    encoder's last state has shape (1, batch, hidden). That is exactly the
    shape the decoder's GRU expects for its starting state.
    """,
)

ENCDEC1_SOURCE = source_file(
    '''
    # encdec1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    START = 10
    END = 11

    starts = torch.full((2, 1), 7)                                   # tensor([[7], [7]])
    joined = torch.cat([starts, torch.tensor([[4], [1]])], dim=1)    # tensor([[7, 4], [7, 1]])


    # TODO: target has shape (batch, length). Return two tensors, the decoder's
    #   input and the decoder's target, both of shape (batch, length + 1). The
    #   input is START followed by target. The decoder's target is target
    #   followed by END.
    #   Example: decoder_pairs(torch.tensor([[4, 7, 2]])) returns
    #   (tensor([[10,  4,  7,  2]]), tensor([[ 4,  7,  2, 11]])).
    def decoder_pairs(target: Tensor) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish forward. Embed symbols, of shape (batch, length), with
    #   self.embedding, and run self.gru on the result from a state of zeros.
    #   Return only the GRU's last state, of shape (1, batch, hidden).
    class Encoder(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, 16)
            self.gru = nn.GRU(16, hidden, batch_first=True)

        def forward(self, symbols: Tensor) -> Tensor:
            ...


    # TODO: Finish forward. Embed symbols, of shape (batch, length), and run
    #   self.gru on the result, starting from state. Return two things. The
    #   first is the score of every possible next symbol at every step, from
    #   self.out, of shape (batch, length, vocabulary). The second is the
    #   GRU's last state.
    class Decoder(nn.Module):
        def __init__(
            self,
            vocabulary: int,
            hidden: int,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, 16)
            self.gru = nn.GRU(16, hidden, batch_first=True)
            self.out = nn.Linear(hidden, vocabulary)

        def forward(
            self,
            symbols: Tensor,
            state: Tensor,
        ) -> tuple[Tensor, Tensor]:
            ...


    # TODO: Return the teacher forced loss for turning source into target. Run
    #   encoder on source to get the starting state. Run decoder on the input
    #   from decoder_pairs(target), starting from that state. The loss is the
    #   cross entropy of the decoder's scores against the target from
    #   decoder_pairs, over every step of every row.
    def sequence_loss(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        ...
    ''',
)

ENCDEC1_SOLUTION = source_file(
    '''
    # encdec1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    starts = torch.full((2, 1), 7)
    joined = torch.cat([starts, torch.tensor([[4], [1]])], dim=1)


''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


''' + indented(LOSS_PART) + '''
    ''',
)


def check_decoder_pairs(found: object) -> None:
    for seed, count, length in ((430, 2, 3), (431, 3, 5)):
        target = digit_rows(seed, count, length)
        call = f"decoder_pairs on {count} rows of {length} digits"
        actual = found(target.clone())

        if not isinstance(actual, tuple) or len(actual) != 2:
            raise Mismatch(f"{call} should give two tensors in a tuple, the decoder's input and then its target.")

        expected_inputs, expected_outputs = REFERENCE["decoder_pairs"](target)
        expect_tensor(
            call=f"the input from {call}",
            actual=actual[0],
            expected=expected_inputs,
        )
        expect_tensor(
            call=f"the target from {call}",
            actual=actual[1],
            expected=expected_outputs,
        )


def check_encoder(found: type) -> None:
    torch.manual_seed(432)
    encoder = expect_module("Encoder(12, 8)", found(VOCABULARY, 8))
    torch.manual_seed(432)
    reference = REFERENCE["Encoder"](VOCABULARY, 8)
    source = digit_rows(433, 3, 5)

    with torch.no_grad():
        expected = reference(source)

    expect_tensor(
        call="Encoder(12, 8) on 3 rows of 5 digits",
        actual=encoder(source),
        expected=expected,
    )


def check_decoder(found: type) -> None:
    torch.manual_seed(434)
    decoder = expect_module("Decoder(12, 8)", found(VOCABULARY, 8))
    torch.manual_seed(434)
    reference = REFERENCE["Decoder"](VOCABULARY, 8)
    symbols = digit_rows(435, 3, 4)
    state = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(436))
    call = "Decoder(12, 8) on 3 rows of 4 symbols"
    actual = decoder(symbols, state.clone())

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give two things in a tuple, the scores and then the last state.")

    with torch.no_grad():
        expected_scores, expected_state = reference(symbols, state)

    expect_tensor(
        call=f"the scores from {call}",
        actual=actual[0],
        expected=expected_scores,
    )
    expect_tensor(
        call=f"the last state from {call}",
        actual=actual[1],
        expected=expected_state,
    )


def check_sequence_loss(found: object) -> None:
    source = digit_rows(437, 4, 5)
    target = digit_rows(438, 4, 3)
    encoder, decoder = seeded_pair(8, 439)

    with torch.no_grad():
        expected = REFERENCE["sequence_loss"](encoder, decoder, source, target).item()
        actual = found(encoder, decoder, source, target)

    expect_close(
        call="sequence_loss on 4 sources of 5 digits and targets of 3",
        actual=actual.item() if isinstance(actual, Tensor) else actual,
        expected=expected,
        tolerance=1e-5,
    )


def check_encdec1(checker: Checker) -> None:
    checker.test(
        function="decoder_pairs",
        description="puts START before the input and END after the target",
        run=check_decoder_pairs,
    )
    checker.test_method(
        class_name="Encoder",
        method="forward",
        description="returns the GRU's last state",
        run=check_encoder,
    )
    checker.test_method(
        class_name="Decoder",
        method="forward",
        description="scores every step and returns the last state",
        run=check_decoder,
    )
    checker.test(
        function="sequence_loss",
        description="takes the teacher forced cross entropy over every step",
        run=check_sequence_loss,
    )


ENCDEC1 = Exercise(
    name="encdec1",
    chapter=SEQ2SEQ,
    title="An encoder and a decoder",
    lesson=ENCDEC1_LESSON,
    source=ENCDEC1_SOURCE,
    solution=ENCDEC1_SOLUTION,
    hints={
        "decoder_pairs": "torch.full((len(target), 1), START) is a column of START. Join it before target with "
        "torch.cat and dim=1. Do the same with a column of END after target.",
        "Encoder.forward": "self.gru(self.embedding(symbols)) returns the outputs and the last state. Return the "
        "last state.",
        "Decoder.forward": "self.gru(self.embedding(symbols), state) returns the outputs and the last state. Return "
        "self.out(outputs) and the state.",
        "sequence_loss": "Split target into inputs and outputs with decoder_pairs. Run decoder on the inputs, "
        "starting from the state encoder returns for source, and keep only the scores. Then take F.cross_entropy "
        "of scores.reshape(-1, scores.shape[-1]) and outputs.reshape(-1).",
    },
    introduces=(),
    check=check_encdec1,
)


# ---------------------------------------------------------------------------
# encdec2
# ---------------------------------------------------------------------------

ENCDEC2_LESSON = prose(
    """
    The decoder sees the input only through the encoder's last state. That
    state has the same number of entries however long the input is. A copying
    task shows the cost of this. The model reads a row of digits and has to
    write the same row back. For each length below, a new model with a hidden
    size of 32 trains for 1,500 steps. Then it is tested on 500 new rows. The
    table gives the fraction of digits copied correctly, and the fraction of
    rows where every digit is correct.

    | Length | Digits right | Rows all right |
    |---|---|---|
    | 5 | 0.988 | 0.942 |
    | 10 | 0.876 | 0.232 |
    | 20 | 0.494 | 0.000 |
    | 40 | 0.233 | 0.000 |

    With 5 digits, it copies almost every row. With 20, it gets fewer than
    half the digits right and no whole row. With 40, it gets the first digit
    right 0.48 of the time and the last digit 0.13 of the time. That is close
    to the 0.1 of a random guess.

    A bigger state helps. With a hidden size of 128, rows of 20 are copied
    with 0.909 of their digits right, compared with 0.494. But only 12.2% of
    whole rows are right. And the vector still has one fixed size, however
    long the input grows.

    Cho and colleagues found the same thing in translation in 2014. Their
    encoder and decoder translated short sentences well, and got worse as
    sentences grew longer. The fix in Section 7 lets the decoder look back at
    every step of the input, instead of only one vector.

    The measurements here feed the decoder the true previous digit before
    each guess, the same way training does. `.all(dim=1)` is `True` for a row
    only when every entry in that row is `True`.

    ```python
    right = torch.tensor([[True, True], [True, False]])
    right.all(dim=1)    # tensor([ True, False])
    ```
    """,
)

ENCDEC2_SOURCE = source_file(
    '''
    # encdec2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    right = torch.tensor([[True, True], [True, False]])
    whole_rows = right.all(dim=1)    # tensor([ True, False])


    # decoder_pairs, Encoder, Decoder and sequence_loss from encdec1.
''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


''' + indented(LOSS_PART) + '''


    # TODO: Finish the loop. At each step, draw a new batch of 64 rows of
    #   random digits from 0 to 9 with generator, each row length digits long.
    #   The task is copying, so the same rows are both the source and the
    #   target. Compute sequence_loss and take one step of optimizer. Call the
    #   loss loss, because the last line returns it.
    def train_copier(
        encoder: nn.Module,
        decoder: nn.Module,
        length: int,
        steps: int,
        generator: torch.Generator,
    ) -> float:
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.01)

        for _ in range(steps):
            ...

        return loss.item()


    # TODO: source has shape (rows, length). Return the decoder's guess for
    #   each digit of the copy, as a tensor of the same shape. Use teacher
    #   forcing, so the decoder reads the true previous digit before each
    #   guess. Each guess is the symbol with the highest score. Leave out the
    #   decoder's last step, the one that should write END. Do not record
    #   anything for autograd.
    def copy_guesses(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the fraction of the rows of source that copy_guesses gets
    #   right at every position, as a Python float.
    def exact_match(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
    ) -> float:
        ...
    ''',
)

ENCDEC2_SOLUTION = source_file(
    '''
    # encdec2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    right = torch.tensor([[True, True], [True, False]])
    whole_rows = right.all(dim=1)


''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


''' + indented(LOSS_PART) + '''


''' + indented(TRAIN_PART) + '''


    def copy_guesses(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
    ) -> Tensor:
        inputs, _ = decoder_pairs(source)

        with torch.no_grad():
            scores, _ = decoder(inputs, encoder(source))

        return scores[:, :-1].argmax(dim=2)


    def exact_match(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
    ) -> float:
        return (copy_guesses(encoder, decoder, source) == source).all(dim=1).float().mean().item()
    ''',
)


def check_train_copier(found: object) -> None:
    runs = []

    for train in (found, REFERENCE["train_copier"]):
        encoder, decoder = seeded_pair(8, 440)
        loss = train(encoder, decoder, 4, 5, torch.Generator().manual_seed(441))
        runs.append((loss, decoder.out.weight.detach().clone()))

    (loss, weight), (expected_loss, expected_weight) = runs

    try:
        expect_close(
            call="train_copier for 5 steps on rows of 4",
            actual=loss,
            expected=expected_loss,
            tolerance=1e-5,
        )
    except Mismatch as mismatch:
        raise Mismatch(
            f"{mismatch} Each step should draw one batch with torch.randint(0, 10, (64, length), "
            "generator=generator), which never draws 10 itself, and use it as both source and target.",
        ) from None

    expect_tensor(
        call="the decoder's output weights after train_copier for 5 steps",
        actual=weight,
        expected=expected_weight,
    )


def check_copier_learns(found: object) -> None:
    encoder, decoder = seeded_pair(32, 0)
    found(encoder, decoder, 4, 300, torch.Generator().manual_seed(1))
    rows_right = reference_exact_match(encoder, decoder, digit_rows(2, 500, 4))

    if rows_right < 0.8:
        raise Mismatch(
            f"After train_copier for 300 steps on rows of 4, the model copies {rows_right:.1%} of new rows exactly, "
            "but it should pass 80%.",
        )


def check_encdec2(checker: Checker) -> None:
    checker.test(
        function="train_copier",
        description="takes one step on a new batch of rows each time",
        run=check_train_copier,
    )
    checker.test(
        function="train_copier",
        description="learns to copy rows of 4 digits",
        run=check_copier_learns,
    )
    encoder, decoder = seeded_pair(8, 442)
    pair = (
        Described(encoder, "an encoder of hidden size 8"),
        Described(decoder, "its decoder"),
    )
    check_on_inputs(
        checker=checker,
        function="copy_guesses",
        description="guesses each digit, reading the true ones before it",
        inputs=(
            (*pair, digit_rows(443, 3, 4)),
            (*pair, digit_rows(444, 5, 6)),
        ),
        reference=reference_predictions,
        expect=expect_tensor,
    )
    checker.test(
        function="copy_guesses",
        description="records nothing for autograd",
        run=lambda found: expect_nothing_recorded(
            function="copy_guesses",
            modules=[encoder, decoder],
            run=lambda: found(encoder, decoder, digit_rows(443, 3, 4)),
        ),
    )
    trained_encoder, trained_decoder = seeded_pair(16, 445)
    REFERENCE["train_copier"](trained_encoder, trained_decoder, 3, 60, torch.Generator().manual_seed(446))
    check_on_inputs(
        checker=checker,
        function="exact_match",
        description="gives the fraction of rows copied right at every position",
        inputs=(
            (Described(trained_encoder, "a lightly trained encoder"), Described(trained_decoder, "its decoder"), digit_rows(447, 40, 3)),
        ),
        reference=reference_exact_match,
        expect=expect_close,
    )


ENCDEC2 = Exercise(
    name="encdec2",
    chapter=SEQ2SEQ,
    title="Why one vector runs out of room",
    lesson=ENCDEC2_LESSON,
    source=ENCDEC2_SOURCE,
    solution=ENCDEC2_SOLUTION,
    hints={
        "train_copier": "Draw source with torch.randint, with digits below 10, a shape of (64, length), and "
        "generator. The loss is sequence_loss with source as both the source and the target. Then zero the "
        "gradients, call backward, and call optimizer.step().",
        "copy_guesses": "Take the input from decoder_pairs(source) and run the decoder on it from encoder(source), "
        "inside torch.no_grad(). Drop the last step of the scores, then take the argmax over dim 2.",
        "exact_match": "Compare copy_guesses with source, and take .all(dim=1). Then convert to floats, take the "
        "mean and call .item().",
    },
    introduces=(),
    check=check_encdec2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# greedy1
# ---------------------------------------------------------------------------

GREEDY1_LESSON = prose(
    """
    In training, the decoder reads the true previous symbol at each step.
    When it writes on its own, there are no true symbols to read, so it reads
    its own guesses. Greedy decoding picks the symbol with the highest score
    at each step and gives it to the decoder as the input of the next step.

    1. The encoder reads the source and returns the starting state.
    2. The decoder reads START and gives a score to every possible first symbol.
    3. The symbol with the highest score becomes the next input, together with the state from that step.
    4. It stops when it writes END, or after a maximum length, so a model that never writes END still stops.

    Here the decoder reads one symbol at a time, as a tensor of shape (1, 1).
    It passes its state from one step to the next.

    ```python
    scores = torch.tensor([[[0.1, 2.0, -1.0]]])    # (1 row, 1 step, 3 symbols)
    scores[0, -1].argmax().item()                  # 1
    ```

    Teacher forcing hides a problem. Once greedy decoding writes a wrong
    digit, every later step reads that wrong digit, and more mistakes follow.
    Here are the copiers from encdec2, measured both ways on the same 500
    rows. A digit counts as right only when it is correct at its position.

    | Length | Digits right, reading true digits | Digits right, greedy | Rows of the wrong length |
    |---|---|---|---|
    | 10 | 0.876 | 0.703 | 4 of 500 |
    | 20 | 0.494 | 0.226 | 125 of 500 |

    The fraction of rows that are entirely right is the same both ways, 0.232
    at length 10. A row is entirely right only when every guess is right. In
    that case greedy decoding reads exactly the true digits anyway.
    """,
)

GREEDY1_SOURCE = source_file(
    '''
    # greedy1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    scores = torch.tensor([[[0.1, 2.0, -1.0]]])
    best = scores[0, -1].argmax().item()    # 1


    # START, END and decoder_pairs, Encoder and Decoder from encdec1.
''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


    # TODO: Return the symbols the decoder writes for source, of shape
    #   (length,), as a list of ints without END. Run the encoder on source as
    #   a batch of one to get the starting state. The decoder reads START
    #   first. At each step, take the symbol with the highest score and feed it
    #   back as the next input. Stop at END or after max_length steps. Do not
    #   record anything for autograd.
    def greedy_decode(
        encoder: nn.Module,
        decoder: nn.Module,
        source: Tensor,
        max_length: int,
    ) -> list[int]:
        ...


    # TODO: sources has shape (rows, length). Run greedy_decode on each row
    #   with a maximum length of length + 5. Return the fraction of all the
    #   digits in sources that greedy_decode writes back at the same position,
    #   as a Python float. A position past the end of what it writes counts as
    #   wrong.
    def greedy_digits(
        encoder: nn.Module,
        decoder: nn.Module,
        sources: Tensor,
    ) -> float:
        ...
    ''',
)

GREEDY1_SOLUTION = source_file(
    '''
    # greedy1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    scores = torch.tensor([[[0.1, 2.0, -1.0]]])
    best = scores[0, -1].argmax().item()


''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


    # With no_grad as a decorator, nothing inside the function is recorded.
    @torch.no_grad()
''' + indented(GREEDY_PART) + '''


    def greedy_digits(
        encoder: nn.Module,
        decoder: nn.Module,
        sources: Tensor,
    ) -> float:
        right = 0

        for row in sources:
            written = greedy_decode(encoder, decoder, row, len(row) + 5)
            right += sum(guess == digit for guess, digit in zip(written, row.tolist()))

        return right / sources.numel()
    ''',
)


def reference_greedy(
    encoder: nn.Module,
    decoder: nn.Module,
    source: Tensor,
    max_length: int,
) -> list[int]:
    with torch.no_grad():
        return REFERENCE["greedy_decode"](encoder, decoder, source, max_length)


def reference_greedy_digits(
    encoder: nn.Module,
    decoder: nn.Module,
    sources: Tensor,
) -> float:
    right = 0

    for row in sources:
        written = reference_greedy(encoder, decoder, row, len(row) + 5)
        right += sum(guess == digit for guess, digit in zip(written, row.tolist()))

    return right / sources.numel()


def ending_decoder(
    hidden: int,
    seed: int,
) -> nn.Module:
    """A decoder whose output bias leans hard on END, so greedy decoding stops early on its own."""
    _, decoder = seeded_pair(hidden, seed)

    with torch.no_grad():
        decoder.out.bias[END] += 4.0

    return decoder


class ScriptedEncoder(nn.Module):
    """Starts the step count at 0, whatever the source."""

    def forward(self, symbols: Tensor) -> Tensor:
        return torch.zeros(1, len(symbols), 1)


class ScriptedDecoder(nn.Module):
    """Writes the symbols of script in turn, whatever it reads, counting the steps in its state. A trained
    decoder rarely writes on after END, so this one does it on purpose, to show whether greedy decoding stops.
    Its last step counts the symbols read before it in this call too, so a greedy_decode that rereads the
    whole prefix from the encoder's state each time gets the same script as one that carries the state."""

    def __init__(self, script: list[int]) -> None:
        super().__init__()
        self.script = script

    def forward(
        self,
        symbols: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step = min(int(state.flatten()[0].item()) + symbols.shape[1] - 1, len(self.script) - 1)
        scores = torch.zeros(len(symbols), symbols.shape[1], VOCABULARY)
        scores[:, -1, self.script[step]] = 1.0

        return scores, state + symbols.shape[1]


def expect_symbols(
    call: str,
    actual: object,
    expected: list[int],
) -> None:
    if isinstance(actual, Tensor):
        raise Mismatch(f"{call} is a tensor. Return a list of ints, which .tolist() or .item() on each symbol gives.")

    if isinstance(actual, list) and any(isinstance(symbol, Tensor) for symbol in actual):
        raise Mismatch(
            f"{call} is a list that holds tensors. Return a list of plain ints, which .item() on each symbol or "
            ".tolist() on a tensor of them gives.",
        )

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_greedy1(checker: Checker) -> None:
    encoder, decoder = seeded_pair(8, 450)
    early_encoder, _ = seeded_pair(8, 451)
    early_decoder = ending_decoder(8, 451)
    copier_encoder, copier_decoder = seeded_pair(16, 452)
    REFERENCE["train_copier"](copier_encoder, copier_decoder, 3, 120, torch.Generator().manual_seed(453))
    check_on_inputs(
        checker=checker,
        function="greedy_decode",
        description="feeds back its highest scoring symbol until END or the maximum length",
        inputs=(
            (Described(encoder, "an untrained encoder"), Described(decoder, "its decoder"), torch.tensor([3, 1, 4]), 6),
            (Described(early_encoder, "an encoder"), Described(early_decoder, "a decoder that likes END"), torch.tensor([2, 7]), 6),
            (Described(copier_encoder, "a lightly trained copier"), Described(copier_decoder, "its decoder"), torch.tensor([5, 0, 9]), 8),
            (Described(copier_encoder, "a lightly trained copier"), Described(copier_decoder, "its decoder"), torch.tensor([1, 1, 6]), 2),
            (Described(ScriptedEncoder(), "an encoder"), Described(ScriptedDecoder([3, 8, END, 7, 2, 5]), "a decoder that writes 3, 8, END, 7, 2, 5"), torch.tensor([4, 4]), 6),
        ),
        reference=reference_greedy,
        expect=expect_symbols,
    )
    checker.test(
        function="greedy_decode",
        description="records nothing for autograd",
        run=lambda found: expect_nothing_recorded(
            function="greedy_decode",
            modules=[encoder, decoder],
            run=lambda: found(encoder, decoder, torch.tensor([3, 1, 4]), 6),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="greedy_digits",
        description="gives the fraction of digits written back at the right position",
        inputs=(
            (Described(copier_encoder, "a lightly trained copier"), Described(copier_decoder, "its decoder"), digit_rows(454, 30, 3)),
            (Described(early_encoder, "an encoder"), Described(early_decoder, "a decoder that likes END"), digit_rows(455, 4, 3)),
        ),
        reference=reference_greedy_digits,
        expect=expect_close,
    )


GREEDY1 = Exercise(
    name="greedy1",
    chapter=SEQ2SEQ,
    title="Writing greedily",
    lesson=GREEDY1_LESSON,
    source=GREEDY1_SOURCE,
    solution=GREEDY1_SOLUTION,
    hints={
        "greedy_decode": "Get the starting state by running encoder on source.unsqueeze(0), which makes a batch of "
        "one. Loop max_length times. Each time, run decoder on torch.tensor([[symbol]]) and the state, and take "
        "scores[0, -1].argmax().item(). Break when that symbol is END.",
        "greedy_digits": "For each row, zip greedy_decode(..., row, len(row) + 5) with row.tolist() and count the "
        "pairs that match. Divide the count by sources.numel().",
    },
    introduces=(),
    check=check_greedy1,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# beam1
# ---------------------------------------------------------------------------

BEAM_SEARCH_GIVEN = '''
def beam_search(
    step: Callable[[int, object], tuple[Tensor, object]],
    state: object,
    width: int,
    max_length: int,
    alpha: float,
) -> list[int]:
    beams = [([START], 0.0, state)]
    finished = []

    for _ in range(max_length):
        candidates = []

        for symbols, total, state in beams:
            log_probs, next_state = step(symbols[-1], state)
            best = log_probs.topk(width)

            for log_prob, symbol in zip(best.values.tolist(), best.indices.tolist()):
                candidates.append((symbols + [symbol], total + log_prob, next_state))

        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = []

        for symbols, total, state in candidates[:width]:
            if symbols[-1] == END:
                finished.append((symbols, total))
            else:
                beams.append((symbols, total, state))

        if not beams:
            break

    finished.extend((symbols, total) for symbols, total, _ in beams)
    best_symbols, _ = max(finished, key=lambda written: sequence_score(written[1], len(written[0]) - 1, alpha))

    return [symbol for symbol in best_symbols[1:] if symbol != END]
'''

BEAM1_LESSON = prose(
    """
    Greedy decoding picks the most likely first symbol, then the most likely
    next one, and so on. But the most likely whole output can start with a
    less likely first symbol. Take a model with two symbols, A and B, plus
    END. The probability of a whole output is the product of the
    probabilities of its steps.

    | First symbol | Its probability | Best second | Its probability | Whole output |
    |---|---|---|---|---|
    | A | 0.6 | END | 0.4 | 0.24 |
    | B | 0.4 | END | 0.9 | 0.36 |

    Greedy decoding picks A, and the output A then END has probability 0.24.
    But B then END is more likely, at 0.36.

    **Beam search** keeps the `width` best partial outputs instead of only
    one. Each of these partial outputs is called a beam. At each step, beam
    search extends every beam by each of its `width` most likely next
    symbols. Then it keeps the `width` best of all those candidates. A beam
    that writes END is finished and set aside. With a width of 1, beam search
    is the same as greedy decoding.

    Beam search ranks beams by their total log probability. That is the sum
    of the log probabilities of all the steps, which equals the log of their
    product. Adding logs is safer than multiplying many small probabilities.
    A product of many small numbers can underflow, which means it gets too
    small for a float and becomes 0.

    Every extra symbol adds a negative log probability, so totals favour short
    outputs. Length normalization fixes this. It divides each finished total
    by the output's length raised to the power `alpha`, with END counted in
    the length. An `alpha` of 0 keeps the raw total, and an `alpha` of 1 ranks
    by the average log probability per symbol. Wu and colleagues used a
    variant of this in Google's translation system.

    ```python
    logs = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    logs.topk(2)    # values tensor([-0.6931, -1.3863]), indices tensor([0, 1])
    ```

    Here is the copier from encdec2 on rows of 10 digits, tested on the same
    500 rows as before.

    | Decoding | Rows all right |
    |---|---|
    | greedy, a width of 1 | 0.232 |
    | beam search, width 3 | 0.280 |
    | beam search, width 5 | 0.280 |

    Beam search fixes some rows where greedy decoding picked a wrong digit
    early on, because two digits had close scores.

    Length normalization matters when outputs of different lengths compete.
    Here a copier with a hidden size of 96 trained for 3,000 steps on rows of
    1 to 12 digits. Then it copied 300 new rows of 6 to 12 digits. Each cell
    gives the results for seeds 0, 1 and 2.

    | Decoding | Ends too early | Rows exactly right | Digits right |
    |---|---|---|---|
    | greedy | 20.3%, 29.0%, 29.7% | 51.3%, 32.0%, 44.0% | 0.814, 0.699, 0.767 |
    | width 10, `alpha` 0 | 19.7%, 31.7%, 29.0% | 52.3%, 33.7%, 47.7% | 0.822, 0.710, 0.788 |
    | width 10, `alpha` 1 | 18.7%, 27.3%, 26.3% | 52.7%, 34.7%, 48.0% | 0.824, 0.719, 0.794 |

    With an `alpha` of 1, the beam ends too early least often and gets the
    most right, on every seed. Its gain over raw totals is small, 1 to 3 more
    correct rows out of 300. It gets 4 to 12 more rows right than greedy
    decoding, and most of that gain comes from the beam itself. In
    translation, Koehn and Knowles found that wider beams gave shorter and
    worse translations. They also found that dividing by length reduced the
    problem.
    """,
)

BEAM1_SOURCE = source_file(
    '''
    # beam1.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    logs = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    best_two = logs.topk(2)    # values tensor([-0.6931, -1.3863]), indices tensor([0, 1])

    # The log of the softmax in one call, more numerically stable than torch.log(torch.softmax(...)).
    chances = F.log_softmax(torch.tensor([0.0, 1.0]), dim=0)    # tensor([-1.3133, -0.3133])


    # START, END and decoder_pairs, Encoder and Decoder from encdec1.
''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


    # TODO: Return the score that ranks finished outputs, which is total
    #   divided by length raised to the power alpha.
    #   Example: sequence_score(-3.0, 4, 1.0) returns -0.75.
    def sequence_score(
        total: float,
        length: int,
        alpha: float,
    ) -> float:
        ...


    # TODO: Finish the loop over beams. step(symbol, state) returns two
    #   things, the log probabilities of every possible next symbol and the
    #   state after that step. Extend each beam by each of its width most
    #   likely next symbols. For each new symbol, append a tuple of three
    #   things to candidates. They are the beam's symbols with the new one
    #   added, total plus the new symbol's log probability, and the new state.
    def beam_search(
        step: Callable[[int, object], tuple[Tensor, object]],
        state: object,
        width: int,
        max_length: int,
        alpha: float,
    ) -> list[int]:
        beams = [([START], 0.0, state)]
        finished = []

        for _ in range(max_length):
            candidates = []

            for symbols, total, state in beams:
                ...

            candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            beams = []

            for symbols, total, state in candidates[:width]:
                if symbols[-1] == END:
                    finished.append((symbols, total))
                else:
                    beams.append((symbols, total, state))

            if not beams:
                break

        finished.extend((symbols, total) for symbols, total, _ in beams)
        best_symbols, _ = max(finished, key=lambda written: sequence_score(written[1], len(written[0]) - 1, alpha))

        return [symbol for symbol in best_symbols[1:] if symbol != END]


    # TODO: Return a step function for beam_search, built from decoder.
    #   step(symbol, state) runs the decoder on the one symbol, as a tensor of
    #   shape (1, 1), starting from state. It returns the log_softmax of the
    #   decoder's scores at that step, of shape (vocabulary,), and the new
    #   state. step must not record anything for autograd.
    def decoder_step(decoder: nn.Module) -> Callable[[int, Tensor], tuple[Tensor, Tensor]]:
        ...
    ''',
)

BEAM1_SOLUTION = source_file(
    '''
    # beam1.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    logs = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    best_two = logs.topk(2)

    chances = F.log_softmax(torch.tensor([0.0, 1.0]), dim=0)


''' + indented(PAIRS_PART) + '''


''' + indented(ENCODER_DECODER_PART) + '''


    def sequence_score(
        total: float,
        length: int,
        alpha: float,
    ) -> float:
        return total / length ** alpha


''' + indented(BEAM_SEARCH_GIVEN) + '''


    def decoder_step(decoder: nn.Module) -> Callable[[int, Tensor], tuple[Tensor, Tensor]]:
        @torch.no_grad()
        def step(
            symbol: int,
            state: Tensor,
        ) -> tuple[Tensor, Tensor]:
            scores, state = decoder(torch.tensor([[symbol]]), state)

            return F.log_softmax(scores[0, -1], dim=0), state

        return step
    ''',
)


def reference_beam_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "Tensor": Tensor,
        "Callable": Callable,
        "START": START,
        "END": END,
        "sequence_score": lambda total, length, alpha: total / length ** alpha,
    }
    exec(BEAM_SEARCH_GIVEN, namespace)

    return namespace


BEAM_REFERENCE = reference_beam_namespace()


def reference_decoder_step(decoder: nn.Module) -> Callable[[int, Tensor], tuple[Tensor, Tensor]]:
    @torch.no_grad()
    def step(
        symbol: int,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scores, state = decoder(torch.tensor([[symbol]]), state)

        return F.log_softmax(scores[0, -1], dim=0), state

    return step


def table_step(
    table: dict[tuple[int, ...], list[float]],
) -> Callable[[int, tuple[int, ...]], tuple[Tensor, tuple[int, ...]]]:
    """A made up model whose next symbol's probabilities depend on everything written so far, kept as the state."""

    def step(
        symbol: int,
        written: tuple[int, ...],
    ) -> tuple[Tensor, tuple[int, ...]]:
        written = (*written, symbol)
        probabilities = torch.tensor(table.get(written[1:], [0.0] * VOCABULARY), dtype=torch.float64)

        return probabilities.clamp_min(1e-12).log(), written

    return step


def table_row(**chances: float) -> list[float]:
    """Probabilities over the 12 symbols, named like s3=0.5 or end=0.4, the rest 0."""
    row = [0.0] * VOCABULARY

    for name, chance in chances.items():
        row[END if name == "end" else int(name[1:])] = chance

    return row


# The lesson's model, A as 1 and B as 2. Greedy writes A, beam search finds B.
LESSON_TABLE = {
    (): table_row(s1=0.6, s2=0.4),
    (1,): table_row(end=0.4, s3=0.35, s4=0.25),
    (2,): table_row(end=0.9, s3=0.1),
    (1, 3): table_row(end=1.0),
    (1, 4): table_row(end=1.0),
    (2, 3): table_row(end=1.0),
}

# A short output with the better total, 5 then END, and a long one with the better average, 6 7 8 then END,
# so alpha changes the winner.
LENGTH_TABLE = {
    (): table_row(s5=0.55, s6=0.45),
    (5,): table_row(end=0.9, s9=0.1),
    (6,): table_row(s7=0.85, end=0.15),
    (5, 9): table_row(end=1.0),
    (6, 7): table_row(s8=0.9, end=0.1),
    (6, 7, 8): table_row(end=0.9, s9=0.1),
    (6, 7, 8, 9): table_row(end=1.0),
}


def check_beam_search(found: object) -> None:
    cases = (
        (LESSON_TABLE, 1, 5, 0.0, "the lesson's model, width 1"),
        (LESSON_TABLE, 2, 5, 0.0, "the lesson's model, width 2"),
        (LENGTH_TABLE, 3, 6, 0.0, "a model where length matters, width 3, alpha 0"),
        (LENGTH_TABLE, 3, 6, 1.0, "a model where length matters, width 3, alpha 1"),
        (LENGTH_TABLE, 2, 2, 0.7, "a model where length matters, width 2, at most 2 steps"),
    )

    for table, width, max_length, alpha, name in cases:
        expected = BEAM_REFERENCE["beam_search"](table_step(table), (), width, max_length, alpha)
        expect_symbols(
            call=f"beam_search on {name}",
            actual=found(table_step(table), (), width, max_length, alpha),
            expected=expected,
        )


def check_decoder_step(found: object) -> None:
    encoder, decoder = seeded_pair(8, 460)
    state = encoder(torch.tensor([[2, 5, 1]])).detach()
    step = found(decoder)

    if not callable(step):
        raise Mismatch(f"decoder_step(decoder) is a {type(step).__name__}. It should return a function, step.")

    actual = step(START, state.clone())

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch("step(START, state) should give two things in a tuple, the log probabilities and the new state.")

    expected_log_probs, expected_state = reference_decoder_step(decoder)(START, state)
    expect_tensor(
        call="the log probabilities from step(START, state)",
        actual=actual[0],
        expected=expected_log_probs,
    )
    expect_tensor(
        call="the new state from step(START, state)",
        actual=actual[1],
        expected=expected_state,
    )
    expect_nothing_recorded(
        function="decoder_step's step",
        modules=[decoder],
        run=lambda: step(START, state.clone()),
    )


def check_beam1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="sequence_score",
        description="divides the total by the length raised to alpha",
        inputs=(
            (-3.0, 4, 0.0),
            (-3.0, 4, 1.0),
            (-2.5, 6, 0.7),
        ),
        reference=lambda total, length, alpha: total / length ** alpha,
        expect=expect_close,
    )
    checker.test(
        function="beam_search",
        description="keeps the width best outputs, ranked by total log probability",
        run=check_beam_search,
    )
    checker.test(
        function="decoder_step",
        description="gives the decoder's log probabilities and its new state",
        run=check_decoder_step,
    )


BEAM1 = Exercise(
    name="beam1",
    chapter=SEQ2SEQ,
    title="Beam search",
    lesson=BEAM1_LESSON,
    source=BEAM1_SOURCE,
    solution=BEAM1_SOLUTION,
    hints={
        "sequence_score": "Python's ** raises a number to a power, so divide total by length ** alpha.",
        "beam_search": "Call step on the last symbol of symbols and on state, and keep the results as log_probs "
        "and next_state. Take the width best with .topk. For each pair from best.values.tolist() and "
        "best.indices.tolist(), append (symbols + [symbol], total + log_prob, next_state).",
        "decoder_step": "Define step(symbol, state) inside decoder_step. It runs decoder(torch.tensor([[symbol]]), "
        "state) and returns F.log_softmax(scores[0, -1], dim=0) and the state. Return step.",
    },
    introduces=(),
    check=check_beam1,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def train_mixed_copier(
    hidden: int,
    steps: int,
    seed: int,
) -> tuple[nn.Module, nn.Module]:
    """A copier trained on rows of 1 to 12 digits, one length per batch of 64, with Adam at 0.01."""
    encoder, decoder = seeded_pair(hidden, seed)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.01)
    generator = torch.Generator().manual_seed(seed + 100)

    for _ in range(steps):
        length = int(torch.randint(1, 13, (1,), generator=generator))
        source = torch.randint(0, 10, (64, length), generator=generator)
        loss = REFERENCE["sequence_loss"](encoder, decoder, source, source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return encoder.eval(), decoder.eval()


def measure_one_vector() -> dict[tuple[int, int], dict[str, float]]:
    """encdec2's table and the numbers after it: for each length and hidden size, a copier trained for 1,500
    steps, measured on 500 new rows reading the true digits. The share of digits right, of rows all right, and
    of the first and the last digit right."""
    results: dict[tuple[int, int], dict[str, float]] = {}

    with pinned_threads():
        for length, hidden in ((5, 32), (10, 32), (20, 32), (40, 32), (20, 128)):
            encoder, decoder = trained_copier(length, hidden, 1500)
            rows = digit_rows(2, 500, length)
            by_position = reference_position_accuracy(encoder, decoder, rows)
            results[(length, hidden)] = {
                "digits": by_position.mean().item(),
                "rows": reference_exact_match(encoder, decoder, rows),
                "first": by_position[0].item(),
                "last": by_position[-1].item(),
            }

    return results


def measure_greedy_gap() -> dict[int, dict[str, float]]:
    """greedy1's table: encdec2's copiers of rows of 10 and 20, on the same 500 rows. The share of digits right
    reading the true digits and greedily, the rows greedy decoding writes at the wrong length, and the share of
    rows all right both ways."""
    results: dict[int, dict[str, float]] = {}

    with pinned_threads():
        for length in (10, 20):
            encoder, decoder = trained_copier(length, 32, 1500)
            rows = digit_rows(2, 500, length)
            written = [reference_greedy(encoder, decoder, row, length + 5) for row in rows]
            results[length] = {
                "digits read true": reference_position_accuracy(encoder, decoder, rows).mean().item(),
                "digits greedy": reference_greedy_digits(encoder, decoder, rows),
                "wrong length": sum(len(symbols) != length for symbols in written),
                "rows read true": reference_exact_match(encoder, decoder, rows),
                "rows greedy": sum(symbols == row.tolist() for symbols, row in zip(written, rows)) / len(rows),
            }

    return results


def measure_beam_widths(widths: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]:
    """beam1's first table: encdec2's copier of rows of 10, the share of the same 500 rows each width of beam
    search copies exactly, with an alpha of 0."""
    with pinned_threads():
        encoder, decoder = trained_copier(10, 32, 1500)
        rows = digit_rows(2, 500, 10)
        step = reference_decoder_step(decoder)
        results: dict[int, float] = {}

        for width in widths:
            right = 0

            for row in rows:
                with torch.no_grad():
                    state = encoder(row.unsqueeze(0))

                right += BEAM_REFERENCE["beam_search"](step, state, width, 15, 0.0) == row.tolist()

            results[width] = right / len(rows)

    return results


def measure_length_penalty(
    seeds: range = range(3),
    widths: tuple[int, ...] = (1, 10),
) -> dict[tuple[int, int, float], tuple[float, float, float]]:
    """beam1's second table: for each seed, width and alpha, the share of 300 rows of 6 to 12 digits that the
    copier of size 96, trained for 3,000 steps, ends too early, the share it copies exactly, and the share of
    digits it writes back in place."""
    generator = torch.Generator().manual_seed(77)
    rows = [
        torch.randint(0, 10, (int(torch.randint(6, 13, (1,), generator=generator)),), generator=generator)
        for _ in range(300)
    ]
    digit_count = sum(len(row) for row in rows)
    results: dict[tuple[int, int, float], tuple[float, float, float]] = {}

    with pinned_threads():
        for seed in seeds:
            encoder, decoder = train_mixed_copier(96, 3000, seed)
            step = reference_decoder_step(decoder)

            with torch.no_grad():
                states = [encoder(row.unsqueeze(0)) for row in rows]

            for width in widths:
                for alpha in (0.0, 1.0):
                    short = right = digits = 0

                    for row, state in zip(rows, states):
                        written = BEAM_REFERENCE["beam_search"](step, state, width, len(row) + 5, alpha)
                        short += len(written) < len(row)
                        right += written == row.tolist()
                        digits += sum(guess == digit for guess, digit in zip(written, row.tolist()))

                    results[(seed, width, alpha)] = (short / len(rows), right / len(rows), digits / digit_count)

    return results
