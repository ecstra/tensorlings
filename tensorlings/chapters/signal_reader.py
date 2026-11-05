"""103_signal_reader, Section 13's project. Your GPT answers questions about signals two ways, reading each signal written
as letters, or reading it through 101_reading_a_new_modality's encoder and projector, and the two are scored side by
side."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    DeviceRecorder,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_on_device,
    expect_tensor,
    expect_value,
    reject_missing_return,
    value_text,
)
from ..paths import CHECKPOINTS_DIR
from ..text import prose, source_file
from ..training import MissingCheckpoint
from .finetuning import GPT_SETTINGS, LOAD_PART, MODEL_PART
from .lora import EVERYWHERE_SOLUTION
from .new_modality import (
    BRIDGE_IMPORTS,
    BRIDGE_PARTS,
    BRIDGE_SOLUTION,
    ENCODER_PART,
    LORA_LAYER_PART,
    PROJECTOR_SOLUTION,
    ROWS_PART,
    SIGNAL_PART,
    STAGES_SOLUTION,
    WRITE_PART,
    indented,
    joined,
    lora_layers,
    namespace_of,
    pinned_threads,
)

SIGNAL_READER = Chapter(
    folder="103_signal_reader",
    title="Project: your GPT reads signals",
    readme=source_file(
        """
        # Project: your GPT reads signals

        Section 13's project. Your GPT from gpt3 learns to answer questions
        about signals two ways. One writes each signal out as letters it can
        read. The other runs it through a frozen encoder and a projector, as
        101_reading_a_new_modality built, with LLaVA's two stages of training.
        Then both are scored on the same questions.

        ## Further reading

        - [Liu and colleagues, visual instruction tuning, LLaVA](https://arxiv.org/abs/2304.08485)
        - [Gruver and colleagues, large language models are zero shot time series forecasters](https://arxiv.org/abs/2310.07820)
        """,
    ),
)

QUESTION_PART = '''
QUESTIONS = ["does it rise?", "how many waves?", "where is the spike?"]


# The right answer to question number asked, from a signal's three facts.
def answer_for(
    asked: int,
    rise: int,
    waves: int,
    third: int,
) -> str:
    if asked == 0:
        return "yes" if rise else "no"

    if asked == 1:
        return WAVE_WORDS[waves - 1]

    return THIRDS[third]


# count new signals, a question drawn for each, and its right answer.
def question_batch(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, list[int], list[str]]:
    signals, rises, waves, thirds = make_signals(count, generator)
    asked = torch.randint(0, len(QUESTIONS), (count,), generator=generator).tolist()
    facts = zip(rises.tolist(), waves.tolist(), thirds.tolist())
    answers = [answer_for(question, *fact) for question, fact in zip(asked, facts)]

    return signals, asked, answers
'''

LETTERS_SOLUTION = '''
def as_letters(signal: Tensor) -> str:
    levels = ((signal - signal.min()) / (signal.max() - signal.min()) * 26).long().clamp(0, 25)

    return "".join("abcdefghijklmnopqrstuvwxyz"[level] for level in levels.tolist())


def text_prompt(
    signal: Tensor,
    asked: int,
) -> str:
    return as_letters(signal) + "\\nQ: " + QUESTIONS[asked] + "\\nA: "
'''

TEXT_STEP_SOLUTION = '''
def text_step(
    gpt: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    scores = gpt(x.to(device))
    loss = F.cross_entropy(scores.flatten(0, 1), y.to(device).flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

# Greedy writing for prompts of one length, as post1's answer wrote, a batch at a time.
TEXT_WRITE_PART = '''
# What gpt writes after each prompt, the likeliest character each step, until a newline or 8 characters. Every
# prompt must be as long as the others.
def write_text(
    gpt: nn.Module,
    prompts: list[str],
    positions: dict[str, int],
    characters: list[str],
) -> list[str]:
    gpt.eval()
    device = gpt.wte.weight.device
    ids = torch.tensor([[positions[character] for character in prompt] for prompt in prompts], device=device)
    written = [""] * len(prompts)
    done = [False] * len(prompts)

    with torch.no_grad():
        for _ in range(8):
            tokens = gpt(ids)[:, -1].argmax(dim=1)

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

QUESTION_ROWS_SOLUTION = '''
def signal_prompt(asked: int) -> str:
    return "Q: " + QUESTIONS[asked] + "\\nA: "


def question_rows(
    asked: list[int],
    answers: list[str],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    return answer_rows([signal_prompt(question) for question in asked], answers, positions)
'''

TRAIN_READER_SOLUTION = '''
def train_reader(
    gpt: nn.Module,
    encoder: nn.Module,
    projector: nn.Module,
    stage_one_steps: int,
    stage_two_steps: int,
    positions: dict[str, int],
    generator: torch.Generator,
    device: torch.device,
) -> None:
    optimizer = stage_one(gpt, projector)

    for _ in range(stage_one_steps):
        signals, x, y = caption_batch(32, positions, generator)
        bridge_step(gpt, encoder, projector, optimizer, signals, x, y, device)

    optimizer = stage_two(gpt, projector)

    for _ in range(stage_two_steps):
        signals, asked, answers = question_batch(32, generator)
        x, y = question_rows(asked, answers, positions)
        bridge_step(gpt, encoder, projector, optimizer, signals, x, y, device)
'''

# The signals reader3 scores both models on, the same every run, each asked the three questions in turn.
HELD_OUT_PART = '''
def held_out(count: int) -> tuple[Tensor, list[int], list[str]]:
    signals, rises, waves, thirds = make_signals(count, torch.Generator().manual_seed(999))
    asked = [place % len(QUESTIONS) for place in range(count)]
    facts = zip(rises.tolist(), waves.tolist(), thirds.tolist())
    answers = [answer_for(question, *fact) for question, fact in zip(asked, facts)]

    return signals, asked, answers
'''

WRITTEN_SOLUTION = '''
def text_written(
    gpt: nn.Module,
    signals: Tensor,
    asked: list[int],
    positions: dict[str, int],
    characters: list[str],
) -> list[str]:
    written = [""] * len(signals)

    for question in range(len(QUESTIONS)):
        rows = [row for row, which in enumerate(asked) if which == question]
        prompts = [text_prompt(signals[row], question) for row in rows]

        for row, said in zip(rows, write_text(gpt, prompts, positions, characters)):
            written[row] = said

    return written


def signal_written(
    reader: nn.Module,
    signals: Tensor,
    asked: list[int],
    positions: dict[str, int],
    characters: list[str],
) -> list[str]:
    written = [""] * len(signals)

    for question in range(len(QUESTIONS)):
        rows = [row for row, which in enumerate(asked) if which == question]
        prompt = signal_prompt(question)
        said = write_after(reader["gpt"], reader["encoder"], reader["projector"], signals[rows], prompt, positions, characters)

        for row, text in zip(rows, said):
            written[row] = text

    return written


def shares_by_question(
    asked: list[int],
    written: list[str],
    answers: list[str],
) -> list[float]:
    shares = []

    for question in range(len(QUESTIONS)):
        marks = [said == answer for said, answer, which in zip(written, answers, asked) if which == question]
        shares.append(sum(marks) / len(marks))

    return shares
'''

# The trained models read back, reader1's text GPT and reader2's signal reader.
LOAD_MODELS_PART = '''
def text_gpt() -> nn.Module:
    return load_checkpoint(add_lora_everywhere(GPT(65, BLOCK_SIZE, SIZE, LAYERS, HEADS), 8, 8.0), "reader1")


def signal_reader() -> nn.Module:
    gpt = add_lora_everywhere(GPT(65, BLOCK_SIZE, SIZE, LAYERS, HEADS), 8, 8.0)
    reader = nn.ModuleDict({"encoder": SignalEncoder(), "projector": Projector(32, 128), "gpt": gpt})

    return load_checkpoint(reader, "reader2")
'''

READER_PARTS = joined(
    BRIDGE_PARTS,
    QUESTION_PART,
    LETTERS_SOLUTION,
    TEXT_STEP_SOLUTION,
    TEXT_WRITE_PART,
    QUESTION_ROWS_SOLUTION,
    TRAIN_READER_SOLUTION,
    HELD_OUT_PART,
    WRITTEN_SOLUTION,
    LOAD_MODELS_PART,
)

READER_REFERENCE = namespace_of(READER_PARTS)
TEXT_STEPS = 1500
STAGE_ONE_STEPS = 300
STAGE_TWO_STEPS = 1200

READER_IMPORTS = BRIDGE_IMPORTS.replace(
    "from tensorlings.training import load_checkpoint",
    "from tensorlings.training import best_device, load_checkpoint, save_checkpoint",
)

# Everything both of the project's models share: the GPT, LoRA, the signals and their questions.
COMMON_GIVEN = '''
''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, your model from gpt3, and LoRA from lora1 and lora3.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


    # 101_reading_a_new_modality's signals, the questions about them, and sft1's answer only rows.
''' + indented(SIGNAL_PART) + '''


''' + indented(QUESTION_PART) + '''


''' + indented(ROWS_PART) + '''
'''

# What the signal reader adds: the frozen encoder, and bridge1's and bridge2's code.
BRIDGE_GIVEN = '''
''' + indented(ENCODER_PART) + '''


''' + indented(PROJECTOR_SOLUTION) + '''


''' + indented(BRIDGE_SOLUTION) + '''


''' + indented(STAGES_SOLUTION) + '''


''' + indented(WRITE_PART) + '''
'''

# What reader3 asks of the trained models, on 300 held out signals. Over 3 seeds the text GPT reached 100%, 70% to 86%
# and 87% to 99% on the three questions, and the signal reader 100%, 100% and 88% to 99%. The bars leave room for a
# run that ends a little lower, since runs on a GPU vary.
HELD_OUT = 300
TEXT_BARS = (0.9, 0.5, 0.5)
SIGNAL_BARS = (0.9, 0.9, 0.7)

SMALL_CHARACTERS = sorted(set("Q: A\nabcdefghijklmnopqrstuvwxyz?"))
SMALL_POSITIONS = Described(
    {character: place for place, character in enumerate(SMALL_CHARACTERS)},
    "the places of Q, A, the colon, the space, the question mark, the newline and a to z",
)


def small_lora_gpt(seed: int) -> nn.Module:
    """A GPT over SMALL_CHARACTERS with rank 2 LoRA on every Linear, its B drawn too, so the update shows."""
    torch.manual_seed(seed)
    gpt = READER_REFERENCE["add_lora_everywhere"](READER_REFERENCE["GPT"](len(SMALL_CHARACTERS), 96, 32, 2, 4), 2, 2.0)
    generator = torch.Generator().manual_seed(seed + 1)

    with torch.no_grad():
        for layer in lora_layers(gpt):
            layer.B.copy_(torch.randn(layer.B.shape, generator=generator) * 0.2)

    return gpt


def reference_model(
    name: str,
    loader: str,
) -> nn.Module:
    """A trained model from the project's own checkpoint, read with the reference's code."""
    try:
        return READER_REFERENCE[loader]()
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            f"The saved model in checkpoints/{name}.pt doesn't fit the model as it is now. Train it again from the "
            f"project folder.\n\n    .venv/bin/python run.py train {name}",
        ) from None


def held_out_shares(
    model: nn.Module,
    written_by: str,
) -> list[float]:
    characters, positions = shakespeare_positions()
    signals, asked, answers = READER_REFERENCE["held_out"](HELD_OUT)
    written = READER_REFERENCE[written_by](model, signals, asked, positions, characters)

    return READER_REFERENCE["shares_by_question"](asked, written, answers)


# Scoring a model on the held out signals takes seconds, so each checkpoint's shares are kept until it is trained again.
SCORED: dict[tuple[str, float], list[float]] = {}


def scored_checkpoint(
    checkpoint: str,
    loader: str,
    written_by: str,
) -> list[float]:
    path = CHECKPOINTS_DIR / f"{checkpoint}.pt"
    key = (checkpoint, path.stat().st_mtime if path.exists() else 0.0)

    if key not in SCORED:
        SCORED[key] = held_out_shares(reference_model(checkpoint, loader), written_by)

    return SCORED[key]


def check_shares_name(
    name: str,
    checkpoint: str,
    loader: str,
    written_by: str,
    bars: tuple[float, ...],
):
    """A name's shares must match the reference's scoring of the same trained model, and reach that model's bars."""

    def check(value: object) -> None:
        expected = scored_checkpoint(checkpoint, loader, written_by)
        expect_shares(
            call=name,
            actual=value,
            expected=expected,
        )

        for question, share, bar in zip(READER_REFERENCE["QUESTIONS"], expected, bars):
            if share < bar:
                raise Mismatch(
                    f"Your model from {checkpoint} answers {share:.0%} of the held out {question!r} questions right, "
                    f"but it should reach {bar:.0%}. Check {checkpoint} and train it again.",
                )

    return check


# ---------------------------------------------------------------------------
# reader1
# ---------------------------------------------------------------------------

READER1_MAIN = '''
    # Run by .venv/bin/python run.py train reader1, never by the app.
    def main() -> None:
        characters = sorted(set(load_shakespeare()))
        positions = {character: place for place, character in enumerate(characters)}
        device = best_device()
        torch.manual_seed(0)
        gpt = add_lora_everywhere(trained_gpt(), 8, 8.0).to(device)
        trained = [parameter for parameter in gpt.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trained, lr=1e-3, weight_decay=0.0)
        generator = torch.Generator().manual_seed(7)
        steps = 1500
        print(f"Training on {device} for {steps} steps, on questions about signals written as letters.")

        for step in range(steps):
            signals, asked, answers = question_batch(32, generator)
            prompts = [text_prompt(signal, question) for signal, question in zip(signals, asked)]
            x, y = answer_rows(prompts, answers, positions)
            loss = text_step(gpt, optimizer, x, y, device)

            if (step + 1) % 500 == 0:
                print(f"Step {step + 1}, loss {loss:.3f}")

        save_checkpoint(gpt, "reader1")
        print("Saved the model in checkpoints/reader1.pt. reader3 reads it back.")
'''

READER1_LESSON = prose(
    """
    This is Section 13's project. Your GPT from gpt3 learns to answer
    questions about the signals of 101_reading_a_new_modality, two ways.
    reader1 writes each signal out as characters. reader2 reads it through
    bridge1's projector, and reader3 scores both.

    Each question asks about one of a signal's three facts.

    | Question | Answers |
    |---|---|
    | `does it rise?` | yes or no |
    | `how many waves?` | one to four |
    | `where is the spike?` | early, middle or late |

    Your GPT's characters hold no digit but 3. So `as_letters` scales each
    signal from its own lowest step to its highest, and cuts that range into
    26 levels, a to z. Its 64 steps become 64 letters.

    ```text
    oopppqssututtsqpnmllkjjkzlmmnooooomlljihgfdeffghijjjijihhfeebaaa
    Q: how many waves?
    A: three
    ```

    The spike is the `z` among the `k`s. Writing a series out as text is how
    Gruver and colleagues fed time series to large language models, as
    digits.

    `text_step` trains as post1's `sft_step` did, on the answer's loss alone.
    main puts rank 8 LoRA on every Linear, as lora3 did, and trains 1,500
    steps of 32 questions.

    Once every check passes, train from the project folder. It took 43 to 97
    seconds on an Apple M5's GPU, and 1.1 to 3.8 minutes on 4 CPU threads.

    ```text
    .venv/bin/python run.py train reader1
    ```
    """,
)

READER1_GIVEN = COMMON_GIVEN + '''

    # What a GPT writes after each prompt, as post1's answer wrote, a batch at a time.
''' + indented(TEXT_WRITE_PART) + '''
'''

READER1_SOURCE = source_file(
    '''
    # reader1.py
    #
    # Needs your trained model from gpt3, in checkpoints/gpt3.pt.
''' + READER_IMPORTS + '''
    levels = (torch.tensor([0.0, 0.5, 1.0]) * 26).long().clamp(0, 25)    # tensor([ 0, 13, 25])
    letter = "abcdefghijklmnopqrstuvwxyz"[13]    # 'n'
''' + READER1_GIVEN + '''

    # TODO: Write signal as 64 letters. Scale it so its lowest step is 0 and its
    #   highest is 1, times 26, as whole numbers clamped to 0 to 25. Each level
    #   picks a letter, 0 for a and 25 for z.
    def as_letters(signal: Tensor) -> str:
        ...


    # TODO: Return the letters of signal, a newline, then "Q: " and the
    #   question QUESTIONS[asked], a newline, then "A: ".
    def text_prompt(
        signal: Tensor,
        asked: int,
    ) -> str:
        ...


    # TODO: Take one training step on x and y, moved to device, as post1's
    #   sft_step did. The loss is the cross entropy of gpt's scores against y,
    #   which skips -100. Zero the old gradients, backpropagate, step, and
    #   return the loss as a plain number.
    def text_step(
        gpt: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        device: torch.device,
    ) -> float:
        ...

''' + READER1_MAIN,
)

READER1_SOLUTION = source_file(
    '''
    # reader1.py, a solution for comparison
''' + READER_IMPORTS + '''
    levels = (torch.tensor([0.0, 0.5, 1.0]) * 26).long().clamp(0, 25)
    letter = "abcdefghijklmnopqrstuvwxyz"[13]
''' + READER1_GIVEN + '''

''' + indented(LETTERS_SOLUTION) + '''


''' + indented(TEXT_STEP_SOLUTION) + '''

''' + READER1_MAIN,
)


def drawn_signals(
    seed: int,
    count: int,
) -> Described:
    signals, _, _, _ = READER_REFERENCE["make_signals"](count, torch.Generator().manual_seed(seed))

    return Described(signals, f"{count} signals made with seed {seed}")


def check_as_letters(found: object) -> None:
    for seed in (931, 932):
        signals = drawn_signals(seed, 3).value

        for row in range(3):
            expect_value(
                call=f"as_letters of signal {row} made with seed {seed}",
                actual=found(signals[row].clone()),
                expected=READER_REFERENCE["as_letters"](signals[row]),
            )

    ramp = torch.linspace(-2.0, 5.0, 64)
    expect_value(
        call="as_letters of a ramp from -2 to 5",
        actual=found(ramp.clone()),
        expected=READER_REFERENCE["as_letters"](ramp),
    )


def check_text_prompt(found: object) -> None:
    for seed, asked in ((933, 0), (934, 1), (935, 2)):
        signal = drawn_signals(seed, 1).value[0]
        expect_value(
            call=f"text_prompt of a signal made with seed {seed}, question {asked}",
            actual=found(signal.clone(), asked),
            expected=READER_REFERENCE["text_prompt"](signal, asked),
        )


def text_batches() -> list[tuple[Tensor, Tensor]]:
    positions = SMALL_POSITIONS.value

    return [
        READER_REFERENCE["answer_rows"](["abc\nQ: does it rise?\nA: ", "zzy\nQ: does it rise?\nA: "], ["yes", "no"], positions),
        READER_REFERENCE["answer_rows"](["qrs\nQ: how many waves?\nA: "], ["two"], positions),
    ]


def check_text_step(found: object) -> None:
    for seed in (936, 937):
        results = []

        for step in (found, READER_REFERENCE["text_step"]):
            gpt = small_lora_gpt(seed)
            optimizer = torch.optim.AdamW([parameter for parameter in gpt.parameters() if parameter.requires_grad], lr=0.01)
            # Two steps, since gradients left from a first step only show in a second.
            losses = [step(gpt, optimizer, x, y, torch.device("cpu")) for x, y in text_batches()]
            results.append((losses, gpt.h[1].mlp.c_fc.A.detach().clone()))

        (losses, weight), (expected_losses, expected_weight) = results

        for place, (loss, expected) in enumerate(zip(losses, expected_losses)):
            expect_close(
                call=f"text_step's loss on a small GPT with LoRA, step {place + 1}",
                actual=loss,
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="one LoRA A after two calls of text_step",
            actual=weight,
            expected=expected_weight,
        )

    x, y = text_batches()[0]
    recorder = DeviceRecorder(
        names=("x",),
        output=lambda ids: torch.zeros(*ids.shape, len(SMALL_CHARACTERS), device="meta"),
    )
    expect_on_device(
        function="text_step",
        run=lambda: found(recorder, torch.optim.SGD(recorder.parameters(), lr=0.1), x, y, torch.device("meta")),
        recorders=[recorder],
        advice="Move x and y to device before the model and the loss.",
    )


def check_reader1(checker: Checker) -> None:
    checker.test(
        function="as_letters",
        description="writes each step as a letter by its level",
        run=check_as_letters,
    )
    checker.test(
        function="text_prompt",
        description="writes the letters, then the question, then A",
        run=check_text_prompt,
    )
    checker.test(
        function="text_step",
        description="trains on the answer's loss, on device",
        run=check_text_step,
    )


READER1 = Exercise(
    name="reader1",
    chapter=SIGNAL_READER,
    title="Project: signals written as letters",
    lesson=READER1_LESSON,
    source=READER1_SOURCE,
    solution=READER1_SOLUTION,
    hints={
        "as_letters": "Subtract signal.min(), divide by signal.max() minus signal.min(), times 26, then .long() "
        "and .clamp(0, 25). Join the letter at each level, from its .tolist().",
        "text_prompt": "as_letters(signal) + a newline + \"Q: \" + QUESTIONS[asked] + a newline + \"A: \".",
        "text_step": "Run gpt on x moved to device for the scores. F.cross_entropy of scores.flatten(0, 1) and "
        "y.to(device).flatten(). Then zero_grad, backward, step, and loss.item().",
    },
    introduces=(),
    check=check_reader1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# reader2
# ---------------------------------------------------------------------------

READER2_MAIN = '''
    # Run by .venv/bin/python run.py train reader2, never by the app.
    def main() -> None:
        characters = sorted(set(load_shakespeare()))
        positions = {character: place for place, character in enumerate(characters)}
        device = best_device()
        print("Pretraining the encoder as an autoencoder.")
        encoder = pretrained_encoder(0).to(device)
        gpt = trained_gpt().to(device)
        torch.manual_seed(0)
        projector = Projector(32, 128).to(device)
        generator = torch.Generator().manual_seed(7)
        print(f"Training on {device}, 300 steps of captions, then 1,200 of questions.")
        train_reader(gpt, encoder, projector, 300, 1200, positions, generator, device)
        reader = nn.ModuleDict({"encoder": encoder, "projector": projector, "gpt": gpt})
        save_checkpoint(reader, "reader2")
        print("Saved the encoder, the projector and the GPT in checkpoints/reader2.pt. reader3 reads them back.")
'''

READER2_LESSON = prose(
    """
    reader2 gives your GPT the same signals through 101_reading_a_new_modality's
    bridge. The frozen encoder makes 8 tokens of each signal, and the
    projector maps them into the GPT's embeddings. LoRA lets the GPT learn to
    read them. What follows the tokens is the question alone.

    ```text
    Q: how many waves?
    A: three
    ```

    `train_reader` runs LLaVA's two stages as bridge2 built them. Stage one
    trains the projector alone on captions, 300 steps. Stage two adds LoRA and
    trains on questions, 1,200 steps. That is the same 1,500 steps reader1's
    text GPT took.

    main saves the encoder with the rest, since the projector learned to read
    that encoder's tokens and no other.

    Once every check passes, train from the project folder. It took 27 to 69
    seconds on an Apple M5's GPU, and 46 to 90 on 4 CPU threads. Its rows are
    far shorter than reader1's, 8 tokens and a question in place of 64
    letters.

    ```text
    .venv/bin/python run.py train reader2
    ```
    """,
)

READER2_GIVEN = COMMON_GIVEN + '''

    # The frozen encoder, and bridge1's and bridge2's code.
''' + BRIDGE_GIVEN

READER2_SOURCE = source_file(
    '''
    # reader2.py
    #
    # Needs your trained model from gpt3, in checkpoints/gpt3.pt.
''' + READER_IMPORTS + '''
    asked_twice = ["Q: " + question for question in ["a?", "b?"]]    # ['Q: a?', 'Q: b?']
''' + READER2_GIVEN + '''

    # TODO: Return the prompt that follows the signal's tokens, "Q: ", then the
    #   question QUESTIONS[asked], a newline, then "A: ".
    def signal_prompt(asked: int) -> str:
        ...


    # TODO: Return answer_rows of each question's signal_prompt and its answer.
    def question_rows(
        asked: list[int],
        answers: list[str],
        positions: dict[str, int],
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Train with LLaVA's two stages. Make stage one's optimizer, then
    #   take stage_one_steps steps of bridge_step on caption_batch's 32 signals
    #   and captions each. Then make stage two's optimizer, and take
    #   stage_two_steps steps on question_batch's 32 signals and questions, as
    #   question_rows. Draw every batch from generator.
    def train_reader(
        gpt: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
        stage_one_steps: int,
        stage_two_steps: int,
        positions: dict[str, int],
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        ...

''' + READER2_MAIN,
)

READER2_SOLUTION = source_file(
    '''
    # reader2.py, a solution for comparison
''' + READER_IMPORTS + '''
    asked_twice = ["Q: " + question for question in ["a?", "b?"]]
''' + READER2_GIVEN + '''

''' + indented(QUESTION_ROWS_SOLUTION) + '''


''' + indented(TRAIN_READER_SOLUTION) + '''

''' + READER2_MAIN,
)


def check_signal_prompt(found: object) -> None:
    for asked in (0, 1, 2):
        expect_value(
            call=f"signal_prompt({asked})",
            actual=found(asked),
            expected=READER_REFERENCE["signal_prompt"](asked),
        )


def expect_id_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise Mismatch(f"{call} should give two things, x and y, but it gives {value_text(actual)}.")

    for name, part, wanted in zip(("x", "y"), actual, expected):
        if isinstance(part, Tensor) and part.dtype != torch.long:
            raise Mismatch(f"{name} from {call} holds {part.dtype} numbers, but it should hold whole numbers, torch.long, as ids do.")

        expect_tensor(
            call=f"{name} from {call}",
            actual=part,
            expected=wanted,
        )


def check_question_rows(found: object) -> None:
    positions = SMALL_POSITIONS.value

    for asked, answers in (([0, 2], ["yes", "late"]), ([1], ["three"])):
        expect_id_pair(
            call=f"question_rows({asked}, {answers}, the small positions)",
            actual=found(asked, answers, positions),
            expected=READER_REFERENCE["question_rows"](asked, answers, positions),
        )


def check_train_reader(found: object) -> None:
    namespace = found.__globals__
    calls: list[tuple[str, object]] = []
    optimizers = {"one": object(), "two": object()}
    handed = torch.Generator()
    strays: list[str] = []

    def scripted_stage_one(
        gpt: nn.Module,
        projector: nn.Module,
    ) -> object:
        calls.append(("stage_one", None))

        return optimizers["one"]

    def scripted_stage_two(
        gpt: nn.Module,
        projector: nn.Module,
    ) -> object:
        calls.append(("stage_two", None))

        return optimizers["two"]

    def scripted_captions(
        count: int,
        positions: dict[str, int],
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        calls.append(("caption_batch", count))

        if generator is not handed:
            strays.append("caption_batch")

        return torch.zeros(count, 64), torch.full((count, 4), 1), torch.full((count, 4), 2)

    def scripted_questions(
        count: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, list[int], list[str]]:
        calls.append(("question_batch", count))

        if generator is not handed:
            strays.append("question_batch")

        return torch.ones(count, 64), [0] * count, ["yes"] * count

    def scripted_rows(
        asked: list[int],
        answers: list[str],
        positions: dict[str, int],
    ) -> tuple[Tensor, Tensor]:
        calls.append(("question_rows", len(asked)))

        return torch.full((len(asked), 4), 3), torch.full((len(asked), 4), 4)

    def scripted_step(
        gpt: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
        optimizer: object,
        signals: Tensor,
        x: Tensor,
        y: Tensor,
        device: torch.device,
    ) -> float:
        stage = "one" if optimizer is optimizers["one"] else "two" if optimizer is optimizers["two"] else "neither"
        data = "captions" if x[0, 0].item() == 1 else "questions" if x[0, 0].item() == 3 else "something else"
        calls.append(("bridge_step", (stage, data, device.type)))

        return 0.5

    stand_ins = {
        "stage_one": scripted_stage_one,
        "stage_two": scripted_stage_two,
        "caption_batch": scripted_captions,
        "question_batch": scripted_questions,
        "question_rows": scripted_rows,
        "bridge_step": scripted_step,
    }
    kept = {name: namespace.get(name) for name in stand_ins}
    namespace.update(stand_ins)

    try:
        found(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), 3, 2, SMALL_POSITIONS.value, handed, torch.device("meta"))
    finally:
        namespace.update(kept)

    steps = [detail for name, detail in calls if name == "bridge_step"]
    order = [name for name, _ in calls if name in ("stage_one", "stage_two")]
    call = "train_reader with 3 steps of stage one and 2 of stage two"

    if order != ["stage_one", "stage_two"]:
        raise Mismatch(f"{call} made the optimizers in the order {order}, but it should call stage_one once, then stage_two once.")

    expected = [("one", "captions", "meta")] * 3 + [("two", "questions", "meta")] * 2

    if steps != expected:
        raise Mismatch(
            f"{call} took the steps {steps}, each as (optimizer, data, device), but they should be {expected}. Stage one "
            "trains on captions, stage two on questions, each with its own optimizer, and every step on device.",
        )

    sizes = {detail for name, detail in calls if name in ("caption_batch", "question_batch")}

    if sizes != {32}:
        raise Mismatch(f"{call} drew batches of {sorted(sizes)} signals, but each should hold 32.")

    if strays:
        raise Mismatch(
            f"{call} called {' and '.join(sorted(set(strays)))} with another generator than the one it was handed. Draw "
            "every batch from generator, so each step gets new signals.",
        )


def check_reader2(checker: Checker) -> None:
    checker.test(
        function="signal_prompt",
        description="writes Q, the question, then A",
        run=check_signal_prompt,
    )
    checker.test(
        function="question_rows",
        description="turns questions and answers into x and y",
        run=check_question_rows,
    )
    checker.test(
        function="train_reader",
        description="trains stage one on captions, then stage two on questions",
        run=check_train_reader,
    )


READER2 = Exercise(
    name="reader2",
    chapter=SIGNAL_READER,
    title="Project: a signal reader, the LLaVA way",
    lesson=READER2_LESSON,
    source=READER2_SOURCE,
    solution=READER2_SOLUTION,
    hints={
        "signal_prompt": "\"Q: \" + QUESTIONS[asked] + a newline + \"A: \".",
        "question_rows": "answer_rows of [signal_prompt(question) for question in asked], answers and positions.",
        "train_reader": "stage_one of gpt and projector gives the first optimizer, then a loop of caption_batch(32, "
        "positions, generator) and bridge_step. stage_two gives the second optimizer, then a loop of "
        "question_batch(32, generator), question_rows, and bridge_step.",
    },
    introduces=(),
    check=check_reader2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# reader3
# ---------------------------------------------------------------------------

READER3_LESSON = prose(
    """
    reader3 scores your two models on the same 300 held out signals, a third
    of them for each question.

    `text_written` and `signal_written` collect what each model writes, one
    question at a time, since a batch of prompts must share one length.
    `shares_by_question` gives the share right for each question.

    Here are both models as reader1's and reader2's mains train them, over 3
    seeds.

    | Question | Text GPT | Signal reader |
    |---|---|---|
    | `does it rise?` | 100% | 100% |
    | `how many waves?` | 70% to 86% | 100% |
    | `where is the spike?` | 87% to 99% | 88% to 99% |

    Both read the trend, and both find the spike, which stands out in either
    form. Its `z` towers over the letters around it, and one of the encoder's
    tokens holds it whole.

    The wave count is where the bridge wins. Here is the text GPT's share of
    wave counts right as it trains on, over the same 3 seeds.

    | Steps | 500 | 1,000 | 1,500 | 3,000 |
    |---|---|---|---|---|
    | Wave counts right | 50% to 59% | 58% to 70% | 70% to 86% | 86% to 96% |

    At reader1's 1,500 steps, 68 of its 71 wrong counts were off by one,
    among two, three and four waves. It got every signal of one wave right.
    So it is still learning to count the cycles in 64 letters, and twice the
    steps leave it short. The signal reader got every wave count right in
    the same 1,500 steps, on every seed.

    It also reads each signal in 8 positions, where the text needs 64. That
    is LLaVA's recipe, and now yours, for any kind of input you want your
    GPT to read.
    """,
)

READER3_GIVEN = COMMON_GIVEN + '''

    # reader1's letters and what a GPT writes after a prompt.
''' + indented(LETTERS_SOLUTION) + '''


''' + indented(TEXT_WRITE_PART) + '''


    # The frozen encoder, bridge1's and bridge2's code, and reader2's prompt and rows.
''' + BRIDGE_GIVEN + '''

''' + indented(QUESTION_ROWS_SOLUTION) + '''


    # The held out signals, and both trained models read back.
''' + indented(HELD_OUT_PART) + '''


''' + indented(LOAD_MODELS_PART) + '''
'''

READER3_NAMES = '''

    # Your two models, each scored on the same 300 held out signals.
    characters = sorted(set(load_shakespeare()))
    positions = {character: place for place, character in enumerate(characters)}
    signals, asked, answers = held_out(300)
    text_shares = shares_by_question(asked, text_written(text_gpt(), signals, asked, positions, characters), answers)
    signal_shares = shares_by_question(asked, signal_written(signal_reader(), signals, asked, positions, characters), answers)
'''

READER3_SOURCE = source_file(
    '''
    # reader3.py
    #
    # Needs your trained models from reader1 and reader2, in checkpoints/reader1.pt and checkpoints/reader2.pt.
''' + READER_IMPORTS + '''
    rows = [row for row, which in enumerate([2, 0, 2]) if which == 2]    # [0, 2]
''' + READER3_GIVEN + '''

    # TODO: Return what gpt writes for each signal's question, in the signals'
    #   order. write_text needs prompts of one length, so take one question at
    #   a time. Find the rows asked it, write their text_prompts, and put each
    #   answer back at its row.
    def text_written(
        gpt: nn.Module,
        signals: Tensor,
        asked: list[int],
        positions: dict[str, int],
        characters: list[str],
    ) -> list[str]:
        ...


    # TODO: The same for reader, a ModuleDict of "gpt", "encoder" and
    #   "projector". For each question, write_after reads the rows' signals,
    #   then that question's signal_prompt.
    def signal_written(
        reader: nn.Module,
        signals: Tensor,
        asked: list[int],
        positions: dict[str, int],
        characters: list[str],
    ) -> list[str]:
        ...


    # TODO: Return the share written right for each question, in the order of
    #   QUESTIONS, as plain numbers.
    def shares_by_question(
        asked: list[int],
        written: list[str],
        answers: list[str],
    ) -> list[float]:
        ...
''' + READER3_NAMES,
)

READER3_SOLUTION = source_file(
    '''
    # reader3.py, a solution for comparison
''' + READER_IMPORTS + '''
    rows = [row for row, which in enumerate([2, 0, 2]) if which == 2]
''' + READER3_GIVEN + '''

''' + indented(WRITTEN_SOLUTION) + READER3_NAMES,
)


def check_text_written(found: object) -> None:
    namespace = found.__globals__
    prompts_seen: list[list[str]] = []

    def scripted_write(
        gpt: nn.Module,
        prompts: list[str],
        positions: dict[str, int],
        characters: list[str],
    ) -> list[str]:
        if len({len(prompt) for prompt in prompts}) > 1:
            raise Mismatch("text_written handed write_text prompts of different lengths. Take one question at a time.")

        prompts_seen.append(prompts)

        return [prompt[:3] + str(len(prompt)) for prompt in prompts]

    for seed, asked in ((941, [2, 0, 1, 0, 2]), (942, [1, 1, 0])):
        signals = drawn_signals(seed, len(asked)).value
        kept = namespace.get("write_text")
        namespace["write_text"] = scripted_write

        try:
            actual = found(nn.Linear(1, 1), signals.clone(), asked, SMALL_POSITIONS.value, SMALL_CHARACTERS)
        finally:
            namespace["write_text"] = kept

        expected = [READER_REFERENCE["text_prompt"](signals[row], question)[:3] + str(len(READER_REFERENCE["text_prompt"](signals[row], question))) for row, question in enumerate(asked)]
        expect_value(
            call=f"text_written for questions {asked}, with a write_text that echoes each prompt's start and length",
            actual=actual,
            expected=expected,
        )


def check_signal_written(found: object) -> None:
    namespace = found.__globals__

    def scripted_write(
        gpt: nn.Module,
        encoder: nn.Module,
        projector: nn.Module,
        signals: Tensor,
        prompt: str,
        positions: dict[str, int],
        characters: list[str],
    ) -> list[str]:
        if gpt is not reader["gpt"] or encoder is not reader["encoder"] or projector is not reader["projector"]:
            raise Mismatch("signal_written handed write_after the wrong parts. Use reader[\"gpt\"], reader[\"encoder\"] and reader[\"projector\"].")

        return [f"{prompt[3:6]}{signal[0].item():.2f}" for signal in signals]

    for seed, asked in ((943, [0, 2, 2, 1]), (944, [1, 0])):
        reader = nn.ModuleDict({"gpt": nn.Linear(1, 1), "encoder": nn.Linear(1, 2), "projector": nn.Linear(2, 1)})
        signals = drawn_signals(seed, len(asked)).value
        kept = namespace.get("write_after")
        namespace["write_after"] = scripted_write

        try:
            actual = found(reader, signals.clone(), asked, SMALL_POSITIONS.value, SMALL_CHARACTERS)
        finally:
            namespace["write_after"] = kept

        expected = [f"{READER_REFERENCE['signal_prompt'](question)[3:6]}{signals[row][0].item():.2f}" for row, question in enumerate(asked)]
        expect_value(
            call=f"signal_written for questions {asked}, with a write_after that echoes each prompt and signal",
            actual=actual,
            expected=expected,
        )


def expect_shares(
    call: str,
    actual: object,
    expected: list[float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} shares, one per question.")

    for place, (share, wanted) in enumerate(zip(actual, expected)):
        expect_close(
            call=f"share {place} of {call}",
            actual=share,
            expected=wanted,
        )


def check_reader3(checker: Checker) -> None:
    checker.test(
        function="text_written",
        description="writes each question's answers, one question at a time",
        run=check_text_written,
    )
    checker.test(
        function="signal_written",
        description="writes each question's answers after the signals",
        run=check_signal_written,
    )
    check_on_inputs(
        checker=checker,
        function="shares_by_question",
        description="gives the share right for each question",
        inputs=(
            ([0, 1, 2, 0], ["yes", "two", "late", "no"], ["yes", "one", "late", "yes"]),
            ([2, 2, 1, 0, 1, 0], ["early", "middle", "four", "no", "four", "no"], ["early", "early", "four", "no", "one", "yes"]),
        ),
        reference=READER_REFERENCE["shares_by_question"],
        expect=expect_shares,
    )

    for name, checkpoint, loader, written_by, bars in (
        ("text_shares", "reader1", "text_gpt", "text_written", TEXT_BARS),
        ("signal_shares", "reader2", "signal_reader", "signal_written", SIGNAL_BARS),
    ):
        checker.check_name(
            name=name,
            description="scores your trained model on the 300 held out signals",
            run=check_shares_name(
                name=name,
                checkpoint=checkpoint,
                loader=loader,
                written_by=written_by,
                bars=bars,
            ),
        )


READER3 = Exercise(
    name="reader3",
    chapter=SIGNAL_READER,
    title="Project: the two readers scored",
    lesson=READER3_LESSON,
    source=READER3_SOURCE,
    solution=READER3_SOLUTION,
    hints={
        "text_written": "Start with written = [\"\"] * len(signals). For each question, rows = the places where asked "
        "holds it. write_text of their text_prompts gives answers in the rows' order, so zip rows with them and fill "
        "written.",
        "signal_written": "As text_written, but call write_after(reader[\"gpt\"], reader[\"encoder\"], "
        "reader[\"projector\"], signals[rows], signal_prompt(question), positions, characters).",
        "shares_by_question": "For each question, collect said == answer over the rows asked it, then divide the sum "
        "by the count.",
        "text_shares": "Train reader1 and reader2 from the project folder first.",
        "signal_shares": "Train reader1 and reader2 from the project folder first.",
    },
    introduces=(),
    check=check_reader3,
    time_limit=120,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def shakespeare_positions() -> tuple[list[str], dict[str, int]]:
    characters = sorted(set(load_shakespeare()))

    return characters, {character: place for place, character in enumerate(characters)}


def text_gpt_start(
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.Generator]:
    """reader1's main up to its loop, for a seed: rank 8 LoRA on gpt3's GPT, its AdamW and the batches' generator."""
    torch.manual_seed(seed)
    gpt = READER_REFERENCE["add_lora_everywhere"](READER_REFERENCE["trained_gpt"](), 8, 8.0).to(device)
    trained = [parameter for parameter in gpt.parameters() if parameter.requires_grad]

    return gpt, torch.optim.AdamW(trained, lr=1e-3, weight_decay=0.0), torch.Generator().manual_seed(seed + 7)


def train_text_steps(
    gpt: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    steps: int,
    device: torch.device,
) -> None:
    """reader1's loop, steps steps on questions about signals written as letters."""
    _, positions = shakespeare_positions()

    for _ in range(steps):
        signals, asked, answers = READER_REFERENCE["question_batch"](32, generator)
        prompts = [READER_REFERENCE["text_prompt"](signal, question) for signal, question in zip(signals, asked)]
        x, y = READER_REFERENCE["answer_rows"](prompts, answers, positions)
        READER_REFERENCE["text_step"](gpt, optimizer, x, y, device)


def train_text_gpt(
    seed: int,
    steps: int,
    device: torch.device,
) -> nn.Module:
    """reader1's main, for a seed: rank 8 LoRA on gpt3's GPT, trained steps steps on questions about letters."""
    gpt, optimizer, generator = text_gpt_start(seed, device)
    train_text_steps(gpt, optimizer, generator, steps, device)

    return gpt


def train_signal_reader(
    seed: int,
    stage_one_steps: int,
    stage_two_steps: int,
    device: torch.device,
) -> nn.ModuleDict:
    """reader2's main, for a seed: the encoder pretrained, then LLaVA's two stages."""
    _, positions = shakespeare_positions()
    encoder = READER_REFERENCE["pretrained_encoder"](seed).to(device)
    gpt = READER_REFERENCE["trained_gpt"]().to(device)
    torch.manual_seed(seed)
    projector = READER_REFERENCE["Projector"](32, 128).to(device)
    generator = torch.Generator().manual_seed(seed + 7)
    READER_REFERENCE["train_reader"](gpt, encoder, projector, stage_one_steps, stage_two_steps, positions, generator, device)

    return nn.ModuleDict({"encoder": encoder, "projector": projector, "gpt": gpt})


def measure_wave_counts(
    seeds: range = range(3),
    marks: tuple[int, ...] = (500, 1000, 1500, 3000),
    device: str = "mps",
) -> dict[str, list[object]]:
    """reader3's reason: the text GPT's share right on each question after each mark of steps, trained on as reader1's
    main trains, and at reader1's 1,500 steps the wave counts it said against the right ones, scored on the 300 held
    out signals."""
    place = torch.device(device)
    characters, positions = shakespeare_positions()
    signals, asked, answers = READER_REFERENCE["held_out"](300)
    results: dict[str, list[object]] = {"shares": [], "counts said": []}

    with pinned_threads():
        for seed in seeds:
            gpt, optimizer, generator = text_gpt_start(seed, place)
            shares: dict[int, list[float]] = {}
            done = 0

            for mark in marks:
                train_text_steps(gpt, optimizer, generator, mark - done, place)
                done = mark
                written = READER_REFERENCE["text_written"](gpt, signals, asked, positions, characters)
                shares[mark] = READER_REFERENCE["shares_by_question"](asked, written, answers)

                if mark == TEXT_STEPS:
                    said = [(answer, text) for answer, text, question in zip(answers, written, asked) if question == 1]
                    results["counts said"].append(said)

            results["shares"].append(shares)

    return results


def measure_readers(
    seeds: range = range(3),
    device: str = "mps",
) -> dict[str, list[list[float]]]:
    """reader3's table: the share right on each question, for the text GPT and the signal reader trained as reader1
    and reader2's mains train them, scored on 300 held out signals, for each seed."""
    place = torch.device(device)
    characters, positions = shakespeare_positions()
    signals, asked, answers = READER_REFERENCE["held_out"](300)
    results: dict[str, list[list[float]]] = {"text": [], "signal": []}

    with pinned_threads():
        for seed in seeds:
            gpt = train_text_gpt(seed, TEXT_STEPS, place).cpu()
            written = READER_REFERENCE["text_written"](gpt, signals, asked, positions, characters)
            results["text"].append(READER_REFERENCE["shares_by_question"](asked, written, answers))
            reader = train_signal_reader(seed, STAGE_ONE_STEPS, STAGE_TWO_STEPS, place).cpu()
            written = READER_REFERENCE["signal_written"](reader, signals, asked, positions, characters)
            results["signal"].append(READER_REFERENCE["shares_by_question"](asked, written, answers))

    return results
