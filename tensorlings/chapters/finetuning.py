"""76_finetuning, Section 11, your Section 7 GPT trained further on a novel, what it forgets, and which layers to freeze."""

from __future__ import annotations

import math
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import MissingData, load_persuasion, load_shakespeare
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
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, best_device, load_checkpoint
from .gpt import (
    DATA_PART,
    GIVEN_MODEL,
    GPT_FORWARD,
    GPT_INIT,
    GPT_SETTINGS,
    INIT_SOLUTION,
    TRAIN_PARTS,
    seeded_gpt,
)

FINETUNING = Chapter(
    folder="76_finetuning",
    title="Fine tuning",
    readme=source_file(
        """
        # Fine tuning

        Your GPT from Section 7 learned Shakespeare. Here it keeps training on
        a novel by Jane Austen, starting from the weights it has. You measure
        what it learns and what it forgets, and which layers are worth
        training at all. Download the books once from the project folder.

        ```bash
        .venv/bin/python run.py data gutenberg
        ```

        ## Further reading

        - [McCloskey and Cohen's catastrophic interference, on Wikipedia](https://en.wikipedia.org/wiki/Catastrophic_interference)
        - [Kirkpatrick and colleagues, overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
        - [Yosinski and colleagues, how transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
        - [Howard and Ruder, universal language model fine tuning](https://arxiv.org/abs/1801.06146)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# gpt3's validation_loss, the last of its training parts.
VALIDATION_PART = TRAIN_PARTS[TRAIN_PARTS.index("def validation_loss("):]

# gpt2's model whole, its forward included.
MODEL_PART = GIVEN_MODEL + "\n\n" + INIT_SOLUTION + "\n\n" + GPT_INIT + GPT_FORWARD

# Your model from gpt3, read back from checkpoints/gpt3.pt.
LOAD_PART = '''
def trained_gpt() -> nn.Module:
    model = GPT(65, BLOCK_SIZE, SIZE, LAYERS, HEADS)

    return load_checkpoint(model, "gpt3")
'''

ENCODE_SOLUTION = '''
def encode_known(
    text: str,
    characters: list[str],
) -> Tensor:
    positions = {character: position for position, character in enumerate(characters)}

    return torch.tensor([positions[character] for character in text if character in positions])
'''

FINE_TUNE_SOLUTION = '''
def fine_tune(
    model: nn.Module,
    data: Tensor,
    steps: int,
    lr: float,
    generator: torch.Generator,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []

    for _ in range(steps):
        x, y = get_batch(data, model.block_size, 32, generator)
        scores = model(x.to(device))
        loss = F.cross_entropy(scores.flatten(0, 1), y.to(device).flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
'''

LOSSES_SOLUTION = '''
def losses_on(
    model: nn.Module,
    texts: dict[str, Tensor],
    device: torch.device,
) -> dict[str, float]:
    return {name: validation_loss(model, data, 20, torch.Generator().manual_seed(1), device) for name, data in texts.items()}
'''

# Persuasion as ids of the model's own characters, cut 90 to 10 as gpt1 cut Shakespeare.
PERSUASION_PART = '''
def persuasion_splits(characters: list[str]) -> tuple[Tensor, Tensor]:
    data = encode_known(load_persuasion(), characters)
    cut = int(0.9 * len(data))

    return data[:cut], data[cut:]
'''

# Everything finetune1 hands over, and what finetune2 and 77_lora are handed after it.
FINE_TUNE_PARTS = ENCODE_SOLUTION + "\n\n" + FINE_TUNE_SOLUTION + "\n\n" + LOSSES_SOLUTION + "\n\n" + PERSUASION_PART


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
        "load_persuasion": load_persuasion,
        "load_checkpoint": load_checkpoint,
    }
    exec(code, namespace)

    return namespace


FINE_TUNE_REFERENCE = namespace_of(GPT_SETTINGS + MODEL_PART + DATA_PART + VALIDATION_PART + LOAD_PART + FINE_TUNE_PARTS)


def reference_trained_gpt() -> nn.Module:
    """The learner's model from gpt3, in the reference GPT, so a check fails on the checkpoint and not on given code."""
    try:
        return FINE_TUNE_REFERENCE["trained_gpt"]()
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model doesn't fit gpt2's GPT. Train it again from gpt3's project folder.\n\n"
            "    .venv/bin/python run.py train gpt3",
        ) from None


def reference_texts() -> dict[str, Tensor]:
    """Both validation parts, Persuasion's and Shakespeare's."""
    characters, _, shakespeare_validation = FINE_TUNE_REFERENCE["shakespeare_splits"]()

    try:
        _, persuasion_validation = FINE_TUNE_REFERENCE["persuasion_splits"](characters)
    except MissingData as missing:
        raise Mismatch(str(missing)) from None

    return {"persuasion": persuasion_validation, "shakespeare": shakespeare_validation}


def persuasion_training() -> Tensor:
    characters = sorted(set(load_shakespeare()))

    try:
        persuasion_train, _ = FINE_TUNE_REFERENCE["persuasion_splits"](characters)
    except MissingData as missing:
        raise Mismatch(str(missing)) from None

    return persuasion_train


def measure_rates() -> None:
    """The numbers in finetune1's table. Each rate fine tunes the trained gpt3 model for 1,000 steps on the best
    device, and the from scratch row trains a fresh GPT seeded with 0 the same way. Takes about 4 minutes on MPS."""
    device = best_device()
    texts = reference_texts()
    train_data = persuasion_training()
    runs = [(f"rate {lr}", FINE_TUNE_REFERENCE["trained_gpt"], lr) for lr in (1e-4, 3e-4, 1e-3)]
    runs.append(("from scratch at 0.001", lambda: seeded_gpt(FINE_TUNE_REFERENCE["GPT"], (65, 128, 128, 4, 4)), 1e-3))

    for label, make, lr in runs:
        model = make().to(device)
        generator = torch.Generator().manual_seed(0)
        done = 0
        print(label, 0, FINE_TUNE_REFERENCE["losses_on"](model, texts, device))

        for mark in (100, 300, 1000):
            FINE_TUNE_REFERENCE["fine_tune"](model, train_data, mark - done, lr, generator, device)
            done = mark
            print(label, mark, FINE_TUNE_REFERENCE["losses_on"](model, texts, device))


# ---------------------------------------------------------------------------
# finetune1
# ---------------------------------------------------------------------------

FINETUNE1_LESSON = prose(
    """
    Your gpt3 model learned Shakespeare from random weights. Fine tuning
    trains a trained model further, on new text, starting from the weights it
    has. Language models are built this way. A base model trains once on a
    large pile of text, then fine tuning shapes it for a task.

    The new text is Jane Austen's novel Persuasion, from NLTK's copy of
    Project Gutenberg. Download it once with `.venv/bin/python run.py data
    gutenberg`.

    The model reads only its own 65 characters. Persuasion uses 15 others,
    1,833 times in all, mostly double quotes, 0.4% of the book. `encode_known`
    drops them, so the vocabulary and every trained weight stay as they are.

    ```python
    known = {"a", "b"}
    [letter for letter in "a-b!" if letter in known]    # ['a', 'b']
    ```

    Here is the run behind this chapter on MPS. AdamW at a fixed rate, with
    batches of 32, and each loss on a text's validation part.

    | Rate | Steps | Persuasion | Shakespeare |
    |---|---|---|---|
    | | 0 | 1.744 | 1.535 |
    | 0.0001 | 100 | 1.353 | 1.834 |
    | 0.0001 | 1,000 | 1.260 | 2.020 |
    | 0.001 | 100 | 1.409 | 2.268 |
    | 0.001 | 1,000 | 1.287 | 2.498 |
    | from scratch, 0.001 | 1,000 | 1.510 | 2.671 |

    A hundred steps buy most of the gain. A fresh GPT trained on Persuasion
    alone is still far behind after 1,000, since the fine tune starts out
    knowing English spelling and words.

    Shakespeare's loss climbs while Persuasion's falls. McCloskey and Cohen
    named this catastrophic interference in 1989, and it is often called
    catastrophic forgetting. Nothing in the loss asks the model to keep the
    old text, so the weights that held it move too.

    gpt3's peak rate of 0.001 learns Persuasion worse and forgets more. A
    trained model already sits near a good answer, and a small rate keeps it
    near. A rate of 0.0003 learned Persuasion as well as 0.0001 but forgot
    more, 2.207 on Shakespeare after 1,000 steps.

    `losses_on` scores every text on the same batches each time it runs. So a
    difference before and after comes from the model, not from luck in the
    batches drawn.
    """,
)

FINETUNE1_SOURCE = source_file(
    '''
    # finetune1.py
    #
    # Needs your trained model from gpt3, in checkpoints/gpt3.pt, and the books,
    # downloaded once from the project folder with
    #
    #     .venv/bin/python run.py data gutenberg

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_persuasion, load_shakespeare
    from tensorlings.training import load_checkpoint

    known = {"a", "b"}
    kept = [letter for letter in "a-b!" if letter in known]    # ['a', 'b']
    lengths = {name: len(name) for name in ("old", "new")}    # {'old': 3, 'new': 3}


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2.
''' + indented(MODEL_PART) + '''


    # get_batch and the Shakespeare splits from gpt1, and validation_loss from gpt3.
''' + indented(DATA_PART) + '''


''' + indented(VALIDATION_PART) + '''


''' + indented(LOAD_PART) + '''


    # TODO: Return the ids of the characters of text that characters holds, each
    #   its position in characters, as a tensor. Drop every other character.
    def encode_known(
        text: str,
        characters: list[str],
    ) -> Tensor:
        ...


    # TODO: Train model for steps steps with AdamW at the fixed rate lr, betas of
    #   (0.9, 0.95) and no weight decay. Each step draws a batch of 32 from data
    #   with get_batch, model.block_size long, moved to device. Take the cross
    #   entropy as gpt3 did, zero the old gradients, backpropagate and step,
    #   with no clipping. Return every step's loss as a plain number.
    def fine_tune(
        model: nn.Module,
        data: Tensor,
        steps: int,
        lr: float,
        generator: torch.Generator,
        device: torch.device,
    ) -> list[float]:
        ...


    # TODO: Return a dict from each name of texts to model's validation_loss on
    #   its data, over 20 batches, with a new generator seeded with 1 for each.
    def losses_on(
        model: nn.Module,
        texts: dict[str, Tensor],
        device: torch.device,
    ) -> dict[str, float]:
        ...


    # Persuasion cut 90 to 10, as gpt1 cut Shakespeare, through your encode_known.
''' + indented(PERSUASION_PART) + '''
    ''',
)

FINETUNE1_SOLUTION = source_file(
    '''
    # finetune1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_persuasion, load_shakespeare
    from tensorlings.training import load_checkpoint

    known = {"a", "b"}
    kept = [letter for letter in "a-b!" if letter in known]
    lengths = {name: len(name) for name in ("old", "new")}


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(DATA_PART) + '''


''' + indented(VALIDATION_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(ENCODE_SOLUTION) + '''


''' + indented(FINE_TUNE_SOLUTION) + '''


''' + indented(LOSSES_SOLUTION) + '''


''' + indented(PERSUASION_PART) + '''
    ''',
)

SAMPLE_CHARACTERS = sorted(set("To be, or not to be: that is the question."))


def small_gpt() -> nn.Module:
    return seeded_gpt(FINE_TUNE_REFERENCE["GPT"])


def small_data(seed: int) -> Tensor:
    return torch.randint(0, 20, (300,), generator=torch.Generator().manual_seed(seed))


def expect_ids(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype.is_floating_point:
        raise Mismatch(
            f"{call} holds {actual.dtype} numbers, but ids are whole numbers, which nn.Embedding and F.cross_entropy "
            "need. Make it with torch.tensor of whole numbers, not torch.Tensor.",
        )


def loud_gpt() -> nn.Module:
    """The small GPT with its table 5 times larger, so its first gradients pass a norm of 1 and clipping would show."""
    model = small_gpt()

    with torch.no_grad():
        model.wte.weight.mul_(5.0)

    return model


def train_small(
    train: object,
    model: nn.Module,
) -> list[float]:
    try:
        return train(model, small_data(760), 2, 0.01, torch.Generator().manual_seed(761), torch.device("cpu"))
    except IndexError:
        raise Mismatch(
            f"fine_tune crashed with an index out of range. The check's small GPT reads {model.block_size} tokens, "
            "so draw each batch model.block_size long, not BLOCK_SIZE.",
        ) from None


def check_fine_tune(found: object) -> None:
    """Two steps, so gradients never zeroed show, from the same small GPT and the same batches as the reference.
    The loud GPT's gradients pass a norm of 1, so a clipped step shows too."""
    for make, name in ((small_gpt, "a small GPT"), (loud_gpt, "a small GPT with a large table")):
        results = []

        for train in (found, FINE_TUNE_REFERENCE["fine_tune"]):
            model = make()
            losses = train_small(train, model)
            results.append((losses, model.wte.weight.detach().clone(), model.h[0].attn.c_attn.bias.detach().clone()))

        (losses, weight, bias), (expected_losses, expected_weight, expected_bias) = results

        if not isinstance(losses, list) or len(losses) != 2:
            raise Mismatch(f"fine_tune for 2 steps should return a list of 2 losses, but it returned {losses!r}.")

        for number, (loss, expected_loss) in enumerate(zip(losses, expected_losses), start=1):
            expect_close(
                call=f"fine_tune's loss on {name}, step {number}, batches of 32 from 300 tokens",
                actual=loss,
                expected=expected_loss,
                tolerance=1e-5,
            )

        expect_tensor(
            call=f"wte's weights of {name} after two steps of fine_tune at 0.01",
            actual=weight,
            expected=expected_weight,
        )
        expect_tensor(
            call=f"the first block's c_attn bias of {name} after two steps of fine_tune",
            actual=bias,
            expected=expected_bias,
        )


def check_fine_tune_device(found: object) -> None:
    recorder = DeviceRecorder(
        names=("the batch x",),
        output=lambda ids: ids.unsqueeze(-1).expand(-1, -1, 20).float(),
    )
    recorder.block_size = 16
    expect_on_device(
        function="fine_tune",
        run=lambda: found(recorder, small_data(762), 2, 0.01, torch.Generator().manual_seed(763), torch.device("meta")),
        recorders=[recorder],
        advice="Draw each batch on the CPU with get_batch, then move its x and its y to device.",
    )


def expect_losses(
    call: str,
    actual: object,
    expected: dict[str, float],
) -> None:
    if not isinstance(actual, dict) or sorted(actual) != sorted(expected):
        raise Mismatch(f"{call} should be a dict with the names {sorted(expected)}, but it is {actual!r}.")

    for name, loss in expected.items():
        expect_close(
            call=f"{call}[{name!r}]",
            actual=actual[name],
            expected=loss,
            tolerance=1e-5,
        )


def check_losses_on(found: object) -> None:
    for seeds in ((764, 765), (766, 767, 768)):
        model = small_gpt()
        texts = {f"text {index}": small_data(seed) for index, seed in enumerate(seeds)}
        expect_losses(
            call=f"losses_on(a small GPT, {len(texts)} texts of 300 tokens)",
            actual=found(model, texts, torch.device("cpu")),
            expected=FINE_TUNE_REFERENCE["losses_on"](model, texts, torch.device("cpu")),
        )


def persuasion_loss(model: nn.Module) -> float:
    """Five batches only, since the check runs on the CPU, the same five before and after."""
    return FINE_TUNE_REFERENCE["validation_loss"](model, reference_texts()["persuasion"], 5, torch.Generator().manual_seed(1), torch.device("cpu"))


def check_fine_tuned_gpt(found: object) -> None:
    """40 steps on the CPU, of the learner's fine_tune on their own gpt3 model."""
    model = reference_trained_gpt()
    train_data = persuasion_training()
    before = persuasion_loss(model)
    found(model, train_data, 40, 1e-4, torch.Generator().manual_seed(0), torch.device("cpu"))
    after = persuasion_loss(model)

    if after > before - 0.15:
        raise Mismatch(
            f"40 steps of fine_tune at 0.0001 took your gpt3 model's Persuasion loss from {before:.3f} to {after:.3f}, "
            "but it should fall by more than 0.15.",
        )


def check_finetune1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="encode_known",
        description="gives each known character's position and drops the rest",
        inputs=(
            ("Not \"to\" be!", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
            ("(question)\nto be", Described(SAMPLE_CHARACTERS, "the characters of a line of Hamlet")),
        ),
        reference=FINE_TUNE_REFERENCE["encode_known"],
        expect=expect_ids,
    )
    checker.test(
        function="fine_tune",
        description="takes AdamW steps at a fixed rate and gives every loss",
        run=check_fine_tune,
    )
    checker.test(
        function="fine_tune",
        description="moves each batch to the device",
        run=check_fine_tune_device,
    )
    checker.test(
        function="losses_on",
        description="scores every text on the same batches",
        run=check_losses_on,
    )
    checker.test(
        function="fine_tune",
        description="on your gpt3 model, 40 steps lower its Persuasion loss",
        run=check_fine_tuned_gpt,
    )


FINETUNE1 = Exercise(
    name="finetune1",
    chapter=FINETUNING,
    title="Fine tuning your GPT on a novel",
    lesson=FINETUNE1_LESSON,
    source=FINETUNE1_SOURCE,
    solution=FINETUNE1_SOLUTION,
    hints={
        "encode_known": "Make a dict from character to position with enumerate(characters). Look up only the "
        "characters of text that are in it, and pass the list to torch.tensor.",
        "fine_tune": "torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0). Each step, "
        "get_batch, the loss on scores.flatten(0, 1) and y.flatten(), then zero_grad, backward, step and loss.item().",
        "losses_on": "A dict comprehension over texts.items(), calling validation_loss(model, data, 20, "
        "torch.Generator().manual_seed(1), device) for each.",
    },
    introduces=(),
    check=check_finetune1,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# finetune2
# ---------------------------------------------------------------------------

FREEZE_SOLUTION = '''
def freeze_except(
    model: nn.Module,
    prefixes: tuple[str, ...],
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))


def trainable_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
'''

FINE_TUNE_REFERENCE.update(namespace_of(GPT_SETTINGS + MODEL_PART + DATA_PART + VALIDATION_PART + FINE_TUNE_PARTS + FREEZE_SOLUTION))

FREEZING_CHOICES: dict[str, tuple[str, ...]] = {
    "everything": ("",),
    "the last block": ("h.3.",),
    "the first block": ("h.0.",),
    "wte and wpe": ("wte", "wpe"),
    "the LayerNorms": ("h.0.ln", "h.1.ln", "h.2.ln", "h.3.ln", "ln_f"),
}


def measure_freezing() -> None:
    """The numbers in finetune2's table, 300 steps from the trained gpt3 model for each choice, at 0.0001 and at
    0.001, on the best device. Takes about 3 minutes on MPS."""
    device = best_device()
    texts = reference_texts()
    train_data = persuasion_training()

    for lr in (1e-4, 1e-3):
        for label, prefixes in FREEZING_CHOICES.items():
            model = FINE_TUNE_REFERENCE["trained_gpt"]()
            FINE_TUNE_REFERENCE["freeze_except"](model, prefixes)
            model.to(device)
            FINE_TUNE_REFERENCE["fine_tune"](model, train_data, 300, lr, torch.Generator().manual_seed(0), device)
            print(lr, label, FINE_TUNE_REFERENCE["trainable_count"](model), FINE_TUNE_REFERENCE["losses_on"](model, texts, device))


FINETUNE2_LESSON = prose(
    """
    Freezing a parameter means `requires_grad_(False)`. Autograd then gives
    it no gradient, and the optimizer leaves it alone.

    ```python
    layer = nn.Linear(2, 2)
    layer.weight.requires_grad_(False)    # the weight stays, the bias still trains
    ```

    `fine_tune` hands AdamW every parameter, frozen ones too. A frozen one
    never gets a `.grad`, and AdamW skips a parameter whose `.grad` is None.
    So it never moves and holds no optimizer state.

    `model.named_parameters()` gives each parameter with its name, such as
    `h.3.attn.c_attn.weight`. A name's start says where it sits, so `"h.3."`
    picks the last block. `str.startswith` also takes a tuple of starts.

    GPT-2's tied head is a trap here. `lm_head.weight` is `wte.weight`
    itself, so `named_parameters` lists it once, as `wte.weight`. Freezing the
    embeddings freezes the head too.

    Here is the run behind this exercise on MPS. Each choice trained for 300
    steps at 0.0001 from your gpt3 model.

    | Trained | Weights | Persuasion | Shakespeare |
    |---|---|---|---|
    | nothing | 0 | 1.744 | 1.535 |
    | everything | 818,048 | 1.302 | 1.897 |
    | the last block | 198,272 | 1.361 | 1.763 |
    | the first block | 198,272 | 1.472 | 1.778 |
    | `wte` and `wpe` | 24,704 | 1.554 | 1.708 |
    | the LayerNorms | 2,304 | 1.657 | 1.567 |

    The fewer weights train, the less the model forgets, and the less it
    learns. The last block gives the best trade. It takes most of the gain with
    a quarter of the weights, and forgets less.

    The first block, the same size, learns far less. Yosinski and colleagues
    found image networks move from general features in their first layers to
    specific ones in their last. Here too, the late block is the one to
    retrain.

    Fewer training weights can take a larger rate. At 0.001, training
    everything reached 1.358 on Persuasion and 2.316 on Shakespeare. The last
    block alone did better on both, 1.313 and 2.014.

    A frozen weight also costs less memory. It needs no gradient, and none of
    the two numbers AdamW keeps for each weight it trains. 77_lora makes that
    saving the whole point.
    """,
)

FINETUNE2_SOURCE = source_file(
    '''
    # finetune2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_persuasion, load_shakespeare
    from tensorlings.training import load_checkpoint

    layer = nn.Linear(2, 2)
    layer.weight.requires_grad_(False)                            # the weight stays, the bias still trains
    starts = "h.3.mlp.c_fc.weight".startswith(("h.3.", "ln_f"))    # True
    sizes = [parameter.numel() for parameter in layer.parameters()]    # [4, 2]


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2.
''' + indented(MODEL_PART) + '''


    # get_batch and the Shakespeare splits from gpt1, and validation_loss from gpt3.
''' + indented(DATA_PART) + '''


''' + indented(VALIDATION_PART) + '''


''' + indented(LOAD_PART) + '''


    # encode_known, fine_tune, losses_on and persuasion_splits from finetune1.
''' + indented(FINE_TUNE_PARTS) + '''


    # TODO: Let every parameter of model whose name starts with one of prefixes
    #   train, and freeze every other one. A parameter frozen before but named
    #   in prefixes trains again.
    def freeze_except(
        model: nn.Module,
        prefixes: tuple[str, ...],
    ) -> None:
        ...


    # TODO: Return how many numbers of model's parameters still train.
    def trainable_count(model: nn.Module) -> int:
        ...
    ''',
)

FINETUNE2_SOLUTION = source_file(
    '''
    # finetune2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_persuasion, load_shakespeare
    from tensorlings.training import load_checkpoint

    layer = nn.Linear(2, 2)
    layer.weight.requires_grad_(False)
    starts = "h.3.mlp.c_fc.weight".startswith(("h.3.", "ln_f"))
    sizes = [parameter.numel() for parameter in layer.parameters()]


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(DATA_PART) + '''


''' + indented(VALIDATION_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(FINE_TUNE_PARTS) + '''


''' + indented(FREEZE_SOLUTION) + '''
    ''',
)


def training_flags(model: nn.Module) -> dict[str, bool]:
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def check_freeze_except(found: object) -> None:
    """Each choice from a model that trains whole and from one frozen whole, so both freezing and thawing show."""
    # No name starts with "attn.", so it freezes everything, while a prefix found mid name would not.
    for prefixes in (("h.1.", "ln_f"), ("wte",), ("",), ("h.0.attn.", "wpe"), ("attn.",)):
        for start_frozen in (False, True):
            model = small_gpt()
            reference = small_gpt()
            model.requires_grad_(not start_frozen)
            reference.requires_grad_(not start_frozen)
            found(model, prefixes)
            FINE_TUNE_REFERENCE["freeze_except"](reference, prefixes)
            wrong = [name for name, flag in training_flags(model).items() if flag != training_flags(reference)[name]]
            start = "frozen whole" if start_frozen else "training whole"

            if wrong:
                raise Mismatch(
                    f"After freeze_except(a small GPT {start}, {prefixes!r}), {wrong[0]} has requires_grad "
                    f"{training_flags(model)[wrong[0]]}, but it should be {training_flags(reference)[wrong[0]]}.",
                )


def check_frozen_stay(found: object) -> None:
    model = small_gpt()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    found(model, ("h.1.",))
    FINE_TUNE_REFERENCE["fine_tune"](model, small_data(769), 2, 0.01, torch.Generator().manual_seed(770), torch.device("cpu"))

    for name, parameter in model.named_parameters():
        moved = not torch.equal(parameter.detach(), before[name])

        if moved and not name.startswith("h.1."):
            raise Mismatch(f"After freeze_except(model, ('h.1.',)) and two steps of fine_tune, {name} moved, but it should be frozen.")

        if not moved and name.startswith("h.1.") and name.endswith("weight"):
            raise Mismatch(f"After freeze_except(model, ('h.1.',)) and two steps of fine_tune, {name} stayed, but it should train.")


def partly_frozen_gpt(prefixes: tuple[str, ...]) -> nn.Module:
    model = small_gpt()
    FINE_TUNE_REFERENCE["freeze_except"](model, prefixes)

    return model


def check_finetune2(checker: Checker) -> None:
    checker.test(
        function="freeze_except",
        description="lets the parameters named by the prefixes train and freezes the rest",
        run=check_freeze_except,
    )
    checker.test(
        function="freeze_except",
        description="keeps frozen weights still through fine_tune",
        run=check_frozen_stay,
    )
    check_on_inputs(
        checker=checker,
        function="trainable_count",
        description="counts the numbers that still train, the tied head once",
        inputs=(
            (Described(partly_frozen_gpt(("h.0.",)), "a small GPT with only its first block training"),),
            (Described(partly_frozen_gpt(("",)), "a small GPT training whole"),),
            (Described(partly_frozen_gpt(("wte", "ln_f")), "a small GPT with only wte and ln_f training"),),
        ),
        reference=FINE_TUNE_REFERENCE["trainable_count"],
        expect=expect_value,
    )


FINETUNE2 = Exercise(
    name="finetune2",
    chapter=FINETUNING,
    title="Which layers to freeze",
    lesson=FINETUNE2_LESSON,
    source=FINETUNE2_SOURCE,
    solution=FINETUNE2_SOLUTION,
    hints={
        "freeze_except": "Loop over model.named_parameters() and call parameter.requires_grad_ with "
        "name.startswith(prefixes), which is True or False.",
        "trainable_count": "Sum parameter.numel() over model.parameters(), keeping the ones whose requires_grad "
        "is True. model.parameters() gives the tied head once.",
    },
    introduces=(),
    check=check_finetune2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
