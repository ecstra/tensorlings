"""83_post_training, Section 11's project. gpt3's GPT learns pig latin through LoRA from demonstrations that carry
one systematic slip, DPO moves it toward the preferred answers, and exact match and a reward model score both."""

from __future__ import annotations

import copy
import math
import re
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
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
)
from ..text import prose, source_file
from ..training import MissingCheckpoint, best_device, load_checkpoint, save_checkpoint
from .dpo import DPO_LOSS_SOLUTION, LOGPROB_SOLUTION
from .finetuning import GPT_SETTINGS, LOAD_PART, MODEL_PART
from .lora import EVERYWHERE_SOLUTION, LORA_PART, MERGE_PART
from .reward_models import REWARD_MODEL_PART, REWARD_SOLUTION

POST_TRAINING = Chapter(
    folder="83_post_training",
    title="Post training your GPT",
    readme=source_file(
        """
        # Post training your GPT

        Section 11's project runs post training end to end on your GPT from
        gpt3, the way nanochat runs SFT and then RL on its own base model. LoRA
        teaches it a made up task, pig latin, from demonstrations whose writer
        had one rule wrong. DPO then moves it toward the answers a judge
        preferred, and a reward model and exact match score what each stage
        gained.

        ## Further reading

        - [Karpathy, nanochat](https://github.com/karpathy/nanochat)
        - [Hu and colleagues, LoRA](https://arxiv.org/abs/2106.09685)
        - [Rafailov and colleagues, direct preference optimization](https://arxiv.org/abs/2305.18290)
        - [Ouyang and colleagues, training language models to follow instructions](https://arxiv.org/abs/2203.02155)
        - [Hugging Face TRL, the DPO trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
        """,
    ),
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def joined(*parts: str) -> str:
    """Parts of code one after another, two blank lines apart as in a file."""
    return "\n\n\n".join(part.strip("\n") for part in parts) + "\n"


# lora1's layer without add_lora, which puts it on the attention alone, and lora2's merge of one layer.
LORA_LAYER_PART = LORA_PART[: LORA_PART.index("def add_lora(")]
MERGE_ONE_PART = MERGE_PART[MERGE_PART.index("def merge(layer") : MERGE_PART.index("def merge_lora(")]

# The made up task. Each answer follows a rule, so every answer can be scored exactly.
TASK_PART = '''
VOWELS = "aeiou"


# How many consonants lead word, 0 when it starts with a vowel. A y after the first letter counts as a vowel.
def leading_consonants(word: str) -> int:
    if word[0] in VOWELS:
        return 0

    for place in range(1, len(word)):
        if word[place] in VOWELS + "y":
            return place

    return len(word)


# The right answer. A word that starts with a vowel gets way. Any other moves its leading consonants to the end and
# gets ay.
def pig_latin(word: str) -> str:
    lead = leading_consonants(word)

    if lead == 0:
        return word + "way"

    return word[lead:] + word[:lead] + "ay"


# A near miss, the slip a hurried writer makes. ay and way swap, or only the first of several consonants moves.
def slip(word: str) -> str:
    lead = leading_consonants(word)

    if lead == 0:
        return word + "ay"

    if lead == 1:
        return word[1:] + word[0] + "way"

    return word[1:] + word[0] + "ay"


# The demonstrations' writer always moved only the first consonant, so every word led by several got the slip.
def demonstration(word: str) -> str:
    if leading_consonants(word) >= 2:
        return slip(word)

    return pig_latin(word)


# Shakespeare's lowercase words of 3 to 8 letters, shuffled the same way every run, and 500 held out for tests.
def task_words() -> tuple[list[str], list[str]]:
    found = re.findall(r"\\b[a-z]+\\b", load_shakespeare())
    words = sorted({word for word in found if 3 <= len(word) <= 8 and slip(word) != pig_latin(word)})
    order = torch.randperm(len(words), generator=torch.Generator().manual_seed(0)).tolist()
    shuffled = [words[place] for place in order]

    return shuffled[500:], shuffled[:500]


# The plain text format, a question line and an answer line. A newline ends the answer.
def prompt_text(word: str) -> str:
    return f"Q: {word}\\nA: "
'''

CHAT_SOLUTION = '''
def chat_example(
    word: str,
    answer: str,
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    prompt = prompt_text(word)
    ids = torch.tensor([positions[character] for character in prompt + answer + "\\n"])
    x = ids[:-1]
    y = ids[1:].clone()
    y[: len(prompt) - 1] = -100

    return x, y
'''

BATCH_PART = '''
# Rows of different lengths as one batch, x padded with newlines and y with -100, which the loss skips.
def pad_examples(
    rows: list[tuple[Tensor, Tensor]],
    newline: int,
) -> tuple[Tensor, Tensor]:
    longest = max(len(row_x) for row_x, _ in rows)
    x = torch.full((len(rows), longest), newline)
    y = torch.full((len(rows), longest), -100)

    for place, (row_x, row_y) in enumerate(rows):
        x[place, : len(row_x)] = row_x
        y[place, : len(row_y)] = row_y

    return x, y


# A batch of count demonstrations, of words drawn at random.
def demonstration_batch(
    words: list[str],
    positions: dict[str, int],
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    picked = [words[place] for place in torch.randint(0, len(words), (count,), generator=generator).tolist()]
    rows = [chat_example(word, demonstration(word), positions) for word in picked]

    return pad_examples(rows, positions["\\n"])
'''

STEP_SOLUTION = '''
def sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    scores = model(x.to(device))
    loss = F.cross_entropy(scores.flatten(0, 1), y.to(device).flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

ANSWER_SOLUTION = '''
def answer(
    model: nn.Module,
    word: str,
    positions: dict[str, int],
    characters: list[str],
    device: torch.device,
) -> str:
    ids = torch.tensor([[positions[character] for character in prompt_text(word)]], device=device)
    written = ""

    with torch.no_grad():
        for _ in range(12):
            token = model(ids)[0, -1].argmax().item()

            if characters[token] == "\\n":
                break

            written += characters[token]
            ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=1)

    return written


def scores(
    model: nn.Module,
    words: list[str],
    positions: dict[str, int],
    characters: list[str],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    answers = [answer(model, word, positions, characters, device) for word in words]
    model.train()
    right = sum(written == pig_latin(word) for word, written in zip(words, answers))
    slipped = sum(written == slip(word) for word, written in zip(words, answers))

    return right / len(words), slipped / len(words)
'''

POST1_MAIN = '''
    # Run by .venv/bin/python run.py train post1, never by the app.
    def main() -> None:
        characters = sorted(set(load_shakespeare()))
        positions = {character: place for place, character in enumerate(characters)}
        train_words, test_words = task_words()
        device = best_device()
        torch.manual_seed(0)
        model = add_lora_everywhere(trained_gpt(), 4, 4.0).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.0)
        generator = torch.Generator().manual_seed(0)
        steps = 1000
        print(f"Teaching on {device} for {steps} steps, from the demonstrations.")

        for step in range(steps):
            x, y = demonstration_batch(train_words, positions, 64, generator)
            loss = sft_step(model, optimizer, x, y, device)

            if (step + 1) % 250 == 0:
                right, slipped = scores(model, test_words[:200], positions, characters, device)
                print(f"Step {step + 1}, loss {loss:.3f}, right {right:.1%}, the slip {slipped:.1%}")

        save_checkpoint(model, "post1")
        print("Saved the model in checkpoints/post1.pt. post2 reads it back.")
'''

MERGE_EVERYWHERE_SOLUTION = '''
def merge_everywhere(model: nn.Module) -> nn.Module:
    for block in model.h:
        block.attn.c_attn = merge(block.attn.c_attn)
        block.attn.c_proj = merge(block.attn.c_proj)
        block.mlp.c_fc = merge(block.mlp.c_fc)
        block.mlp.c_proj = merge(block.mlp.c_proj)

    return model
'''

REFERENCE_SOLUTION = '''
def reference_logprobs(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    layers = [module for module in model.modules() if isinstance(module, LoRALinear)]
    kept_scales = [layer.scale for layer in layers]

    for layer in layers:
        layer.scale = 0.0

    with torch.no_grad():
        logprobs = answer_logprobs(model, x, y)

    for layer, scale in zip(layers, kept_scales):
        layer.scale = scale

    return logprobs
'''

DPO_STEP_SOLUTION = '''
def dpo_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen: tuple[Tensor, Tensor],
    rejected: tuple[Tensor, Tensor],
    beta: float,
    nll: float,
    device: torch.device,
) -> float:
    chosen_x, chosen_y = chosen[0].to(device), chosen[1].to(device)
    rejected_x, rejected_y = rejected[0].to(device), rejected[1].to(device)
    reference_chosen = reference_logprobs(model, chosen_x, chosen_y)
    reference_rejected = reference_logprobs(model, rejected_x, rejected_y)
    policy_chosen = answer_logprobs(model, chosen_x, chosen_y)
    policy_rejected = answer_logprobs(model, rejected_x, rejected_y)
    chosen_nll = -policy_chosen.sum() / (chosen_y != -100).sum()
    loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta) + nll * chosen_nll
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

PAIRS_PART = '''
# The preference pairs, the right answer chosen and the slip rejected, as two batches of the same words.
def pair_batch(
    words: list[str],
    positions: dict[str, int],
    count: int,
    generator: torch.Generator,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    picked = [words[place] for place in torch.randint(0, len(words), (count,), generator=generator).tolist()]
    chosen = pad_examples([chat_example(word, pig_latin(word), positions) for word in picked], positions["\\n"])
    rejected = pad_examples([chat_example(word, slip(word), positions) for word in picked], positions["\\n"])

    return chosen, rejected


# post1's model read back from checkpoints/post1.pt, with LoRA on every Linear as it was trained.
def taught_gpt() -> nn.Module:
    model = add_lora_everywhere(GPT(65, BLOCK_SIZE, SIZE, LAYERS, HEADS), 4, 4.0)

    return load_checkpoint(model, "post1")
'''

POST2_MAIN = '''
    # Run by .venv/bin/python run.py train post2, never by the app.
    def main() -> None:
        characters = sorted(set(load_shakespeare()))
        positions = {character: place for place, character in enumerate(characters)}
        train_words, test_words = task_words()
        device = best_device()
        torch.manual_seed(0)
        model = add_lora_everywhere(merge_everywhere(taught_gpt()), 4, 4.0).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
        generator = torch.Generator().manual_seed(1)
        steps = 300
        print(f"Training on {device} for {steps} steps of DPO, from the preference pairs.")

        for step in range(steps):
            chosen, rejected = pair_batch(train_words, positions, 32, generator)
            loss = dpo_step(model, optimizer, chosen, rejected, 0.1, 0.2, device)

            if (step + 1) % 100 == 0:
                right, slipped = scores(model, test_words[:200], positions, characters, device)
                print(f"Step {step + 1}, loss {loss:.3f}, right {right:.1%}, the slip {slipped:.1%}")

        save_checkpoint(model, "post2")
        print("Saved the model in checkpoints/post2.pt. post3 reads it back.")
'''

REWARD_ROWS_SOLUTION = '''
def reward_rows(
    words: list[str],
    answers: list[str],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    texts = [prompt_text(word) + written + "\\n" for word, written in zip(words, answers)]
    ids = torch.full((len(texts), max(len(text) for text in texts)), positions["\\n"])

    for place, text in enumerate(texts):
        ids[place, : len(text)] = torch.tensor([positions[character] for character in text])

    return ids, torch.tensor([len(text) - 1 for text in texts])


def mean_reward(
    reward_model: nn.Module,
    words: list[str],
    answers: list[str],
    positions: dict[str, int],
) -> float:
    reward_model.eval()

    with torch.no_grad():
        rewards = reward_model(*reward_rows(words, answers, positions))

    reward_model.train()

    return rewards.mean().item()
'''

POST3_MAIN = '''
    # Run by .venv/bin/python run.py train post3, never by the app.
    def main() -> None:
        characters = sorted(set(load_shakespeare()))
        positions = {character: place for place, character in enumerate(characters)}
        train_words, test_words = task_words()
        device = best_device()
        torch.manual_seed(0)
        reward_model = RewardModel(trained_gpt()).to(device)
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=3e-4, weight_decay=0.0)
        generator = torch.Generator().manual_seed(2)
        steps = 200
        print(f"Training the reward model on {device} for {steps} steps, from the preference pairs.")

        for step in range(steps):
            picked = [train_words[place] for place in torch.randint(0, len(train_words), (32,), generator=generator).tolist()]
            chosen_ids, chosen_last = reward_rows(picked, [pig_latin(word) for word in picked], positions)
            rejected_ids, rejected_last = reward_rows(picked, [slip(word) for word in picked], positions)
            chosen = (chosen_ids.to(device), chosen_last.to(device))
            rejected = (rejected_ids.to(device), rejected_last.to(device))
            loss = reward_step(reward_model, optimizer, chosen, rejected)

            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}, loss {loss:.3f}")

        save_checkpoint(reward_model, "post3")
        print("Saved the reward model in checkpoints/post3.pt.")
        reward_model.cpu()
        words = test_words[:200]

        for name, model in (("post1", taught_gpt()), ("post2", preferred_gpt())):
            right, slipped = scores(model, words, positions, characters, torch.device("cpu"))
            answers = [answer(model, word, positions, characters, torch.device("cpu")) for word in words]
            reward = mean_reward(reward_model, words, answers, positions)
            print(f"{name}: right {right:.1%}, the slip {slipped:.1%}, mean reward {reward:.2f}")
'''

PREFERRED_PART = '''
# post2's model read back from checkpoints/post2.pt, its layers merged once and a fresh LoRA on top, as DPO trained it.
def preferred_gpt() -> nn.Module:
    model = add_lora_everywhere(GPT(65, BLOCK_SIZE, SIZE, LAYERS, HEADS), 4, 4.0)

    return load_checkpoint(model, "post2")
'''

# Everything post1 hands over, then what post2 adds, then what post3 adds.
POST1_PARTS = joined(TASK_PART, CHAT_SOLUTION, BATCH_PART, STEP_SOLUTION, ANSWER_SOLUTION)
POST2_GIVEN = joined(MERGE_ONE_PART, LOGPROB_SOLUTION, DPO_LOSS_SOLUTION, PAIRS_PART)
POST2_SOLUTION_PARTS = joined(MERGE_EVERYWHERE_SOLUTION, REFERENCE_SOLUTION, DPO_STEP_SOLUTION)
POST3_GIVEN = joined(PREFERRED_PART, REWARD_MODEL_PART, REWARD_SOLUTION)


def post_namespace() -> dict[str, object]:
    """The given parts and the solutions, run once, so the checks compare against exactly the code the files hold."""
    namespace: dict[str, object] = {
        "math": math,
        "re": re,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
        "load_checkpoint": load_checkpoint,
        "save_checkpoint": save_checkpoint,
        "best_device": best_device,
    }
    exec(
        joined(
            GPT_SETTINGS,
            MODEL_PART,
            LOAD_PART,
            LORA_LAYER_PART,
            EVERYWHERE_SOLUTION,
            POST1_PARTS,
            POST2_GIVEN,
            POST2_SOLUTION_PARTS,
            POST3_GIVEN,
            REWARD_ROWS_SOLUTION,
        ),
        namespace,
    )

    return namespace


POST_REFERENCE = post_namespace()

# What the checks on the trained models ask for, below the 3 seeds' lowest in measure_post_training, which were
# 71.5% right for post1, 84.5% for post2 and 98.2% of pairs for the reward model.
POST1_BAR = 0.6
POST2_BAR = 0.75
REWARD_BAR = 0.9

SMALL_CHARACTERS = sorted(set("Q: A\nabcdefghijklmnopqrstuvwxyz"))
SMALL_POSITIONS = Described(
    {character: place for place, character in enumerate(SMALL_CHARACTERS)},
    "the places of Q, A, the colon, the space, the newline and a to z",
)
SMALL_NAMES = Described(SMALL_CHARACTERS, "those 31 characters in order")
NEWLINE = SMALL_POSITIONS.value["\n"]


def is_lora(module: nn.Module) -> bool:
    """Any LoRA layer, the reference's or the one a learner's file defines, which is another class of the same name."""
    return type(module).__name__ == "LoRALinear"


def small_lora_gpt(
    seed: int,
    drawn: bool,
    namespace: dict[str, object] = POST_REFERENCE,
) -> nn.Module:
    """A GPT over SMALL_CHARACTERS with rank 2 LoRA on every Linear, its B drawn too when drawn, so the update shows.
    A learner's file defines its own LoRALinear, so a check builds the model from that file's namespace, where an
    isinstance test finds it."""
    torch.manual_seed(seed)
    model = namespace["add_lora_everywhere"](namespace["GPT"](len(SMALL_CHARACTERS), 24, 32, 2, 4), 2, 2.0)

    if drawn:
        generator = torch.Generator().manual_seed(seed + 1)

        with torch.no_grad():
            for module in model.modules():
                if is_lora(module):
                    module.B.copy_(torch.randn(module.B.shape, generator=generator) * 0.2)

    return model


def small_batch(
    words: list[str],
    answers: list[str],
) -> tuple[Tensor, Tensor]:
    rows = [POST_REFERENCE["chat_example"](word, written, SMALL_POSITIONS.value) for word, written in zip(words, answers)]

    return POST_REFERENCE["pad_examples"](rows, NEWLINE)


# ---------------------------------------------------------------------------
# post1
# ---------------------------------------------------------------------------

POST1_LESSON = prose(
    """
    This is Section 11's project. Your GPT from gpt3 learns a made up task
    with LoRA. post2 then trains it toward the answers a judge preferred, and
    post3 scores both.

    The task is pig latin. A word that starts with a vowel gets way. Any other
    word moves its leading consonants to the end, then gets ay.

    | Word | Answer |
    |---|---|
    | `apple` | `appleway` |
    | `hello` | `ellohay` |
    | `string` | `ingstray` |

    The demonstrations come from a writer who had one rule wrong. They always
    moved only the first consonant, so `string` became `tringsay`. That slip
    hits every word led by several consonants, 23% of the words.

    Each chat is plain text, a question line and an answer line. A newline
    ends the answer.

    ```text
    Q: string
    A: tringsay
    ```

    sft2 grew the vocabulary for its new tokens. Every character here is one
    gpt3's GPT already knows, so LoRA can leave every original weight frozen.
    `add_lora_everywhere` from lora3 puts rank 4 LoRA on the four Linears of
    each block, 32,768 weights in all. `chat_example` scores the answer
    alone, as sft1 did.

    Once every check passes, train from the project folder. It took about
    three and a half minutes on this Mac's GPU.

    ```text
    .venv/bin/python run.py train post1
    ```

    Here is what it learned, on 200 held out words, the average of 3 seeds.

    | Steps | 250 | 500 | 750 | 1,000 |
    |---|---|---|---|---|
    | Right | 28% | 72% | 77% | 76% |
    | The slip | 10% | 20% | 20% | 21% |

    The model copied the writer, slip and all. From 500 steps on it writes
    the slip for about one word in five. More demonstrations from the same
    writer would only teach the slip again.
    """,
)

POST1_SOURCE = source_file(
    '''
    # post1.py
    #
    # Needs your trained model from gpt3, in checkpoints/gpt3.pt.

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    targets = torch.tensor([7, 8, 9, 10])[1:].clone()    # tensor([ 8,  9, 10])
    made_there = torch.tensor([[1, 2]], device=torch.device("cpu"))    # a tensor made on a device, not moved after
    top = torch.tensor([0.1, 2.0, 0.5]).argmax().item()    # 1
    matches = sum(a == b for a, b in zip("abc", "abd"))    # 2


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, and your model from gpt3 read back.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


    # The LoRA layer from lora1, and add_lora_everywhere from lora3.
''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


    # The made up task and its plain text format.
''' + indented(TASK_PART) + '''


    # TODO: Return x and y for one chat, the prompt of word, then answer, then a
    #   newline, as ids from positions. x is every id but the last, and y every
    #   id but the first. Set y to -100 at every target before the answer's
    #   first character, so the loss scores the answer and its newline alone.
    def chat_example(
        word: str,
        answer: str,
        positions: dict[str, int],
    ) -> tuple[Tensor, Tensor]:
        ...


''' + indented(BATCH_PART) + '''


    # TODO: Take one training step on x and y, moved to device. The loss is the
    #   cross entropy of model's scores against y, which skips -100. Zero the old
    #   gradients, backpropagate, step, and return the loss as a plain number.
    def sft_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
        device: torch.device,
    ) -> float:
        ...


    # TODO: Return the answer model writes to word, without recording for
    #   autograd. Start from the ids of prompt_text(word), made on device. Each
    #   step adds the character with the top score at the last place. Stop at a
    #   newline, which is not part of the answer, or after 12 characters.
    def answer(
        model: nn.Module,
        word: str,
        positions: dict[str, int],
        characters: list[str],
        device: torch.device,
    ) -> str:
        ...


    # TODO: Return the share of words answered right, then the share answered
    #   with the slip, as plain numbers. Answer with model in eval mode, and put
    #   it back in training mode after.
    def scores(
        model: nn.Module,
        words: list[str],
        positions: dict[str, int],
        characters: list[str],
        device: torch.device,
    ) -> tuple[float, float]:
        ...

''' + POST1_MAIN,
)

POST1_SOLUTION = source_file(
    '''
    # post1.py, a solution for comparison

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    targets = torch.tensor([7, 8, 9, 10])[1:].clone()
    made_there = torch.tensor([[1, 2]], device=torch.device("cpu"))
    top = torch.tensor([0.1, 2.0, 0.5]).argmax().item()
    matches = sum(a == b for a, b in zip("abc", "abd"))


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(POST1_PARTS) + '''

''' + POST1_MAIN,
)


def expect_whole_parts(*names: str):
    """An expect function for a pair of tensors that must both hold whole numbers, as ids and targets do."""

    def expect(
        call: str,
        actual: object,
        expected: tuple[Tensor, Tensor],
    ) -> None:
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (tuple, list)) or len(actual) != len(names):
            raise Mismatch(f"{call} should give {len(names)} things, {' and '.join(names)}, but it gives {actual!r}.")

        for name, part, wanted in zip(names, actual, expected):
            if isinstance(part, Tensor) and part.dtype != torch.long:
                raise Mismatch(f"{name} from {call} holds {part.dtype} numbers, but it should hold whole numbers, torch.long, as ids do.")

            expect_tensor(
                call=f"{name} from {call}",
                actual=part,
                expected=wanted,
            )

    return expect


def check_sft_step(found: object) -> None:
    for seed in (831, 832):
        batches = [
            small_batch(["cat", "sky", "oak"], ["atcay", "yskay", "oakway"]),
            small_batch(["trip", "egg", "lip", "spa"], ["iptray", "eggway", "iplay", "aspay"]),
        ]
        results = []

        for step in (found, POST_REFERENCE["sft_step"]):
            model = small_lora_gpt(seed, drawn=True, namespace=step.__globals__)
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
            # Two steps, since gradients left from a first step only show in a second.
            losses = [step(model, optimizer, x, y, torch.device("cpu")) for x, y in batches]
            results.append((losses, model.h[1].mlp.c_fc.A.detach().clone()))

        (losses, weight), (expected_losses, expected_weight) = results

        for place, (loss, expected) in enumerate(zip(losses, expected_losses)):
            expect_close(
                call=f"sft_step's loss on a small GPT with LoRA, step {place + 1}",
                actual=loss,
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="one LoRA A after two calls of sft_step",
            actual=weight,
            expected=expected_weight,
        )

    x, y = small_batch(["cat", "sky"], ["atcay", "yskay"])
    recorder = DeviceRecorder(
        names=("x",),
        output=lambda ids: torch.zeros(*ids.shape, len(SMALL_CHARACTERS), device="meta"),
    )
    expect_on_device(
        function="sft_step",
        run=lambda: found(recorder, torch.optim.SGD(recorder.parameters(), lr=0.1), x, y, torch.device("meta")),
        recorders=[recorder],
        advice="Move x and y to device before the model and the loss.",
    )


class ScriptedWriter(nn.Module):
    """A stand in model that writes script, one character per call, and notes each call's ids and mode."""

    def __init__(
        self,
        prompt_length: int,
        script: list[int],
    ) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.script = script
        self.calls: list[tuple[list[int], bool]] = []

    def forward(self, ids: Tensor) -> Tensor:
        self.calls.append((ids[0].tolist(), torch.is_grad_enabled()))
        written = ids.shape[1] - self.prompt_length
        scores = torch.zeros(ids.shape[0], ids.shape[1], len(SMALL_CHARACTERS))
        scores[:, -1, self.script[min(written, len(self.script) - 1)]] = 5.0

        return scores


def check_answer(found: object) -> None:
    positions = SMALL_POSITIONS.value

    for word, script, text in (
        ("cat", "atcay\n", "atcay, then a newline"),
        ("oak", "oakway\nzz", "oakway, then a newline"),
        ("sky", "y", "y forever"),
    ):
        prompt = [positions[character] for character in f"Q: {word}\nA: "]
        ids = [positions[character] for character in script]
        model = ScriptedWriter(len(prompt), ids)
        expected = POST_REFERENCE["answer"](ScriptedWriter(len(prompt), ids), word, positions, SMALL_CHARACTERS, torch.device("cpu"))
        actual = found(model, word, positions, SMALL_CHARACTERS, torch.device("cpu"))
        call = f"answer of a model that writes {text}, to {word!r}"

        if not model.calls:
            raise Mismatch(f"{call} never ran the model.")

        first_ids, recording = model.calls[0]

        if first_ids != prompt:
            raise Mismatch(f"{call} first ran the model on {first_ids}, but it should read the ids of prompt_text(word), {prompt}.")

        if recording:
            raise Mismatch(f"{call} ran the model while recording for autograd. Write inside torch.no_grad().")

        expect_value(
            call=call,
            actual=actual,
            expected=expected,
        )

    recorder = DeviceRecorder(
        names=("the prompt's ids",),
        output=lambda ids: torch.zeros(*ids.shape, len(SMALL_CHARACTERS), device="meta"),
    )
    expect_on_device(
        function="answer",
        run=lambda: found(recorder, "cat", positions, SMALL_CHARACTERS, torch.device("meta")),
        recorders=[recorder],
        advice="Make the prompt's ids on device, with device=device, and each new id too.",
    )


def check_scores(found: object) -> None:
    for words, written in (
        (["string", "apple", "hello", "sky"], ["tringsay", "appleway", "ellohay", "skyay"]),
        (["oak", "tree", "lip"], ["oakay", "eetray", "iplay"]),
    ):
        model = small_lora_gpt(833, drawn=False)
        modes: list[bool] = []
        replies = dict(zip(words, written))

        def scripted_answer(
            model: nn.Module,
            word: str,
            positions: dict[str, int],
            characters: list[str],
            device: torch.device,
        ) -> str:
            modes.append(model.training)

            return replies[word]

        namespace = found.__globals__
        original = namespace.get("answer")
        namespace["answer"] = scripted_answer

        try:
            actual = found(model, words, SMALL_POSITIONS.value, SMALL_CHARACTERS, torch.device("cpu"))
        finally:
            namespace["answer"] = original

        if len(modes) != len(words):
            raise Mismatch(f"scores should call answer once for each of {len(words)} words, but it called it {len(modes)} times.")

        if any(modes):
            raise Mismatch("scores ran answer with the model in training mode. Call model.eval() first.")

        if not model.training:
            raise Mismatch("scores left the model in eval mode. Call model.train() after.")

        right = sum(reply == POST_REFERENCE["pig_latin"](word) for word, reply in replies.items())
        slipped = sum(reply == POST_REFERENCE["slip"](word) for word, reply in replies.items())
        call = f"scores when answer writes {written} for {words}"
        reject_missing_return(
            call=call,
            actual=actual,
        )

        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise Mismatch(f"{call} should give two shares, right then the slip, but it gives {actual!r}.")

        for name, share, wanted in (("the share right", actual[0], right), ("the share with the slip", actual[1], slipped)):
            expect_close(
                call=f"{name} from {call}",
                actual=share,
                expected=wanted / len(words),
            )


def check_post1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="chat_example",
        description="scores the answer and its newline alone",
        inputs=(
            ("cat", "atcay", SMALL_POSITIONS),
            ("string", "tringsay", SMALL_POSITIONS),
            ("oak", "", SMALL_POSITIONS),
        ),
        reference=POST_REFERENCE["chat_example"],
        expect=expect_whole_parts("x", "y"),
    )
    checker.test(
        function="sft_step",
        description="trains on the answer loss, on device",
        run=check_sft_step,
    )
    checker.test(
        function="answer",
        description="writes the likeliest character until a newline, at most 12",
        run=check_answer,
    )
    checker.test(
        function="scores",
        description="gives the shares right and with the slip, in eval mode",
        run=check_scores,
    )


POST1 = Exercise(
    name="post1",
    chapter=POST_TRAINING,
    title="Project: teaching pig latin with LoRA",
    lesson=POST1_LESSON,
    source=POST1_SOURCE,
    solution=POST1_SOLUTION,
    hints={
        "chat_example": "prompt = prompt_text(word). Make one tensor of the ids of prompt + answer + a newline. x is "
        "ids[:-1] and y is ids[1:].clone(). The answer starts at place len(prompt), so its first target sits at place "
        "len(prompt) - 1. Set y before that to -100.",
        "sft_step": "scores = model(x.to(device)). F.cross_entropy of scores.flatten(0, 1) and y.to(device).flatten(). "
        "Then zero_grad, backward, step, and loss.item().",
        "answer": "Make ids with torch.tensor([[...]], device=device). Inside torch.no_grad(), loop 12 times. Take "
        "the argmax of model(ids)[0, -1] with .item(), and break if characters[token] is a newline. Otherwise add "
        "the character, and torch.cat the token onto ids, made on device too.",
        "scores": "model.eval(), then one answer per word, then model.train(). Count the answers equal to "
        "pig_latin(word) and the ones equal to slip(word), and divide each count by len(words).",
    },
    introduces=(),
    check=check_post1,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# post2
# ---------------------------------------------------------------------------

POST2_LESSON = prose(
    """
    post1's model writes the slip for most words led by several consonants.
    New demonstrations cost a lot to write. Judging which of two answers is
    right costs far less.

    Each preference pair holds a right answer, chosen, and its slip,
    rejected. dpo1's loss raises the chosen answer's log chance against the
    rejected one's, each measured against a reference that stays fixed.

    LoRA hands you that reference. `merge_everywhere` folds post1's update
    into its layers with lora2's `merge`, and a fresh LoRA goes on top. Its B
    starts at zero, so the model starts as post1's. Set every update's scale
    to 0, and it is post1's again at any step. Hugging Face's TRL trainer
    gets its reference the same way for a new LoRA.

    ```python
    layer.scale = 0.0    # the update adds nothing, so the layer is its base alone
    ```

    dpo2 found that plain DPO can pull the chosen answer down along with the
    rejected one. So `dpo_step` adds Llama 3's term, 0.2 times the chosen
    answers' cross entropy. That is minus their summed log chance over the
    count of their scored tokens, which `answer_logprobs` already gives.

    Here is DPO from post1's model, 300 steps of 32 pairs, on 200 held out
    words, the average of 3 seeds.

    | Run | Right | The slip |
    |---|---|---|
    | post1's model | 76% | 21% |
    | Plain DPO, at a rate of 0.001 | 1% | 0% |
    | Plain DPO, at a rate of 0.0001 | 4% | 0% |
    | DPO with the 0.2 term | 89% | 0% |

    Plain DPO wrecked the model at both rates. It pushed the slip down, and
    the right answers fell with it. With the term, the slip is gone and right
    answers rise. The path is bumpy, as one seed dipped to 55% at step 200.

    Not every slip turned right. In one run, about half the words led by
    several consonants got a new mistake. The whole cluster moved, but a
    letter of it was miscopied, as in `slop` to `opstay`.

    Once every check passes, train from the project folder. It took about a
    minute on this Mac's GPU.

    ```text
    .venv/bin/python run.py train post2
    ```
    """,
)

POST2_SOURCE = source_file(
    '''
    # post2.py
    #
    # Needs your trained model from post1, in checkpoints/post1.pt.

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    linears = [module for module in nn.Sequential(nn.Linear(2, 2), nn.ReLU()).modules() if isinstance(module, nn.Linear)]    # the one Linear
    counted = (torch.tensor([3, -100, 5]) != -100).sum()    # tensor(2)


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, your model from gpt3, LoRA, the task and post1's code.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(POST1_PARTS) + '''


    # lora2's merge of one layer, dpo1's log chances and loss, and the pairs.
''' + indented(POST2_GIVEN) + '''


    # TODO: Fold every LoRA layer of model back into a plain Linear with merge,
    #   the four Linears of each block in model.h, and return model.
    def merge_everywhere(model: nn.Module) -> nn.Module:
        ...


    # TODO: Return answer_logprobs of model on x and y with every LoRA update
    #   switched off, without recording for autograd. Set the scale of every
    #   LoRALinear in model.modules() to 0, then put each one's scale back.
    def reference_logprobs(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...


    # TODO: Take one DPO step on the pairs, each an (x, y) moved to device. The
    #   reference's log chances come from reference_logprobs, the policy's from
    #   answer_logprobs. The loss is dpo_loss at beta, plus nll times the chosen
    #   answers' cross entropy, minus their summed log chance over the count of
    #   their targets that are not -100. Step, and return the loss as a plain
    #   number.
    def dpo_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chosen: tuple[Tensor, Tensor],
        rejected: tuple[Tensor, Tensor],
        beta: float,
        nll: float,
        device: torch.device,
    ) -> float:
        ...

''' + POST2_MAIN,
)

POST2_SOLUTION = source_file(
    '''
    # post2.py, a solution for comparison

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    linears = [module for module in nn.Sequential(nn.Linear(2, 2), nn.ReLU()).modules() if isinstance(module, nn.Linear)]
    counted = (torch.tensor([3, -100, 5]) != -100).sum()


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(POST1_PARTS) + '''


''' + indented(POST2_GIVEN) + '''


''' + indented(POST2_SOLUTION_PARTS) + '''

''' + POST2_MAIN,
)


def reference_model(
    name: str,
    loader: str,
) -> nn.Module:
    """A trained model from the project's own checkpoint, read with the reference's code."""
    try:
        return POST_REFERENCE[loader]()
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            f"The saved model in checkpoints/{name}.pt doesn't fit the model as it is now. Train it again from the "
            f"project folder.\n\n    .venv/bin/python run.py train {name}",
        ) from None


def held_out_scores(
    model: nn.Module,
    count: int,
) -> tuple[float, float]:
    characters = sorted(set(load_shakespeare()))
    positions = {character: place for place, character in enumerate(characters)}
    _, test_words = POST_REFERENCE["task_words"]()

    return POST_REFERENCE["scores"](model, test_words[:count], positions, characters, torch.device("cpu"))


def check_taught_gpt(found: object) -> None:
    right, slipped = held_out_scores(reference_model("post1", "taught_gpt"), 100)

    if right < POST1_BAR:
        raise Mismatch(
            f"Your model from post1 answers {right:.0%} of 100 held out words right, but it should reach "
            f"{POST1_BAR:.0%}. Check post1 and train it again.",
        )


def outputs_of(
    model: nn.Module,
    seed: int,
) -> Tensor:
    ids = torch.randint(0, len(SMALL_CHARACTERS), (2, 9), generator=torch.Generator().manual_seed(seed))

    with torch.no_grad():
        return model(ids)


def check_merge_everywhere(found: object) -> None:
    for seed in (841, 842):
        model = small_lora_gpt(seed, drawn=True, namespace=found.__globals__)
        expected = outputs_of(POST_REFERENCE["merge_everywhere"](small_lora_gpt(seed, drawn=True)), seed)
        merged = found(model)
        call = f"merge_everywhere of a small GPT with LoRA on every Linear, seed {seed}"
        reject_missing_return(
            call=call,
            actual=merged,
        )
        left = [name for name, module in merged.named_modules() if is_lora(module)]

        if left:
            raise Mismatch(f"After {call}, {', '.join(left)} still hold LoRA layers. Merge all four Linears of every block.")

        expect_tensor(
            call=f"the scores of the model {call} gives",
            actual=outputs_of(merged, seed),
            expected=expected,
        )


def check_reference_logprobs(found: object) -> None:
    for seed, words, answers in (
        (843, ["cat", "sky"], ["atcay", "kysay"]),
        (844, ["oak", "trip", "egg"], ["oakway", "iptray", "eggay"]),
    ):
        model = small_lora_gpt(seed, drawn=True, namespace=found.__globals__)
        x, y = small_batch(words, answers)
        before = outputs_of(model, seed)
        switched_off = small_lora_gpt(seed, drawn=True)

        for module in switched_off.modules():
            if is_lora(module):
                module.B.data.zero_()

        expected = POST_REFERENCE["answer_logprobs"](switched_off, x, y).detach()
        actual = found(model, x, y)
        call = f"reference_logprobs of a small GPT with LoRA, for {answers}"
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )

        if actual.requires_grad:
            raise Mismatch(f"{call} recorded for autograd. Work out the reference inside torch.no_grad().")

        expect_tensor(
            call=f"the model's scores after {call}, which should have every scale put back",
            actual=outputs_of(model, seed),
            expected=before,
        )


def check_dpo_step(found: object) -> None:
    chosen = small_batch(["cat", "string", "oak"], ["atcay", "ingstray", "oakway"])
    rejected = small_batch(["cat", "string", "oak"], ["atcway", "tringsay", "oakay"])
    second_chosen = small_batch(["sky", "lip"], ["yskay", "iplay"])
    second_rejected = small_batch(["sky", "lip"], ["kysay", "iplway"])

    for seed in (845, 846):
        results = []

        for step in (found, POST_REFERENCE["dpo_step"]):
            model = small_lora_gpt(seed, drawn=True, namespace=step.__globals__)
            optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
            # Two steps, at other settings, so leftover gradients and a typed in beta or nll show.
            losses = [
                step(model, optimizer, chosen, rejected, 0.1, 0.2, torch.device("cpu")),
                step(model, optimizer, second_chosen, second_rejected, 0.5, 1.0, torch.device("cpu")),
            ]
            results.append((losses, model.h[0].attn.c_attn.B.detach().clone()))

        (losses, weight), (expected_losses, expected_weight) = results

        for place, (loss, expected) in enumerate(zip(losses, expected_losses)):
            expect_close(
                call=f"dpo_step's loss on a small GPT with LoRA, step {place + 1}",
                actual=loss,
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="one LoRA B after two calls of dpo_step",
            actual=weight,
            expected=expected_weight,
        )

    recorder = DeviceRecorder(
        names=("the ids",),
        output=lambda ids: torch.zeros(*ids.shape, len(SMALL_CHARACTERS), device="meta"),
    )
    expect_on_device(
        function="dpo_step",
        run=lambda: found(recorder, torch.optim.SGD(recorder.parameters(), lr=0.1), chosen, rejected, 0.1, 0.2, torch.device("meta")),
        recorders=[recorder],
        advice="Move each pair's x and y to device first.",
    )


def check_post2(checker: Checker) -> None:
    checker.test(
        function="merge_everywhere",
        description="folds every LoRA layer back into a Linear",
        run=check_merge_everywhere,
    )
    checker.test(
        function="reference_logprobs",
        description="scores with every update switched off, then puts it back",
        run=check_reference_logprobs,
    )
    checker.test(
        function="dpo_step",
        description="takes a DPO step with the NLL term, on device",
        run=check_dpo_step,
    )
    checker.test(
        function="taught_gpt",
        description="your post1 model answers most held out words right",
        run=check_taught_gpt,
    )


POST2 = Exercise(
    name="post2",
    chapter=POST_TRAINING,
    title="Project: preferring the right answer",
    lesson=POST2_LESSON,
    source=POST2_SOURCE,
    solution=POST2_SOLUTION,
    hints={
        "merge_everywhere": "For each block in model.h, set block.attn.c_attn to merge(block.attn.c_attn), and the "
        "same for block.attn.c_proj, block.mlp.c_fc and block.mlp.c_proj. Return model.",
        "reference_logprobs": "Gather the LoRALinear modules and keep their scales in a list. Set each scale to 0.0, "
        "run answer_logprobs inside torch.no_grad(), then set each scale back.",
        "dpo_step": "Move the four tensors to device. Two reference_logprobs and two answer_logprobs give the four "
        "log chances. The NLL is minus the policy's chosen log chances summed, over (chosen y != -100).sum(). Add "
        "nll times it to dpo_loss, then zero_grad, backward, step, and loss.item().",
        "taught_gpt": "Your post1 model is read from checkpoints/post1.pt. Train post1 again if it scores low.",
    },
    introduces=(),
    check=check_post2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# post3
# ---------------------------------------------------------------------------

POST3_LESSON = prose(
    """
    Exact match needs a rule that knows every right answer. Real tasks rarely
    have one, which is why reward1 trained a model to score answers. Pig latin
    has the rule, so post3 checks one against the other.

    The reward model is reward1's, on gpt3's GPT, trained with the Bradley
    Terry loss on the same pairs post2 used. It reads a whole chat and gives
    one number, read at the chat's last newline. `reward_rows` builds those
    chats, with the place of each one's last newline.

    ```python
    torch.full((2, 3), 9)    # tensor([[9, 9, 9], [9, 9, 9]])
    ```

    Here is the reward model on the 500 held out words, each pair's right
    answer against its slip, the average of 3 seeds.

    | Words led by | Right answer ranked first |
    |---|---|
    | a vowel | 97.7% |
    | one consonant | 99.6% |
    | several consonants | 100% |

    Here is each stage's own answers to 200 held out words, scored both
    ways, the average of 3 seeds.

    | Model | Right | The slip | Mean reward |
    |---|---|---|---|
    | post1 | 76% | 21% | 3.36 |
    | post2 | 89% | 0% | 5.68 |

    Both scores agree that post2 is better. Each seed's reward model has its
    own scale, so compare rewards within one seed, where post2 won every time.

    Ranking pairs well doesn't make it a judge of right answers. In one run
    it scored right answers about 7 and slips about -7. Mistakes it never saw,
    like `opstay`, scored 4 to 6, close to right. It learned what tells the
    slip from the right answer, nothing more. A model trained against it
    could learn mistakes it can't see, the hacking reward2 showed on a toy.

    Once every check passes, train from the project folder. It took about
    half a minute on this Mac's GPU, and prints the second table for your
    own models.

    ```text
    .venv/bin/python run.py train post3
    ```
    """,
)

POST3_SOURCE = source_file(
    '''
    # post3.py
    #
    # Needs your trained models from gpt3, post1 and post2, in checkpoints/.

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    filled = torch.full((2, 3), 9)    # tensor([[9, 9, 9], [9, 9, 9]])
    lasts = torch.tensor([len(text) - 1 for text in ("ab", "abcd")])    # tensor([1, 3])


''' + indented(GPT_SETTINGS) + '''


    # The model from gpt2, your model from gpt3, LoRA, the task and the code of
    # post1 and post2.
''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(POST1_PARTS) + '''


''' + indented(POST2_GIVEN) + '''


''' + indented(POST2_SOLUTION_PARTS) + '''


    # post2's model read back, and reward1's reward model with its training step.
''' + indented(POST3_GIVEN) + '''


    # TODO: Return the ids of each word's whole chat, its prompt, its answer and
    #   a newline, as rows padded at the end with newlines. Return also the
    #   place of each chat's own last newline, where the reward model reads it.
    def reward_rows(
        words: list[str],
        answers: list[str],
        positions: dict[str, int],
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return reward_model's mean score of the answers, as a plain number,
    #   without recording for autograd. Score in eval mode, and put
    #   reward_model back in training mode after.
    def mean_reward(
        reward_model: nn.Module,
        words: list[str],
        answers: list[str],
        positions: dict[str, int],
    ) -> float:
        ...

''' + POST3_MAIN,
)

POST3_SOLUTION = source_file(
    '''
    # post3.py, a solution for comparison

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare
    from tensorlings.training import best_device, load_checkpoint, save_checkpoint

    filled = torch.full((2, 3), 9)
    lasts = torch.tensor([len(text) - 1 for text in ("ab", "abcd")])


''' + indented(GPT_SETTINGS) + '''


''' + indented(MODEL_PART) + '''


''' + indented(LOAD_PART) + '''


''' + indented(LORA_LAYER_PART) + '''


''' + indented(EVERYWHERE_SOLUTION) + '''


''' + indented(POST1_PARTS) + '''


''' + indented(POST2_GIVEN) + '''


''' + indented(POST2_SOLUTION_PARTS) + '''


''' + indented(POST3_GIVEN) + '''


''' + indented(REWARD_ROWS_SOLUTION) + '''

''' + POST3_MAIN,
)


class ScriptedReward(nn.Module):
    """A stand in reward model whose reward is each chat's last place, and that notes its mode and autograd."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls: list[tuple[bool, bool]] = []

    def forward(
        self,
        ids: Tensor,
        last: Tensor,
    ) -> Tensor:
        self.calls.append((self.training, torch.is_grad_enabled()))

        return last.float() * self.weight


def check_mean_reward(found: object) -> None:
    for words, answers in (
        (["cat", "string"], ["atcay", "tringsay"]),
        (["oak", "sky", "lip"], ["oakway", "yskay", "iplwayy"]),
    ):
        model = ScriptedReward()
        actual = found(model, words, answers, SMALL_POSITIONS.value)
        call = f"mean_reward of a reward model that gives each chat's last place, for {answers}"

        if len(model.calls) != 1:
            raise Mismatch(f"{call} ran the reward model {len(model.calls)} times, but one batch of every chat is enough.")

        training, recording = model.calls[0]

        if training:
            raise Mismatch(f"{call} ran the reward model in training mode. Call reward_model.eval() first.")

        if recording:
            raise Mismatch(f"{call} recorded for autograd. Score inside torch.no_grad().")

        if not model.training:
            raise Mismatch(f"{call} left the reward model in eval mode. Call reward_model.train() after.")

        _, last = POST_REFERENCE["reward_rows"](words, answers, SMALL_POSITIONS.value)
        expect_close(
            call=call,
            actual=actual,
            expected=last.float().mean().item(),
        )


def held_out_pair_accuracy(
    reward_model: nn.Module,
    count: int,
) -> dict[str, float]:
    """The share of held out pairs whose right answer the reward model scores above the slip, by kind of word."""
    characters = sorted(set(load_shakespeare()))
    positions = {character: place for place, character in enumerate(characters)}
    _, test_words = POST_REFERENCE["task_words"]()
    words = test_words[:count]
    chosen = POST_REFERENCE["reward_rows"](words, [POST_REFERENCE["pig_latin"](word) for word in words], positions)
    rejected = POST_REFERENCE["reward_rows"](words, [POST_REFERENCE["slip"](word) for word in words], positions)
    reward_model.eval()

    with torch.no_grad():
        wins = reward_model(*chosen) > reward_model(*rejected)

    reward_model.train()
    kinds = {"vowel": [], "one consonant": [], "several consonants": []}

    for word, won in zip(words, wins.tolist()):
        lead = POST_REFERENCE["leading_consonants"](word)
        kinds["vowel" if lead == 0 else "one consonant" if lead == 1 else "several consonants"].append(won)

    shares = {kind: sum(results) / len(results) for kind, results in kinds.items()}
    shares["all"] = wins.float().mean().item()

    return shares


def trained_reward_model() -> nn.Module:
    model = POST_REFERENCE["RewardModel"](POST_REFERENCE["GPT"](65, 128, 128, 4, 4))

    try:
        return load_checkpoint(model, "post3")
    except MissingCheckpoint as missing:
        raise Mismatch(str(missing)) from None
    except RuntimeError:
        raise Mismatch(
            "The saved model in checkpoints/post3.pt doesn't fit RewardModel as it is now. Train it again from the "
            "project folder.\n\n    .venv/bin/python run.py train post3",
        ) from None


def check_trained_post2(found: object) -> None:
    right, slipped = held_out_scores(reference_model("post2", "preferred_gpt"), 100)

    if right < POST2_BAR:
        raise Mismatch(
            f"Your model from post2 answers {right:.0%} of 100 held out words right, but it should reach "
            f"{POST2_BAR:.0%}. Check post2 and train it again.",
        )


def check_trained_reward(found: object) -> None:
    share = held_out_pair_accuracy(trained_reward_model(), 200)["all"]

    if share < REWARD_BAR:
        raise Mismatch(
            f"Your reward model from post3 ranks the right answer first in {share:.0%} of 200 held out pairs, but it "
            f"should reach {REWARD_BAR:.0%}. Check post3 and train it again.",
        )


def check_post3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="reward_rows",
        description="gives each whole chat padded, and its last newline's place",
        inputs=(
            (["cat", "string"], ["atcay", "tringsay"], SMALL_POSITIONS),
            (["oak", "sky", "lip"], ["oakway", "", "iplay"], SMALL_POSITIONS),
        ),
        reference=POST_REFERENCE["reward_rows"],
        expect=expect_whole_parts("the ids", "the last places"),
    )
    checker.test(
        function="mean_reward",
        description="averages the rewards in eval mode, without autograd",
        run=check_mean_reward,
    )
    checker.test(
        function="preferred_gpt",
        description="your post2 model answers more held out words right",
        run=check_trained_post2,
    )
    checker.test(
        function="mean_reward",
        description="your reward model ranks most held out pairs right",
        run=check_trained_reward,
    )


POST3 = Exercise(
    name="post3",
    chapter=POST_TRAINING,
    title="Project: scoring every stage",
    lesson=POST3_LESSON,
    source=POST3_SOURCE,
    solution=POST3_SOLUTION,
    hints={
        "reward_rows": "Build each text as prompt_text(word) + answer + a newline. torch.full makes the rows of "
        "newline ids, one per text, as long as the longest. Copy each text's ids into its row. The last places are "
        "each text's length minus 1.",
        "mean_reward": "reward_model.eval(), then inside torch.no_grad() call reward_model(*reward_rows(...)). Call "
        "reward_model.train(), then return the rewards' mean().item().",
        "preferred_gpt": "Your post2 model is read from checkpoints/post2.pt. Train post2 again if it scores low.",
    },
    introduces=(),
    check=check_post3,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons' tables
# ---------------------------------------------------------------------------


def task_setup() -> tuple[list[str], dict[str, int], list[str], list[str]]:
    characters = sorted(set(load_shakespeare()))
    positions = {character: place for place, character in enumerate(characters)}
    train_words, test_words = POST_REFERENCE["task_words"]()

    return characters, positions, train_words, test_words


def taught_reference(
    seed: int,
    device: torch.device,
    steps: int = 1000,
    every: int = 250,
) -> tuple[nn.Module, list[tuple[int, float, float]]]:
    """post1's main with seed in place of 0, and the held out scores every every steps."""
    characters, positions, train_words, test_words = task_setup()
    torch.manual_seed(seed)
    model = POST_REFERENCE["add_lora_everywhere"](POST_REFERENCE["trained_gpt"](), 4, 4.0).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    rows = []

    for step in range(steps):
        x, y = POST_REFERENCE["demonstration_batch"](train_words, positions, 64, generator)
        POST_REFERENCE["sft_step"](model, optimizer, x, y, device)

        if (step + 1) % every == 0:
            right, slipped = POST_REFERENCE["scores"](model, test_words[:200], positions, characters, device)
            rows.append((step + 1, right, slipped))
            print("post1", seed, rows[-1], flush=True)

    return model.cpu(), rows


def preferred_reference(
    taught_state: dict[str, Tensor],
    seed: int,
    device: torch.device,
    lr: float = 1e-3,
    nll: float = 0.2,
    steps: int = 300,
    every: int = 100,
) -> tuple[nn.Module, list[tuple[int, float, float]]]:
    """post2's main with seed + 1 in place of 1, from a post1 model's state, at the given rate and NLL weight."""
    characters, positions, train_words, test_words = task_setup()
    taught = POST_REFERENCE["add_lora_everywhere"](POST_REFERENCE["GPT"](65, 128, 128, 4, 4), 4, 4.0)
    taught.load_state_dict(taught_state)
    torch.manual_seed(seed)
    model = POST_REFERENCE["add_lora_everywhere"](POST_REFERENCE["merge_everywhere"](taught), 4, 4.0).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 1)
    rows = []

    for step in range(steps):
        chosen, rejected = POST_REFERENCE["pair_batch"](train_words, positions, 32, generator)
        POST_REFERENCE["dpo_step"](model, optimizer, chosen, rejected, 0.1, nll, device)

        if (step + 1) % every == 0:
            right, slipped = POST_REFERENCE["scores"](model, test_words[:200], positions, characters, device)
            rows.append((step + 1, right, slipped))
            print("post2", seed, lr, nll, rows[-1], flush=True)

    return model.cpu(), rows


def reward_reference(
    seed: int,
    device: torch.device,
    steps: int = 200,
) -> nn.Module:
    """post3's reward model training with seed + 2 in place of 2."""
    _, positions, train_words, _ = task_setup()
    torch.manual_seed(seed)
    reward_model = POST_REFERENCE["RewardModel"](POST_REFERENCE["trained_gpt"]()).to(device)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=3e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 2)

    for _ in range(steps):
        picked = [train_words[place] for place in torch.randint(0, len(train_words), (32,), generator=generator).tolist()]
        chosen = POST_REFERENCE["reward_rows"](picked, [POST_REFERENCE["pig_latin"](word) for word in picked], positions)
        rejected = POST_REFERENCE["reward_rows"](picked, [POST_REFERENCE["slip"](word) for word in picked], positions)
        POST_REFERENCE["reward_step"](
            reward_model,
            optimizer,
            tuple(part.to(device) for part in chosen),
            tuple(part.to(device) for part in rejected),
        )

    return reward_model.cpu()


def stage_rewards(
    model: nn.Module,
    reward_model: nn.Module,
) -> dict[str, float]:
    characters, positions, _, test_words = task_setup()
    words = test_words[:200]
    right, slipped = POST_REFERENCE["scores"](model, words, positions, characters, torch.device("cpu"))
    model.eval()
    answers = [POST_REFERENCE["answer"](model, word, positions, characters, torch.device("cpu")) for word in words]
    model.train()
    reward = POST_REFERENCE["mean_reward"](reward_model, words, answers, positions)

    return {"right": right, "slip": slipped, "reward": reward}


def measure_reward_by_kind() -> dict[str, dict[str, float]]:
    """For the models in checkpoints/, the run main makes, the reward model's mean score of each stage's answers to
    200 held out words, split into right ones, slips, and other mistakes it never saw in training."""
    characters, positions, _, test_words = task_setup()
    words = test_words[:200]
    reward_model = trained_reward_model()
    results = {}

    for name, loader in (("post1", "taught_gpt"), ("post2", "preferred_gpt")):
        model = POST_REFERENCE[loader]().eval()
        answers = [POST_REFERENCE["answer"](model, word, positions, characters, torch.device("cpu")) for word in words]
        reward_model.eval()

        with torch.no_grad():
            rewards = reward_model(*POST_REFERENCE["reward_rows"](words, answers, positions)).tolist()

        kinds: dict[str, list[float]] = {"right": [], "slip": [], "other": []}

        for word, written, reward in zip(words, answers, rewards):
            kind = "right" if written == POST_REFERENCE["pig_latin"](word) else "slip" if written == POST_REFERENCE["slip"](word) else "other"
            kinds[kind].append(reward)

        results[name] = {f"{kind} count": len(values) for kind, values in kinds.items()}
        results[name].update({f"{kind} reward": sum(values) / len(values) for kind, values in kinds.items() if values})
        print(name, results[name], flush=True)

    return results


def measure_cluster_share() -> float:
    """The share of the task's words led by several consonants, the ones the demonstrations got wrong."""
    train_words, test_words = POST_REFERENCE["task_words"]()
    words = train_words + test_words

    return sum(POST_REFERENCE["leading_consonants"](word) >= 2 for word in words) / len(words)


def measure_post_training(
    seeds: tuple[int, ...] = (0, 1, 2),
    device: torch.device | None = None,
) -> dict[str, object]:
    """Every number in the three lessons' tables, on this Mac's GPU. Seed 0 runs main's settings, though two runs on
    MPS can differ by a few points, as post2's 80% from main against 84.5% here."""
    device = device or best_device()
    results: dict[str, object] = {"cluster share": measure_cluster_share()}

    for seed in seeds:
        taught, rows = taught_reference(seed, device)
        results[f"post1 {seed}"] = rows
        state = copy.deepcopy(taught.state_dict())
        preferred = None

        for lr, nll in ((1e-3, 0.0), (1e-4, 0.0), (1e-3, 0.2)):
            model, rows = preferred_reference(state, seed, device, lr=lr, nll=nll)
            results[f"post2 {seed} {lr} {nll}"] = rows
            preferred = model

        reward_model = reward_reference(seed, device)
        results[f"pairs {seed}"] = held_out_pair_accuracy(reward_model, 500)
        results[f"post1 rewards {seed}"] = stage_rewards(taught, reward_model)
        results[f"post2 rewards {seed}"] = stage_rewards(preferred, reward_model)
        print(seed, results[f"pairs {seed}"], results[f"post1 rewards {seed}"], results[f"post2 rewards {seed}"], flush=True)

    return results
