"""78_sft, supervised fine tuning, a chat format with special tokens and a loss on the answer alone, then gpt3's
GPT taught to answer in it."""

from __future__ import annotations

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
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
)
from ..text import prose, source_file
from .gpt import GIVEN_MODEL, GPT_FORWARD, GPT_INIT, INIT_SOLUTION, MODEL_REFERENCE, trained_gpt

SFT = Chapter(
    folder="78_sft",
    title="Chat fine tuning",
    readme=source_file(
        """
        # Chat fine tuning

        A base model continues text. Supervised fine tuning, SFT, trains it
        further on questions paired with their answers, so it learns to answer.
        This chapter builds the chat format, with tokens that mark each turn,
        and a loss that scores the answer alone. Then it teaches gpt3's GPT to
        answer a made up question, so the result can be scored exactly.

        ## Further reading

        - [Ouyang and colleagues, training language models to follow instructions](https://arxiv.org/abs/2203.02155)
        - [Karpathy, nanochat](https://github.com/karpathy/nanochat)
        - [PyTorch, F.cross_entropy and its ignore_index](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)
        """,
    ),
)


def indented_part(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# The three tokens after gpt1's 65 characters, the vocabulary they make, and the made up task's words.
CHAT_PART = '''
USER = 65
ASSISTANT = 66
END = 67


# Every token's name, the special ones after the 65 characters, and the id of each character.
def chat_vocabulary() -> tuple[list[str], dict[str, int]]:
    characters = sorted(set(load_shakespeare()))
    positions = {character: position for position, character in enumerate(characters)}

    return characters + ["<user>", "<assistant>", "<end>"], positions


# Shakespeare's lowercase words of 3 to 8 letters, shuffled the same way every run, and 500 held out for tests.
def question_words() -> tuple[list[str], list[str]]:
    words = sorted({word for word in re.findall(r"\\b[a-z]+\\b", load_shakespeare()) if 3 <= len(word) <= 8})
    order = torch.randperm(len(words), generator=torch.Generator().manual_seed(0)).tolist()
    shuffled = [words[position] for position in order]

    return shuffled[500:], shuffled[:500]
'''

CHAT_SOLUTION = '''
def chat_ids(
    question: str,
    answer: str,
    positions: dict[str, int],
) -> list[int]:
    return [USER] + [positions[character] for character in question] + [ASSISTANT] + [positions[character] for character in answer] + [END]


def answer_targets(ids: list[int]) -> tuple[Tensor, Tensor]:
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    y[: ids.index(ASSISTANT)] = -100

    return x, y


def pad_rows(rows: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    longest = max(len(row_x) for row_x, _ in rows)
    x = torch.full((len(rows), longest), END)
    y = torch.full((len(rows), longest), -100)

    for row, (row_x, row_y) in enumerate(rows):
        x[row, : len(row_x)] = row_x
        y[row, : len(row_y)] = row_y

    return x, y


def answer_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    scores = model(x)

    return F.cross_entropy(scores.flatten(0, 1), y.flatten())
'''

# The made up task's pairs, a word and the word in capitals, turned into one padded batch.
BATCH_PART = '''
def word_pairs(
    words: list[str],
    count: int,
    generator: torch.Generator,
) -> list[tuple[str, str]]:
    picked = torch.randint(0, len(words), (count,), generator=generator).tolist()

    return [(words[position], words[position].upper()) for position in picked]


def chat_batch(
    pairs: list[tuple[str, str]],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    return pad_rows([answer_targets(chat_ids(question, answer, positions)) for question, answer in pairs])
'''

GROW_SOLUTION = '''
def grow_vocabulary(
    model: nn.Module,
    extra: int,
) -> None:
    rows = model.wte.weight.data
    grown = torch.cat([rows, rows.mean(dim=0).expand(extra, -1)])
    model.wte = nn.Embedding.from_pretrained(grown, freeze=False)
    model.lm_head = nn.Linear(grown.shape[1], grown.shape[0], bias=False)
    model.lm_head.weight = model.wte.weight
'''

TEACH_SOLUTION = '''
def sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
) -> float:
    loss = answer_loss(model, x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def answer(
    model: nn.Module,
    question: str,
    positions: dict[str, int],
    names: list[str],
) -> str:
    ids = torch.tensor([[USER] + [positions[character] for character in question] + [ASSISTANT]])
    written = ""

    with torch.no_grad():
        for _ in range(12):
            token = model(ids)[0, -1].argmax().item()

            if token == END:
                break

            written += names[token]
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)

    return written


def exact_match(
    model: nn.Module,
    words: list[str],
    positions: dict[str, int],
    names: list[str],
) -> float:
    model.eval()
    right = sum(answer(model, word, positions, names) == word.upper() for word in words)
    model.train()

    return right / len(words)
'''



def joined(*parts: str) -> str:
    """Parts of code one after another, two blank lines apart as in a file."""
    return "\n\n\n".join(part.strip("\n") for part in parts) + "\n"


SFT_SOLUTION = joined(GROW_SOLUTION, TEACH_SOLUTION)

# GPT_FORWARD is the class's last method, so it follows GPT_INIT with one blank line.
GPT_PARTS = joined(GIVEN_MODEL, INIT_SOLUTION, GPT_INIT.rstrip("\n") + "\n\n" + GPT_FORWARD.strip("\n"))

CHAT_PARTS = joined(CHAT_PART, CHAT_SOLUTION, BATCH_PART)


def sft_namespace() -> dict[str, object]:
    """The given parts and the solutions, run once, so the checks compare against exactly the code the files hold."""
    namespace: dict[str, object] = {
        "math": math,
        "re": re,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
    }
    exec(joined(GPT_PARTS, CHAT_PARTS, SFT_SOLUTION), namespace)

    return namespace


SFT_REFERENCE = sft_namespace()


def small_chat_gpt() -> nn.Module:
    torch.manual_seed(0)

    return MODEL_REFERENCE["GPT"](68, 24, 32, 2, 4)


# ---------------------------------------------------------------------------
# The runs behind the lessons' tables
# ---------------------------------------------------------------------------


def every_token_batch(
    pairs: list[tuple[str, str]],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    """chat_batch with every target scored, the question's too, for the comparison in sft1's table."""
    rows = []

    for question, answer in pairs:
        ids = SFT_REFERENCE["chat_ids"](question, answer, positions)
        rows.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:])))

    return SFT_REFERENCE["pad_rows"](rows)


def taught_gpt(
    steps: int,
    every_token: bool = False,
    pretrained: bool = True,
    seed: int = 0,
) -> nn.Module:
    """gpt3's GPT, or a fresh one, taught the made up task for steps steps of 32 pairs, as sft2's check does
    with seed 0."""
    torch.manual_seed(seed)
    model = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)

    if pretrained:
        model = trained_gpt(MODEL_REFERENCE["GPT"])

    SFT_REFERENCE["grow_vocabulary"](model, 3)
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, _ = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    make_batch = every_token_batch if every_token else SFT_REFERENCE["chat_batch"]

    for _ in range(steps):
        x, y = make_batch(SFT_REFERENCE["word_pairs"](train_words, 32, generator), positions)
        SFT_REFERENCE["sft_step"](model, optimizer, x, y)

    return model


def held_out_score(model: nn.Module) -> float:
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    _, test_words = SFT_REFERENCE["question_words"]()

    return SFT_REFERENCE["exact_match"](model, test_words[:200], positions, names)


def measure_sft() -> dict[str, float]:
    """Every number in sft1's and sft2's tables, the share of 200 held out words answered exactly, on the CPU,
    averaged over seeds 0, 1 and 2."""
    runs = {
        "answer scored": dict(),
        "every token scored": dict(every_token=True),
        "untrained GPT": dict(pretrained=False),
    }
    results: dict[str, float] = {}

    for name, settings in runs.items():
        for steps in (0, 100, 200, 300, 400):
            scores = [held_out_score(taught_gpt(steps, seed=seed, **settings)) for seed in (0, 1, 2)]
            results[f"{name}, {steps} steps"] = sum(scores) / len(scores)
            print(name, steps, [round(score, 3) for score in scores], flush=True)

    return results


# ---------------------------------------------------------------------------
# sft1
# ---------------------------------------------------------------------------

SFT1_LESSON = prose(
    """
    gpt3's GPT continues text. A chat model answers instead. The step between
    them is supervised fine tuning, SFT, which trains the same model further
    on questions paired with their answers.

    The model has to see where the question ends and the answer begins. Chat
    models mark each turn with tokens that never occur in plain text. Here
    there are three, after gpt1's 65 characters.

    - `USER`, 65: the user's turn starts.
    - `ASSISTANT`, 66: the answer starts.
    - `END`, 67: the answer is done, so the model can say it has finished.

    The task is made up, so every answer can be scored exactly. The question
    is a lowercase word from Shakespeare, and the answer is the word in
    capitals.

    ```text
    <user>romeo<assistant>ROMEO<end>
    ```

    Training reads the whole chat, but only the answer is scored. The user
    writes the question, so learning to predict it is wasted effort.
    `F.cross_entropy` skips every target of -100, its default
    `ignore_index`, as adder2 skipped padding.

    ```python
    F.cross_entropy(torch.zeros(2, 3), torch.tensor([2, -100]))    # tensor(1.0986), the second row skipped
    ```

    Chats differ in length, so a batch pads each one at the end. A GPT's
    position never reads a later one, so the padding changes no earlier
    score, and its targets are -100 too.

    Here is gpt3's GPT taught as sft2 teaches it, scored on 200 held out
    words. Each number is the share answered exactly, averaged over 3 seeds.

    | Steps of 32 chats | 100 | 200 | 300 | 400 |
    |---|---|---|---|---|
    | Answer scored | 8% | 74% | 90% | 95% |
    | Every token scored | 15% | 84% | 87% | 79% |

    Scoring every token led at 100 and 200 steps, by an amount within the
    spread between seeds. At 300 the two are close. At 400, scoring the
    answer alone reaches 95%, and scoring every token slips to 79%, with one
    seed down at 56%. Three seeds are few, but the answer alone was the
    steadier.
    """,
)

SFT1_SOURCE = source_file(
    '''
    # sft1.py

    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    targets = torch.tensor([5, 6, 7, 8])
    targets[:2] = -100                            # tensor([-100, -100,    7,    8])
    where = [3, 65, 9, 66].index(66)              # 3, the place of the first 66
    padded = torch.full((2, 3), 67)               # tensor([[67, 67, 67], [67, 67, 67]])


    # The three special tokens, the vocabulary and the made up task's words.
''' + indented_part(CHAT_PART) + '''


    # TODO: Return the ids of one chat. USER comes first, then the question's
    #   characters, ASSISTANT, the answer's characters, and END. positions maps
    #   each character to its id, as gpt1's did.
    def chat_ids(
        question: str,
        answer: str,
        positions: dict[str, int],
    ) -> list[int]:
        ...


    # TODO: Return the inputs and the targets of ids, as tensors. The inputs
    #   are every id but the last, and the targets every id but the first.
    #   Every target before the answer's first character becomes -100.
    def answer_targets(ids: list[int]) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Stack rows of inputs and targets into two tensors of shape (rows,
    #   longest). Pad the inputs at the end with END and the targets with -100.
    def pad_rows(rows: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the cross entropy of model's scores on x against y, as a
    #   tensor. Targets of -100 count for nothing.
    def answer_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...
    ''',
)

SFT1_SOLUTION = source_file(
    '''
    # sft1.py, a solution for comparison

    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    targets = torch.tensor([5, 6, 7, 8])
    targets[:2] = -100
    where = [3, 65, 9, 66].index(66)
    padded = torch.full((2, 3), 67)


''' + indented_part(CHAT_PART) + '''


''' + indented_part(CHAT_SOLUTION) + '''
    ''',
)

SMALL_POSITIONS = Described(
    value={"a": 1, "b": 2, "c": 3, "A": 4, "B": 5, "C": 6},
    text="positions of a, b, c, A, B and C",
)


def chat_rows(chats: list[tuple[str, str]]) -> list[tuple[Tensor, Tensor]]:
    return [SFT_REFERENCE["answer_targets"](SFT_REFERENCE["chat_ids"](question, answer, SMALL_POSITIONS.value)) for question, answer in chats]


def expect_id_pair(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    expect_parts("inputs", "targets")(
        call=call,
        actual=actual,
        expected=expected,
    )

    for name, part in zip(("inputs", "targets"), actual):
        if part.dtype.is_floating_point:
            raise Mismatch(
                f"The {name} from {call} hold {part.dtype} numbers, but ids and targets are whole numbers, which "
                "nn.Embedding and F.cross_entropy need. Build them from whole numbers, with torch.tensor, or "
                "torch.full with a whole number to fill.",
            )


def check_answer_loss(found: object) -> None:
    for chats in ((("ab", "AB"), ("cab", "CAB")), (("c", "C"), ("abba", "ABBA"), ("bc", "BC"))):
        x, y = SFT_REFERENCE["pad_rows"](chat_rows(list(chats)))
        model = small_chat_gpt()

        with torch.no_grad():
            expected = SFT_REFERENCE["answer_loss"](model, x, y)

        expect_tensor(
            call=f"answer_loss of a small GPT on {len(chats)} padded chats",
            actual=found(model, x, y),
            expected=expected,
        )


def check_sft1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="chat_ids",
        description="wraps the question and the answer in the special tokens",
        inputs=(
            ("ab", "AB", SMALL_POSITIONS),
            ("cab", "C", SMALL_POSITIONS),
        ),
        reference=SFT_REFERENCE["chat_ids"],
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="answer_targets",
        description="shifts by one and scores only the answer",
        inputs=(
            ([65, 1, 2, 66, 4, 5, 67],),
            ([65, 3, 1, 2, 3, 66, 6, 67],),
        ),
        reference=SFT_REFERENCE["answer_targets"],
        expect=expect_id_pair,
    )
    check_on_inputs(
        checker=checker,
        function="pad_rows",
        description="pads every row to the longest, with END and -100",
        inputs=(
            (Described(chat_rows([("ab", "AB"), ("c", "C")]), "the inputs and targets of the chats ab and c"),),
            (Described(chat_rows([("a", "A"), ("cab", "CAB"), ("bc", "BC")]), "the inputs and targets of the chats a, cab and bc"),),
        ),
        reference=SFT_REFERENCE["pad_rows"],
        expect=expect_id_pair,
    )
    checker.test(
        function="answer_loss",
        description="scores the answer's tokens and skips the rest",
        run=check_answer_loss,
    )


SFT1 = Exercise(
    name="sft1",
    chapter=SFT,
    title="The chat format",
    lesson=SFT1_LESSON,
    source=SFT1_SOURCE,
    solution=SFT1_SOLUTION,
    hints={
        "chat_ids": "Lists join with +. Start from [USER], add positions[character] for each character of question, "
        "then [ASSISTANT], the answer's ids the same way, and [END].",
        "answer_targets": "torch.tensor(ids[:-1]) and torch.tensor(ids[1:]). ids.index(ASSISTANT) is how many targets "
        "come before the answer, and assigning -100 to that many first targets masks them.",
        "pad_rows": "longest is the largest len of the inputs. Make one tensor of END and one of -100 with torch.full, "
        "then copy each row into the start of its line.",
        "answer_loss": "model(x) gives (rows, steps, tokens). Flatten it to one row of scores per prediction with "
        ".flatten(0, 1), flatten y to match, and F.cross_entropy skips the -100s on its own.",
    },
    introduces=(),
    check=check_sft1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# sft2
# ---------------------------------------------------------------------------

SFT2_LESSON = prose(
    """
    gpt3's GPT has rows for 65 tokens. The three new ones need rows of their
    own, in `wte` and in the head that shares its weight. So the table grows
    by three rows, and the head is tied to it again.

    A new row needs a start. Each new row here starts at the mean of the old
    ones. The head scores a token by its row, so a token at the mean scores
    the average of the others. It starts neither likely nor impossible.

    ```python
    rows = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    torch.cat([rows, rows.mean(dim=0).expand(1, -1)])    # tensor([[1., 2.], [3., 6.], [2., 4.]])
    ```

    `nn.Embedding.from_pretrained(rows, freeze=False)` makes a table that
    starts from given rows and keeps training.

    Then SFT is plain training on batches of chats, at a rate of 0.001 with
    AdamW. To answer, the model reads `USER`, the question and `ASSISTANT`,
    then writes its likeliest token, one at a time, until it writes `END`.

    Here is the share of 200 held out words answered exactly, by steps of 32
    chats, for gpt3's GPT and for the same GPT untrained. Each is the
    average of 3 seeds.

    | Steps | 0 | 100 | 200 | 300 | 400 |
    |---|---|---|---|---|---|
    | gpt3's GPT | 0% | 8% | 74% | 90% | 95% |
    | Untrained GPT | 0% | 36% | 68% | 73% | 87% |

    The untrained GPT starts faster, likely because gpt3's GPT first has to
    unlearn continuing the text as Shakespeare. From 200 steps on, gpt3's GPT
    is ahead. The check trains your gpt3 model for 400 steps and asks for 85%
    of 100 held out words.
    """,
)

SFT2_SOURCE = source_file(
    '''
    # sft2.py
    #
    # The last check needs your trained model from gpt3, in checkpoints/gpt3.pt.

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    rows = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    grown = torch.cat([rows, rows.mean(dim=0).expand(2, -1)])    # tensor([[1., 2.], [3., 6.], [2., 4.], [2., 4.]])
    table = nn.Embedding.from_pretrained(grown, freeze=False)    # an Embedding that starts from those rows


    # The GPT from gpt2.
''' + indented_part(GPT_PARTS) + '''


    # The chat format from sft1, and batches of the made up task.
''' + indented_part(CHAT_PARTS) + '''


    # TODO: Give model extra more tokens. Its wte grows by extra rows, each the
    #   mean of the old rows, and still trains. lm_head becomes a Linear with no
    #   bias whose weight is the grown table, tied as before.
    def grow_vocabulary(
        model: nn.Module,
        extra: int,
    ) -> None:
        ...


    # TODO: Take one training step on x and y with answer_loss. Zero the old
    #   gradients, backpropagate, step, and return the loss as a plain number.
    def sft_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Return the text model writes after USER, the question's ids and
    #   ASSISTANT, without recording for autograd. Each step adds the token
    #   with the top score at the last place. Stop at END, or after writing 12
    #   tokens. names[token] is each token's text.
    def answer(
        model: nn.Module,
        question: str,
        positions: dict[str, int],
        names: list[str],
    ) -> str:
        ...


    # TODO: Return the share of words whose answer is the word in capitals,
    #   as a plain number, with model in eval mode. Put it back in training
    #   mode after.
    def exact_match(
        model: nn.Module,
        words: list[str],
        positions: dict[str, int],
        names: list[str],
    ) -> float:
        ...
    ''',
)

SFT2_SOLUTION = source_file(
    '''
    # sft2.py, a solution for comparison

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    rows = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    grown = torch.cat([rows, rows.mean(dim=0).expand(2, -1)])
    table = nn.Embedding.from_pretrained(grown, freeze=False)


''' + indented_part(GPT_PARTS) + '''


''' + indented_part(CHAT_PARTS) + '''


''' + indented_part(SFT_SOLUTION) + '''
    ''',
)


def check_grow_vocabulary(found: object) -> None:
    for vocabulary, extra in ((20, 3), (12, 1)):
        torch.manual_seed(0)
        model = MODEL_REFERENCE["GPT"](vocabulary, 16, 32, 2, 4)
        old = model.wte.weight.detach().clone()
        found(model, extra)
        call = f"grow_vocabulary of a GPT with {vocabulary} tokens and extra {extra}"
        expected = torch.cat([old, old.mean(dim=0).expand(extra, -1)])

        expect_tensor(
            call=f"wte's weight after {call}",
            actual=model.wte.weight.detach(),
            expected=expected,
        )

        if not model.wte.weight.requires_grad:
            raise Mismatch(f"After {call}, wte's weight no longer trains. Pass freeze=False.")

        if model.lm_head.weight is not model.wte.weight:
            raise Mismatch(f"After {call}, lm_head's weight should be wte's grown weight itself, tied as gpt2 tied it.")

        if model.lm_head.bias is not None:
            raise Mismatch(f"After {call}, lm_head has a bias, but GPT's head has none.")

        scores = model(torch.zeros(1, 4, dtype=torch.long))

        if scores.shape[-1] != vocabulary + extra:
            raise Mismatch(f"After {call}, the model scores {scores.shape[-1]} tokens, but it should score {vocabulary + extra}.")


def chat_batches(seed: int) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    words = ["ab", "ba", "cab", "abc", "c", "bca"]
    batches = []

    for _ in range(2):
        picked = torch.randint(0, len(words), (4,), generator=generator).tolist()
        batches.append(SFT_REFERENCE["pad_rows"](chat_rows([(words[position], words[position].upper()) for position in picked])))

    return batches


def check_sft_step(found: object) -> None:
    for seed in (781, 782):
        batches = chat_batches(seed)
        model = small_chat_gpt()
        reference = small_chat_gpt()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)

        for step, (x, y) in enumerate(batches):
            expected = SFT_REFERENCE["sft_step"](reference, reference_optimizer, x, y)
            expect_close(
                call=f"sft_step on a small GPT, step {step + 1}",
                actual=found(model, optimizer, x, y),
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="wte's weight after two calls of sft_step",
            actual=model.wte.weight.detach(),
            expected=reference.wte.weight.detach(),
        )


class ScriptedWriter(nn.Module):
    """A stand in model that writes script, one token per call, and notes each call's ids and mode."""

    def __init__(
        self,
        prompt_length: int,
        script: list[int],
    ) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.script = script
        self.calls: list[tuple[list[int], bool]] = []
        # An answer that crops to the block, as gpt4's generate did, reads this.
        self.block_size = 128

    def forward(self, ids: Tensor) -> Tensor:
        self.calls.append((ids[0].tolist(), torch.is_grad_enabled()))
        written = ids.shape[1] - self.prompt_length
        scores = torch.zeros(ids.shape[0], ids.shape[1], 68)
        scores[:, -1, self.script[min(written, len(self.script) - 1)]] = 5.0

        return scores


def small_token_names() -> list[str]:
    """The names of a vocabulary of 68 tokens whose characters are SMALL_POSITIONS', with ? for the unused ones."""
    names = ["?"] * 65

    for character, position in SMALL_POSITIONS.value.items():
        names[position] = character

    return names + ["<user>", "<assistant>", "<end>"]


TOKEN_NAMES = small_token_names()


def check_answer(found: object) -> None:
    for question, script, text in (
        ("ab", [4, 5, 67], "A, then B, then END"),
        ("cab", [6, 66, 4, 67], "C, then ASSISTANT, then A, then END"),
        ("a", [5], "B forever"),
    ):
        prompt = [65] + [SMALL_POSITIONS.value[character] for character in question] + [66]
        model = ScriptedWriter(len(prompt), script)
        expected = SFT_REFERENCE["answer"](ScriptedWriter(len(prompt), script), question, SMALL_POSITIONS.value, TOKEN_NAMES)
        actual = found(model, question, SMALL_POSITIONS.value, TOKEN_NAMES)
        call = f"answer of a model that writes {text}, to {question!r}"

        if not model.calls:
            raise Mismatch(f"{call} never ran the model.")

        first_ids, recording = model.calls[0]

        if first_ids != prompt:
            raise Mismatch(f"{call} first ran the model on {first_ids}, but it should read USER, the question and ASSISTANT, {prompt}.")

        if recording:
            raise Mismatch(f"{call} ran the model while recording for autograd. Write inside torch.no_grad().")

        expect_value(
            call=call,
            actual=actual,
            expected=expected,
        )


def check_exact_match(found: object) -> None:
    for words, written in (
        (["ab", "cab", "bc", "a"], ["AB", "CAB", "BCC", "a"]),
        (["abc", "ba", "c"], ["ABC", "BA", "C"]),
    ):
        model = small_chat_gpt()
        modes: list[bool] = []
        replies = dict(zip(words, written))

        def scripted_answer(
            model: nn.Module,
            question: str,
            positions: dict[str, int],
            names: list[str],
        ) -> str:
            modes.append(model.training)

            return replies[question]

        namespace = found.__globals__
        original = namespace.get("answer")
        namespace["answer"] = scripted_answer

        try:
            actual = found(model, words, SMALL_POSITIONS.value, TOKEN_NAMES)
        finally:
            namespace["answer"] = original

        if len(modes) != len(words):
            raise Mismatch(f"exact_match should call answer once for each of {len(words)} words, but it called it {len(modes)} times.")

        if any(modes):
            raise Mismatch("exact_match ran answer with the model in training mode. Call model.eval() first.")

        if not model.training:
            raise Mismatch("exact_match left the model in eval mode. Call model.train() after.")

        right = sum(reply == word.upper() for word, reply in replies.items())
        expect_close(
            call=f"exact_match when answer writes {written} for {words}",
            actual=actual,
            expected=right / len(words),
        )


def check_taught_gpt(found: object) -> None:
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    SFT_REFERENCE["grow_vocabulary"](model, 3)
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, test_words = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)

    for _ in range(400):
        x, y = SFT_REFERENCE["chat_batch"](SFT_REFERENCE["word_pairs"](train_words, 32, generator), positions)
        found(model, optimizer, x, y)

    score = SFT_REFERENCE["exact_match"](model, test_words[:100], positions, names)

    if score < 0.85:
        raise Mismatch(f"After 400 steps of sft_step, your gpt3 model answers {score:.0%} of 100 held out words exactly, but it should reach 85%.")


def check_sft2(checker: Checker) -> None:
    checker.test(
        function="grow_vocabulary",
        description="adds rows at the mean of the old ones, the head tied",
        run=check_grow_vocabulary,
    )
    checker.test(
        function="sft_step",
        description="zeroes, backpropagates and steps on the answer loss",
        run=check_sft_step,
    )
    checker.test(
        function="answer",
        description="writes the likeliest token until END, at most 12",
        run=check_answer,
    )
    checker.test(
        function="exact_match",
        description="gives the share answered in capitals, in eval mode",
        run=check_exact_match,
    )
    checker.test(
        function="sft_step",
        description="teaches your gpt3 model to answer 85% of held out words",
        run=check_taught_gpt,
    )


SFT2 = Exercise(
    name="sft2",
    chapter=SFT,
    title="Teaching the GPT to answer",
    lesson=SFT2_LESSON,
    source=SFT2_SOURCE,
    solution=SFT2_SOLUTION,
    hints={
        "grow_vocabulary": "torch.cat the old rows, model.wte.weight.data, with their mean over dim=0 expanded to extra "
        "rows. Make model.wte from that with nn.Embedding.from_pretrained and freeze=False, then a new lm_head whose "
        "weight you set to model.wte.weight.",
        "sft_step": "answer_loss(model, x, y), then optimizer.zero_grad(), loss.backward(), optimizer.step(), and "
        "return loss.item().",
        "answer": "Start ids as a tensor of one row. Inside torch.no_grad(), loop 12 times: argmax of model(ids)[0, -1], "
        "break at END, else add names[token] to the text and torch.cat the token onto ids.",
        "exact_match": "model.eval(), count the words where answer(...) equals word.upper(), model.train(), then divide "
        "by len(words).",
    },
    introduces=(),
    check=check_sft2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)
