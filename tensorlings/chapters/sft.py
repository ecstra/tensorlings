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

        A base model, like the GPT you trained in gpt3, only continues text.
        Supervised fine tuning, SFT, trains it further on questions paired with
        their answers, so that it learns to answer questions. This chapter
        builds the chat format, which uses special tokens to mark where each
        turn starts and ends. It also builds a loss that counts only the
        answer's tokens. Then it trains gpt3's GPT on a made up task that has
        one right answer per question, so every answer can be scored exactly.

        ## Further reading

        - [Ouyang and colleagues, training language models to follow instructions](https://arxiv.org/abs/2203.02155)
        - [Karpathy, nanochat](https://github.com/karpathy/nanochat)
        - [Huerta-Enochian and Ko, instruction fine tuning, does prompt loss matter?](https://arxiv.org/abs/2401.13586)
        - [Shi and colleagues, instruction tuning with loss over instructions](https://arxiv.org/abs/2405.14394)
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


# Returns the name of every token, with the three special tokens after the 65 characters, and each character's id.
def chat_vocabulary() -> tuple[list[str], dict[str, int]]:
    characters = sorted(set(load_shakespeare()))
    positions = {character: position for position, character in enumerate(characters)}

    return characters + ["<user>", "<assistant>", "<end>"], positions


# Shakespeare's lowercase words of 3 to 8 letters, shuffled the same way on every run. Returns the training words
# and 500 held out words for testing.
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


def held_out_score(model: nn.Module) -> float:
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    _, test_words = SFT_REFERENCE["question_words"]()

    return SFT_REFERENCE["exact_match"](model, test_words[:200], positions, names)


def learning_curve(
    lr: float,
    every_token: bool = False,
    pretrained: bool = True,
    seed: int = 0,
) -> list[float]:
    """gpt3's GPT, or a fresh one, taught the made up task in steps of 32 pairs at rate lr, as sft2's check does
    with seed 0. Gives the share of 200 held out words answered exactly after 100, 200, 300 and 400 steps."""
    torch.manual_seed(seed)
    model = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)

    if pretrained:
        model = trained_gpt(MODEL_REFERENCE["GPT"])

    SFT_REFERENCE["grow_vocabulary"](model, 3)
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, _ = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    make_batch = every_token_batch if every_token else SFT_REFERENCE["chat_batch"]
    shares = []

    for step in range(1, 401):
        x, y = make_batch(SFT_REFERENCE["word_pairs"](train_words, 32, generator), positions)
        SFT_REFERENCE["sft_step"](model, optimizer, x, y)

        if step % 100 == 0:
            shares.append(held_out_score(model))

    return shares


def measure_sft(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, list[list[float]]]:
    """Every number in sft1's and sft2's tables, and the rates sft2 tried, on the CPU with 2 threads. Each run gives
    the share of 200 held out words answered exactly after 100 to 400 steps, one list per seed."""
    runs = {
        "answer scored, 0.0003": dict(lr=3e-4),
        "every token scored, 0.0003": dict(lr=3e-4, every_token=True),
        "gpt3's GPT, 0.0001": dict(lr=1e-4),
        "gpt3's GPT, 0.001": dict(lr=1e-3),
        "untrained, 0.0001": dict(lr=1e-4, pretrained=False),
        "untrained, 0.0003": dict(lr=3e-4, pretrained=False),
        "untrained, 0.001": dict(lr=1e-3, pretrained=False),
        "untrained, 0.003": dict(lr=3e-3, pretrained=False),
    }
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    results: dict[str, list[list[float]]] = {}

    try:
        for name, settings in runs.items():
            results[name] = [learning_curve(seed=seed, **settings) for seed in seeds]
            print(name, [[round(share, 3) for share in curve] for curve in results[name]], flush=True)
    finally:
        torch.set_num_threads(threads)

    return results


# ---------------------------------------------------------------------------
# sft1
# ---------------------------------------------------------------------------

SFT1_LESSON = prose(
    """
    gpt3's GPT continues whatever text it is given. A chat model answers
    questions instead. The training that turns one into the other is called
    supervised fine tuning, SFT. It trains the same model further on
    questions paired with their answers.

    The model needs a way to see where the question ends and the answer
    begins. Chat models mark each turn with special tokens that never appear
    in ordinary text. This chapter uses three of them. They get the ids 65,
    66 and 67, right after the ids 0 to 64 of gpt1's 65 characters.

    - `USER`, 65: marks the start of the user's question.
    - `ASSISTANT`, 66: marks the start of the answer.
    - `END`, 67: marks the end of the answer. The model writes it to say it
      has finished.

    The task is made up so that every answer is either exactly right or
    wrong. The question is a lowercase word from Shakespeare, and the answer
    is the same word in capitals.

    ```text
    <user>romeo<assistant>ROMEO<end>
    ```

    During training the model reads the whole chat, but the loss counts only
    the answer's tokens. The user writes the question, so the model never
    needs to learn to write questions. It only has to write answers.

    To leave a target out of the loss, set it to -100. `F.cross_entropy`
    skips every target equal to its `ignore_index` argument, which is -100
    by default. adder2 used the same trick to skip padding.

    ```python
    F.cross_entropy(torch.zeros(2, 3), torch.tensor([2, -100]))    # tensor(1.0986), the second row skipped
    ```

    Chats have different lengths, so a batch pads each chat at the end to
    the length of the longest one. In a GPT, each position reads only
    earlier positions, never later ones. So padding at the end cannot change
    the scores at any earlier position. The padding's targets are set to
    -100 as well, so they add nothing to the loss.

    The table below trains gpt3's GPT the way sft2 does, then tests it on
    200 held out words that never appear in training. Each number is the
    percentage of those words answered exactly right, averaged over 3 seeds.

    | Steps of 32 chats | 100 | 200 | 300 | 400 |
    |---|---|---|---|---|
    | Loss on the answer only | 54% | 95% | 97% | 98% |
    | Loss on every token | 19% | 87% | 95% | 97% |

    Counting only the answer in the loss learns faster. It was ahead on
    every seed at 100 and 200 steps. By 400 steps the two are within two
    percentage points of each other.

    Huerta-Enochian and Ko multiplied the loss on the question's tokens by a
    weight between 0 and 1. Their models were trained on short answers. On
    tests with short answers, they did best with a small weight, from 0.01
    to 0.5. Shi and colleagues found that including the question in the loss
    helps in one case. That is when the questions are long, the answers are
    short and there are few training examples.
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
    where = [3, 65, 9, 66].index(66)              # 3, the index of the first 66
    padded = torch.full((2, 3), 67)               # tensor([[67, 67, 67], [67, 67, 67]])


    # The three special tokens, the vocabulary, and the words for the made up task.
''' + indented_part(CHAT_PART) + '''


    # TODO: Return the list of token ids for one chat. It starts with USER,
    #   then has the id of each character of question, then ASSISTANT, then
    #   the id of each character of answer, and ends with END. positions maps
    #   each character to its id, as in gpt1.
    #   Example: chat_ids("ab", "AB", {"a": 1, "b": 2, "A": 4, "B": 5})
    #   returns [65, 1, 2, 66, 4, 5, 67].
    def chat_ids(
        question: str,
        answer: str,
        positions: dict[str, int],
    ) -> list[int]:
        ...


    # TODO: Return the inputs and the targets for ids, as two tensors. The
    #   inputs are every id except the last, and the targets are every id
    #   except the first, so target i is the token that follows input i. Then
    #   set every target before the answer's first character to -100, so the
    #   loss counts only the answer and END.
    #   Example: answer_targets([65, 1, 2, 66, 4, 5, 67]) returns
    #   (tensor([65, 1, 2, 66, 4, 5]), tensor([-100, -100, -100, 4, 5, 67])).
    def answer_targets(ids: list[int]) -> tuple[Tensor, Tensor]:
        ...


    # TODO: rows is a list of (inputs, targets) pairs of different lengths.
    #   Return two tensors of shape (len(rows), longest), where longest is the
    #   length of the longest inputs. Copy each pair into the start of its
    #   row, then fill the rest of the inputs with END and the rest of the
    #   targets with -100.
    #   Example: for the pairs (tensor([1, 2]), tensor([2, 3])) and
    #   (tensor([5]), tensor([6])), pad_rows returns
    #   (tensor([[1, 2], [5, 67]]), tensor([[2, 3], [6, -100]])).
    def pad_rows(rows: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the cross entropy loss of model on inputs x and targets y,
    #   as a tensor. x and y have shape (rows, steps), and model(x) returns
    #   scores of shape (rows, steps, tokens). Targets of -100 must not count
    #   toward the loss.
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
                f"The {name} from {call} hold {part.dtype} numbers, but ids and targets must be integers, because "
                "nn.Embedding and F.cross_entropy need integers. Build them from integers, with torch.tensor, or with "
                "torch.full and an integer fill value.",
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
        description="shifts the ids by one and masks every target before the answer",
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
        description="counts the answer's targets in the loss and skips the -100s",
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
        "chat_ids": "Join lists with +. Start from [USER] and add a list of positions[character] for each character of "
        "question. Then add [ASSISTANT], the answer's ids built the same way, and [END].",
        "answer_targets": "The inputs are torch.tensor(ids[:-1]) and the targets are torch.tensor(ids[1:]). "
        "ids.index(ASSISTANT) is the number of targets that come before the answer. Assign -100 to that many targets "
        "at the start, with a slice.",
        "pad_rows": "longest is the largest len among the inputs. Make one tensor full of END and one full of -100 with "
        "torch.full. Then copy each pair's inputs and targets into the start of its row.",
        "answer_loss": "model(x) returns scores of shape (rows, steps, tokens). Use .flatten(0, 1) to turn them into "
        "one row of scores per prediction, and flatten y to match. F.cross_entropy skips the targets of -100 by itself.",
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
    gpt3's GPT has an embedding table, `wte`, with one row for each of its 65
    tokens. Its output head, `lm_head`, shares that same weight, as gpt2 set
    up. The three new tokens need rows of their own. So `wte` grows by three
    rows, and the head is tied to the grown table again.

    Each new row needs a starting value. Here each new row starts as the
    mean of the old rows. The head scores each token with that token's row.
    So a new token whose row is the mean of the others gets the average of
    their scores. That means the new token starts out neither likely nor
    impossible.

    ```python
    rows = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    torch.cat([rows, rows.mean(dim=0).expand(1, -1)])    # tensor([[1., 2.], [3., 6.], [2., 4.]])
    ```

    `nn.Embedding.from_pretrained(rows, freeze=False)` makes an embedding
    table whose rows start as the given `rows`. `freeze=False` means
    training still updates them.

    After that, SFT is ordinary training with AdamW on batches of chats. A
    model that is already trained needs a small learning rate, as finetune1
    found, so the rate here is 0.0003. To answer a question, the model reads
    `USER`, the question and `ASSISTANT`. Then it writes one token at a
    time, each time picking the token with the highest score, until it
    writes `END`.

    The table shows the percentage of 200 held out words answered exactly,
    after each number of steps of 32 chats. It compares gpt3's GPT with the
    same GPT untrained. Both were tried at learning rates 0.0001, 0.0003 and
    0.001, and the untrained one also at 0.003. For both models, 0.0003 was
    the best rate, or within half a point of the best, at every step. The
    table shows 0.0003, averaged over 3 seeds.

    | Steps | 0 | 100 | 200 | 300 | 400 |
    |---|---|---|---|---|---|
    | gpt3's GPT | 0% | 54% | 95% | 97% | 98% |
    | Untrained GPT | 0% | 37% | 80% | 92% | 96% |

    gpt3's GPT learns faster. It was ahead on every seed at 200 and 300
    steps. By 400 steps the untrained one has nearly caught up. The check
    trains your model from gpt3 for 400 steps. Then it requires the model
    to answer at least 85% of 100 held out words exactly.
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
    table = nn.Embedding.from_pretrained(grown, freeze=False)    # an Embedding whose rows start as grown, and train


    # The GPT from gpt2.
''' + indented_part(GPT_PARTS) + '''


    # The chat format from sft1, and batches of the made up task.
''' + indented_part(CHAT_PARTS) + '''


    # TODO: Add extra new tokens to model, changing it in place. Replace
    #   model.wte with an embedding whose rows are the old rows followed by
    #   extra new rows, each equal to the mean of the old rows. The new wte
    #   must still train. Then replace model.lm_head with a Linear layer with
    #   no bias, and tie its weight to the grown wte weight, as gpt2 did.
    #   Example: after grow_vocabulary(model, 3) on a GPT with 65 tokens, the
    #   model scores 68 tokens.
    def grow_vocabulary(
        model: nn.Module,
        extra: int,
    ) -> None:
        ...


    # TODO: Take one training step on inputs x and targets y. Compute
    #   answer_loss, zero the old gradients, backpropagate, call the
    #   optimizer's step, and return the loss as a Python float.
    def sft_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...


    # TODO: Return the answer model writes to question, as a string. Start
    #   from the ids of USER, the question's characters and ASSISTANT, as a
    #   tensor with one row. At each step run the model, take the token with
    #   the highest score at the last position, and append it to the ids.
    #   Stop when that token is END, which is not added to the text, or after
    #   12 steps. names[token] is the text of each token. Run the model inside
    #   torch.no_grad(), so autograd records nothing.
    def answer(
        model: nn.Module,
        question: str,
        positions: dict[str, int],
        names: list[str],
    ) -> str:
        ...


    # TODO: Return the fraction of words for which answer returns the word in
    #   capitals, as a Python float. Put model in eval mode before answering,
    #   and back in training mode afterwards.
    #   Example: if 3 of 4 words are answered right, exact_match returns 0.75.
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
            raise Mismatch(f"After {call}, lm_head's weight should be the same tensor as wte's grown weight, tied as in gpt2.")

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
            raise Mismatch(f"{call} first ran the model on {first_ids}, but it should first read USER, the question and ASSISTANT, which are {prompt}.")

        if recording:
            raise Mismatch(f"{call} ran the model while autograd was recording. Run the model inside torch.no_grad().")

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
            raise Mismatch("exact_match left the model in eval mode. Call model.train() afterwards.")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
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
        description="adds rows equal to the mean of the old rows and ties the head",
        run=check_grow_vocabulary,
    )
    checker.test(
        function="sft_step",
        description="zeroes the gradients, backpropagates the answer loss and steps",
        run=check_sft_step,
    )
    checker.test(
        function="answer",
        description="writes the highest scoring token until END, at most 12 tokens",
        run=check_answer,
    )
    checker.test(
        function="exact_match",
        description="returns the fraction answered in capitals, in eval mode",
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
        "grow_vocabulary": "The old rows are model.wte.weight.data. Join them with torch.cat to their mean over dim=0, "
        "expanded to extra rows. Build model.wte from the result with nn.Embedding.from_pretrained and freeze=False. "
        "Then make a new lm_head and set its weight to model.wte.weight.",
        "sft_step": "Compute answer_loss(model, x, y), then clear the old gradients with the optimizer's zero_grad. Call "
        "loss.backward() and optimizer.step(), and return loss.item().",
        "answer": "Start ids as a tensor with one row. Inside torch.no_grad(), loop 12 times. Each time, take the argmax "
        "of model(ids)[0, -1] and break if it is END. Otherwise add names[token] to the text and torch.cat the token "
        "onto ids.",
        "exact_match": "Call model.eval(), count the words where answer(...) equals word.upper(), call model.train(), "
        "then divide the count by len(words).",
    },
    introduces=(),
    check=check_sft2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)
