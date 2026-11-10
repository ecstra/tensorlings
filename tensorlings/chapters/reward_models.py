"""81_reward_models, a reward model trained on preference pairs with the Bradley Terry loss, then the KL penalty
that keeps a policy near its reference, and reward hacking on a toy."""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
)
from ..text import prose, source_file
from .gpt import MODEL_REFERENCE, trained_gpt
from .sft import CHAT_PARTS, GPT_PARTS, GROW_SOLUTION, indented_part, joined

REWARD_MODELS = Chapter(
    folder="81_reward_models",
    title="Reward models",
    readme=source_file(
        """
        # Reward models

        People find it easier to pick the better of two answers than to write
        the best answer themselves. A reward model learns from such picks to
        give any answer a score. It trains with the Bradley Terry loss.
        Reinforcement learning from human feedback, RLHF, then trains the chat
        model to get high scores. A penalty stops the chat model from drifting
        too far from where it started. This chapter trains a reward model on
        your GPT, then uses a toy example to show why the penalty is needed.

        ## Further reading

        - [Ouyang and colleagues, training language models to follow instructions](https://arxiv.org/abs/2203.02155)
        - [Christiano and colleagues, deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
        - [Rafailov and colleagues, direct preference optimization, section 4](https://arxiv.org/abs/2305.18290)
        """,
    ),
)

# A right answer and a near miss for each question, and the chats of a batch with the place of each one's END.
PAIRS_PART = '''
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Returns the word in capitals with one mistake, which is a wrong letter, a letter left out, or a letter written twice.
def near_miss(
    word: str,
    generator: torch.Generator,
) -> str:
    right = word.upper()
    spot = torch.randint(0, len(right), (1,), generator=generator).item()
    slip = torch.randint(0, 3, (1,), generator=generator).item()

    if slip == 0:
        others = LETTERS.replace(right[spot], "")

        return right[:spot] + others[torch.randint(0, 25, (1,), generator=generator).item()] + right[spot + 1 :]

    if slip == 1:
        return right[:spot] + right[spot + 1 :]

    return right[:spot] + right[spot] + right[spot:]


# Returns count questions, each with a chosen answer, the word in capitals, and a rejected answer, a near miss.
def preference_pairs(
    words: list[str],
    count: int,
    generator: torch.Generator,
) -> list[tuple[str, str, str]]:
    picked = torch.randint(0, len(words), (count,), generator=generator).tolist()

    return [(words[position], words[position].upper(), near_miss(words[position], generator)) for position in picked]


# Returns the ids of the chats, padded at the end with END, and the index of each chat's own END token.
def scored_batch(
    chats: list[tuple[str, str]],
    positions: dict[str, int],
) -> tuple[Tensor, Tensor]:
    rows = [chat_ids(question, answer, positions) for question, answer in chats]
    ids = torch.full((len(rows), max(len(row) for row in rows)), END)

    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row)

    return ids, torch.tensor([len(row) - 1 for row in rows])
'''

REWARD_INIT = '''
class RewardModel(nn.Module):
    def __init__(self, gpt: nn.Module) -> None:
        super().__init__()
        self.gpt = gpt
        self.score = nn.Linear(gpt.wte.weight.shape[1], 1)
'''

REWARD_FORWARD = '''
    def forward(
        self,
        ids: Tensor,
        last: Tensor,
    ) -> Tensor:
        x = self.gpt.wte(ids) + self.gpt.wpe(torch.arange(ids.shape[1], device=ids.device))

        for block in self.gpt.h:
            x = block(x)

        hidden = self.gpt.ln_f(x)[torch.arange(len(ids), device=ids.device), last]

        return self.score(hidden).squeeze(-1)
'''

REWARD_SOLUTION = '''
def bradley_terry_loss(
    chosen: Tensor,
    rejected: Tensor,
) -> Tensor:
    return -F.logsigmoid(chosen - rejected).mean()


def pair_accuracy(
    chosen: Tensor,
    rejected: Tensor,
) -> float:
    return (chosen > rejected).float().mean().item()


def reward_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen: tuple[Tensor, Tensor],
    rejected: tuple[Tensor, Tensor],
) -> float:
    loss = bradley_terry_loss(model(*chosen), model(*rejected))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

REWARD_MODEL_PART = REWARD_INIT.rstrip("\n") + "\n\n" + REWARD_FORWARD.strip("\n") + "\n"


def reward_namespace() -> dict[str, object]:
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
    exec(joined(GPT_PARTS, CHAT_PARTS, GROW_SOLUTION, PAIRS_PART, REWARD_MODEL_PART, REWARD_SOLUTION), namespace)

    return namespace


REWARD_REFERENCE = reward_namespace()


def pair_batches(
    pairs: list[tuple[str, str, str]],
    positions: dict[str, int],
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    chosen = REWARD_REFERENCE["scored_batch"]([(question, right) for question, right, _ in pairs], positions)
    rejected = REWARD_REFERENCE["scored_batch"]([(question, wrong) for question, _, wrong in pairs], positions)

    return chosen, rejected


def trained_reward_model(
    steps: int,
    pretrained: bool = True,
    seed: int = 0,
    lr: float = 3e-4,
) -> nn.Module:
    """A reward model on gpt3's GPT, or on a fresh one, trained on steps batches of 32 pairs."""
    torch.manual_seed(seed)
    gpt = MODEL_REFERENCE["GPT"](65, 128, 128, 4, 4)

    if pretrained:
        gpt = trained_gpt(MODEL_REFERENCE["GPT"])

    REWARD_REFERENCE["grow_vocabulary"](gpt, 3)
    torch.manual_seed(seed)
    model = REWARD_REFERENCE["RewardModel"](gpt)
    _, positions = REWARD_REFERENCE["chat_vocabulary"]()
    train_words, _ = REWARD_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        chosen, rejected = pair_batches(REWARD_REFERENCE["preference_pairs"](train_words, 32, generator), positions)
        REWARD_REFERENCE["reward_step"](model, optimizer, chosen, rejected)

    return model


def held_out_pairs(count: int = 500) -> list[tuple[str, str, str]]:
    _, test_words = REWARD_REFERENCE["question_words"]()

    return REWARD_REFERENCE["preference_pairs"](test_words, count, torch.Generator().manual_seed(812))


def held_out_accuracy(
    model: nn.Module,
    pairs: list[tuple[str, str, str]],
) -> float:
    _, positions = REWARD_REFERENCE["chat_vocabulary"]()
    chosen, rejected = pair_batches(pairs, positions)
    model.eval()

    with torch.no_grad():
        accuracy = REWARD_REFERENCE["pair_accuracy"](model(*chosen), model(*rejected))

    model.train()

    return accuracy


def slip_of(
    right: str,
    wrong: str,
) -> str:
    if len(wrong) < len(right):
        return "letter left out"

    if len(wrong) > len(right):
        return "letter doubled"

    return "wrong letter"


def measure_reward_models() -> dict[str, float]:
    """Every number in reward1's table, the share of 500 held out pairs ranked right, for each kind of slip, averaged
    over seeds 0 to 2, on 4 threads, since the thread count moves how sums round."""
    pairs = held_out_pairs()
    kinds = ("wrong letter", "letter left out", "letter doubled")
    results: dict[str, float] = {}
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        for name, pretrained in (("gpt3's GPT", True), ("untrained GPT", False)):
            for steps in (0, 50, 200, 1000):
                totals = dict.fromkeys(kinds + ("all",), 0.0)

                for seed in (0, 1, 2):
                    model = trained_reward_model(steps, pretrained=pretrained, seed=seed)
                    totals["all"] += held_out_accuracy(model, pairs) / 3

                    for kind in kinds:
                        some = [pair for pair in pairs if slip_of(pair[1], pair[2]) == kind]
                        totals[kind] += held_out_accuracy(model, some) / 3

                for kind, total in totals.items():
                    results[f"{name}, {steps} steps, {kind}"] = total
                    print(name, steps, kind, round(total, 3), flush=True)
    finally:
        torch.set_num_threads(threads)

    return results


# ---------------------------------------------------------------------------
# reward1
# ---------------------------------------------------------------------------

REWARD1_LESSON = prose(
    """
    People find it easier to pick the better of two answers than to write
    the best answer themselves. InstructGPT is GPT-3 trained to follow
    instructions, by Ouyang and colleagues at OpenAI. Its labelers ranked 4
    to 9 answers to each prompt. Every pair of those answers then gave one
    comparison. A reward model learns from such comparisons to give any
    answer a score.

    The Bradley Terry model says how scores turn into choices. The
    probability that answer a is preferred over answer b is the sigmoid of
    the difference between their scores.

    ```text
    P(a beats b) = sigmoid(r(a) - r(b))
    ```

    Only the difference matters, so adding the same number to every score
    changes nothing. The loss is minus the log of that probability for the
    answer that was preferred, as in InstructGPT. `F.logsigmoid` computes
    the log of the sigmoid in one step, as in gan1. That way a sigmoid that
    rounds to 0 never turns the log into minus infinity.

    ```python
    F.logsigmoid(torch.tensor([2.0, -200.0]))    # tensor([-1.2693e-01, -2.0000e+02]), the -200 kept exact
    ```

    The reward model is the GPT itself with a new head. InstructGPT started
    from its SFT model, removed the last layer, and added a layer that
    outputs one number. Here a Linear layer reads the GPT's output at the
    position of the chat's END token. That position has seen the whole
    question and the whole answer.

    Chats in a batch have different lengths, so `last` holds the index of
    each chat's END. Indexing with two tensors picks one position per row.

    ```python
    h = torch.arange(12.0).view(2, 3, 2)
    h[torch.arange(2), torch.tensor([2, 0])]    # tensor([[4., 5.], [6., 7.]]), position 2 of row 0, position 0 of row 1
    ```

    Each pair holds a right answer and a near miss, which has one letter
    wrong, left out or doubled. The table shows the percentage of 500 held
    out pairs ranked correctly, after each number of steps of 32 pairs. It
    averages 3 seeds.

    | Mistake | 0 steps | 50 steps | 200 steps | 1,000 steps |
    |---|---|---|---|---|
    | A letter left out | 44% | 100% | 100% | 99.8% |
    | A letter doubled | 60% | 100% | 100% | 100% |
    | A wrong letter | 50% | 79% | 78% | 90% |
    | All | 51% | 93% | 93% | 97% |

    Two kinds of mistake change the length of the answer. By 50 steps the
    model ranks every one of those pairs correctly. It learned the easy cue
    first, whether the answer's length matches the question's. A wrong
    letter keeps the length the same, so the model has to read the letters,
    and that takes longer. Between 50 and 200 steps it stays near 78%, then
    rises to 90% by 1,000 steps. reward2 shows what a policy does with
    whatever mistakes a reward model still makes.

    An untrained GPT learns the length cue as quickly, reaching 92% overall
    by 50 steps. Pretraining helps with reading the letters. On wrong
    letters the untrained GPT gets 74%, 73% and 85% at 50, 200 and 1,000
    steps, while gpt3's GPT gets 79%, 78% and 90%.
    """,
)

REWARD1_SOURCE = source_file(
    '''
    # reward1.py

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    logs = F.logsigmoid(torch.tensor([2.0, -200.0]))         # tensor([-1.2693e-01, -2.0000e+02])
    h = torch.arange(12.0).view(2, 3, 2)
    picked = h[torch.arange(2), torch.tensor([2, 0])]        # tensor([[4., 5.], [6., 7.]])


    # The GPT from gpt2.
''' + indented_part(GPT_PARTS) + '''


    # The chat format from sft1, and grow_vocabulary from sft2.
''' + indented_part(joined(CHAT_PARTS, GROW_SOLUTION)) + '''


    # Pairs of a right answer and a near miss, and batches of chats.
''' + indented_part(PAIRS_PART) + '''


    # TODO: Finish forward. ids has shape (rows, steps), and last holds the
    #   index of each row's END. Run ids through the GPT's wte and wpe, its
    #   blocks and ln_f, as GPT.forward does, but skip lm_head. For each row,
    #   take the output at position last[row] and pass it through self.score.
    #   Return one score per row, a tensor of shape (rows,).
''' + indented_part(REWARD_INIT) + '''

        def forward(
            self,
            ids: Tensor,
            last: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the Bradley Terry loss, as a tensor with one number. For each
    #   pair, take the log probability that the chosen answer is preferred
    #   over the rejected one, computed from the difference of their scores.
    #   Return minus the mean of those log probabilities.
    #   Example: bradley_terry_loss(torch.tensor([1.0]), torch.tensor([1.0]))
    #   returns tensor(0.6931).
    def bradley_terry_loss(
        chosen: Tensor,
        rejected: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the fraction of pairs whose chosen score is higher than the
    #   rejected score, as a Python float. A tie counts as wrong.
    #   Example: pair_accuracy(torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0]))
    #   returns 0.5.
    def pair_accuracy(
        chosen: Tensor,
        rejected: Tensor,
    ) -> float:
        ...


    # TODO: Take one training step. chosen and rejected are each a pair
    #   (ids, last), ready to pass to model. Compute the Bradley Terry loss of
    #   model's scores for chosen and rejected. Zero the old gradients,
    #   backpropagate, call the optimizer's step, and return the loss as a
    #   Python float.
    def reward_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chosen: tuple[Tensor, Tensor],
        rejected: tuple[Tensor, Tensor],
    ) -> float:
        ...
    ''',
)

REWARD1_SOLUTION = source_file(
    '''
    # reward1.py, a solution for comparison

    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    logs = F.logsigmoid(torch.tensor([2.0, -200.0]))
    h = torch.arange(12.0).view(2, 3, 2)
    picked = h[torch.arange(2), torch.tensor([2, 0])]


''' + indented_part(GPT_PARTS) + '''


''' + indented_part(joined(CHAT_PARTS, GROW_SOLUTION)) + '''


''' + indented_part(PAIRS_PART) + '''


''' + indented_part(REWARD_MODEL_PART) + '''


''' + indented_part(REWARD_SOLUTION) + '''
    ''',
)


def small_reward_model(make: type) -> nn.Module:
    torch.manual_seed(0)
    gpt = MODEL_REFERENCE["GPT"](68, 24, 32, 2, 4)
    torch.manual_seed(1)

    return make(gpt)


# Ids for every letter, since a near miss can write any capital, within a small GPT's 65 characters.
SMALL_POSITIONS = {letter: position + 1 for position, letter in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def small_batch(chats: list[tuple[str, str]]) -> tuple[Tensor, Tensor]:
    return REWARD_REFERENCE["scored_batch"](chats, SMALL_POSITIONS)


def check_reward_forward(found: type) -> None:
    for chats in ((("ab", "AB"), ("cab", "CA")), (("c", "CC"), ("abc", "ABC"), ("ba", "B"))):
        ids, last = small_batch(list(chats))
        model = small_reward_model(found)
        reference = small_reward_model(REWARD_REFERENCE["RewardModel"])

        with torch.no_grad():
            expected = reference(ids, last)
            actual = model(ids, last)

        call = f"RewardModel of a small GPT on {len(chats)} chats"
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )
        reference(ids, last).sum().backward()

        # enable_grad puts autograd back as it was, even after a forward that turns it off for the whole thread.
        with torch.enable_grad():
            scores = model(ids, last)

        if not scores.requires_grad:
            raise Mismatch(
                f"{call} returns scores that autograd did not record, so training would never reach the GPT. Remove "
                "torch.no_grad, set_grad_enabled and .detach() from forward.",
            )

        scores.sum().backward()

        if model.gpt.wte.weight.grad is None:
            raise Mismatch(
                f"{call} sends no gradient back to the GPT's wte, so training would leave the GPT unchanged. Keep "
                "autograd on through the whole forward.",
            )

        expect_tensor(
            call=f"the GPT's wte gradient after backward from the scores of {call}",
            actual=model.gpt.wte.weight.grad,
            expected=reference.gpt.wte.weight.grad,
        )


def check_reward_step(found: object) -> None:
    for seed in (821, 822):
        generator = torch.Generator().manual_seed(seed)
        words = ["ab", "ba", "cab", "abc", "bca"]
        model = small_reward_model(REWARD_REFERENCE["RewardModel"])
        reference = small_reward_model(REWARD_REFERENCE["RewardModel"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)

        for step in range(2):
            pairs = REWARD_REFERENCE["preference_pairs"](words, 4, generator)
            chosen = small_batch([(question, right) for question, right, _ in pairs])
            rejected = small_batch([(question, wrong) for question, _, wrong in pairs])
            expected = REWARD_REFERENCE["reward_step"](reference, reference_optimizer, chosen, rejected)
            expect_close(
                call=f"reward_step on a small reward model, step {step + 1}",
                actual=found(model, optimizer, chosen, rejected),
                expected=expected,
                tolerance=1e-5,
            )

        expect_tensor(
            call="the score layer's weight after two calls of reward_step",
            actual=model.score.weight.detach(),
            expected=reference.score.weight.detach(),
        )


def check_trained_reward_model(found: object) -> None:
    gpt = trained_gpt(MODEL_REFERENCE["GPT"])
    REWARD_REFERENCE["grow_vocabulary"](gpt, 3)
    torch.manual_seed(0)
    model = REWARD_REFERENCE["RewardModel"](gpt)
    _, positions = REWARD_REFERENCE["chat_vocabulary"]()
    train_words, _ = REWARD_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        chosen, rejected = pair_batches(REWARD_REFERENCE["preference_pairs"](train_words, 32, generator), positions)
        found(model, optimizer, chosen, rejected)

    accuracy = held_out_accuracy(model, held_out_pairs(200))

    if accuracy < 0.8:
        raise Mismatch(f"After 50 steps of reward_step on your gpt3 model, it ranks {accuracy:.0%} of 200 held out pairs correctly, but it should reach 80%.")


def check_reward1(checker: Checker) -> None:
    checker.test_method(
        class_name="RewardModel",
        method="forward",
        description="scores each chat at its END with one number",
        run=check_reward_forward,
    )
    check_on_inputs(
        checker=checker,
        function="bradley_terry_loss",
        description="is minus the mean log sigmoid of the score differences",
        inputs=(
            (torch.tensor([1.0, 0.5, -2.0]), torch.tensor([0.0, 1.5, -2.0])),
            (torch.tensor([3.0, -150.0]), torch.tensor([-1.0, 60.0])),
        ),
        reference=REWARD_REFERENCE["bradley_terry_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="pair_accuracy",
        description="returns the fraction of chosen scores above rejected ones",
        inputs=(
            (torch.tensor([1.0, 0.5, -2.0, 4.0]), torch.tensor([0.0, 1.5, -2.0, 3.0])),
            (torch.tensor([2.0, 2.0, 1.0]), torch.tensor([1.0, 3.0, 0.0])),
        ),
        reference=REWARD_REFERENCE["pair_accuracy"],
        expect=expect_close,
    )
    checker.test(
        function="reward_step",
        description="zeroes the gradients, backpropagates the Bradley Terry loss and steps",
        run=check_reward_step,
    )
    checker.test(
        function="reward_step",
        description="trains your gpt3 model to rank 80% of held out pairs correctly",
        run=check_trained_reward_model,
    )


REWARD1 = Exercise(
    name="reward1",
    chapter=REWARD_MODELS,
    title="A reward model from pairs",
    lesson=REWARD1_LESSON,
    source=REWARD1_SOURCE,
    solution=REWARD1_SOLUTION,
    hints={
        "RewardModel.forward": "Add self.gpt.wte(ids) and self.gpt.wpe of torch.arange(ids.shape[1]). Run each block of "
        "self.gpt.h, then self.gpt.ln_f. Pick each row's position with torch.arange(len(ids)) and last, then apply "
        "self.score and .squeeze(-1).",
        "bradley_terry_loss": "The differences are chosen - rejected. Take F.logsigmoid of them, then the mean, then put "
        "a minus in front.",
        "pair_accuracy": "chosen > rejected gives True or False for each pair. Turn that into floats, take the mean, and "
        "call .item().",
        "reward_step": "model(*chosen) and model(*rejected) give the scores. Compute bradley_terry_loss of them, then "
        "call zero_grad, backward and step, and return loss.item().",
    },
    introduces=(),
    check=check_reward1,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# reward2
# ---------------------------------------------------------------------------

# The toy, 40 answers whose only feature is their length, a reference that writes about 8 characters, a true reward
# that is best at 12, and a reward model that sees only the length, fit to pairs the reference wrote.
TOY_PART = '''
LENGTHS = torch.arange(1, 41).float()


def reference_policy() -> Tensor:
    weights = torch.exp(-0.5 * ((LENGTHS - 8) / 3) ** 2) + 1e-6

    return weights / weights.sum()


def true_reward() -> Tensor:
    return -(((LENGTHS - 12) / 3) ** 2)


# 2,000 pairs of answers drawn from the reference. The first answer of a pair wins with the Bradley Terry probability
# from the true rewards. Returns the chosen and the rejected answers.
def judged_pairs(generator: torch.Generator) -> tuple[Tensor, Tensor]:
    reference, truth = reference_policy(), true_reward()
    first = torch.multinomial(reference, 2000, replacement=True, generator=generator)
    second = torch.multinomial(reference, 2000, replacement=True, generator=generator)
    first_won = torch.rand(2000, generator=generator) < torch.sigmoid(truth[first] - truth[second])

    return torch.where(first_won, first, second), torch.where(first_won, second, first)


# A reward model whose reward is a learned slope times the length, fit to the pairs with the Bradley Terry loss.
def learned_reward(generator: torch.Generator) -> Tensor:
    chosen, rejected = judged_pairs(generator)
    slope = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([slope], lr=0.05)

    for _ in range(300):
        rewards = slope * LENGTHS
        loss = -F.logsigmoid(rewards[chosen] - rewards[rejected]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (slope * LENGTHS).detach()
'''

TOY_SOLUTION = '''
def best_policy(
    reference: Tensor,
    rewards: Tensor,
    beta: float,
) -> Tensor:
    return torch.softmax(reference.log() + rewards / beta, dim=0)


def kl_divergence(
    policy: Tensor,
    reference: Tensor,
) -> float:
    used = policy > 0

    return (policy[used] * (policy[used] / reference[used]).log()).sum().item()


def expected_reward(
    policy: Tensor,
    rewards: Tensor,
) -> float:
    return (policy * rewards).sum().item()
'''


def toy_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "F": F,
        "Tensor": Tensor,
    }
    exec(joined(TOY_PART, TOY_SOLUTION), namespace)

    return namespace


TOY_REFERENCE = toy_namespace()


def measure_hacking() -> list[tuple[str, float, float, float, float]]:
    """reward2's table, for each beta the policy's KL from the reference, its learned and true rewards and its
    mean length."""
    reference = TOY_REFERENCE["reference_policy"]()
    truth = TOY_REFERENCE["true_reward"]()
    learned = TOY_REFERENCE["learned_reward"](torch.Generator().manual_seed(0))
    lengths = TOY_REFERENCE["LENGTHS"]
    rows = [("reference", 0.0, TOY_REFERENCE["expected_reward"](reference, learned), TOY_REFERENCE["expected_reward"](reference, truth), TOY_REFERENCE["expected_reward"](reference, lengths))]

    for beta in (10.0, 5.0, 3.0, 2.0, 1.0, 0.5):
        policy = TOY_REFERENCE["best_policy"](reference, learned, beta)
        rows.append(
            (
                str(beta),
                TOY_REFERENCE["kl_divergence"](policy, reference),
                TOY_REFERENCE["expected_reward"](policy, learned),
                TOY_REFERENCE["expected_reward"](policy, truth),
                TOY_REFERENCE["expected_reward"](policy, lengths),
            ),
        )

    return rows


def measure_floors() -> dict[tuple[str, float], tuple[float, float]]:
    """What reward2's collapse owes to the reference's floor of 1e-6 on every length. For each floor and beta, the
    best policy's mean length and true reward, with the learned reward kept from the reference as shipped."""
    lengths = TOY_REFERENCE["LENGTHS"]
    truth = TOY_REFERENCE["true_reward"]()
    learned = TOY_REFERENCE["learned_reward"](torch.Generator().manual_seed(0))
    results = {}

    for name, floor in (("1e-6", 1e-6), ("1e-9", 1e-9), ("none", 0.0)):
        weights = torch.exp(-0.5 * ((lengths - 8) / 3) ** 2) + floor
        reference = weights / weights.sum()

        for beta in (2.0, 1.0, 0.5):
            policy = TOY_REFERENCE["best_policy"](reference, learned, beta)
            results[(name, beta)] = (
                TOY_REFERENCE["expected_reward"](policy, lengths),
                TOY_REFERENCE["expected_reward"](policy, truth),
            )

    results[("learned reward at 40", 0.0)] = (learned[-1].item(), reference_chance_at_40())

    return results


def reference_chance_at_40() -> float:
    return TOY_REFERENCE["reference_policy"]()[-1].item()


REWARD2_LESSON = prose(
    """
    A reward model learns from pairs, so it is only accurate for answers
    similar to the ones in those pairs. If you train a policy to get the
    highest score it can, it finds the answers where the reward model is
    wrong. This is called reward hacking.

    Reinforcement learning from human feedback, RLHF, holds the policy back
    with a penalty. It maximizes the expected reward minus beta times the KL
    divergence from the reference. The reference is the model before RLHF,
    and the KL divergence measures how far the policy has moved away from
    it. InstructGPT adds this penalty to keep the policy from over
    optimizing the reward model.

    ```text
    E[r] - beta * KL(policy || reference)
    ```

    The best policy has a closed form, from section 4 of the DPO paper. Each
    answer's probability under the reference is multiplied by
    `exp(r(y) / beta)`. Then all the probabilities are divided by their sum
    Z, so they add up to 1.

    ```text
    policy(y) = reference(y) * exp(r(y) / beta) / Z
    ```

    Computing Z means summing over every possible answer. That is easy for
    the 40 answers here.

    Taking logs, the best policy is the softmax of
    `log reference(y) + r(y) / beta`. A large beta keeps the policy near the
    reference. A small beta puts almost all the probability on the answer
    with the highest reward.

    The toy has 40 possible answers, and the only thing that differs between
    them is their length, from 1 to 40 characters. The reference writes
    about 8 characters, and the true reward is highest at 12. A reward model
    that sees only the length is fit to 2,000 pairs from the reference, with
    reward1's loss. Near 8 characters, longer answers were better. So the
    reward model learned that longer is always better.

    | beta | KL | Learned reward | True reward | Mean length |
    |---|---|---|---|---|
    | reference | 0 | 4.40 | -2.69 | 8.1 |
    | 10 | 0.01 | 4.66 | -2.31 | 8.5 |
    | 3 | 0.15 | 5.27 | -1.62 | 9.7 |
    | 2 | 0.35 | 5.75 | -1.43 | 10.5 |
    | 1 | 10.6 | 17.77 | -60.63 | 32.5 |
    | 0.5 | 14.9 | 21.57 | -84.08 | 39.5 |

    As beta goes down to 2, the smaller penalty gives a better true reward.
    At a beta of 1, the policy jumps to answers of 30 characters or more,
    which the reference almost never wrote. The learned reward triples while
    the true reward collapses.

    The jump comes from a floor in the reference. Before its weights are
    normalized into probabilities, the reference adds 1e-6 to every length's
    weight. A language model also gives every answer some small probability.
    At 40 characters, the floor leaves a probability of about 1 in 7.5
    million. A beta of 1 multiplies each length's probability by
    `exp(learned reward)`. That is e^22 at 40 characters against e^4.4 at 8,
    which is enough for the longest answers to win. Without the floor, the
    probability at 40 characters is about e^-59. Then the best policy at a
    beta of 1 has a mean length near 13 characters, with a true reward of
    -1.09.

    A KL divergence is a sum over every answer of
    `policy * log(policy / reference)`. An answer the policy never gives has
    probability 0 and adds nothing to the sum, so those terms are left out.
    """,
)

REWARD2_SOURCE = source_file(
    '''
    # reward2.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    chances = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)    # tensor([0.2689, 0.7311])
    used = torch.tensor([0.5, 0.0, 0.5]) > 0                    # tensor([ True, False,  True])


    # The toy's answer lengths, reference policy, true reward and learned reward.
''' + indented_part(TOY_PART) + '''


    # TODO: Return the policy that maximizes the expected reward minus beta
    #   times the KL divergence from reference. It is the softmax of
    #   log(reference) + rewards / beta.
    #   Example: best_policy(torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0]),
    #   1.0) returns tensor([0.2689, 0.7311]).
    def best_policy(
        reference: Tensor,
        rewards: Tensor,
        beta: float,
    ) -> Tensor:
        ...


    # TODO: Return the KL divergence of policy from reference, as a Python
    #   float. It is the sum over the answers of
    #   policy * log(policy / reference). Leave out every answer where policy
    #   is 0.
    #   Example: kl_divergence(torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5]))
    #   returns about 0.6931, which is log(2).
    def kl_divergence(
        policy: Tensor,
        reference: Tensor,
    ) -> float:
        ...


    # TODO: Return the expected reward of policy, as a Python float. Multiply
    #   each answer's probability by its reward, and add up the products.
    def expected_reward(
        policy: Tensor,
        rewards: Tensor,
    ) -> float:
        ...
    ''',
)

REWARD2_SOLUTION = source_file(
    '''
    # reward2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    chances = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    used = torch.tensor([0.5, 0.0, 0.5]) > 0


''' + indented_part(TOY_PART) + '''


''' + indented_part(TOY_SOLUTION) + '''
    ''',
)


def check_reward2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="best_policy",
        description="scales the reference by the exponential of the rewards over beta",
        inputs=(
            (torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.0, 1.0, 2.0]), 1.0),
            (torch.tensor([0.7, 0.1, 0.1, 0.1]), torch.tensor([1.0, -1.0, 3.0, 0.5]), 0.25),
            (torch.tensor([0.25, 0.25, 0.5]), torch.tensor([4.0, 0.0, -2.0]), 8.0),
        ),
        reference=TOY_REFERENCE["best_policy"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kl_divergence",
        description="sums policy times log policy over reference, skipping zeros",
        inputs=(
            (torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.2, 0.3, 0.5])),
            (torch.tensor([0.0, 0.6, 0.4]), torch.tensor([0.5, 0.25, 0.25])),
        ),
        reference=TOY_REFERENCE["kl_divergence"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="expected_reward",
        description="weights each reward by its probability",
        inputs=(
            (torch.tensor([0.5, 0.3, 0.2]), torch.tensor([1.0, -2.0, 4.0])),
            (torch.tensor([0.1, 0.0, 0.9]), torch.tensor([3.0, 100.0, -1.0])),
        ),
        reference=TOY_REFERENCE["expected_reward"],
        expect=expect_close,
    )


REWARD2 = Exercise(
    name="reward2",
    chapter=REWARD_MODELS,
    title="The KL penalty and reward hacking",
    lesson=REWARD2_LESSON,
    source=REWARD2_SOURCE,
    solution=REWARD2_SOLUTION,
    hints={
        "best_policy": "Add reference.log() and rewards divided by beta, then take torch.softmax over dim=0.",
        "kl_divergence": "Make a mask of the answers where policy > 0. On those answers, sum policy times the log of "
        "policy divided by reference, then call .item().",
        "expected_reward": "Multiply policy by rewards, sum the result, and call .item().",
    },
    introduces=(),
    check=check_reward2,
)
