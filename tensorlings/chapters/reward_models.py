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
        the best one. A reward model learns from such picks, a score for any
        answer, with the Bradley Terry loss. RLHF then trains the chat model
        to score high, with a penalty for drifting from where it started.
        This chapter trains a reward model on your GPT, then shows on a toy
        what the penalty is for.

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


# The word in capitals with one slip, a wrong letter, a letter left out, or a letter written twice.
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


# Questions with a chosen answer, the word in capitals, and a rejected one, a near miss.
def preference_pairs(
    words: list[str],
    count: int,
    generator: torch.Generator,
) -> list[tuple[str, str, str]]:
    picked = torch.randint(0, len(words), (count,), generator=generator).tolist()

    return [(words[position], words[position].upper(), near_miss(words[position], generator)) for position in picked]


# The chats of questions and answers, padded at the end with END, and the place of each chat's own END.
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
    over seeds 0 to 2."""
    pairs = held_out_pairs()
    kinds = ("wrong letter", "letter left out", "letter doubled")
    results: dict[str, float] = {}

    for name, pretrained in (("gpt3's GPT", True), ("untrained GPT", False)):
        for steps in (0, 50, 200):
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

    return results


# ---------------------------------------------------------------------------
# reward1
# ---------------------------------------------------------------------------

REWARD1_LESSON = prose(
    """
    People find it easier to pick the better of two answers than to write
    the best one. InstructGPT's labelers ranked 4 to 9 answers to each prompt,
    and every two of them made one comparison. A reward model learns a score
    for any answer from such picks.

    The Bradley Terry model says how scores turn into picks. The chance that
    answer a beats answer b is the sigmoid of the gap between their scores.

    ```text
    P(a beats b) = sigmoid(r(a) - r(b))
    ```

    Only the gap counts, so adding a number to every score changes nothing.
    The loss is minus the log of that chance for the answer that won, as in
    InstructGPT. `F.logsigmoid` takes it without rounding to 0, as gan1 did.

    ```python
    F.logsigmoid(torch.tensor([2.0, -200.0]))    # tensor([-1.2693e-01, -2.0000e+02]), the -200 kept exact
    ```

    The reward model is the GPT itself with a new head. InstructGPT started
    from its SFT model with the last layer removed, and added a layer that
    gives one number. Here a Linear reads the GPT's output at the chat's END.
    That place has seen the whole question and the whole answer.

    Chats in a batch differ in length, so `last` holds the place of each
    one's END. Indexing with two tensors picks one place per row.

    ```python
    h = torch.arange(12.0).view(2, 3, 2)
    h[torch.arange(2), torch.tensor([2, 0])]    # tensor([[4., 5.], [6., 7.]]), place 2 of row 0, place 0 of row 1
    ```

    The pairs are a right answer and a near miss, with one letter wrong,
    left out or doubled. Here is the share of 500 held out pairs ranked
    right, by steps of 32 pairs, averaged over 3 seeds.

    | Slip | 0 steps | 50 steps | 200 steps |
    |---|---|---|---|
    | A letter left out | 44% | 100% | 100% |
    | A letter doubled | 60% | 100% | 100% |
    | A wrong letter | 50% | 79% | 78% |
    | All | 51% | 93% | 93% |

    Two slips change the length, and the model ranks every one of them right.
    A wrong letter keeps the length, and more training doesn't help it. The
    model learned the easy shortcut, a length that matches the question, and
    only part of the real task. reward2 shows what a policy does with such a
    gap. An untrained GPT reaches 92% too, so pretraining adds little here.
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


    # TODO: Finish forward. Run ids through the GPT's wte and wpe, its blocks
    #   and ln_f, as GPT.forward does, but not lm_head. Then score the output
    #   at the place last gives for each row, one number per row.
''' + indented_part(REWARD_INIT) + '''

        def forward(
            self,
            ids: Tensor,
            last: Tensor,
        ) -> Tensor:
            ...


    # TODO: Return the Bradley Terry loss, minus the mean log chance that each
    #   chosen answer beats its rejected one.
    def bradley_terry_loss(
        chosen: Tensor,
        rejected: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the share of pairs whose chosen score is above the rejected
    #   one, as a plain number. A tie counts as wrong.
    def pair_accuracy(
        chosen: Tensor,
        rejected: Tensor,
    ) -> float:
        ...


    # TODO: Take one training step on the Bradley Terry loss of model's scores
    #   for chosen and rejected, each a pair of ids and last. Zero the old
    #   gradients, backpropagate, step, and return the loss as a plain number.
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

        expect_tensor(
            call=f"RewardModel of a small GPT on {len(chats)} chats",
            actual=actual,
            expected=expected,
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
        raise Mismatch(f"After 50 steps of reward_step on your gpt3 model, it ranks {accuracy:.0%} of 200 held out pairs right, but it should reach 80%.")


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
        description="is minus the mean log sigmoid of the gaps",
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
        description="gives the share of chosen scores above rejected ones",
        inputs=(
            (torch.tensor([1.0, 0.5, -2.0, 4.0]), torch.tensor([0.0, 1.5, -2.0, 3.0])),
            (torch.tensor([2.0, 2.0, 1.0]), torch.tensor([1.0, 3.0, 0.0])),
        ),
        reference=REWARD_REFERENCE["pair_accuracy"],
        expect=expect_close,
    )
    checker.test(
        function="reward_step",
        description="zeroes, backpropagates and steps on the Bradley Terry loss",
        run=check_reward_step,
    )
    checker.test(
        function="reward_step",
        description="trains your gpt3 model to rank 80% of held out pairs right",
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
        "RewardModel.forward": "Add self.gpt.wte(ids) and self.gpt.wpe of torch.arange(ids.shape[1]), run each block of "
        "self.gpt.h, then self.gpt.ln_f. Pick each row's place with torch.arange(len(ids)) and last, then self.score "
        "and .squeeze(-1).",
        "bradley_terry_loss": "The gaps are chosen - rejected. F.logsigmoid of them, the mean, then minus.",
        "pair_accuracy": "chosen > rejected gives True or False per pair. Turn it into floats, take the mean, and "
        ".item().",
        "reward_step": "model(*chosen) and model(*rejected) give the scores. bradley_terry_loss of them, then "
        "zero_grad, backward, step, and loss.item().",
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


# 2,000 pairs from the reference, each won by an answer with the Bradley Terry chance its true reward gives.
def judged_pairs(generator: torch.Generator) -> tuple[Tensor, Tensor]:
    reference, truth = reference_policy(), true_reward()
    first = torch.multinomial(reference, 2000, replacement=True, generator=generator)
    second = torch.multinomial(reference, 2000, replacement=True, generator=generator)
    first_won = torch.rand(2000, generator=generator) < torch.sigmoid(truth[first] - truth[second])

    return torch.where(first_won, first, second), torch.where(first_won, second, first)


# A reward model that is a slope times the length, fit to the pairs with the Bradley Terry loss.
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


REWARD2_LESSON = prose(
    """
    A reward model learns from pairs, so it is only right near the answers
    those pairs held. Train a policy to score as high as it can, and it finds
    where the reward model is wrong. That is reward hacking.

    RLHF holds the policy back with a penalty. It maximizes the expected
    reward minus beta times the KL divergence from the reference, the model
    before RLHF. InstructGPT adds it to keep the policy from over optimizing
    the reward model.

    ```text
    E[r] - beta * KL(policy || reference)
    ```

    Over a few answers the best policy has a closed form, from section 4 of
    the DPO paper. Each answer keeps its reference chance, scaled by the
    exponential of its reward over beta, then all are scaled to add up to 1.

    ```text
    policy(y) = reference(y) * exp(r(y) / beta) / Z
    ```

    In logs that is a softmax of the log of the reference plus the rewards
    over beta. A large beta stays near the reference, and a small one piles
    everything on the top reward.

    The toy has 40 answers whose only feature is their length. The reference
    writes about 8 characters, and the true reward is best at 12. A reward
    model that sees only the length is fit, with reward1's loss, to 2,000
    pairs from the reference. Near the reference longer was better, so it
    learned that longer is always better.

    | beta | KL | Learned reward | True reward | Mean length |
    |---|---|---|---|---|
    | reference | 0 | 4.40 | -2.69 | 8.1 |
    | 10 | 0.01 | 4.66 | -2.31 | 8.5 |
    | 3 | 0.15 | 5.27 | -1.62 | 9.7 |
    | 2 | 0.35 | 5.75 | -1.43 | 10.5 |
    | 1 | 10.6 | 17.77 | -60.63 | 32.5 |
    | 0.5 | 14.9 | 21.57 | -84.08 | 39.5 |

    Down to a beta of 2, a smaller penalty buys a better true reward. At 1
    the policy jumps to answers of 30 characters and more, which the
    reference almost never wrote. The learned reward triples while the true
    one collapses.

    A KL divergence sums over every answer, and a choice the policy never
    makes adds nothing, so those terms are left out.
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


    # The toy's answers, its reference, its true reward and its learned one.
''' + indented_part(TOY_PART) + '''


    # TODO: Return the policy that maximizes the expected rewards minus beta
    #   times the KL divergence from reference. It is the softmax of the log of
    #   reference plus rewards over beta.
    def best_policy(
        reference: Tensor,
        rewards: Tensor,
        beta: float,
    ) -> Tensor:
        ...


    # TODO: Return the KL divergence of policy from reference, as a plain
    #   number. Sum policy times the log of policy over reference, leaving out
    #   every answer policy gives no chance.
    def kl_divergence(
        policy: Tensor,
        reference: Tensor,
    ) -> float:
        ...


    # TODO: Return the reward policy expects, each answer's chance times its
    #   reward, summed, as a plain number.
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
        description="weights each reward by its chance",
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
        "best_policy": "reference.log() plus rewards divided by beta, then torch.softmax over dim=0.",
        "kl_divergence": "Keep the answers where policy > 0 with a mask. Sum policy times the log of policy over "
        "reference on those, then .item().",
        "expected_reward": "Multiply policy by rewards, sum, and .item().",
    },
    introduces=(),
    check=check_reward2,
)
