"""82_dpo, the DPO loss from its paper, the GPT trained with it next to the reward model and PPO route, and GRPO's
advantages taken relative to a group of samples."""

from __future__ import annotations

import copy
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
from .reward_models import PAIRS_PART, held_out_pairs, trained_reward_model
from .sft import CHAT_PARTS, GPT_PARTS, GROW_SOLUTION, SFT_REFERENCE, chat_rows, indented_part, joined, small_chat_gpt

DPO = Chapter(
    folder="82_dpo",
    title="Direct preference optimization",
    readme=source_file(
        """
        # Direct preference optimization

        RLHF trains a reward model, then runs PPO against it with a penalty
        for drifting. DPO gets the same goal from the pairs alone, with one
        loss and no sampling. This chapter derives the DPO loss from reward2's
        closed form, trains your GPT with it next to the reward model route,
        then builds GRPO's advantages, which need no value model.

        ## Further reading

        - [Rafailov and colleagues, direct preference optimization](https://arxiv.org/abs/2305.18290)
        - [Pal and colleagues, Smaug and the failure modes of DPO](https://arxiv.org/abs/2402.13228)
        - [Shao and colleagues, DeepSeekMath and GRPO](https://arxiv.org/abs/2402.03300)
        - [Grattafiori and colleagues, the Llama 3 herd of models, section 4.1](https://arxiv.org/abs/2407.21783)
        """,
    ),
)

LOGPROB_SOLUTION = '''
def answer_logprobs(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    logs = F.log_softmax(model(x), dim=-1)
    kept = y != -100
    picked = logs.gather(-1, y.masked_fill(~kept, 0).unsqueeze(-1)).squeeze(-1)

    return (picked * kept).sum(dim=1)
'''

DPO_LOSS_SOLUTION = '''
def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> Tensor:
    gap = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)

    return -F.logsigmoid(beta * gap).mean()


def implicit_rewards(
    policy_logprobs: Tensor,
    reference_logprobs: Tensor,
    beta: float,
) -> Tensor:
    return beta * (policy_logprobs - reference_logprobs)
'''

DPO1_SOLUTION_PART = joined(LOGPROB_SOLUTION, DPO_LOSS_SOLUTION)

DPO_STEP_SOLUTION = '''
def frozen_copy(model: nn.Module) -> nn.Module:
    reference = copy.deepcopy(model)
    reference.eval()
    reference.requires_grad_(False)

    return reference


def dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen: tuple[Tensor, Tensor],
    rejected: tuple[Tensor, Tensor],
    beta: float,
    nll: float,
) -> float:
    with torch.no_grad():
        reference_chosen = answer_logprobs(reference, *chosen)
        reference_rejected = answer_logprobs(reference, *rejected)

    policy_chosen = answer_logprobs(policy, *chosen)
    policy_rejected = answer_logprobs(policy, *rejected)
    loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta) + nll * answer_loss(policy, *chosen)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
'''

GRPO_SOLUTION = '''
def group_advantages(rewards: Tensor) -> Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    spread = rewards.std(dim=1, keepdim=True)

    return (rewards - mean) / (spread + 1e-4)


def kl_estimate(
    logprobs: Tensor,
    reference_logprobs: Tensor,
) -> Tensor:
    gap = reference_logprobs - logprobs

    return gap.exp() - gap - 1


def grpo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    reference_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    epsilon: float,
    beta: float,
) -> Tensor:
    ratio = (logprobs - old_logprobs).exp()
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    gain = torch.minimum(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1))
    per_token = gain - beta * kl_estimate(logprobs, reference_logprobs)
    per_answer = (per_token * mask).sum(dim=1) / mask.sum(dim=1)

    return -per_answer.mean()
'''


def dpo_namespace() -> dict[str, object]:
    """The given parts and the solutions, run once, so the checks compare against exactly the code the files hold."""
    namespace: dict[str, object] = {
        "copy": copy,
        "math": math,
        "re": re,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "load_shakespeare": load_shakespeare,
    }
    exec(joined(GPT_PARTS, CHAT_PARTS, GROW_SOLUTION, PAIRS_PART, DPO1_SOLUTION_PART, DPO_STEP_SOLUTION, GRPO_SOLUTION), namespace)

    return namespace


DPO_REFERENCE = dpo_namespace()


# ---------------------------------------------------------------------------
# The runs behind the lessons' tables
# ---------------------------------------------------------------------------


def sft_start(
    steps: int,
    seed: int = 0,
) -> nn.Module:
    """gpt3's GPT taught the made up task for steps steps, as sft2 teaches it, the start of every DPO run."""
    model = trained_gpt(MODEL_REFERENCE["GPT"])
    SFT_REFERENCE["grow_vocabulary"](model, 3)
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, _ = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        x, y = SFT_REFERENCE["chat_batch"](SFT_REFERENCE["word_pairs"](train_words, 32, generator), positions)
        SFT_REFERENCE["sft_step"](model, optimizer, x, y)

    return model


def chat_pair_batches(
    pairs: list[tuple[str, str, str]],
    positions: dict[str, int],
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    chosen = SFT_REFERENCE["chat_batch"]([(question, right) for question, right, _ in pairs], positions)
    rejected = SFT_REFERENCE["chat_batch"]([(question, wrong) for question, _, wrong in pairs], positions)

    return chosen, rejected


def after_preferences(
    model: nn.Module,
    steps: int,
    route: str,
    seed: int = 0,
) -> nn.Module:
    """model trained steps more steps of 32 pairs, by more SFT on the right answers or by DPO with beta 0.1, with no
    NLL term or with Llama 3's 0.2."""
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, _ = SFT_REFERENCE["question_words"]()
    generator = torch.Generator().manual_seed(100 + seed)

    if route == "sft":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        for _ in range(steps):
            x, y = SFT_REFERENCE["chat_batch"](SFT_REFERENCE["word_pairs"](train_words, 32, generator), positions)
            SFT_REFERENCE["sft_step"](model, optimizer, x, y)

        return model

    nll = {"dpo": 0.0, "dpo with nll": 0.2, "dpo with nll 1": 1.0}[route]
    reference = DPO_REFERENCE["frozen_copy"](model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    for _ in range(steps):
        chosen, rejected = chat_pair_batches(DPO_REFERENCE["preference_pairs"](train_words, 32, generator), positions)
        DPO_REFERENCE["dpo_step"](model, reference, optimizer, chosen, rejected, 0.1, nll)

    return model


def held_out_logprobs(model: nn.Module) -> tuple[float, float]:
    """The mean log chance of the right answer and of the near miss, over 200 held out pairs."""
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    chosen, rejected = chat_pair_batches(held_out_pairs(200), positions)
    model.eval()

    with torch.no_grad():
        right = DPO_REFERENCE["answer_logprobs"](model, *chosen).mean().item()
        wrong = DPO_REFERENCE["answer_logprobs"](model, *rejected).mean().item()

    model.train()

    return right, wrong


def held_out_score(model: nn.Module) -> float:
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    _, test_words = SFT_REFERENCE["question_words"]()

    return SFT_REFERENCE["exact_match"](model, test_words[:200], positions, names)


def sample_answers(
    model: nn.Module,
    question: str,
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, list[str]]:
    """count answers to question drawn at temperature 1, the ids of each whole chat padded with END, and the text."""
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    prompt = [65] + [positions[character] for character in question] + [66]
    ids = torch.tensor([prompt] * count)
    done = torch.zeros(count, dtype=torch.bool)

    with torch.no_grad():
        for _ in range(12):
            chances = torch.softmax(model(ids)[:, -1], dim=-1)
            token = torch.multinomial(chances, 1, generator=generator).squeeze(1)
            token = torch.where(done, torch.full_like(token, 67), token)
            ids = torch.cat([ids, token.unsqueeze(1)], dim=1)
            done = done | (token == 67)

            if done.all():
                break

    texts = []

    for row in ids[:, len(prompt) :].tolist():
        written = row[: row.index(67)] if 67 in row else row
        texts.append("".join(names[token] for token in written))

    return ids, texts


def best_of(
    model: nn.Module,
    reward_model: nn.Module,
    count: int,
    seed: int = 0,
) -> float:
    """The share of 200 held out words answered right when the reward model picks the best of count samples."""
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    _, test_words = SFT_REFERENCE["question_words"]()
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    reward_model.eval()
    right = 0

    for word in test_words[:200]:
        ids, texts = sample_answers(model, word, count, generator)
        written = ids[:, len(word) + 2 :]
        ends = torch.where((written == 67).any(dim=1), (written == 67).int().argmax(dim=1), written.shape[1] - 1)

        with torch.no_grad():
            scores = reward_model(ids, ends + len(word) + 2)

        right += texts[scores.argmax().item()] == word.upper()

    model.train()

    return right / 200


def measure_dpo() -> dict[str, float]:
    """Every number in dpo2's table, from a GPT given 200 steps of SFT, averaged over seeds 0 to 2."""
    results: dict[str, float] = {}

    for seed in (0, 1, 2):
        start = sft_start(200, seed)
        state = copy.deepcopy(start.state_dict())
        rows = {"start": start}

        for route in ("sft", "dpo", "dpo with nll", "dpo with nll 1"):
            model = sft_start(0, seed)
            model.load_state_dict(state)
            rows[route] = after_preferences(model, 100, route, seed)

        for name, model in rows.items():
            right, wrong = held_out_logprobs(model)
            numbers = {"exact": held_out_score(model), "right logprob": right, "near miss logprob": wrong}

            for key, value in numbers.items():
                results[f"{name}, {key}"] = results.get(f"{name}, {key}", 0.0) + value / 3

            print(seed, name, {key: round(value, 3) for key, value in numbers.items()}, flush=True)

        reward_model = trained_reward_model(50, seed=seed)

        for count in (1, 4, 16):
            score = best_of(start, reward_model, count, seed)
            results[f"best of {count}"] = results.get(f"best of {count}", 0.0) + score / 3
            print(seed, "best of", count, round(score, 3), flush=True)

    return results


def token_logprobs(
    model: nn.Module,
    ids: Tensor,
    prompt_length: int,
) -> tuple[Tensor, Tensor]:
    """The log chance of each written token of ids after the prompt, and a mask of the ones up to the first END."""
    logs = F.log_softmax(model(ids[:, :-1]), dim=-1)
    written = ids[:, 1:]
    picked = logs.gather(-1, written.unsqueeze(-1)).squeeze(-1)[:, prompt_length - 1 :]
    answer = ids[:, prompt_length:]
    ended_before = torch.cumsum((answer == 67).int(), dim=1) - (answer == 67).int()

    return picked, (ended_before == 0).float()


def measure_grpo(
    steps: int = 60,
    seed: int = 0,
) -> list[tuple[int, float, float]]:
    """grpo1's table, GRPO on the GPT after 200 steps of SFT, 8 questions a step with 8 answers each, a reward of
    1 for an exact answer, epsilon 0.2 and beta 0.04. Each row is the step, the share of 200 held out words
    answered right, and the share of the step's groups whose rewards were all the same."""
    policy = sft_start(200, seed)
    reference = DPO_REFERENCE["frozen_copy"](policy)
    train_words, _ = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(200 + seed)
    rows = [(0, held_out_score(policy), float("nan"))]
    same = 0

    for step in range(1, steps + 1):
        picked = torch.randint(0, len(train_words), (8,), generator=generator).tolist()
        loss = torch.zeros(())
        policy.eval()

        groups = []

        for position in picked:
            word = train_words[position]
            ids, texts = sample_answers(policy, word, 8, generator)
            rewards = torch.tensor([[float(text == word.upper()) for text in texts]])
            same += int(rewards.min() == rewards.max())
            groups.append((word, ids, rewards))

        policy.train()

        for word, ids, rewards in groups:
            advantages = DPO_REFERENCE["group_advantages"](rewards)[0]
            logprobs, mask = token_logprobs(policy, ids, len(word) + 2)

            with torch.no_grad():
                reference_logprobs, _ = token_logprobs(reference, ids, len(word) + 2)

            loss = loss + DPO_REFERENCE["grpo_loss"](logprobs, logprobs.detach(), reference_logprobs, advantages, mask, 0.2, 0.04) / len(groups)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            rows.append((step, held_out_score(policy), same / (20 * len(picked))))
            print(rows[-1], flush=True)
            same = 0

    return rows


# ---------------------------------------------------------------------------
# dpo1
# ---------------------------------------------------------------------------

DPO1_LESSON = prose(
    """
    reward2 found the best policy for a reward in closed form. DPO, from
    Rafailov and colleagues, turns that around. Take logs and solve for the
    reward instead.

    ```text
    r(y) = beta * log(policy(y) / reference(y)) + beta * log Z
    ```

    Put that into reward1's Bradley Terry chance. Only the gap between two
    rewards counts, so the two `log Z` cancel, and what is left needs only
    the policy and the reference.

    ```text
    P(chosen beats rejected) = sigmoid(beta * (log ratio of chosen - log ratio of rejected))
    ```

    A log ratio is the policy's log chance of an answer minus the
    reference's. The DPO loss is minus the log of that chance, averaged over
    pairs. It trains the policy straight on the pairs, with no reward model
    and no sampling. In the paper's title, your language model is secretly a
    reward model, with `beta * (log policy - log reference)` as its implicit
    reward.

    An answer's log chance is the sum of its tokens' log chances, each given
    the tokens before it. Only the answer's tokens count, the targets sft1
    kept. `gather` picks one entry of each row by index.

    ```python
    logs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), dim=-1)
    logs.gather(-1, torch.tensor([[2], [0]]))    # tensor([[-0.4076], [-1.0986]]), entry 2, then entry 0
    ```

    The paper uses a beta of 0.1 unless it says otherwise. Its gradient
    weighs each pair by the sigmoid of minus the implicit reward gap. A pair
    the policy already orders well counts for little.

    | Implicit reward gap | -2 | 0 | 2 | 4 |
    |---|---|---|---|---|
    | Weight of the pair | 0.88 | 0.50 | 0.12 | 0.02 |
    """,
)

DPO1_SOURCE = source_file(
    '''
    # dpo1.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    logs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), dim=-1)
    picked = logs.gather(-1, torch.tensor([[2], [0]]))                       # tensor([[-0.4076], [-1.0986]])
    safe = torch.tensor([4, -100]).masked_fill(torch.tensor([False, True]), 0)    # tensor([4, 0])


    # TODO: Return each row's log chance of its answer, of shape (rows,). Take
    #   log_softmax of model's scores on x, gather each target's entry, and sum
    #   the places whose target isn't -100. gather can't read place -100, so
    #   swap those targets for 0 before gathering.
    def answer_logprobs(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the DPO loss, minus the mean log sigmoid of beta times the
    #   gap between the chosen and the rejected log ratios. A log ratio is the
    #   policy's log chance minus the reference's.
    def dpo_loss(
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
        beta: float,
    ) -> Tensor:
        ...


    # TODO: Return the implicit rewards, beta times the policy's log chances
    #   minus the reference's.
    def implicit_rewards(
        policy_logprobs: Tensor,
        reference_logprobs: Tensor,
        beta: float,
    ) -> Tensor:
        ...
    ''',
)

DPO1_SOLUTION = source_file(
    '''
    # dpo1.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    logs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), dim=-1)
    picked = logs.gather(-1, torch.tensor([[2], [0]]))
    safe = torch.tensor([4, -100]).masked_fill(torch.tensor([False, True]), 0)


''' + indented_part(DPO1_SOLUTION_PART) + '''
    ''',
)


def check_answer_logprobs(found: object) -> None:
    for chats in ((("ab", "AB"), ("cab", "CB")), (("c", "C"), ("abba", "ABBA"), ("bc", "BCC"))):
        x, y = SFT_REFERENCE["pad_rows"](chat_rows(list(chats)))
        model = small_chat_gpt()

        with torch.no_grad():
            expected = DPO_REFERENCE["answer_logprobs"](model, x, y)

        actual = found(model, x, y)
        call = f"answer_logprobs of a small GPT on {len(chats)} padded chats"
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )

        if not actual.requires_grad:
            raise Mismatch(f"{call} holds no gradient, but DPO trains through it. Leave out torch.no_grad here.")


def check_dpo1(checker: Checker) -> None:
    checker.test(
        function="answer_logprobs",
        description="sums the log chances of the answer's tokens",
        run=check_answer_logprobs,
    )
    check_on_inputs(
        checker=checker,
        function="dpo_loss",
        description="is minus the mean log sigmoid of beta times the gap of log ratios",
        inputs=(
            (torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.5]), torch.tensor([-1.5, -2.0]), torch.tensor([-2.0, -2.5]), 0.1),
            (torch.tensor([-4.0, -1.0, -6.0]), torch.tensor([-2.0, -9.0, -1.0]), torch.tensor([-3.0, -3.0, -3.0]), torch.tensor([-3.0, -3.0, -900.0]), 0.5),
        ),
        reference=DPO_REFERENCE["dpo_loss"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="implicit_rewards",
        description="is beta times the log ratio",
        inputs=(
            (torch.tensor([-1.0, -2.0, -0.5]), torch.tensor([-1.5, -1.0, -0.5]), 0.1),
            (torch.tensor([-8.0, -3.0]), torch.tensor([-2.0, -6.0]), 2.0),
        ),
        reference=DPO_REFERENCE["implicit_rewards"],
        expect=expect_tensor,
    )


DPO1 = Exercise(
    name="dpo1",
    chapter=DPO,
    title="The DPO loss",
    lesson=DPO1_LESSON,
    source=DPO1_SOURCE,
    solution=DPO1_SOLUTION,
    hints={
        "answer_logprobs": "F.log_softmax of model(x) over dim=-1. kept = y != -100. Gather with y.masked_fill(~kept, 0) "
        "unsqueezed at the end, squeeze it back, multiply by kept and sum over dim=1.",
        "dpo_loss": "The gap is (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected). Then "
        "F.logsigmoid of beta times it, the mean, and minus.",
        "implicit_rewards": "beta times policy_logprobs - reference_logprobs.",
    },
    introduces=(),
    check=check_dpo1,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# dpo2
# ---------------------------------------------------------------------------

DPO2_LESSON = prose(
    """
    DPO needs a reference, the model before DPO, held still. `copy.deepcopy`
    makes a separate copy, and `requires_grad_(False)` stops every one of
    its weights from training.

    ```python
    layer = nn.Linear(2, 2)
    frozen = copy.deepcopy(layer).requires_grad_(False)    # its own weights, which never train
    ```

    Here DPO starts from gpt3's GPT after 200 steps of sft2's training. The
    pairs are reward1's, a right answer and a near miss. Each run takes 100
    more steps of 32 pairs, with a beta of 0.1. The log chances are averages
    over 200 held out pairs, and every number is the average of 3 seeds.

    | 100 more steps of | Exact answers | Right answer's log chance | Near miss's log chance |
    |---|---|---|---|
    | nothing, the start | 74% | -1.08 | -14.7 |
    | SFT on the right answers | 92% | -0.32 | -15.9 |
    | DPO | 53% | -2.78 | -36.8 |
    | DPO plus 0.2 of the SFT loss | 90% | -0.46 | -32.7 |
    | DPO plus 1.0 of the SFT loss | 93% | -0.20 | -29.6 |

    DPO alone did what its loss asks, and opened the gap from 14 to 34. But
    it pushed the right answers down too, so fewer came out on top. Pal and
    colleagues found this when the two answers differ in a few tokens, as
    near misses do.

    Llama 3 adds the SFT loss on the chosen answers, times 0.2. That holds
    the right answer up while DPO pushes the near miss down. The check trains
    your GPT this way. Here every right answer is known, so plain SFT does as
    well. Preferences pay when people can pick the better answer but can't
    write the best one.

    In InstructGPT, RLHF with PPO holds four models at once. They are the
    policy, the reference, the reward model, and a value model started from
    the reward model. It also writes answers during training, which is slow.
    DPO holds two and writes nothing.

    The reward model can work with no training at all, picking the best of
    several answers drawn at a temperature of 1.

    | Answers drawn per question | 1 | 4 | 16 |
    |---|---|---|---|
    | Exact answers, reward1's model picking | 60% | 65% | 47% |

    Best of 4 helps, but best of 16 hurts. The more answers it sees, the more
    chances the reward model gets to pick one it overrates, reward2's hacking
    with no training at all.
    """,
)

DPO2_SOURCE = source_file(
    '''
    # dpo2.py

    import copy
    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    layer = nn.Linear(2, 2)
    frozen = copy.deepcopy(layer).requires_grad_(False)    # its own weights, which never train


    # The GPT from gpt2.
''' + indented_part(GPT_PARTS) + '''


    # The chat format from sft1, grow_vocabulary from sft2, and reward1's pairs.
''' + indented_part(joined(CHAT_PARTS, GROW_SOLUTION, PAIRS_PART)) + '''


    # answer_logprobs, dpo_loss and implicit_rewards from dpo1.
''' + indented_part(DPO1_SOLUTION_PART) + '''


    # TODO: Return a separate copy of model in eval mode, whose weights never
    #   train. model itself stays as it was.
    def frozen_copy(model: nn.Module) -> nn.Module:
        ...


    # TODO: Take one DPO step. chosen and rejected are each a pair of inputs
    #   and targets. Work out reference's log chances of both without recording
    #   for autograd, then policy's. The loss is dpo_loss with beta, plus nll
    #   times answer_loss of policy on chosen. Zero the old gradients,
    #   backpropagate, step, and return the loss as a plain number.
    def dpo_step(
        policy: nn.Module,
        reference: nn.Module,
        optimizer: torch.optim.Optimizer,
        chosen: tuple[Tensor, Tensor],
        rejected: tuple[Tensor, Tensor],
        beta: float,
        nll: float,
    ) -> float:
        ...
    ''',
)

DPO2_SOLUTION = source_file(
    '''
    # dpo2.py, a solution for comparison

    import copy
    import math
    import re

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    from tensorlings.data import load_shakespeare

    layer = nn.Linear(2, 2)
    frozen = copy.deepcopy(layer).requires_grad_(False)


''' + indented_part(GPT_PARTS) + '''


''' + indented_part(joined(CHAT_PARTS, GROW_SOLUTION, PAIRS_PART)) + '''


''' + indented_part(DPO1_SOLUTION_PART) + '''


''' + indented_part(DPO_STEP_SOLUTION) + '''
    ''',
)


def check_frozen_copy(found: object) -> None:
    for make, text in ((small_chat_gpt, "a small GPT"), (lambda: nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5)), "a Linear with dropout")):
        torch.manual_seed(0)
        model = make()
        copied = found(model)
        call = f"frozen_copy of {text}"

        if copied is model or copied is None:
            raise Mismatch(f"{call} should give a new model, a copy, not the model itself.")

        for (name, original), (_, kept) in zip(model.named_parameters(), copied.named_parameters()):
            if not torch.equal(original, kept):
                raise Mismatch(f"{call} changed {name}, but a copy starts with the same weights.")

            if kept.data_ptr() == original.data_ptr():
                raise Mismatch(f"{call} shares {name} with the model, so training the model would move the reference too. Use copy.deepcopy.")

            if kept.requires_grad:
                raise Mismatch(f"{call} left {name} able to train. Call requires_grad_(False) on the copy.")

            if not original.requires_grad:
                raise Mismatch(f"{call} froze the model's own {name}. Freeze the copy alone.")

        if copied.training:
            raise Mismatch(f"{call} is in training mode, but the reference runs in eval mode.")

        if not model.training:
            raise Mismatch(f"{call} put the model itself in eval mode. Only the copy goes to eval mode.")


def check_dpo_step(found: object) -> None:
    for seed, beta, nll in ((891, 0.1, 0.0), (892, 0.5, 0.3)):
        generator = torch.Generator().manual_seed(seed)
        words = ["ab", "ba", "cab", "abc", "bca"]
        policy = small_chat_gpt()
        expected_policy = small_chat_gpt()
        reference = DPO_REFERENCE["frozen_copy"](small_chat_gpt())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
        expected_optimizer = torch.optim.AdamW(expected_policy.parameters(), lr=0.01)
        recording: list[bool] = []
        hook = reference.register_forward_pre_hook(lambda module, inputs: recording.append(torch.is_grad_enabled()))

        try:
            for step in range(2):
                picked = torch.randint(0, len(words), (4,), generator=generator).tolist()
                chats = [(words[position], words[position].upper()) for position in picked]
                misses = [(question, answer[:-1] + "C" if answer[-1] != "C" else answer[:-1] + "A") for question, answer in chats]
                chosen = SFT_REFERENCE["pad_rows"](chat_rows(chats))
                rejected = SFT_REFERENCE["pad_rows"](chat_rows(misses))
                expected = DPO_REFERENCE["dpo_step"](expected_policy, reference, expected_optimizer, chosen, rejected, beta, nll)
                recording.clear()
                actual = found(policy, reference, optimizer, chosen, rejected, beta, nll)

                if any(recording):
                    raise Mismatch("dpo_step ran the reference while recording for autograd. Work out its log chances inside torch.no_grad().")

                expect_close(
                    call=f"dpo_step on a small GPT with beta {beta} and nll {nll}, step {step + 1}",
                    actual=actual,
                    expected=expected,
                    tolerance=1e-5,
                )
        finally:
            hook.remove()

        expect_tensor(
            call="the policy's wte after two calls of dpo_step",
            actual=policy.wte.weight.detach(),
            expected=expected_policy.wte.weight.detach(),
        )


def check_trained_dpo(found: object) -> None:
    policy = sft_start(200)
    before_right, before_wrong = held_out_logprobs(policy)
    reference = DPO_REFERENCE["frozen_copy"](policy)
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    train_words, _ = SFT_REFERENCE["question_words"]()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(100)

    for _ in range(50):
        chosen, rejected = chat_pair_batches(DPO_REFERENCE["preference_pairs"](train_words, 32, generator), positions)
        found(policy, reference, optimizer, chosen, rejected, 0.1, 0.2)

    right, wrong = held_out_logprobs(policy)

    if wrong > before_wrong - 2:
        raise Mismatch(
            f"After 50 steps of dpo_step, the near misses' mean log chance went from {before_wrong:.2f} to {wrong:.2f}, "
            "but DPO should push it down by more than 2.",
        )

    if right < before_right - 1:
        raise Mismatch(
            f"After 50 steps of dpo_step with nll 0.2, the right answers' mean log chance fell from {before_right:.2f} "
            f"to {right:.2f}. The nll term should hold it up.",
        )


def check_dpo2(checker: Checker) -> None:
    checker.test(
        function="frozen_copy",
        description="copies the model, frozen and in eval mode",
        run=check_frozen_copy,
    )
    checker.test(
        function="dpo_step",
        description="takes the DPO loss plus nll times the SFT loss, the reference untracked",
        run=check_dpo_step,
    )
    checker.test(
        function="dpo_step",
        description="trains your gpt3 model away from near misses, the right answers held up",
        run=check_trained_dpo,
    )


DPO2 = Exercise(
    name="dpo2",
    chapter=DPO,
    title="Your GPT trained with DPO",
    lesson=DPO2_LESSON,
    source=DPO2_SOURCE,
    solution=DPO2_SOLUTION,
    hints={
        "frozen_copy": "copy.deepcopy(model), then .eval() and .requires_grad_(False) on the copy, and return it.",
        "dpo_step": "Inside torch.no_grad(), answer_logprobs(reference, *chosen) and the same for rejected. Then the "
        "policy's two, dpo_loss of the four with beta, plus nll * answer_loss(policy, *chosen). zero_grad, backward, "
        "step, and loss.item().",
    },
    introduces=(),
    check=check_dpo2,
    time_limit=120,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# grpo1
# ---------------------------------------------------------------------------

GRPO1_LESSON = prose(
    """
    PPO needs a value model to know what reward to expect, and it is usually
    about as large as the policy. GRPO, from DeepSeekMath, drops it. For each question it writes a
    group of answers and scores each one. The group's mean reward is the
    baseline, as in reinforce2, and each advantage is measured against it.

    ```text
    advantage = (reward - mean of the group) / spread of the group
    ```

    The spread is `std`, PyTorch's standard deviation. A small number, 0.0001
    here, is added to it, since a group of equal rewards has no spread at all.
    Such a group gives every answer an advantage of 0, so it teaches nothing.

    The loss is ppo1's clipped objective on each token, with each answer's
    advantage on all its tokens. It is averaged over each answer's tokens,
    then over the group, and minus that is the loss. The mask marks the
    answer's tokens, so padding counts for nothing.

    The KL penalty goes into the loss, not the reward, with an estimate that
    is never below 0. With `gap` the reference's log chance of a token minus
    the policy's, it is the following, per token.

    ```text
    kl = exp(gap) - gap - 1
    ```

    It is 0 when the two agree and grows either way. GRPO subtracts beta
    times it from each token's objective. DeepSeekMath wrote 64 answers per
    question, with a beta of 0.04 and a rate of 0.000001.

    Here is GRPO on your GPT after 200 steps of sft2's training, 8 questions
    a step with 8 answers each. The reward is 1 for an exact answer and 0
    otherwise, with an epsilon of 0.2 and a beta of 0.04.

    | Steps | 0 | 20 | 40 | 60 |
    |---|---|---|---|---|
    | Exact answers on held out words | 74% | 87% | 87% | 89% |
    | Groups all right or all wrong, in the 20 steps before | | 46% | 55% | 59% |

    Both rows are averages of 3 seeds. GRPO never saw a right answer written
    out, only a check of its own. As the GPT improved, more groups came back
    all right, and those teach nothing. By step 60 most groups gave no
    signal, and the gains slowed.
    """,
)

GRPO1_SOURCE = source_file(
    '''
    # grpo1.py

    import torch
    from torch import Tensor

    rewards = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    means = rewards.mean(dim=1, keepdim=True)              # tensor([[0.7500], [0.0000]])
    spreads = rewards.std(dim=1, keepdim=True)             # tensor([[0.5000], [0.0000]])
    smaller = torch.minimum(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))    # tensor([1., 2.])


    # TODO: Return each answer's advantage within its group. rewards has one
    #   row per question and one column per answer. Subtract each row's mean,
    #   then divide by each row's std plus 0.0001.
    def group_advantages(rewards: Tensor) -> Tensor:
        ...


    # TODO: Return the KL estimate of each token, exp(gap) - gap - 1, where
    #   gap is reference_logprobs minus logprobs.
    def kl_estimate(
        logprobs: Tensor,
        reference_logprobs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return GRPO's loss for one group. logprobs, old_logprobs and
    #   reference_logprobs hold each answer's token log chances, of shape
    #   (answers, tokens), and mask is 1 on the answer's tokens and 0 after.
    #   Per token, take the smaller of ratio times advantage and the clipped
    #   ratio times advantage, as ppo1 did, minus beta times kl_estimate.
    #   Average over each answer's masked tokens, then over the answers, and
    #   return minus that.
    def grpo_loss(
        logprobs: Tensor,
        old_logprobs: Tensor,
        reference_logprobs: Tensor,
        advantages: Tensor,
        mask: Tensor,
        epsilon: float,
        beta: float,
    ) -> Tensor:
        ...
    ''',
)

GRPO1_SOLUTION = source_file(
    '''
    # grpo1.py, a solution for comparison

    import torch
    from torch import Tensor

    rewards = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    means = rewards.mean(dim=1, keepdim=True)
    spreads = rewards.std(dim=1, keepdim=True)
    smaller = torch.minimum(torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0]))


''' + indented_part(GRPO_SOLUTION) + '''
    ''',
)


def check_grpo1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="group_advantages",
        description="measures each reward against its own group",
        inputs=(
            (torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),),
            (torch.tensor([[2.0, -1.0, 0.5], [3.0, 3.0, 3.0]]),),
        ),
        reference=DPO_REFERENCE["group_advantages"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="kl_estimate",
        description="is exp(gap) - gap - 1, never below 0",
        inputs=(
            (torch.tensor([[-1.0, -2.0, -0.5]]), torch.tensor([[-1.0, -1.0, -3.0]])),
            (torch.tensor([[-0.1, -4.0], [-2.0, -2.5]]), torch.tensor([[-0.3, -3.0], [-2.0, -0.5]])),
        ),
        reference=DPO_REFERENCE["kl_estimate"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="grpo_loss",
        description="clips, subtracts the KL, and averages per answer",
        inputs=(
            (
                torch.tensor([[-1.0, -2.0, -0.5], [-0.2, -3.0, -1.0]]),
                torch.tensor([[-1.5, -2.0, -0.1], [-0.2, -2.0, -1.0]]),
                torch.tensor([[-1.2, -1.8, -0.4], [-0.5, -2.5, -1.0]]),
                torch.tensor([1.0, -0.5]),
                torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
                0.2,
                0.04,
            ),
            (
                torch.tensor([[-0.3, -0.9], [-2.0, -0.1], [-1.0, -1.0]]),
                torch.tensor([[-0.6, -0.4], [-1.5, -0.1], [-1.0, -2.0]]),
                torch.tensor([[-0.3, -0.5], [-1.0, -0.2], [-1.2, -1.1]]),
                torch.tensor([-1.0, 2.0, 0.5]),
                torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
                0.1,
                0.5,
            ),
        ),
        reference=DPO_REFERENCE["grpo_loss"],
        expect=expect_tensor,
    )


GRPO1 = Exercise(
    name="grpo1",
    chapter=DPO,
    title="GRPO and group advantages",
    lesson=GRPO1_LESSON,
    source=GRPO1_SOURCE,
    solution=GRPO1_SOLUTION,
    hints={
        "group_advantages": "rewards.mean(dim=1, keepdim=True) and rewards.std(dim=1, keepdim=True) keep one number per "
        "row that broadcasts back. Subtract the mean, then divide by the std plus 1e-4.",
        "kl_estimate": "gap = reference_logprobs - logprobs, then gap.exp() minus gap minus 1.",
        "grpo_loss": "ratio = (logprobs - old_logprobs).exp(). advantages.unsqueeze(1) lines up with the tokens. "
        "torch.minimum of ratio times it and ratio.clamp(1 - epsilon, 1 + epsilon) times it, minus beta * "
        "kl_estimate. Multiply by mask, sum over dim=1, divide by mask.sum(dim=1), take the mean, and negate.",
    },
    introduces=(),
    check=check_grpo1,
)

