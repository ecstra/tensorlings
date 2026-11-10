"""82_dpo, the DPO loss from its paper, the GPT trained with it next to the reward model and PPO route, and GRPO's
advantages taken relative to a group of samples."""

from __future__ import annotations

import contextlib
import copy
import math
import re
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..data import load_shakespeare
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    small_tensor,
    value_text,
)
from ..paths import CHECKPOINTS_DIR
from ..text import prose, source_file
from .cost import tied_pair
from .gpt import MODEL_REFERENCE, trained_gpt
from .reward_models import PAIRS_PART, held_out_pairs, trained_reward_model
from .sft import CHAT_PARTS, GPT_PARTS, GROW_SOLUTION, SFT_REFERENCE, chat_rows, indented_part, joined, small_chat_gpt

DPO = Chapter(
    folder="82_dpo",
    title="Direct preference optimization",
    readme=source_file(
        """
        # Direct preference optimization

        RLHF trains a reward model, then trains the policy with PPO to get high
        rewards, with a penalty for drifting from the reference. Direct
        preference optimization, DPO, reaches the same goal from the preference
        pairs alone. It uses one loss and samples no answers during training.
        This chapter derives the DPO loss from reward2's closed form and trains
        your GPT with it, then compares that with using a reward model. Last,
        it builds GRPO's advantages, which need no value network.

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


def drawn_answers(
    model: nn.Module,
    words: list[str],
    generator: torch.Generator,
) -> dict[str, str | None]:
    """An answer to each word, drawn at a temperature of 1 with the words of one length in one batch. None marks an
    answer that wrote a chat token other than END, which no chat can hold."""
    names, positions = SFT_REFERENCE["chat_vocabulary"]()
    drawn: dict[str, str | None] = {}
    model.eval()

    for length in sorted({len(word) for word in words}):
        group = [word for word in words if len(word) == length]
        ids = torch.tensor([[65] + [positions[character] for character in word] + [66] for word in group])
        written: list[list[int]] = [[] for _ in group]
        done = [False] * len(group)

        with torch.no_grad():
            for _ in range(12):
                tokens = torch.multinomial(torch.softmax(model(ids)[:, -1], dim=-1), 1, generator=generator)

                for place, token in enumerate(tokens.squeeze(1).tolist()):
                    if done[place]:
                        continue

                    if token == 67:
                        done[place] = True
                    else:
                        written[place].append(token)

                ids = torch.cat([ids, tokens], dim=1)

        for word, answer in zip(group, written):
            drawn[word] = "".join(names[token] for token in answer) if all(token < 65 for token in answer) else None

    model.train()

    return drawn


def letters_right(
    written: str,
    right: str,
) -> int:
    """The judge's score, the places that hold the right letter less one for each letter too many or too few."""
    return sum(mine == wanted for mine, wanted in zip(written, right)) - abs(len(written) - len(right))


def judged_pairs(
    model: nn.Module,
    generator: torch.Generator,
) -> list[tuple[str, str, str]]:
    """Two of model's answers to every training word, the one with more letters right chosen and a tie going to the
    first. Two answers that came out the same make no pair."""
    train_words, _ = SFT_REFERENCE["question_words"]()
    first = drawn_answers(model, train_words, generator)
    second = drawn_answers(model, train_words, generator)
    pairs = []

    for word in train_words:
        if first[word] is None or second[word] is None or first[word] == second[word]:
            continue

        if letters_right(second[word], word.upper()) > letters_right(first[word], word.upper()):
            pairs.append((word, second[word], first[word]))
        else:
            pairs.append((word, first[word], second[word]))

    return pairs


def after_preferences(
    model: nn.Module,
    pairs: list[tuple[str, str, str]],
    steps: int,
    route: str,
    seed: int = 0,
) -> nn.Module:
    """model trained steps more steps of 32 pairs, by SFT on the chosen answers alone or by DPO with beta 0.1, with
    no NLL term or with Llama 3's 0.2."""
    _, positions = SFT_REFERENCE["chat_vocabulary"]()
    generator = torch.Generator().manual_seed(100 + seed)

    if route == "sft":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        for _ in range(steps):
            picked = [pairs[place] for place in torch.randint(0, len(pairs), (32,), generator=generator).tolist()]
            chosen, _ = chat_pair_batches(picked, positions)
            SFT_REFERENCE["sft_step"](model, optimizer, *chosen)

        return model

    nll = {"dpo": 0.0, "dpo with nll": 0.2}[route]
    reference = DPO_REFERENCE["frozen_copy"](model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    for _ in range(steps):
        picked = [pairs[place] for place in torch.randint(0, len(pairs), (32,), generator=generator).tolist()]
        chosen, rejected = chat_pair_batches(picked, positions)
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


# The CPU's thread count changes how sums round, which moves dpo2's and grpo1's tables by a point or two.
MEASURE_THREADS = 4


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def measure_dpo() -> dict[str, float]:
    """Every number in dpo2's tables, from a GPT given 200 steps of SFT, averaged over seeds 0 to 2, on 4 threads. The
    pairs are the start's own answers, two to each training word, ranked by letters right. Best of n picks with
    reward1's model after 50 steps and after 1,000."""
    results: dict[str, float] = {}

    with pinned_threads():
        for seed in (0, 1, 2):
            start = sft_start(200, seed)
            state = copy.deepcopy(start.state_dict())
            pairs = judged_pairs(start, torch.Generator().manual_seed(600 + seed))
            chosen_right = sum(chosen == word.upper() for word, chosen, _ in pairs) / len(pairs)
            results["pairs"] = results.get("pairs", 0.0) + len(pairs) / 3
            results["chosen right"] = results.get("chosen right", 0.0) + chosen_right / 3
            print(seed, "pairs", len(pairs), "chosen right", round(chosen_right, 3), flush=True)
            rows = {"start": start}

            for route in ("sft", "dpo", "dpo with nll"):
                model = sft_start(0, seed)
                model.load_state_dict(state)
                rows[route] = after_preferences(model, pairs, 100, route, seed)

            for name, model in rows.items():
                right, wrong = held_out_logprobs(model)
                numbers = {"exact": held_out_score(model), "right logprob": right, "near miss logprob": wrong}

                for key, value in numbers.items():
                    results[f"{name}, {key}"] = results.get(f"{name}, {key}", 0.0) + value / 3

                print(seed, name, {key: round(value, 3) for key, value in numbers.items()}, flush=True)

            for reward_steps in (50, 1000):
                reward_model = trained_reward_model(reward_steps, seed=seed)

                for count in (1, 4, 16):
                    score = best_of(start, reward_model, count, seed)
                    key = f"best of {count}, reward model after {reward_steps} steps"
                    results[key] = results.get(key, 0.0) + score / 3
                    print(seed, key, round(score, 3), flush=True)

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
    1 for an exact answer, epsilon 0.2 and beta 0.04, on 4 threads. Each row is the step, the share of 200 held out
    words answered right, and the share of the step's groups whose rewards were all the same."""
    with pinned_threads():
        return grpo_rows(steps, seed)


def grpo_rows(
    steps: int,
    seed: int,
) -> list[tuple[int, float, float]]:
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
    reward2 found the best policy for a given reward in closed form. DPO,
    from Rafailov and colleagues, turns that formula around. Take logs of it
    and solve for the reward instead of the policy.

    ```text
    r(y) = beta * log(policy(y) / reference(y)) + beta * log Z
    ```

    Put that reward into reward1's Bradley Terry probability. That
    probability depends only on the difference between two rewards, so the
    two `log Z` terms cancel. What is left needs only the policy and the
    reference.

    ```text
    P(chosen beats rejected) = sigmoid(beta * (log ratio of chosen - log ratio of rejected))
    ```

    The log ratio of an answer is the policy's log probability of it minus
    the reference's log probability of it. The DPO loss is minus the log of
    that probability, averaged over the pairs. It trains the policy directly
    on the pairs, with no reward model and no sampling. The paper's title
    says your language model is secretly a reward model. Its implicit reward
    is `beta * (log policy - log reference)`.

    The log probability of an answer is the sum of the log probabilities of
    its tokens, each given the tokens before it. Only the answer's tokens
    count, which are the targets that sft1 did not set to -100. `gather`
    picks one entry from each row, using a tensor of indices.

    ```python
    logs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), dim=-1)
    logs.gather(-1, torch.tensor([[2], [0]]))    # tensor([[-0.4076], [-1.0986]]), entry 2 of row 0, entry 0 of row 1
    ```

    The paper uses a beta of 0.1 unless it says otherwise. The gradient of
    the DPO loss weights each pair by `sigmoid(-gap)`. Here gap is the
    chosen answer's implicit reward minus the rejected answer's. So a pair
    that the policy already ranks correctly by a wide margin counts for
    little.

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


    # TODO: Return the log probability of each row's answer, a tensor of shape
    #   (rows,). x and y are inputs and targets of shape (rows, steps), as
    #   from sft1's pad_rows. Take log_softmax of the model's scores on x, and
    #   use gather to pick the entry of each target. Sum those over the
    #   positions whose target is not -100. gather cannot use -100 as an
    #   index, so replace those targets with 0 in a new tensor before
    #   gathering. Do not change y itself.
    def answer_logprobs(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the DPO loss, as a tensor with one number. The log ratio of
    #   an answer is the policy's log probability minus the reference's. For
    #   each pair, take beta times the chosen log ratio minus the rejected log
    #   ratio, and take its log sigmoid. Return minus the mean over the pairs.
    #   Example: dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]),
    #   torch.tensor([-2.0]), torch.tensor([-2.0]), 1.0) returns tensor(0.1269).
    def dpo_loss(
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
        beta: float,
    ) -> Tensor:
        ...


    # TODO: Return the implicit reward of each answer, which is beta times the
    #   policy's log probability minus the reference's log probability.
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

        targets = y.clone()
        actual = found(model, x, y)
        call = f"answer_logprobs of a small GPT on {len(chats)} padded chats"
        expect_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )

        if not actual.requires_grad:
            raise Mismatch(f"{call} returns log probabilities that autograd did not record, but DPO trains through them. Remove torch.no_grad here.")

        if not torch.equal(y, targets):
            raise Mismatch(
                f"{call} changed y, the caller's targets, so a later loss on the same batch would count the wrong "
                "tokens. Replace the -100s in a new tensor, which masked_fill returns.",
            )


def check_dpo1(checker: Checker) -> None:
    checker.test(
        function="answer_logprobs",
        description="sums the log probabilities of the answer's tokens",
        run=check_answer_logprobs,
    )
    check_on_inputs(
        checker=checker,
        function="dpo_loss",
        description="is minus the mean log sigmoid of beta times the difference of log ratios",
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
        "answer_logprobs": "Take F.log_softmax of model(x) over dim=-1, and make a mask kept = y != -100. Gather with "
        "y.masked_fill(~kept, 0), unsqueezed at the end, then squeeze the result back. Multiply by kept and sum over "
        "dim=1.",
        "dpo_loss": "The gap is (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected). Take "
        "F.logsigmoid of beta times the gap, then the mean, and put a minus in front.",
        "implicit_rewards": "Multiply beta by policy_logprobs - reference_logprobs.",
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
    DPO needs a reference, which is the model as it was before DPO, kept
    fixed during training. `copy.deepcopy` makes a separate copy of a model.
    `requires_grad_(False)` stops every one of the copy's weights from
    training.

    ```python
    layer = nn.Linear(2, 2)
    frozen = copy.deepcopy(layer).requires_grad_(False)    # a copy with its own weights, which never train
    ```

    Compute the reference's log probabilities inside `torch.no_grad()`. Its
    weights are frozen, so autograd would record nothing anyway. But
    `torch.no_grad()` makes that explicit in the code. It also keeps the
    reference out of autograd if someone forgot to freeze it.

    Here DPO starts from gpt3's GPT after 200 steps of sft2's training. The
    pairs come from the model's own answers. It answers every training word
    twice, sampling at a temperature of 1. A judge counts the letters in the
    right places, and the answer with more of them is chosen. Two identical
    answers make no pair. That left about 3,600 pairs. About half of their
    chosen answers are right, and the rest are the better of two wrong
    answers.

    Each run trains for 100 more steps of 32 pairs, with a beta of 0.1.
    Every number is the average of 3 seeds, measured on 200 held out words.

    | 100 more steps of | Exact answers | Right answer's log probability |
    |---|---|---|
    | nothing, the start | 74% | -1.08 |
    | SFT on the chosen answers | 73% | -1.34 |
    | DPO | 46% | -3.71 |
    | DPO plus 0.2 of the SFT loss | 83% | -0.76 |

    SFT learns to copy every chosen answer, so it learns their mistakes too,
    and ends about where it began. DPO learns what makes the chosen answer
    better than the rejected one. A wrong answer is pushed down each time it
    loses, even when it loses to another wrong answer. Per seed, SFT scored
    81%, 75.5% and 62%, and DPO with the SFT term scored 81%, 89% and 80%.

    DPO alone also lowered the probability of the right answers, so the
    right answer came out on top less often. Pal and colleagues found this
    happens when the two answers differ in only a few tokens, as these do.
    Llama 3 adds the SFT loss on the chosen answers, multiplied by 0.2,
    which keeps their probability up. The check trains your GPT this way, on
    reward1's pairs.

    In InstructGPT, RLHF with PPO keeps four models in memory at once. They
    are the policy, the reference, the reward model, and a value network
    like ppo2's, initialized from the reward model. PPO also samples answers
    during training, which is slow. DPO keeps two models, and here the
    answers were sampled once, before training.

    The reward model can also be used without training the policy at all.
    Sample several answers at a temperature of 1, and let the reward model
    pick the best one. The start's 74% came from always writing the most
    likely character. A single sampled answer is right less often.

    | Answers sampled per question | 1 | 4 | 16 |
    |---|---|---|---|
    | Exact answers, picked by reward1's model after 50 steps | 60% | 65% | 47% |
    | Exact answers, picked by reward1's model after 1,000 steps | 60% | 69% | 57.5% |

    On average, picking the best of 4 beats a single sampled answer, but
    picking from 16 does worse. The more answers the reward model sees, the
    more chances it has to pick one it scores too high. That is the reward
    hacking of reward2, without any training of the policy. Training the
    reward model 20 times longer makes the drop smaller but does not stop
    it. No pick reaches the start's 74%.
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
    frozen = copy.deepcopy(layer).requires_grad_(False)    # a copy with its own weights, which never train


    # The GPT from gpt2.
''' + indented_part(GPT_PARTS) + '''


    # The chat format from sft1, grow_vocabulary from sft2, and reward1's pairs.
''' + indented_part(joined(CHAT_PARTS, GROW_SOLUTION, PAIRS_PART)) + '''


    # answer_logprobs, dpo_loss and implicit_rewards from dpo1.
''' + indented_part(DPO1_SOLUTION_PART) + '''


    # TODO: Return a separate copy of model, in eval mode, whose weights never
    #   train. Do not change model itself.
    def frozen_copy(model: nn.Module) -> nn.Module:
        ...


    # TODO: Take one DPO step. chosen and rejected are each a pair (x, y) of
    #   inputs and targets. Compute the reference's log probabilities of both
    #   answers with answer_logprobs, inside torch.no_grad(). Then compute the
    #   policy's log probabilities of both, with autograd on. The loss is
    #   dpo_loss with beta, plus nll times answer_loss of policy on chosen.
    #   Zero the old gradients, backpropagate, call the optimizer's step, and
    #   return the loss as a Python float.
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
            raise Mismatch(f"{call} should return a new model, a copy, not the model itself.")

        for (name, original), (_, kept) in zip(model.named_parameters(), copied.named_parameters()):
            if not torch.equal(original, kept):
                raise Mismatch(f"{call} changed {name}, but a copy starts with the same weights.")

            if kept.data_ptr() == original.data_ptr():
                raise Mismatch(f"{call} shares {name} with the model, so training the model would change the reference too. Use copy.deepcopy.")

            if kept.requires_grad:
                raise Mismatch(f"{call} left {name} able to train. Call requires_grad_(False) on the copy.")

            if not original.requires_grad:
                raise Mismatch(f"{call} froze the model's own {name}. Freeze only the copy.")

        if copied.training:
            raise Mismatch(f"{call} is in training mode, but the reference should be in eval mode.")

        if not model.training:
            raise Mismatch(f"{call} put the model itself in eval mode. Only the copy should be in eval mode.")


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
                    raise Mismatch(
                        "dpo_step ran the reference with autograd on. The TODO asks for its log probabilities inside "
                        "torch.no_grad(), which keeps the reference out of the computation graph even when it is not "
                        "frozen.",
                    )

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


# dpo2's last test starts from 200 steps of SFT on the learner's gpt3 model. They run once per process and per
# checkpoint, before the test's clock starts, since under the clock's trace they could pass the limit on a slow laptop.
SFT_STARTS: dict[int, nn.Module | Mismatch] = {}


def checkpoint_key() -> int:
    path = CHECKPOINTS_DIR / "gpt3.pt"

    return path.stat().st_mtime_ns if path.exists() else 0


def build_sft_start() -> None:
    key = checkpoint_key()

    if key in SFT_STARTS:
        return

    try:
        SFT_STARTS[key] = sft_start(200)
    except Mismatch as missing:
        SFT_STARTS[key] = missing


def fresh_sft_start() -> nn.Module:
    build_sft_start()
    start = SFT_STARTS[checkpoint_key()]

    if isinstance(start, Mismatch):
        raise Mismatch(str(start))

    return copy.deepcopy(start)


def check_trained_dpo(found: object) -> None:
    policy = fresh_sft_start()
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
            f"After 50 steps of dpo_step, the near misses' mean log probability went from {before_wrong:.2f} to {wrong:.2f}, "
            "but DPO should push it down by more than 2.",
        )

    if right < before_right - 1:
        raise Mismatch(
            f"After 50 steps of dpo_step with nll 0.2, the right answers' mean log probability fell from {before_right:.2f} "
            f"to {right:.2f}. The nll term should keep it from falling by more than 1.",
        )


def check_dpo2(checker: Checker) -> None:
    checker.test(
        function="frozen_copy",
        description="copies the model, frozen and in eval mode",
        run=check_frozen_copy,
    )
    checker.test(
        function="dpo_step",
        description="steps on the DPO loss plus nll times the SFT loss, with the reference under no_grad",
        run=check_dpo_step,
    )

    if callable(getattr(checker.module, "dpo_step", None)):
        build_sft_start()

    checker.test(
        function="dpo_step",
        description="trains your gpt3 model away from near misses while keeping the right answers likely",
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
        "frozen_copy": "Call copy.deepcopy(model), then call .eval() and .requires_grad_(False) on the copy, and return "
        "it.",
        "dpo_step": "Inside torch.no_grad(), compute answer_logprobs(reference, *chosen) and the same for rejected. Then "
        "compute the policy's two outside it. The loss is dpo_loss of the four with beta, plus "
        "nll * answer_loss(policy, *chosen). Call zero_grad, backward and step, and return loss.item().",
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
    PPO needs a value network, ppo2's V, to predict what reward to expect.
    That network is usually about as large as the policy. GRPO, group
    relative policy optimization, from DeepSeekMath, removes it. For each
    question, GRPO samples a group of answers and scores each one. The
    group's mean reward is the baseline, as in reinforce2. Each advantage is
    measured relative to that mean.

    ```text
    advantage = (reward - mean of the group) / standard deviation of the group
    ```

    The standard deviation here is the default `.std()`, which divides by
    n - 1. batchnorm1's variance divided by n instead. A small number,
    0.0001 here, is added to the standard deviation, because a group of
    equal rewards has a standard deviation of 0. Such a group gives every
    answer an advantage of 0, so it teaches nothing.

    The loss applies ppo1's clipped objective to each token, and every token
    of an answer gets that answer's advantage. The objective is averaged
    over each answer's tokens, then over the group. The loss is minus that
    average. The mask marks the answer's tokens, so padding adds nothing.

    GRPO puts the KL penalty into the loss instead of the reward. It uses an
    estimate of the KL divergence that is never below 0. Let `gap` be the
    reference's log probability of a token minus the policy's. The estimate
    for that token is this.

    ```text
    kl = exp(gap) - gap - 1
    ```

    It is 0 when the two log probabilities are equal, and it grows as they
    move apart in either direction. GRPO subtracts beta times it from each
    token's objective. DeepSeekMath sampled 64 answers per question, with a
    beta of 0.04 and a learning rate of 0.000001.

    The table runs GRPO on your GPT after 200 steps of sft2's training. Each
    step uses 8 questions with 8 answers each. The reward is 1 for an exact
    answer and 0 otherwise, with an epsilon of 0.2 and a beta of 0.04.

    | Steps | 0 | 20 | 40 | 60 |
    |---|---|---|---|---|
    | Exact answers on held out words | 74% | 87% | 87% | 89% |
    | Groups all right or all wrong, over the previous 20 steps | | 46% | 55% | 59% |

    Both rows are averages over 3 seeds. GRPO never saw a right answer
    written out. It only saw the reward from checking its own answers. As
    the GPT improved, more groups came back all right, and those groups
    teach nothing. By step 60 most groups gave no learning signal, and the
    improvement slowed.

    dpo2 started from the same GPT and tested on the same 200 words. There
    DPO with the SFT term reached 83%, and SFT on the chosen answers reached
    73%. GRPO needs an automatic check it can run on any answer it writes,
    and it samples answers throughout training. DPO needs only pairs that
    were judged once. So DPO fits tasks where answers can be compared but
    not checked automatically.
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


    # TODO: Return each answer's advantage within its group, a tensor with the
    #   shape of rewards. rewards has one row per question and one column per
    #   answer. Subtract each row's mean, then divide by that row's standard
    #   deviation plus 0.0001.
    #   Example: group_advantages(torch.tensor([[1.0, 0.0, 1.0, 1.0]])) returns
    #   tensor([[ 0.4999, -1.4997,  0.4999,  0.4999]]).
    def group_advantages(rewards: Tensor) -> Tensor:
        ...


    # TODO: Return the KL estimate of each token, exp(gap) - gap - 1, where
    #   gap is reference_logprobs minus logprobs.
    #   Example: kl_estimate(torch.tensor([-1.0, -2.0]), torch.tensor([-1.0, -1.0]))
    #   returns tensor([0.0000, 0.7183]).
    def kl_estimate(
        logprobs: Tensor,
        reference_logprobs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return GRPO's loss for one group, as a tensor with one number.
    #   logprobs, old_logprobs and reference_logprobs hold the log probability
    #   of each token of each answer, with shape (answers, tokens). advantages
    #   has one entry per answer, used for every token of that answer. mask is
    #   1 on the answer's tokens and 0 after them. For each token, take the
    #   smaller of ratio * advantage and clipped_ratio * advantage, as in
    #   ppo1, then subtract beta * kl_estimate. Average over each answer's
    #   masked tokens, then over the answers, and return minus that average.
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
        "group_advantages": "rewards.mean(dim=1, keepdim=True) and rewards.std(dim=1, keepdim=True) give one number per "
        "row, in a shape that broadcasts back over the row. Subtract the mean, then divide by the standard deviation "
        "plus 1e-4.",
        "kl_estimate": "The gap is the reference's log probabilities minus the policy's. Then compute gap.exp() minus "
        "gap minus 1.",
        "grpo_loss": "The ratio is .exp() of the log probabilities minus the old log probabilities. "
        "advantages.unsqueeze(1) has shape (answers, 1), which broadcasts over the tokens. Take torch.minimum of ratio "
        "times it and ratio.clamp(1 - epsilon, 1 + epsilon) times it, then subtract beta * kl_estimate. Multiply by "
        "mask, sum over dim=1, divide by mask.sum(dim=1), take the mean, and negate.",
    },
    introduces=(),
    check=check_grpo1,
)


# ---------------------------------------------------------------------------
# quiz14
# ---------------------------------------------------------------------------

QUIZ14_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines steps
    from Section 11. If a step is unclear, the exercise named in the table
    shows how to do it.

    | Function | What it computes | Shown in |
    |---|---|---|
    | `trainable_share` | the count of numbers in parameters that require a gradient, divided by the count of all numbers | finetune2 |
    | `merged_weight` | the weight plus the update `B @ A`, scaled by alpha divided by the rank | lora2 |
    | `answer_only_targets` | the ids shifted by one position, with every target before the answer set to -100 | sft1 |
    | `baseline_loss` | REINFORCE's loss, with the batch's mean reward subtracted from each reward | reinforce2 |
    | `ppo_loss_from_rewards` | PPO's clipped loss, with each reward minus the batch's mean reward as its advantage | reinforce2, ppo1 |
    | `dpo_loss_from_tokens` | each answer's token log probabilities summed over its mask, then the DPO loss | dpo1 |
    | `exact_match_advantages` | a reward of 1 for an exact answer and 0 otherwise, then GRPO's group advantages | sft2, grpo1 |
    """,
)

QUIZ14_SOURCE = source_file(
    '''
    # quiz14.py
    #
    # No examples and no new tools. Each function combines steps from Section 11.

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the fraction of model's numbers that will train, as a Python
    #   float. Those are the numbers in parameters that require a gradient.
    def trainable_share(model: nn.Module) -> float:
        ...


    # TODO: Return weight with LoRA's update merged into it. A has shape
    #   (rank, inputs), B has shape (outputs, rank), and the update B @ A is
    #   scaled by alpha divided by the rank.
    def merged_weight(
        weight: Tensor,
        A: Tensor,
        B: Tensor,
        alpha: float,
    ) -> Tensor:
        ...


    # TODO: Return the targets for one chat, given its ids as a 1D tensor.
    #   Target i is the id that follows position i, so there is one target
    #   fewer than ids. Set every target before the answer to -100. The
    #   answer's first id is ids[start]. Keep the targets as integers, and
    #   leave ids itself unchanged.
    def answer_only_targets(
        ids: Tensor,
        start: int,
    ) -> Tensor:
        ...


    # TODO: Return the REINFORCE loss with the batch's mean reward as the
    #   baseline. log_probs and rewards have one entry per episode.
    def baseline_loss(
        log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return PPO's clipped loss, averaged over the actions. You get the
    #   new and old log probability of each action, and its reward. Each
    #   action's advantage is its reward minus the mean of all the rewards.
    def ppo_loss_from_rewards(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
        rewards: Tensor,
        epsilon: float,
    ) -> Tensor:
        ...


    # TODO: Return the DPO loss, averaged over the pairs. The first four
    #   arguments hold the log probability of each token, with shape
    #   (pairs, tokens), for the chosen and rejected answers under the policy
    #   and the reference. A mask is 1 on an answer's tokens and 0 on its
    #   padding. First sum each answer's masked token log probabilities to get
    #   the answer's log probability.
    def dpo_loss_from_tokens(
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
        chosen_mask: Tensor,
        rejected_mask: Tensor,
        beta: float,
    ) -> Tensor:
        ...


    # TODO: Return each answer's advantage within its group, as a tensor with
    #   one row per question. answers holds each question's group of answers,
    #   and right holds each question's right answer. An answer's reward is
    #   1.0 when it equals the right answer and 0.0 otherwise. Subtract each
    #   row's mean, then divide by the row's standard deviation plus 1e-4.
    def exact_match_advantages(
        answers: list[list[str]],
        right: list[str],
    ) -> Tensor:
        ...
    ''',
)

QUIZ14_SOLUTION = source_file(
    '''
    # quiz14.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def trainable_share(model: nn.Module) -> float:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

        return trainable / total


    def merged_weight(
        weight: Tensor,
        A: Tensor,
        B: Tensor,
        alpha: float,
    ) -> Tensor:
        return weight + alpha / A.shape[0] * B @ A


    def answer_only_targets(
        ids: Tensor,
        start: int,
    ) -> Tensor:
        targets = ids[1:].clone()
        targets[: start - 1] = -100

        return targets


    def baseline_loss(
        log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        return -(log_probs * (rewards - rewards.mean())).mean()


    def ppo_loss_from_rewards(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
        rewards: Tensor,
        epsilon: float,
    ) -> Tensor:
        advantages = rewards - rewards.mean()
        ratio = (new_log_probs - old_log_probs).exp()
        clipped = ratio.clamp(1 - epsilon, 1 + epsilon)

        return -torch.min(ratio * advantages, clipped * advantages).mean()


    def dpo_loss_from_tokens(
        policy_chosen: Tensor,
        policy_rejected: Tensor,
        reference_chosen: Tensor,
        reference_rejected: Tensor,
        chosen_mask: Tensor,
        rejected_mask: Tensor,
        beta: float,
    ) -> Tensor:
        chosen_ratio = ((policy_chosen - reference_chosen) * chosen_mask).sum(dim=1)
        rejected_ratio = ((policy_rejected - reference_rejected) * rejected_mask).sum(dim=1)

        return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()


    def exact_match_advantages(
        answers: list[list[str]],
        right: list[str],
    ) -> Tensor:
        rewards = torch.tensor([[float(answer == wanted) for answer in group] for group, wanted in zip(answers, right)])

        return (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-4)
    ''',
)


def quiz14_model(
    seed: int,
    frozen: tuple[int, ...],
) -> nn.Module:
    """A small stack of Linears with the ones at the places in frozen needing no gradient."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 6), nn.Linear(6, 2, bias=False))

    for place in frozen:
        model[place].requires_grad_(False)

    return model


def quiz14_tied_model() -> nn.Module:
    """cost1's tied Embedding and Linear, then a frozen Linear, so a count over state_dict() counts the table twice."""
    model = nn.Sequential(*tied_pair(), nn.Linear(10, 3))
    model[2].requires_grad_(False)

    return model


def reference_trainable_share(model: nn.Module) -> float:
    total = sum(parameter.numel() for parameter in model.parameters())

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / total


def reference_answer_only_targets(
    ids: Tensor,
    start: int,
) -> Tensor:
    targets = ids[1:].clone()
    targets[: start - 1] = -100

    return targets


def expect_whole_targets(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dtype != torch.long:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but F.cross_entropy needs integer targets, torch.long, like ids.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_answer_only_targets(found: object) -> None:
    for ids, start in ((torch.tensor([5, 1, 2, 7, 3, 4, 9]), 4), (torch.tensor([8, 0, 6, 6, 2]), 2)):
        given = ids.clone()
        call = f"answer_only_targets({value_text(ids)}, {start})"
        actual = found(given, start)

        if not torch.equal(given, ids):
            raise Mismatch(
                f"{call} changed ids, the caller's tensor, so the chat it came from now holds -100s. ids[1:] shares "
                "memory with ids, so .clone() it before writing -100.",
            )

        expect_whole_targets(
            call=call,
            actual=actual,
            expected=reference_answer_only_targets(ids, start),
        )


def reference_ppo_loss_from_rewards(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    epsilon: float,
) -> Tensor:
    advantages = rewards - rewards.mean()
    ratio = (new_log_probs - old_log_probs).exp()

    return -torch.min(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


def reference_dpo_loss_from_tokens(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    chosen_mask: Tensor,
    rejected_mask: Tensor,
    beta: float,
) -> Tensor:
    return DPO_REFERENCE["dpo_loss"](
        (policy_chosen * chosen_mask).sum(dim=1),
        (policy_rejected * rejected_mask).sum(dim=1),
        (reference_chosen * chosen_mask).sum(dim=1),
        (reference_rejected * rejected_mask).sum(dim=1),
        beta,
    )


def reference_exact_match_advantages(
    answers: list[list[str]],
    right: list[str],
) -> Tensor:
    rewards = torch.tensor([[float(answer == wanted) for answer in group] for group, wanted in zip(answers, right)])

    return DPO_REFERENCE["group_advantages"](rewards)


def check_quiz14(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="trainable_share",
        description="counts the numbers that will train",
        inputs=(
            (Described(quiz14_model(980, (0,)), "Linears 3 to 4, 4 to 6 and 6 to 2, the first frozen"),),
            (Described(quiz14_model(981, (2, 3)), "Linears 3 to 4, 4 to 6 and 6 to 2, the last two frozen"),),
            (Described(quiz14_tied_model(), "an Embedding(10, 4), a Linear sharing its weight, and a frozen Linear from 10 to 3"),),
        ),
        reference=reference_trainable_share,
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="merged_weight",
        description="adds the scaled update B @ A",
        inputs=(
            (small_tensor(982, 4, 3), small_tensor(983, 2, 3), small_tensor(984, 4, 2), 4.0),
            (small_tensor(985, 2, 5), small_tensor(986, 1, 5), small_tensor(987, 2, 1), 3.0),
        ),
        reference=lambda weight, A, B, alpha: weight + alpha / A.shape[0] * B @ A,
        expect=expect_tensor,
    )
    checker.test(
        function="answer_only_targets",
        description="shifts the ids by one and masks the question, leaving ids unchanged",
        run=check_answer_only_targets,
    )
    check_on_inputs(
        checker=checker,
        function="baseline_loss",
        description="weights each log probability by its reward minus the mean",
        inputs=(
            (torch.tensor([-0.5, -1.2, -0.1, -2.0]), torch.tensor([1.0, 0.0, 1.0, 0.0])),
            (torch.tensor([-0.3, -0.9, -1.5]), torch.tensor([3.0, 5.0, 4.0])),
        ),
        reference=lambda log_probs, rewards: -(log_probs * (rewards - rewards.mean())).mean(),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="ppo_loss_from_rewards",
        description="clips the ratio, with each reward minus the mean as its advantage",
        inputs=(
            (torch.tensor([-0.2, -1.5, -0.7, -0.1]), torch.tensor([-0.6, -1.0, -0.7, -0.9]), torch.tensor([1.0, 0.0, 1.0, 0.0]), 0.2),
            (torch.tensor([-1.0, -0.4, -2.2]), torch.tensor([-1.3, -0.3, -1.0]), torch.tensor([3.0, 5.0, 4.0]), 0.1),
        ),
        reference=reference_ppo_loss_from_rewards,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="dpo_loss_from_tokens",
        description="sums each answer's masked tokens, then compares the log ratios",
        inputs=(
            (
                torch.tensor([[-1.0, -2.0, -0.5], [-0.3, -1.2, -4.0]]),
                torch.tensor([[-2.0, -0.7, -3.0], [-1.5, -2.5, -0.9]]),
                torch.tensor([[-1.2, -1.8, -0.9], [-0.6, -1.0, -2.0]]),
                torch.tensor([[-1.5, -1.0, -2.0], [-1.0, -2.0, -1.5]]),
                torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
                torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
                0.1,
            ),
            (
                torch.tensor([[-0.4, -3.0], [-1.0, -0.2], [-2.0, -2.5]]),
                torch.tensor([[-0.9, -1.0], [-3.0, -0.5], [-0.5, -6.0]]),
                torch.tensor([[-0.6, -2.0], [-1.1, -0.8], [-1.0, -2.0]]),
                torch.tensor([[-0.8, -2.0], [-2.0, -0.4], [-0.6, -1.0]]),
                torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
                torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
                0.5,
            ),
        ),
        reference=reference_dpo_loss_from_tokens,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="exact_match_advantages",
        description="rewards exact answers, then measures each against its own group",
        inputs=(
            ([["CAT", "CT", "CAT", "CAAT"], ["SKY", "SKY", "SKY", "SKY"]], ["CAT", "SKY"]),
            ([["AB", "BA", "AB"], ["ABC", "AB", "ABC"], ["BC", "BC", "AB"]], ["BA", "AB", "XY"]),
        ),
        reference=reference_exact_match_advantages,
        expect=expect_tensor,
    )


QUIZ14 = Exercise(
    name="quiz14",
    chapter=QUIZZES,
    title="Post training, start to finish",
    lesson=QUIZ14_LESSON,
    source=QUIZ14_SOURCE,
    solution=QUIZ14_SOLUTION,
    hints={
        "trainable_share": "Sum numel() over model.parameters() for the total, and again over the ones whose "
        "requires_grad is True. Divide the second by the first.",
        "merged_weight": "The rank is A.shape[0]. B @ A has weight's shape, so scale it by alpha divided by the rank and "
        "add it to weight.",
        "answer_only_targets": "ids[1:] gives the next id at every position. Clone it, then set every target before "
        "index start - 1 to -100.",
        "baseline_loss": "Subtract rewards.mean() from every reward, multiply by log_probs, take the mean, and negate.",
        "ppo_loss_from_rewards": "The advantages are rewards - rewards.mean(). The ratio is the exp of new minus old. "
        "Clamp it to the range from 1 - epsilon to 1 + epsilon. Take torch.min of the two ratios times the "
        "advantages, then the mean, and negate.",
        "dpo_loss_from_tokens": "Multiply each tensor of token log probabilities by its mask and sum over dim=1, which "
        "gives one log probability per answer. Then compute the chosen log ratio minus the rejected log ratio, times "
        "beta. Pass that through F.logsigmoid, take the mean, and negate.",
        "exact_match_advantages": "Build rewards with torch.tensor of a list of rows, with float(answer == wanted) for "
        "each answer of a group. Then mean(dim=1, keepdim=True) and std(dim=1, keepdim=True) give one number per row "
        "that broadcasts back over the row.",
    },
    introduces=(),
    check=check_quiz14,
)

