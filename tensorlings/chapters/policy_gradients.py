"""79_policy_gradients, the words of reinforcement learning on a bandit, REINFORCE, a baseline that cuts its noise, and why a score with no gradient can still train a policy."""

from __future__ import annotations

import textwrap

import torch
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    seeded,
)
from ..text import prose, source_file

POLICY_GRADIENTS = Chapter(
    folder="79_policy_gradients",
    title="Policy gradients",
    readme=source_file(
        """
        # Policy gradients

        Reinforcement learning trains a model from scores instead of right
        answers. This chapter introduces its standard terms on a row of slot
        machines. Then it derives REINFORCE, the classic method for following
        the gradient of the expected score. A baseline makes REINFORCE's
        gradient estimate less noisy. Finally, a model learns to spell a hidden
        word from a score computed with `==`, which shows that REINFORCE needs
        no gradient from the score at all.

        ## Further reading

        - [Williams, simple statistical gradient following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696)
        - [Sutton and Barto, Reinforcement Learning, an Introduction, chapters 2 and 13](http://incompleteideas.net/book/the-book-2nd.html)
        - [OpenAI Spinning Up, intro to policy optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
        """,
    ),
)

# The pieces later exercises and chapters are handed, written once so the files never drift apart.
BANDIT_PART = '''
ARM_MEANS = torch.tensor([0.2, -0.5, 1.0, 0.4, -0.1])
'''

PLAY_PART = '''
def play(
    preferences: Tensor,
    arm_means: Tensor,
    episodes: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    chances = torch.softmax(preferences, dim=0)
    actions = torch.multinomial(chances, episodes, replacement=True, generator=generator)
    rewards = arm_means[actions] + torch.randn(episodes, generator=generator)

    return actions, rewards
'''

EXPECTED_PART = '''
def expected_reward(
    preferences: Tensor,
    arm_means: Tensor,
) -> Tensor:
    return (torch.softmax(preferences, dim=0) * arm_means).sum()
'''

AVERAGE_PART = '''
def average_reward(
    preferences: Tensor,
    arm_means: Tensor,
    episodes: int,
    generator: torch.Generator,
) -> float:
    _, rewards = play(preferences, arm_means, episodes, generator)

    return rewards.mean().item()


def best_arm_share(
    actions: Tensor,
    arm_means: Tensor,
) -> float:
    return (actions == arm_means.argmax()).float().mean().item()
'''

LOG_PROB_PART = '''
def log_probabilities(
    preferences: Tensor,
    actions: Tensor,
) -> Tensor:
    return torch.log_softmax(preferences, dim=0)[actions]
'''

REINFORCE_LOSS_PART = '''
def reinforce_loss(
    log_probs: Tensor,
    rewards: Tensor,
) -> Tensor:
    return -(log_probs * rewards).mean()
'''

TRAIN_BANDIT_PART = '''
def train_bandit(
    arm_means: Tensor,
    steps: int,
    episodes: int,
    lr: float,
    generator: torch.Generator,
) -> Tensor:
    preferences = torch.zeros(len(arm_means), requires_grad=True)
    optimizer = torch.optim.SGD([preferences], lr=lr)

    for _ in range(steps):
        actions, rewards = play(preferences, arm_means, episodes, generator)
        loss = reinforce_loss(log_probabilities(preferences, actions), rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return preferences.detach()
'''

ADVANTAGE_PART = '''
def advantages(rewards: Tensor) -> Tensor:
    return rewards - rewards.mean()
'''

SAMPLES_PART = '''
def gradient_samples(
    preferences: Tensor,
    arm_means: Tensor,
    episodes: int,
    trials: int,
    baseline: bool,
    generator: torch.Generator,
) -> Tensor:
    samples = []

    for _ in range(trials):
        leaf = preferences.clone().requires_grad_()
        actions, rewards = play(leaf, arm_means, episodes, generator)
        weights = advantages(rewards) if baseline else rewards
        reinforce_loss(log_probabilities(leaf, actions), weights).backward()
        samples.append(-leaf.grad)

    return torch.stack(samples)


def gradient_noise(samples: Tensor) -> float:
    return samples.std(dim=0).mean().item()
'''

TRAIN_BASELINE_PART = '''
def train_bandit(
    arm_means: Tensor,
    steps: int,
    episodes: int,
    lr: float,
    baseline: bool,
    generator: torch.Generator,
) -> Tensor:
    preferences = torch.zeros(len(arm_means), requires_grad=True)
    optimizer = torch.optim.SGD([preferences], lr=lr)

    for _ in range(steps):
        actions, rewards = play(preferences, arm_means, episodes, generator)
        weights = advantages(rewards) if baseline else rewards
        loss = reinforce_loss(log_probabilities(preferences, actions), weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return preferences.detach()
'''

TRAIN_BASELINE_STUB = TRAIN_BASELINE_PART.replace("advantages(rewards) if baseline else rewards", "...")

EXACT_PART = '''
def exact_gradient(
    preferences: Tensor,
    arm_means: Tensor,
) -> Tensor:
    leaf = preferences.clone().requires_grad_()
    expected_reward(leaf, arm_means).backward()

    return leaf.grad


def estimate_gap(
    preferences: Tensor,
    arm_means: Tensor,
    episodes: int,
    generator: torch.Generator,
) -> float:
    estimate = gradient_samples(preferences, arm_means, episodes, 1, False, generator)[0]

    return (estimate - exact_gradient(preferences, arm_means)).abs().max().item()
'''

SPELLER_PART = '''
LETTERS = 26


def spell(
    logits: Tensor,
    batch: int,
    generator: torch.Generator,
) -> Tensor:
    chances = torch.softmax(logits, dim=1)

    return torch.multinomial(chances, batch, replacement=True, generator=generator).T


def matches(
    words: Tensor,
    target: Tensor,
) -> Tensor:
    return (words == target).sum(dim=1).float()
'''

SPELLER_SOLUTION = '''
def word_log_probs(
    logits: Tensor,
    words: Tensor,
) -> Tensor:
    return torch.log_softmax(logits, dim=1)[torch.arange(len(logits)), words].sum(dim=1)


def train_speller(
    target: Tensor,
    steps: int,
    batch: int,
    lr: float,
    generator: torch.Generator,
) -> Tensor:
    logits = torch.zeros(len(target), LETTERS, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    for _ in range(steps):
        words = spell(logits, batch, generator)
        rewards = matches(words, target)
        loss = reinforce_loss(word_log_probs(logits, words), advantages(rewards))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return logits.detach()
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace(*parts: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "Tensor": Tensor,
    }
    exec("\n\n".join(parts), namespace)

    return namespace


REFERENCE = reference_namespace(
    BANDIT_PART,
    PLAY_PART,
    EXPECTED_PART,
    AVERAGE_PART,
    LOG_PROB_PART,
    REINFORCE_LOSS_PART,
    ADVANTAGE_PART,
    SAMPLES_PART,
    TRAIN_BASELINE_PART,
    EXACT_PART,
    SPELLER_PART,
    SPELLER_SOLUTION,
)
FIRST_TRAIN = reference_namespace(PLAY_PART, LOG_PROB_PART, REINFORCE_LOSS_PART, TRAIN_BANDIT_PART)

ARM_MEANS = REFERENCE["ARM_MEANS"]
SHIFTED_MEANS = ARM_MEANS + 10
TORCH = torch.tensor([19, 14, 17, 2, 7])


def measure_policy_gradients() -> dict[str, object]:
    """Every number of this chapter's lessons. A few seconds on a laptop CPU. Not run by any check."""
    seeds = range(5)
    results: dict[str, object] = {}

    # A seed per row, so the rows don't share one draw of noise.
    for name, preferences, seed in (
        ("all 0", torch.zeros(5), 790),
        ("[0, 0, 2, 0, 0]", torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), 791),
        ("[0, 0, 6, 0, 0]", torch.tensor([0.0, 0.0, 6.0, 0.0, 0.0]), 792),
    ):
        # play draws the same 1,000 episodes average_reward does, from a generator seeded the same.
        actions, _ = REFERENCE["play"](preferences, ARM_MEANS, 1000, torch.Generator().manual_seed(seed))
        results[f"rl1 {name}"] = (
            REFERENCE["expected_reward"](preferences, ARM_MEANS).item(),
            REFERENCE["average_reward"](preferences, ARM_MEANS, 1000, torch.Generator().manual_seed(seed)),
            REFERENCE["best_arm_share"](actions, ARM_MEANS),
        )

    for steps in (50, 200, 500):
        trained = [FIRST_TRAIN["train_bandit"](ARM_MEANS, steps, 32, 0.1, torch.Generator().manual_seed(seed)) for seed in seeds]
        results[f"reinforce1 {steps} steps"] = (
            [REFERENCE["expected_reward"](preferences, ARM_MEANS).item() for preferences in trained],
            [torch.softmax(preferences, dim=0)[2].item() for preferences in trained],
        )

    for name, means in (("plain", ARM_MEANS), ("shifted", SHIFTED_MEANS)):
        for baseline in (False, True):
            samples = REFERENCE["gradient_samples"](torch.zeros(5), means, 32, 1000, baseline, torch.Generator().manual_seed(791))
            results[f"reinforce2 noise {name} baseline {baseline}"] = (
                REFERENCE["gradient_noise"](samples),
                samples.mean(dim=0).tolist(),
            )

    # Ten times reinforce1's rate, on the shifted bandit and on the plain one, so the shift shows as the cause.
    for name, means in (("plain", ARM_MEANS), ("shifted", SHIFTED_MEANS)):
        for baseline in (False, True):
            trained = [REFERENCE["train_bandit"](means, 100, 32, 1.0, baseline, torch.Generator().manual_seed(seed)) for seed in range(10)]
            results[f"reinforce2 train {name} at 1.0 baseline {baseline}"] = (
                [REFERENCE["expected_reward"](preferences, ARM_MEANS).item() for preferences in trained],
                sum(int(preferences.argmax()) == 2 for preferences in trained),
            )

    results["reinforce3 exact"] = REFERENCE["exact_gradient"](torch.zeros(5), ARM_MEANS).tolist()

    for episodes in (10, 1000, 100000):
        results[f"reinforce3 gap {episodes}"] = [
            REFERENCE["estimate_gap"](torch.zeros(5), ARM_MEANS, episodes, torch.Generator().manual_seed(seed)) for seed in seeds
        ]

    for steps in (10, 20, 40, 80):
        spellers = [REFERENCE["train_speller"](TORCH, steps, 64, 0.1, torch.Generator().manual_seed(seed)) for seed in seeds]
        results[f"reinforce3 speller {steps} steps"] = (
            [int((logits.argmax(dim=1) == TORCH).sum()) for logits in spellers],
            [REFERENCE["matches"](REFERENCE["spell"](logits, 1000, torch.Generator().manual_seed(792)), TORCH).mean().item() for logits in spellers],
        )

    return results


# ---------------------------------------------------------------------------
# rl1
# ---------------------------------------------------------------------------

RL1_LESSON = prose(
    """
    Every model so far learned from right answers, with a label for each
    output. Reinforcement learning has no labels. A program takes an action,
    and the world responds with a score that says how good the action was.
    Four standard terms name the parts.

    - **Policy**: the rule that decides what to do. Here it is a probability distribution over the possible choices.
    - **Action**: one choice the policy makes.
    - **Reward**: the number the world gives back for an action. Higher is better.
    - **Episode**: one run from start to finish. Here an episode is a single action and its reward. In 80_ppo an episode lasts hundreds of steps.

    The world here is a bandit, a row of 5 slot machines called arms. Each
    time an arm is played, its reward is that arm's mean plus random noise
    with mean 0 and standard deviation 1. The policy never sees the means. It
    only sees the rewards of the arms it played.

    ```python
    ARM_MEANS = torch.tensor([0.2, -0.5, 1.0, 0.4, -0.1])
    ```

    The policy holds one number per arm, called its preference. The
    probabilities of playing the arms, called `chances` in the code, are the
    softmax of the preferences. The goal is the highest expected reward,
    which is the average reward per episode over very many episodes. When
    the probabilities and the means are known, the expected reward is a sum.

    ```text
    expected reward = sum over arms of probability(arm) * mean(arm)
    ```

    Without knowing the means, you can still estimate the expected reward by
    playing many episodes and averaging their rewards. Here are three
    policies. Each row shows the exact expected reward and the average
    reward over 1,000 played episodes.

    | Preferences | Exact expected reward | Average over 1,000 episodes | Plays of the best arm |
    |---|---|---|---|
    | all 0 | 0.200 | 0.154 | 19.8% |
    | [0, 0, 2, 0, 0] | 0.649 | 0.672 | 67.8% |
    | [0, 0, 6, 0, 0] | 0.990 | 1.002 | 98.6% |

    Each average is off by a few hundredths, sometimes above and sometimes
    below, because of the random noise in its 1,000 rewards. The best
    possible policy puts all its probability on arm 2, which gives an
    expected reward of 1.0.

    The last column is `best_arm_share`, the fraction of played actions that
    chose arm 2, the best arm. It stays close to arm 2's probability in each
    policy, which is 0.2, 0.65 and 0.99. The next exercises learn the
    preferences from the rewards alone, which pushes this fraction toward 1.
    """,
)

RL1_SOURCE = source_file(
    '''
    # rl1.py

    import torch
    from torch import Tensor

''' + indented(BANDIT_PART) + '''

    chances = torch.softmax(torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), dim=0)
    # tensor([0.0878, 0.0878, 0.6488, 0.0878, 0.0878])
    picks = torch.multinomial(chances, 6, replacement=True, generator=torch.Generator().manual_seed(0))
    # tensor([4, 2, 2, 4, 2, 2]), six actions drawn this way, mostly arm 2


    # TODO: Play the bandit for `episodes` episodes. First draw all the actions
    #   at once with torch.multinomial, from the softmax of preferences, with
    #   replacement=True and generator. Then draw the noise for all the rewards
    #   with torch.randn(episodes, generator=generator). Each reward is the
    #   mean of its arm, from arm_means, plus its noise. Return the actions and
    #   the rewards, two tensors of shape (episodes,).
    def play(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the policy's expected reward, as a tensor with one number.
    #   Multiply each arm's probability, from the softmax of preferences, by
    #   that arm's mean, and add up the products.
    #   Example: expected_reward(torch.zeros(5), ARM_MEANS) returns
    #   tensor(0.2000).
    def expected_reward(
        preferences: Tensor,
        arm_means: Tensor,
    ) -> Tensor:
        ...


    # TODO: Play `episodes` episodes with play, and return the mean of their
    #   rewards as a Python float.
    def average_reward(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> float:
        ...


    # TODO: Return the fraction of actions that chose the best arm, the arm
    #   with the highest mean, as a Python float.
    #   Example: best_arm_share(torch.tensor([2, 0, 2, 4]), ARM_MEANS) returns
    #   0.5.
    def best_arm_share(
        actions: Tensor,
        arm_means: Tensor,
    ) -> float:
        ...
    ''',
)

RL1_SOLUTION = source_file(
    '''
    # rl1.py, a solution for comparison

    import torch
    from torch import Tensor

''' + indented(BANDIT_PART) + '''

    chances = torch.softmax(torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), dim=0)
    picks = torch.multinomial(chances, 6, replacement=True, generator=torch.Generator().manual_seed(0))


''' + indented(PLAY_PART) + '''


''' + indented(EXPECTED_PART) + '''


''' + indented(AVERAGE_PART) + '''
    ''',
)


def check_rl1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="play",
        description="draws the actions first, then the noise of each reward",
        inputs=(
            (torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), ARM_MEANS, 8, seeded(793)),
            (torch.tensor([1.0, -1.0, 0.5]), torch.tensor([3.0, -2.0, 0.0]), 5, seeded(794)),
        ),
        reference=REFERENCE["play"],
        expect=expect_parts("the actions", "the rewards"),
    )
    check_on_inputs(
        checker=checker,
        function="expected_reward",
        description="multiplies each arm's mean by its probability and sums",
        inputs=(
            (torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0]), ARM_MEANS),
            (torch.tensor([1.0, -1.0, 0.5]), torch.tensor([3.0, -2.0, 0.0])),
        ),
        reference=REFERENCE["expected_reward"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="average_reward",
        description="averages the rewards of played episodes",
        inputs=(
            (torch.zeros(5), ARM_MEANS, 50, seeded(795)),
            (torch.tensor([2.0, 0.0, -1.0]), torch.tensor([1.0, 4.0, -3.0]), 12, seeded(796)),
        ),
        reference=REFERENCE["average_reward"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="best_arm_share",
        description="returns the fraction of actions on the best arm",
        inputs=(
            (torch.tensor([2, 0, 2, 4, 2, 1, 2, 3]), ARM_MEANS),
            (torch.tensor([0, 1, 1, 0, 2]), torch.tensor([-1.0, 0.5, 0.2])),
            # The arm played most is not the best one, so a share of the likeliest action shows.
            (torch.tensor([0, 0, 0, 2, 2, 4]), ARM_MEANS),
        ),
        reference=REFERENCE["best_arm_share"],
        expect=expect_close,
    )


RL1 = Exercise(
    name="rl1",
    chapter=POLICY_GRADIENTS,
    title="Policy, action, reward, episode",
    lesson=RL1_LESSON,
    source=RL1_SOURCE,
    solution=RL1_SOLUTION,
    hints={
        "play": "torch.softmax(preferences, dim=0) gives the probabilities, and torch.multinomial draws the actions "
        "from them. arm_means[actions] gives the mean of each action's arm. Then add "
        "torch.randn(episodes, generator=generator).",
        "expected_reward": "Multiply torch.softmax(preferences, dim=0) by arm_means and call .sum() on the result.",
        "average_reward": "Take the rewards that play returns, then call .mean() and .item() on them.",
        "best_arm_share": "arm_means.argmax() is the best arm. Compare the actions with it using ==, turn the result "
        "into floats, then take the mean and call .item().",
    },
    introduces=("policy", "episode"),
    check=check_rl1,
)


# ---------------------------------------------------------------------------
# reinforce1
# ---------------------------------------------------------------------------

REINFORCE1_LESSON = prose(
    """
    The policy should shift its probability toward the arms that pay more.
    Gradient descent needs the gradient of the expected reward with respect
    to the preferences. But played episodes cannot give that gradient
    directly. An action is an integer drawn at random, so it has no
    gradient. A reward comes from the world, outside autograd, so it has
    none either.

    REINFORCE, published by Williams in 1992, gets the gradient anyway.
    Write the expected reward as J, a sum over the actions a. Here p(a) is
    the policy's probability of action a, and r(a) is the mean reward of
    that arm.

    ```text
    J = sum over a of p(a) * r(a)
    gradient of J = sum over a of p(a) * r(a) * gradient of log p(a)
    ```

    The second line uses `gradient of p = p * gradient of log p`, which is
    the chain rule applied to `log p`. The second line has the same form as
    the first, a sum of p(a) times something. So it is an average over the
    actions the policy picks, and played episodes can estimate it. Each
    episode contributes its reward times the gradient of the log probability
    of its action.

    In code, this estimate becomes a loss. Autograd computes the gradient of
    the log probabilities, and the rewards are treated as constant weights.

    ```text
    loss = minus the mean over episodes of log p(action) * reward
    ```

    An optimizer minimizes a loss, but we want to maximize the expected
    reward. The minus sign turns one into the other. An action with a high
    reward gets its log probability pushed up strongly. An action with a
    negative reward gets its log probability pushed down.

    `torch.log_softmax(preferences, dim=0)` gives the log probability of
    every arm at once. It computes them in the numerically safe way that
    likelihood2 built by hand.

    The table shows training with SGD at a learning rate of 0.1 and 32
    episodes per step, starting from equal preferences. Each cell gives the
    range over 5 seeds.

    | Steps | 50 | 200 | 500 |
    |---|---|---|---|
    | Expected reward | 0.47 to 0.53 | 0.92 to 0.93 | 0.979 to 0.980 |
    | Probability of arm 2 | 0.42 to 0.48 | 0.91 to 0.92 | 0.976 to 0.977 |

    Every seed finds arm 2 from the rewards alone. The improvement slows
    down near the top. By then the other arms are rarely played, so their
    rewards are rarely seen.
    """,
)

REINFORCE1_SOURCE = source_file(
    '''
    # reinforce1.py

    import torch
    from torch import Tensor

    log_chances = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)    # tensor([-2.4076, -1.4076, -0.4076])
    picked = log_chances[torch.tensor([2, 2, 0])]                           # tensor([-0.4076, -0.4076, -2.4076])


    # The bandit and play from rl1.
''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


    # TODO: Return the log probability of each action in actions, where the
    #   probabilities are the softmax of preferences. The result has one entry
    #   per action.
    #   Example: log_probabilities(torch.tensor([0.0, 1.0, 2.0]),
    #   torch.tensor([2, 0])) returns tensor([-0.4076, -2.4076]).
    def log_probabilities(
        preferences: Tensor,
        actions: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the REINFORCE loss, as a tensor with one number. Multiply
    #   each log probability by its reward, take the mean of the products, and
    #   negate it.
    #   Example: reinforce_loss(torch.tensor([-0.5, -1.0]),
    #   torch.tensor([2.0, 1.0])) returns tensor(1.).
    def reinforce_loss(
        log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train the preferences and return them. Start them as zeros, one
    #   per arm, that record gradients, and make a torch.optim.SGD optimizer
    #   for them with learning rate lr. Run `steps` training steps. In each
    #   one, play `episodes` episodes with play and generator, compute
    #   reinforce_loss, and take one optimizer step. Zero the gradients before
    #   each backward call. Return the trained preferences detached.
    def train_bandit(
        arm_means: Tensor,
        steps: int,
        episodes: int,
        lr: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

REINFORCE1_SOLUTION = source_file(
    '''
    # reinforce1.py, a solution for comparison

    import torch
    from torch import Tensor

    log_chances = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    picked = log_chances[torch.tensor([2, 2, 0])]


''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


''' + indented(LOG_PROB_PART) + '''


''' + indented(REINFORCE_LOSS_PART) + '''


''' + indented(TRAIN_BANDIT_PART) + '''
    ''',
)


def expect_loss_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    if isinstance(actual, Tensor) and actual.dim() != 0:
        raise Mismatch(f"{call} has shape {tuple(actual.shape)}, but a loss is one number. Take the mean.")

    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )


def check_loss_has_gradient(found: object) -> None:
    preferences = torch.tensor([0.5, -0.5, 0.0], requires_grad=True)
    log_probs = torch.log_softmax(preferences, dim=0)[torch.tensor([0, 2, 2])]
    loss = found(log_probs, torch.tensor([1.0, -2.0, 0.5]))

    if not isinstance(loss, Tensor) or loss.grad_fn is None:
        raise Mismatch(
            "reinforce_loss returns a loss that autograd did not record, so no gradient can reach the preferences. "
            "Return a tensor computed from log_probs, without .item().",
        )


def check_train_bandit(found: object) -> None:
    for steps, means, seed in ((2, ARM_MEANS, 797), (6, torch.tensor([1.0, -1.0, 0.0]), 798)):
        call = f"train_bandit(arm_means, {steps}, 8, 0.5, a generator seeded with {seed})"
        actual = found(means, steps, 8, 0.5, torch.Generator().manual_seed(seed))
        expect_tensor(
            call=call,
            actual=actual,
            expected=FIRST_TRAIN["train_bandit"](means, steps, 8, 0.5, torch.Generator().manual_seed(seed)),
        )

        if actual.requires_grad:
            raise Mismatch(f"{call} gives preferences that still record gradients. Return preferences.detach().")


def check_bandit_learns(found: object) -> None:
    preferences = found(ARM_MEANS, 300, 32, 0.1, torch.Generator().manual_seed(799))
    chance = torch.softmax(preferences, dim=0)[2].item()

    if chance < 0.6:
        raise Mismatch(f"After 300 steps, train_bandit gives arm 2 a probability of {chance:.2f}, but it should be above 0.6.")


def check_reinforce1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="log_probabilities",
        description="returns each action's log probability",
        inputs=(
            (torch.tensor([0.0, 1.0, 2.0]), torch.tensor([2, 0, 0, 1])),
            (torch.tensor([3.0, -1.0, 0.5, 0.0, 2.0]), torch.tensor([4, 4, 1])),
        ),
        reference=REFERENCE["log_probabilities"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="reinforce_loss",
        description="is minus the mean of log probability times reward",
        inputs=(
            (torch.tensor([-0.4, -1.2, -2.0]), torch.tensor([1.0, -0.5, 2.0])),
            (torch.tensor([-0.1, -3.0]), torch.tensor([10.0, 9.0])),
        ),
        reference=REFERENCE["reinforce_loss"],
        expect=expect_loss_tensor,
    )
    checker.test(
        function="reinforce_loss",
        description="keeps the gradient of the log probabilities",
        run=check_loss_has_gradient,
    )
    checker.test(
        function="train_bandit",
        description="plays episodes, then takes an SGD step on reinforce_loss",
        run=check_train_bandit,
    )
    checker.test(
        function="train_bandit",
        description="learns to favor the best arm",
        run=check_bandit_learns,
    )


REINFORCE1 = Exercise(
    name="reinforce1",
    chapter=POLICY_GRADIENTS,
    title="REINFORCE",
    lesson=REINFORCE1_LESSON,
    source=REINFORCE1_SOURCE,
    solution=REINFORCE1_SOLUTION,
    hints={
        "log_probabilities": "torch.log_softmax(preferences, dim=0) gives every arm's log probability. Index it "
        "with the actions.",
        "reinforce_loss": "Multiply log_probs by rewards, take the mean, and put a minus in front.",
        "train_bandit": "Start the preferences as torch.zeros, one per arm, with requires_grad=True, and make SGD "
        "over [preferences]. In the loop, call play, then compute reinforce_loss of "
        "log_probabilities(preferences, actions) and the rewards. Then call zero_grad, backward and step.",
    },
    introduces=(),
    check=check_reinforce1,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# reinforce2
# ---------------------------------------------------------------------------

REINFORCE2_LESSON = prose(
    """
    REINFORCE's gradient estimate is correct on average, but it is noisy. To
    see why, add 10 to every arm's mean, so every reward is about 10. The
    best arm is still arm 2. But now every reward is positive, so every
    played action gets its probability pushed up. The good actions are
    pushed only a little more than the bad ones. So the direction of each
    update depends heavily on which actions happened to be drawn in that
    batch.

    A **baseline** fixes this. Subtract a number b from every reward before
    multiplying it by the log probabilities. On average, the estimate still
    equals the same gradient. Subtracting b adds the term
    `-b * gradient of log p(a)` to each episode. Averaged over actions, that
    term is `-b` times the gradient of the sum of all the probabilities. The
    probabilities always sum to 1, so that gradient is 0.

    The reward minus the baseline is called the **advantage**. It says how
    much better an action did than usual. A common baseline is the average
    reward of the batch. That average includes the reward it is subtracted
    from. Because of this, the estimate comes out smaller by a factor of
    31/32 with 32 episodes. GRPO, in 82_dpo, uses a version of this baseline
    computed separately for each question.

    The table shows how noisy the estimate is. Each number is the standard
    deviation of the estimate over 1,000 batches of 32 episodes, at equal
    preferences, averaged over the 5 arms.

    | Bandit | Without a baseline | With the batch average |
    |---|---|---|
    | `ARM_MEANS` | 0.078 | 0.076 |
    | `ARM_MEANS + 10` | 0.72 | 0.076 |

    On the shifted bandit, the baseline cuts the noise by a factor of about
    10. On `ARM_MEANS` the rewards already average near 0, so the baseline
    changes little. The mean of the 1,000 estimates is about the same in
    every cell of the table.

    Less noise allows a higher learning rate. The next table runs
    reinforce1's training on the shifted bandit for 100 steps, on 10 seeds.
    The learning rate is 1.0, which is 10 times reinforce1's. The expected
    reward is measured on `ARM_MEANS`, without the shift of 10.

    | Log probabilities weighted by | Lowest | Highest | Average | Seeds that favor arm 2 |
    |---|---|---|---|---|
    | Rewards | -0.50 | 1.00 | 0.70 | 7 of 10 |
    | Advantages | 0.99 | 0.99 | 0.99 | 10 of 10 |

    Without the baseline, 3 seeds end up favoring a worse arm, and one of
    them has an expected reward of -0.50. With the baseline, every seed finds
    arm 2. On the unshifted `ARM_MEANS`, both versions reach 0.99 at this
    rate. So the shift is what breaks the version without a baseline. The
    baseline makes a learning rate 10 times higher safe to use.

    `.clone().requires_grad_()` makes a fresh copy of a tensor that records
    gradients. Each batch gets its own copy, so each batch's gradient ends up
    in a separate `.grad`.
    """,
)

REINFORCE2_SOURCE = source_file(
    '''
    # reinforce2.py

    import torch
    from torch import Tensor

    rewards = torch.tensor([10.5, 9.0, 11.5])
    centered = rewards - rewards.mean()                          # tensor([ 0.1667, -1.3333,  1.1667])
    leaf = torch.zeros(3).clone().requires_grad_()               # a fresh copy that records gradients


    # The bandit, play, log_probabilities and reinforce_loss from rl1 and reinforce1.
''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


''' + indented(LOG_PROB_PART) + '''


''' + indented(REINFORCE_LOSS_PART) + '''


    # TODO: Return the advantages, which are each reward minus the mean of all
    #   the rewards.
    def advantages(rewards: Tensor) -> Tensor:
        ...


    # TODO: Return `trials` separate estimates of the gradient of the expected
    #   reward, stacked into a tensor of shape (trials, arms). For each trial,
    #   make a fresh copy of preferences that records gradients, and play
    #   `episodes` episodes with it. Weight the log probabilities by the
    #   advantages when baseline is True, or by the rewards when it is False,
    #   and backpropagate reinforce_loss. The estimate is the negative of the
    #   copy's gradient, because the loss has a minus sign.
    def gradient_samples(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        trials: int,
        baseline: bool,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return how noisy samples is, as a Python float. Take the standard
    #   deviation over the trials of each arm's estimate, then average those
    #   over the arms.
    #   Example: gradient_noise(torch.tensor([[0.5, -0.5], [1.5, 0.5]]))
    #   returns about 0.7071.
    def gradient_noise(samples: Tensor) -> float:
        ...


    # TODO: Finish reinforce1's training loop. Set weights to the advantages
    #   when baseline is True, and to the rewards when it is False.
''' + indented(TRAIN_BASELINE_STUB) + '''
    ''',
)

REINFORCE2_SOLUTION = source_file(
    '''
    # reinforce2.py, a solution for comparison

    import torch
    from torch import Tensor

    rewards = torch.tensor([10.5, 9.0, 11.5])
    centered = rewards - rewards.mean()
    leaf = torch.zeros(3).clone().requires_grad_()


''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


''' + indented(LOG_PROB_PART) + '''


''' + indented(REINFORCE_LOSS_PART) + '''


''' + indented(ADVANTAGE_PART) + '''


''' + indented(SAMPLES_PART) + '''


''' + indented(TRAIN_BASELINE_PART) + '''
    ''',
)


def check_samples_shape_and_values(found: object) -> None:
    for preferences, means, episodes, trials, baseline, seed in (
        (torch.zeros(5), SHIFTED_MEANS, 6, 3, True, 800),
        (torch.zeros(5), SHIFTED_MEANS, 6, 3, False, 800),
        (torch.tensor([1.0, -1.0, 0.0]), torch.tensor([2.0, 0.0, -1.0]), 4, 2, True, 801),
    ):
        call = f"gradient_samples(preferences, arm_means, {episodes}, {trials}, {baseline}, a generator seeded with {seed})"
        expect_tensor(
            call=call,
            actual=found(preferences.clone(), means, episodes, trials, baseline, torch.Generator().manual_seed(seed)),
            expected=REFERENCE["gradient_samples"](preferences.clone(), means, episodes, trials, baseline, torch.Generator().manual_seed(seed)),
        )


def check_train_with_baseline(found: object) -> None:
    for steps, means, baseline, seed in ((2, SHIFTED_MEANS, True, 802), (2, SHIFTED_MEANS, False, 802), (5, torch.tensor([1.0, -1.0, 0.0]), True, 803)):
        call = f"train_bandit(arm_means, {steps}, 8, 0.1, {baseline}, a generator seeded with {seed})"
        expect_tensor(
            call=call,
            actual=found(means, steps, 8, 0.1, baseline, torch.Generator().manual_seed(seed)),
            expected=REFERENCE["train_bandit"](means, steps, 8, 0.1, baseline, torch.Generator().manual_seed(seed)),
        )


def check_reinforce2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="advantages",
        description="subtracts the mean reward",
        inputs=(
            (torch.tensor([10.5, 9.0, 11.5, 10.0]),),
            (torch.tensor([-1.0, 2.0]),),
        ),
        reference=REFERENCE["advantages"],
        expect=expect_tensor,
    )
    checker.test(
        function="gradient_samples",
        description="returns one gradient estimate per trial",
        run=check_samples_shape_and_values,
    )
    check_on_inputs(
        checker=checker,
        function="gradient_noise",
        description="averages each arm's standard deviation over the trials",
        inputs=(
            (torch.tensor([[1.0, 0.0, -2.0], [3.0, 0.0, 2.0], [2.0, 1.0, 0.0]]),),
            (torch.tensor([[0.5, -0.5], [1.5, 0.5]]),),
        ),
        reference=REFERENCE["gradient_noise"],
        expect=expect_close,
    )
    checker.test(
        function="train_bandit",
        description="weights by the advantages when baseline is True",
        run=check_train_with_baseline,
    )


REINFORCE2 = Exercise(
    name="reinforce2",
    chapter=POLICY_GRADIENTS,
    title="A baseline, and the advantage",
    lesson=REINFORCE2_LESSON,
    source=REINFORCE2_SOURCE,
    solution=REINFORCE2_SOLUTION,
    hints={
        "advantages": "Subtract rewards.mean() from rewards.",
        "gradient_samples": "In a loop, make a fresh leaf with preferences.clone() and requires_grad_, then play with "
        "the leaf. Call .backward() on reinforce_loss of log_probabilities(leaf, actions) and the weights. Keep "
        "-leaf.grad in a list, and torch.stack the list at the end.",
        "gradient_noise": "samples.std(dim=0) gives each arm's standard deviation over the trials. Take its .mean(), "
        "then .item().",
        "train_bandit": "Use a conditional expression that gives advantages(rewards) when baseline is True and the "
        "rewards otherwise.",
    },
    introduces=(),
    check=check_reinforce2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# reinforce3
# ---------------------------------------------------------------------------

REINFORCE3_LESSON = prose(
    """
    You can check the estimate against the true gradient. On the bandit,
    `expected_reward` is a sum of probabilities times means, so autograd can
    compute its exact gradient. REINFORCE's estimate from one batch gets
    close to it when the batch has enough episodes. The table shows the
    largest difference between the estimate and the exact gradient over the
    5 arms. It uses equal preferences and gives the range over 5 seeds.

    | Episodes in the batch | 10 | 1,000 | 100,000 |
    |---|---|---|---|
    | Largest difference | 0.095 to 0.19 | 0.016 to 0.047 | 0.0007 to 0.0025 |

    At equal preferences the exact gradient is `[0, -0.14, 0.16, 0.04, -0.06]`.
    The entry for each arm is its probability, 0.2, times the arm's mean
    minus the expected reward of 0.2.

    The estimate never needed the gradient of the reward. The reward only
    multiplies the log probabilities, so any score works as a reward, even
    one computed with `==`.

    A speller shows this. The policy writes a word of 5 letters. Each letter
    is drawn from its own softmax over the 26 letters of the alphabet. The
    reward is the number of letters that match a hidden word, `torch`, at
    the same position. Counting matches with `==` gives integers, which have
    no gradient. The policy never sees the hidden word.

    ```python
    (torch.tensor([[19, 14, 17, 2, 7]]) == torch.tensor([19, 0, 17, 2, 0])).sum(dim=1)    # tensor([3])
    ```

    The letters are drawn independently, so the log probability of a word is
    the sum of the log probabilities of its letters. To pick them out, index
    the log probabilities with `torch.arange(5)` and the words together. For
    each word and position, this picks the row for that position and the
    column for the word's letter there. The arange lines up with the
    positions, which are the last dimension of the words. The `picked` line
    in the file shows this on a small example.

    The table shows training with Adam at a learning rate of 0.1 and 64
    words per step, with the batch average as the baseline. Each cell gives
    the range over 5 seeds.

    | Steps | 10 | 20 | 40 | 80 |
    |---|---|---|---|---|
    | Average reward of drawn words | 0.54 to 0.58 | 1.49 to 1.57 | 4.49 to 4.54 | 4.93 to 4.94 |

    Random letters score 5/26 on average, about 0.19. After 10 steps, the
    most likely word is already `torch` on every seed, even though the drawn
    words still miss most letters. The later steps make the policy more and
    more certain of `torch`.

    This is how a model can train on scores like a passed test, a solved
    puzzle or a person's thumbs up. The rest of Section 11 does this with a
    language model. If the hidden word were known, ordinary cross entropy on
    its letters would train faster. REINFORCE is for when the score is the
    only signal available.
    """,
)

REINFORCE3_SOURCE = source_file(
    '''
    # reinforce3.py

    import torch
    from torch import Tensor

    rows = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    picked = rows[torch.arange(2), torch.tensor([[2, 0], [1, 1]])]    # tensor([[0.3000, 0.4000], [0.2000, 0.5000]])


    # The bandit and the pieces of reinforce1 and reinforce2.
''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


''' + indented(EXPECTED_PART) + '''


''' + indented(LOG_PROB_PART) + '''


''' + indented(REINFORCE_LOSS_PART) + '''


''' + indented(ADVANTAGE_PART) + '''


''' + indented(SAMPLES_PART) + '''


    # TODO: Return the exact gradient of expected_reward with respect to
    #   preferences. Make a fresh copy of preferences that records gradients,
    #   backpropagate expected_reward through it, and return the copy's
    #   gradient.
    #   Example: exact_gradient(torch.zeros(5), ARM_MEANS) returns
    #   tensor([ 0.0000, -0.1400,  0.1600,  0.0400, -0.0600]).
    def exact_gradient(
        preferences: Tensor,
        arm_means: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return how far one REINFORCE estimate is from the exact gradient,
    #   as a Python float. Get the estimate from gradient_samples with one
    #   trial of `episodes` episodes and no baseline. Subtract exact_gradient,
    #   and return the largest absolute difference over the arms.
    def estimate_gap(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> float:
        ...


    # The speller's world. logits has shape (5, 26), one row of letter scores
    # per position. spell draws batch words from the softmax of each row.
    # matches counts the letters of each word that equal the target's letter
    # at the same position.
''' + indented(SPELLER_PART) + '''


    # TODO: words has shape (batch, 5), and logits has shape (5, 26). Return
    #   the log probability of each word, a tensor of shape (batch,). It is the
    #   sum over the 5 positions of the log softmax of that position's row of
    #   logits, taken at the word's letter.
    def word_log_probs(
        logits: Tensor,
        words: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train the logits and return them. Start them as zeros of shape
    #   (len(target), LETTERS) that record gradients, and use torch.optim.Adam
    #   with learning rate lr. Run `steps` training steps. In each one, spell
    #   `batch` words with generator and score them with matches to get the
    #   rewards. Then take one step on reinforce_loss of the words' log
    #   probabilities, weighted by the advantages. Return the trained logits
    #   detached.
    def train_speller(
        target: Tensor,
        steps: int,
        batch: int,
        lr: float,
        generator: torch.Generator,
    ) -> Tensor:
        ...
    ''',
)

REINFORCE3_SOLUTION = source_file(
    '''
    # reinforce3.py, a solution for comparison

    import torch
    from torch import Tensor

    rows = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    picked = rows[torch.arange(2), torch.tensor([[2, 0], [1, 1]])]


''' + indented(BANDIT_PART) + '''


''' + indented(PLAY_PART) + '''


''' + indented(EXPECTED_PART) + '''


''' + indented(LOG_PROB_PART) + '''


''' + indented(REINFORCE_LOSS_PART) + '''


''' + indented(ADVANTAGE_PART) + '''


''' + indented(SAMPLES_PART) + '''


''' + indented(EXACT_PART) + '''


''' + indented(SPELLER_PART) + '''


''' + indented(SPELLER_SOLUTION) + '''
    ''',
)


def check_train_speller(found: object) -> None:
    for target, steps, seed in ((TORCH, 2, 804), (torch.tensor([3, 0, 25]), 4, 805)):
        call = f"train_speller({target.tolist()}, {steps}, 16, 0.1, a generator seeded with {seed})"
        expect_tensor(
            call=call,
            actual=found(target, steps, 16, 0.1, torch.Generator().manual_seed(seed)),
            expected=REFERENCE["train_speller"](target, steps, 16, 0.1, torch.Generator().manual_seed(seed)),
        )


def check_speller_learns(found: object) -> None:
    target = torch.tensor([6, 17, 0, 3, 4])
    logits = found(target, 100, 64, 0.1, torch.Generator().manual_seed(806))
    right = int((logits.argmax(dim=1) == target).sum())

    if right < 5:
        raise Mismatch(
            f"After 100 steps, the most likely word from train_speller gets {right} of the 5 letters of 'grade' right, "
            "but it should get all 5.",
        )


def check_reinforce3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="exact_gradient",
        description="returns the exact gradient of the expected reward",
        inputs=(
            (torch.zeros(5), ARM_MEANS),
            (torch.tensor([1.0, -1.0, 0.5]), torch.tensor([3.0, -2.0, 0.0])),
        ),
        reference=REFERENCE["exact_gradient"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="estimate_gap",
        description="compares one estimate with the exact gradient",
        inputs=(
            (torch.zeros(5), ARM_MEANS, 20, seeded(807)),
            (torch.tensor([1.0, -1.0, 0.5]), torch.tensor([3.0, -2.0, 0.0]), 200, seeded(808)),
        ),
        reference=REFERENCE["estimate_gap"],
        expect=expect_close,
    )
    check_on_inputs(
        checker=checker,
        function="word_log_probs",
        description="sums the log probabilities of each word's letters",
        inputs=(
            (Described(torch.randn(5, 26, generator=torch.Generator().manual_seed(809)), "logits of shape (5, 26)"), torch.tensor([[19, 14, 17, 2, 7], [0, 0, 0, 0, 0], [25, 1, 17, 3, 9]])),
            (Described(torch.randn(3, 4, generator=torch.Generator().manual_seed(810)), "logits of shape (3, 4)"), torch.tensor([[3, 0, 1], [2, 2, 2]])),
        ),
        reference=REFERENCE["word_log_probs"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_speller",
        description="takes Adam steps on the REINFORCE loss with advantages",
        run=check_train_speller,
    )
    checker.test(
        function="train_speller",
        description="learns a hidden word from its score alone",
        run=check_speller_learns,
    )


REINFORCE3 = Exercise(
    name="reinforce3",
    chapter=POLICY_GRADIENTS,
    title="Training on a score with no gradient",
    lesson=REINFORCE3_LESSON,
    source=REINFORCE3_SOURCE,
    solution=REINFORCE3_SOLUTION,
    hints={
        "exact_gradient": "Make leaf = preferences.clone().requires_grad_(), as gradient_samples does. Call .backward() "
        "on expected_reward(leaf, arm_means) and return leaf.grad.",
        "estimate_gap": "gradient_samples(preferences, arm_means, episodes, 1, False, generator)[0] is the one "
        "estimate. Subtract exact_gradient from it, then call .abs().max().item().",
        "word_log_probs": "Take torch.log_softmax(logits, dim=1) and index it with torch.arange(len(logits)) and words "
        "together. Then sum over the positions with .sum(dim=1).",
        "train_speller": "Start the logits as torch.zeros with one row of LETTERS entries per letter of the target, and "
        "requires_grad=True. In the loop, call spell, then matches. Then compute reinforce_loss of word_log_probs and "
        "advantages(rewards), and call zero_grad, backward and step.",
    },
    introduces=(),
    check=check_reinforce3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
