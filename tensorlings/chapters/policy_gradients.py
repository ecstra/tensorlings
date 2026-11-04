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

        Reinforcement learning trains from scores in place of right answers.
        This chapter names its parts on a row of slot machines, then derives
        REINFORCE, the classic way to follow the gradient of an expected score.
        A baseline cuts its noise, and a speller trained on a score made with
        `==` shows why it needs no gradient from the score at all.

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
    Every model so far learned from right answers, a label for each output.
    Reinforcement learning has none. A program acts, and the world answers
    with a score. Four words name the parts.

    - **Policy**: the rule that picks what to do, here a probability distribution over the choices.
    - **Action**: one choice the policy makes.
    - **Reward**: the number the world hands back for an action, higher for better.
    - **Episode**: one run from start to end. Here it is a single action and its reward. In 80_ppo it lasts hundreds of steps.

    The world here is a bandit, a row of 5 slot machines called arms. Each
    arm pays its own average plus noise from a standard normal. The policy
    never sees the averages, only the rewards of the arms it played.

    ```python
    ARM_MEANS = torch.tensor([0.2, -0.5, 1.0, 0.4, -0.1])
    ```

    The policy holds one preference per arm, and its chances are their
    softmax. Its goal is the highest expected reward, the average reward per
    episode over very many episodes. With the chances and the averages known,
    that is each chance times its arm's average, summed.

    Playing estimates it without knowing the averages. Here are three
    policies, each with its exact expected reward and the average over 1,000
    played episodes.

    | Preferences | Exact | Average over 1,000 episodes | Best arm played |
    |---|---|---|---|
    | all 0 | 0.200 | 0.154 | 19.8% |
    | [0, 0, 2, 0, 0] | 0.649 | 0.672 | 67.8% |
    | [0, 0, 6, 0, 0] | 0.990 | 1.002 | 98.6% |

    Each average misses by a few hundredths, either way, from the noise of
    its own 1,000 draws. The best policy puts all its chance on arm 2, for
    an expected reward of 1.0.

    The last column is `best_arm_share`, the share of played actions that
    went to arm 2, the best arm. It tracks arm 2's chance, 0.2, 0.65 and
    0.99. The next exercises learn the preferences from rewards alone, and
    push this share toward 1.
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
    # tensor([4, 2, 2, 4, 2, 2]), six actions, mostly arm 2


    # TODO: Play episodes episodes of the bandit with arm_means. Draw every
    #   action at once from the softmax of preferences, with torch.multinomial,
    #   replacement=True and generator. Then draw the noise of every reward with
    #   torch.randn(episodes, generator=generator). Each reward is its arm's
    #   average plus its noise. Return the actions and the rewards.
    def play(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Return the policy's expected reward as a tensor, each arm's chance
    #   times its average, summed.
    def expected_reward(
        preferences: Tensor,
        arm_means: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the average reward of episodes episodes from play, as a
    #   plain number.
    def average_reward(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> float:
        ...


    # TODO: Return the share of actions that played the arm with the highest
    #   average, as a plain number.
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
        description="draws the actions, then each reward's noise",
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
        description="weighs each arm's average by its chance",
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
        description="gives the share of actions on the best arm",
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
        "play": "torch.softmax(preferences, dim=0) gives the chances. torch.multinomial draws the actions, "
        "arm_means[actions] gives each one's average, then add torch.randn(episodes, generator=generator).",
        "expected_reward": "Multiply torch.softmax(preferences, dim=0) by arm_means and .sum() the result.",
        "average_reward": "Take the rewards that play gives, their .mean(), then .item().",
        "best_arm_share": "arm_means.argmax() is the best arm. Compare the actions with it, turn that into floats, "
        "take the mean and .item().",
    },
    introduces=("policy", "episode"),
    check=check_rl1,
)


# ---------------------------------------------------------------------------
# reinforce1
# ---------------------------------------------------------------------------

REINFORCE1_LESSON = prose(
    """
    The policy should move its chance toward the arms that pay more. Gradient
    descent needs the gradient of the expected reward with respect to the
    preferences. Played episodes have none to give. An action is a whole
    number drawn at random, and a reward comes from the world.

    Williams's REINFORCE, from 1992, finds the gradient anyway. Write the
    expected reward J as a sum over the actions a, with p the policy's chance
    and r an arm's average.

    ```text
    J = sum over a of p(a) * r(a)
    gradient of J = sum over a of p(a) * r(a) * gradient of log p(a)
    ```

    The second line uses `gradient of p = p * gradient of log p`, the chain
    rule through the log. It is again an average over actions the policy
    picks. So played episodes estimate it, each one adding its reward times
    the gradient of its action's log probability.

    In code that becomes a loss. Autograd takes the gradient of the log
    probabilities, and the rewards are plain weights.

    ```text
    loss = minus the mean over episodes of log p(action) * reward
    ```

    The minus sign turns climbing the expected reward into the descent an
    optimizer does. A high reward pushes its action's log probability up
    hard, and a negative one pushes it down.

    `torch.log_softmax(preferences, dim=0)` gives every arm's log
    probability at once, the safe way likelihood2 built by hand.

    Here is SGD at a learning rate of 0.1 with 32 episodes a step, from equal
    preferences, on 5 seeds.

    | Steps | 50 | 200 | 500 |
    |---|---|---|---|
    | Expected reward | 0.47 to 0.53 | 0.92 to 0.93 | 0.979 to 0.980 |
    | Chance of arm 2 | 0.42 to 0.48 | 0.91 to 0.92 | 0.976 to 0.977 |

    Every seed finds arm 2 from rewards alone. The climb slows near the top,
    where the other arms are rarely played and their rewards rarely seen.
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


    # TODO: Return the log probability of each action under the softmax of
    #   preferences, one per action.
    def log_probabilities(
        preferences: Tensor,
        actions: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return REINFORCE's loss, minus the mean of each log probability
    #   times its reward.
    def reinforce_loss(
        log_probs: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train preferences, starting at zeros that record gradients, with
    #   torch.optim.SGD at lr. Each of steps steps plays episodes episodes with
    #   play and generator, then takes one step on reinforce_loss. Zero the
    #   gradients before each backward. Return the trained preferences detached.
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
        raise Mismatch("reinforce_loss gives a loss that records nothing, so no gradient reaches the preferences. Keep it a tensor.")


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
        raise Mismatch(f"After 300 steps, train_bandit gives arm 2 a chance of {chance:.2f}, but it should pass 0.6.")


def check_reinforce1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="log_probabilities",
        description="gives each action's log probability",
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
        description="plays, then steps SGD on reinforce_loss",
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
        "train_bandit": "preferences = torch.zeros(len(arm_means), requires_grad=True), and SGD over [preferences]. "
        "In the loop, play, then reinforce_loss of log_probabilities(preferences, actions) and the rewards, then "
        "zero_grad, backward and step.",
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
    REINFORCE's estimate is right on average, but noisy. Shift every arm's
    average up by 10, so every reward is about 10. The best arm is the same.
    Now every played action gets pushed up, the good ones only a little more
    than the bad. The push swings with the luck of each batch.

    A **baseline** fixes that. Subtract a number b from every reward before
    it weighs the log probabilities. The estimate still averages to the same
    gradient. The extra part, b times the gradient of log p, averages to b
    times the gradient of the summed chances. Those always sum to 1, so that
    gradient is 0.

    The reward minus the baseline is the **advantage**, how much better an
    action did than usual. A common baseline is the average reward of the
    batch. That one also counts the reward it is subtracted from, which
    shrinks the estimate by a factor of 31 in 32 at 32 episodes. GRPO, in
    82_dpo, uses a version of it taken per question.

    Here is the spread of the estimate over 1,000 batches of 32 episodes, at
    equal preferences, averaged over the 5 arms.

    | Bandit | Without a baseline | With the batch average |
    |---|---|---|
    | `ARM_MEANS` | 0.078 | 0.076 |
    | `ARM_MEANS + 10` | 0.72 | 0.076 |

    On the shifted bandit the baseline cuts the noise tenfold. On `ARM_MEANS`
    the rewards already average near 0, so it changes little. Averaged over
    the 1,000 batches, every column gives about the same gradient.

    Less noise lets the rate go higher. Here is reinforce1's training on the
    shifted bandit for 100 steps at a rate of 1.0, 10 times reinforce1's, on
    10 seeds. Its expected reward is taken on `ARM_MEANS`, leaving out the
    shift of 10.

    | Weights | Lowest | Highest | Average | Seeds that favor arm 2 |
    |---|---|---|---|---|
    | Rewards | -0.50 | 1.00 | 0.70 | 7 of 10 |
    | Advantages | 0.99 | 0.99 | 0.99 | 10 of 10 |

    Without the baseline, 3 seeds end on a worse arm, one at -0.50. With it,
    every seed finds arm 2. On `ARM_MEANS` itself both reach 0.99 at this
    rate, so the shift is what breaks the plain version. The baseline lets
    the rate go 10 times higher safely.

    `.clone().requires_grad_()` makes a fresh copy that records gradients,
    so each batch gets its own gradient in `.grad`.
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


    # TODO: Return the advantages of rewards, each reward minus the average of
    #   all of them.
    def advantages(rewards: Tensor) -> Tensor:
        ...


    # TODO: Return trials estimates of the expected reward's gradient, of shape
    #   (trials, arms). For each trial, make a fresh copy of preferences that
    #   records gradients, play episodes episodes with it, and weigh the log
    #   probabilities by the advantages when baseline is True, by the rewards
    #   when False. The estimate is minus the gradient of reinforce_loss.
    def gradient_samples(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        trials: int,
        baseline: bool,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Return the noise of samples as a plain number, the spread over the
    #   trials of each arm's estimate, averaged over the arms.
    def gradient_noise(samples: Tensor) -> float:
        ...


    # TODO: Finish reinforce1's training, so the weights are the advantages
    #   when baseline is True and the rewards when it is False.
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
        description="subtracts the average reward",
        inputs=(
            (torch.tensor([10.5, 9.0, 11.5, 10.0]),),
            (torch.tensor([-1.0, 2.0]),),
        ),
        reference=REFERENCE["advantages"],
        expect=expect_tensor,
    )
    checker.test(
        function="gradient_samples",
        description="gives one gradient estimate per trial",
        run=check_samples_shape_and_values,
    )
    check_on_inputs(
        checker=checker,
        function="gradient_noise",
        description="averages each arm's spread over the trials",
        inputs=(
            (torch.tensor([[1.0, 0.0, -2.0], [3.0, 0.0, 2.0], [2.0, 1.0, 0.0]]),),
            (torch.tensor([[0.5, -0.5], [1.5, 0.5]]),),
        ),
        reference=REFERENCE["gradient_noise"],
        expect=expect_close,
    )
    checker.test(
        function="train_bandit",
        description="weighs by the advantages when baseline is True",
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
        "gradient_samples": "In a loop, leaf = preferences.clone().requires_grad_(), then play with the leaf. Call "
        ".backward() on reinforce_loss of log_probabilities(leaf, actions) and the weights, keep -leaf.grad, and "
        "torch.stack the list.",
        "gradient_noise": "samples.std(dim=0) gives each arm's spread over the trials. Take its .mean() and .item().",
        "train_bandit": "advantages(rewards) if baseline else rewards.",
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
    Check the estimate against the truth. On the bandit, `expected_reward` is
    a sum of chances times averages, so autograd gives its exact gradient.
    REINFORCE's estimate from one batch gets close with enough episodes. Here
    is the largest gap over the 5 arms, at equal preferences, on 5 seeds.

    | Episodes in the batch | 10 | 1,000 | 100,000 |
    |---|---|---|---|
    | Largest gap | 0.095 to 0.19 | 0.016 to 0.047 | 0.0007 to 0.0025 |

    The exact gradient there is `[0, -0.14, 0.16, 0.04, -0.06]`. Each arm's
    entry is its chance, 0.2, times how far its average sits from the expected
    reward of 0.2.

    The estimate never asked the reward for a gradient. The reward only
    weighs log probabilities, so any score works, even one made with `==`.

    A speller shows it. The policy writes a word of 5 letters, each from its
    own softmax over 26 letters. The reward counts the letters that match a
    hidden word, `torch`. Counting with `==` gives whole numbers and no
    gradient, and the policy never sees the hidden word.

    ```python
    (torch.tensor([[19, 14, 17, 2, 7]]) == torch.tensor([19, 0, 17, 2, 0])).sum(dim=1)    # tensor([3])
    ```

    A word's log probability is the sum of its letters' log probabilities.
    Indexing with `torch.arange(5)` and the words together picks each
    position's row and each word's letter there. The arange lines up with the
    positions, the last dimension of the words, as the `picked` line in the
    file shows.

    Here is Adam at a learning rate of 0.1, 64 words a step, with the batch
    average as the baseline, on 5 seeds.

    | Steps | 10 | 20 | 40 | 80 |
    |---|---|---|---|---|
    | Average reward of drawn words | 0.54 to 0.58 | 1.49 to 1.57 | 4.49 to 4.54 | 4.93 to 4.94 |

    Random letters score 5 in 26, about 0.19. After 10 steps the likeliest
    word is already `torch` on every seed, though drawn words still miss most
    letters. The steps after that make the policy sure of it.

    This is how a model trains on scores like a passed test, a solved puzzle
    or a person's thumbs up. The rest of Section 11 does it to a language
    model. With the hidden word known, plain cross entropy on its letters
    would train faster. REINFORCE is for when the score is all there is.
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
    #   preferences, through a fresh copy that records gradients.
    def exact_gradient(
        preferences: Tensor,
        arm_means: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the largest gap, as a plain number, between one estimate of
    #   the gradient from episodes episodes, gradient_samples with one trial and
    #   no baseline, and exact_gradient.
    def estimate_gap(
        preferences: Tensor,
        arm_means: Tensor,
        episodes: int,
        generator: torch.Generator,
    ) -> float:
        ...


    # The speller's world. spell draws batch words from logits of shape (5, 26),
    # one row per position, and matches scores each word against the target.
''' + indented(SPELLER_PART) + '''


    # TODO: words has shape (batch, 5). Return one log probability per word,
    #   of shape (batch,), the sum over the positions of each letter's log
    #   softmax in its row of logits.
    def word_log_probs(
        logits: Tensor,
        words: Tensor,
    ) -> Tensor:
        ...


    # TODO: Train logits, zeros of shape (len(target), LETTERS) that record
    #   gradients, with torch.optim.Adam at lr. Each of steps steps spells batch
    #   words with generator, scores them with matches, and steps on
    #   reinforce_loss of their log probabilities weighed by the advantages.
    #   Return the trained logits detached.
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
            f"After 100 steps, the likeliest word of train_speller gets {right} of the 5 letters of grade right, "
            "but it should get all 5.",
        )


def check_reinforce3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="exact_gradient",
        description="takes the gradient of the expected reward",
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
        description="sums each word's letter log probabilities",
        inputs=(
            (Described(torch.randn(5, 26, generator=torch.Generator().manual_seed(809)), "logits of shape (5, 26)"), torch.tensor([[19, 14, 17, 2, 7], [0, 0, 0, 0, 0], [25, 1, 17, 3, 9]])),
            (Described(torch.randn(3, 4, generator=torch.Generator().manual_seed(810)), "logits of shape (3, 4)"), torch.tensor([[3, 0, 1], [2, 2, 2]])),
        ),
        reference=REFERENCE["word_log_probs"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_speller",
        description="steps Adam on REINFORCE with the advantages",
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
    title="Why, a score with no gradient",
    lesson=REINFORCE3_LESSON,
    source=REINFORCE3_SOURCE,
    solution=REINFORCE3_SOLUTION,
    hints={
        "exact_gradient": "leaf = preferences.clone().requires_grad_(). Call .backward() on expected_reward(leaf, "
        "arm_means) and return leaf.grad.",
        "estimate_gap": "gradient_samples(preferences, arm_means, episodes, 1, False, generator)[0] is the one "
        "estimate. Subtract exact_gradient, then .abs().max().item().",
        "word_log_probs": "torch.log_softmax(logits, dim=1), indexed with torch.arange(len(logits)) and words, then "
        ".sum(dim=1).",
        "train_speller": "logits = torch.zeros(len(target), LETTERS, requires_grad=True). In the loop, spell, then "
        "matches, then reinforce_loss of word_log_probs and advantages(rewards), then zero_grad, backward and step.",
    },
    introduces=(),
    check=check_reinforce3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)
