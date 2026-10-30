"""80_ppo, the probability ratio and PPO's clipped objective, generalized advantage estimation, and a PPO loop that balances cart poles written in plain PyTorch."""

from __future__ import annotations

import math
import textwrap

import torch
import torch.nn as nn
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    small_tensor,
)
from ..text import prose, source_file
from .policy_gradients import BANDIT_PART, PLAY_PART

PPO = Chapter(
    folder="80_ppo",
    title="PPO",
    readme=source_file(
        """
        # PPO

        REINFORCE uses each batch of episodes once. Proximal policy
        optimization reuses it for several steps, with a probability ratio
        that corrects for the policy's change and a clip that keeps the change
        small. This chapter builds its objective from the paper, adds
        generalized advantage estimation for long episodes, and trains a
        policy that balances a pole on a cart, all in plain PyTorch.

        ## Further reading

        - [Schulman and colleagues, proximal policy optimization algorithms](https://arxiv.org/abs/1707.06347)
        - [Schulman and colleagues, generalized advantage estimation](https://arxiv.org/abs/1506.02438)
        - [Barto, Sutton and Anderson, neuronlike adaptive elements, the cart pole's source](https://doi.org/10.1109/TSMC.1983.6313077)
        - [Gymnasium, the CartPole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
        """,
    ),
)

# The pieces later exercises and chapters are handed, written once so the files never drift apart.
PPO_LOSS_PART = '''
def probability_ratio(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
) -> Tensor:
    return torch.exp(new_log_probs - old_log_probs)


def ppo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    epsilon: float,
) -> Tensor:
    ratio = probability_ratio(new_log_probs, old_log_probs)
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)

    return -torch.min(ratio * advantages, clipped * advantages).mean()
'''

SLOPES_PART = '''
def objective_slopes(
    ratios: Tensor,
    advantage: float,
    epsilon: float,
) -> Tensor:
    leaf = ratios.clone().requires_grad_()
    clipped = leaf.clamp(1 - epsilon, 1 + epsilon)
    torch.min(leaf * advantage, clipped * advantage).sum().backward()

    return leaf.grad
'''

RETURNS_PART = '''
def discounted_returns(
    rewards: Tensor,
    dones: Tensor,
    gamma: float,
) -> Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])

    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1 - dones[t].float())
        returns[t] = running

    return returns
'''

GAE_PART = '''
def td_errors(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float,
) -> Tensor:
    return rewards + gamma * next_values * (1 - dones.float()) - values


def gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float,
) -> Tensor:
    deltas = td_errors(rewards, values, next_values, dones, gamma)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])

    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * lam * (1 - dones[t].float()) * running
        advantages[t] = running

    return advantages
'''

CARTPOLE_PART = '''
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
HALF_LENGTH = 0.5
PUSH = 10.0
TICK = 0.02
LONGEST = 500


class CartPole:
    """Many cart poles at once, one per row of states, with the physics of gym's CartPole-v1. A state is the cart's
    position and speed, then the pole's angle and how fast it turns. A cart that falls, or reaches 500 steps,
    starts over."""

    def __init__(
        self,
        count: int,
        generator: torch.Generator,
    ) -> None:
        self.generator = generator
        self.states = self.fresh(count)
        self.steps = torch.zeros(count, dtype=torch.long)

    def fresh(
        self,
        count: int,
    ) -> Tensor:
        return torch.rand(count, 4, generator=self.generator) * 0.1 - 0.05

    def step(
        self,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pushes each cart left for action 0 and right for action 1. Gives each cart's reward, 1 for every step,
        and whether its episode ended."""
        x, speed, angle, turn = self.states.unbind(dim=1)
        force = torch.where(actions == 1, PUSH, -PUSH)
        cos, sin = torch.cos(angle), torch.sin(angle)
        total = CART_MASS + POLE_MASS
        temp = (force + POLE_MASS * HALF_LENGTH * turn**2 * sin) / total
        angle_acceleration = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4 / 3 - POLE_MASS * cos**2 / total))
        acceleration = temp - POLE_MASS * HALF_LENGTH * angle_acceleration * cos / total

        self.states = torch.stack(
            [
                x + TICK * speed,
                speed + TICK * acceleration,
                angle + TICK * turn,
                turn + TICK * angle_acceleration,
            ],
            dim=1,
        )
        self.steps += 1

        fell = (self.states[:, 0].abs() > 2.4) | (self.states[:, 2].abs() > 12 * 2 * math.pi / 360)
        dones = fell | (self.steps >= LONGEST)
        self.states[dones] = self.fresh(int(dones.sum()))
        self.steps[dones] = 0

        return torch.ones(len(actions)), dones


def mlp(outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, outputs),
    )
'''

COLLECT_PART = '''
def collect(
    policy: nn.Module,
    value: nn.Module,
    carts: CartPole,
    steps: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    rollout: dict[str, list[Tensor]] = {name: [] for name in ("states", "actions", "log_probs", "values", "rewards", "dones")}

    with torch.no_grad():
        for _ in range(steps):
            states = carts.states
            log_chances = torch.log_softmax(policy(states), dim=1)
            actions = torch.multinomial(log_chances.exp(), 1, generator=generator).squeeze(1)
            rewards, dones = carts.step(actions)

            rollout["states"].append(states)
            rollout["actions"].append(actions)
            rollout["log_probs"].append(log_chances[torch.arange(len(actions)), actions])
            rollout["values"].append(value(states).squeeze(1))
            rollout["rewards"].append(rewards)
            rollout["dones"].append(dones)

        last_value = value(carts.states).squeeze(1)

    stacked = {name: torch.stack(parts) for name, parts in rollout.items()}
    stacked["next_values"] = torch.cat([stacked["values"][1:], last_value.unsqueeze(0)])

    return stacked
'''

COLLECT_STUB = (
    COLLECT_PART.replace("log_chances = torch.log_softmax(policy(states), dim=1)", "log_chances = ...")
    .replace("actions = torch.multinomial(log_chances.exp(), 1, generator=generator).squeeze(1)", "actions = ...")
    .replace('rollout["log_probs"].append(log_chances[torch.arange(len(actions)), actions])', 'rollout["log_probs"].append(...)')
)

UPDATE_PART = '''
def minibatch_loss(
    policy: nn.Module,
    value: nn.Module,
    states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    epsilon: float,
) -> Tensor:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_chances = torch.log_softmax(policy(states), dim=1)
    new_log_probs = log_chances[torch.arange(len(actions)), actions]
    value_loss = ((value(states).squeeze(1) - returns) ** 2).mean()

    return ppo_loss(new_log_probs, old_log_probs, advantages, epsilon) + 0.5 * value_loss


def ppo_update(
    policy: nn.Module,
    value: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Tensor],
    advantages: Tensor,
    returns: Tensor,
    epochs: int,
    minibatch: int,
    epsilon: float,
    generator: torch.Generator,
) -> float:
    states = rollout["states"].flatten(0, 1)
    actions = rollout["actions"].flatten()
    old_log_probs = rollout["log_probs"].flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()

    for _ in range(epochs):
        order = torch.randperm(len(states), generator=generator)

        for start in range(0, len(states), minibatch):
            batch = order[start : start + minibatch]
            loss = minibatch_loss(policy, value, states[batch], actions[batch], old_log_probs[batch], advantages[batch], returns[batch], epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()
'''

TRAIN_PART = '''
def train_ppo(
    updates: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    policy = mlp(2)
    value = mlp(1)
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    carts = CartPole(16, generator)

    for _ in range(updates):
        rollout = collect(policy, value, carts, 128, generator)
        advantages = gae(rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.99, 0.95)
        returns = advantages + rollout["values"]
        ppo_update(policy, value, optimizer, rollout, advantages, returns, 4, 256, 0.2, generator)

    return policy


def balance_time(
    policy: nn.Module,
    count: int,
    generator: torch.Generator,
) -> float:
    carts = CartPole(count, generator)
    standing = torch.ones(count, dtype=torch.bool)
    survived = torch.zeros(count)

    with torch.no_grad():
        for _ in range(LONGEST):
            _, dones = carts.step(policy(carts.states).argmax(dim=1))
            survived += standing.float()
            standing &= ~dones

    return survived.mean().item()
'''

TRAIN_STUB = TRAIN_PART.split("\n\n\ndef balance_time")[0].replace(
    """    for _ in range(updates):
        rollout = collect(policy, value, carts, 128, generator)
        advantages = gae(rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.99, 0.95)
        returns = advantages + rollout["values"]
        ppo_update(policy, value, optimizer, rollout, advantages, returns, 4, 256, 0.2, generator)

    return policy""",
    """    ...""",
)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def reference_namespace(*parts: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec("\n\n".join(parts), namespace)

    return namespace


REFERENCE = reference_namespace(
    BANDIT_PART,
    PLAY_PART,
    PPO_LOSS_PART,
    SLOPES_PART,
    RETURNS_PART,
    GAE_PART,
    CARTPOLE_PART,
    COLLECT_PART,
    UPDATE_PART,
    TRAIN_PART,
)


def reuse_one_batch(
    clip: bool,
    seed: int,
    steps: int,
) -> tuple[float, int]:
    """ppo1's run. One batch of 16 bandit episodes from equal preferences, then steps of Adam at 0.1 on it alone. Gives
    the largest chance the policy ends with and the arm that holds it."""
    generator = torch.Generator().manual_seed(seed)
    actions, rewards = REFERENCE["play"](torch.zeros(5), REFERENCE["ARM_MEANS"], 16, generator)
    old_log_probs = torch.log_softmax(torch.zeros(5), dim=0)[actions]
    advantages = rewards - rewards.mean()
    preferences = torch.zeros(5, requires_grad=True)
    optimizer = torch.optim.Adam([preferences], lr=0.1)

    for _ in range(steps):
        new_log_probs = torch.log_softmax(preferences, dim=0)[actions]

        if clip:
            loss = REFERENCE["ppo_loss"](new_log_probs, old_log_probs, advantages, 0.2)
        else:
            loss = -(REFERENCE["probability_ratio"](new_log_probs, old_log_probs) * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    chances = torch.softmax(preferences.detach(), dim=0)

    return chances.max().item(), int(chances.argmax())


def train_variant(
    updates: int,
    seed: int,
    epochs: int,
    epsilon: float,
    lam: float,
) -> nn.Module:
    """train_ppo with its epochs, clip and lambda free, for the tables of ppo2 and ppo3."""
    torch.manual_seed(seed)
    policy = REFERENCE["mlp"](2)
    value = REFERENCE["mlp"](1)
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    carts = REFERENCE["CartPole"](16, generator)

    for _ in range(updates):
        rollout = REFERENCE["collect"](policy, value, carts, 128, generator)
        advantages = REFERENCE["gae"](rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.99, lam)
        returns = advantages + rollout["values"]
        REFERENCE["ppo_update"](policy, value, optimizer, rollout, advantages, returns, epochs, 256, epsilon, generator)

    return policy


def scored(policy: nn.Module) -> float:
    return REFERENCE["balance_time"](policy, 16, torch.Generator().manual_seed(7))


def measure_ppo() -> dict[str, object]:
    """Every number of this chapter's lessons. About two minutes on a laptop CPU. Not run by any check."""
    seeds = range(5)
    results: dict[str, object] = {}

    for clip in (False, True):
        for steps in (10, 50):
            results[f"ppo1 clip {clip} steps {steps}"] = [reuse_one_batch(clip, seed, steps) for seed in range(10)]

    results["ppo1 slopes"] = [
        REFERENCE["objective_slopes"](torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5]), advantage, 0.2).tolist() for advantage in (1.0, -1.0)
    ]
    results["untrained policy"] = sorted(scored(seeded_networks(seed)[0]) for seed in range(40))

    for lam in (0.0, 0.95, 1.0):
        results[f"ppo2 lambda {lam}"] = [scored(train_variant(10, seed, 4, 0.2, lam)) for seed in seeds]

    for updates in (5, 10, 20, 30):
        results[f"ppo3 updates {updates}"] = [scored(train_variant(updates, seed, 4, 0.2, 0.95)) for seed in seeds]

    for epochs, epsilon in ((1, 0.2), (4, 0.2), (10, 0.2), (10, 1e9)):
        results[f"ppo3 epochs {epochs} epsilon {epsilon}"] = [scored(train_variant(10, seed, epochs, epsilon, 0.95)) for seed in seeds]

    return results


# ---------------------------------------------------------------------------
# ppo1
# ---------------------------------------------------------------------------

PPO1_LESSON = prose(
    """
    REINFORCE plays a batch, takes one step and throws the batch away. After
    the step the policy has changed, and the episodes came from the old one.
    Playing is often the slow part, so reusing each batch would pay.

    The **probability ratio** corrects for the change. It divides each played
    action's chance under the new policy by its chance under the old one that
    played it.

    ```text
    ratio = p_new(action) / p_old(action) = exp(log p_new - log p_old)
    ```

    Weighting each advantage by the ratio gives an objective whose gradient,
    at a ratio of 1, is REINFORCE's. Many steps on it overfit the batch. Here
    is one batch of 16 bandit episodes from 79_policy_gradients, used alone
    for many steps of Adam, on 10 seeds.

    | Steps on one batch | 10 | 50 |
    |---|---|---|
    | Largest chance, ratio times advantage | 0.36 to 0.65 | 0.96 to 0.99 |
    | Largest chance, PPO's clipped objective | 0.25 to 0.41 | 0.24 to 0.45 |

    Without a limit, 16 episodes make the policy sure of one arm. On 2 of the
    10 seeds it is arm 3, a worse arm the batch happened to favor.

    PPO, from Schulman and colleagues in 2017, clips the ratio, with epsilon a
    small number like the paper's 0.2.

    ```text
    objective = mean of min(ratio * A, clip(ratio, 1 - epsilon, 1 + epsilon) * A)
    ```

    Taking the smaller of the two makes it pessimistic, a lower bound on the
    plain objective. Past the clip, a change that would raise the objective
    stops counting, and one that lowers it still counts in full. Here is the
    slope of one sample's objective with respect to its ratio.

    | Ratio | 0.5 | 0.9 | 1.0 | 1.1 | 1.5 |
    |---|---|---|---|---|---|
    | Advantage +1 | 1 | 1 | 1 | 1 | 0 |
    | Advantage -1 | 0 | -1 | -1 | -1 | -1 |

    The clip doesn't stop a ratio at 1.2 outright, since other samples push
    the same preferences. It keeps one batch from deciding everything.
    """,
)

PPO1_SOURCE = source_file(
    '''
    # ppo1.py

    import torch
    from torch import Tensor

    log_new = torch.tensor([-0.5, -2.0])
    log_old = torch.tensor([-0.7, -1.9])
    ratios = (log_new - log_old).exp()                                 # tensor([1.2214, 0.9048])
    clipped = torch.tensor([0.5, 1.3]).clamp(0.8, 1.2)                 # tensor([0.8000, 1.2000])
    smaller = torch.min(torch.tensor([1.0, 5.0]), torch.tensor([2.0, 3.0]))    # tensor([1., 3.])


    # TODO: Return the probability ratio of each action, its chance under the
    #   new policy over its chance under the old, from their log probabilities.
    def probability_ratio(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return PPO's loss, minus the mean over the actions of the smaller of
    #   ratio times advantage and the ratio clipped to 1 - epsilon and
    #   1 + epsilon times advantage.
    def ppo_loss(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        epsilon: float,
    ) -> Tensor:
        ...


    # TODO: Return the slope of each sample's clipped objective, the smaller of
    #   ratio times advantage and the clipped ratio times advantage, with
    #   respect to its ratio. Take it through a fresh copy of ratios that
    #   records gradients.
    def objective_slopes(
        ratios: Tensor,
        advantage: float,
        epsilon: float,
    ) -> Tensor:
        ...
    ''',
)

PPO1_SOLUTION = source_file(
    '''
    # ppo1.py, a solution for comparison

    import torch
    from torch import Tensor

    log_new = torch.tensor([-0.5, -2.0])
    log_old = torch.tensor([-0.7, -1.9])
    ratios = (log_new - log_old).exp()
    clipped = torch.tensor([0.5, 1.3]).clamp(0.8, 1.2)
    smaller = torch.min(torch.tensor([1.0, 5.0]), torch.tensor([2.0, 3.0]))


''' + indented(PPO_LOSS_PART) + '''


''' + indented(SLOPES_PART) + '''
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


def log_probs_for_check(seed: int) -> Tensor:
    return -torch.rand(6, generator=torch.Generator().manual_seed(seed)) * 2


def check_ppo_loss_gradient(found: object) -> None:
    new_log_probs = log_probs_for_check(822).requires_grad_()
    loss = found(new_log_probs, log_probs_for_check(823), small_tensor(824, 6), 0.2)

    if not isinstance(loss, Tensor) or loss.grad_fn is None:
        raise Mismatch("ppo_loss gives a loss that records nothing, so no gradient reaches the new log probabilities. Keep it a tensor.")


def check_ppo1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="probability_ratio",
        description="divides the new chance by the old one",
        inputs=(
            (torch.tensor([-0.5, -2.0, -1.0]), torch.tensor([-0.7, -1.9, -1.0])),
            (log_probs_for_check(825), log_probs_for_check(826)),
        ),
        reference=REFERENCE["probability_ratio"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="ppo_loss",
        description="takes minus the mean of the smaller of the two terms",
        inputs=(
            (log_probs_for_check(827), log_probs_for_check(828), small_tensor(829, 6), 0.2),
            (torch.tensor([-0.1, -3.0, -0.2, -1.5]), torch.tensor([-1.0, -0.5, -0.4, -1.4]), torch.tensor([2.0, -1.0, -3.0, 0.5]), 0.1),
        ),
        reference=REFERENCE["ppo_loss"],
        expect=expect_loss_tensor,
    )
    checker.test(
        function="ppo_loss",
        description="keeps the gradient of the new log probabilities",
        run=check_ppo_loss_gradient,
    )
    check_on_inputs(
        checker=checker,
        function="objective_slopes",
        description="gives each sample's slope against its ratio",
        inputs=(
            (torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5]), 1.0, 0.2),
            (torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5]), -2.0, 0.2),
            (torch.tensor([0.7, 0.85, 1.05, 1.2]), 3.0, 0.1),
        ),
        reference=REFERENCE["objective_slopes"],
        expect=expect_tensor,
    )


PPO1 = Exercise(
    name="ppo1",
    chapter=PPO,
    title="The ratio and the clip",
    lesson=PPO1_LESSON,
    source=PPO1_SOURCE,
    solution=PPO1_SOLUTION,
    hints={
        "probability_ratio": "The ratio of two chances is the exp of the difference of their logs.",
        "ppo_loss": "ratio from probability_ratio, and ratio.clamp(1 - epsilon, 1 + epsilon). torch.min of the two "
        "times the advantages, then .mean() with a minus in front.",
        "objective_slopes": "leaf = ratios.clone().requires_grad_(). Build torch.min of leaf * advantage and the "
        "clamped leaf * advantage, .sum().backward(), and return leaf.grad.",
    },
    introduces=(),
    check=check_ppo1,
)


# ---------------------------------------------------------------------------
# ppo2
# ---------------------------------------------------------------------------

PPO2_LESSON = prose(
    """
    In the bandit an episode is one action, so its reward plainly belongs to
    it. ppo3's cart pole runs for hundreds of steps and pays 1 for each step
    the pole stays up. Which push deserves the credit?

    The **return** of a step adds its reward and every later one in the
    episode, each shrunk by a discount gamma per step.

    ```text
    G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
    ```

    As a baseline, a value network V learns to guess a state's return from
    the state alone. The TD error compares one real step against that guess.

    ```text
    delta_t = r_t + gamma * V(s_(t+1)) - V(s_t)
    ```

    GAE, from Schulman and colleagues in 2015, mixes the two. Its advantage
    adds up the TD errors ahead, each shrunk by gamma times lambda per step.
    One pass from the last step back to the first computes it.

    ```text
    A_t = delta_t + gamma * lambda * A_(t+1)
    ```

    At lambda 1 it is the return minus the guess, unbiased but noisy. At
    lambda 0 it is one TD error, quiet but only as right as V. PPO's paper
    uses gamma 0.99 and lambda 0.95.

    When an episode ends at step t, nothing after it belongs to it. The next
    state's value counts as 0, and the sums stop there. A done flag marks
    those steps.

    Here is ppo3's training after 10 updates, scored by how many steps the
    greedy policy keeps 16 poles up, out of 500, on 5 seeds.

    | Lambda | 0 | 0.95 | 1 |
    |---|---|---|---|
    | Steps balanced | 9 to 12 | 283 to 423 | 146 to 473 |

    At lambda 0 nothing is learned yet. Every step pays 1, so while V knows
    nothing every TD error looks alike. Lambda 1 learns, with the widest
    spread over seeds, and 0.95 lands between the two.
    """,
)

PPO2_SOURCE = source_file(
    '''
    # ppo2.py

    import torch
    from torch import Tensor

    dones = torch.tensor([False, True, False])
    keep = 1 - dones.float()                    # tensor([1., 0., 1.]), 0 where an episode ended
    backwards = list(reversed(range(3)))        # [2, 1, 0]


    # Every function takes tensors of shape (steps, carts), or (steps,) for one
    # cart, in the order the steps were played. dones holds True at a step
    # whose episode ended there.

    # TODO: Return the return of every step, its reward plus gamma times the
    #   next step's return, where an episode's last step has no next step.
    #   Work from the last step back to the first.
    def discounted_returns(
        rewards: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Tensor:
        ...


    # TODO: Return the TD error of every step, its reward plus gamma times the
    #   next state's value, minus its own value. Count the next value as 0 at
    #   a step whose episode ended.
    def td_errors(
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Tensor:
        ...


    # TODO: Return GAE's advantage of every step, its TD error plus gamma times
    #   lam times the next step's advantage, where an episode's last step has
    #   no next step. Work from the last step back to the first.
    def gae(
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        dones: Tensor,
        gamma: float,
        lam: float,
    ) -> Tensor:
        ...
    ''',
)

PPO2_SOLUTION = source_file(
    '''
    # ppo2.py, a solution for comparison

    import torch
    from torch import Tensor

    dones = torch.tensor([False, True, False])
    keep = 1 - dones.float()
    backwards = list(reversed(range(3)))


''' + indented(RETURNS_PART) + '''


''' + indented(GAE_PART) + '''
    ''',
)


def rollout_for_check(
    seed: int,
    steps: int,
    carts: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Rewards, values, next values and done flags of a made up rollout, with a few episodes ending in it."""
    generator = torch.Generator().manual_seed(seed)
    rewards = torch.randint(0, 3, (steps, carts), generator=generator).float()
    values = torch.randn(steps, carts, generator=generator) * 2
    next_values = torch.randn(steps, carts, generator=generator) * 2
    dones = torch.rand(steps, carts, generator=generator) < 0.25

    return rewards, values, next_values, dones


def described_rollout(
    seed: int,
    steps: int,
    carts: int,
) -> tuple[Described, Described, Described, Described]:
    shape = f"of shape ({steps}, {carts})"

    return tuple(
        Described(part, f"{name} {shape}")
        for name, part in zip(("rewards", "values", "next values", "dones"), rollout_for_check(seed, steps, carts))
    )


def check_ppo2(checker: Checker) -> None:
    first = described_rollout(830, 6, 2)
    second = described_rollout(831, 5, 3)
    one_cart = (
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        torch.tensor([3.0, 2.5, 1.0, 4.0, 3.5]),
        torch.tensor([2.5, 1.0, 4.0, 3.5, 3.0]),
        torch.tensor([False, False, True, False, False]),
    )

    check_on_inputs(
        checker=checker,
        function="discounted_returns",
        description="adds the discounted rewards ahead within each episode",
        inputs=(
            (first[0], first[3], 0.9),
            (second[0], second[3], 0.5),
            (one_cart[0], one_cart[3], 0.99),
        ),
        reference=REFERENCE["discounted_returns"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="td_errors",
        description="compares one step with the value's guess",
        inputs=(
            (*first, 0.9),
            (*second, 0.5),
        ),
        reference=REFERENCE["td_errors"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="gae",
        description="adds the TD errors ahead within each episode",
        inputs=(
            (*first, 0.9, 0.8),
            (*second, 0.5, 0.95),
            (*one_cart, 0.99, 0.95),
        ),
        reference=REFERENCE["gae"],
        expect=expect_tensor,
    )


PPO2 = Exercise(
    name="ppo2",
    chapter=PPO,
    title="Generalized advantage estimation",
    lesson=PPO2_LESSON,
    source=PPO2_SOURCE,
    solution=PPO2_SOLUTION,
    hints={
        "discounted_returns": "Start running at torch.zeros_like(rewards[0]). For t in reversed(range(len(rewards))), "
        "running becomes rewards[t] plus gamma times running times 1 - dones[t].float(). Store it in returns[t].",
        "td_errors": "rewards + gamma * next_values * (1 - dones.float()), minus values.",
        "gae": "Take td_errors first. Then walk back as discounted_returns does, adding gamma * lam times the running "
        "advantage, cut to 0 where dones[t] is True.",
    },
    introduces=(),
    check=check_ppo2,
)


# ---------------------------------------------------------------------------
# ppo3
# ---------------------------------------------------------------------------

PPO3_LESSON = prose(
    """
    Cart pole is the classic balancing task, from Barto, Sutton and Anderson
    in 1983. A cart rolls on a track with a pole hinged on top, and each step
    the policy pushes it left or right. An episode ends when the pole tips
    past 12 degrees or the cart rolls 2.4 from the center. Each step pays 1,
    up to 500.

    The file holds it in plain PyTorch, with the physics of gym's
    CartPole-v1. Sixteen carts run at once as the rows of one tensor, and a
    cart that falls starts over.

    One update of PPO has three parts.

    - Collect: run the carts for 128 steps under no_grad, keeping states, actions, log probabilities, values, rewards and done flags.
    - Advantages: GAE with gamma 0.99 and lambda 0.95. Advantage plus value is the return V learns to guess.
    - Update: 4 epochs over the 2,048 samples in shuffled minibatches of 256, each step on PPO's loss plus half V's squared error.

    That sum is the paper's combined objective, less its entropy bonus.
    Common implementations also scale each minibatch's advantages to mean 0
    and spread 1, which keeps the step size steady.

    The score is how many steps the greedy policy, always taking its likelier
    action, keeps 16 fresh poles up. An untrained policy usually falls within
    10 steps. Of 40 untrained ones, 32 fell within 18, and one held all 500
    by luck.

    | Updates | 5 | 10 | 20 | 30 |
    |---|---|---|---|---|
    | Steps balanced, 5 seeds | 265 to 434 | 283 to 423 | 280 to 500 | 205 to 476 |

    Ten thousand steps of play lift the policy from falling at once to
    balancing for hundreds of steps. After that it swings from update to
    update in place of settling at 500. RL's numbers are noisy, which is why
    papers report many seeds.

    Reuse and the clip both pay. Here are 10 updates with other numbers of
    epochs.

    | Epochs | 1 | 4 | 10 | 10, no clip |
    |---|---|---|---|---|
    | Steps balanced | 75 to 300 | 283 to 423 | 215 to 443 | 146 to 394 |

    The averages are 168, 344, 355 and 269. Five seeds are few, so the ranges
    overlap.
    """,
)

PPO3_SOURCE = source_file(
    '''
    # ppo3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    shuffled = torch.randperm(6, generator=torch.Generator().manual_seed(0))    # the numbers 0 to 5 in a random order
    flat = torch.zeros(128, 16, 4).flatten(0, 1)                                 # shape (2048, 4), steps and carts in one


    # The cart poles, the networks, and the pieces of ppo1 and ppo2.
''' + indented(CARTPOLE_PART) + '''


''' + indented(PPO_LOSS_PART) + '''


''' + indented(GAE_PART) + '''


    # TODO: Finish collect. At each step, take the log softmax of the policy's
    #   scores for the states, draw one action per cart from its chances with
    #   torch.multinomial(..., 1, generator=generator), and keep the log
    #   probability of each cart's drawn action.
''' + indented(COLLECT_STUB) + '''


    # TODO: Return the loss of one minibatch. Scale advantages to mean 0 and
    #   spread 1, with 1e-8 added to the spread. Take each action's new log
    #   probability from the policy, then ppo_loss plus 0.5 times the mean
    #   squared error of the value network's guesses against returns.
    def minibatch_loss(
        policy: nn.Module,
        value: nn.Module,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        epsilon: float,
    ) -> Tensor:
        ...


    # TODO: Flatten the rollout's states, actions and log probabilities, and
    #   advantages and returns, so steps and carts become one. For each of
    #   epochs epochs, shuffle with torch.randperm and generator, and take one
    #   optimizer step on minibatch_loss for each minibatch of that order.
    #   Return the last loss as a plain number.
    def ppo_update(
        policy: nn.Module,
        value: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: dict[str, Tensor],
        advantages: Tensor,
        returns: Tensor,
        epochs: int,
        minibatch: int,
        epsilon: float,
        generator: torch.Generator,
    ) -> float:
        ...


    # TODO: Finish train_ppo. Each of updates updates collects 128 steps,
    #   takes gae with gamma 0.99 and lam 0.95, adds the values to get the
    #   returns, and calls ppo_update with 4 epochs, minibatches of 256 and
    #   epsilon 0.2. Return the policy.
''' + indented(TRAIN_STUB) + '''


    # TODO: Return how many steps policy keeps count fresh carts up on
    #   average, made with generator, up to LONGEST. Push each cart toward its
    #   likelier action, count each cart's steps until its first done, and
    #   record nothing for autograd.
    def balance_time(
        policy: nn.Module,
        count: int,
        generator: torch.Generator,
    ) -> float:
        ...
    ''',
)

PPO3_SOLUTION = source_file(
    '''
    # ppo3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    shuffled = torch.randperm(6, generator=torch.Generator().manual_seed(0))
    flat = torch.zeros(128, 16, 4).flatten(0, 1)


''' + indented(CARTPOLE_PART) + '''


''' + indented(PPO_LOSS_PART) + '''


''' + indented(GAE_PART) + '''


''' + indented(COLLECT_PART) + '''


''' + indented(UPDATE_PART) + '''


''' + indented(TRAIN_PART) + '''
    ''',
)


def seeded_networks(seed: int) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)

    return REFERENCE["mlp"](2), REFERENCE["mlp"](1)


def check_collect(found: object) -> None:
    for steps, count, seed in ((5, 3, 832), (8, 2, 833)):
        call = f"collect(a policy, a value network, {count} carts, {steps}, a generator seeded with {seed})"
        rollouts = []

        for function in (found, REFERENCE["collect"]):
            policy, value = seeded_networks(seed)
            generator = torch.Generator().manual_seed(seed)
            rollouts.append(function(policy, value, REFERENCE["CartPole"](count, generator), steps, generator))

        rollout, expected = rollouts

        if not isinstance(rollout, dict):
            raise Mismatch(f"{call} should give a dict of tensors, as its last lines build.")

        for name, wanted in expected.items():
            expect_tensor(
                call=f'{call}["{name}"]',
                actual=rollout.get(name),
                expected=wanted,
            )


def minibatch_for_check(seed: int) -> tuple[Described, ...]:
    generator = torch.Generator().manual_seed(seed)
    policy, value = seeded_networks(seed)
    states = torch.randn(8, 4, generator=generator)
    actions = torch.randint(0, 2, (8,), generator=generator)
    old_log_probs = torch.log_softmax(torch.randn(8, 2, generator=generator), dim=1)[torch.arange(8), actions]
    advantages = torch.randn(8, generator=generator) * 3 + 1
    returns = torch.randn(8, generator=generator) * 5 + 20

    return (
        Described(policy, f"a policy seeded with {seed}"),
        Described(value, f"a value network seeded with {seed}"),
        Described(states, "states of shape (8, 4)"),
        Described(actions, "8 actions"),
        Described(old_log_probs, "8 old log probabilities"),
        Described(advantages, "8 advantages"),
        Described(returns, "8 returns"),
    )


def check_minibatch_gradient(found: object) -> None:
    policy, value, states, actions, old_log_probs, advantages, returns = [part.value for part in minibatch_for_check(834)]
    loss = found(policy, value, states, actions, old_log_probs, advantages, returns, 0.2)

    if not isinstance(loss, Tensor) or loss.grad_fn is None:
        raise Mismatch("minibatch_loss gives a loss that records nothing, so no gradient reaches the networks. Keep it a tensor.")

    loss.backward()

    if policy[0].weight.grad is None or value[0].weight.grad is None:
        raise Mismatch("minibatch_loss's loss reaches only one network. It should add the policy's ppo_loss and the value network's error.")


class swapped:
    """Puts reference functions into the learner's module for the length of a check, so one function is tested alone."""

    def __init__(
        self,
        found: object,
        names: tuple[str, ...],
    ) -> None:
        self.namespace = found.__globals__
        self.names = names
        self.saved: dict[str, object] = {}

    def __enter__(self) -> None:
        for name in self.names:
            self.saved[name] = self.namespace.get(name)
            self.namespace[name] = REFERENCE[name]

    def __exit__(self, *_: object) -> None:
        for name in self.names:
            self.namespace[name] = self.saved[name]


def rollout_and_targets(seed: int) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    policy, value = seeded_networks(seed)
    generator = torch.Generator().manual_seed(seed)
    rollout = REFERENCE["collect"](policy, value, REFERENCE["CartPole"](3, generator), 4, generator)
    advantages = REFERENCE["gae"](rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.99, 0.95)

    return rollout, advantages, advantages + rollout["values"]


def check_ppo_update(found: object) -> None:
    rollout, advantages, returns = rollout_and_targets(835)
    results = []

    with swapped(found, ("minibatch_loss",)):
        for function in (found, REFERENCE["ppo_update"]):
            policy, value = seeded_networks(836)
            optimizer = torch.optim.SGD([*policy.parameters(), *value.parameters()], lr=0.05)
            loss = function(policy, value, optimizer, rollout, advantages, returns, 2, 4, 0.2, torch.Generator().manual_seed(837))
            results.append((loss, policy[0].weight.detach().clone(), value[0].weight.detach().clone()))

    (loss, policy_weight, value_weight), (expected_loss, expected_policy, expected_value) = results
    call = "ppo_update(a policy, a value network, SGD, a rollout of 4 steps of 3 carts, 2 epochs, minibatches of 4)"

    expect_close(
        call=f"the last loss of {call}",
        actual=loss,
        expected=expected_loss,
        tolerance=1e-5,
    )
    expect_tensor(
        call=f"the policy's first weights after {call}",
        actual=policy_weight,
        expected=expected_policy,
    )
    expect_tensor(
        call=f"the value network's first weights after {call}",
        actual=value_weight,
        expected=expected_value,
    )


def check_train_ppo(found: object) -> None:
    with swapped(found, ("collect", "ppo_update")):
        policy = found(2, 838)

    if not isinstance(policy, nn.Module):
        raise Mismatch(f"train_ppo(2, 838) gives {type(policy).__name__}, but it should give the trained policy.")

    expect_tensor(
        call="the policy's first weights after train_ppo(2, 838)",
        actual=policy[0].weight.detach(),
        expected=REFERENCE["train_ppo"](2, 838)[0].weight.detach(),
    )


def check_ppo_learns(found: object) -> None:
    # Over 20 seeds the reference reached 181 to 489 steps after 10 updates. Seed 839's untrained policy falls in 9.5.
    steps = REFERENCE["balance_time"](found(10, 839), 16, torch.Generator().manual_seed(7))

    if steps < 100:
        raise Mismatch(f"After 10 updates of train_ppo, the greedy policy keeps 16 poles up for {steps:.1f} steps on average, but it should pass 100.")


def check_balance_time(found: object) -> None:
    for name, policy in (("an untrained policy", seeded_networks(840)[0]), ("a policy after 5 updates", REFERENCE["train_ppo"](5, 841))):
        grad_seen: list[bool] = []
        hook = policy.register_forward_pre_hook(lambda module, inputs: grad_seen.append(torch.is_grad_enabled()))

        try:
            actual = found(policy, 8, torch.Generator().manual_seed(842))
        finally:
            hook.remove()

        expect_close(
            call=f"balance_time({name}, 8, a generator seeded with 842)",
            actual=actual,
            expected=REFERENCE["balance_time"](policy, 8, torch.Generator().manual_seed(842)),
        )

        if any(grad_seen):
            raise Mismatch("balance_time recorded for autograd. Run the carts inside torch.no_grad().")


def check_ppo3(checker: Checker) -> None:
    checker.test(
        function="collect",
        description="draws each cart's action and keeps its log probability",
        run=check_collect,
    )
    check_on_inputs(
        checker=checker,
        function="minibatch_loss",
        description="adds PPO's loss and half the value error",
        inputs=(
            (*minibatch_for_check(843), 0.2),
            (*minibatch_for_check(844), 0.1),
        ),
        reference=REFERENCE["minibatch_loss"],
        expect=expect_loss_tensor,
    )
    checker.test(
        function="minibatch_loss",
        description="reaches both networks",
        run=check_minibatch_gradient,
    )
    checker.test(
        function="ppo_update",
        description="takes shuffled minibatch steps for every epoch",
        run=check_ppo_update,
    )
    checker.test(
        function="train_ppo",
        description="collects, takes GAE, then updates",
        run=check_train_ppo,
    )
    checker.test(
        function="balance_time",
        description="counts the steps until each cart's first fall",
        run=check_balance_time,
    )
    checker.test(
        function="train_ppo",
        description="learns to balance within 10 updates",
        run=check_ppo_learns,
    )


PPO3 = Exercise(
    name="ppo3",
    chapter=PPO,
    title="Balancing a pole with PPO",
    lesson=PPO3_LESSON,
    source=PPO3_SOURCE,
    solution=PPO3_SOLUTION,
    hints={
        "collect": "log_chances = torch.log_softmax(policy(states), dim=1). Draw from log_chances.exp() and .squeeze(1) "
        "the draws. Index log_chances with torch.arange(len(actions)) and actions.",
        "minibatch_loss": "Index the policy's log softmax with torch.arange(len(actions)) and actions. "
        "value(states).squeeze(1) are the guesses.",
        "ppo_update": "rollout[\"states\"].flatten(0, 1), the others .flatten(). order = torch.randperm(len(states), "
        "generator=generator), and batch = order[start : start + minibatch] for start in range(0, len(states), "
        "minibatch). Zero the gradients before each backward.",
        "train_ppo": "rollout = collect(policy, value, carts, 128, generator). gae takes the rollout's rewards, values, "
        "next_values and dones. returns = advantages + rollout[\"values\"].",
        "balance_time": "Make CartPole(count, generator), a standing mask of ones as bools and a survived count of "
        "zeros. Each step, add standing to survived, then set standing to False where dones is True.",
    },
    introduces=(),
    check=check_ppo3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
