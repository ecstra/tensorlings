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

        REINFORCE uses each batch of episodes for one training step and then
        throws it away. Proximal policy optimization, PPO, reuses each batch
        for several steps. A probability ratio corrects for how much the
        policy has changed since the batch was played, and a clip keeps that
        change small. This chapter builds PPO's objective from the paper. It
        adds generalized advantage estimation, which assigns credit to each
        step of a long episode. Then it trains a policy that balances a pole on
        a cart, all in plain PyTorch.

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
        """Pushes each cart left for action 0 and right for action 1. Returns each cart's reward, which is 1 for
        every step, and whether its episode ended."""
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


def networks() -> tuple[nn.Sequential, nn.Sequential]:
    """Returns the policy, which outputs 2 scores, and the value network, which outputs 1. Both are initialized the
    way Stable Baselines3 does it. The weights are orthogonal, scaled by sqrt(2) in the hidden layers, by 0.01 in
    the policy's last layer and by 1 in the value network's last layer. The biases start at zero."""
    made = []

    for outputs, last_gain in ((2, 0.01), (1, 1.0)):
        network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, outputs),
        )

        for layer, gain in zip(network[::2], (math.sqrt(2), math.sqrt(2), last_gain)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

        made.append(network)

    return made[0], made[1]
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
            nn.utils.clip_grad_norm_([*policy.parameters(), *value.parameters()], 0.5)
            optimizer.step()

    return loss.item()
'''

TRAIN_PART = '''
def train_ppo(
    updates: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    policy, value = networks()
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=1e-3, eps=1e-5)
    generator = torch.Generator().manual_seed(seed)
    carts = CartPole(8, generator)

    for update in range(updates):
        left = 1 - update / updates
        optimizer.param_groups[0]["lr"] = 1e-3 * left
        rollout = collect(policy, value, carts, 32, generator)
        advantages = gae(rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.98, 0.8)
        returns = advantages + rollout["values"]
        ppo_update(policy, value, optimizer, rollout, advantages, returns, 20, 256, 0.2 * left, generator)

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
    """    for update in range(updates):
        left = 1 - update / updates
        optimizer.param_groups[0]["lr"] = 1e-3 * left
        rollout = collect(policy, value, carts, 32, generator)
        advantages = gae(rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.98, 0.8)
        returns = advantages + rollout["values"]
        ppo_update(policy, value, optimizer, rollout, advantages, returns, 20, 256, 0.2 * left, generator)

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


def scored(
    policy: nn.Module,
    count: int = 16,
    seed: int = 7,
) -> float:
    return REFERENCE["balance_time"](policy, count, torch.Generator().manual_seed(seed))


def train_variant(
    updates: int,
    seed: int,
    epochs: int,
    epsilon: float,
    lam: float,
    every: int = 0,
) -> tuple[nn.Module, nn.Module, list[float]]:
    """train_ppo with its epochs, clip and lambda free, for the tables of ppo2 and ppo3. With 20, 0.2 and 0.8 it is
    train_ppo itself. Gives the policy, the value network and, when every is set, the policy's score after every that
    many updates, which draws on a generator of its own and leaves the training as it was."""
    torch.manual_seed(seed)
    policy, value = REFERENCE["networks"]()
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=1e-3, eps=1e-5)
    generator = torch.Generator().manual_seed(seed)
    carts = REFERENCE["CartPole"](8, generator)
    scores = []

    for update in range(updates):
        left = 1 - update / updates
        optimizer.param_groups[0]["lr"] = 1e-3 * left
        rollout = REFERENCE["collect"](policy, value, carts, 32, generator)
        advantages = REFERENCE["gae"](rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.98, lam)
        returns = advantages + rollout["values"]
        REFERENCE["ppo_update"](policy, value, optimizer, rollout, advantages, returns, epochs, 256, epsilon * left, generator)

        if every and (update + 1) % every == 0:
            scores.append(scored(policy))

    return policy, value, scores


def lambda_bias_and_spread(
    seed: int,
    starts: int = 32,
    copies: int = 256,
    horizon: int = 300,
) -> dict[float, tuple[float, float]]:
    """ppo2's table. A policy and value network trained 20 updates, then copies rollouts of horizon steps from each
    of starts states the policy reached. For each lambda, GAE's estimate of a start state's return, its advantage
    plus its value, against the average discounted return of separate rollouts from the same state. Gives how far
    the estimates' average misses that, and how far one estimate spreads, each averaged over the states."""
    policy, value, _ = train_variant(20, seed, 20, 0.2, 0.8)
    generator = torch.Generator().manual_seed(1000 + seed)
    carts = REFERENCE["CartPole"](starts, generator)
    REFERENCE["collect"](policy, value, carts, 40, generator)
    start_states = carts.states.clone()

    def rollouts() -> dict[str, Tensor]:
        many = REFERENCE["CartPole"](starts * copies, generator)
        many.states = start_states.repeat_interleave(copies, dim=0)
        many.steps.zero_()

        return REFERENCE["collect"](policy, value, many, horizon, generator)

    # The true return comes from rollouts of its own, so lambda 1's estimates aren't scored against themselves.
    apart = rollouts()
    truth = REFERENCE["discounted_returns"](apart["rewards"], apart["dones"], 0.98)[0].view(starts, copies).mean(dim=1)
    rollout = rollouts()
    results = {}

    for lam in (0.0, 0.5, 0.8, 0.95, 1.0):
        advantages = REFERENCE["gae"](rollout["rewards"], rollout["values"], rollout["next_values"], rollout["dones"], 0.98, lam)
        estimates = (advantages[0] + rollout["values"][0]).view(starts, copies)
        results[lam] = (
            (estimates.mean(dim=1) - truth).abs().mean().item(),
            estimates.std(dim=1).mean().item(),
        )

    return results


def measure_ppo() -> dict[str, object]:
    """Every number of this chapter's lessons, with 4 threads. About 6 minutes on a busy laptop CPU. Not run by any
    check."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        seeds = range(5)
        results: dict[str, object] = {}

        for clip in (False, True):
            for steps in (10, 50):
                results[f"ppo1 clip {clip} steps {steps}"] = [reuse_one_batch(clip, seed, steps) for seed in range(10)]

        results["ppo1 slopes"] = [
            REFERENCE["objective_slopes"](torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5]), advantage, 0.2).tolist() for advantage in (1.0, -1.0)
        ]
        results["untrained policy"] = sorted(scored(seeded_networks(seed)[0]) for seed in range(40))
        results["ppo2 lambda"] = [lambda_bias_and_spread(seed) for seed in range(3)]

        for epochs, epsilon in ((1, 0.2), (4, 0.2), (20, 0.2), (20, 1e9)):
            runs = [train_variant(390, seed, epochs, epsilon, 0.8, every=39) for seed in seeds]
            results[f"ppo3 epochs {epochs} epsilon {epsilon} curves"] = [scores for _, _, scores in runs]
            results[f"ppo3 epochs {epochs} epsilon {epsilon} on 100 carts"] = [scored(policy, 100, 12345) for policy, _, _ in runs]

        results["ppo3 check runs of 20 updates"] = [scored(train_variant(20, seed, 20, 0.2, 0.8)[0]) for seed in (*range(10), 839)]

        return results
    finally:
        torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# ppo1
# ---------------------------------------------------------------------------

PPO1_LESSON = prose(
    """
    REINFORCE plays a batch of episodes, takes one training step and throws
    the batch away. The reason is that after the step the policy has
    changed, but the episodes were played by the old policy. Playing
    episodes is often the slow part of training, so reusing each batch
    would save a lot of time.

    The **probability ratio** corrects for the change. For each played
    action, it divides the action's probability under the new policy by its
    probability under the old policy, the one that played it.

    ```text
    ratio = p_new(action) / p_old(action) = exp(log p_new - log p_old)
    ```

    Multiply each advantage by its ratio and take the mean. At a ratio of 1,
    the gradient of this objective equals REINFORCE's gradient. But taking
    many steps on it overfits the batch. The table uses one batch of 16
    bandit episodes from 79_policy_gradients. It takes many steps of Adam on
    that batch alone, on 10 seeds. Each cell is the largest probability the
    final policy gives to any arm, as a range over the seeds.

    | Steps on one batch | 10 | 50 |
    |---|---|---|
    | Ratio times advantage | 0.36 to 0.65 | 0.96 to 0.99 |
    | PPO's clipped objective | 0.25 to 0.41 | 0.24 to 0.45 |

    Without a limit, 16 episodes are enough to make the policy almost
    certain of one arm. On 2 of the 10 seeds that arm is arm 3, a worse arm
    that the batch happened to favor.

    PPO, from Schulman and colleagues in 2017, clips the ratio to the range
    from `1 - epsilon` to `1 + epsilon`. Here epsilon is a small number, like
    the paper's 0.2.

    ```text
    objective = mean of min(ratio * A, clip(ratio, 1 - epsilon, 1 + epsilon) * A)
    ```

    Taking the smaller of the two terms makes the objective pessimistic. It
    is never larger than the unclipped objective, `ratio * A`. Once the
    ratio is outside the clip range, a change that would raise the objective
    no longer counts. A change that would lower it still counts in full. The
    table shows the derivative of one sample's objective with respect to its
    ratio.

    | Ratio | 0.5 | 0.9 | 1.0 | 1.1 | 1.5 |
    |---|---|---|---|---|---|
    | Advantage +1 | 1 | 1 | 1 | 1 | 0 |
    | Advantage -1 | 0 | -1 | -1 | -1 | -1 |

    The clip does not stop a ratio from going past 1.2, because other
    samples still push on the same preferences. What the clip does is keep a
    single batch from changing the policy too much.
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


    # TODO: Return the probability ratio of each action, which is its
    #   probability under the new policy divided by its probability under the
    #   old policy. The inputs are log probabilities, so compute the ratio from
    #   them.
    def probability_ratio(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return PPO's loss, as a tensor with one number. For each action,
    #   compute two terms, ratio * advantage and clipped_ratio * advantage.
    #   clipped_ratio is the ratio clamped to the range from 1 - epsilon to
    #   1 + epsilon. Take the smaller term for each action, then return minus
    #   the mean over the actions.
    #   Example: ppo_loss(torch.tensor([-0.5, -2.0]), torch.tensor([-0.7, -1.9]),
    #   torch.tensor([1.0, -1.0]), 0.2) returns tensor(-0.1476).
    def ppo_loss(
        new_log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        epsilon: float,
    ) -> Tensor:
        ...


    # TODO: Return the derivative of each sample's clipped objective with
    #   respect to its ratio. One sample's objective is the smaller of
    #   ratio * advantage and clipped_ratio * advantage. Make a fresh copy of
    #   ratios that records gradients, compute the objectives from it, sum them
    #   and backpropagate. Return the copy's gradient.
    #   Example: objective_slopes(torch.tensor([0.5, 1.0, 1.5]), 1.0, 0.2)
    #   returns tensor([1., 1., 0.]).
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
        raise Mismatch(
            "ppo_loss returns a loss that autograd did not record, so no gradient can reach the new log probabilities. "
            "Return a tensor computed from the inputs, without .item().",
        )


def check_ppo1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="probability_ratio",
        description="divides the new probability by the old one",
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
        description="returns each sample's derivative with respect to its ratio",
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
        "probability_ratio": "The ratio of two probabilities is the exp of the difference of their logs.",
        "ppo_loss": "Get the ratio from probability_ratio and the clipped ratio from ratio.clamp(1 - epsilon, "
        "1 + epsilon). Multiply each by the advantages and take torch.min of the two products. Then take .mean() and "
        "put a minus in front.",
        "objective_slopes": "Make a leaf from ratios.clone() and requires_grad_. Compute torch.min of leaf * advantage "
        "and the clamped leaf times advantage. Call .sum().backward() on it, and return leaf.grad.",
    },
    introduces=(),
    check=check_ppo1,
)


# ---------------------------------------------------------------------------
# ppo2
# ---------------------------------------------------------------------------

PPO2_LESSON = prose(
    """
    In the bandit an episode is one action, so the reward belongs to that one
    action. ppo3's cart pole runs for hundreds of steps and gives a reward of
    1 for each step the pole stays up. So it is not clear which pushes
    deserve the credit for keeping the pole up.

    The **return** of a step is its reward plus every later reward in the
    episode. Each later reward is multiplied by a discount factor gamma once
    per step of delay, so rewards far in the future count less.

    ```text
    G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
    ```

    For a baseline, a second network V, called the value network, learns to
    predict a state's return from the state alone. The TD error, short for
    temporal difference error, compares one real step with that prediction.
    It is the reward actually received plus the discounted prediction for
    the next state, minus the prediction for this state.

    ```text
    delta_t = r_t + gamma * V(s_(t+1)) - V(s_t)
    ```

    GAE, generalized advantage estimation, from Schulman and colleagues in
    2015, mixes the two. Its advantage for a step is a sum of the TD errors
    from that step onward. Each later TD error is multiplied by
    `gamma * lambda` once per step of delay. One pass from the last step
    back to the first computes it with this recurrence.

    ```text
    A_t = delta_t + gamma * lambda * A_(t+1)
    ```

    At lambda 1 the advantage equals the return minus V's prediction. That
    estimate is right on average, which is called unbiased, but it is noisy.
    At lambda 0 the advantage is a single TD error. That has little noise,
    but it is only as accurate as V. PPO's paper uses gamma 0.99 and lambda
    0.95.

    When an episode ends at step t, the steps after t belong to a new
    episode. So the next state's value counts as 0, and the sums stop at
    step t. A done flag marks the steps where an episode ended. ppo3's carts
    also stop when they reach 500 steps. Gymnasium calls that a truncation
    and treats it differently from a fall. Here both count as the end of an
    episode.

    The table measures this tradeoff. ppo3's loop trained a policy and V for
    20 updates. Then each of 32 states the policy reached was played out 256
    times, for 300 steps each. For each lambda, GAE estimates the return of
    a start state as its advantage plus V. The estimate is compared with the
    average return of 256 separate playouts from the same state. Each cell
    is a range over 3 seeds, measured in steps of reward.

    | Lambda | 0 | 0.5 | 0.8 | 0.95 | 1 |
    |---|---|---|---|---|---|
    | Error of the average estimate | 6.4 to 14.0 | 6.1 to 13.5 | 5.0 to 12.2 | 2.5 to 8.2 | 0.6 |
    | Standard deviation of one estimate | 0.05 to 0.16 | 0.08 to 0.31 | 0.53 to 1.24 | 3.1 to 4.4 | 8.9 to 10.0 |

    The error of the average comes from V's mistakes. Lambda 0 inherits all
    of V's error, and lambda 1 removes it. The standard deviation comes from
    the randomness of a single playout. Lambda 0 is not affected by it, and
    lambda 1 inherits all of it. The 0.6 left at lambda 1 is the noise from
    averaging only 256 playouts. ppo3 uses lambda 0.8, the value that RL
    Baselines3 Zoo tuned for cart pole.
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
    # cart, with the steps in the order they were played. dones is True at
    # each step where an episode ended.

    # TODO: Return the return of every step, as a tensor with the shape of
    #   rewards. A step's return is its reward plus gamma times the next
    #   step's return. At a step where dones is True the episode ended, so the
    #   next step's return counts as 0. Work from the last step back to the
    #   first, starting from a next return of 0.
    #   Example: discounted_returns(torch.tensor([1.0, 1.0, 1.0]),
    #   torch.tensor([False, True, False]), 0.5) returns
    #   tensor([1.5000, 1.0000, 1.0000]).
    def discounted_returns(
        rewards: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Tensor:
        ...


    # TODO: Return the TD error of every step. It is the step's reward plus
    #   gamma times the next state's value, minus the step's own value. At a
    #   step where dones is True, count the next state's value as 0.
    def td_errors(
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Tensor:
        ...


    # TODO: Return the GAE advantage of every step. A step's advantage is its
    #   TD error plus gamma * lam times the next step's advantage. At a step
    #   where dones is True, the next step's advantage counts as 0. Work from
    #   the last step back to the first, as in discounted_returns.
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
        description="sums the discounted later rewards within each episode",
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
        description="compares one real step with the value network's prediction",
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
        description="sums the discounted later TD errors within each episode",
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
        "set running to rewards[t] plus gamma times running times (1 - dones[t].float()). Store it in returns[t].",
        "td_errors": "Multiply next_values by (1 - dones.float()) to make them 0 where an episode ended. Then add "
        "gamma times that to the rewards, and subtract values.",
        "gae": "Compute td_errors first. Then loop backwards as discounted_returns does. Each step's advantage is its TD "
        "error plus gamma * lam times the running advantage, which counts as 0 where dones[t] is True.",
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
    in 1983. A cart rolls on a track with a pole hinged on top. At each step
    the policy pushes the cart left or right. An episode ends when the pole
    tilts more than 12 degrees from upright, or the cart moves more than 2.4
    units from the center. Each step gives a reward of 1, for at most 500
    steps.

    The file implements cart pole in plain PyTorch, with the same physics as
    gym's CartPole-v1. The one difference is that reaching 500 steps counts
    as the end of an episode, like a fall, as ppo2 explained. Eight carts
    run at once, one per row of a tensor, and a cart whose episode ends
    starts over.

    Stable Baselines3 is a widely used PyTorch library of reinforcement
    learning methods. RL Baselines3 Zoo is a collection of settings tuned
    for each task with that library. This file uses the Zoo's settings for
    CartPole-v1, with the library's defaults for everything else. One update
    of PPO has three parts.

    - Collect: run the carts for 32 steps inside `torch.no_grad()`, and save the states, actions, log probabilities, values, rewards and done flags.
    - Advantages: compute GAE with gamma 0.98 and lambda 0.8. Each step's advantage plus its value is the return that V learns to predict.
    - Update: run 20 epochs, each one optimizer step on all 256 samples from 32 steps of 8 carts, with gradients clipped to norm 0.5.

    Over the run, the learning rate falls linearly from 1e-3 to 0, and
    epsilon falls linearly from 0.2 to 0. So the last updates change the
    policy less and less. The loss is PPO's loss plus half of V's mean
    squared error. That follows the combined objective in the paper, without
    its entropy bonus. Each minibatch's advantages are scaled to mean 0 and
    standard deviation 1.
    The file gives you the networks, initialized the way Stable Baselines3
    initializes them.

    The score is the average number of steps that the greedy policy keeps 16
    fresh poles up. The greedy policy always takes the action with the
    higher probability. Of 40 untrained policies, 17 scored 12 steps or
    fewer, and one scored the full 500 by luck.

    On all 5 seeds, the policy first keeps all 16 poles up for 500 steps
    within 117 updates, which is about 30,000 steps of play. After all 390
    updates, about 100,000 steps of play, every seed's policy keeps 100
    fresh poles up for all 500 steps.

    Reusing each batch for many epochs helps, and the clip keeps that reuse
    safe. The table runs the same 390 updates with other numbers of epochs,
    scored on 100 fresh poles.

    | Epochs | 1 | 4 | 20 | 20, no clip |
    |---|---|---|---|---|
    | Steps balanced, 5 seeds | 116 to 145 | 108 to 500 | 500 on all 5 | 9 to 500 |

    With one epoch, most of the information in each batch goes unused. Four
    epochs reach 499.8 or more on 3 seeds, but stay below 175 on the other
    2. Twenty epochs reach 500 on all 5 seeds. Without the clip, 20 epochs
    ruined one seed, whose final policy let the poles fall within 9 steps.
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
    flat = torch.zeros(32, 8, 4).flatten(0, 1)                                   # shape (256, 4), steps and carts merged into one dimension
    share_left = 1 - 3 / 10                                                      # 0.7, the fraction of 10 updates still to run at update 3


    # The cart poles, the networks, and the pieces of ppo1 and ppo2.
''' + indented(CARTPOLE_PART) + '''


''' + indented(PPO_LOSS_PART) + '''


''' + indented(GAE_PART) + '''


    # TODO: Finish collect by filling in its three ... placeholders. At each
    #   step, set log_chances to the log softmax of the policy's scores for
    #   the states, over the 2 actions. Draw one action per cart from its
    #   probabilities with torch.multinomial(..., 1, generator=generator), as
    #   a tensor of shape (carts,). Then save the log probability of each
    #   cart's drawn action.
''' + indented(COLLECT_STUB) + '''


    # TODO: Return the loss of one minibatch. First scale advantages to mean 0
    #   and standard deviation 1, by subtracting their mean and dividing by
    #   their standard deviation plus 1e-8. Take each action's new log
    #   probability from the policy's log softmax for states. Return ppo_loss
    #   plus 0.5 times the mean squared error between the value network's
    #   predictions for states and returns.
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


    # TODO: Train both networks on one rollout and return the last loss as a
    #   Python float. First merge the steps and carts dimensions of the
    #   rollout's states, actions and log probabilities, and of advantages and
    #   returns, so each sample is one row. The states become shape
    #   (steps * carts, 4), and the rest become shape (steps * carts,). Then
    #   run `epochs` epochs. In each epoch, shuffle the sample indices with
    #   torch.randperm and generator, and split that order into minibatches
    #   of size minibatch. Take one optimizer step on minibatch_loss for each
    #   minibatch. Before each step, clip the gradients of both networks
    #   together to a total norm of 0.5, as clipping1 did.
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


    # TODO: Finish train_ppo. Loop over the updates. At each update, compute
    #   left, the fraction of updates still to run, which is 1 at the first
    #   update. Set the learning rate to 1e-3 * left in
    #   optimizer.param_groups[0]["lr"]. Collect 32 steps, compute gae with
    #   gamma 0.98 and lam 0.8, and add the rollout's values to the advantages
    #   to get the returns. Call ppo_update with 20 epochs, minibatches of 256
    #   and epsilon 0.2 * left. After the loop, return the policy.
''' + indented(TRAIN_STUB) + '''


    # TODO: Return the average number of steps that policy keeps `count` fresh
    #   carts up, at most LONGEST, as a Python float. Make the carts with
    #   CartPole(count, generator). At each step, push each cart with its more
    #   likely action, the argmax of the policy's scores. Count each cart's
    #   steps up to and including its first done, and stop counting that cart
    #   after it. Run inside torch.no_grad(), so autograd records nothing.
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
    flat = torch.zeros(32, 8, 4).flatten(0, 1)
    share_left = 1 - 3 / 10


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

    return REFERENCE["networks"]()


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
            raise Mismatch(f"{call} should return a dict of tensors, which its last lines build.")

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


# minibatch_loss with the guesses left at shape (rows, 1), so the likeliest slip there can be named.
UNSQUEEZED_REFERENCE = reference_namespace(
    PPO_LOSS_PART,
    UPDATE_PART.replace("value(states).squeeze(1) - returns", "value(states) - returns"),
)


def check_minibatch_loss(found: object) -> None:
    for seed, epsilon in ((843, 0.2), (844, 0.1)):
        parts = minibatch_for_check(seed)
        values = [part.value for part in parts]
        call = f"minibatch_loss({', '.join(part.text for part in parts)}, {epsilon})"
        actual = found(*values, epsilon)
        expected = REFERENCE["minibatch_loss"](*values, epsilon).detach()
        unsqueezed = UNSQUEEZED_REFERENCE["minibatch_loss"](*values, epsilon).detach()
        guesses_unsqueezed = (
            isinstance(actual, Tensor)
            and actual.dim() == 0
            and torch.allclose(actual.detach(), unsqueezed)
            and not torch.allclose(actual.detach(), expected)
        )

        if guesses_unsqueezed:
            raise Mismatch(
                f"{call} is {actual.item():.4f}, which computes the value error between every prediction and every "
                "return. value(states) has shape (8, 1), so subtracting returns of shape (8,) broadcasts to an (8, 8) "
                "grid. Squeeze the predictions to shape (8,) first.",
            )

        expect_loss_tensor(
            call=call,
            actual=actual,
            expected=expected,
        )


def check_minibatch_gradient(found: object) -> None:
    policy, value, states, actions, old_log_probs, advantages, returns = [part.value for part in minibatch_for_check(834)]
    loss = found(policy, value, states, actions, old_log_probs, advantages, returns, 0.2)

    if not isinstance(loss, Tensor) or loss.grad_fn is None:
        raise Mismatch("minibatch_loss returns a loss that autograd did not record, so no gradient can reach the networks. Return a tensor, without .item().")

    loss.backward()

    if policy[0].weight.grad is None or value[0].weight.grad is None:
        raise Mismatch("minibatch_loss's loss sends gradients to only one network. It should add the policy's ppo_loss and the value network's error.")


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
        raise Mismatch(f"train_ppo(2, 838) returns {type(policy).__name__}, but it should return the trained policy.")

    expect_tensor(
        call="the policy's first weights after train_ppo(2, 838)",
        actual=policy[0].weight.detach(),
        expected=REFERENCE["train_ppo"](2, 838)[0].weight.detach(),
    )


def check_ppo_learns(found: object) -> None:
    # After 20 updates the reference reached 79 to 259 steps over seeds 0 to 9, and 99.4 on seed 839, whose untrained
    # policy falls in 9.25.
    steps = REFERENCE["balance_time"](found(20, 839), 16, torch.Generator().manual_seed(7))

    if steps < 50:
        raise Mismatch(f"After 20 updates of train_ppo, the greedy policy keeps 16 poles up for {steps:.1f} steps on average, but it should be above 50.")


def check_balance_time(found: object) -> None:
    # Seed 27's untrained policy happens to hold all 8 carts up to LONGEST, so a loop that stops short shows.
    for name, policy in (
        ("an untrained policy", seeded_networks(840)[0]),
        ("a policy after 5 updates", REFERENCE["train_ppo"](5, 841)),
        ("a policy that holds every cart to the end", seeded_networks(27)[0]),
    ):
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
            raise Mismatch("balance_time ran the policy while autograd was recording. Run the carts inside torch.no_grad().")


def check_ppo3(checker: Checker) -> None:
    checker.test(
        function="collect",
        description="draws each cart's action and saves its log probability",
        run=check_collect,
    )
    checker.test(
        function="minibatch_loss",
        description="adds PPO's loss and half the value error",
        run=check_minibatch_loss,
    )
    checker.test(
        function="minibatch_loss",
        description="sends gradients to both networks",
        run=check_minibatch_gradient,
    )
    checker.test(
        function="ppo_update",
        description="takes shuffled minibatch steps for every epoch",
        run=check_ppo_update,
    )
    checker.test(
        function="train_ppo",
        description="collects, computes GAE, then updates",
        run=check_train_ppo,
    )
    checker.test(
        function="balance_time",
        description="counts each cart's steps until its episode first ends",
        run=check_balance_time,
    )
    checker.test(
        function="train_ppo",
        description="learns to balance within 20 updates",
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
        "collect": "Take the log softmax of the policy's scores over dim 1. torch.multinomial needs probabilities, so "
        "draw from the exp of the log probabilities, one sample per row, and squeeze away the extra dimension. To get "
        "each cart's own log probability, index the rows with torch.arange and the columns with the actions.",
        "minibatch_loss": "Scale the advantages first. Subtract their mean and divide by their standard deviation plus "
        "1e-8. Index the policy's log softmax as collect did, rows with torch.arange and columns with the actions. The "
        "value network returns shape (rows, 1), so squeeze that last dimension before comparing with the returns.",
        "ppo_update": "Flatten the states over their first two dimensions, and flatten the rest completely. Each epoch, "
        "draw a fresh order with torch.randperm and the generator, then go through it in slices of length minibatch. "
        "For each slice, zero the gradients and backpropagate minibatch_loss. Then clip both networks' gradients "
        "together with nn.utils.clip_grad_norm_, and step.",
        "train_ppo": "At each update, left is the fraction of updates not yet run, which is 1 at the first. Set the "
        "learning rate in the optimizer's first param group and collect 32 steps. Pass the rollout's rewards, values, "
        "next values and dones to gae. The returns are the advantages plus the rollout's values.",
        "balance_time": "Make CartPole(count, generator), a standing mask of ones as bools, and a survived count of "
        "zeros. At each step, add standing to survived, then set standing to False where dones is True.",
    },
    introduces=(),
    check=check_ppo3,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)
