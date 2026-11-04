"""56_moe, experts and a router that runs a few of them per token, why the router needs a balancing loss, when more parameters for the same compute pay off, and weights computed from the input as one idea."""

from __future__ import annotations

import contextlib
import math
import textwrap
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_tensor,
    expect_value,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

MOE = Chapter(
    folder="56_moe",
    title="Mixture of experts",
    readme=source_file(
        """
        # Mixture of experts

        A mixture of experts keeps several MLPs and lets a router send each
        token to a few of them, so a model holds far more parameters than it
        runs per token. This chapter builds the router and the sparse layer,
        shows why the router needs a balancing loss, measures when the extra
        parameters pay off, and ties gates, SwiGLU, attention and routing
        together as weights computed from the input.

        ## Further reading

        - [Shazeer and colleagues, the sparsely gated mixture of experts layer](https://arxiv.org/abs/1701.06538)
        - [Fedus, Zoph and Shazeer, scaling to trillion parameter models with simple and efficient sparsity](https://arxiv.org/abs/2101.03961)
        - [Jiang and colleagues, Mixtral of experts](https://arxiv.org/abs/2401.04088)
        - [Hugging Face, mixture of experts explained](https://huggingface.co/blog/moe)
        """,
    ),
)

FEATURES = 8


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


# Every file of the chapter builds its experts the same way, written once so they never drift apart.
EXPERT_PART = '''
def expert(
    features: int,
    hidden: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, features),
    )
'''


def expert(
    features: int,
    hidden: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, features),
    )


# ---------------------------------------------------------------------------
# moe1
# ---------------------------------------------------------------------------

MOE1_LESSON = prose(
    """
    Every token so far went through the same MLP, all of its weights, every
    time. A **mixture of experts** keeps several MLPs, the experts, and sends
    each token to only a few of them.

    A router picks them. It is a Linear that gives each expert a score for the
    token. The top k scores win, and a softmax over only those k gives the
    weights of the chosen experts. The layer's output is the weighted sum of
    their outputs.

    ```python
    scores = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
    top = scores.topk(2, dim=-1)        # values tensor([[2., 1.]]), indices tensor([[0, 3]])
    torch.softmax(top.values, dim=-1)   # tensor([[0.7311, 0.2689]])
    ```

    This is Mixtral's router, a softmax over the top k of `x W_g`, with 8
    experts and k of 2. Each token runs 2 of the 8. So Mixtral holds about 47
    billion parameters and uses about 13 billion per token.

    The Switch paper went further, with k of 1. A softmax over one score is
    always 1, so that router would get no gradient and never learn. Switch
    takes the softmax over every expert's score first, then weights the one
    chosen expert by its probability. moe2's layer does this.

    The saving comes from running only the chosen experts. For each expert,
    find the tokens that picked it, run it on those rows alone, and add its
    weighted output into theirs.

    ```python
    chosen = torch.tensor([[0, 3], [3, 1]])
    rows, slots = (chosen == 3).nonzero(as_tuple=True)    # tensor([0, 1]), tensor([1, 0])
    ```

    `rows` are the tokens that picked expert 3, and `slots` say where it sits
    in their top k, which finds its weight. A token picks an expert at most
    once, so adding into `out[rows]` touches each row once.

    Running every expert and giving the unchosen ones a weight of 0 gives the
    same numbers. It also does all the work the router was meant to save.
    Mixtral's experts are SwiGLU MLPs, from llama2. These experts use ReLU to
    stay small.
    """,
)

MOE1_SOURCE = source_file(
    '''
    # moe1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    scores = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
    top = scores.topk(2, dim=-1)                            # values tensor([[2., 1.]]), indices tensor([[0, 3]])
    chosen = torch.tensor([[0, 3], [3, 1]])
    rows, slots = (chosen == 3).nonzero(as_tuple=True)      # tensor([0, 1]), tensor([1, 0])


    # Every expert is a small MLP.
''' + indented(EXPERT_PART) + '''


    # TODO: Return the weights and the chosen experts of each token, both of
    #   shape (tokens, k), for router scores of shape (tokens, experts). The
    #   weights are a softmax over the k largest scores of each row alone.
    #   Keep the largest score first, as topk gives them.
    def route(
        scores: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish forward for x of shape (tokens, features). Route the router's
    #   scores, then run each expert only on the rows that chose it, and add its
    #   output, times that row's weight for it, into the result.
    class MoE(nn.Module):
        def __init__(
            self,
            features: int,
            hidden: int,
            experts: int,
            k: int,
        ) -> None:
            super().__init__()
            self.router = nn.Linear(features, experts, bias=False)
            self.experts = nn.ModuleList([expert(features, hidden) for _ in range(experts)])
            self.k = k

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the share of all the choices in chosen, of shape (tokens, k),
    #   that went to each expert, one number per expert.
    def expert_load(
        chosen: Tensor,
        experts: int,
    ) -> Tensor:
        ...
    ''',
)

MOE1_SOLUTION = source_file(
    '''
    # moe1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    scores = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
    top = scores.topk(2, dim=-1)
    chosen = torch.tensor([[0, 3], [3, 1]])
    rows, slots = (chosen == 3).nonzero(as_tuple=True)


''' + indented(EXPERT_PART) + '''


    def route(
        scores: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        top = scores.topk(k, dim=-1)

        return torch.softmax(top.values, dim=-1), top.indices


    class MoE(nn.Module):
        def __init__(
            self,
            features: int,
            hidden: int,
            experts: int,
            k: int,
        ) -> None:
            super().__init__()
            self.router = nn.Linear(features, experts, bias=False)
            self.experts = nn.ModuleList([expert(features, hidden) for _ in range(experts)])
            self.k = k

        def forward(self, x: Tensor) -> Tensor:
            weights, chosen = route(self.router(x), self.k)
            out = torch.zeros_like(x)

            for index, expert_net in enumerate(self.experts):
                rows, slots = (chosen == index).nonzero(as_tuple=True)

                if len(rows) == 0:
                    continue

                out[rows] += weights[rows, slots].unsqueeze(1) * expert_net(x[rows])

            return out


    def expert_load(
        chosen: Tensor,
        experts: int,
    ) -> Tensor:
        return torch.bincount(chosen.flatten(), minlength=experts).float() / chosen.numel()
    ''',
)


def reference_route(
    scores: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    top = scores.topk(k, dim=-1)

    return torch.softmax(top.values, dim=-1), top.indices


class ReferenceMoE(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        experts: int,
        k: int,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(features, experts, bias=False)
        self.experts = nn.ModuleList([expert(features, hidden) for _ in range(experts)])
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        """Every expert runs on every token here, the unchosen ones weighted 0, which gives the same numbers."""
        weights, chosen = reference_route(self.router(x), self.k)
        every_weight = torch.zeros(len(x), len(self.experts)).scatter(1, chosen, weights)
        every_output = torch.stack([expert_net(x) for expert_net in self.experts], dim=1)

        return (every_weight.unsqueeze(2) * every_output).sum(dim=1)


def seeded_moe(
    make: type,
    k: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(4, 6, 5, k)


def expect_routes(
    call: str,
    actual: object,
    expected: tuple[Tensor, Tensor],
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise Mismatch(f"{call} should give the weights and the chosen experts, two tensors in a tuple.")

    expect_tensor(
        call=f"the weights of {call}",
        actual=actual[0],
        expected=expected[0],
    )
    expect_tensor(
        call=f"the chosen experts of {call}",
        actual=actual[1],
        expected=expected[1],
    )


def check_moe_forward(found: type) -> None:
    for k in (1, 2):
        model = expect_module(f"MoE(4, 6, 5, {k})", seeded_moe(found, k))
        x = small_tensor(560 + k, 7, 4) / 2
        rows_run = [0]

        # A single token's row of shape (features,) counts as one row, so a loop over tokens counts right.
        def count_rows(
            module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            rows_run[0] += inputs[0].shape[0] if inputs[0].dim() > 1 else 1

        hooks = [expert_net.register_forward_hook(count_rows) for expert_net in model.experts]

        try:
            actual = model(x)
        finally:
            for hook in hooks:
                hook.remove()

        with torch.no_grad():
            expected = seeded_moe(ReferenceMoE, k)(x)

        expect_tensor(
            call=f"MoE(4, 6, 5, {k}) on 7 tokens",
            actual=actual,
            expected=expected,
        )

        if rows_run[0] != 7 * k:
            raise Mismatch(
                f"MoE(4, 6, 5, {k}) ran its experts on {rows_run[0]} rows in all for 7 tokens, but it should run "
                f"only the chosen ones, {7 * k} rows.",
            )

        # With k of 1 every weight is a softmax over one score, always 1, so only k of 2 can show a gradient.
        if k == 1:
            continue

        if actual.requires_grad:
            actual.sum().backward()

        gradient = model.router.weight.grad

        if gradient is None or not gradient.abs().sum() > 0:
            raise Mismatch(
                f"MoE(4, 6, 5, {k})'s router got no gradient. Keep the weights from route in the sum, outside "
                "torch.no_grad() and without .detach(), since they are how the router learns.",
            )


def check_moe1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="route",
        description="keeps the top k scores and softmaxes them alone",
        inputs=(
            (torch.tensor([[2.0, -1.0, 0.5, 1.0], [0.0, 3.0, -2.0, 1.0]]), 2),
            (small_tensor(563, 5, 6) / 2 + torch.linspace(0.0, 0.1, 6), 3),
            (small_tensor(564, 4, 3) / 2 + torch.linspace(0.0, 0.1, 3), 1),
        ),
        reference=reference_route,
        expect=expect_routes,
    )
    checker.test_method(
        class_name="MoE",
        method="forward",
        description="runs only the chosen experts and weights their outputs",
        run=check_moe_forward,
    )
    check_on_inputs(
        checker=checker,
        function="expert_load",
        description="gives each expert's share of the choices",
        inputs=(
            (torch.tensor([[0, 3], [3, 1], [3, 0]]), 4),
            (torch.tensor([[2], [2], [0], [2], [1]]), 5),
        ),
        reference=lambda chosen, experts: torch.bincount(chosen.flatten(), minlength=experts).float() / chosen.numel(),
        expect=expect_tensor,
    )


MOE1 = Exercise(
    name="moe1",
    chapter=MOE,
    title="Experts and a router",
    lesson=MOE1_LESSON,
    source=MOE1_SOURCE,
    solution=MOE1_SOLUTION,
    hints={
        "route": "scores.topk(k, dim=-1) gives .values and .indices. Return torch.softmax of the values over dim=-1, "
        "then the indices.",
        "MoE.forward": "weights, chosen = route(self.router(x), self.k). For each expert, (chosen == index).nonzero("
        "as_tuple=True) gives rows and slots. Skip an expert with no rows, and add weights[rows, slots].unsqueeze(1) "
        "times the expert's output on x[rows] into out[rows].",
        "expert_load": "torch.bincount(chosen.flatten(), minlength=experts) counts the choices. Divide by "
        "chosen.numel() as floats.",
    },
    introduces=(),
    check=check_moe1,
)


# ---------------------------------------------------------------------------
# moe2
# ---------------------------------------------------------------------------

# The target the Switch layers of moe2 learn, built from its own generator so it is the same everywhere.
TARGET_PART = '''
def target_of(x: Tensor) -> Tensor:
    generator = torch.Generator().manual_seed(100)
    first = torch.randn(64, 8, generator=generator) / 8**0.5
    second = torch.randn(8, 64, generator=generator) / 64**0.5

    return torch.tanh(x @ first.T) @ second.T
'''

SWITCH_PART = '''
# A Switch layer, top 1 routing with a capacity for each expert, from the Switch paper.
class Switch(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        experts: int,
        factor: float,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(features, experts)
        self.experts = nn.ModuleList([expert(features, hidden) for _ in range(experts)])
        self.factor = factor

    # Each token goes to its likeliest expert, weighted by that probability, if the expert has room.
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        probs = torch.softmax(self.router(x), dim=-1)
        weight, chosen = probs.max(dim=-1)
        capacity = math.ceil(self.factor * len(x) / len(self.experts))
        kept = within_capacity(chosen, len(self.experts), capacity)
        output = torch.zeros_like(x)

        for index, expert_net in enumerate(self.experts):
            rows = ((chosen == index) & kept).nonzero().flatten()

            if len(rows) == 0:
                continue

            output[rows] = weight[rows].unsqueeze(1) * expert_net(x[rows])

        return output, probs, chosen, kept
'''

BALANCING_SOLUTION = '''
def balancing_loss(
    probs: Tensor,
    chosen: Tensor,
    experts: int,
) -> Tensor:
    sent = torch.bincount(chosen, minlength=experts).float() / len(chosen)

    return experts * (sent * probs.mean(dim=0)).sum()
'''

CAPACITY_SOLUTION = '''
def within_capacity(
    chosen: Tensor,
    experts: int,
    capacity: int,
) -> Tensor:
    kept = torch.zeros(len(chosen), dtype=torch.bool)

    for index in range(experts):
        rows = (chosen == index).nonzero().flatten()
        kept[rows[:capacity]] = True

    return kept
'''

TRAIN_SWITCH_SOLUTION = '''
def train_switch(
    model: nn.Module,
    steps: int,
    alpha: float,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(steps):
        x = torch.randn(256, 8, generator=generator)
        output, probs, chosen, _ = model(x)
        loss = F.mse_loss(output, target_of(x)) + alpha * balancing_loss(probs, chosen, len(model.experts))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''

MOE2_LESSON = prose(
    """
    A router learns like any other layer, and left alone it picks favourites.
    Shazeer and colleagues saw the imbalance feed itself. The favoured experts
    train faster, so the router picks them even more.

    Here are 4 runs of a top 1 router over 4 experts, 400 steps each, on the
    same smooth target. Its router has a bias, which lets it favour one expert
    for every input and shows the effect quickly. The routers of Switch and
    Mixtral have no bias. This layer's softmax runs over all 4 scores, as
    Switch's does. Each token's one expert is weighted by its probability,
    for the reason moe1 gave.

    | Balancing | Tokens to the busiest expert | Tokens dropped | Error |
    |---|---|---|---|
    | none | 53% to 65% | 23% to 34% | 0.096 to 0.120 |
    | weight 0.01 | 32% to 39% | 1% to 8% | 0.021 to 0.042 |

    An even share is 25%, and always answering 0 has an error of 0.351.

    Imbalance wastes the experts that get little, and it does worse than
    that. Each expert runs on its own share of the hardware, with room for a
    fixed number of tokens per batch, its **capacity**. The Switch paper sets
    the capacity to a capacity factor times an even share, here 1.25.

    ```python
    math.ceil(1.25 * 256 / 4)    # 80 tokens per expert, for a batch of 256
    ```

    Tokens past an expert's capacity are dropped, and the layer gives them 0.
    With a residual around the layer they pass through unchanged, but the
    layer has done nothing for them. The busier the favourite, the more of the
    batch is dropped.

    The Switch paper's fix is a balancing loss, added to the task's loss with
    a weight of 0.01.

    ```text
    balance = experts * sum over i of f_i * P_i
    ```

    `f_i` is the share of tokens sent to expert i, and `P_i` is the router's
    average probability for expert i. An even split gives exactly 1. One
    expert with every token and all the probability gives the number of
    experts. `f_i` comes from an argmax and passes no gradient, as soft1 showed.
    So the gradient reaches the router through `P_i`, and it lowers the
    probability of the busiest experts most.
    """,
)

MOE2_SOURCE = source_file(
    '''
    # moe2.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    sent = torch.bincount(torch.tensor([2, 0, 2, 2]), minlength=3)    # tensor([1, 0, 3])
    room = math.ceil(1.25 * 256 / 4)                                    # 80


    # moe1's experts.
''' + indented(EXPERT_PART) + '''


    # The smooth target the layer learns.
''' + indented(TARGET_PART) + '''


''' + indented(SWITCH_PART) + '''


    # TODO: Return the Switch paper's balancing loss for router probabilities of
    #   shape (tokens, experts) and the chosen expert of each token. It is
    #   experts times the sum of f_i * P_i, with f_i the share of tokens sent to
    #   expert i and P_i the mean of its column of probs.
    def balancing_loss(
        probs: Tensor,
        chosen: Tensor,
        experts: int,
    ) -> Tensor:
        ...


    # TODO: Return a tensor of True and False, one per token, that keeps the
    #   first capacity tokens sent to each expert, in the order they come, and
    #   drops the rest.
    def within_capacity(
        chosen: Tensor,
        experts: int,
        capacity: int,
    ) -> Tensor:
        ...


    # TODO: Finish the line that sets loss. It is the mean squared error of
    #   output against target_of(x), plus alpha times the balancing loss.
    def train_switch(
        model: nn.Module,
        steps: int,
        alpha: float,
        generator: torch.Generator,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(steps):
            x = torch.randn(256, 8, generator=generator)
            output, probs, chosen, _ = model(x)
            loss = ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ''',
)

MOE2_SOLUTION = source_file(
    '''
    # moe2.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    sent = torch.bincount(torch.tensor([2, 0, 2, 2]), minlength=3)
    room = math.ceil(1.25 * 256 / 4)


''' + indented(EXPERT_PART) + '''


''' + indented(TARGET_PART) + '''


''' + indented(SWITCH_PART) + '''


''' + indented(BALANCING_SOLUTION) + '''


''' + indented(CAPACITY_SOLUTION) + '''


''' + indented(TRAIN_SWITCH_SOLUTION) + '''
    ''',
)


def switch_namespace() -> dict[str, object]:
    """The given parts and the solution's functions, run once, so the checks compare against exactly the
    code the learner's file holds."""
    namespace: dict[str, object] = {
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(
        EXPERT_PART + TARGET_PART + SWITCH_PART + BALANCING_SOLUTION + CAPACITY_SOLUTION + TRAIN_SWITCH_SOLUTION,
        namespace,
    )

    return namespace


SWITCH_REFERENCE = switch_namespace()


def seeded_switch(seed: int) -> nn.Module:
    torch.manual_seed(seed)

    return SWITCH_REFERENCE["Switch"](FEATURES, 16, 4, 1.25)


def switch_results(
    model: nn.Module,
) -> tuple[float, float, float, float]:
    """The share of tokens to the busiest expert, the share dropped, the error and the error of always
    answering 0, on 4,096 new inputs."""
    x = torch.randn(4096, FEATURES, generator=torch.Generator().manual_seed(999))

    with torch.no_grad():
        output, _, chosen, kept = model(x)
        target = SWITCH_REFERENCE["target_of"](x)

    busiest = torch.bincount(chosen, minlength=len(model.experts)).max().item() / len(chosen)
    dropped = 1 - kept.float().mean().item()

    return busiest, dropped, F.mse_loss(output, target).item(), F.mse_loss(torch.zeros_like(target), target).item()


def check_train_switch(found: object) -> None:
    trained = seeded_switch(570)
    found(trained, 3, 0.01, torch.Generator().manual_seed(571))
    expected = seeded_switch(570)
    SWITCH_REFERENCE["train_switch"](expected, 3, 0.01, torch.Generator().manual_seed(571))
    expect_tensor(
        call="the router's weights after train_switch for 3 steps with alpha 0.01",
        actual=trained.router.weight.detach(),
        expected=expected.router.weight.detach(),
    )

    balanced = seeded_switch(0)
    found(balanced, 400, 0.01, torch.Generator().manual_seed(1))
    _, dropped, error, _ = switch_results(balanced)

    if dropped > 0.2 or error > 0.07:
        raise Mismatch(
            f"After train_switch for 400 steps with alpha 0.01, the layer drops {dropped:.1%} of the tokens and has "
            f"an error of {error:.4f}, but it should drop under 20% with an error under 0.07.",
        )


def check_moe2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="balancing_loss",
        description="gives experts times the sum of f_i * P_i",
        inputs=(
            (torch.softmax(small_tensor(572, 6, 4) / 2, dim=-1), torch.tensor([0, 3, 3, 1, 3, 0]), 4),
            (torch.full((4, 2), 0.5), torch.tensor([0, 1, 1, 0]), 2),
            (torch.softmax(small_tensor(573, 5, 3), dim=-1), torch.tensor([2, 2, 2, 2, 2]), 3),
        ),
        reference=SWITCH_REFERENCE["balancing_loss"],
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual.item() if isinstance(actual, Tensor) else actual,
            expected=expected.item(),
            tolerance=1e-5,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="within_capacity",
        description="keeps the first tokens up to each expert's capacity",
        inputs=(
            (torch.tensor([1, 1, 0, 1, 2, 1, 0]), 3, 2),
            (torch.tensor([0, 0, 0, 0]), 2, 3),
            (torch.tensor([2, 0, 1, 2, 2, 0]), 3, 1),
        ),
        reference=SWITCH_REFERENCE["within_capacity"],
        expect=expect_tensor,
    )
    checker.test(
        function="train_switch",
        description="adds the balancing loss, so the layer spreads its tokens",
        run=check_train_switch,
    )


MOE2 = Exercise(
    name="moe2",
    chapter=MOE,
    title="Why the balancing loss",
    lesson=MOE2_LESSON,
    source=MOE2_SOURCE,
    solution=MOE2_SOLUTION,
    hints={
        "balancing_loss": "torch.bincount(chosen, minlength=experts) over len(chosen) gives f as floats, and "
        "probs.mean(dim=0) gives P. Return experts * (f * P).sum().",
        "within_capacity": "Start from torch.zeros(len(chosen), dtype=torch.bool). For each expert, "
        "(chosen == index).nonzero().flatten() lists its tokens in order, and the first capacity of them are kept.",
        "train_switch": "F.mse_loss(output, target_of(x)) + alpha * balancing_loss(probs, chosen, len(model.experts)).",
    },
    introduces=(),
    check=check_moe2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


def measure_moe2(
    seeds: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[float, list[tuple[float, float, float]]]:
    """moe2's table: the Switch layer from each seed, trained 400 steps on batches drawn from seed + 1, with no
    balancing and with a weight of 0.01. Gives (share to the busiest expert, share dropped, error) for each, on
    4,096 new inputs."""
    results: dict[float, list[tuple[float, float, float]]] = {}

    with pinned_threads():
        for alpha in (0.0, 0.01):
            for seed in seeds:
                model = seeded_switch(seed)
                SWITCH_REFERENCE["train_switch"](model, 400, alpha, torch.Generator().manual_seed(seed + 1))
                busiest, dropped, error, _ = switch_results(model)
                results.setdefault(alpha, []).append((busiest, dropped, error))

    return results


# ---------------------------------------------------------------------------
# moe3
# ---------------------------------------------------------------------------

REGIONS = 16


def regions_task() -> tuple[Tensor, Tensor]:
    """16 centres in 8 features, each region of input with its own linear map."""
    generator = torch.Generator().manual_seed(0)
    centers = 3 * torch.randn(REGIONS, FEATURES, generator=generator)
    maps = torch.randn(REGIONS, FEATURES, FEATURES, generator=generator) / FEATURES**0.5

    return centers, maps


def regions_batch(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    centers, maps = regions_task()
    region = torch.randint(0, REGIONS, (count,), generator=generator)
    x = centers[region] + torch.randn(count, FEATURES, generator=generator)

    return x, torch.einsum("nij,nj->ni", maps[region], x), region


def smooth_batch(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    x = torch.randn(count, FEATURES, generator=generator)

    return x, SWITCH_REFERENCE["target_of"](x), torch.zeros(count, dtype=torch.long)


class Dense(nn.Module):
    def __init__(
        self,
        hidden: int,
    ) -> None:
        super().__init__()
        self.net = expert(FEATURES, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_for_moe3(
    model: nn.Module,
    batch: object,
    steps: int,
) -> tuple[float, float, float]:
    """The runs moe3's table comes from, Adam at 0.01 on batches of 256, with the balancing loss at 0.01 for
    a Switch layer. Gives the error on 4,000 new inputs and, for a Switch layer, the share of each region's
    inputs that go to its most common expert and the share of inputs that found their expert full."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(1)
    switch = isinstance(model, SWITCH_REFERENCE["Switch"])

    for _ in range(steps):
        x, y, _ = batch(256, generator)

        if not switch:
            loss = F.mse_loss(model(x), y)
        else:
            output, probs, chosen, _ = model(x)
            loss = F.mse_loss(output, y) + 0.01 * SWITCH_REFERENCE["balancing_loss"](probs, chosen, len(model.experts))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x, y, region = batch(4000, torch.Generator().manual_seed(2))

    with torch.no_grad():
        if not switch:
            return F.mse_loss(model(x), y).item(), 0.0, 0.0

        output, _, chosen, kept = model(x)

    owned = [torch.bincount(chosen[region == index]).max().item() / (region == index).sum().item() for index in region.unique().tolist()]

    return F.mse_loss(output, y).item(), sum(owned) / len(owned), 1 - kept.float().mean().item()


# A factor this large never fills an expert, so it stands for no limit.
MOE3_MODELS: dict[str, Callable[[], nn.Module]] = {
    "dense, 16 wide": lambda: Dense(16),
    "8 experts, no limit": lambda: SWITCH_REFERENCE["Switch"](FEATURES, 16, 8, 100.0),
    "8 experts, factor 2.0": lambda: SWITCH_REFERENCE["Switch"](FEATURES, 16, 8, 2.0),
    "dense, 128 wide": lambda: Dense(128),
}


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """Four CPU threads while measuring, so the sums add up in the same order on every run."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def measure_moe3(
    steps: int = 3000,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
    """moe3's table: each model on each task from each seed, as (error, share owned, share dropped). The factor
    2.0 layer runs on the regions task alone, since the table shows capacity there."""
    results: dict[tuple[str, str], list[tuple[float, float, float]]] = {}

    with pinned_threads():
        for task, batch in (("regions", regions_batch), ("smooth", smooth_batch)):
            for name, make in MOE3_MODELS.items():
                if task == "smooth" and "factor" in name:
                    continue

                for seed in seeds:
                    torch.manual_seed(seed)
                    results.setdefault((task, name), []).append(train_for_moe3(make(), batch, steps))

    return results


MOE3_LESSON = prose(
    """
    A mixture of experts holds many parameters and runs few of them for each
    token. This exercise counts both and measures when the trade pays.

    An expert from 8 features to 16 and back holds 280 parameters. Eight of
    them and a router with a bias hold 2,312, yet each token runs one expert
    and the router, 352. A dense MLP 128 wide holds 2,184 and runs every one.

    Here they are, trained 3,000 steps on two tasks, 3 seeds each. In the
    regions task the input falls near one of 16 centres, and each region has
    its own linear map. The smooth task is moe2's target, one function
    everywhere.

    | Model | Run per token | Regions error | Smooth error |
    |---|---|---|---|
    | dense, 16 wide | 280 | 1.34 to 1.43 | 0.0148 to 0.0156 |
    | 8 experts | 352 | 0.105 to 0.151 | 0.0118 to 0.0123 |
    | dense, 128 wide | 2,184 | 0.262 to 0.270 | 0.0011 to 0.0013 |

    On the regions task the router sent 99% or more of each region's inputs
    to one expert. Each expert came to own a couple of regions and learned
    their maps. The experts beat the dense layer of the same compute about
    tenfold. They also beat the dense layer with as many parameters, with
    about half its error.

    On the smooth task the experts still beat the dense layer of the same
    compute, 0.012 against 0.015. One function everywhere doesn't split into
    parts, though. There the dense layer with as many parameters, all of them
    running, does ten times better.

    That is the trade. Experts buy parameters for the same compute, and they
    pay most when the work splits into parts that different experts can own.
    In Mixtral and in Google's ST-MoE the experts did not split by topic or by
    language. They split by kinds of tokens, like punctuation, numbers, names,
    or `self` in Python code.

    The experts above ran with no limit on their load. With room for twice an
    even share, a capacity factor of 2.0, under half a percent of the inputs
    found their expert full. The regions error came out at 0.112 to 0.185.

    The Switch Transformer paper ran factors from 1.0 to 2.0, and each beat
    the dense model of the same compute by about the same margin. There a token
    with no room passes on through the residual. This layer has no residual, so
    a dropped token gets 0, which is why it wants the room.

    The cost is memory, every expert held at once, and a router to keep
    balanced. That memory is why large models spread their experts across
    many GPUs. The saving is multiplications per token.
    """,
)

MOE3_SOURCE = source_file(
    '''
    # moe3.py

    layer = 8 * 16 + 16    # 144, the weights and biases of a Linear from 8 to 16


    # TODO: Return the parameters of one expert, a Linear with bias from features
    #   to hidden and one from hidden back to features.
    def dense_parameters(
        features: int,
        hidden: int,
    ) -> int:
        ...


    # TODO: Return the parameters of a mixture of experts, experts copies of
    #   that expert, plus a router with a bias, a Linear from features to
    #   experts.
    def moe_parameters(
        features: int,
        hidden: int,
        experts: int,
    ) -> int:
        ...


    # TODO: Return the parameters one token runs through, the router and the k
    #   experts it picks.
    def active_parameters(
        features: int,
        hidden: int,
        experts: int,
        k: int,
    ) -> int:
        ...
    ''',
)

MOE3_SOLUTION = source_file(
    '''
    # moe3.py, a solution for comparison

    layer = 8 * 16 + 16


    def dense_parameters(
        features: int,
        hidden: int,
    ) -> int:
        return features * hidden + hidden + hidden * features + features


    def moe_parameters(
        features: int,
        hidden: int,
        experts: int,
    ) -> int:
        return experts * dense_parameters(features, hidden) + features * experts + experts


    def active_parameters(
        features: int,
        hidden: int,
        experts: int,
        k: int,
    ) -> int:
        return k * dense_parameters(features, hidden) + features * experts + experts
    ''',
)


def counted(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def reference_moe_parameters(
    features: int,
    hidden: int,
    experts: int,
) -> int:
    return counted(SWITCH_REFERENCE["Switch"](features, hidden, experts, 1.0))


def reference_active_parameters(
    features: int,
    hidden: int,
    experts: int,
    k: int,
) -> int:
    return k * counted(expert(features, hidden)) + counted(nn.Linear(features, experts))


def check_moe3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="dense_parameters",
        description="counts one expert's parameters",
        inputs=(
            (8, 16),
            (5, 3),
            (4096, 14336),
        ),
        reference=lambda features, hidden: 2 * features * hidden + hidden + features,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="moe_parameters",
        description="counts every expert and the router",
        inputs=(
            (8, 16, 8),
            (6, 5, 3),
        ),
        reference=reference_moe_parameters,
        expect=expect_value,
    )
    check_on_inputs(
        checker=checker,
        function="active_parameters",
        description="counts the router and the k experts a token runs",
        inputs=(
            (8, 16, 8, 1),
            (6, 5, 4, 2),
        ),
        reference=reference_active_parameters,
        expect=expect_value,
    )


MOE3 = Exercise(
    name="moe3",
    chapter=MOE,
    title="When, more parameters for the same compute",
    lesson=MOE3_LESSON,
    source=MOE3_SOURCE,
    solution=MOE3_SOLUTION,
    hints={
        "dense_parameters": "Each Linear has in times out weights plus out biases. Add the two Linears.",
        "moe_parameters": "experts times dense_parameters, plus the router's features * experts weights and experts "
        "biases.",
        "active_parameters": "k times dense_parameters, plus the whole router, which runs for every token.",
    },
    introduces=(),
    check=check_moe3,
)


# ---------------------------------------------------------------------------
# dynamic1
# ---------------------------------------------------------------------------

DYNAMIC1_LESSON = prose(
    """
    Four layers in this course make some of their weights from the input
    itself, fresh on every forward pass.

    | Layer | Weights made from the input | What they weigh | Shown in |
    |---|---|---|---|
    | gate | `sigmoid(x W_g)` | each feature of `x W_v` | gating1, derive1 |
    | SwiGLU | `silu(x W_g)` | each feature of `x W_up` | llama2 |
    | attention | `softmax(q k^T / sqrt(d))` | the values of every position | attention5 |
    | routing | a softmax over the top k router scores | the outputs of the chosen experts | moe1 |

    A Linear's weights stop changing once training ends, and the same numbers
    multiply every input. In these four, the trained weights make other
    weights, and the input decides those each time.

    ```python
    W_g = torch.tensor([[1.0, -1.0]])
    torch.sigmoid(torch.tensor([[2.0, 0.0]]) @ W_g.T)    # tensor([[0.8808]])
    torch.sigmoid(torch.tensor([[0.0, 2.0]]) @ W_g.T)    # tensor([[0.1192]])
    ```

    One gate, two inputs, and two different weights.

    That buys multiplication. A Linear scales its inputs and adds them. These
    layers multiply one function of the input by another. So a gate can pass a
    value only while a switch is on, as gating1 showed. Attention can fetch
    whatever a query asks for, from any position. Routing multiplies the
    skipped experts by 0, and so runs a different network for each token.

    It is a design tool too. When a model must choose, look something up or
    switch between paths, make the choice a weight computed from the input.
    Make it soft so it passes a gradient, as 38_soft_design did, and train it.

    Routing needs one more tool. `.scatter(dim, index, values)` writes each
    value of a row into the column that `index` names for it, leaving every
    other column as it was. On a row of zeros it turns the top k into a full
    row of weights, 0 for each expert left out.

    ```python
    torch.zeros(2, 4).scatter(1, torch.tensor([[3, 0], [1, 2]]), torch.tensor([[0.6, 0.4], [0.9, 0.1]]))
    # tensor([[0.4000, 0.0000, 0.0000, 0.6000], [0.0000, 0.9000, 0.1000, 0.0000]])
    ```
    """,
)

DYNAMIC1_SOURCE = source_file(
    '''
    # dynamic1.py

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    W_g = torch.tensor([[1.0, -1.0]])
    gate = torch.sigmoid(torch.tensor([[2.0, 0.0]]) @ W_g.T)    # tensor([[0.8808]])
    picked = torch.zeros(2, 3).scatter(1, torch.tensor([[2], [0]]), torch.tensor([[0.7], [0.4]]))    # tensor([[0.0, 0.0, 0.7], [0.4, 0.0, 0.0]])


    # TODO: Return each feature of x @ W_value.T times its gate,
    #   sigmoid(x @ W_gate.T). x has shape (tokens, features).
    def gate_layer(
        x: Tensor,
        W_gate: Tensor,
        W_value: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return llama2's SwiGLU before its last Linear, x @ W_up.T with
    #   each feature weighted by F.silu(x @ W_gate.T).
    def swiglu_layer(
        x: Tensor,
        W_gate: Tensor,
        W_up: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return attention over x, of shape (steps, features), with no mask.
    #   q, k and v are x times W_q.T, W_k.T and W_v.T, and the weights are a
    #   softmax over each row of q k^T divided by the square root of q's size.
    def attention_layer(
        x: Tensor,
        W_q: Tensor,
        W_k: Tensor,
        W_v: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return a routed layer for x, of shape (tokens, features). Expert i is
    #   x @ experts[i].T, and the router's scores are x @ router.T. Each token
    #   weights its top k experts by a softmax over their scores alone and the
    #   rest by 0. Running every expert is fine here.
    def routed_layer(
        x: Tensor,
        router: Tensor,
        experts: Tensor,
        k: int,
    ) -> Tensor:
        ...
    ''',
)

DYNAMIC1_SOLUTION = source_file(
    '''
    # dynamic1.py, a solution for comparison

    import math

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    W_g = torch.tensor([[1.0, -1.0]])
    gate = torch.sigmoid(torch.tensor([[2.0, 0.0]]) @ W_g.T)
    picked = torch.zeros(2, 3).scatter(1, torch.tensor([[2], [0]]), torch.tensor([[0.7], [0.4]]))


    def gate_layer(
        x: Tensor,
        W_gate: Tensor,
        W_value: Tensor,
    ) -> Tensor:
        return torch.sigmoid(x @ W_gate.T) * (x @ W_value.T)


    def swiglu_layer(
        x: Tensor,
        W_gate: Tensor,
        W_up: Tensor,
    ) -> Tensor:
        return F.silu(x @ W_gate.T) * (x @ W_up.T)


    def attention_layer(
        x: Tensor,
        W_q: Tensor,
        W_k: Tensor,
        W_v: Tensor,
    ) -> Tensor:
        q, k, v = x @ W_q.T, x @ W_k.T, x @ W_v.T
        weights = torch.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1)

        return weights @ v


    def routed_layer(
        x: Tensor,
        router: Tensor,
        experts: Tensor,
        k: int,
    ) -> Tensor:
        top = (x @ router.T).topk(k, dim=-1)
        weights = torch.zeros(len(x), len(experts)).scatter(1, top.indices, torch.softmax(top.values, dim=-1))
        outputs = torch.stack([x @ weight.T for weight in experts], dim=1)

        return (weights.unsqueeze(2) * outputs).sum(dim=1)
    ''',
)


def reference_routed_layer(
    x: Tensor,
    router: Tensor,
    experts: Tensor,
    k: int,
) -> Tensor:
    """Each token's chosen experts one by one, a different road to the same numbers."""
    rows = []

    for token in x:
        top = (router @ token).topk(k)
        weights = torch.softmax(top.values, dim=0)
        rows.append(sum(weight * (experts[index] @ token) for weight, index in zip(weights, top.indices.tolist())))

    return torch.stack(rows)


def check_dynamic1(checker: Checker) -> None:
    x = small_tensor(580, 5, 4) / 2
    check_on_inputs(
        checker=checker,
        function="gate_layer",
        description="weights each feature by a gate made from the input",
        inputs=(
            (x, small_tensor(581, 3, 4) / 2, small_tensor(582, 3, 4) / 2),
        ),
        reference=lambda x, W_gate, W_value: torch.sigmoid(x @ W_gate.T) * (x @ W_value.T),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="swiglu_layer",
        description="weights each feature by silu of another projection",
        inputs=(
            (x, small_tensor(583, 6, 4) / 2, small_tensor(584, 6, 4) / 2),
        ),
        reference=lambda x, W_gate, W_up: F.silu(x @ W_gate.T) * (x @ W_up.T),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="attention_layer",
        description="mixes the values of every position by softmax weights",
        inputs=(
            (x, small_tensor(585, 3, 4) / 2, small_tensor(586, 3, 4) / 2, small_tensor(587, 2, 4) / 2),
        ),
        reference=lambda x, W_q, W_k, W_v: F.scaled_dot_product_attention(
            (x @ W_q.T).unsqueeze(0),
            (x @ W_k.T).unsqueeze(0),
            (x @ W_v.T).unsqueeze(0),
        ).squeeze(0),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="routed_layer",
        description="weights each token's top k experts by a softmax over their scores",
        inputs=(
            (x, small_tensor(588, 4, 4) / 2 + torch.linspace(0.0, 0.1, 4).unsqueeze(1), small_tensor(589, 4, 3, 4) / 2, 2),
            (x, small_tensor(590, 3, 4) / 2 + torch.linspace(0.0, 0.1, 3).unsqueeze(1), small_tensor(591, 3, 4, 4) / 2, 1),
        ),
        reference=reference_routed_layer,
        expect=expect_tensor,
    )


DYNAMIC1 = Exercise(
    name="dynamic1",
    chapter=MOE,
    title="Weights computed from the input",
    lesson=DYNAMIC1_LESSON,
    source=DYNAMIC1_SOURCE,
    solution=DYNAMIC1_SOLUTION,
    hints={
        "gate_layer": "torch.sigmoid(x @ W_gate.T) times (x @ W_value.T).",
        "swiglu_layer": "F.silu(x @ W_gate.T) times (x @ W_up.T).",
        "attention_layer": "q @ k.T over math.sqrt(q.shape[-1]), a softmax with dim=-1, then @ v.",
        "routed_layer": "(x @ router.T).topk(k, dim=-1) gives the scores and the experts. scatter the softmax of the "
        "scores into torch.zeros(len(x), len(experts)), then weight torch.stack([x @ e.T for e in experts], dim=1) by "
        "it and sum over dim=1.",
    },
    introduces=(),
    check=check_dynamic1,
)
