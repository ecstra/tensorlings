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

        A mixture of experts has several MLPs, called experts, and a router
        that sends each token to only a few of them. So the model holds far
        more parameters than it uses for any one token. This chapter builds
        the router and the sparse layer. It shows why the router needs a
        balancing loss and measures when the extra parameters are worth it.
        Last, it shows that gates, SwiGLU, attention and routing all share one
        idea, weights computed from the input.

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
    So far, every token has gone through the same MLP and used all of its
    weights. A **mixture of experts** has several MLPs, called the experts,
    and sends each token to only a few of them.

    A router picks the experts for each token. The router is a Linear layer
    that gives each expert a score for the token. The k experts with the
    highest scores are chosen. A softmax over only those k scores gives the
    weights of the chosen experts. The layer's output is the weighted sum of
    the chosen experts' outputs.

    ```python
    scores = torch.tensor([[2.0, -1.0, 0.5, 1.0]])
    top = scores.topk(2, dim=-1)        # values tensor([[2., 1.]]), indices tensor([[0, 3]])
    torch.softmax(top.values, dim=-1)   # tensor([[0.7311, 0.2689]])
    ```

    This is the router Mixtral uses, a softmax over the top k of `x W_g`.
    Mixtral has 8 experts and k of 2, so each token runs through 2 of the 8.
    That is why Mixtral holds about 47 billion parameters but uses only about
    13 billion for each token.

    The Switch paper went further and used k of 1. A softmax over a single
    score is always 1. So the weight would always be 1, the router would get
    no gradient, and it would never learn. Instead, Switch takes the softmax
    over all the experts' scores first. Then it multiplies the output of the
    one chosen expert by that expert's probability. The layer in moe2 works
    this way.

    The layer saves computation by running only the chosen experts. For each
    expert, find the tokens that picked it and run the expert on only those
    rows. Multiply its output by each token's weight for it, and add the
    result into those tokens' rows of the output.

    ```python
    chosen = torch.tensor([[0, 3], [3, 1]])
    rows, slots = (chosen == 3).nonzero(as_tuple=True)    # tensor([0, 1]), tensor([1, 0])
    ```

    `rows` holds the tokens that picked expert 3. `slots` holds the position
    of expert 3 in each of those tokens' top k, which is where to find its
    weight. A token picks an expert at most once, so no row appears twice in
    `rows`. That means adding into `out[rows]` updates each row exactly once.

    You could also run every expert and give the unchosen ones a weight of 0.
    That gives the same numbers, but it does all the work the router was
    meant to save. Mixtral's experts are SwiGLU MLPs, from llama2. The experts
    here use ReLU to keep them small.
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


    # TODO: Return a pair (weights, chosen) for router scores of shape
    #   (tokens, experts). chosen holds the indices of the k largest scores in
    #   each row, and weights is a softmax over only those k scores. Both have
    #   shape (tokens, k). Keep the largest score first, in the order topk
    #   returns them.
    #   Example: route(scores, 2) for the scores above returns
    #   (tensor([[0.7311, 0.2689]]), tensor([[0, 3]])).
    def route(
        scores: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: Finish forward for x of shape (tokens, features). Call route on
    #   the router's scores to get each token's weights and chosen experts.
    #   Then run each expert only on the rows of x that chose it. Multiply the
    #   expert's output by each row's weight for that expert, and add it into
    #   those rows of the result.
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


    # TODO: Return the fraction of all the choices in chosen that went to each
    #   expert, as a float tensor with one number per expert. chosen has shape
    #   (tokens, k).
    #   Example: expert_load(torch.tensor([[0, 3], [3, 1]]), 4) returns
    #   tensor([0.2500, 0.2500, 0.0000, 0.5000]).
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
        raise Mismatch(f"{call} should return the weights and the chosen experts as a tuple of two tensors.")

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
                f"MoE(4, 6, 5, {k}) ran its experts on {rows_run[0]} rows in total for 7 tokens. It should run "
                f"only the chosen experts, which is {7 * k} rows.",
            )

        # With k of 1 every weight is a softmax over one score, always 1, so only k of 2 can show a gradient.
        if k == 1:
            continue

        if actual.requires_grad:
            actual.sum().backward()

        gradient = model.router.weight.grad

        if gradient is None or not gradient.abs().sum() > 0:
            raise Mismatch(
                f"The router of MoE(4, 6, 5, {k}) got no gradient. The router learns through the weights from route, "
                "so use them in the output sum, outside torch.no_grad() and without .detach().",
            )


def check_moe1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="route",
        description="takes a softmax over only the top k scores",
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
        description="runs only the chosen experts and adds their weighted outputs",
        run=check_moe_forward,
    )
    check_on_inputs(
        checker=checker,
        function="expert_load",
        description="returns the fraction of the choices that went to each expert",
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
        "MoE.forward": "Call route with the router's scores for x and with self.k to get weights and chosen. For "
        "each expert, (chosen == index).nonzero(as_tuple=True) gives rows and slots. Skip an expert with no rows. "
        "Otherwise add weights[rows, slots].unsqueeze(1) times the expert's output on x[rows] into out[rows].",
        "expert_load": "torch.bincount(chosen.flatten(), minlength=experts) counts the choices for each expert. "
        "Convert the counts to floats and divide them by chosen.numel().",
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
# A Switch layer from the Switch paper. It uses top 1 routing, with a
# capacity for each expert.
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

    # Each token goes to its most likely expert, and the expert's output is
    # multiplied by that probability. Tokens past an expert's capacity get 0.
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
    A router is trained like any other layer, and without help it ends up
    favouring a few experts. Shazeer and colleagues saw this imbalance
    reinforce itself. The favoured experts train faster, so the router picks
    them even more often.

    The table below comes from 4 training runs of a layer with 4 experts and
    a top 1 router. Each run trains for 400 steps on the same smooth target
    function. This router has a bias, which lets it favour one expert for
    every input, so the effect shows up quickly. The routers of Switch and
    Mixtral have no bias. As in Switch, the softmax runs over all 4 scores.
    Each token's one expert is weighted by its probability, for the reason
    moe1 gave.

    | Balancing | Tokens to the busiest expert | Tokens dropped | Error |
    |---|---|---|---|
    | none | 53% to 65% | 23% to 34% | 0.096 to 0.120 |
    | weight 0.01 | 32% to 39% | 1% to 8% | 0.021 to 0.042 |

    With 4 experts, an even share is 25% each. For comparison, a model that
    always outputs 0 has an error of 0.351.

    Imbalance wastes the experts that get few tokens, and it causes a worse
    problem too. Each expert runs on its own part of the hardware, which has
    room for a fixed number of tokens per batch. That number is the expert's
    **capacity**. The Switch paper sets the capacity to an even share of the
    batch times a capacity factor, here 1.25.

    ```python
    math.ceil(1.25 * 256 / 4)    # 80 tokens per expert, for a batch of 256
    ```

    When more tokens choose an expert than its capacity allows, the extra
    tokens are dropped, and the layer outputs 0 for them. With a residual
    connection around the layer, a dropped token passes through unchanged,
    but the layer has done nothing for it. The busier the favourite expert,
    the more of the batch gets dropped.

    The Switch paper fixes this with a balancing loss. The balancing loss is
    multiplied by 0.01 and added to the task's loss.

    ```text
    balance = experts * sum over i of f_i * P_i
    ```

    `f_i` is the fraction of tokens sent to expert i, and `P_i` is the
    router's average probability for expert i over the batch. An even split
    gives a balancing loss of exactly 1. If one expert gets every token and
    all the probability, the loss equals the number of experts. `f_i` comes
    from an argmax, so it passes no gradient, as soft1 showed. The gradient
    reaches the router only through `P_i`, and it lowers the probability of
    the busiest experts the most.
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


    # The experts from moe1.
''' + indented(EXPERT_PART) + '''


    # The smooth target function that the layer learns.
''' + indented(TARGET_PART) + '''


''' + indented(SWITCH_PART) + '''


    # TODO: Return the Switch paper's balancing loss as a tensor. probs holds
    #   the router probabilities, shape (tokens, experts), and chosen holds the
    #   chosen expert of each token, shape (tokens,). The loss is experts times
    #   the sum of f_i * P_i over the experts. f_i is the fraction of tokens
    #   sent to expert i, and P_i is the mean of column i of probs.
    #   Example: for probs = torch.full((4, 2), 0.5) and
    #   chosen = torch.tensor([0, 1, 1, 0]), balancing_loss returns tensor(1.).
    def balancing_loss(
        probs: Tensor,
        chosen: Tensor,
        experts: int,
    ) -> Tensor:
        ...


    # TODO: Return a boolean tensor with one entry per token, True for the
    #   tokens that are kept. For each expert, keep the first capacity tokens
    #   sent to it, in the order they appear in chosen, and drop the rest.
    #   Example: within_capacity(torch.tensor([1, 1, 0, 1, 2, 1, 0]), 3, 2)
    #   returns tensor([ True,  True,  True, False,  True, False,  True]).
    def within_capacity(
        chosen: Tensor,
        experts: int,
        capacity: int,
    ) -> Tensor:
        ...


    # TODO: Finish the line that sets loss. It is the mean squared error of
    #   output against target_of(x), plus alpha times the balancing loss of
    #   probs and chosen.
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
        description="returns experts times the sum of f_i * P_i",
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
        description="keeps the first capacity tokens sent to each expert",
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
        description="adds the balancing loss, so the tokens spread across the experts",
        run=check_train_switch,
    )


MOE2 = Exercise(
    name="moe2",
    chapter=MOE,
    title="Why the router needs a balancing loss",
    lesson=MOE2_LESSON,
    source=MOE2_SOURCE,
    solution=MOE2_SOLUTION,
    hints={
        "balancing_loss": "Divide torch.bincount(chosen, minlength=experts) by len(chosen) to get f as floats. "
        "probs.mean(dim=0) gives P. Then return experts * (f * P).sum().",
        "within_capacity": "Start from torch.zeros(len(chosen), dtype=torch.bool). For each expert, "
        "(chosen == index).nonzero().flatten() lists its tokens in order. Set the first capacity of those to True.",
        "train_switch": "Use F.mse_loss for the error against target_of(x). Then add alpha times balancing_loss, "
        "called with probs, chosen and the number of experts, len(model.experts).",
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
    A mixture of experts holds many parameters but uses only a few of them
    for each token. The parameters a token runs through set its compute,
    which is how much arithmetic the model does for that token. This exercise
    counts both numbers and measures when the trade is worth it.

    An expert that maps 8 features to 16 hidden features and back holds 280
    parameters. Eight experts plus a router with a bias hold 2,312. But each
    token runs through only the router and one expert, which is 352
    parameters. A dense MLP with a hidden width of 128 holds 2,184
    parameters, and every token runs through all of them.

    The table compares these models after 3,000 training steps on two tasks,
    with 3 seeds each. In the regions task, each input lies near one of 16
    centres, and each region has its own linear map from input to target.
    The smooth task uses the target from moe2, one smooth function for all
    inputs.

    | Model | Used per token | Regions error | Smooth error |
    |---|---|---|---|
    | dense, 16 wide | 280 | 1.34 to 1.43 | 0.0148 to 0.0156 |
    | 8 experts | 352 | 0.105 to 0.151 | 0.0118 to 0.0123 |
    | dense, 128 wide | 2,184 | 0.262 to 0.270 | 0.0011 to 0.0013 |

    On the regions task, the router sent 99% or more of each region's inputs
    to a single expert. Each expert took over a couple of regions and learned
    their maps. The experts had about a tenth of the error of the 16 wide
    dense layer, which uses about the same compute. They also had about half
    the error of the 128 wide dense layer, which has about as many
    parameters.

    On the smooth task, the experts still beat the 16 wide dense layer, 0.012
    against 0.015. But one smooth function does not split into parts that
    different experts could handle. So on this task the 128 wide dense layer,
    which runs all of its parameters for every token, does ten times better.

    That is the trade. Experts give a model more parameters for the same
    compute. They help most when the work splits into parts that different
    experts can each handle. In Mixtral and in Google's ST-MoE, the experts
    did not specialize by topic or by language. They specialized by kinds of
    tokens, like punctuation, numbers, names, or `self` in Python code.

    The experts above had no limit on how many tokens each could take. Next
    they ran with a capacity factor of 2.0, which gives each expert room for
    twice an even share. Then under half a percent of the inputs found their
    expert full, and the regions error was 0.112 to 0.185.

    The Switch Transformer paper tried capacity factors from 1.0 to 2.0. Each
    one beat the dense model of the same compute by about the same margin. In
    that paper, a token that finds its expert full passes through unchanged
    on the residual connection. The layer here has no residual connection, so
    a dropped token gets an output of 0. That is why this layer needs the
    extra room.

    The costs are memory, because every expert must be held at once, and a
    router that must be kept balanced. That memory is why large models spread
    their experts across many GPUs. The saving is in multiplications per
    token.
    """,
)

MOE3_SOURCE = source_file(
    '''
    # moe3.py

    layer = 8 * 16 + 16    # 144, the weights and biases of a Linear from 8 to 16


    # TODO: Return the number of parameters in one expert. An expert is a
    #   Linear with bias from features to hidden, then a Linear with bias from
    #   hidden back to features.
    #   Example: dense_parameters(8, 16) returns 280.
    def dense_parameters(
        features: int,
        hidden: int,
    ) -> int:
        ...


    # TODO: Return the number of parameters in a mixture of experts. It has
    #   experts copies of that expert, plus a router, which is a Linear with
    #   bias from features to experts.
    #   Example: moe_parameters(8, 16, 8) returns 2312.
    def moe_parameters(
        features: int,
        hidden: int,
        experts: int,
    ) -> int:
        ...


    # TODO: Return the number of parameters one token runs through, which are
    #   the router and the k experts it picks.
    #   Example: active_parameters(8, 16, 8, 1) returns 352.
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
        description="counts the router and the k experts a token runs through",
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
    title="When more parameters for the same compute pay off",
    lesson=MOE3_LESSON,
    source=MOE3_SOURCE,
    solution=MOE3_SOLUTION,
    hints={
        "dense_parameters": "A Linear from n inputs to m outputs has n * m weights plus m biases. Add up the two "
        "Linears.",
        "moe_parameters": "Multiply dense_parameters by experts. Then add the router's features * experts weights "
        "and its experts biases.",
        "active_parameters": "Multiply dense_parameters by k. Then add the whole router, which runs for every token.",
    },
    introduces=(),
    check=check_moe3,
)


# ---------------------------------------------------------------------------
# dynamic1
# ---------------------------------------------------------------------------

DYNAMIC1_LESSON = prose(
    """
    Four layers in this course compute some of their weights from the input
    itself. They compute these weights again on every forward pass.

    | Layer | Weights computed from the input | What they multiply | Shown in |
    |---|---|---|---|
    | gate | `sigmoid(x W_g)` | each feature of `x W_v` | gating1, derive1 |
    | SwiGLU | `silu(x W_g)` | each feature of `x W_up` | llama2 |
    | attention | `softmax(q k^T / sqrt(d))` | the values of every position | attention5 |
    | routing | a softmax over the top k router scores | the outputs of the chosen experts | moe1 |

    A Linear layer's weights stop changing once training ends, and the same
    numbers multiply every input. In these four layers, the trained weights
    are used to compute other weights, and those other weights depend on the
    input each time.

    ```python
    W_g = torch.tensor([[1.0, -1.0]])
    torch.sigmoid(torch.tensor([[2.0, 0.0]]) @ W_g.T)    # tensor([[0.8808]])
    torch.sigmoid(torch.tensor([[0.0, 2.0]]) @ W_g.T)    # tensor([[0.1192]])
    ```

    The same gate gives two different weights for two different inputs.

    This lets a layer multiply parts of the input together. A Linear layer
    can only scale its inputs and add them up. These layers multiply one
    function of the input by another function of the input. So a gate can
    pass a value only while a switch in the input is on, as gating1 showed.
    Attention can fetch whatever a query asks for, from any position. Routing
    multiplies the skipped experts by 0, so it runs a different network for
    each token.

    Weights computed from the input are also a design tool. When a model must
    choose, look something up or switch between paths, make that choice a
    weight computed from the input. Make the weight soft so it passes a
    gradient, as in 38_soft_design, and train it.

    Routing needs one more tool. `.scatter(dim, index, values)` writes each
    value in a row into the column that `index` gives for it, and leaves every
    other column unchanged. Scattering the top k weights into a row of zeros
    gives a full row of weights, with 0 for each expert that was left out.

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


    # TODO: Return the output of a gate layer. Compute x @ W_value.T, and
    #   multiply each feature of it by its gate, sigmoid(x @ W_gate.T). x has
    #   shape (tokens, features).
    def gate_layer(
        x: Tensor,
        W_gate: Tensor,
        W_value: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return llama2's SwiGLU without its last Linear. Compute x @ W_up.T,
    #   and multiply each feature of it by F.silu(x @ W_gate.T).
    def swiglu_layer(
        x: Tensor,
        W_gate: Tensor,
        W_up: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return attention over x, of shape (steps, features), with no mask.
    #   The queries, keys and values are q = x @ W_q.T, k = x @ W_k.T and
    #   v = x @ W_v.T. The attention weights are a softmax over each row of
    #   (q @ k.T) / sqrt(d), where d is the last dimension of q. The output
    #   mixes the values with those weights.
    def attention_layer(
        x: Tensor,
        W_q: Tensor,
        W_k: Tensor,
        W_v: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the output of a routed layer for x, of shape
    #   (tokens, features). The output of expert i is x @ experts[i].T, and the
    #   router's scores are x @ router.T. Each token gives its top k experts
    #   the weights of a softmax over only their scores, and gives every other
    #   expert a weight of 0. The result is the weighted sum of the experts'
    #   outputs. Running every expert is fine here.
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
        description="multiplies each feature by a gate computed from the input",
        inputs=(
            (x, small_tensor(581, 3, 4) / 2, small_tensor(582, 3, 4) / 2),
        ),
        reference=lambda x, W_gate, W_value: torch.sigmoid(x @ W_gate.T) * (x @ W_value.T),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="swiglu_layer",
        description="multiplies each feature by silu of another projection",
        inputs=(
            (x, small_tensor(583, 6, 4) / 2, small_tensor(584, 6, 4) / 2),
        ),
        reference=lambda x, W_gate, W_up: F.silu(x @ W_gate.T) * (x @ W_up.T),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="attention_layer",
        description="takes a weighted sum of the values at every position, with softmax weights",
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
        "gate_layer": "Compute the gate with torch.sigmoid and the values with x @ W_value.T, then multiply them "
        "entry by entry.",
        "swiglu_layer": "Follow gate_layer, using F.silu instead of torch.sigmoid and W_up instead of W_value.",
        "attention_layer": "Divide q @ k.T by math.sqrt(q.shape[-1]), take the softmax with dim=-1, then multiply "
        "the weights by v with @.",
        "routed_layer": "(x @ router.T).topk(k, dim=-1) gives the top scores and their experts. Scatter the softmax "
        "of those scores into torch.zeros(len(x), len(experts)). Stack the expert outputs with "
        "torch.stack([x @ e.T for e in experts], dim=1). Multiply them by the weights and sum over dim=1.",
    },
    introduces=(),
    check=check_dynamic1,
)
