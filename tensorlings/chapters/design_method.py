"""105_design_method, a method for a module on a new problem: a toy task that isolates one skill, a baseline and a diagnosis from the curves, the soft step loop on a harder algorithm, gradient health before training, and each property tested directly."""

from __future__ import annotations

import contextlib
import copy
import math
import textwrap
import time
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
    reject_missing_return,
    seeded,
    small_tensor,
    value_text,
)
from ..text import prose, source_file
from .linear_layers import expect_module

DESIGN_METHOD = Chapter(
    folder="105_design_method",
    title="The design method",
    readme=source_file(
        """
        # The design method

        A new problem often has no paper to copy. This chapter teaches a method
        for designing a module for such a problem. First, isolate the skill the
        problem needs in a small toy task. Then train models on it, compare them
        with a simple baseline, and read their accuracy curves to see why they
        failed. Next, build the module from an exact algorithm by replacing its
        hard steps with soft ones. Check its gradients before training. Last,
        test directly that the module has the properties it should have. Every
        step is measured on real training runs.

        ## Further reading

        - [Arora and colleagues, Zoology](https://arxiv.org/abs/2312.04927)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        - [Vinyals and colleagues, Matching Networks](https://arxiv.org/abs/1606.04080)
        """,
    ),
)

MEASURE_THREADS = 4


@contextlib.contextmanager
def pinned_threads() -> Iterator[None]:
    """The CPU's thread count changes its sums' rounding, so the lessons' runs pin it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(MEASURE_THREADS)

    try:
        yield
    finally:
        torch.set_num_threads(threads)


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "copy": copy,
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
        "Callable": Callable,
    }
    exec(code, namespace)

    return namespace


# ---------------------------------------------------------------------------
# The recall task, shared by design1, design2 and design5
# ---------------------------------------------------------------------------

RECALL_SETTINGS = '''
KEYS = 16
VALUES = 16
VOCABULARY = KEYS + VALUES
'''

RECALL_START = '''
def recall_batch(
    count: int,
    pairs: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    keys = torch.stack([torch.randperm(KEYS, generator=generator)[:pairs] for _ in range(count)])
    values = torch.randint(KEYS, VOCABULARY, (count, pairs), generator=generator)
'''

RECALL_REST = '''
    order = torch.stack([torch.randperm(pairs, generator=generator) for _ in range(count)])
    rows = torch.arange(count).unsqueeze(1)
    tokens = torch.cat([torch.stack([keys, values], dim=2).flatten(1), keys[rows, order]], dim=1)
    targets = torch.full_like(tokens, -100)
    targets[:, 2 * pairs :] = values[rows, order]

    return tokens, targets
'''

EXACT_SOLUTION = '''
def exact_recall(
    tokens: Tensor,
    pairs: int,
) -> Tensor:
    answers = []

    for sequence in tokens.tolist():
        stored = {sequence[2 * i]: sequence[2 * i + 1] for i in range(pairs)}
        answers.append([stored[query] for query in sequence[2 * pairs :]])

    return torch.tensor(answers)
'''

ACCURACY_SOLUTION = '''
def query_accuracy(
    scores: Tensor,
    targets: Tensor,
) -> float:
    asked = targets != -100

    return (scores.argmax(dim=-1)[asked] == targets[asked]).float().mean().item()
'''

RECALL_PART = RECALL_SETTINGS + RECALL_START + RECALL_REST + "\n" + ACCURACY_SOLUTION

# The three models the lessons compare, each reading the tokens and scoring every place.
MODELS_PART = '''
# One causal attention layer with one head, as multihead2 built it.
class CausalAttention(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.c_attn = nn.Linear(size, 3 * size)
        self.c_proj = nn.Linear(size, size)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.c_attn(x).split(x.shape[-1], dim=2)

        return self.c_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True))


# A stack of attention layers with no MLPs, with residual connections and learned positions, like induction1's model.
class AttentionRecaller(nn.Module):
    def __init__(
        self,
        layers: int,
        size: int = 64,
        longest: int = 96,
    ) -> None:
        super().__init__()
        self.ids = nn.Embedding(VOCABULARY, size)
        self.positions = nn.Embedding(longest, size)
        self.norms = nn.ModuleList([nn.LayerNorm(size) for _ in range(layers)])
        self.layers = nn.ModuleList([CausalAttention(size) for _ in range(layers)])
        self.norm = nn.LayerNorm(size)
        self.head = nn.Linear(size, VOCABULARY)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.ids(tokens) + self.positions(torch.arange(tokens.shape[1]))

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        return self.head(self.norm(x))


class GRURecaller(nn.Module):
    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.ids = nn.Embedding(VOCABULARY, size)
        self.gru = nn.GRU(size, size, batch_first=True)
        self.head = nn.Linear(size, VOCABULARY)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.head(self.gru(self.ids(tokens))[0])


# An MLP over the whole flattened sequence. It sees every position at once, but it works for one sequence length only.
class FlatRecaller(nn.Module):
    def __init__(
        self,
        pairs: int,
        size: int = 16,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.steps = 3 * pairs
        self.ids = nn.Embedding(VOCABULARY, size)
        self.mlp = nn.Sequential(
            nn.Linear(self.steps * size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.steps * VOCABULARY),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.mlp(self.ids(tokens).flatten(1)).view(len(tokens), self.steps, VOCABULARY)
'''

RECALL_REFERENCE = namespace_of(RECALL_PART + EXACT_SOLUTION + MODELS_PART)

TRACK_SOLUTION = '''
def track(
    model: nn.Module,
    next_batch: Callable[[], tuple[Tensor, Tensor]],
    held: tuple[Tensor, Tensor],
    steps: int,
    every: int,
) -> list[tuple[int, float, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curve = []

    for step in range(1, steps + 1):
        tokens, targets = next_batch()
        scores = model(tokens)
        loss = F.cross_entropy(scores.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % every == 0:
            with torch.no_grad():
                curve.append((step, query_accuracy(scores, targets), query_accuracy(model(held[0]), held[1])))

    return curve
'''

DIAGNOSE_SOLUTION = '''
def baseline_accuracy(
    train_targets: Tensor,
    held_targets: Tensor,
) -> float:
    common = torch.bincount(train_targets[train_targets != -100]).argmax()
    asked = held_targets[held_targets != -100]

    return (asked == common).float().mean().item()


def diagnose(
    train_accuracy: float,
    held_accuracy: float,
    fit_bar: float,
    gap_bar: float,
) -> str:
    if train_accuracy < fit_bar:
        return "failed to fit"

    if train_accuracy - held_accuracy > gap_bar:
        return "failed to generalize"

    return "works"
'''

DIAGNOSE_REFERENCE = namespace_of(RECALL_PART + MODELS_PART + TRACK_SOLUTION + DIAGNOSE_SOLUTION)


def fresh_batches(
    pairs: int,
    seed: int,
) -> Callable[[], tuple[Tensor, Tensor]]:
    """A new batch of 64 sequences every step, drawn from one generator."""
    generator = torch.Generator().manual_seed(seed)

    return lambda: RECALL_REFERENCE["recall_batch"](64, pairs, generator)


def fixed_batches(
    pairs: int,
    count: int,
    seed: int,
) -> Callable[[], tuple[Tensor, Tensor]]:
    """Batches of 64 drawn from one fixed set of count sequences, all of it when count is 64 or less."""
    tokens, targets = RECALL_REFERENCE["recall_batch"](count, pairs, torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(seed)

    if count <= 64:
        return lambda: (tokens, targets)

    def batch() -> tuple[Tensor, Tensor]:
        picked = torch.randint(0, count, (64,), generator=generator)

        return tokens[picked], targets[picked]

    return batch


def held_out(pairs: int) -> tuple[Tensor, Tensor]:
    return RECALL_REFERENCE["recall_batch"](1000, pairs, torch.Generator().manual_seed(99))


def recall_model(
    kind: str,
    pairs: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)

    if kind == "attention, 1 layer":
        return RECALL_REFERENCE["AttentionRecaller"](1)

    if kind == "attention, 2 layers":
        return RECALL_REFERENCE["AttentionRecaller"](2)

    if kind == "GRU":
        return RECALL_REFERENCE["GRURecaller"]()

    return RECALL_REFERENCE["FlatRecaller"](pairs)


def final_accuracies(
    kind: str,
    pairs: int,
    seed: int,
    batches: Callable[[], tuple[Tensor, Tensor]],
    steps: int = 1500,
) -> tuple[float, float]:
    """The last point of a tracked run, its training accuracy and its held out one."""
    model = recall_model(kind, pairs, seed)
    curve = DIAGNOSE_REFERENCE["track"](model, batches, held_out(pairs), steps, steps)

    return curve[-1][1], curve[-1][2]


# ---------------------------------------------------------------------------
# The soft nearest neighbours of design3
# ---------------------------------------------------------------------------

EPISODES_PART = '''
FEATURES = 8
CLASSES = 4
SHOTS = 5


# Each episode draws 4 new class centres in the first 2 features. The other 6 features are noise, and the
# noise is larger than the gaps between classes. The labels 0 to 3 mean something new in every episode.
def episodes(
    count: int,
    generator: torch.Generator,
    queries: int = 10,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    centres = 2.0 * torch.randn(count, CLASSES, 2, generator=generator)
    labels = torch.arange(CLASSES).repeat_interleave(SHOTS).expand(count, -1).clone()
    query_labels = torch.randint(0, CLASSES, (count, queries), generator=generator)

    def points(classes: Tensor) -> Tensor:
        middles = centres.gather(1, classes.unsqueeze(2).expand(-1, -1, 2))
        signal = middles + 0.3 * torch.randn(count, classes.shape[1], 2, generator=generator)
        noise = 2.0 * torch.randn(count, classes.shape[1], FEATURES - 2, generator=generator)

        return torch.cat([signal, noise], dim=2)

    return points(labels), labels, points(query_labels), query_labels
'''

KNN_SOLUTION = '''
def knn_predict(
    memory: Tensor,
    labels: Tensor,
    queries: Tensor,
    k: int,
) -> Tensor:
    nearest = torch.cdist(queries, memory).topk(k, dim=-1, largest=False).indices
    picked = labels.gather(1, nearest.flatten(1)).view(nearest.shape)
    votes = F.one_hot(picked, CLASSES).sum(dim=2)

    return votes.argmax(dim=-1)
'''

SOFT_VOTES_SOLUTION = '''
def soft_votes(
    memory: Tensor,
    labels: Tensor,
    queries: Tensor,
    temperature: Tensor | float,
) -> Tensor:
    weights = torch.softmax(-torch.cdist(queries, memory) ** 2 / temperature, dim=-1)

    return weights @ F.one_hot(labels, CLASSES).float()
'''

NEIGHBORS_INIT = '''
class SoftNeighbors(nn.Module):
    def __init__(
        self,
        features: int,
        size: int,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(features, size, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
'''

NEIGHBORS_FORWARD = '''
    def forward(
        self,
        memory: Tensor,
        labels: Tensor,
        queries: Tensor,
    ) -> Tensor:
        temperature = self.log_temperature.exp()
        votes = soft_votes(self.project(memory), labels, self.project(queries), temperature)

        return torch.log(votes + 1e-6)
'''

NEIGHBORS_REFERENCE = namespace_of(EPISODES_PART + KNN_SOLUTION + SOFT_VOTES_SOLUTION + NEIGHBORS_INIT + NEIGHBORS_FORWARD)


def train_neighbors(
    model: nn.Module,
    steps: int,
    seed: int,
) -> nn.Module:
    """design3's training, Adam at 0.01 on batches of 32 episodes, scored by the log of the soft votes."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(steps):
        memory, labels, queries, truth = NEIGHBORS_REFERENCE["episodes"](32, generator)
        loss = F.nll_loss(model(memory, labels, queries).flatten(0, 1), truth.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def held_episodes() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return NEIGHBORS_REFERENCE["episodes"](500, torch.Generator().manual_seed(99))


def share_right(
    predicted: Tensor,
    truth: Tensor,
) -> float:
    return (predicted == truth).float().mean().item()


# ---------------------------------------------------------------------------
# The health checks of design4
# ---------------------------------------------------------------------------

HEALTH_SOLUTION = '''
def identity_gap(
    block: nn.Module,
    x: Tensor,
) -> float:
    with torch.no_grad():
        return (block(x) - x).abs().max().item()


def silent_after_one_step(
    model: nn.Module,
    loss_of: Callable[[nn.Module], Tensor],
    lr: float,
) -> list[str]:
    copied = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copied.parameters(), lr=lr)
    loss_of(copied).backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_of(copied).backward()

    return [name for name, parameter in copied.named_parameters() if parameter.grad is None or not parameter.grad.any()]
'''

BLOCK_START = '''
class GatedBlock(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.gate = nn.Linear(size, size)
        self.up = nn.Linear(size, 4 * size)
        self.down = nn.Linear(4 * size, size)
'''

BLOCK_ZERO = '''
        with torch.no_grad():
            self.down.weight.zero_()
            self.down.bias.zero_()
'''

BLOCK_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        gate = torch.sigmoid(self.gate(h))

        return x + gate * self.down(F.relu(self.up(h)))
'''

# The block as handed to the learner, with a random start and a hard gate.
BROKEN_FORWARD = '''
    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        gate = (self.gate(h) > 0).float()

        return x + gate * self.down(F.relu(self.up(h)))
'''

HEALTH_REFERENCE = namespace_of(HEALTH_SOLUTION + BLOCK_START + BLOCK_ZERO + BLOCK_FORWARD)
BROKEN_REFERENCE = namespace_of(BLOCK_START + BROKEN_FORWARD)

SWITCH_INPUTS = 8
BLOCK_SIZE = 32


def switch_data(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """A task where a gate pays: the answer is input 0 when input 1 is above 0, and input 2 when it isn't."""
    x = torch.randn(count, SWITCH_INPUTS, generator=generator)

    return x, torch.where(x[:, 1] > 0, x[:, 0], x[:, 2])


class VariantBlock(nn.Module):
    """design4's block with each fix on or off, for the lesson's table. The table's blocks are loud: no norm, and up
    and down drawn with Kaiming's start, so a block that doesn't start as the identity changes its input a lot."""

    def __init__(
        self,
        size: int,
        zero_start: bool,
        soft_gate: bool,
        residual: bool,
        loud: bool = False,
    ) -> None:
        super().__init__()
        self.norm = None if loud else nn.LayerNorm(size)
        self.gate = nn.Linear(size, size)
        self.up = nn.Linear(size, 4 * size)
        self.down = nn.Linear(4 * size, size)
        self.soft_gate = soft_gate
        self.residual = residual

        if loud:
            nn.init.kaiming_normal_(self.up.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")

        if zero_start:
            with torch.no_grad():
                self.down.weight.zero_()
                self.down.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        h = x if self.norm is None else self.norm(x)
        score = self.gate(h)
        gate = torch.sigmoid(score) if self.soft_gate else (score > 0).float()
        branch = gate * self.down(F.relu(self.up(h)))

        return x + branch if self.residual else branch


class BlockStack(nn.Module):
    """12 blocks between a Linear in and a Linear out, for the switching task."""

    def __init__(
        self,
        make_block: Callable[[], nn.Module],
        depth: int = 12,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(SWITCH_INPUTS, BLOCK_SIZE)
        self.blocks = nn.ModuleList([make_block() for _ in range(depth)])
        self.out = nn.Linear(BLOCK_SIZE, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.inp(x)

        for block in self.blocks:
            h = block(h)

        return self.out(h).squeeze(-1)


# ---------------------------------------------------------------------------
# The property tests of design5
# ---------------------------------------------------------------------------

GAP_PART = '''
# equivariance1's gap, given. It is 0 when applying g before f gives the same result as applying g after f.
def equivariance_gap(
    f: Callable[[Tensor], Tensor],
    g: Callable[[Tensor], Tensor],
    x: Tensor,
) -> float:
    return (f(g(x)) - g(f(x))).abs().max().item()
'''

PROPERTY_SOLUTION = '''
def shuffled(
    x: Tensor,
    order: Tensor,
) -> Tensor:
    return x[:, order]


def shifted(
    x: Tensor,
    steps: int,
) -> Tensor:
    return torch.roll(x, steps, dims=1)


def prefix_gap(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    extra: Tensor,
) -> float:
    with torch.no_grad():
        longer = f(torch.cat([x, extra], dim=1))

        return (longer[:, : x.shape[1]] - f(x)).abs().max().item()


def property_gaps(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    transforms: dict[str, Callable[[Tensor], Tensor]],
) -> dict[str, float]:
    with torch.no_grad():
        return {name: equivariance_gap(f, transform, x) for name, transform in transforms.items()}
'''

PROPERTY_REFERENCE = namespace_of(GAP_PART + PROPERTY_SOLUTION)

TABLE_SIZE = 16
TABLE_STEPS = 12


class TableAttention(nn.Module):
    """One attention layer for design5's table, with or without the causal mask and learned positions."""

    def __init__(
        self,
        causal: bool,
        positions: bool,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.positions = nn.Embedding(64, TABLE_SIZE) if positions else None
        self.c_attn = nn.Linear(TABLE_SIZE, 3 * TABLE_SIZE)
        self.c_proj = nn.Linear(TABLE_SIZE, TABLE_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        if self.positions is not None:
            x = x + self.positions(torch.arange(x.shape[1]))

        q, k, v = self.c_attn(x).split(TABLE_SIZE, dim=2)

        return self.c_proj(F.scaled_dot_product_attention(q, k, v, is_causal=self.causal))


class TableConv(nn.Module):
    """A convolution of width 3 along the steps, padded on both sides, or on the left only when causal."""

    def __init__(
        self,
        padding: str,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(TABLE_SIZE, TABLE_SIZE, 3)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.transpose(1, 2)

        if self.padding == "causal":
            padded = F.pad(channels, (2, 0))
        else:
            padded = F.pad(channels, (1, 1), mode=self.padding)

        return self.conv(padded).transpose(1, 2)


class TableGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(TABLE_SIZE, TABLE_SIZE, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.gru(x)[0]


class TableFlat(nn.Module):
    """A Linear over the whole flattened sequence, which takes one length only."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(TABLE_STEPS * TABLE_SIZE, TABLE_STEPS * TABLE_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1)).view(x.shape)


def table_modules() -> dict[str, nn.Module]:
    torch.manual_seed(0)

    return {
        "MLP on each step": nn.Sequential(nn.Linear(TABLE_SIZE, 32), nn.ReLU(), nn.Linear(32, TABLE_SIZE)),
        "convolution, zero padding": TableConv("constant"),
        "convolution, circular padding": TableConv("circular"),
        "causal convolution": TableConv("causal"),
        "attention, no positions": TableAttention(causal=False, positions=False),
        "causal attention, no positions": TableAttention(causal=True, positions=False),
        "attention, learned positions": TableAttention(causal=False, positions=True),
        "GRU": TableGRU(),
        "MLP over the whole sequence": TableFlat(),
    }


# ---------------------------------------------------------------------------
# design1
# ---------------------------------------------------------------------------

DESIGN1_LESSON = prose(
    """
    A new problem rarely tells you which skill a model needs to solve it. So
    the method starts by isolating one skill. A **toy task** is a small task
    that tests that skill and nothing else. A short exact program can solve
    it, so you know it is solvable. One setting controls how hard it is. And
    you can generate as many fresh examples as you want.

    Arora and colleagues built their Zoology study this way. They compared
    attention models with gated convolution models on real text. 82% of the
    difference in quality between the two came from one skill, recalling
    something that appeared earlier in the text. On that skill, a 70 million
    parameter attention model beat a 1.4 billion parameter convolution model.

    Their toy task is called multi query associative recall. The model reads
    a list of key and value pairs. Then it sees several of the keys again and
    must answer each one's value.

    ```text
    A 4 B 3 C 6 F 1 E 2   then   A ? C ? F ? E ? B ?   answers 4 6 1 2 3
    ```

    In this chapter, keys are token ids 0 to 15 and values are ids 16 to 31. A
    sequence with `pairs` pairs first lists them as key, value, key, value.
    Then the same keys follow again in a new order. Each repeated key is a
    query, and the model must predict its value at that position. Only the
    query positions are scored. Every other position gets the target -100,
    which `F.cross_entropy` ignores, as in sft1.

    The number of pairs, `pairs`, controls how hard the task is. Three models
    were trained for 1,500 steps on freshly generated sequences. The table
    shows the fraction of queries each one answered correctly on new
    sequences, as the range over 3 seeds.

    | Model | 4 pairs | 8 pairs | 16 pairs |
    |---|---|---|---|
    | GRU | 34.1% to 34.9% | 25.8% to 26.0% | 19.3% to 19.5% |
    | attention, 1 layer | 34.0% to 34.1% | 25.9% to 26.1% | 19.4% to 19.6% |
    | attention, 2 layers | 100% | 100% | 99.99% to 100% |

    Your `exact_recall` gets 100%, which shows the task can be solved exactly.
    A random guess among the 16 values gets 1 in 16 right. The GRU and the 1
    layer attention model score within one percentage point of each other,
    well above a random guess.

    There is a shortcut that needs no recall. For every query, answer the
    value that appears most often in that sequence. This shortcut scores
    34.15%, 25.95% and 19.525% at 4, 8 and 16 pairs. Those match the GRU and 1
    layer rows, so both models learned the shortcut and neither one recalls.
    Two attention layers can shift information to the next position and then
    match on it, as induction1 showed. They recall every pair.

    Each training run took under a minute on a laptop CPU. In that time the
    toy task showed which skill one attention layer lacks. On real text, the
    same finding would take far longer to see.
    """,
)

DESIGN1_SOURCE = source_file(
    '''
    # design1.py

    import torch
    from torch import Tensor

''' + indented(RECALL_SETTINGS) + '''

    laid_out = torch.stack([torch.tensor([[3, 9]]), torch.tensor([[20, 17]])], dim=2).flatten(1)    # tensor([[ 3, 20,  9, 17]])
    rows = torch.arange(2).unsqueeze(1)
    picked = torch.tensor([[5, 6], [7, 8]])[rows, torch.tensor([[1, 0], [0, 1]])]    # tensor([[6, 5], [7, 8]])
    asked = torch.tensor([-100, 3, -100]) != -100    # tensor([False,  True, False])


    # TODO: Finish building count sequences of the recall task. keys and values
    #   are already drawn, each of shape (count, pairs). For each sequence, draw
    #   a query order with torch.randperm(pairs, generator=generator). tokens
    #   holds the pairs laid out key, value, key, value, as laid_out shows,
    #   followed by the keys again in the query order. So tokens has shape
    #   (count, 3 * pairs). targets has the same shape. It is -100 everywhere
    #   except at the query positions, which hold each query's value. Return
    #   (tokens, targets), both integer tensors.
    def recall_batch(
        count: int,
        pairs: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        keys = torch.stack([torch.randperm(KEYS, generator=generator)[:pairs] for _ in range(count)])
        values = torch.randint(KEYS, VOCABULARY, (count, pairs), generator=generator)
        ...


    # TODO: Return the exact answer to every query, using plain Python. For each
    #   sequence, build a dict that maps each key to its value, then look up
    #   each query, as exact_lookup in derive2 did with a loop. tokens is laid
    #   out as recall_batch makes it. Return an integer tensor of shape
    #   (count, pairs).
    #   Example: exact_recall(torch.tensor([[3, 20, 9, 17, 9, 3]]), 2) returns
    #   tensor([[17, 20]]).
    def exact_recall(
        tokens: Tensor,
        pairs: int,
    ) -> Tensor:
        ...


    # TODO: Return the accuracy on the query positions, as a Python float. The
    #   query positions are the ones where targets is not -100. At each one,
    #   the prediction is the id with the highest score. The accuracy is the
    #   fraction of query positions where the prediction equals the target.
    #   scores has shape (count, steps, VOCABULARY) and targets has shape
    #   (count, steps).
    def query_accuracy(
        scores: Tensor,
        targets: Tensor,
    ) -> float:
        ...
    ''',
)

DESIGN1_SOLUTION = source_file(
    '''
    # design1.py, a solution for comparison

    import torch
    from torch import Tensor

''' + indented(RECALL_SETTINGS) + '''

    laid_out = torch.stack([torch.tensor([[3, 9]]), torch.tensor([[20, 17]])], dim=2).flatten(1)
    rows = torch.arange(2).unsqueeze(1)
    picked = torch.tensor([[5, 6], [7, 8]])[rows, torch.tensor([[1, 0], [0, 1]])]
    asked = torch.tensor([-100, 3, -100]) != -100


''' + indented(RECALL_START + RECALL_REST) + '''


''' + indented(EXACT_SOLUTION) + '''


''' + indented(ACCURACY_SOLUTION) + '''
    ''',
)


def check_recall_batch(
    found: object,
    count: int,
    pairs: int,
    seed: int,
) -> None:
    """The task's properties, so any way of laying it out that keeps them passes."""
    call = f"recall_batch({count}, {pairs}, a generator seeded with {seed})"
    made = found(count, pairs, torch.Generator().manual_seed(seed))

    if not isinstance(made, tuple) or len(made) != 2:
        raise Mismatch(f"{call} should give tokens and targets in a tuple.")

    tokens, targets = made

    for name, part in (("tokens", tokens), ("targets", targets)):
        if not isinstance(part, Tensor):
            raise Mismatch(f"{call}'s {name} is {value_text(part)}, but it should be a tensor.")

        if part.shape != (count, 3 * pairs):
            raise Mismatch(f"{call}'s {name} has shape {tuple(part.shape)}, but it should be ({count}, {3 * pairs}).")

        if part.dtype.is_floating_point:
            raise Mismatch(f"{call}'s {name} holds {part.dtype} numbers, but ids and targets are whole numbers.")

    generator = torch.Generator().manual_seed(seed)
    keys = torch.stack([torch.randperm(16, generator=generator)[:pairs] for _ in range(count)])
    values = torch.randint(16, 32, (count, pairs), generator=generator)

    if not torch.equal(tokens[:, : 2 * pairs : 2], keys) or not torch.equal(tokens[:, 1 : 2 * pairs : 2], values):
        raise Mismatch(f"{call}'s first {2 * pairs} tokens should be the keys and values drawn for you, laid out key, value, key, value.")

    queries = tokens[:, 2 * pairs :]

    if not torch.equal(queries.sort(dim=1).values, keys.sort(dim=1).values):
        raise Mismatch(f"{call}'s last {pairs} tokens should be each sequence's keys again, in a new order.")

    if torch.equal(queries, keys):
        raise Mismatch(f"{call} asks every sequence's keys in the order they were stored. Draw a new order with torch.randperm.")

    # Where each query sat among its sequence's keys. One order for the whole batch would let a model learn the order.
    orders = (queries.unsqueeze(2) == keys.unsqueeze(1)).int().argmax(dim=2)

    if count > 1 and (orders == orders[0]).all():
        raise Mismatch(f"{call} asks every sequence's keys in the same new order. Draw a new order for each sequence.")

    if (targets[:, : 2 * pairs] != -100).any():
        raise Mismatch(f"{call}'s targets should be -100 at the {2 * pairs} positions of the pairs, so the loss ignores them.")

    answers = RECALL_REFERENCE["exact_recall"](tokens, pairs)

    if not torch.equal(targets[:, 2 * pairs :], answers):
        raise Mismatch(f"{call}'s targets at the query positions should be each query's value, the value stored with that key.")

    again = found(count, pairs, torch.Generator().manual_seed(seed))

    if not torch.equal(again[0], tokens):
        raise Mismatch(f"{call} gave other tokens on a second call with the same seed. Draw every order from generator.")


def recall_tokens(
    count: int,
    pairs: int,
    seed: int,
) -> Described:
    tokens, _ = RECALL_REFERENCE["recall_batch"](count, pairs, torch.Generator().manual_seed(seed))

    return Described(tokens, f"tokens of {count} sequences of {pairs} pairs")


def expect_whole_numbers(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but the answers are value ids, whole numbers.")


def scored_places(
    seed: int,
    count: int,
    steps: int,
) -> tuple[Described, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(count, steps, 32, generator=generator)
    targets = torch.randint(0, 32, (count, steps), generator=generator)
    targets[torch.rand(count, steps, generator=generator) < 0.5] = -100
    # One place right on purpose, so a share of 0 can't pass.
    targets[0, -1] = scores[0, -1].argmax()

    return Described(scores, f"scores of shape ({count}, {steps}, 32)"), targets


def check_design1(checker: Checker) -> None:
    for count, pairs, seed in ((50, 5, 951), (7, 3, 952)):
        checker.test(
            function="recall_batch",
            description="lays out the pairs then the queries, with targets only at the queries",
            run=lambda found, count=count, pairs=pairs, seed=seed: check_recall_batch(found, count, pairs, seed),
        )

    check_on_inputs(
        checker=checker,
        function="exact_recall",
        description="answers every query exactly",
        inputs=(
            (recall_tokens(4, 3, 953), 3),
            (recall_tokens(6, 5, 954), 5),
        ),
        reference=RECALL_REFERENCE["exact_recall"],
        expect=expect_whole_numbers,
    )
    check_on_inputs(
        checker=checker,
        function="query_accuracy",
        description="scores only the query positions",
        inputs=(
            scored_places(955, 4, 6),
            scored_places(956, 8, 9),
        ),
        reference=RECALL_REFERENCE["query_accuracy"],
        expect=expect_close,
    )


DESIGN1 = Exercise(
    name="design1",
    chapter=DESIGN_METHOD,
    title="A toy task for one skill",
    lesson=DESIGN1_LESSON,
    source=DESIGN1_SOURCE,
    solution=DESIGN1_SOLUTION,
    hints={
        "recall_batch": "Build order by stacking one torch.randperm(pairs, generator=generator) per sequence. If "
        "rows is torch.arange(count).unsqueeze(1), then keys[rows, order] reorders each row of keys. Join the laid "
        "out pairs and the reordered keys with torch.cat on dim=1. Start targets from torch.full_like(tokens, -100) "
        "and fill in its last pairs positions.",
        "exact_recall": "For each row of tokens.tolist(), build a dict that maps sequence[2 * i] to "
        "sequence[2 * i + 1]. Then look up every token after the first 2 * pairs. Calling torch.tensor on the "
        "lists of answers gives an integer tensor.",
        "query_accuracy": "Make a mask of the positions where targets != -100. Compare scores.argmax(dim=-1) with "
        "targets at those positions. Convert the result to float, take the mean, and call .item().",
    },
    introduces=("toy task",),
    check=check_design1,
)


# ---------------------------------------------------------------------------
# design2
# ---------------------------------------------------------------------------

DESIGN2_LESSON = prose(
    """
    An accuracy of 60% means little until you know what a trivial method
    would score. Karpathy's recipe for training neural networks sets up "dumb
    baselines" early. The simplest baseline ignores the input entirely. Here
    it always answers the value that was most common in the training targets.
    It gets 6.3% of queries right, about 1 in 16.

    Next, track two accuracy curves during training. One is the accuracy on
    the batches the model trains on. The other is the accuracy on held out
    sequences that the model never trains on. The two ways a model can fail
    look different on these curves.

    - Failed to fit: even the training accuracy stays low. Either the model cannot express the skill, or its training is broken.
    - Failed to generalize: training accuracy is high but held out accuracy is low. The model memorized its training examples instead of learning the skill.

    The table shows each model at 8 pairs after 1,500 steps, as the range
    over 3 seeds. "64 fixed" means the model trains on the same 64 sequences
    over and over. "Fresh sequences" means every batch is newly generated.

    | Model and data | Training | Held out | Diagnosis |
    |---|---|---|---|
    | GRU, fresh sequences | 24.2% to 27.1% | 25.8% to 26.0% | failed to fit |
    | MLP over the sequence, 64 fixed | 100% | 7.3% to 7.6% | failed to generalize |
    | attention, 2 layers, 64 fixed | 100% | 14.9% to 15.0% | failed to generalize |
    | attention, 2 layers, 256 fixed | 99.8% to 100% | 21.6% to 99.5% | depends on the seed |
    | attention, 2 layers, 1,024 fixed | 100% | 100% | works |
    | attention, 2 layers, fresh sequences | 100% | 100% | works |

    The diagnosis tells you what to change. A model that failed to fit needs
    a layer that can express the skill, or a fix to its training. design4
    shows how to check the training. More data will not help such a model.
    The GRU never got past design1's shortcut, at 4 pairs or at 8, while two
    attention layers recalled every pair.

    A model that failed to generalize needs more data, or a layer whose
    structure suits the task better. With 256 fixed sequences, two seeds
    learned the skill and one memorized its examples. With 1,024 fixed
    sequences, and also with 4,096, every seed recalled every pair. A toy
    task can generate unlimited fresh data, which is one more reason to start
    with one.

    Karpathy's order is to fit first and generalize second. First get a model
    that can fit the training set. Then trade some training accuracy for
    better held out accuracy.
    """,
)

DESIGN2_GIVEN = RECALL_PART + "\n" + MODELS_PART

DESIGN2_SOURCE = source_file(
    '''
    # design2.py

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    counts = torch.bincount(torch.tensor([20, 17, 20]))
    common = counts.argmax()    # tensor(20), the id that appears most often
    every = [step for step in range(1, 7) if step % 3 == 0]    # [3, 6]


    # design1's task and accuracy function, and the three models that the
    # lessons compare. These are given.
''' + indented(DESIGN2_GIVEN) + '''


    # TODO: Return the accuracy of a baseline that never reads the input, as a
    #   Python float. Find the value that appears most often in train_targets
    #   at its query positions, the ones that are not -100. Return the fraction
    #   of held_targets' query positions whose target equals that value.
    #   Example: baseline_accuracy(torch.tensor([[-100, 20, 20, 17]]),
    #   torch.tensor([[-100, 20, 17, 17]])) returns about 0.3333.
    def baseline_accuracy(
        train_targets: Tensor,
        held_targets: Tensor,
    ) -> float:
        ...


    # TODO: Train model with torch.optim.AdamW at lr=1e-3 for `steps` steps.
    #   Each step gets (tokens, targets) from next_batch() and minimizes the
    #   cross entropy over every position. Count steps from 1. When the step
    #   number is a multiple of `every`, after that step's update, append a
    #   tuple of 3 numbers to the curve. They are the step number, the
    #   query_accuracy of the scores the model just trained on, and the
    #   query_accuracy on held, a (tokens, targets) pair of held out sequences.
    #   Score held inside torch.no_grad(). Return the curve.
    def track(
        model: nn.Module,
        next_batch: Callable[[], tuple[Tensor, Tensor]],
        held: tuple[Tensor, Tensor],
        steps: int,
        every: int,
    ) -> list[tuple[int, float, float]]:
        ...


    # TODO: Return "failed to fit" when train_accuracy is below fit_bar,
    #   "failed to generalize" when train_accuracy is more than gap_bar above
    #   held_accuracy, and "works" otherwise.
    #   Example: diagnose(0.95, 0.5, 0.9, 0.25) returns "failed to generalize".
    def diagnose(
        train_accuracy: float,
        held_accuracy: float,
        fit_bar: float,
        gap_bar: float,
    ) -> str:
        ...
    ''',
)

DESIGN2_SOLUTION = source_file(
    '''
    # design2.py, a solution for comparison

    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    counts = torch.bincount(torch.tensor([20, 17, 20]))
    common = counts.argmax()
    every = [step for step in range(1, 7) if step % 3 == 0]


''' + indented(DESIGN2_GIVEN) + '''


''' + indented(DIAGNOSE_SOLUTION.split("\n\n\ndef diagnose")[0]) + '''


''' + indented(TRACK_SOLUTION) + '''


''' + indented("def diagnose" + DIAGNOSE_SOLUTION.split("\n\n\ndef diagnose")[1]) + '''
    ''',
)


class HeldRecorder:
    """Notes whether autograd was on for each call that scores the held out sequences, which have their own size."""

    def __init__(
        self,
        model: nn.Module,
        held_count: int,
    ) -> None:
        self.recording: list[bool] = []
        model.register_forward_pre_hook(self.note)
        self.held_count = held_count

    def note(
        self,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
    ) -> None:
        if len(inputs[0]) == self.held_count:
            self.recording.append(torch.is_grad_enabled())


def small_recaller() -> nn.Module:
    torch.manual_seed(0)
    model = RECALL_REFERENCE["AttentionRecaller"](2, 16)

    # The keys' third of c_attn's bias gets a true gradient of 0, since the softmax ignores a shift every key shares.
    # Its float noise differs between valid ways of writing the loss, and AdamW scales noise up to real steps, so
    # the check's model leaves that bias out.
    for layer in model.layers:
        layer.c_attn.bias = None

    return model


def check_track(
    found: object,
    fixed: bool,
) -> None:
    held = RECALL_REFERENCE["recall_batch"](40, 3, torch.Generator().manual_seed(961))
    reference_model = small_recaller()
    expected = DIAGNOSE_REFERENCE["track"](reference_model, track_batches(fixed), held, 6, 3)
    model = small_recaller()
    recorder = HeldRecorder(model, 40)
    data = "a fixed set of 64" if fixed else "fresh sequences"
    call = f"track(a small 2 layer attention model, batches of 3 pairs from {data}, 40 held out sequences, 6, 3)"
    actual = found(model, track_batches(fixed), held, 6, 3)

    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, list) or len(actual) != len(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a list of {len(expected)} points, at steps 3 and 6.")

    for point, wanted in zip(actual, expected):
        if not isinstance(point, tuple) or len(point) != 3:
            raise Mismatch(f"{call} holds {value_text(point)}, but each point should be a tuple of 3 numbers, the step and two accuracies.")

        if point[0] != wanted[0]:
            raise Mismatch(f"{call} has a point at step {value_text(point[0])}, but the points should come at steps 3 and 6.")

        for name, got, should in (("training", point[1], wanted[1]), ("held out", point[2], wanted[2])):
            if isinstance(got, Tensor):
                raise Mismatch(f"{call}'s {name} accuracy at step {wanted[0]} is a tensor. query_accuracy gives a plain number.")

            if abs(got - should) > 1e-6:
                raise Mismatch(
                    f"{call}'s {name} accuracy at step {wanted[0]} is {got:.4f}, but it should be {should:.4f}. Train with "
                    "AdamW at 1e-3, zero the gradients each step, and score the training batch with the scores it trained on.",
                )

    if any(recorder.recording):
        raise Mismatch(f"{call} scored the held out sequences while recording for autograd. Score them inside torch.no_grad().")

    # Six steps barely move the accuracies, so the weights show a slip in the training itself.
    for (name, trained), wanted in zip(model.state_dict().items(), reference_model.state_dict().values()):
        if not torch.allclose(trained, wanted, atol=1e-6):
            raise Mismatch(
                f"After {call}, the model's {name} differs from 6 steps of the TODO's training. Use AdamW at "
                "lr=1e-3, and zero the gradients before each backward.",
            )


def track_batches(fixed: bool) -> Callable[[], tuple[Tensor, Tensor]]:
    if fixed:
        tokens, targets = RECALL_REFERENCE["recall_batch"](64, 3, torch.Generator().manual_seed(962))

        return lambda: (tokens, targets)

    generator = torch.Generator().manual_seed(963)

    return lambda: RECALL_REFERENCE["recall_batch"](64, 3, generator)


def baseline_targets(
    seed: int,
    common: int,
    other: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train = torch.randint(16, 32, (6, 8), generator=generator)
    train[:, :3] = -100
    # One value far more common than the rest, so the answer has no tie.
    train[:, 3] = common
    held = torch.randint(16, 32, (5, 8), generator=generator)
    held[:, 4:] = -100
    held[:, 0] = common
    # The held out places favor another value, so finding the common one there gives another share.
    held[:, 1:3] = other

    return train, held


def check_design2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="baseline_accuracy",
        description="always answers the value most common in training",
        inputs=(
            baseline_targets(964, 23, 18),
            baseline_targets(965, 30, 21),
        ),
        reference=DIAGNOSE_REFERENCE["baseline_accuracy"],
        expect=expect_close,
    )

    for fixed in (False, True):
        checker.test(
            function="track",
            description="trains and records both curves",
            run=lambda found, fixed=fixed: check_track(found, fixed),
        )

    check_on_inputs(
        checker=checker,
        function="diagnose",
        description="tells failing to fit from failing to generalize",
        inputs=(
            (0.6, 0.55, 0.9, 0.25),
            (0.95, 0.5, 0.9, 0.25),
            (1.0, 0.75, 0.9, 0.25),
            (0.9, 0.9, 0.9, 0.25),
            (0.5, 0.0, 0.9, 0.25),
        ),
        reference=DIAGNOSE_REFERENCE["diagnose"],
        expect=expect_value,
    )


DESIGN2 = Exercise(
    name="design2",
    chapter=DESIGN_METHOD,
    title="A baseline and a diagnosis",
    lesson=DESIGN2_LESSON,
    source=DESIGN2_SOURCE,
    solution=DESIGN2_SOLUTION,
    hints={
        "baseline_accuracy": "Count the training targets that are not -100 with torch.bincount, and take .argmax() "
        "to get the most common value. Compare that value with the held targets that are not -100, convert the "
        "result to float and take the mean.",
        "track": "This is gpt3's training loop with a curve added. For F.cross_entropy, flatten the scores to shape "
        "(positions, VOCABULARY) and the targets to one dimension. It ignores the -100 targets. When "
        "step % every == 0, score held[0] under torch.no_grad() and append the three numbers.",
        "diagnose": "Check the fit first, with train_accuracy < fit_bar. Then check the gap between the two "
        "accuracies, train_accuracy - held_accuracy > gap_bar.",
    },
    introduces=(),
    check=check_design2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# design3
# ---------------------------------------------------------------------------

DESIGN3_LESSON = prose(
    """
    In 38_soft_design you made two small algorithms soft. Each had one or two
    hard steps that pass no gradient, and you replaced them with smooth
    versions that do. Now take k nearest neighbours. It labels a query point
    by looking at the labelled points closest to it.

    1. Measure the distance from the query to every point in memory.
    2. Keep the k nearest points. This is a hard choice, made with `topk`.
    3. Count how many of those k points have each label. These counts are the votes.
    4. Answer the class with the most votes. This is another hard choice, made with `argmax`.

    Steps 2 and 4 pass no gradient. The soft version replaces them. It gives
    every point in memory a weight, the softmax over the points of
    `-(distance ** 2) / temperature`, as derive2 weighted its keys. Close points
    get large weights. The vote for a class becomes the sum of the weights of
    the points with that label. So each query gets a probability for each
    class.

    The task is made of episodes. Each episode has 4 new classes, with 5
    labelled points of each class in memory. Every point has 8 features. The
    first 2 features depend on the class. The other 6 are random noise, larger
    than the gaps between classes. The labels mean something new in each
    episode, so no fixed classifier can learn them.

    Because every step is now soft, the distance itself can be trained. A
    Linear layer projects every point before the distances are measured.
    Training learns this projection together with the temperature. The table
    shows the accuracy on 500 new episodes, as the range over 3 seeds.

    | Model | Right |
    |---|---|
    | hard, k 1, all 8 features | 57.7% |
    | hard, k 5, all 8 features | 53.9% |
    | an MLP on the query alone | 24.8% to 25.7% |
    | soft, untrained | 43.2% to 49.0% |
    | soft, trained 300 steps | 95.7% to 95.8% |
    | hard, k 5, the trained projection | 95.8% to 95.9% |
    | hard, k 5, only the 2 features that matter | 96.0% |

    The MLP scores about 1 in 4, the same as a random guess, because the
    labels change every episode. In the trained projection's weight matrix,
    the columns for the 2 class features have length 1.8 or more. The columns
    for the 6 noise features have length 0.09 or less. So training found which
    features matter without being told. The hard algorithm does as well as the
    soft one when it uses the trained projection. So training the soft
    version also improved the exact algorithm.

    This model is attention over a memory of labelled points. The query
    attends to the memory points, and the values are their one hot labels.
    Matching Networks, by Vinyals and colleagues, use the same form with
    cosine similarity in place of distance. They point out that this form
    includes nearest neighbours as a special case.
    """,
)

DESIGN3_SOURCE = source_file(
    '''
    # design3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    points = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    distances = torch.cdist(points, points)    # tensor([[[0., 5.], [5., 0.]]]), the distance between every pair of points
    nearest = torch.tensor([4.0, 1.0, 3.0]).topk(2, largest=False).indices    # tensor([1, 2])
    votes = F.one_hot(torch.tensor([2, 0, 2]), 4).sum(dim=0)    # tensor([1, 0, 2, 0])


''' + indented(EPISODES_PART) + '''


    # TODO: Label every query with hard k nearest neighbours. memory has shape
    #   (episodes, points, FEATURES), labels has shape (episodes, points), and
    #   queries has shape (episodes, queries, FEATURES). For each query,
    #   compute its distance to every point in memory and keep the k nearest
    #   points. Count how many of them have each of the CLASSES labels. Return
    #   the label with the most votes, as an integer tensor of shape
    #   (episodes, queries). A tie goes to the lowest label among the tied
    #   ones, as argmax does.
    def knn_predict(
        memory: Tensor,
        labels: Tensor,
        queries: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return each query's soft votes, of shape (episodes, queries,
    #   CLASSES). First give every point in memory a weight, the softmax over
    #   the points of -(distance ** 2) / temperature, where distance is from the
    #   query to that point. The vote for a class is the sum of the weights of
    #   the points with that label. Multiplying the weights by the one hot
    #   labels gives every vote at once.
    def soft_votes(
        memory: Tensor,
        labels: Tensor,
        queries: Tensor,
        temperature: Tensor | float,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Project memory and queries with self.project. Call
    #   soft_votes on them with the temperature exp(self.log_temperature).
    #   Return the log of the votes plus 1e-6. The 1e-6 keeps a class with
    #   zero votes from giving minus infinity.
''' + indented(NEIGHBORS_INIT) + '''

        def forward(
            self,
            memory: Tensor,
            labels: Tensor,
            queries: Tensor,
        ) -> Tensor:
            ...
    ''',
)

DESIGN3_SOLUTION = source_file(
    '''
    # design3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    points = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    distances = torch.cdist(points, points)
    nearest = torch.tensor([4.0, 1.0, 3.0]).topk(2, largest=False).indices
    votes = F.one_hot(torch.tensor([2, 0, 2]), 4).sum(dim=0)


''' + indented(EPISODES_PART) + '''


''' + indented(KNN_SOLUTION) + '''


''' + indented(SOFT_VOTES_SOLUTION) + '''


''' + indented(NEIGHBORS_INIT + NEIGHBORS_FORWARD) + '''
    ''',
)


def episode_inputs(
    seed: int,
    count: int,
    queries: int,
    k: int,
) -> tuple[Described, Tensor, Tensor, int]:
    memory, labels, query_points, _ = NEIGHBORS_REFERENCE["episodes"](count, torch.Generator().manual_seed(seed), queries)

    return Described(memory, f"memory of {count} episodes of 20 points"), labels, query_points, k


def check_neighbors_forward(found: type) -> None:
    torch.manual_seed(0)
    model = expect_module("SoftNeighbors(8, 8)", found(8, 8))
    torch.manual_seed(0)
    reference = NEIGHBORS_REFERENCE["SoftNeighbors"](8, 8)

    # A temperature other than 1, so a forward that leaves out exp shows.
    with torch.no_grad():
        model.log_temperature.fill_(0.4)
        reference.log_temperature.fill_(0.4)

    memory, labels, queries, _ = NEIGHBORS_REFERENCE["episodes"](3, torch.Generator().manual_seed(971), 4)

    with torch.no_grad():
        expected = reference(memory, labels, queries)

    expect_tensor(
        call="SoftNeighbors(8, 8) with log_temperature 0.4, on 3 episodes of 4 queries",
        actual=model(memory, labels, queries),
        expected=expected,
    )


def check_neighbors_learn(found: type) -> None:
    torch.manual_seed(0)
    model = train_neighbors(found(8, 8), 300, 1)
    memory, labels, queries, truth = held_episodes()

    with torch.no_grad():
        share = share_right(model(memory, labels, queries).argmax(dim=-1), truth)

    if share < 0.9:
        raise Mismatch(f"After 300 steps of training, SoftNeighbors labels {share:.1%} of new queries right, but it should pass 90%.")


def check_design3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="knn_predict",
        description="labels each query by the vote of its k nearest points",
        inputs=(
            episode_inputs(972, 3, 5, 1),
            episode_inputs(973, 4, 6, 3),
        ),
        reference=NEIGHBORS_REFERENCE["knn_predict"],
        expect=expect_whole_numbers,
    )
    check_on_inputs(
        checker=checker,
        function="soft_votes",
        description="weights every point by a softmax of its squared distance",
        inputs=(
            episode_inputs(974, 2, 3, 0)[:3] + (0.5,),
            episode_inputs(975, 3, 4, 0)[:3] + (torch.tensor(2.0),),
        ),
        reference=NEIGHBORS_REFERENCE["soft_votes"],
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="SoftNeighbors",
        method="forward",
        description="projects the points and returns the log of the soft votes",
        run=check_neighbors_forward,
    )
    checker.test_method(
        class_name="SoftNeighbors",
        method="forward",
        description="learns which features matter",
        run=check_neighbors_learn,
    )


DESIGN3 = Exercise(
    name="design3",
    chapter=DESIGN_METHOD,
    title="Soft steps for nearest neighbours",
    lesson=DESIGN3_LESSON,
    source=DESIGN3_SOURCE,
    solution=DESIGN3_SOLUTION,
    hints={
        "knn_predict": "torch.cdist(queries, memory) gives distances of shape (episodes, queries, points). topk "
        "with largest=False and dim=-1 gives the indices of the nearest points. labels.gather on dim 1 picks their "
        "labels. F.one_hot with CLASSES, then .sum(dim=2), counts the votes. Then .argmax(dim=-1) picks the winner.",
        "soft_votes": "Square torch.cdist(queries, memory), negate it and divide by temperature, then take the "
        "softmax over dim=-1. Multiply those weights by F.one_hot(labels, CLASSES).float() with the @ operator.",
        "SoftNeighbors.forward": "Call soft_votes with self.project(memory), labels, self.project(queries) and "
        "self.log_temperature.exp(), then torch.log of the votes plus 1e-6.",
    },
    introduces=(),
    check=check_design3,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# design4
# ---------------------------------------------------------------------------

DESIGN4_LESSON = prose(
    """
    A new module can train and still not work the way it was designed to.
    Two checks run before training catch most of these problems. Both come
    from earlier chapters.

    The first check is that a block starts as the identity function. A block
    added to a working model should begin by doing nothing, so it cannot
    break what already works. residual3 did this by setting the last layer of
    the branch to zero at the start. `identity_gap` measures the largest
    change a block makes to its input. A gap of 0 means the block starts as
    the identity.

    The second check is that gradients reach every parameter. debug3 looked
    for layers with no gradient after one backward pass. But a layer whose
    weights are zero blocks the gradient to the layers before it on that
    first pass. So `silent_after_one_step` first takes one training step on a
    copy of the model. Then it runs a second backward pass and looks for
    parameters with no gradient.

    The table tests a stack of 48 gated blocks, each computing
    `x + gate * branch(x)`. Each variant trained for 500 steps. The blocks
    have no LayerNorm, and their up and down layers start with Kaiming
    initialization, as in a deep model built without care. So a block that
    does not start as the identity changes its input a lot. The target is
    input 0 when input 1 is above 0, and input 2 otherwise. The error is the
    mean squared error on new inputs. Always answering 0 gives an error of
    1.00.

    | Variant | Identity gap | Silent after one step | Held out error |
    |---|---|---|---|
    | as designed | 0 | 0 | 0.038 to 0.044 |
    | random start | 3.5 to 3.9 | 0 | 2e12 to 2e13 |
    | hard gate, `(score > 0).float()` | 0 | 96 | 0.050 to 0.077 |
    | no residual | 4.7 to 5.5 | 0 | 1.00 |

    Starting as the identity is what lets these 48 blocks train. With a
    random start, each block changes its input by up to 3.9. Those changes
    compound through 48 blocks, and the error never comes down. Bachlechner
    and colleagues' ReZero also starts every residual branch at zero. It
    multiplies each branch by one learned scale that starts at 0, and it
    trained transformers with 120 layers this way. Without the residual
    connection, the stack learned nothing.

    The variant with a hard gate trained, so judging by the loss alone you
    would accept it. But its 96 gate tensors, holding 50,688 numbers, got no
    gradient and never changed. The gates kept their random starting values,
    and the rest of the model learned to work around them. You would credit
    the result to a gate that did nothing.

    Starting as the identity matters again when you add a block to a model
    that already works. Insert one new block in the middle of the trained
    stack. When the new block started at zero, the held out error stayed
    exactly the same. When it started at random, the error rose by a factor
    of 1.6 to 3.8 on every seed.

    The block in your file has two of these faults. Fix it so both checks
    pass.
    """,
)

DESIGN4_SOURCE = source_file(
    '''
    # design4.py

    import copy
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    layer = nn.Linear(2, 2)
    twin = copy.deepcopy(layer)    # a separate copy with its own weights, so training twin leaves layer unchanged
    moved = torch.tensor([0.0, 0.0, 3.0]).any()    # tensor(True), because at least one entry is not 0


    # TODO: Return the largest absolute difference between block(x) and x, over
    #   all entries, as a Python float. 0 means the block is the identity, so
    #   it leaves x unchanged.
    def identity_gap(
        block: nn.Module,
        x: Tensor,
    ) -> float:
        ...


    # TODO: Return the names of the parameters that get no gradient after one
    #   training step, in the order named_parameters() gives them. Work on a
    #   copy made with copy.deepcopy(model), so model itself never changes.
    #   Backpropagate loss_of on the copy, take one step of torch.optim.SGD at
    #   lr, zero the gradients, and backpropagate loss_of on the copy again. A
    #   parameter is silent when its gradient is None or all zeros.
    def silent_after_one_step(
        model: nn.Module,
        loss_of: Callable[[nn.Module], Tensor],
        lr: float,
    ) -> list[str]:
        ...


    # TODO: This block does not start as the identity, and its gate never
    #   learns. Fix __init__ so the block starts as the identity, by setting
    #   down's weight and bias to zero as residual3 did. Fix forward so the
    #   gate is soft, the sigmoid of the gate's score instead of a hard
    #   comparison with 0.
''' + indented(BLOCK_START + BROKEN_FORWARD) + '''
    ''',
)

DESIGN4_SOLUTION = source_file(
    '''
    # design4.py, a solution for comparison

    import copy
    from collections.abc import Callable

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    layer = nn.Linear(2, 2)
    twin = copy.deepcopy(layer)
    moved = torch.tensor([0.0, 0.0, 3.0]).any()


''' + indented(HEALTH_SOLUTION) + '''


''' + indented(BLOCK_START + BLOCK_ZERO + BLOCK_FORWARD) + '''
    ''',
)


class ZeroBranch(nn.Module):
    """x plus a branch whose last layer starts at zero, residual3's zero start."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.up = nn.Linear(size, 2 * size)
        self.down = nn.Linear(2 * size, size)

        with torch.no_grad():
            self.down.weight.zero_()
            self.down.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(F.relu(self.up(x)))


class UnusedLayer(nn.Module):
    """A model whose middle Linear is built and never called."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.middle = nn.Linear(6, 6)
        self.last = nn.Linear(6, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.last(F.relu(self.first(x))).squeeze(-1)


def dead_start() -> nn.Module:
    """A hidden layer whose bias pushes every ReLU below 0, so its gradient exists but is all zeros, step after step."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))

    with torch.no_grad():
        model[0].bias.fill_(-50.0)

    return model


def dies_after_step() -> nn.Module:
    """Hidden ReLUs that are on at the start and all off after one step of SGD at 0.1. Every layer gets a gradient on
    the first pass and only the last bias on the second, so a missing zero_grad shows."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1))

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(0.05)
        model[2].weight.fill_(1.0)
        model[2].bias.fill_(2.0)

    return model


def seeded_described(
    make: Callable[[], nn.Module],
    text: str,
) -> Described:
    torch.manual_seed(0)

    return Described(make(), text)


def regression_loss(
    inputs: int,
    seed: int,
) -> Callable[[nn.Module], Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(16, inputs, generator=generator)
    y = torch.randn(16, generator=generator)

    def loss_of(model: nn.Module) -> Tensor:
        return F.mse_loss(model(x).reshape(-1), y)

    return loss_of


def zero_stack() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), ZeroBranch(8), ZeroBranch(8), nn.Linear(8, 1))


def hard_gate_stack() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), BROKEN_REFERENCE["GatedBlock"](8), nn.Linear(8, 1))


def check_silent(
    found: object,
    make: Callable[[], nn.Module],
    text: str,
    seed: int,
) -> None:
    torch.manual_seed(0)
    model = make()
    before = copy.deepcopy(model.state_dict())
    loss_of = regression_loss(4, seed)
    call = f"silent_after_one_step({text}, its loss on 16 rows, 0.1)"
    actual = found(model, loss_of, 0.1)
    expected = HEALTH_REFERENCE["silent_after_one_step"](model, loss_of, 0.1)

    expect_value(
        call=call,
        actual=actual,
        expected=expected,
    )

    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, before[name]):
            raise Mismatch(f"{call} changed the model's own {name}. Train a copy.deepcopy of it, so the check leaves the model alone.")


def check_block_start(found: type) -> None:
    torch.manual_seed(0)
    block = expect_module("GatedBlock(8)", found(8))

    for name in ("weight", "bias"):
        part = getattr(block.down, name)

        if part.abs().max().item() != 0.0:
            raise Mismatch(f"GatedBlock(8)'s down.{name} should start at zero, but its largest number is {part.abs().max().item():.4g}.")

    x = small_tensor(981, 5, 8)
    gap = HEALTH_REFERENCE["identity_gap"](block, x)

    if gap > 1e-6:
        raise Mismatch(f"GatedBlock(8) changes its input by up to {gap:.4g} at the start, but it should start as the identity.")


def check_block_learns(found: type) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), found(8), found(8), nn.Linear(8, 1))
    silent = HEALTH_REFERENCE["silent_after_one_step"](model, regression_loss(4, 982), 0.1)

    if silent:
        raise Mismatch(f"Two GatedBlocks between Linears leave {', '.join(silent)} with no gradient after one step. A gate made with > passes no gradient.")


def check_block_forward(found: type) -> None:
    torch.manual_seed(0)
    block = expect_module("GatedBlock(8)", found(8))
    torch.manual_seed(0)
    reference = HEALTH_REFERENCE["GatedBlock"](8)

    # A branch that isn't zero, so the gate's shape shows.
    with torch.no_grad():
        for made in (block, reference):
            made.down.weight.copy_(small_tensor(983, 8, 32) / 4)
            made.down.bias.copy_(small_tensor(984, 8) / 4)

    x = small_tensor(985, 5, 8)

    with torch.no_grad():
        expected = reference(x)

    expect_tensor(
        call="GatedBlock(8), with down set to other numbers, on 5 rows",
        actual=block(x),
        expected=expected,
    )


def check_design4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="identity_gap",
        description="gives the largest change a block makes",
        inputs=(
            (seeded_described(lambda: ZeroBranch(4), "a residual block with its last layer at zero"), small_tensor(986, 3, 4)),
            (seeded_described(lambda: nn.Linear(4, 4), "nn.Linear(4, 4)"), small_tensor(987, 3, 4)),
            (seeded_described(lambda: BROKEN_REFERENCE["GatedBlock"](4), "the GatedBlock in your file"), small_tensor(988, 2, 4)),
        ),
        reference=HEALTH_REFERENCE["identity_gap"],
        expect=expect_close,
    )

    for make, text, seed in (
        (UnusedLayer, "a model whose middle Linear is never called", 989),
        (zero_stack, "two residual blocks with their last layer at zero", 990),
        (hard_gate_stack, "the GatedBlock in your file between two Linears", 991),
        (dead_start, "a model whose hidden ReLUs all start below 0", 992),
        (dies_after_step, "a model whose hidden ReLUs all die after one step", 992),
    ):
        checker.test(
            function="silent_after_one_step",
            description="names the parameters that get no gradient after one step",
            run=lambda found, make=make, text=text, seed=seed: check_silent(found, make, text, seed),
        )

    checker.test_method(
        class_name="GatedBlock",
        method="__init__",
        description="starts as the identity",
        run=check_block_start,
    )
    checker.test_method(
        class_name="GatedBlock",
        method="forward",
        description="gives every parameter a gradient",
        run=check_block_learns,
    )
    checker.test_method(
        class_name="GatedBlock",
        method="forward",
        description="gates the branch with a sigmoid",
        run=check_block_forward,
    )


DESIGN4 = Exercise(
    name="design4",
    chapter=DESIGN_METHOD,
    title="Checks before training",
    lesson=DESIGN4_LESSON,
    source=DESIGN4_SOURCE,
    solution=DESIGN4_SOLUTION,
    hints={
        "identity_gap": "Take block(x) - x, its .abs().max(), and .item().",
        "silent_after_one_step": "Work on a copy.deepcopy of the model, with an SGD optimizer over "
        "copied.parameters(). After the second backward, keep each name from copied.named_parameters() whose grad "
        "is None or not grad.any().",
        "GatedBlock.__init__": "Under torch.no_grad(), call .zero_() on self.down.weight and self.down.bias, after "
        "the layers are made.",
        "GatedBlock.forward": "(score > 0).float() passes no gradient. torch.sigmoid of self.gate(h) does.",
    },
    introduces=(),
    check=check_design4,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# design5
# ---------------------------------------------------------------------------

DESIGN5_LESSON = prose(
    """
    The structure of a layer promises certain properties. Suppose you shuffle
    the steps of the input, shift them, rotate its points or make it longer.
    The output should change in a known way, or not at all. The last step of
    the method tests each property directly on the module, as equivariance1
    did.

    `property_gaps` computes equivariance1's gap for every transform you give
    it. The gap is the largest difference between transforming the input
    before the module and transforming the output after it. A gap of 0 means
    the property holds exactly. A rotation made with equivariance1's `rotate`
    is one possible transform.
    `prefix_gap` tests a longer input. If you append steps to the end of the
    input, a causal module's outputs at the earlier steps should not change.

    The table shows the gaps of untrained modules on 4 sequences, each with
    12 steps of 16 numbers. A gap near 1e-7 is only float rounding, so that
    property holds.

    | Module | Shuffle | Shift by 1 | Prefix |
    |---|---|---|---|
    | MLP on each step | 0 | 0 | 0 |
    | convolution, zero padding | 2.4 | 1.3 | 0.91 |
    | convolution, circular padding | 2.4 | 4e-7 | 1.6 |
    | causal convolution | 2.3 | 1.4 | 4e-7 |
    | attention, no positions | 1e-7 | 6e-8 | 0.26 |
    | causal attention, no positions | 0.96 | 0.78 | 1e-7 |
    | attention, learned positions | 0.52 | 0.38 | 0.23 |
    | GRU | 0.70 | 0.31 | 0 |
    | MLP over the whole sequence | 3.0 | 3.4 | can't run |

    Use this table to choose layers. Attention without positions treats its
    steps as a set, so it cannot tell their order. Adding positions or a
    causal mask gives it a sense of order, so a shuffle then changes the
    output. A convolution keeps its shift gap at zero except at its edges,
    and everywhere with circular padding. Causal layers pass the prefix test,
    which generating text one token at a time requires.

    A trained model needs these tests too, at the sequence lengths it will
    meet in use. design1's 2 layer attention model was trained on 8 pairs.
    The table shows the fraction of queries it answered correctly on new
    sequences of each length, as the range over 3 seeds.

    | Positions in training | 4 pairs | 8 | 12 | 16 |
    |---|---|---|---|---|
    | 0 to 23 | 98.2% to 99.8% | 100% | 66.8% to 68.3% | 52.2% to 52.7% |
    | from a random start | 99.5% to 99.8% | 99.9% to 100% | 99.8% to 99.9% | 97.6% to 99.4% |

    The test caught a failure on longer sequences, and the cause is the
    learned positions. A training sequence of 8 pairs fills positions 0 to
    23, so the position embeddings past 23 never trained. The fix starts each
    batch's positions at a random place among all 96, so every position
    embedding trains. This is close to Kiyono and colleagues' SHAPE, which
    shifts each sequence's positions by its own random offset. Test at every
    length the module will meet, including lengths it never trained on.
    """,
)

ROTATE_PART = '''
# equivariance1's rotation by angle of points of shape (..., 2), given.
def rotate(
    points: Tensor,
    angle: float,
) -> Tensor:
    turn = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    return points @ turn.T
'''

DESIGN5_SOURCE = source_file(
    '''
    # design5.py

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    x = torch.tensor([[[1.0], [2.0], [3.0]]])
    reordered = x[:, torch.tensor([2, 0, 1])]     # tensor([[[3.], [1.], [2.]]]), steps in a new order
    rolled = torch.roll(x, 1, dims=1)              # tensor([[[3.], [1.], [2.]]]), the last step wraps to the front
    longer = torch.cat([x, x], dim=1).shape        # torch.Size([1, 6, 1])


''' + indented(GAP_PART) + '''


''' + indented(ROTATE_PART) + '''


    # TODO: Return x with its steps, along dim 1, reordered by order. Step i of
    #   the result is step order[i] of x.
    #   Example: with the x at the top of this file,
    #   shuffled(x, torch.tensor([2, 0, 1])) returns tensor([[[3.], [1.], [2.]]]).
    def shuffled(
        x: Tensor,
        order: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return x with its steps, along dim 1, shifted by `steps` places.
    #   Steps pushed off the end wrap around to the front.
    #   Example: with the x at the top of this file, shifted(x, 1) returns
    #   tensor([[[3.], [1.], [2.]]]).
    def shifted(
        x: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Test whether a longer input changes the earlier outputs of f.
    #   Append extra to x along dim 1 and run f on the result. Compare its
    #   first x.shape[1] steps with f(x). Return the largest absolute
    #   difference over all entries, as a Python float. 0 means the longer
    #   input left the earlier outputs unchanged.
    def prefix_gap(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        extra: Tensor,
    ) -> float:
        ...


    # TODO: Return a dict that maps each name in transforms to the
    #   equivariance_gap of f under that name's transform, on x.
    def property_gaps(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        transforms: dict[str, Callable[[Tensor], Tensor]],
    ) -> dict[str, float]:
        ...
    ''',
)

DESIGN5_SOLUTION = source_file(
    '''
    # design5.py, a solution for comparison

    import math
    from collections.abc import Callable

    import torch
    from torch import Tensor

    x = torch.tensor([[[1.0], [2.0], [3.0]]])
    reordered = x[:, torch.tensor([2, 0, 1])]
    rolled = torch.roll(x, 1, dims=1)
    longer = torch.cat([x, x], dim=1).shape


''' + indented(GAP_PART) + '''


''' + indented(ROTATE_PART) + '''


''' + indented(PROPERTY_SOLUTION) + '''
    ''',
)

PROPERTY_REFERENCE.update(namespace_of(ROTATE_PART))


def expect_gaps(
    call: str,
    actual: object,
    expected: dict[str, float],
) -> None:
    reject_missing_return(
        call=call,
        actual=actual,
    )

    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise Mismatch(f"{call} is {value_text(actual)}, but it should be a dict with the names {sorted(expected)}.")

    for name, wanted in expected.items():
        expect_close(
            call=f"{call}[{name!r}]",
            actual=actual[name],
            expected=wanted,
            tolerance=1e-5,
        )


def mixing(x: Tensor) -> Tensor:
    """Each step's cumulative sum, a causal mixing whose gaps aren't all 0."""
    return x.cumsum(dim=1) * torch.linspace(1.0, 2.0, x.shape[-1])


def check_design5(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="shuffled",
        description="reorders the steps",
        inputs=(
            (small_tensor(991, 2, 4, 3), torch.tensor([2, 0, 3, 1])),
            (small_tensor(992, 3, 5, 2), torch.tensor([4, 3, 2, 1, 0])),
        ),
        reference=PROPERTY_REFERENCE["shuffled"],
    )
    check_on_inputs(
        checker=checker,
        function="shifted",
        description="rolls the steps, wrapping around",
        inputs=(
            (small_tensor(993, 2, 4, 3), 1),
            (small_tensor(994, 3, 5, 2), -2),
        ),
        reference=PROPERTY_REFERENCE["shifted"],
    )
    check_on_inputs(
        checker=checker,
        function="prefix_gap",
        description="compares the first steps of a longer input's output",
        inputs=(
            (Described(lambda t: t.cumsum(dim=1), "a running sum along the steps"), small_tensor(995, 2, 4, 3), small_tensor(996, 2, 2, 3)),
            (Described(lambda t: t.flip(1), "the steps in reverse"), small_tensor(997, 2, 4, 3), small_tensor(998, 2, 3, 3)),
            (Described(lambda t: t + t.mean(dim=1, keepdim=True), "each step plus the mean of all steps"), small_tensor(999, 1, 3, 2), small_tensor(1000, 1, 2, 2)),
        ),
        reference=PROPERTY_REFERENCE["prefix_gap"],
        expect=expect_close,
    )
    order = torch.tensor([3, 1, 0, 2])
    points = small_tensor(1001, 2, 4, 2) / 2
    check_on_inputs(
        checker=checker,
        function="property_gaps",
        description="gives the gap under every transform",
        inputs=(
            (
                Described(mixing, "a running sum along the steps, each feature scaled"),
                small_tensor(1002, 2, 4, 3),
                Described(
                    {"shuffle": lambda t: t[:, order], "shift": lambda t: torch.roll(t, 1, dims=1)},
                    "a shuffle and a shift by 1",
                ),
            ),
            (
                Described(lambda p: p * p.norm(dim=-1, keepdim=True), "each point scaled by its length"),
                points,
                Described(
                    {"rotate": lambda p: PROPERTY_REFERENCE["rotate"](p, 0.7), "shuffle": lambda p: p[:, order]},
                    "a turn of 0.7 and a shuffle",
                ),
            ),
        ),
        reference=PROPERTY_REFERENCE["property_gaps"],
        expect=expect_gaps,
    )


DESIGN5 = Exercise(
    name="design5",
    chapter=DESIGN_METHOD,
    title="Each property tested directly",
    lesson=DESIGN5_LESSON,
    source=DESIGN5_SOURCE,
    solution=DESIGN5_SOLUTION,
    hints={
        "shuffled": "Index dim 1 with order and keep all of dim 0, as reordered in the file does.",
        "shifted": "Use torch.roll with dims=1.",
        "prefix_gap": "Join x and extra with torch.cat on dim=1 and run f on the result. Keep the first x.shape[1] "
        "steps of the output and compare them with f(x).",
        "property_gaps": "Use a dict comprehension over transforms.items(), where each value is "
        "equivariance_gap(f, transform, x).",
    },
    introduces=(),
    check=check_design5,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------

RECALL_KINDS = ("GRU", "attention, 1 layer", "attention, 2 layers")


def measure_recall(seeds: range = range(3)) -> dict[tuple[str, int], list[float]]:
    """design1's table: held out accuracy after 1,500 steps on fresh sequences, per model and number of pairs,
    with the seconds each run took on the CPU."""
    results: dict[tuple[str, int], list[float]] = {}

    with pinned_threads():
        for kind in RECALL_KINDS:
            for pairs in (4, 8, 16):
                for seed in seeds:
                    start = time.perf_counter()
                    _, held = final_accuracies(kind, pairs, seed, fresh_batches(pairs, seed + 1))
                    results.setdefault((kind, pairs), []).append(held)
                    results.setdefault((kind, pairs, "seconds"), []).append(time.perf_counter() - start)

        tokens, targets = held_out(8)
        exact = RECALL_REFERENCE["exact_recall"](tokens, 8)
        results[("exact", 8)] = [(exact == targets[:, 16:]).float().mean().item()]

        for pairs in (4, 8, 16):
            results[("most common value", pairs)] = [most_common_value_accuracy(pairs)]

    return results


def most_common_value_accuracy(pairs: int) -> float:
    """The shortcut that needs no recall: answer the value a sequence holds most often, a fair pick among ties."""
    tokens, targets = held_out(pairs)
    shares = []

    for sequence, answers in zip(tokens[:, 1 : 2 * pairs : 2].tolist(), targets[:, 2 * pairs :].tolist()):
        most = max(sequence.count(value) for value in sequence)
        tied = {value for value in sequence if sequence.count(value) == most}
        shares.extend(1 / len(tied) if answer in tied else 0.0 for answer in answers)

    return sum(shares) / len(shares)


def measure_diagnoses(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """design2's table: the training and held out accuracy of each run at 8 pairs, and the baseline."""
    runs = {
        "GRU, fresh sequences": ("GRU", None),
        "MLP over the sequence, 64 fixed": ("MLP", 64),
        "attention, 2 layers, 64 fixed": ("attention, 2 layers", 64),
        "attention, 2 layers, 256 fixed": ("attention, 2 layers", 256),
        "attention, 2 layers, 1,024 fixed": ("attention, 2 layers", 1024),
        "attention, 2 layers, 4,096 fixed": ("attention, 2 layers", 4096),
        "attention, 2 layers, fresh sequences": ("attention, 2 layers", None),
    }
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for name, (kind, count) in runs.items():
            for seed in seeds:
                batches = fresh_batches(8, seed + 1) if count is None else fixed_batches(8, count, seed + 1)
                results.setdefault(name, []).append(final_accuracies(kind, 8, seed, batches))

        train_tokens, train_targets = RECALL_REFERENCE["recall_batch"](4096, 8, torch.Generator().manual_seed(7))
        _, held_targets = held_out(8)
        baseline = DIAGNOSE_REFERENCE["baseline_accuracy"](train_targets, held_targets)
        results["baseline"] = [(baseline, baseline)]

    return results


def measure_neighbors(seeds: range = range(3)) -> dict[str, list[float]]:
    """design3's table, on 500 new episodes."""
    memory, labels, queries, truth = held_episodes()
    knn = NEIGHBORS_REFERENCE["knn_predict"]
    results: dict[str, list[float]] = {
        "hard, k 1": [share_right(knn(memory, labels, queries, 1), truth)],
        "hard, k 5": [share_right(knn(memory, labels, queries, 5), truth)],
        "hard, k 5, the 2 features that matter": [share_right(knn(memory[..., :2], labels, queries[..., :2], 5), truth)],
    }

    with pinned_threads():
        for seed in seeds:
            torch.manual_seed(seed)
            model = NEIGHBORS_REFERENCE["SoftNeighbors"](8, 8)

            with torch.no_grad():
                results.setdefault("soft, untrained", []).append(share_right(model(memory, labels, queries).argmax(dim=-1), truth))

            train_neighbors(model, 300, seed + 1)

            with torch.no_grad():
                results.setdefault("soft, trained", []).append(share_right(model(memory, labels, queries).argmax(dim=-1), truth))
                projected = knn(model.project(memory), labels, model.project(queries), 5)
                results.setdefault("hard, k 5, the learned projection", []).append(share_right(projected, truth))
                columns = model.project.weight.norm(dim=0)
                results.setdefault("signal columns", []).append(columns[:2].min().item())
                results.setdefault("noise columns", []).append(columns[2:].max().item())
                results.setdefault("temperature", []).append(model.log_temperature.exp().item())

            torch.manual_seed(seed)
            mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4))
            optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
            generator = torch.Generator().manual_seed(seed + 1)

            for _ in range(300):
                _, _, train_queries, train_truth = NEIGHBORS_REFERENCE["episodes"](32, generator)
                loss = F.cross_entropy(mlp(train_queries).flatten(0, 1), train_truth.flatten())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                results.setdefault("MLP on the query alone", []).append(share_right(mlp(queries).argmax(dim=-1), truth))

    return results


HEALTH_VARIANTS = {
    "as designed": {"zero_start": True, "soft_gate": True, "residual": True},
    "random start": {"zero_start": False, "soft_gate": True, "residual": True},
    "hard gate": {"zero_start": True, "soft_gate": False, "residual": True},
    "no residual": {"zero_start": False, "soft_gate": True, "residual": False},
}

# Deep enough that a start which isn't the identity compounds, as ReZero's deep stacks did.
HEALTH_DEPTH = 48
HEALTH_STEPS = 500


def measure_health(seeds: range = range(3)) -> dict[str, list[tuple[float, int, float, float]]]:
    """design4's table: for each variant of 48 loud blocks, the largest identity gap at the start, how many
    parameters stay silent after one step, how many numbers those hold, and the held out error after 500 steps of
    Adam at 0.001."""
    held_x, held_y = switch_data(4000, torch.Generator().manual_seed(99))
    identity_gap = HEALTH_REFERENCE["identity_gap"]
    silent_after_one_step = HEALTH_REFERENCE["silent_after_one_step"]
    results: dict[str, list[tuple[float, int, float, float]]] = {"always 0": [(0.0, 0, 0.0, (held_y**2).mean().item())]}

    with pinned_threads():
        for name, kinds in HEALTH_VARIANTS.items():
            for seed in seeds:
                torch.manual_seed(seed)
                model = BlockStack(lambda: VariantBlock(BLOCK_SIZE, **kinds, loud=True), HEALTH_DEPTH)

                with torch.no_grad():
                    h = model.inp(held_x[:256])

                gap = max(identity_gap(block, h) for block in model.blocks)
                x, y = switch_data(128, torch.Generator().manual_seed(5))
                silent = silent_after_one_step(model, lambda copied: F.mse_loss(copied(x), y), 0.01)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                generator = torch.Generator().manual_seed(seed + 1)

                for _ in range(HEALTH_STEPS):
                    x, y = switch_data(128, generator)
                    loss = F.mse_loss(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    error = F.mse_loss(model(held_x), held_y).item()

                numbers = sum(parameter.numel() for parameter_name, parameter in model.named_parameters() if parameter_name in silent)
                results.setdefault(name, []).append((gap, len(silent), numbers, error))

    return results


def measure_insertion(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """Why the identity start matters: a trained stack's held out error before and after one new loud block goes
    in at the middle, started at zero or at random."""
    held_x, held_y = switch_data(4000, torch.Generator().manual_seed(99))
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for seed in seeds:
            torch.manual_seed(seed)
            model = BlockStack(
                lambda: VariantBlock(BLOCK_SIZE, zero_start=True, soft_gate=True, residual=True, loud=True),
                HEALTH_DEPTH,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            generator = torch.Generator().manual_seed(seed + 1)

            for _ in range(HEALTH_STEPS):
                x, y = switch_data(128, generator)
                loss = F.mse_loss(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                before = F.mse_loss(model(held_x), held_y).item()

                for name, zero_start in (("new block at zero", True), ("new block at random", False)):
                    torch.manual_seed(seed + 10)
                    grown = copy.deepcopy(model)
                    grown.blocks.insert(
                        HEALTH_DEPTH // 2,
                        VariantBlock(BLOCK_SIZE, zero_start=zero_start, soft_gate=True, residual=True, loud=True),
                    )
                    results.setdefault(name, []).append((before, F.mse_loss(grown(held_x), held_y).item()))

    return results


def measure_properties() -> dict[str, tuple[float, float, float | None]]:
    """design5's table: each module's shuffle gap, shift gap and prefix gap, on 4 sequences of 12 steps,
    and None for a module that can't run on a longer input."""
    x = torch.randn(4, TABLE_STEPS, TABLE_SIZE, generator=torch.Generator().manual_seed(1))
    extra = torch.randn(4, 4, TABLE_SIZE, generator=torch.Generator().manual_seed(2))
    order = torch.randperm(TABLE_STEPS, generator=torch.Generator().manual_seed(3))
    gaps = PROPERTY_REFERENCE["property_gaps"]
    prefix_gap = PROPERTY_REFERENCE["prefix_gap"]
    transforms = {
        "shuffle": lambda t: PROPERTY_REFERENCE["shuffled"](t, order),
        "shift": lambda t: PROPERTY_REFERENCE["shifted"](t, 1),
    }
    results: dict[str, tuple[float, float, float | None]] = {}

    for name, module in table_modules().items():
        found = gaps(module, x, transforms)

        try:
            prefix: float | None = prefix_gap(module, x, extra)
        except RuntimeError:
            prefix = None

        results[name] = (found["shuffle"], found["shift"], prefix)

    return results


class RandomPositionsRecaller(RECALL_REFERENCE["AttentionRecaller"]):
    """In training, each batch's positions start at a random place among the 96, so every position trains. In use
    it reads positions 0 onwards, as before."""

    def __init__(
        self,
        layers: int,
        seed: int,
    ) -> None:
        super().__init__(layers)
        self.picker = torch.Generator().manual_seed(seed)

    def forward(self, tokens: Tensor) -> Tensor:
        steps = tokens.shape[1]

        start = 0

        if self.training:
            start = int(torch.randint(0, self.positions.num_embeddings - steps + 1, (1,), generator=self.picker))

        places = torch.arange(steps) + start

        x = self.ids(tokens) + self.positions(places)

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))

        return self.head(self.norm(x))


def measure_lengthening(seeds: range = range(3)) -> dict[tuple[str, int], list[float]]:
    """design5's second table: the 2 layer attention trained on 8 pairs, with learned positions as they were and with
    each training batch's positions starting at a random place, scored on new sequences of each length."""
    results: dict[tuple[str, int], list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            plain = recall_model("attention, 2 layers", 8, seed)
            torch.manual_seed(seed)
            fixed = RandomPositionsRecaller(2, 500 + seed)

            for name, model in (("positions as trained", plain), ("random positions in training", fixed)):
                DIAGNOSE_REFERENCE["track"](model, fresh_batches(8, seed + 1), held_out(8), 1500, 1500)
                model.eval()

                with torch.no_grad():
                    for pairs in (4, 8, 12, 16):
                        tokens, targets = held_out(pairs)
                        results.setdefault((name, pairs), []).append(RECALL_REFERENCE["query_accuracy"](model(tokens), targets))

    return results
