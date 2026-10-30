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

        A new problem has no paper to copy. This chapter is a method for one.
        Isolate the skill in a toy task, beat a baseline and read the curves to
        see what failed, build the module from the exact algorithm, check its
        gradients before training, and test the property it should have
        directly. Each step is measured on real runs.

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


# Layers of attention alone, with residuals and learned positions, as induction1's model.
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


# An MLP over the whole flattened sequence, so it reads every place at once but only one length.
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


# Each episode draws 4 new class centres on the first 2 features. The other 6 features are noise, and
# larger than the gaps between classes. The labels 0 to 3 mean something new in every episode.
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
    """design4's block with each fix on or off, for the lesson's table."""

    def __init__(
        self,
        size: int,
        zero_start: bool,
        soft_gate: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.gate = nn.Linear(size, size)
        self.up = nn.Linear(size, 4 * size)
        self.down = nn.Linear(4 * size, size)
        self.soft_gate = soft_gate
        self.residual = residual

        if zero_start:
            with torch.no_grad():
                self.down.weight.zero_()
                self.down.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
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
# equivariance1's gap, given.
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
    A new problem rarely says which skill it needs. The method starts by
    isolating one. A **toy task** asks for that skill alone. An exact program
    can answer it, a knob sets how hard it is, and fresh examples never run
    out.

    Arora and colleagues built Zoology this way. On real text, 82% of the gap
    between attention and convolution models with gates came from one skill,
    recalling something said earlier. At recalling, a 70 million parameter
    attention model beat a 1.4 billion parameter convolution model.

    Their task, multi query associative recall, reads pairs, then asks for
    several of them back.

    ```text
    A 4 B 3 C 6 F 1 E 2   then   A ? C ? F ? E ? B ?   answers 4 6 1 2 3
    ```

    Here keys are ids 0 to 15 and values 16 to 31. A sequence of `pairs` pairs
    lays them out key, value, key, value. The same keys follow, in a new order,
    as queries. Only the query places are scored. Every other place gets the
    target -100, which `F.cross_entropy` skips, as sft1's targets did.

    `pairs` is the knob. Trained 1,500 steps on fresh sequences, three models
    got this share of queries right on new ones, the range over 3 seeds.

    | Model | 4 pairs | 8 pairs | 16 pairs |
    |---|---|---|---|
    | GRU | 34.1% to 34.9% | 25.8% to 26.0% | 19.3% to 19.5% |
    | attention, 1 layer | 34.0% to 34.1% | 25.9% to 26.1% | 19.4% to 19.6% |
    | attention, 2 layers | 100% | 100% | 99.99% to 100% |

    Your `exact_recall` gets 100%, which shows the task is sound. A guess among
    the values gets 1 in 16. The GRU and one attention layer land within a
    point of each other, well above a guess.

    Answering the value a sequence holds most often scores 34.15%, 25.95% and
    19.525%. So both learned that shortcut, and neither recalls. Two layers can
    shift, then match, as induction1 showed, and they recall every pair.

    Each run took under a minute on a laptop CPU. The task found the skill one
    layer lacks, which on real text would take far longer to see.
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
    #   are drawn for you, each of shape (count, pairs). Draw each sequence's
    #   query order with torch.randperm(pairs, generator=generator). Lay the
    #   pairs out key, value, key, value, as laid_out shows, then the keys again
    #   in that order. targets is -100 everywhere but those query places, which
    #   hold each query's value. Return tokens and targets, whole numbers.
    def recall_batch(
        count: int,
        pairs: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        keys = torch.stack([torch.randperm(KEYS, generator=generator)[:pairs] for _ in range(count)])
        values = torch.randint(KEYS, VOCABULARY, (count, pairs), generator=generator)
        ...


    # TODO: Return the exact answer to every query, from a plain dict per
    #   sequence of each key to its value, as exact_lookup in derive2 used a
    #   loop. tokens is laid out as recall_batch makes it. Return whole numbers
    #   of shape (count, pairs).
    def exact_recall(
        tokens: Tensor,
        pairs: int,
    ) -> Tensor:
        ...


    # TODO: Return the share of query places, where targets isn't -100, whose
    #   highest score is the target, as a plain number. scores has shape
    #   (count, steps, VOCABULARY) and targets (count, steps).
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

    if (targets[:, : 2 * pairs] != -100).any():
        raise Mismatch(f"{call}'s targets should be -100 at the {2 * pairs} places of the pairs, so the loss skips them.")

    answers = RECALL_REFERENCE["exact_recall"](tokens, pairs)

    if not torch.equal(targets[:, 2 * pairs :], answers):
        raise Mismatch(f"{call}'s targets at the query places should be each query's value, the one stored with its key.")

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
            description="lays out pairs then queries, and targets only at the queries",
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
        description="scores the query places only",
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
        "recall_batch": "order stacks one torch.randperm(pairs, generator=generator) per sequence. rows is "
        "torch.arange(count).unsqueeze(1), so keys[rows, order] reorders each row. torch.cat the laid out pairs "
        "with it on dim=1. Start targets from torch.full_like(tokens, -100) and fill its last pairs places.",
        "exact_recall": "For each row of tokens.tolist(), build a dict from sequence[2 * i] to sequence[2 * i + 1]. "
        "Then look up every token after the first 2 * pairs. torch.tensor of the lists gives whole numbers.",
        "query_accuracy": "Keep the places where targets != -100. Compare scores.argmax(dim=-1) with targets "
        "there, then take the mean as floats and .item().",
    },
    introduces=("toy task",),
    check=check_design1,
)


# ---------------------------------------------------------------------------
# design2
# ---------------------------------------------------------------------------

DESIGN2_LESSON = prose(
    """
    A score of 60% means little until you know what 60% beats. Karpathy's
    recipe for training networks sets up "dumb baselines" early, and the
    simplest ignores the input. Here that is always answering the value most
    common in training. It gets 6.3% of queries right, about 1 in 16.

    Then read two curves, accuracy on the batches the model trains on and on
    held out sequences. The two ways to fail look different there.

    - Failed to fit: training accuracy itself stays low. The model can't express the skill, or its training is broken.
    - Failed to generalize: training accuracy is high and held out accuracy isn't. It learned its examples, not the skill.

    Here is each model at 8 pairs after 1,500 steps, the range over 3 seeds.
    A fixed set is all the sequences it ever trains on.

    | Model and data | Training | Held out | Diagnosis |
    |---|---|---|---|
    | GRU, fresh sequences | 24.2% to 27.1% | 25.8% to 26.0% | failed to fit |
    | MLP over the sequence, 64 fixed | 100% | 7.3% to 7.6% | failed to generalize |
    | attention, 2 layers, 64 fixed | 100% | 14.9% to 15.0% | failed to generalize |
    | attention, 2 layers, 256 fixed | 99.8% to 100% | 21.6% to 99.5% | depends on the seed |
    | attention, 2 layers, 1,024 fixed | 100% | 100% | works |
    | attention, 2 layers, fresh sequences | 100% | 100% | works |

    The diagnosis says what to change. A model that failed to fit needs a layer
    that can express the skill, or a fix to its training, which design4 checks.
    More data won't help it. The GRU's fixed state can't hold 8 pairs, as
    attention6 found.

    A model that failed to generalize needs more data, or a layer whose shape
    matches the task. With 256 sequences, two seeds found the skill and one
    learned its examples. From 1,024 on, every seed recalled every pair, at
    4,096 too. A toy task makes fresh data forever, one more reason to start
    with one.

    Karpathy's order is fit first, then generalize. Get a model that can fit
    the training set, then give up some training accuracy for held out
    accuracy.
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
    common = counts.argmax()    # tensor(20), the id seen most
    every = [step for step in range(1, 7) if step % 3 == 0]    # [3, 6]


    # design1's task and accuracy, and the three models the lessons compare, given.
''' + indented(DESIGN2_GIVEN) + '''


    # TODO: Return the share of held_targets' query places, the ones that aren't
    #   -100, equal to the value most common among train_targets' query places,
    #   as a plain number. This baseline never reads the input.
    def baseline_accuracy(
        train_targets: Tensor,
        held_targets: Tensor,
    ) -> float:
        ...


    # TODO: Train model with torch.optim.AdamW at lr=1e-3 for steps steps. Each
    #   step takes a batch from next_batch() and the cross entropy over every
    #   place. Every every steps, counting from 1, add (step, query_accuracy of
    #   the scores it just trained on, query_accuracy on held) to the curve, the
    #   held out one inside torch.no_grad(). Return the curve.
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

    return RECALL_REFERENCE["AttentionRecaller"](2, 16)


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
            raise Mismatch(f"{call} holds {value_text(point)}, but each point should be a tuple of 3, step and two accuracies.")

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
        "baseline_accuracy": "Count the training targets that aren't -100 with torch.bincount and take .argmax(). "
        "Then compare it with the held targets that aren't -100, and take the mean as floats.",
        "track": "This is gpt3's loop with a curve. Flatten the scores to (places, VOCABULARY) and the targets to "
        "places for F.cross_entropy, which skips the -100s. When step % every == 0, score held[0] under "
        "torch.no_grad() and append the three numbers.",
        "diagnose": "Check the fit first, with train_accuracy < fit_bar. Then check the gap, train_accuracy - "
        "held_accuracy > gap_bar.",
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
    38_soft_design made two small algorithms soft, with one or two hard steps
    each. Now take k nearest neighbours, which labels a query by the labelled
    points closest to it.

    1. Measure the distance from the query to every point in memory.
    2. Keep the k nearest, a hard pick with `topk`.
    3. Count each class's votes among them.
    4. Answer the class with the most votes, a hard `argmax`.

    Steps 2 and 4 pass no gradient. The soft version weights every point in
    memory by a softmax of minus its squared distance over a temperature, as
    derive2 weighted its keys. The votes become those weights summed over one
    hot labels, a chance per class.

    The task gives each episode 4 new classes and 5 labelled points of each.
    The first 2 of 8 features carry the class, and the other 6 are larger
    noise. The labels mean something new in each episode, so no fixed
    classifier can learn them.

    The soft steps make the distance itself trainable. A Linear projects every
    point before the distances, and training learns it with the temperature.
    Here are 500 new episodes, the range over 3 seeds.

    | Model | Right |
    |---|---|
    | hard, k 1, all 8 features | 57.7% |
    | hard, k 5, all 8 features | 53.9% |
    | an MLP on the query alone | 24.8% to 25.7% |
    | soft, untrained | 43.2% to 49.0% |
    | soft, trained 300 steps | 95.7% to 95.8% |
    | hard, k 5, the trained projection | 95.8% to 95.9% |
    | hard, k 5, only the 2 features that matter | 96.0% |

    The MLP sits at a guess, 1 in 4, since the labels change every episode.
    The trained projection gives the 2 class features columns of length 1.8
    or more, and the 6 noise features 0.09 or less. It found which features
    matter with no one telling it. The hard algorithm, given that projection,
    does as well, so the soft steps also taught the exact one.

    This model is attention over a memory of labelled points, with one hot
    labels as the values. Matching Networks, by Vinyals and colleagues, use the
    same form with cosine similarity. They note it covers both nearest
    neighbours and kernel density estimates.
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
    distances = torch.cdist(points, points)    # tensor([[[0., 5.], [5., 0.]]]), every pair's distance
    nearest = torch.tensor([4.0, 1.0, 3.0]).topk(2, largest=False).indices    # tensor([1, 2])
    votes = F.one_hot(torch.tensor([2, 0, 2]), 4).sum(dim=0)    # tensor([1, 0, 2, 0])


''' + indented(EPISODES_PART) + '''


    # TODO: Label every query by its k nearest points in memory, the hard way.
    #   memory has shape (episodes, points, FEATURES), labels (episodes, points)
    #   and queries (episodes, queries, FEATURES). Find each query's distances
    #   to memory, keep the k nearest, count their labels' votes over CLASSES,
    #   and return the class with the most votes, of shape (episodes, queries).
    def knn_predict(
        memory: Tensor,
        labels: Tensor,
        queries: Tensor,
        k: int,
    ) -> Tensor:
        ...


    # TODO: Return each query's soft votes, of shape (episodes, queries,
    #   CLASSES). Weight every point in memory by a softmax over the points of
    #   minus its squared distance divided by temperature. Then sum those
    #   weights over the one hot labels.
    def soft_votes(
        memory: Tensor,
        labels: Tensor,
        queries: Tensor,
        temperature: Tensor | float,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Project memory and queries with self.project, take
    #   soft_votes at the temperature exp(self.log_temperature), and return the
    #   log of the votes with 1e-6 added, so a class with no weight doesn't
    #   give minus infinity.
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
        description="projects, takes the soft votes and their log",
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
        "knn_predict": "torch.cdist(queries, memory) gives (episodes, queries, points). topk with largest=False and "
        "dim=-1 gives the nearest places. labels.gather on dim 1 picks their labels, F.one_hot with CLASSES and "
        ".sum(dim=2) counts the votes, and .argmax(dim=-1) answers.",
        "soft_votes": "Square torch.cdist(queries, memory), negate it and divide by temperature, then softmax over "
        "dim=-1. Multiply those weights by F.one_hot(labels, CLASSES).float() with @.",
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
    A new module can train and still not work as designed. Two checks before
    training catch most of it, and both come from earlier chapters.

    The first is the identity at the start. A block added to a working model
    should begin as "do nothing", so it can't break what works. residual3's
    blocks did this with their last layer at zero. `identity_gap` measures the
    largest change a block makes to its input.

    The second is gradients that reach every parameter. debug3 looked for
    layers with no gradient after one backward pass. A layer at zero blocks
    the gradient to the layers before it on that first pass, though. So
    `silent_after_one_step` takes one step on a copy of the model, then looks.

    Here are 12 gated blocks, `x + gate * branch(x)`, each variant trained
    1,500 steps. The task suits a gate. It answers input 0 when input 1 is
    above 0, and input 2 when it isn't. The error is the mean squared error on
    new inputs, and always answering 0 gets 1.00.

    | Variant | Identity gap | Silent after one step | Held out error |
    |---|---|---|---|
    | as designed | 0 | 0 | 0.016 to 0.032 |
    | random start | 0.66 to 0.79 | 0 | 0.016 to 0.023 |
    | hard gate, `(score > 0).float()` | 0 | 24 | 0.015 to 0.026 |
    | no residual | 2.6 to 2.8 | 0 | 0.023 to 0.050 |

    Every variant trained, so the loss alone would pass all four. The checks
    show which ones aren't the module that was designed. The hard gate's 24
    gate parameters never moved, so its gates stayed as drawn and the rest of
    the model worked around them. Credit for the result would go to a gate
    that did nothing. Two variants change their input before learning
    anything at all.

    Why the identity start matters shows when a block goes into a model that
    already works. Insert one new block in the middle of the trained stack.
    Started at zero, the held out error stays exactly where it was. Started at
    random, it doubled on one seed, went up 6 times on another, and barely
    moved on the third.

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
    twin = copy.deepcopy(layer)    # its own weights, so training twin leaves layer alone
    moved = torch.tensor([0.0, 0.0, 3.0]).any()    # tensor(True), some entry isn't 0


    # TODO: Return the largest change block makes to x, entry by entry, as a
    #   plain number. 0 means the block starts as "do nothing".
    def identity_gap(
        block: nn.Module,
        x: Tensor,
    ) -> float:
        ...


    # TODO: Return the names of model's parameters that get no gradient after
    #   one step, in order. Work on copy.deepcopy(model), so model itself never
    #   changes. Backpropagate loss_of(copied), take one step of
    #   torch.optim.SGD at lr, zero the gradients, and backpropagate again. A
    #   parameter is silent when its gradient is None or all zeros.
    def silent_after_one_step(
        model: nn.Module,
        loss_of: Callable[[nn.Module], Tensor],
        lr: float,
    ) -> list[str]:
        ...


    # TODO: This block starts far from "do nothing", and its gate never learns.
    #   Fix __init__ so the block starts as the identity, with down's weight and
    #   bias at zero as residual3 did. Fix forward so the gate is soft, a
    #   sigmoid of the score.
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
    """residual3's block, x plus a branch whose last layer starts at zero."""

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
        raise Mismatch(f"Two GatedBlocks between Linears leave {', '.join(silent)} with no gradient after one step. A gate made with > passes none.")


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
        "silent_after_one_step": "copied = copy.deepcopy(model), and an SGD over copied.parameters(). After the "
        "second backward, keep each name from copied.named_parameters() whose grad is None or not grad.any().",
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
    A layer's structure promises properties. Shuffling the steps, shifting
    them, turning points or making the input longer should change the output
    in a known way, or not at all. The last move of the method tests each
    property directly on the module, as equivariance1 did.

    `property_gaps` takes equivariance1's gap for every change you name, a
    rotation from its `rotate` among them. `prefix_gap` tests a longer input.
    Append steps, and a causal module's earlier outputs shouldn't move.

    Here are untrained modules on 4 sequences of 12 steps of 16 numbers. A gap
    near 1e-7 is float rounding, so that property holds.

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

    Choose layers by this table. Attention without positions treats its
    steps as a set, so it can't tell their order. Positions or a causal mask
    add order and break the shuffle. A convolution keeps shifts except at its
    edges, and fully with circular padding. Causal layers keep the prefix,
    which writing text one token at a time needs.

    A trained model needs the test too, at the lengths it will meet. design1's
    2 layer attention, trained on 8 pairs, got this share of queries right on
    new sequences, the range over 3 seeds.

    | Pairs | 4 | 8 | 12 | 16 |
    |---|---|---|---|---|
    | Right | 98.2% to 99.8% | 100% | 66.8% to 68.3% | 52.2% to 52.7% |

    Its learned positions past the 24 places of training never trained, so it
    breaks past them. Test at the lengths the module will meet, not only the
    ones it learned.
    """,
)

ROTATE_PART = '''
# equivariance1's turn of points of shape (..., 2), given.
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


    # TODO: Return x with its steps, dim 1, put in the order order gives.
    def shuffled(
        x: Tensor,
        order: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return x with its steps, dim 1, rolled by steps places, the ones
    #   pushed off the end wrapping to the front.
    def shifted(
        x: Tensor,
        steps: int,
    ) -> Tensor:
        ...


    # TODO: Return the largest gap, entry by entry, between f of x and the first
    #   steps of f of x with extra appended along dim 1, as a plain number. 0
    #   means a longer input left the earlier outputs alone.
    def prefix_gap(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        extra: Tensor,
    ) -> float:
        ...


    # TODO: Return a dict from each name in transforms to equivariance_gap of
    #   f under that transform, on x.
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
        "shifted": "torch.roll with dims=1.",
        "prefix_gap": "torch.cat x and extra on dim=1 and run f on it. Keep its first x.shape[1] steps and compare "
        "them with f(x).",
        "property_gaps": "A dict comprehension over transforms.items(), each value equivariance_gap(f, transform, x).",
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


def measure_health(seeds: range = range(3)) -> dict[str, list[tuple[float, int, float]]]:
    """design4's table: for each variant, the largest identity gap of its 12 blocks at the start, how many
    parameters stay silent after one step, and the held out error after 1,500 steps of Adam at 0.001."""
    held_x, held_y = switch_data(4000, torch.Generator().manual_seed(99))
    identity_gap = HEALTH_REFERENCE["identity_gap"]
    silent_after_one_step = HEALTH_REFERENCE["silent_after_one_step"]
    results: dict[str, list[tuple[float, int, float]]] = {"always 0": [(0.0, 0, (held_y**2).mean().item())]}

    with pinned_threads():
        for name, kinds in HEALTH_VARIANTS.items():
            for seed in seeds:
                torch.manual_seed(seed)
                model = BlockStack(lambda: VariantBlock(BLOCK_SIZE, **kinds))

                with torch.no_grad():
                    h = model.inp(held_x[:256])

                gap = max(identity_gap(block, h) for block in model.blocks)
                x, y = switch_data(128, torch.Generator().manual_seed(5))
                silent = silent_after_one_step(model, lambda copied: F.mse_loss(copied(x), y), 0.01)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                generator = torch.Generator().manual_seed(seed + 1)

                for _ in range(1500):
                    x, y = switch_data(128, generator)
                    loss = F.mse_loss(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    error = F.mse_loss(model(held_x), held_y).item()

                results.setdefault(name, []).append((gap, len(silent), error))

    return results


def measure_insertion(seeds: range = range(3)) -> dict[str, list[tuple[float, float]]]:
    """Why the identity start matters: a trained stack's held out error before and after one new block goes
    in at the middle, started at zero or at random."""
    held_x, held_y = switch_data(4000, torch.Generator().manual_seed(99))
    results: dict[str, list[tuple[float, float]]] = {}

    with pinned_threads():
        for seed in seeds:
            torch.manual_seed(seed)
            model = BlockStack(lambda: VariantBlock(BLOCK_SIZE, zero_start=True, soft_gate=True, residual=True))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            generator = torch.Generator().manual_seed(seed + 1)

            for _ in range(1500):
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
                    grown.blocks.insert(6, VariantBlock(BLOCK_SIZE, zero_start=zero_start, soft_gate=True, residual=True))
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


def measure_lengthening(seeds: range = range(3)) -> dict[int, list[float]]:
    """design5's second table: the 2 layer attention trained on 8 pairs, scored on new sequences of each length."""
    results: dict[int, list[float]] = {}

    with pinned_threads():
        for seed in seeds:
            model = recall_model("attention, 2 layers", 8, seed)
            DIAGNOSE_REFERENCE["track"](model, fresh_batches(8, seed + 1), held_out(8), 1500, 1500)

            with torch.no_grad():
                for pairs in (4, 8, 12, 16):
                    tokens, targets = held_out(pairs)
                    results.setdefault(pairs, []).append(RECALL_REFERENCE["query_accuracy"](model(tokens), targets))

    return results
