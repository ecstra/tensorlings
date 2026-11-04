"""75_shortest_paths, Section 10's project, a network shaped like Bellman-Ford against a generic one and an MLP, on graphs bigger than training had."""

from __future__ import annotations

import textwrap

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
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import pinned_threads

SHORTEST_PATHS = Chapter(
    folder="75_shortest_paths",
    title="Project: shortest paths",
    readme=source_file(
        """
        # Project: shortest paths

        Section 10's project. Three networks learn the length of the
        shortest path from one node to every other in a weighted graph. One
        is shaped like the Bellman-Ford algorithm, one is a generic message
        passing network, and one is an MLP that reads the whole graph at once.
        Then all three meet graphs bigger than any they trained on.

        ## Further reading

        - [Xu and colleagues, what can neural networks reason about?](https://arxiv.org/abs/1905.13211)
        - [Xu and colleagues, how neural networks extrapolate](https://arxiv.org/abs/2009.11848)
        - [Veličković and colleagues, neural execution of graph algorithms](https://arxiv.org/abs/1910.10593)
        """,
    ),
)

GRAPHS_PART = '''
# Weighted graphs as weight matrices, 0 where there is no edge, with weights from 0.1 to 1.
# A random path through every node keeps each graph connected. chance is the chance of a
# random edge each way between two nodes, so a smaller chance leaves longer shortest paths.
def random_graphs(
    count: int,
    nodes: int,
    generator: torch.Generator,
    chance: float = 0.3,
) -> Tensor:
    weights = torch.rand(count, nodes, nodes, generator=generator) * 0.9 + 0.1
    edges = torch.rand(count, nodes, nodes, generator=generator) < chance

    for graph in range(count):
        order = torch.randperm(nodes, generator=generator)
        edges[graph, order[:-1], order[1:]] = True

    edges = (edges | edges.transpose(1, 2)) & ~torch.eye(nodes, dtype=torch.bool)

    return torch.where(edges, torch.minimum(weights, weights.transpose(1, 2)), torch.zeros(()))
'''

COSTS_SOLUTION = '''
def edge_costs(weights: Tensor) -> Tensor:
    return torch.where(weights > 0, weights, torch.full_like(weights, float("inf")))


def relax(
    distance: Tensor,
    costs: Tensor,
) -> Tensor:
    offers = (distance.unsqueeze(2) + costs).min(dim=1).values

    return torch.minimum(distance, offers)


def shortest_paths(weights: Tensor) -> Tensor:
    count, nodes, _ = weights.shape
    distance = torch.full((count, nodes), float("inf"))
    distance[:, 0] = 0
    costs = edge_costs(weights)

    for _ in range(nodes - 1):
        distance = relax(distance, costs)

    return distance
'''

INDENTED_GRAPHS = textwrap.indent(GRAPHS_PART.strip("\n"), "    ")
INDENTED_COSTS = textwrap.indent(COSTS_SOLUTION.strip("\n"), "    ")


def namespace_of(code: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Tensor": Tensor,
    }
    exec(code, namespace)

    return namespace


PATHS_REFERENCE = namespace_of(GRAPHS_PART + COSTS_SOLUTION)


# ---------------------------------------------------------------------------
# paths1
# ---------------------------------------------------------------------------

PATHS1_LESSON = prose(
    """
    This is Section 10's project. Each graph has weighted edges, and the task
    is the length of the shortest path from node 0 to every other node.
    message3 ran Bellman-Ford as message passing on one graph's edge list.
    Here many graphs come at once, each a weight matrix. That is graph1's
    adjacency matrix with each edge's weight in place of its 1, and 0 where
    there is no edge.

    A round is the same as in message3. Every node takes the best offer from
    its neighbours, each neighbour's distance plus the edge's weight, if that
    beats what it has.

    ```text
    d[v] = min(d[v], min over neighbours u of d[u] + w(u, v))
    ```

    On a matrix, one round is a min down each column of `d[u] + w(u, v)` for
    the whole batch at once. A shortest path has at most n - 1 edges, so
    n - 1 rounds find them all. paths2 builds a network on exactly this shape.

    Missing edges get a cost of infinity, so their offers never win.
    `torch.minimum(a, b)` keeps the smaller of each pair of entries.

    ```python
    torch.tensor([3.0, float("inf")]).min()                                # tensor(3.)
    torch.minimum(torch.tensor([3.0, 1.0]), torch.tensor([2.0, 5.0]))    # tensor([2., 1.])
    ```
    """,
)

PATHS1_SOURCE = source_file(
    '''
    # paths1.py

    import torch
    from torch import Tensor

    smallest = torch.tensor([3.0, float("inf")]).min()                             # tensor(3.)
    smaller = torch.minimum(torch.tensor([3.0, 1.0]), torch.tensor([2.0, 5.0]))    # tensor([2., 1.])


''' + INDENTED_GRAPHS + '''


    # TODO: Return weights with every 0, a missing edge, replaced by infinity,
    #   so a missing edge's offer never wins.
    def edge_costs(weights: Tensor) -> Tensor:
        ...


    # TODO: Return the distances after one round of Bellman-Ford. distance has
    #   shape (graphs, nodes), and costs[g, u, v] is the cost of the edge from
    #   u to v. Each node v keeps the smaller of distance[v] and the best
    #   distance[u] + costs[u, v] over every u.
    def relax(
        distance: Tensor,
        costs: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the shortest distance from node 0 to every node of each graph
    #   of weights, of shape (graphs, nodes, nodes). Start node 0 at 0 and every
    #   other node at infinity, then run nodes - 1 rounds of relax.
    def shortest_paths(weights: Tensor) -> Tensor:
        ...
    ''',
)

PATHS1_SOLUTION = source_file(
    '''
    # paths1.py, a solution for comparison

    import torch
    from torch import Tensor

    smallest = torch.tensor([3.0, float("inf")]).min()
    smaller = torch.minimum(torch.tensor([3.0, 1.0]), torch.tensor([2.0, 5.0]))


''' + INDENTED_GRAPHS + '''


''' + INDENTED_COSTS + '''
    ''',
)

LINE_GRAPH = torch.tensor(
    [
        [
            [0.0, 0.2, 0.0, 0.9],
            [0.2, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.1],
            [0.9, 0.0, 0.1, 0.0],
        ],
    ],
)

# Edges from 0 to 1, 1 to 2 and 2 to 0 only, so reading a cost the wrong way round shows.
ONE_WAY_GRAPH = torch.tensor(
    [
        [
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.0, 0.0],
        ],
    ],
)


def check_paths1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="edge_costs",
        description="turns every missing edge into infinity",
        inputs=(
            (LINE_GRAPH,),
        ),
        reference=PATHS_REFERENCE["edge_costs"],
        expect=expect_tensor,
    )
    costs = PATHS_REFERENCE["edge_costs"](LINE_GRAPH)
    start = torch.tensor([[0.0, float("inf"), float("inf"), float("inf")]])
    check_on_inputs(
        checker=checker,
        function="relax",
        description="takes the best offer from every neighbour, once",
        inputs=(
            (start, Described(costs, "the costs of a graph of 4 nodes")),
            (torch.tensor([[0.0, 0.2, float("inf"), 0.9]]), Described(costs, "the costs of a graph of 4 nodes")),
            (
                torch.tensor([[0.0, float("inf"), float("inf")]]),
                Described(PATHS_REFERENCE["edge_costs"](ONE_WAY_GRAPH), "the costs of a one way graph, 0 to 1 to 2 to 0"),
            ),
        ),
        reference=PATHS_REFERENCE["relax"],
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="shortest_paths",
        description="finds every shortest distance from node 0",
        inputs=(
            (LINE_GRAPH,),
            (Described(PATHS_REFERENCE["random_graphs"](5, 7, torch.Generator().manual_seed(750)), "5 random graphs of 7 nodes"),),
        ),
        reference=PATHS_REFERENCE["shortest_paths"],
        expect=expect_tensor,
    )


PATHS1 = Exercise(
    name="paths1",
    chapter=SHORTEST_PATHS,
    title="Project: Bellman-Ford",
    lesson=PATHS1_LESSON,
    source=PATHS1_SOURCE,
    solution=PATHS1_SOLUTION,
    hints={
        "edge_costs": "torch.where picks from weights where weights > 0, and elsewhere from a tensor of weights' shape "
        "full of float(\"inf\"), which torch.full_like makes.",
        "relax": "distance.unsqueeze(2) + costs puts each u's offer to each v along dim 1. Take .min(dim=1).values, "
        "then torch.minimum with distance.",
        "shortest_paths": "torch.full((count, nodes), float(\"inf\")) with distance[:, 0] = 0. Then loop nodes - 1 "
        "times over relax with edge_costs of weights.",
    },
    introduces=(),
    check=check_paths1,
)


# ---------------------------------------------------------------------------
# paths2
# ---------------------------------------------------------------------------

# A node no path has reached yet starts this far away, much farther than any real path.
FAR = 100.0

MIN_PASSING_INIT = '''
class MinPassing(nn.Module):
    def __init__(
        self,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        self.offer = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
'''

MIN_PASSING_FORWARD = '''
    def forward(self, weights: Tensor) -> Tensor:
        count, nodes, _ = weights.shape
        distance = torch.full((count, nodes), FAR)
        distance[:, 0] = 0
        edges = weights > 0

        for _ in range(nodes - 1):
            pairs = torch.stack([distance.unsqueeze(2).expand(count, nodes, nodes), weights], dim=3)
            offers = self.offer(pairs).squeeze(3).masked_fill(~edges, float("inf"))
            distance = torch.minimum(distance, offers.min(dim=1).values)

        return distance
'''

FLAT_MLP_PART = '''
# An MLP that reads a graph's whole weight matrix at once, padded with zeros to 20 nodes.
class FlatMLP(nn.Module):
    def __init__(
        self,
        nodes: int = 20,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.net = nn.Sequential(
            nn.Linear(nodes * nodes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nodes),
        )

    def forward(self, weights: Tensor) -> Tensor:
        nodes = weights.shape[1]
        padded = F.pad(weights, (0, self.nodes - nodes, 0, self.nodes - nodes))

        return self.net(padded.flatten(1))[:, :nodes]
'''

TRAINING_SOLUTION = '''
def within(
    guess: Tensor,
    truth: Tensor,
    tolerance: float,
) -> float:
    close = (guess - truth).abs()[:, 1:] <= tolerance

    return close.float().mean().item()


def train_paths(
    model: nn.Module,
    steps: int,
    nodes: int,
    generator: torch.Generator,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(steps):
        chance = 0.03 if step % 2 else 0.3
        weights = random_graphs(32, nodes, generator, chance=chance)
        loss = F.mse_loss(model(weights), shortest_paths(weights))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
'''

INDENTED_MIN_INIT = textwrap.indent(MIN_PASSING_INIT.strip("\n"), "    ")
INDENTED_MIN_FORWARD = textwrap.indent(MIN_PASSING_FORWARD.strip("\n"), "    ")
INDENTED_FLAT = textwrap.indent(FLAT_MLP_PART.strip("\n"), "    ")
INDENTED_TRAINING = textwrap.indent(TRAINING_SOLUTION.strip("\n"), "    ")

PATHS2_LESSON = prose(
    """
    Two networks learn the same distances. The first is shaped like
    Bellman-Ford. Each node holds one number, its distance so far, starting at
    0 for node 0 and `FAR` everywhere else. In each of nodes - 1 rounds,
    every neighbour makes an offer from its own distance and the edge's
    weight. The node keeps the smallest offer if it beats what it has.

    The algorithm's offer is `d[u] + w(u, v)`. Here a small MLP makes the
    offer, and it has to learn that sum from examples. Everything else, the
    min and the rounds, is built in. It has 65 weights.

    The second is an MLP that reads the graph's whole weight matrix at once
    and gives every distance, with 173,588 weights. It reads 20 nodes, so a
    graph of 10 is padded with zeros.

    Both train on graphs of 10. `random_graphs` takes a `chance` of a random
    edge each way between two nodes. At 0.3, Bellman-Ford needs at most 4
    rounds on 200 graphs, and at 0.03 up to 9. Training switches between
    the two every step. Xu and colleagues found that varied training graphs
    let such a network handle graphs unlike the ones it saw.

    Here is the share of distances within 0.05 on 200 new graphs of 10 at
    0.3, over 3 seeds. One column trains for 1,500 steps of 32 new graphs,
    48,000 in all. The other trains as long on only 8 graphs, 4 at each
    chance.

    | Network | Weights | 48,000 graphs | 8 graphs |
    |---|---|---|---|
    | Shaped like Bellman-Ford | 65 | 100% | 100% |
    | MLP on the whole matrix | 173,588 | 22% to 23% | 15% to 16% |

    The MLP has to discover from examples that distances add along paths and
    that the shortest one wins. The other network was handed both facts in
    its shape, and only has to learn the sum. Its offer stayed within 0.1 of
    `d + w` for every distance up to 4, in every seed.
    """,
)

PATHS2_SOURCE = source_file(
    '''
    # paths2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    sparse = 0.03 if 7 % 2 else 0.3    # 0.03, since 7 % 2 is 1, the remainder after dividing by 2

    FAR = 100.0


    # random_graphs and the Bellman-Ford of paths1.
''' + INDENTED_GRAPHS + '''


''' + INDENTED_COSTS + '''


''' + INDENTED_FLAT + '''


    # TODO: Finish forward, shaped like Bellman-Ford. distance starts at FAR,
    #   with node 0 at 0. Each of nodes - 1 rounds, self.offer reads every pair
    #   (distance[u], weights[u, v]), stacked on a last dimension of 2. Offers
    #   along missing edges become infinity, and each node v keeps the smaller
    #   of its distance and its best offer. Return the distances, of shape
    #   (graphs, nodes).
''' + INDENTED_MIN_INIT + '''

        def forward(self, weights: Tensor) -> Tensor:
            ...


    # TODO: Return the share of distances in guess within tolerance of truth,
    #   both of shape (graphs, nodes), leaving out node 0, whose distance is 0
    #   by construction.
    def within(
        guess: Tensor,
        truth: Tensor,
        tolerance: float,
    ) -> float:
        ...


    # TODO: Train model with Adam at a rate of 0.003 for steps steps. Each step
    #   draws 32 graphs of nodes nodes with random_graphs and generator, at a
    #   chance of 0.03 on odd steps and 0.3 on even ones, and takes the mean
    #   squared error against shortest_paths.
    def train_paths(
        model: nn.Module,
        steps: int,
        nodes: int,
        generator: torch.Generator,
    ) -> nn.Module:
        ...
    ''',
)

PATHS2_SOLUTION = source_file(
    '''
    # paths2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    sparse = 0.03 if 7 % 2 else 0.3

    FAR = 100.0


''' + INDENTED_GRAPHS + '''


''' + INDENTED_COSTS + '''


''' + INDENTED_FLAT + '''


''' + INDENTED_MIN_INIT + '''

''' + INDENTED_MIN_FORWARD + '''


''' + INDENTED_TRAINING + '''
    ''',
)

MODELS_REFERENCE = namespace_of("FAR = 100.0\n" + GRAPHS_PART + COSTS_SOLUTION + FLAT_MLP_PART + MIN_PASSING_INIT + MIN_PASSING_FORWARD + TRAINING_SOLUTION)


def seeded_model(
    make: type,
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)

    return make()


def check_min_forward(found: type) -> None:
    model = expect_module("MinPassing()", seeded_model(found))
    weights = PATHS_REFERENCE["random_graphs"](3, 6, torch.Generator().manual_seed(751))

    with torch.no_grad():
        expected = seeded_model(MODELS_REFERENCE["MinPassing"])(weights)

    expect_tensor(
        call="MinPassing() on 3 random graphs of 6 nodes",
        actual=model(weights),
        expected=expected,
    )


def check_min_learns(found: type) -> None:
    model = MODELS_REFERENCE["train_paths"](seeded_model(found), 300, 8, torch.Generator().manual_seed(0))
    weights = PATHS_REFERENCE["random_graphs"](100, 10, torch.Generator().manual_seed(752))

    with torch.no_grad():
        share = MODELS_REFERENCE["within"](model(weights), PATHS_REFERENCE["shortest_paths"](weights), 0.05)

    if share < 0.9:
        raise Mismatch(f"After 300 steps on graphs of 8, MinPassing gets {share:.1%} of distances within 0.05 on new graphs of 10, but it should pass 90%.")


def check_train_paths(found: object) -> None:
    results = []

    for train in (found, MODELS_REFERENCE["train_paths"]):
        model = seeded_model(MODELS_REFERENCE["MinPassing"])
        train(model, 5, 6, torch.Generator().manual_seed(753))
        results.append(model.offer[0].weight.detach().clone())

    expect_tensor(
        call="the offer's first weights after train_paths for 5 steps on graphs of 6",
        actual=results[0],
        expected=results[1],
    )


def check_paths2(checker: Checker) -> None:
    checker.test_method(
        class_name="MinPassing",
        method="forward",
        description="runs rounds of learned offers, keeping the smallest",
        run=check_min_forward,
    )
    truth = torch.tensor([[0.0, 0.5, 0.9, 1.2], [0.0, 0.3, 0.6, 0.7]])
    check_on_inputs(
        checker=checker,
        function="within",
        description="gives the share of distances close enough, node 0 left out",
        inputs=(
            (torch.tensor([[0.3, 0.52, 1.0, 1.2], [0.0, 0.3, 0.64, 0.5]]), truth, 0.05),
            (torch.tensor([[9.0, 0.5, 0.9, 1.0], [0.0, 0.4, 0.6, 0.7]]), truth, 0.15),
        ),
        reference=MODELS_REFERENCE["within"],
        expect=expect_close,
    )
    checker.test(
        function="train_paths",
        description="trains with Adam on the mean squared error",
        run=check_train_paths,
    )
    checker.test_method(
        class_name="MinPassing",
        method="forward",
        description="learns distances within 0.05",
        run=check_min_learns,
    )


PATHS2 = Exercise(
    name="paths2",
    chapter=SHORTEST_PATHS,
    title="Project: a network shaped like the algorithm",
    lesson=PATHS2_LESSON,
    source=PATHS2_SOURCE,
    solution=PATHS2_SOLUTION,
    hints={
        "MinPassing.forward": "torch.stack([distance.unsqueeze(2).expand(count, nodes, nodes), weights], dim=3) makes "
        "the pairs. self.offer(pairs).squeeze(3), masked_fill with infinity where there is no edge, then the min "
        "over dim=1 and torch.minimum with distance.",
        "within": "Take (guess - truth).abs() from column 1 on, compare with tolerance, and average as floats.",
        "train_paths": "Loop with for step in range(steps). Pick the chance from step % 2 as the example's sparse line "
        "does, and pass it as chance= to random_graphs(32, nodes, generator). Then F.mse_loss of model(weights) and "
        "shortest_paths(weights), zero_grad, backward, step.",
    },
    introduces=(),
    check=check_paths2,
    time_limit=60,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# paths3
# ---------------------------------------------------------------------------

GENERIC_PART = '''
# A generic message passing network, message1's with message3's minimum in place of the sum. A vector per
# node, a learned message along each edge, the smallest of them gathered, and a residual update, for
# nodes - 1 rounds as Bellman-Ford takes, or for rounds rounds when told.
class MessagePassing(nn.Module):
    def __init__(
        self,
        size: int = 32,
    ) -> None:
        super().__init__()
        self.start = nn.Linear(1, size)
        self.message = nn.Sequential(nn.Linear(size + 1, size), nn.ReLU(), nn.Linear(size, size))
        self.update = nn.Sequential(nn.Linear(2 * size, size), nn.ReLU(), nn.Linear(size, size))
        self.read = nn.Linear(size, 1)

    def forward(
        self,
        weights: Tensor,
        rounds: int | None = None,
    ) -> Tensor:
        count, nodes, _ = weights.shape
        source = torch.zeros(count, nodes, 1)
        source[:, 0] = 1
        state = self.start(source)
        edges = weights > 0

        for _ in range(nodes - 1 if rounds is None else rounds):
            pairs = torch.cat([state.unsqueeze(2).expand(count, nodes, nodes, -1), weights.unsqueeze(3)], dim=3)
            messages = self.message(pairs).masked_fill(~edges.unsqueeze(3), float("inf"))
            state = state + self.update(torch.cat([state, messages.min(dim=1).values], dim=2))

        return self.read(state).squeeze(2)
'''

SIZES_SOLUTION = '''
def scores_by_size(
    model: nn.Module,
    sizes: list[int],
    count: int,
    generator: torch.Generator,
) -> list[float]:
    scores = []

    with torch.no_grad():
        for nodes in sizes:
            weights = random_graphs(count, nodes, generator, chance=0.4 / nodes)
            scores.append(within(model(weights), shortest_paths(weights), 0.05))

    return scores


def offer_gap(
    model: nn.Module,
    distance: Tensor,
    weight: Tensor,
) -> Tensor:
    with torch.no_grad():
        offers = model.offer(torch.stack([distance, weight], dim=-1)).squeeze(-1)

    return offers - (distance + weight)
'''

INDENTED_GENERIC = textwrap.indent(GENERIC_PART.strip("\n"), "    ")
INDENTED_SIZES = textwrap.indent(SIZES_SOLUTION.strip("\n"), "    ")

PATHS3_LESSON = prose(
    """
    Now all three networks meet graphs bigger than any they trained on. The
    third is a generic message passing network, message1's with message3's
    minimum in place of the sum. Each node holds a vector of 32 numbers, a
    learned message runs along every edge, and a residual update takes the
    smallest messages in. Like the first, it runs nodes - 1 rounds.

    The test graphs have a chance of 0.4 / nodes, so a node keeps about as
    many random edges at every size. Bigger graphs then have longer shortest
    paths. Bellman-Ford needs up to 8 rounds at 10 nodes, 17 at 20 and 18 at
    40, and distances reach 6.7.

    Each network trained as in paths2. Here is the share within 0.05 on 200
    new graphs of each size, over 3 seeds.

    | Network | Weights | 10 nodes | 20 nodes | 40 nodes |
    |---|---|---|---|---|
    | Shaped like Bellman-Ford | 65 | 100% | 99.4% to 100% | 99.7% to 100% |
    | Generic, nodes - 1 rounds | 5,377 | 54% to 74% | 0% to 55% | 0% to 37% |
    | Generic, the 9 rounds it trained with | 5,377 | 54% to 74% | 44% to 57% | 39% to 44% |
    | MLP on the whole matrix | 173,588 | 8% to 12% | 3% to 4% | can't read it |

    The generic network learned a computation for exactly 9 rounds. Given
    more, two of the three seeds drifted to 0%. Held to 9, it can't follow a
    path of more edges. The first network keeps the smaller distance every
    round, so extra rounds change nothing once a distance is right. The MLP
    can't even read a graph bigger than its input.

    `offer_gap` shows what the first network learned. Read at the true
    distances, its worst offer along shortest paths on graphs of 10 was off
    by 0.006 to 0.034. Training distances reach about 4.7, and up to 4 every
    offer stayed within 0.1 of the sum. Past that one seed's offers drifted
    by up to 1.1, while another stayed within 0.006 all the way to 8. Fewer
    than 1 in 100 test distances lie past 4, so the scores stay near 100%.

    Xu and colleagues call this algorithmic alignment. A network learns from
    few examples when its structure matches the steps of the algorithm, so
    only a simple piece is left to learn. Section 14 asks you to design
    modules this way.
    """,
)

PATHS3_SOURCE = source_file(
    '''
    # paths3.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    FAR = 100.0

    pairs = torch.stack([torch.tensor([0.5, 2.0]), torch.tensor([0.3, 0.1])], dim=-1)    # shape (2, 2)


    # The pieces of paths1 and paths2.
''' + INDENTED_GRAPHS + '''


''' + INDENTED_COSTS + '''


''' + INDENTED_FLAT + '''


''' + INDENTED_MIN_INIT + '''

''' + INDENTED_MIN_FORWARD + '''


''' + INDENTED_TRAINING + '''


''' + INDENTED_GENERIC + '''


    # TODO: Return model's share of distances within 0.05 for count graphs of
    #   each size in sizes, in order. Draw each size's graphs with
    #   random_graphs, generator and a chance of 0.4 / nodes, and score them
    #   with paths2's within. Record nothing for autograd.
    def scores_by_size(
        model: nn.Module,
        sizes: list[int],
        count: int,
        generator: torch.Generator,
    ) -> list[float]:
        ...


    # TODO: Return how far a MinPassing model's offer is from the true sum,
    #   which is model.offer of each pair (distance, weight) minus the sum
    #   (distance + weight). distance and weight have the same shape. Record
    #   nothing for autograd.
    def offer_gap(
        model: nn.Module,
        distance: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ...
    ''',
)

PATHS3_SOLUTION = source_file(
    '''
    # paths3.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    FAR = 100.0

    pairs = torch.stack([torch.tensor([0.5, 2.0]), torch.tensor([0.3, 0.1])], dim=-1)


''' + INDENTED_GRAPHS + '''


''' + INDENTED_COSTS + '''


''' + INDENTED_FLAT + '''


''' + INDENTED_MIN_INIT + '''

''' + INDENTED_MIN_FORWARD + '''


''' + INDENTED_TRAINING + '''


''' + INDENTED_GENERIC + '''


''' + INDENTED_SIZES + '''
    ''',
)

SIZES_REFERENCE = namespace_of(
    "FAR = 100.0\n" + GRAPHS_PART + COSTS_SOLUTION + FLAT_MLP_PART + MIN_PASSING_INIT + MIN_PASSING_FORWARD
    + TRAINING_SOLUTION + GENERIC_PART + SIZES_SOLUTION,
)


def close(
    found: Tensor,
    expected: Tensor,
) -> bool:
    return found.shape == expected.shape and torch.allclose(found, expected, atol=1e-6)


def check_scores_by_size(found: object) -> None:
    """The file's within is swapped for one that records what it scores and answers by the graphs' size,
    so the check sees which graphs scores_by_size drew and in what order."""
    scored: list[tuple[Tensor, Tensor]] = []

    def scripted_within(
        guess: Tensor,
        truth: Tensor,
        tolerance: float,
    ) -> float:
        scored.append((guess.detach(), truth.detach()))

        if tolerance != 0.05:
            raise Mismatch(f"scores_by_size used a tolerance of {tolerance}, but it should use 0.05.")

        return truth.shape[1] / 100

    namespace = found.__globals__
    original = namespace.get("within")
    namespace["within"] = scripted_within
    modes: list[bool] = []
    model = seeded_model(SIZES_REFERENCE["MinPassing"])
    hook = model.register_forward_pre_hook(lambda module, inputs: modes.append(torch.is_grad_enabled()))

    try:
        actual = found(model, [6, 9, 4], 5, torch.Generator().manual_seed(754))
    finally:
        namespace["within"] = original
        hook.remove()

    if any(modes):
        raise Mismatch("scores_by_size ran the model while recording for autograd. Run it inside torch.no_grad().")

    if not scored:
        raise Mismatch("scores_by_size never called within. Score each size's graphs with paths2's within, at 0.05.")

    expect_value(
        call="scores_by_size(model, [6, 9, 4], 5, a generator seeded with 754)",
        actual=actual,
        expected=[0.06, 0.09, 0.04],
    )

    if [tuple(truth.shape) for _, truth in scored] != [(5, 6), (5, 9), (5, 4)]:
        raise Mismatch("scores_by_size should draw count graphs of each size in turn and score each batch with within.")

    replay = torch.Generator().manual_seed(754)

    for nodes, (guess, truth) in zip([6, 9, 4], scored):
        weights = SIZES_REFERENCE["random_graphs"](5, nodes, replay, chance=0.4 / nodes)
        right_truth = SIZES_REFERENCE["shortest_paths"](weights)

        with torch.no_grad():
            right_guess = model(weights)

        # within gives the same share with its first two arguments swapped, so either order passes.
        in_order = close(guess, right_guess) and close(truth, right_truth)
        swapped = close(guess, right_truth) and close(truth, right_guess)

        if not (in_order or swapped):
            raise Mismatch(
                f"For graphs of {nodes} nodes, scores_by_size scored something other than model(weights) against "
                "shortest_paths(weights), with weights drawn by random_graphs from the generator it was given, at a "
                "chance of 0.4 / nodes.",
            )


def check_offer_gap(found: object) -> None:
    model = seeded_model(SIZES_REFERENCE["MinPassing"])
    distance = torch.tensor([[0.0, 0.5, 2.0], [4.0, 1.5, 0.2]])
    weight = torch.tensor([[0.3, 0.9, 0.1], [0.5, 0.2, 0.7]])
    modes: list[bool] = []
    hook = model.offer.register_forward_pre_hook(lambda module, inputs: modes.append(torch.is_grad_enabled()))

    try:
        actual = found(model, distance, weight)
    finally:
        hook.remove()

    if any(modes):
        raise Mismatch("offer_gap ran model.offer while recording for autograd. Run it inside torch.no_grad().")

    expect_tensor(
        call="offer_gap of an untrained MinPassing on 2 rows of 3 pairs",
        actual=actual,
        expected=SIZES_REFERENCE["offer_gap"](model, distance, weight),
    )


def check_paths3(checker: Checker) -> None:
    checker.test(
        function="scores_by_size",
        description="scores each size of graph in turn, without autograd",
        run=check_scores_by_size,
    )
    checker.test(
        function="offer_gap",
        description="gives how far each offer is from distance plus weight",
        run=check_offer_gap,
    )


PATHS3 = Exercise(
    name="paths3",
    chapter=SHORTEST_PATHS,
    title="Project: graphs bigger than training had",
    lesson=PATHS3_LESSON,
    source=PATHS3_SOURCE,
    solution=PATHS3_SOLUTION,
    hints={
        "scores_by_size": "Inside torch.no_grad(), for each nodes in sizes draw random_graphs(count, nodes, generator, "
        "chance=0.4 / nodes) and append within(model(weights), shortest_paths(weights), 0.05).",
        "offer_gap": "torch.stack([distance, weight], dim=-1) gives the pairs. Run model.offer, .squeeze(-1), and take "
        "away distance + weight, all inside torch.no_grad().",
    },
    introduces=(),
    check=check_paths3,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# The runs behind paths2's and paths3's tables
# ---------------------------------------------------------------------------

TEST_SIZES = (10, 20, 40)


def bellman_ford_rounds(weights: Tensor) -> int:
    """How many rounds of paths1's relax change something, for the hardest graph of the batch."""
    costs = PATHS_REFERENCE["edge_costs"](weights)
    distance = torch.full(weights.shape[:2], float("inf"))
    distance[:, 0] = 0
    rounds = 0

    while True:
        relaxed = PATHS_REFERENCE["relax"](distance, costs)

        if torch.equal(relaxed, distance):
            return rounds

        distance = relaxed
        rounds += 1


def train_on_pool(
    model: nn.Module,
    pool: Tensor,
    steps: int,
    generator: torch.Generator,
) -> nn.Module:
    """paths2's training, with every batch of 32 drawn from a fixed pool of graphs."""
    truth = PATHS_REFERENCE["shortest_paths"](pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        batch = torch.randint(0, len(pool), (32,), generator=generator)
        loss = F.mse_loss(model(pool[batch]), truth[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def offers_on_paths(
    model: nn.Module,
    weights: Tensor,
) -> tuple[float, float]:
    """The worst offer along shortest paths, where the offers decide each answer, and the furthest any other offer
    came in under its node's true distance. Every offer is read at the true distances."""
    distance = SIZES_REFERENCE["shortest_paths"](weights)
    count, nodes, _ = weights.shape
    start = distance.unsqueeze(2).expand(count, nodes, nodes)
    gap = SIZES_REFERENCE["offer_gap"](model, start, weights)
    edges = weights > 0
    on_path = edges & ((start + weights - distance.unsqueeze(1)).abs() < 1e-5)
    on_path[:, :, 0] = False
    under = distance.unsqueeze(1) - (start + weights + gap)

    return gap[on_path].abs().max().item(), under[edges & ~on_path].max().item()


def measure_paths(seeds: range = range(3)) -> dict[str, object]:
    """paths2's and paths3's tables, by seed. Each seed trains the three networks with train_paths for 1,500 steps
    on graphs of 10, and the aligned network and the MLP for 1,500 steps on a pool of only 8 graphs, 4 at each
    chance. paths2 scores on 200 new graphs of 10 at a chance of 0.3, and paths3 on 200 of each size at 0.4 / nodes,
    the generic network both at nodes - 1 rounds and at the 9 it trained with."""
    within = SIZES_REFERENCE["within"]
    shortest_paths = SIZES_REFERENCE["shortest_paths"]
    random_graphs = SIZES_REFERENCE["random_graphs"]
    makers = {
        "aligned": SIZES_REFERENCE["MinPassing"],
        "generic": SIZES_REFERENCE["MessagePassing"],
        "MLP": SIZES_REFERENCE["FlatMLP"],
    }
    results: dict[str, object] = {}

    with pinned_threads():
        dense = random_graphs(200, 10, torch.Generator().manual_seed(900))
        sparse = random_graphs(200, 10, torch.Generator().manual_seed(901), chance=0.03)
        tests = {nodes: random_graphs(200, nodes, torch.Generator().manual_seed(900 + nodes), chance=0.4 / nodes) for nodes in TEST_SIZES}
        results["rounds needed"] = {
            "10 at 0.3": bellman_ford_rounds(dense),
            "10 at 0.03": bellman_ford_rounds(sparse),
            **{f"{nodes} at 0.4 / nodes": bellman_ford_rounds(weights) for nodes, weights in tests.items()},
        }
        results["largest distance"] = {
            "10 at 0.3": shortest_paths(dense).max().item(),
            "10 at 0.03": shortest_paths(sparse).max().item(),
            **{f"{nodes} at 0.4 / nodes": shortest_paths(weights).max().item() for nodes, weights in tests.items()},
        }

        for seed in seeds:
            trained = {}

            for name, make in makers.items():
                torch.manual_seed(seed)
                trained[name] = SIZES_REFERENCE["train_paths"](make(), 1500, 10, torch.Generator().manual_seed(seed))

            pool_generator = torch.Generator().manual_seed(500 + seed)
            pool = torch.cat([random_graphs(4, 10, pool_generator), random_graphs(4, 10, pool_generator, chance=0.03)])

            with torch.no_grad():
                for name in ("aligned", "MLP"):
                    results[f"{seed} paths2 {name}"] = within(trained[name](dense), shortest_paths(dense), 0.05)

            for name in ("aligned", "MLP"):
                torch.manual_seed(seed)
                on_pool = train_on_pool(makers[name](), pool, 1500, torch.Generator().manual_seed(seed))

                with torch.no_grad():
                    results[f"{seed} paths2 {name}, 8 graphs"] = within(on_pool(dense), shortest_paths(dense), 0.05)

            with torch.no_grad():
                for nodes, weights in tests.items():
                    truth = shortest_paths(weights)
                    results[f"{seed} paths3 aligned {nodes}"] = within(trained["aligned"](weights), truth, 0.05)
                    results[f"{seed} paths3 generic {nodes}"] = within(trained["generic"](weights), truth, 0.05)
                    results[f"{seed} paths3 generic, 9 rounds, {nodes}"] = within(trained["generic"](weights, rounds=9), truth, 0.05)

                    if nodes <= 20:
                        results[f"{seed} paths3 MLP {nodes}"] = within(trained["MLP"](weights), truth, 0.05)

            for nodes, weights in tests.items():
                results[f"{seed} offers {nodes}"] = offers_on_paths(trained["aligned"], weights)

            for largest in (2.0, 4.0, 8.0):
                distance, weight = torch.meshgrid(torch.linspace(0, largest, 81), torch.linspace(0.1, 1.0, 10), indexing="ij")
                gap = SIZES_REFERENCE["offer_gap"](trained["aligned"], distance, weight)
                results[f"{seed} worst offer up to {largest:g}"] = gap.abs().max().item()

    return results
