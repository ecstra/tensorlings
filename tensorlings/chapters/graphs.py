"""73_graphs, graphs as an adjacency matrix and an edge list, message passing, the GCN's normalization and oversmoothing, and why message passing lines up with Bellman-Ford."""

from __future__ import annotations

import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import Chapter, Exercise
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

GRAPHS = Chapter(
    folder="73_graphs",
    title="Graphs and message passing",
    readme=source_file(
        """
        # Graphs and message passing

        A graph is a set of nodes with edges between them. Its nodes have no
        order, like a set's elements, and its edges say which nodes affect
        which. This chapter stores graphs, builds message passing, the layer
        every graph network is made of, normalizes it as the GCN does, and
        shows why it fits algorithms on graphs like shortest paths.

        ## Further reading

        - [Sanchez-Lengeling and colleagues, a gentle introduction to graph neural networks](https://distill.pub/2021/gnn-intro/)
        - [Gilmer and colleagues, neural message passing for quantum chemistry](https://arxiv.org/abs/1704.01212)
        - [Kipf and Welling, graph convolutional networks](https://arxiv.org/abs/1609.02907)
        - [Xu and colleagues, what can neural networks reason about?](https://arxiv.org/abs/1905.13211)
        """,
    ),
)

# graph1's and message1's pieces, which later files are handed, written once so they never drift apart.
# An edge list has shape (2, edges): sources in row 0, targets in row 1, as PyTorch Geometric stores it.
GRAPH_PART = '''
def adjacency_from_edges(
    edges: Tensor,
    nodes: int,
) -> Tensor:
    adjacency = torch.zeros(nodes, nodes)
    adjacency[edges[0], edges[1]] = 1.0

    return adjacency


def both_ways(edges: Tensor) -> Tensor:
    return torch.cat([edges, edges.flip(0)], dim=1)
'''

AGGREGATE_PART = '''
def aggregate(
    messages: Tensor,
    targets: Tensor,
    nodes: int,
) -> Tensor:
    return torch.zeros(nodes, messages.shape[1]).index_add_(0, targets, messages)
'''

MESSAGE_PASSING_PART = '''
class MessagePassing(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.message = nn.Linear(features, features)
        self.update = nn.Linear(2 * features, features)

    def forward(
        self,
        h: Tensor,
        edges: Tensor,
    ) -> Tensor:
        incoming = aggregate(self.message(h[edges[0]]), edges[1], len(h))

        return torch.relu(self.update(torch.cat([h, incoming], dim=1)))
'''


def indented(part: str) -> str:
    """Indented like the files around it, which source_file takes the indentation off."""
    return textwrap.indent(part.strip("\n"), "    ")


def graph_namespace() -> dict[str, object]:
    """The pieces above, run once, so checks and later chapters use exactly the code the files hold."""
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
    }
    exec(GRAPH_PART + AGGREGATE_PART + MESSAGE_PASSING_PART, namespace)

    return namespace


GRAPH_REFERENCE = graph_namespace()
adjacency_from_edges = GRAPH_REFERENCE["adjacency_from_edges"]
both_ways = GRAPH_REFERENCE["both_ways"]
aggregate = GRAPH_REFERENCE["aggregate"]
MessagePassing = GRAPH_REFERENCE["MessagePassing"]


def random_graph(
    nodes: int,
    chords: int,
    generator: torch.Generator,
) -> Tensor:
    """A ring through every node, so every node can reach every other, plus chords between random pairs,
    with each edge listed both ways. The first half of the list holds each edge once."""
    pairs = {(node, (node + 1) % nodes) for node in range(nodes)}

    while len(pairs) < nodes + chords:
        a, b = torch.randint(0, nodes, (2,), generator=generator).tolist()

        if a != b and (a, b) not in pairs and (b, a) not in pairs:
            pairs.add((a, b))

    return both_ways(torch.tensor(sorted(pairs)).T)


def random_lengths(
    edges: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Whole lengths from 1 to 9, the same both ways along an edge, for a list from random_graph."""
    half = torch.randint(1, 10, (edges.shape[1] // 2,), generator=generator).float()

    return torch.cat([half, half])


# ---------------------------------------------------------------------------
# graph1
# ---------------------------------------------------------------------------

GRAPH1_LESSON = prose(
    """
    A graph is a set of nodes and the edges between them. Think of the atoms
    of a molecule and its bonds, or towns and their roads. Its nodes have no
    order, like a set's elements. Its edges carry what a set doesn't have,
    which element affects which.

    There are two common ways to store the edges.

    - An adjacency matrix has a row and a column per node, with a 1 where an edge runs from the row's node to the column's.
    - An edge list has one column per edge, its source on top and its target below, as PyTorch Geometric stores it.

    ```python
    edges = torch.tensor([[0, 1, 1], [1, 2, 0]])    # 0 to 1, 1 to 2 and 1 to 0
    ```

    The matrix takes nodes squared numbers whatever the graph. The list takes
    2 per edge. A million towns joined by 2 million roads need a matrix of a
    million million entries. As a list, each road both ways, they need 8
    million numbers.

    A road runs both ways and a follow runs one way. An undirected graph lists
    each edge in both directions, so its matrix is symmetric. The degree of a
    node is how many edges arrive at it.

    Two index tensors set many entries at once, one per pair.

    ```python
    grid = torch.zeros(2, 3)
    grid[torch.tensor([0, 1]), torch.tensor([2, 0])] = 1.0    # tensor([[0., 0., 1.], [1., 0., 0.]])
    ```

    `.nonzero()` goes the other way, one row per entry that isn't 0, in
    reading order.
    """,
)

GRAPH1_SOURCE = source_file(
    '''
    # graph1.py

    import torch
    from torch import Tensor

    grid = torch.zeros(2, 3)
    grid[torch.tensor([0, 1]), torch.tensor([2, 0])] = 1.0    # tensor([[0., 0., 1.], [1., 0., 0.]])
    where = grid.nonzero()                                   # tensor([[0, 2], [1, 0]]), one row per 1


    # TODO: Return the adjacency matrix of a graph of nodes nodes, as floats,
    #   with a 1 at row s and column t for every edge from s to t. edges has
    #   shape (2, edges), sources in row 0 and targets in row 1.
    def adjacency_from_edges(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return the edge list of adjacency, shape (2, edges), sources in row 0
    #   and targets in row 1, ordered by source, then by target.
    def edges_from_adjacency(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return how many edges arrive at each node, one count per node, for
    #   nodes nodes.
    def in_degrees(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return edges followed by every edge turned around, so the graph runs
    #   both ways. The result has twice as many columns.
    def both_ways(edges: Tensor) -> Tensor:
        ...
    ''',
)

GRAPH1_SOLUTION = source_file(
    '''
    # graph1.py, a solution for comparison

    import torch
    from torch import Tensor

    grid = torch.zeros(2, 3)
    grid[torch.tensor([0, 1]), torch.tensor([2, 0])] = 1.0
    where = grid.nonzero()


    def adjacency_from_edges(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        adjacency = torch.zeros(nodes, nodes)
        adjacency[edges[0], edges[1]] = 1.0

        return adjacency


    def edges_from_adjacency(adjacency: Tensor) -> Tensor:
        return adjacency.nonzero().T


    def in_degrees(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        return torch.bincount(edges[1], minlength=nodes)


    def both_ways(edges: Tensor) -> Tensor:
        return torch.cat([edges, edges.flip(0)], dim=1)
    ''',
)

SMALL_EDGES = torch.tensor([[0, 1, 1, 3, 2], [1, 2, 0, 2, 4]])
OTHER_EDGES = torch.tensor([[0, 0, 2, 5, 4, 1], [2, 3, 1, 1, 0, 3]])


def expect_integer_tensor(
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
        raise Mismatch(f"{call} holds floats, but indices have to be whole numbers. Keep the result as integers.")


def expect_float_tensor(
    call: str,
    actual: object,
    expected: Tensor,
) -> None:
    expect_tensor(
        call=call,
        actual=actual,
        expected=expected,
    )

    if not actual.dtype.is_floating_point:
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but it should hold floats, so the matrix can multiply features.")


def check_graph1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="adjacency_from_edges",
        description="puts a 1 where each edge runs",
        inputs=(
            (SMALL_EDGES, 5),
            (OTHER_EDGES, 7),
        ),
        reference=adjacency_from_edges,
        expect=expect_float_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="edges_from_adjacency",
        description="lists the edges by source, then target",
        inputs=(
            (adjacency_from_edges(SMALL_EDGES, 5),),
            (adjacency_from_edges(OTHER_EDGES, 7),),
        ),
        reference=lambda adjacency: adjacency.nonzero().T,
        expect=expect_integer_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="in_degrees",
        description="counts the edges arriving at each node",
        inputs=(
            (SMALL_EDGES, 5),
            (OTHER_EDGES, 7),
        ),
        reference=lambda edges, nodes: torch.bincount(edges[1], minlength=nodes),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="both_ways",
        description="adds every edge turned around",
        inputs=(
            (SMALL_EDGES,),
            (OTHER_EDGES,),
        ),
        reference=both_ways,
        expect=expect_integer_tensor,
    )


GRAPH1 = Exercise(
    name="graph1",
    chapter=GRAPHS,
    title="A graph as a matrix and as a list",
    lesson=GRAPH1_LESSON,
    source=GRAPH1_SOURCE,
    solution=GRAPH1_SOLUTION,
    hints={
        "adjacency_from_edges": "Start from torch.zeros(nodes, nodes). Set every entry an edge names to 1.0 in one go, "
        "with edges[0] and edges[1] as the two index tensors.",
        "edges_from_adjacency": ".nonzero() gives one row per edge, a source and a target. Transpose it with .T to get "
        "the two rows.",
        "in_degrees": "torch.bincount of the targets, edges[1], with minlength=nodes so a node no edge reaches counts 0.",
        "both_ways": ".flip(0) swaps the two rows of edges. torch.cat it after edges along dim=1.",
    },
    introduces=(),
    check=check_graph1,
)


# ---------------------------------------------------------------------------
# message1
# ---------------------------------------------------------------------------

MESSAGE1_LESSON = prose(
    """
    Message passing, as Gilmer and colleagues framed it, updates every node
    from its neighbours. Each edge carries a message from its source to its
    target. Each node adds up the messages arriving, then updates its own
    features from that sum and from itself.

    ```text
    m_v  = sum over edges w to v of M(h_v, h_w, e_vw)
    h_v' = U(h_v, m_v)
    ```

    Here the message M is a Linear of the source's features alone. The update
    U is a Linear of the node's features and its sum, then a ReLU.

    Adding up messages by target is what `index_add_` does. It adds each row
    of a tensor into the row an index names, so many messages can land on one
    node.

    ```python
    totals = torch.zeros(3, 2)
    totals.index_add_(0, torch.tensor([2, 0, 2]), torch.ones(3, 2))    # rows [1, 1], [0, 0], [2, 2]
    ```

    The same sum is a matrix product, the transposed adjacency matrix times
    the features. The product does work for every pair of nodes, the list for
    every edge, so big sparse graphs use the list.

    A layer lets each node hear from nodes one edge away. k layers reach k
    edges away. Like an equivariant set layer, it doesn't care how the nodes
    are numbered. Renumber the 20 nodes of a random graph, and here the
    outputs came out renumbered, every number exactly the same.
    """,
)

MESSAGE1_SOURCE = source_file(
    '''
    # message1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    totals = torch.zeros(3, 2)
    totals.index_add_(0, torch.tensor([2, 0, 2]), torch.ones(3, 2))    # rows [1, 1], [0, 0], [2, 2]


    # TODO: Return the sum of the messages arriving at each node, shape (nodes,
    #   features). messages has one row per edge and targets the node each
    #   one goes to.
    def aggregate(
        messages: Tensor,
        targets: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return the sum of the features h of every node's sources, as
    #   aggregate of h[sources] would give, using the adjacency matrix and no
    #   edge list.
    def aggregate_by_matrix(
        h: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. The message along each edge is message of its
    #   source's row of h. Each node adds up the messages arriving with
    #   aggregate, then gets the ReLU of update of its own features and that
    #   sum side by side.
    class MessagePassing(nn.Module):
        def __init__(self, features: int) -> None:
            super().__init__()
            self.message = nn.Linear(features, features)
            self.update = nn.Linear(2 * features, features)

        def forward(
            self,
            h: Tensor,
            edges: Tensor,
        ) -> Tensor:
            ...
    ''',
)

MESSAGE1_SOLUTION = source_file(
    '''
    # message1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    totals = torch.zeros(3, 2)
    totals.index_add_(0, torch.tensor([2, 0, 2]), torch.ones(3, 2))


''' + indented(AGGREGATE_PART) + '''


    def aggregate_by_matrix(
        h: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        return adjacency.T @ h


''' + indented(MESSAGE_PASSING_PART) + '''
    ''',
)


def seeded_layer(
    make: type,
    features: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(features)


def renumber(
    h: Tensor,
    edges: Tensor,
    order: Tensor,
) -> tuple[Tensor, Tensor]:
    """The same graph with node order[i] called i, its features moved to match."""
    new_name = torch.empty_like(order)
    new_name[order] = torch.arange(len(order))

    return h[order], new_name[edges]


def check_message_passing(found: type) -> None:
    model = expect_module("MessagePassing(3)", seeded_layer(found, 3))
    h = torch.randn(5, 3, generator=torch.Generator().manual_seed(741))

    with torch.no_grad():
        expected = seeded_layer(MessagePassing, 3)(h, SMALL_EDGES)

    expect_tensor(
        call="MessagePassing(3) on 5 nodes with 5 edges",
        actual=model(h, SMALL_EDGES),
        expected=expected,
    )


def check_message1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="aggregate",
        description="adds up the messages arriving at each node",
        inputs=(
            (torch.randn(5, 3, generator=torch.Generator().manual_seed(742)), SMALL_EDGES[1], 5),
            (torch.randn(6, 2, generator=torch.Generator().manual_seed(743)), OTHER_EDGES[1], 7),
        ),
        reference=aggregate,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="aggregate_by_matrix",
        description="gives the same sums from the adjacency matrix",
        inputs=(
            (torch.randn(5, 3, generator=torch.Generator().manual_seed(744)), Described(adjacency_from_edges(SMALL_EDGES, 5), "a graph of 5 nodes")),
            (torch.randn(7, 2, generator=torch.Generator().manual_seed(745)), Described(adjacency_from_edges(OTHER_EDGES, 7), "a graph of 7 nodes")),
        ),
        reference=lambda h, adjacency: adjacency.T @ h,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="MessagePassing",
        method="forward",
        description="sends, adds up and updates",
        run=check_message_passing,
    )


MESSAGE1 = Exercise(
    name="message1",
    chapter=GRAPHS,
    title="Message passing",
    lesson=MESSAGE1_LESSON,
    source=MESSAGE1_SOURCE,
    solution=MESSAGE1_SOLUTION,
    hints={
        "aggregate": "Start from torch.zeros(nodes, messages.shape[1]) and call .index_add_ with dim 0, the targets "
        "and the messages.",
        "aggregate_by_matrix": "Row t of the adjacency matrix's transpose has a 1 for every source of t, so multiply it "
        "by h.",
        "MessagePassing.forward": "h[edges[0]] gives each edge its source's row. Pass self.message of it to aggregate "
        "with edges[1] and len(h). torch.cat h and the sum on dim=1, then self.update and torch.relu.",
    },
    introduces=(),
    check=check_message1,
)


# ---------------------------------------------------------------------------
# message2
# ---------------------------------------------------------------------------


def reference_gcn_matrix(adjacency: Tensor) -> Tensor:
    looped = adjacency + torch.eye(len(adjacency))
    scale = looped.sum(dim=1) ** -0.5

    return scale.unsqueeze(1) * looped * scale.unsqueeze(0)


def reference_propagate(
    h: Tensor,
    normalized: Tensor,
    layers: int,
) -> Tensor:
    for _ in range(layers):
        h = normalized @ h

    return h


def reference_mean_cosine(h: Tensor) -> float:
    unit = F.normalize(h, dim=1)
    similarity = unit @ unit.T
    nodes = len(h)

    return ((similarity.sum() - similarity.diagonal().sum()) / (nodes * (nodes - 1))).item()


MESSAGE2_LESSON = prose(
    """
    Kipf and Welling's GCN is message passing with a fixed rule for weighing
    the messages. Each node gets an edge to itself, `A + I`. Then the message
    from u to v is divided by the square root of both their degrees, counted
    with that self loop.

    ```text
    H' = ReLU(D^(-1/2) (A + I) D^(-1/2) H W)
    ```

    Without the division, a node with 100 neighbours adds 100 messages and a
    node with 2 adds 2. Features grow with degree, and stacking layers makes
    them grow without end. The self loop keeps a node's own features in its
    sum. The paper calls this rewriting the renormalization trick.

    Each layer mixes a node with its neighbours, nearly an average. Stack many
    and every node drifts to the same mix, which is called oversmoothing. Here
    is the GCN's rule with no weights, repeated on a random graph of 40 nodes,
    starting from random features. The table gives the average cosine
    similarity between every two nodes' features.

    | Layers | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
    |---|---|---|---|---|---|---|---|
    | Average cosine similarity | -0.001 | 0.085 | 0.176 | 0.344 | 0.626 | 0.895 | 0.991 |

    At 0 the random features are unrelated. Deep enough, every node points the
    same way, and no classifier can tell the nodes apart. Kipf and Welling
    found their best results with 2 or 3 layers, and their main model has 2.

    Without the division, the same graph's features grew 20,964 times larger
    over 8 layers. With it, they shrank to 0.21 of their start.

    Cosine similarity compares directions. `F.normalize(h, dim=1)` scales every
    row to a length of 1, and the dot products of those rows are the cosines.
    """,
)

MESSAGE2_SOURCE = source_file(
    '''
    # message2.py

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    degrees = torch.tensor([4.0, 1.0])
    scale = degrees ** -0.5                    # tensor([0.5000, 1.0000])
    rows = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)    # tensor([[0.6000, 0.8000]])


    # TODO: Return the GCN's matrix for adjacency, of shape (nodes, nodes): add
    #   the identity, then divide each entry at row v and column u by the
    #   square root of the degrees of v and u. A degree here is a row's sum
    #   after the identity is added.
    def gcn_matrix(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return one GCN layer: the ReLU of normalized times h times weight.
    def gcn_layer(
        h: Tensor,
        normalized: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return h after multiplying it by normalized from the left layers
    #   times, with no weights and no ReLU.
    def propagate(
        h: Tensor,
        normalized: Tensor,
        layers: int,
    ) -> Tensor:
        ...


    # TODO: Return the average cosine similarity between the features of every
    #   two different nodes, each row of h being one node, as a plain number.
    def mean_cosine(h: Tensor) -> float:
        ...
    ''',
)

MESSAGE2_SOLUTION = source_file(
    '''
    # message2.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor

    degrees = torch.tensor([4.0, 1.0])
    scale = degrees ** -0.5
    rows = F.normalize(torch.tensor([[3.0, 4.0]]), dim=1)


    def gcn_matrix(adjacency: Tensor) -> Tensor:
        looped = adjacency + torch.eye(len(adjacency))
        inverse_root = looped.sum(dim=1) ** -0.5

        return inverse_root.unsqueeze(1) * looped * inverse_root.unsqueeze(0)


    def gcn_layer(
        h: Tensor,
        normalized: Tensor,
        weight: Tensor,
    ) -> Tensor:
        return torch.relu(normalized @ h @ weight)


    def propagate(
        h: Tensor,
        normalized: Tensor,
        layers: int,
    ) -> Tensor:
        for _ in range(layers):
            h = normalized @ h

        return h


    def mean_cosine(h: Tensor) -> float:
        unit = F.normalize(h, dim=1)
        similarity = unit @ unit.T
        nodes = len(h)

        return ((similarity.sum() - similarity.diagonal().sum()) / (nodes * (nodes - 1))).item()
    ''',
)

GRAPH_A = adjacency_from_edges(random_graph(7, 4, torch.Generator().manual_seed(746)), 7)
GRAPH_B = adjacency_from_edges(random_graph(9, 2, torch.Generator().manual_seed(747)), 9)


def check_message2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="gcn_matrix",
        description="adds self loops and divides by the roots of both degrees",
        inputs=(
            (Described(GRAPH_A, "a graph of 7 nodes"),),
            (Described(adjacency_from_edges(SMALL_EDGES, 5), "a one way graph of 5 nodes"),),
        ),
        reference=reference_gcn_matrix,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="gcn_layer",
        description="mixes by the matrix, applies the weight, then a ReLU",
        inputs=(
            (torch.randn(7, 3, generator=torch.Generator().manual_seed(748)), Described(reference_gcn_matrix(GRAPH_A), "gcn_matrix of a graph of 7 nodes"), torch.randn(3, 4, generator=torch.Generator().manual_seed(749))),
        ),
        reference=lambda h, normalized, weight: torch.relu(normalized @ h @ weight),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="propagate",
        description="applies the matrix layers times",
        inputs=(
            (torch.randn(7, 3, generator=torch.Generator().manual_seed(750)), Described(reference_gcn_matrix(GRAPH_A), "gcn_matrix of a graph of 7 nodes"), 3),
            (torch.randn(9, 2, generator=torch.Generator().manual_seed(751)), Described(reference_gcn_matrix(GRAPH_B), "gcn_matrix of a graph of 9 nodes"), 0),
        ),
        reference=reference_propagate,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="mean_cosine",
        description="averages the cosine over every pair of different nodes",
        inputs=(
            (torch.randn(6, 4, generator=torch.Generator().manual_seed(752)),),
            (torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),),
        ),
        reference=reference_mean_cosine,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


MESSAGE2 = Exercise(
    name="message2",
    chapter=GRAPHS,
    title="The GCN, and oversmoothing",
    lesson=MESSAGE2_LESSON,
    source=MESSAGE2_SOURCE,
    solution=MESSAGE2_SOLUTION,
    hints={
        "gcn_matrix": "torch.eye(len(adjacency)) adds the self loops. Raise the row sums to -0.5, then multiply the "
        "matrix by them as a column with .unsqueeze(1) and as a row with .unsqueeze(0).",
        "gcn_layer": "normalized @ h mixes the nodes, @ weight changes the features, and torch.relu comes last.",
        "propagate": "A loop of layers steps, each setting h to normalized @ h.",
        "mean_cosine": "F.normalize the rows, multiply by their transpose, take away the diagonal of 1s from the sum, "
        "and divide by nodes * (nodes - 1) pairs.",
    },
    introduces=(),
    check=check_message2,
)


# ---------------------------------------------------------------------------
# message3
# ---------------------------------------------------------------------------


def reference_relax(
    distances: Tensor,
    edges: Tensor,
    lengths: Tensor,
) -> Tensor:
    return distances.scatter_reduce(0, edges[1], distances[edges[0]] + lengths, reduce="amin")


def starting_distances(
    source: int,
    nodes: int,
) -> Tensor:
    distances = torch.full((nodes,), float("inf"))
    distances[source] = 0.0

    return distances


def reference_bellman_ford(
    source: int,
    edges: Tensor,
    lengths: Tensor,
    nodes: int,
) -> Tensor:
    distances = starting_distances(source, nodes)

    for _ in range(nodes - 1):
        distances = reference_relax(distances, edges, lengths)

    return distances


def reference_rounds_needed(
    source: int,
    edges: Tensor,
    lengths: Tensor,
    nodes: int,
) -> int:
    distances = starting_distances(source, nodes)
    rounds = 0

    while True:
        relaxed = reference_relax(distances, edges, lengths)

        if torch.equal(relaxed, distances):
            return rounds

        distances = relaxed
        rounds += 1


MESSAGE3_LESSON = prose(
    """
    Bellman-Ford finds the shortest distance from one node to every other.
    Every node starts at infinity, and the source at 0. In each round, an edge
    from u to v with length w offers v the distance `d_u + w`. Each node keeps
    the smallest of its offers and its own.

    That round is a layer of message passing. The message is `d_u + w`, and
    the nodes combine their messages by a minimum where message1 took a sum.
    After k rounds every node knows its shortest distance over paths of at
    most k edges. So a graph whose shortest paths take at most k edges needs k
    rounds, as k layers reach k edges away. No path needs more than nodes - 1
    edges, so that many rounds always do.

    Here it is on a random graph of 30 nodes with lengths from 1 to 9, from
    node 0.

    | Rounds | 1 | 2 | 4 | 8 |
    |---|---|---|---|---|
    | Nodes with the right distance | 5 | 11 | 21 | 30 |

    Nothing changed after 7 rounds here, far fewer than 29.

    Xu and colleagues call this algorithmic alignment. A network whose layers
    match an algorithm's steps only has to learn one small step, here a sum
    and a minimum. They proved that such a network needs fewer examples to
    learn the task, and 75_shortest_paths measures it.

    `scatter_reduce` gathers by index like `index_add_`, and can take the
    minimum instead of the sum. By default the value already there joins in.

    ```python
    start = torch.tensor([9.0, 9.0, 9.0])
    start.scatter_reduce(0, torch.tensor([0, 2, 0]), torch.tensor([4.0, 1.0, 7.0]), reduce="amin")
    # tensor([4., 9., 1.])
    ```
    """,
)

MESSAGE3_SOURCE = source_file(
    '''
    # message3.py

    import torch
    from torch import Tensor

    start = torch.tensor([9.0, 9.0, 9.0])
    lowest = start.scatter_reduce(0, torch.tensor([0, 2, 0]), torch.tensor([4.0, 1.0, 7.0]), reduce="amin")
    # tensor([4., 9., 1.])


    # TODO: Return the distances after one round of Bellman-Ford. Each edge from
    #   u to v, of length lengths at its column, offers v distances[u] plus its
    #   length, and each node keeps the smallest of its offers and its own.
    def relax(
        distances: Tensor,
        edges: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the shortest distance from source to every node of a graph of
    #   nodes nodes, as floats. Start every node at float("inf") and source at
    #   0, then run nodes - 1 rounds of relax.
    def bellman_ford(
        source: int,
        edges: Tensor,
        lengths: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return how many rounds of relax change something, counted from the
    #   same start, before a round changes nothing.
    def rounds_needed(
        source: int,
        edges: Tensor,
        lengths: Tensor,
        nodes: int,
    ) -> int:
        ...
    ''',
)

MESSAGE3_SOLUTION = source_file(
    '''
    # message3.py, a solution for comparison

    import torch
    from torch import Tensor

    start = torch.tensor([9.0, 9.0, 9.0])
    lowest = start.scatter_reduce(0, torch.tensor([0, 2, 0]), torch.tensor([4.0, 1.0, 7.0]), reduce="amin")


    def relax(
        distances: Tensor,
        edges: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        offers = distances[edges[0]] + lengths

        return distances.scatter_reduce(0, edges[1], offers, reduce="amin")


    def bellman_ford(
        source: int,
        edges: Tensor,
        lengths: Tensor,
        nodes: int,
    ) -> Tensor:
        distances = torch.full((nodes,), float("inf"))
        distances[source] = 0.0

        for _ in range(nodes - 1):
            distances = relax(distances, edges, lengths)

        return distances


    def rounds_needed(
        source: int,
        edges: Tensor,
        lengths: Tensor,
        nodes: int,
    ) -> int:
        distances = torch.full((nodes,), float("inf"))
        distances[source] = 0.0
        rounds = 0

        while True:
            relaxed = relax(distances, edges, lengths)

            if torch.equal(relaxed, distances):
                return rounds

            distances = relaxed
            rounds += 1
    ''',
)

PATH_EDGES = random_graph(12, 6, torch.Generator().manual_seed(753))
PATH_LENGTHS = random_lengths(PATH_EDGES, torch.Generator().manual_seed(754))
LINE_EDGES = both_ways(torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
LINE_LENGTHS = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0])


def check_message3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="relax",
        description="keeps each node's smallest offer or its own",
        inputs=(
            (torch.tensor([0.0, float("inf"), 4.0, float("inf"), 2.0, float("inf")]), Described(LINE_EDGES, "a line of 6 nodes, both ways"), LINE_LENGTHS),
            (torch.tensor([3.0, 0.0, 9.0, 1.0, 7.0, 2.0, 8.0, 5.0, 6.0, 4.0, 9.0, 9.0]), Described(PATH_EDGES, "a graph of 12 nodes"), PATH_LENGTHS),
        ),
        reference=reference_relax,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="bellman_ford",
        description="finds every shortest distance from the source",
        inputs=(
            (0, Described(LINE_EDGES, "a line of 6 nodes, both ways"), LINE_LENGTHS, 6),
            (5, Described(PATH_EDGES, "a graph of 12 nodes"), PATH_LENGTHS, 12),
            (2, Described(SMALL_EDGES, "a one way graph of 5 nodes"), torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0]), 5),
        ),
        reference=reference_bellman_ford,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="rounds_needed",
        description="counts the rounds that change something",
        inputs=(
            (0, Described(LINE_EDGES, "a line of 6 nodes, both ways"), LINE_LENGTHS, 6),
            (5, Described(PATH_EDGES, "a graph of 12 nodes"), PATH_LENGTHS, 12),
            (3, Described(LINE_EDGES, "a line of 6 nodes, both ways"), LINE_LENGTHS, 6),
        ),
        reference=reference_rounds_needed,
        expect=expect_value,
    )


MESSAGE3 = Exercise(
    name="message3",
    chapter=GRAPHS,
    title="Why message passing fits graph algorithms",
    lesson=MESSAGE3_LESSON,
    source=MESSAGE3_SOURCE,
    solution=MESSAGE3_SOLUTION,
    hints={
        "relax": "distances[edges[0]] + lengths gives every edge's offer. scatter_reduce them into distances at the "
        "targets, edges[1], with reduce=\"amin\".",
        "bellman_ford": "torch.full((nodes,), float(\"inf\")) starts every node, then set the source to 0.0. Loop "
        "nodes - 1 times, setting the distances to relax of them.",
        "rounds_needed": "Count up while relax gives something new, and stop when torch.equal says the round changed "
        "nothing.",
    },
    introduces=(),
    check=check_message3,
)
