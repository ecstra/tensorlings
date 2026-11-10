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
    reject_missing_return,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import pinned_threads

GRAPHS = Chapter(
    folder="73_graphs",
    title="Graphs and message passing",
    readme=source_file(
        """
        # Graphs and message passing

        A graph is a set of nodes with edges between them. Like the elements
        of a set, its nodes have no order. Its edges say which nodes affect
        which. This chapter shows two ways to store a graph. Then it builds
        message passing, the layer every graph network is made of, and
        normalizes it the way the GCN does. Last, it shows why message
        passing suits algorithms on graphs, such as finding shortest paths.

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
    of a molecule and its bonds, or towns and their roads. Like the elements
    of a set, the nodes have no order. The edges add something a set does
    not have. They say which node affects which.

    There are two common ways to store the edges.

    - An adjacency matrix has a row and a column per node. Entry `[s, t]` is 1 if an edge runs from s to t.
    - An edge list has one column per edge. Row 0 holds the sources and row 1 the targets, as in PyTorch Geometric.

    ```python
    edges = torch.tensor([[0, 1, 1], [1, 2, 0]])    # edges from 0 to 1, from 1 to 2 and from 1 to 0
    ```

    The matrix always holds `nodes ** 2` numbers, however few edges there
    are. The list holds 2 numbers per edge. A million towns joined by 2
    million roads need a matrix of a million million entries. As a list,
    with each road stored in both directions, they need 8 million numbers.

    Some edges go both ways, like a road. Others go one way, like one person
    following another online. A graph whose edges all go both ways is
    undirected. Its edge list holds each edge in both directions, so its
    adjacency matrix is symmetric. The degree of a node is the number of
    edges that arrive at it.

    Indexing with two tensors sets many entries at once. Each pair of
    matching entries, one from each tensor, gives the row and the column of
    one entry to set.

    ```python
    grid = torch.zeros(2, 3)
    grid[torch.tensor([0, 1]), torch.tensor([2, 0])] = 1.0    # tensor([[0., 0., 1.], [1., 0., 0.]])
    ```

    `.nonzero()` goes the other way. It returns the position of every entry
    that is not 0, one row per entry, in reading order. Reading order means
    row by row, and left to right within a row.
    """,
)

GRAPH1_SOURCE = source_file(
    '''
    # graph1.py

    import torch
    from torch import Tensor

    grid = torch.zeros(2, 3)
    grid[torch.tensor([0, 1]), torch.tensor([2, 0])] = 1.0    # tensor([[0., 0., 1.], [1., 0., 0.]])
    where = grid.nonzero()                                   # tensor([[0, 2], [1, 0]]), the row and column of each 1


    # TODO: Return the adjacency matrix of a graph with nodes nodes, as a float
    #   tensor of shape (nodes, nodes). Put a 1 at row s and column t for
    #   every edge from s to t, and 0 everywhere else. edges has shape
    #   (2, edges), with sources in row 0 and targets in row 1.
    #   Example: adjacency_from_edges(torch.tensor([[0, 1], [1, 2]]), 3)
    #   returns tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]).
    def adjacency_from_edges(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return the edge list of the graph in adjacency, with shape
    #   (2, edges), sources in row 0 and targets in row 1. Order the edges by
    #   source, and edges with the same source by target.
    #   Example: for the matrix in the example above, edges_from_adjacency
    #   returns tensor([[0, 1], [1, 2]]).
    def edges_from_adjacency(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return the number of edges that arrive at each node, as a tensor
    #   with one count for each of the nodes nodes. A node that no edge
    #   reaches gets a count of 0.
    #   Example: in_degrees(torch.tensor([[0, 1, 2], [1, 2, 1]]), 3) returns
    #   tensor([0, 2, 1]).
    def in_degrees(
        edges: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Return edges followed by every edge reversed, so that each edge
    #   appears in both directions. An edge from s to t reversed is an edge
    #   from t to s. The result has twice as many columns as edges.
    #   Example: both_ways(torch.tensor([[0, 1], [1, 2]])) returns
    #   tensor([[0, 1, 1, 2], [1, 2, 0, 1]]).
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
        raise Mismatch(f"{call} holds {actual.dtype} numbers, but it should hold floats, so that the matrix can be multiplied with float features.")


def check_graph1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="adjacency_from_edges",
        description="puts a 1 at the row and column of each edge",
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
        description="adds every edge reversed",
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
        "adjacency_from_edges": "Start from torch.zeros(nodes, nodes). Set the entry of every edge to 1.0 in one "
        "assignment, with edges[0] and edges[1] as the two index tensors.",
        "edges_from_adjacency": ".nonzero() gives one row per edge, holding its source and its target. Transpose the "
        "result with .T to get the two rows of an edge list.",
        "in_degrees": "Use torch.bincount on the targets, edges[1]. Pass minlength=nodes so that a node with no "
        "incoming edges still gets a count of 0.",
        "both_ways": "edges.flip(0) swaps the two rows of edges, which reverses every edge. Join it after edges with "
        "torch.cat along dim=1.",
    },
    introduces=(),
    check=check_graph1,
)


# ---------------------------------------------------------------------------
# message1
# ---------------------------------------------------------------------------

MESSAGE1_LESSON = prose(
    """
    Message passing, as Gilmer and colleagues described it, updates the
    features of every node using its neighbours. Each edge carries a message
    from its source node to its target node. Each node adds up the messages
    that arrive at it. Then it computes its new features from that sum and
    its own current features.

    ```text
    m_v  = sum over edges w to v of M(h_v, h_w, e_vw)
    h_v' = U(h_v, m_v)
    ```

    h_v is the feature vector of node v. e_vw holds the features of the edge
    itself, if it has any, like the type of a chemical bond. In this file the
    message function M is a Linear of the source node's features only. The
    update function U is a Linear of the node's features and its sum of
    messages side by side, followed by a ReLU.

    `index_add_` adds up messages by target. It adds each row of one tensor
    into the row of another tensor that the index names. When several
    messages have the same target, they are all added into that node's row.

    ```python
    totals = torch.zeros(3, 2)
    totals.index_add_(0, torch.tensor([2, 0, 2]), torch.ones(3, 2))    # totals now has rows [1, 1], [0, 0], [2, 2]
    ```

    The same sums also come from one matrix product, the transpose of the
    adjacency matrix times the feature matrix. But the product does work for
    every pair of nodes, while the edge list does work only for each edge. So
    big graphs with few edges use the list.

    One layer lets each node receive information from nodes one edge away.
    With k layers, information reaches nodes up to k edges away. Like an
    equivariant set layer, message passing does not depend on how the nodes
    are numbered. When the 20 nodes of a random graph were renumbered here,
    the outputs came out renumbered in the same way. Every number was exactly
    the same.
    """,
)

MESSAGE1_SOURCE = source_file(
    '''
    # message1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    totals = torch.zeros(3, 2)
    totals.index_add_(0, torch.tensor([2, 0, 2]), torch.ones(3, 2))    # totals now has rows [1, 1], [0, 0], [2, 2]


    # TODO: Return the sum of the messages that arrive at each node, as a
    #   tensor of shape (nodes, features). messages has shape (edges,
    #   features), one row per edge. targets has one entry per edge, the node
    #   that edge's message goes to. A node with no messages gets zeros.
    #   Example: aggregate(torch.ones(3, 2), torch.tensor([2, 0, 2]), 3)
    #   returns tensor([[1., 1.], [0., 0.], [2., 2.]]).
    def aggregate(
        messages: Tensor,
        targets: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: For every node, return the sum of the features of all the nodes
    #   with an edge into it. h has shape (nodes, features). The result is
    #   what aggregate(h[sources], targets, nodes) would give, but compute it
    #   from the adjacency matrix with a matrix product, not from an edge list.
    def aggregate_by_matrix(
        h: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. h has shape (nodes, features) and edges has shape
    #   (2, edges). The message along each edge is self.message applied to the
    #   source node's row of h. Add up the messages arriving at each node with
    #   aggregate. Each node's new features are the ReLU of self.update applied
    #   to its own features and that sum, joined side by side.
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
        description="sends messages, adds them up and updates each node",
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
        "aggregate_by_matrix": "Row t of the transposed adjacency matrix has a 1 in the column of every node with an "
        "edge into t. So multiplying the transpose by h adds up exactly those nodes' features.",
        "MessagePassing.forward": "h[edges[0]] gives the source node's row for each edge. Pass self.message of that to "
        "aggregate, with edges[1] and len(h). Join h and the sum with torch.cat on dim=1, then apply self.update and "
        "torch.relu.",
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
    Kipf and Welling's GCN, short for graph convolutional network, is message
    passing with a fixed rule for scaling the messages. First each node gets
    an edge to itself, which turns A into `A + I`. Then the message from u to
    v is divided by the square root of u's degree and by the square root of
    v's degree. These degrees count the self loop too.

    ```text
    H' = ReLU(D^(-1/2) (A + I) D^(-1/2) H W)
    ```

    A is the adjacency matrix, and I is the identity matrix, which adds the
    self loops. D is a diagonal matrix that holds each node's degree. H holds
    the features, one row per node, and W is a weight matrix. Kipf and
    Welling wrote this for undirected graphs, which list each edge in both
    directions. Their matrix is symmetric, so the sum of a node's row equals
    its degree as graph1 counted it.

    Without the division, a node with 100 neighbours adds up 100 messages and
    a node with 2 neighbours adds up 2. So the features grow with the degree,
    and stacking layers makes them grow without limit.

    The paper's first version divided first and added each node's own
    features afterwards. That gives the matrix `I + D^(-1/2) A D^(-1/2)`. It
    can double the length of a vector, so stacked layers still blow up the
    features. Adding the self loop before dividing prevents this. The paper
    calls this the renormalization trick.

    Each layer replaces a node's features with something close to an average
    of its own and its neighbours' features. Stack many layers, and the
    features of every node become nearly the same. This is called
    oversmoothing. The table repeats the GCN's matrix with no weights on a
    random graph of 40 nodes, starting from random features. It gives the
    average cosine similarity between the features of every two nodes.

    | Layers | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
    |---|---|---|---|---|---|---|---|
    | Average cosine similarity | -0.001 | 0.085 | 0.176 | 0.344 | 0.626 | 0.895 | 0.991 |

    With 0 layers, the random features are unrelated. After enough layers,
    every node's feature vector points in the same direction, so no
    classifier can tell the nodes apart. Kipf and Welling got their best
    results with 2 or 3 layers, and their main model has 2.

    Here is how the overall size of the features changed over 8 layers on the
    same graph. With no division it grew 20,964 times, and with the first
    version's matrix it grew 62 times. With the GCN's matrix, it shrank to
    0.21 times its starting size.

    Cosine similarity compares the directions of two vectors and ignores
    their lengths. `F.normalize(h, dim=1)` scales every row to a length of 1.
    The dot product of two such rows is the cosine similarity of the original
    rows.
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


    # TODO: Return the GCN's matrix for adjacency, of shape (nodes, nodes).
    #   First add the identity matrix. Then divide the entry at row v and
    #   column u by the square root of v's degree and by the square root of
    #   u's degree. Here a node's degree is the sum of its row after the
    #   identity is added.
    #   Example: gcn_matrix(torch.tensor([[0.0, 1.0], [1.0, 0.0]])) returns
    #   tensor([[0.5000, 0.5000], [0.5000, 0.5000]]).
    def gcn_matrix(adjacency: Tensor) -> Tensor:
        ...


    # TODO: Return the output of one GCN layer. That is the ReLU of the matrix
    #   product of normalized, h and weight, in that order.
    def gcn_layer(
        h: Tensor,
        normalized: Tensor,
        weight: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return h after multiplying it on the left by normalized, layers
    #   times in a row, with no weights and no ReLU. When layers is 0, return
    #   h unchanged.
    def propagate(
        h: Tensor,
        normalized: Tensor,
        layers: int,
    ) -> Tensor:
        ...


    # TODO: Each row of h holds the features of one node. Return the average
    #   cosine similarity over every pair of two different nodes, as a Python
    #   float.
    #   Example: mean_cosine(torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]]))
    #   returns about 0.3333.
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
            (Described(GRAPH_B, "a graph of 9 nodes"),),
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
        description="multiplies by the matrix layers times",
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
        "gcn_matrix": "Add torch.eye(len(adjacency)) to get the self loops. Raise the row sums to the power -0.5. "
        "Multiply the matrix by them once as a column with .unsqueeze(1) and once as a row with .unsqueeze(0).",
        "gcn_layer": "normalized @ h mixes each node with its neighbours. Multiplying that by weight with @ changes "
        "the features, and torch.relu comes last.",
        "propagate": "Write a loop that runs layers times and sets h to normalized @ h each time.",
        "mean_cosine": "F.normalize the rows and multiply the result by its transpose to get the cosine of every pair. "
        "The diagonal pairs each node with itself and holds 1s. Subtract its sum from the total, then divide by "
        "nodes * (nodes - 1), the number of pairs of different nodes.",
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
    The Bellman-Ford algorithm finds the shortest distance from one source
    node to every other node. Each node holds a current distance. It starts
    at infinity, except at the source, where it starts at 0. In each round,
    every edge from u to v with length w offers v the distance `d_u + w`.
    Each node then keeps the smallest of its offers and its current distance.

    One round is a layer of message passing. The message is `d_u + w`, and
    each node combines its messages with a minimum instead of the sum that
    message1 used. After k rounds, every node knows its shortest distance
    over paths of at most k edges. So if every shortest path in a graph has
    at most k edges, k rounds are enough. This matches how k layers reach k
    edges away. No shortest path needs more than `nodes - 1` edges, so `nodes - 1`
    rounds are always enough.

    Here is Bellman-Ford from node 0 on a random graph of 30 nodes, with edge
    lengths from 1 to 9. The table counts the nodes whose distance is already
    correct after each number of rounds.

    | Rounds | 1 | 2 | 4 | 8 |
    |---|---|---|---|---|
    | Nodes with the right distance | 5 | 11 | 21 | 30 |

    On this graph no distance changed after round 7. That is far fewer than
    the 29 rounds that are always enough.

    Xu and colleagues call this match algorithmic alignment. When a network's
    layers match the steps of an algorithm, the network only has to learn one
    small step, here a sum and a minimum. For networks trained one step at a
    time, they proved a bound on how many training examples are needed. The
    closer the match, the smaller the bound. 75_shortest_paths measures this
    with networks trained on only 8 graphs.

    `scatter_reduce` combines values by index like `index_add_`, but it can
    take the minimum instead of the sum. By default, the value already in the
    tensor takes part in the minimum. Without an underscore at the end,
    `scatter_reduce` returns a new tensor and leaves `start` unchanged.
    `torch.equal(a, b)` returns True when two tensors hold exactly the same
    numbers.

    ```python
    start = torch.tensor([9.0, 9.0, 9.0])
    start.scatter_reduce(0, torch.tensor([0, 2, 0]), torch.tensor([4.0, 1.0, 7.0]), reduce="amin")
    # tensor([4., 9., 1.])
    torch.equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))    # True
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
    same = torch.equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))    # True


    # TODO: Return the distances after one round of Bellman-Ford. lengths holds
    #   one length per edge, in the same order as the columns of edges. Each
    #   edge from u to v offers v the distance distances[u] plus its length.
    #   Each node keeps the smallest of its offers and its current distance.
    #   Return a new tensor and leave distances unchanged.
    #   Example: with distances = torch.tensor([0.0, float("inf"), float("inf")]),
    #   edges = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]) and
    #   lengths = torch.tensor([4.0, 1.0, 4.0, 1.0]), relax returns
    #   tensor([0., 4., inf]).
    def relax(
        distances: Tensor,
        edges: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return the shortest distance from source to every node, as a float
    #   tensor with one entry for each of the nodes nodes. Start every node at
    #   float("inf") and the source at 0, then run nodes - 1 rounds of relax.
    #   Example: on the graph in the example above, from source 0, bellman_ford
    #   returns tensor([0., 4., 5.]).
    def bellman_ford(
        source: int,
        edges: Tensor,
        lengths: Tensor,
        nodes: int,
    ) -> Tensor:
        ...


    # TODO: Start from the same distances as bellman_ford and run relax again
    #   and again. Return how many rounds changed at least one distance before
    #   the first round that changes nothing.
    #   Example: on the same graph from source 0, rounds_needed returns 2.
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
    same = torch.equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))


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


def check_relax_leaves_input(found: object) -> None:
    """check_on_inputs hands each call a fresh copy, so only a call on a kept tensor shows relax changing it."""
    distances = torch.tensor([0.0, float("inf"), 4.0, float("inf"), 2.0, float("inf")])
    before = distances.clone()
    reject_missing_return(
        call="relax(distances, a line of 6 nodes, its lengths)",
        actual=found(distances, LINE_EDGES, LINE_LENGTHS),
    )

    if not torch.equal(distances, before):
        raise Mismatch(
            "relax changed the distances it was given. scatter_reduce_, with the underscore, changes the tensor in "
            "place, as index_add_ does. Use scatter_reduce, which returns a new tensor. Otherwise rounds_needed "
            "cannot compare one round with the next.",
        )


def check_message3(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="relax",
        description="keeps the smallest of each node's offers and its distance",
        inputs=(
            (torch.tensor([0.0, float("inf"), 4.0, float("inf"), 2.0, float("inf")]), Described(LINE_EDGES, "a line of 6 nodes, both ways"), LINE_LENGTHS),
            (torch.tensor([3.0, 0.0, 9.0, 1.0, 7.0, 2.0, 8.0, 5.0, 6.0, 4.0, 9.0, 9.0]), Described(PATH_EDGES, "a graph of 12 nodes"), PATH_LENGTHS),
        ),
        reference=reference_relax,
        expect=expect_tensor,
    )
    checker.test(
        function="relax",
        description="leaves the distances it was given alone",
        run=check_relax_leaves_input,
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
        "relax": "distances[edges[0]] + lengths gives every edge's offer. Use distances.scatter_reduce to combine "
        "the offers at the targets, edges[1], with reduce=\"amin\".",
        "bellman_ford": "torch.full((nodes,), float(\"inf\")) starts every node at infinity. Then set the source to "
        "0.0. Loop nodes - 1 times, each time replacing the distances with relax of them.",
        "rounds_needed": "Count the rounds while relax keeps returning new distances. Stop when torch.equal shows that "
        "a round changed nothing.",
    },
    introduces=(),
    check=check_message3,
)


# ---------------------------------------------------------------------------
# The runs behind message2's and message3's numbers
# ---------------------------------------------------------------------------


def measure_oversmoothing() -> dict[str, object]:
    """message2's numbers, on a random graph of 40 nodes with 20 chords and 16 random features per node. The average
    cosine after each count of GCN layers with no weights, and how much the features' size grows over 8 layers with
    no division, with the paper's first matrix, and with the GCN's, with the largest eigenvalue of each matrix."""
    adjacency = adjacency_from_edges(random_graph(40, 20, torch.Generator().manual_seed(3)), 40)
    h = torch.randn(40, 16, generator=torch.Generator().manual_seed(4))
    inverse_root = adjacency.sum(dim=1) ** -0.5
    matrices = {
        "no division": adjacency + torch.eye(40),
        "first version": torch.eye(40) + inverse_root.unsqueeze(1) * adjacency * inverse_root.unsqueeze(0),
        "GCN": reference_gcn_matrix(adjacency),
    }

    with pinned_threads():
        return {
            "cosine": {layers: reference_mean_cosine(reference_propagate(h, matrices["GCN"], layers)) for layers in (0, 1, 2, 4, 8, 16, 32)},
            "growth over 8 layers": {name: (reference_propagate(h, matrix, 8).norm() / h.norm()).item() for name, matrix in matrices.items()},
            "largest eigenvalue": {name: torch.linalg.eigvalsh(matrix).max().item() for name, matrix in matrices.items()},
        }


def measure_rounds() -> dict[str, object]:
    """message3's table, on a random graph of 30 nodes with 11 chords and lengths from 1 to 9, from node 0. The
    nodes with their final distance after each round, and the rounds that change something."""
    edges = random_graph(30, 11, torch.Generator().manual_seed(5))
    lengths = random_lengths(edges, torch.Generator().manual_seed(8))
    truth = reference_bellman_ford(0, edges, lengths, 30)
    distances = starting_distances(0, 30)
    right = {}

    for rounds in range(1, 9):
        distances = reference_relax(distances, edges, lengths)
        right[rounds] = int((distances == truth).sum())

    return {
        "right after each round": right,
        "rounds needed": reference_rounds_needed(0, edges, lengths, 30),
    }
