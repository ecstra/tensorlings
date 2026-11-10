"""72_sets, models whose answer can't depend on the order of their input: Deep Sets, why a flat MLP fails on shuffled sets, and layers that reorder their output with their input."""

from __future__ import annotations

import math

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
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module
from .tensors import pinned_threads

SETS = Chapter(
    folder="72_sets",
    title="Sets",
    readme=source_file(
        """
        # Sets

        Section 10 is about building a problem's symmetries into the model.
        This chapter covers the simplest one. A set has no order, so a model
        of a set should give the same answer in whatever order its elements
        are listed. The chapter builds Deep Sets and shows a flat MLP failing
        on shuffled sets. Then it builds layers whose output is reordered in
        the same way when their input is reordered. Attention is one of them.

        ## Further reading

        - [Zaheer and colleagues, Deep Sets](https://arxiv.org/abs/1703.06114)
        - [Lee and colleagues, Set Transformer](https://arxiv.org/abs/1810.00825)
        """,
    ),
)

ELEMENTS = 8


def reference_pool(
    features: Tensor,
    how: str,
) -> Tensor:
    if how == "max":
        return features.amax(dim=1)

    if how == "mean":
        return features.mean(dim=1)

    return features.sum(dim=1)


class ReferenceDeepSet(nn.Module):
    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.rho(self.phi(x).sum(dim=1))


class FlatLinear(nn.Module):
    """A Linear on the flattened list, the order-bound model sets1's lesson compares against."""

    def __init__(
        self,
        elements: int,
        features: int,
        outputs: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(elements * features, outputs)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.flatten(1))


def reference_order_gap(
    model: nn.Module,
    x: Tensor,
    order: Tensor,
) -> float:
    with torch.no_grad():
        return (model(x) - model(x[:, order])).abs().max().item()


def seeded_model(
    make: type,
    *arguments: int,
) -> nn.Module:
    torch.manual_seed(0)

    return make(*arguments)


# ---------------------------------------------------------------------------
# sets1
# ---------------------------------------------------------------------------

SETS1_LESSON = prose(
    """
    Some inputs are sets, like the points of a shape, the items in a basket
    or the players of a team. They arrive as a list, but the order of the
    list means nothing. A model of a set should give the same answer for
    every order of the same elements.

    A Linear layer on the flattened list cannot do that. Each position in
    the list has its own weights. So when an element moves to another
    position, it is multiplied by different weights and the answer changes.

    Zaheer and colleagues proved that every function that ignores order can
    be written in one form, called Deep Sets. Their proof covers elements
    from a countable set, one whose members you could number 0, 1, 2 and so
    on, like words. A Deep Set has three steps.

    - phi is a small network that runs on each element separately.
    - A sum over the elements combines them into one vector. This step is called pooling.
    - rho is a second network that runs on the pooled vector and gives the answer.

    phi can be made of ordinary Linear layers. A Linear on a tensor of shape
    `(batch, elements, features)` runs on each element separately, because
    it acts only on the last dimension. And a sum gives the same result in
    any order, so the answer ignores the order of the elements.

    A mean or a max also ignores order, so either one can do the pooling.
    But the mean forgets how many elements there were, and the max keeps
    only the largest value of each feature.

    ```python
    torch.tensor([[2.0], [2.0]]).mean(dim=0), torch.tensor([[2.0]]).mean(dim=0)    # tensor([2.]) both times
    torch.tensor([[2.0], [2.0]]).sum(dim=0), torch.tensor([[2.0]]).sum(dim=0)      # tensor([4.]) and tensor([2.])
    ```

    `x[:, order]` takes the elements of every set in a new order. The table
    shows how much each untrained model's answer changes when 4 random sets
    are reordered. Each set has 8 elements with 3 features each.

    | Model | Largest change in the answer |
    |---|---|
    | A Linear on the flattened list | 1.53 |
    | A Deep Set | 0.00000006 |

    The Deep Set's answer changes only by rounding error. Adding floats in a
    different order rounds them a little differently.
    """,
)

SETS1_SOURCE = source_file(
    '''
    # sets1.py

    import torch
    import torch.nn as nn
    from torch import Tensor

    features = torch.ones(2, 3, 4)            # 2 sets of 3 elements, each with 4 features
    biggest = features.amax(dim=1)            # shape (2, 4), the largest of each feature in each set


    # TODO: Pool the elements of each set into one row. features has shape
    #   (batch, elements, features). how is "sum", "mean" or "max", and says
    #   whether to take the sum, the mean or the max over the elements. The
    #   result has shape (batch, features).
    #   Example: for features = torch.tensor([[[1.0, 5.0], [3.0, 2.0]]]),
    #   pool(features, "max") returns tensor([[3., 5.]]).
    def pool(
        features: Tensor,
        how: str,
    ) -> Tensor:
        ...


    # TODO: Finish forward. x has shape (batch, elements, inputs). Run phi on
    #   every element of x, sum the results over the elements, and run rho on
    #   that sum. The result has shape (batch, outputs).
    class DeepSet(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            outputs: int,
        ) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, outputs),
            )

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Measure how much reordering the elements changes the model's
    #   answer. Run model on x, and on x with every set's elements taken in
    #   the order given by order, a tensor of positions like tensor([2, 0, 1]).
    #   Return the largest absolute difference between the two answers as a
    #   Python float. Do not let autograd record the computation.
    def order_gap(
        model: nn.Module,
        x: Tensor,
        order: Tensor,
    ) -> float:
        ...
    ''',
)

SETS1_SOLUTION = source_file(
    '''
    # sets1.py, a solution for comparison

    import torch
    import torch.nn as nn
    from torch import Tensor

    features = torch.ones(2, 3, 4)
    biggest = features.amax(dim=1)


    def pool(
        features: Tensor,
        how: str,
    ) -> Tensor:
        if how == "max":
            return features.amax(dim=1)

        if how == "mean":
            return features.mean(dim=1)

        return features.sum(dim=1)


    class DeepSet(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            outputs: int,
        ) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, outputs),
            )

        def forward(self, x: Tensor) -> Tensor:
            pooled = self.phi(x).sum(dim=1)

            return self.rho(pooled)


    def order_gap(
        model: nn.Module,
        x: Tensor,
        order: Tensor,
    ) -> float:
        with torch.no_grad():
            return (model(x) - model(x[:, order])).abs().max().item()
    ''',
)


def recording_hook(
    model: nn.Module,
    seen: list[bool],
) -> torch.utils.hooks.RemovableHandle:
    """Notes on every call whether autograd was recording, since a model's outputs can't show it."""
    return model.register_forward_pre_hook(lambda module, inputs: seen.append(torch.is_grad_enabled()))


def check_deep_set(found: type) -> None:
    model = expect_module("DeepSet(3, 8, 2)", seeded_model(found, 3, 8, 2))
    x = small_tensor(720, 4, 5, 3) / 3

    with torch.no_grad():
        expected = seeded_model(ReferenceDeepSet, 3, 8, 2)(x)

    expect_tensor(
        call="DeepSet(3, 8, 2) on 4 sets of 5 elements",
        actual=model(x),
        expected=expected,
    )


def check_order_gap(found: object) -> None:
    x = small_tensor(721, 4, 6, 2) / 3
    order = torch.tensor([3, 0, 5, 1, 4, 2])

    for name, model in (("a Linear on the flattened list", seeded_model(FlatLinear, 6, 2, 3)), ("a Deep Set", seeded_model(ReferenceDeepSet, 2, 8, 3))):
        seen: list[bool] = []
        hook = recording_hook(model, seen)

        try:
            actual = found(model, x, order)
        finally:
            hook.remove()

        if any(seen):
            raise Mismatch("order_gap recorded for autograd. Run the model inside torch.no_grad().")

        expect_close(
            call=f"order_gap of {name} on 4 sets of 6",
            actual=actual,
            expected=reference_order_gap(model, x, order),
            tolerance=1e-6,
        )


def check_sets1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="pool",
        description="pools each set by a sum, a mean or a max",
        inputs=tuple((small_tensor(722, 3, 4, 2), how) for how in ("sum", "mean", "max")),
        reference=reference_pool,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="DeepSet",
        method="forward",
        description="runs phi on each element, sums, then runs rho",
        run=check_deep_set,
    )
    checker.test(
        function="order_gap",
        description="measures how much reordering changes the answer",
        run=check_order_gap,
    )


SETS1 = Exercise(
    name="sets1",
    chapter=SETS,
    title="Deep Sets",
    lesson=SETS1_LESSON,
    source=SETS1_SOURCE,
    solution=SETS1_SOLUTION,
    hints={
        "pool": "Use one if for \"max\" with .amax(dim=1) and one for \"mean\" with .mean(dim=1). Use .sum(dim=1) "
        "for anything else.",
        "DeepSet.forward": "self.phi(x) still has one row per element. Sum it over dim=1, then pass the sum to "
        "self.rho.",
        "order_gap": "Inside torch.no_grad(), subtract model(x[:, order]) from model(x), then call .abs().max().item().",
    },
    introduces=(),
    check=check_sets1,
)


# ---------------------------------------------------------------------------
# sets2
# ---------------------------------------------------------------------------


def square_sets(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sets of 8 numbers from -1 to 1, one feature each, and the sum of their squares."""
    x = torch.rand(count, ELEMENTS, 1, generator=generator) * 2 - 1

    return x, (x**2).sum(dim=(1, 2))


def reference_sort_each(x: Tensor) -> Tensor:
    return x.sort(dim=1).values


def reference_shuffle_each(
    x: Tensor,
    generator: torch.Generator,
) -> Tensor:
    order = torch.rand(x.shape[0], x.shape[1], generator=generator).argsort(dim=1)
    rows = torch.arange(len(x)).unsqueeze(1)

    return x[rows, order]


class ReferenceFlatMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ELEMENTS, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1)).squeeze(1)


def reference_squared_error(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> float:
    with torch.no_grad():
        return F.mse_loss(model(x).flatten(), y).item()


def train_regressor(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
) -> nn.Module:
    """The run the lesson's table comes from, Adam at 0.003 on batches of 64."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(5)

    for _ in range(steps):
        batch = torch.randint(0, len(x), (64,), generator=generator)
        loss = F.mse_loss(model(x[batch]).flatten(), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


SETS2_LESSON = prose(
    """
    Here is a task on sets. Each set holds 8 numbers between -1 and 1. The
    answer is the sum of their squares, which is the same in any order. Three
    models were trained for 2,000 steps on the same 1,000 sets, then tested
    on 2,000 new sets. The table gives each model's mean squared error. The
    Deep Set is the one from sets1, with 32 hidden features.

    | Model | Trained on | Error, sorted tests | Error, shuffled tests |
    |---|---|---|---|
    | An MLP on the flat list | sorted sets | 0.0014 | 1.59 |
    | An MLP on the flat list | shuffled sets | 0.0086 | 0.025 |
    | A Deep Set | shuffled sets | 0.00015 | 0.00015 |

    A model that always guesses the average answer gets an error of 0.70.

    The MLP trained on sorted sets learned that position 0 always holds the
    smallest number. On shuffled sets that is no longer true, and the MLP
    does worse than always guessing the average. Sorting every input first
    would fix it for this task, but sorting needs an order to sort by.
    Points with many features, or the atoms of a molecule, have no natural
    order.

    The MLP trained on shuffled sets has to learn from examples that the
    order does not matter. That is hard, because 8 numbers can be ordered in
    40,320 ways. The Deep Set ignores order by design, so all of its training
    goes into the task itself. Over 5 seeds, the Deep Set's error on shuffled
    tests was 13 to 168 times smaller than this MLP's.

    `.argsort()` returns the positions that would sort a tensor. The argsort
    of random numbers is a random order. To give every set its own order,
    index with two tensors, as masked2 did for its patches. The first tensor
    gives the row of each element you pick, and the second gives its position
    within that row.

    ```python
    torch.tensor([0.7, 0.1, 0.4]).argsort()    # tensor([1, 2, 0])
    rows = torch.arange(2).unsqueeze(1)          # tensor([[0], [1]]), one row number per set
    ```
    """,
)

SETS2_SOURCE = source_file(
    '''
    # sets2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    positions = torch.tensor([0.7, 0.1, 0.4]).argsort()    # tensor([1, 2, 0])
    sets = torch.tensor([[10, 11, 12], [20, 21, 22]])
    picked = sets[torch.arange(2).unsqueeze(1), torch.tensor([[2, 0, 1], [1, 2, 0]])]
    # tensor([[12, 10, 11], [21, 22, 20]])


    # TODO: Return x, of shape (batch, elements, 1), with each set's elements
    #   sorted from smallest to largest.
    #   Example: sort_each(torch.tensor([[[3.0], [1.0], [2.0]]])) returns
    #   tensor([[[1.], [2.], [3.]]]).
    def sort_each(x: Tensor) -> Tensor:
        ...


    # TODO: Return x with the elements of each set in a different random
    #   order. Draw random numbers once with
    #   torch.rand(batch, elements, generator=generator), and argsort each row
    #   along the elements to get one order per set. Then pick the elements of
    #   x in those orders.
    def shuffle_each(
        x: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Finish forward. x has shape (batch, elements, 1). Flatten each set
    #   into one row of 8 numbers, run net on the rows, and return one answer
    #   per set, as a tensor of shape (batch,).
    class FlatMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return the mean squared error between model(x) and y as a Python
    #   float. Flatten the model's answers first, because some models return
    #   them as a column of shape (batch, 1). Do not let autograd record the
    #   computation.
    def squared_error(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        ...
    ''',
)

SETS2_SOLUTION = source_file(
    '''
    # sets2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    positions = torch.tensor([0.7, 0.1, 0.4]).argsort()
    sets = torch.tensor([[10, 11, 12], [20, 21, 22]])
    picked = sets[torch.arange(2).unsqueeze(1), torch.tensor([[2, 0, 1], [1, 2, 0]])]


    def sort_each(x: Tensor) -> Tensor:
        return x.sort(dim=1).values


    def shuffle_each(
        x: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        order = torch.rand(x.shape[0], x.shape[1], generator=generator).argsort(dim=1)
        rows = torch.arange(len(x)).unsqueeze(1)

        return x[rows, order]


    class FlatMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x.flatten(1)).squeeze(1)


    def squared_error(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            return F.mse_loss(model(x).flatten(), y).item()
    ''',
)


def check_flat_mlp(found: type) -> None:
    model = expect_module("FlatMLP()", seeded_model(found))
    x, _ = square_sets(5, torch.Generator().manual_seed(723))

    with torch.no_grad():
        expected = seeded_model(ReferenceFlatMLP)(x)

    expect_tensor(
        call="FlatMLP() on 5 sets of 8",
        actual=model(x),
        expected=expected,
    )


def check_squared_error(found: object) -> None:
    x, y = square_sets(30, torch.Generator().manual_seed(724))

    for name, model in (("an untrained FlatMLP", seeded_model(ReferenceFlatMLP)), ("an untrained Deep Set with a column of answers", seeded_model(ReferenceDeepSet, 1, 8, 1))):
        seen: list[bool] = []
        hook = recording_hook(model, seen)

        try:
            actual = found(model, x, y)
        finally:
            hook.remove()

        if any(seen):
            raise Mismatch("squared_error recorded for autograd. Run the model inside torch.no_grad().")

        expect_close(
            call=f"squared_error of {name} on 30 sets",
            actual=actual,
            expected=reference_squared_error(model, x, y),
            tolerance=1e-6,
        )


def check_sorted_mlp_fails(found: type) -> None:
    x, y = square_sets(1000, torch.Generator().manual_seed(0))
    test_x, test_y = square_sets(500, torch.Generator().manual_seed(1))
    model = train_regressor(seeded_model(found), reference_sort_each(x), y, 1000)
    on_sorted = reference_squared_error(model, reference_sort_each(test_x), test_y)
    on_shuffled = reference_squared_error(model, reference_shuffle_each(test_x, torch.Generator().manual_seed(2)), test_y)

    if on_sorted > 0.05 or on_shuffled < 0.5:
        raise Mismatch(
            f"Trained on sorted sets, FlatMLP should do well on sorted tests and badly on shuffled ones. Its error "
            f"is {on_sorted:.3f} on sorted tests and {on_shuffled:.3f} on shuffled tests. They should be below 0.05 "
            f"and above 0.5.",
        )


def check_sets2(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="sort_each",
        description="sorts the elements of each set",
        inputs=(
            (small_tensor(725, 3, 5, 1),),
            (small_tensor(726, 2, 4, 1),),
        ),
        reference=reference_sort_each,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="shuffle_each",
        description="gives each set its own random order",
        inputs=(
            (small_tensor(727, 3, 5, 1), seeded(728)),
            (small_tensor(729, 2, 4, 3), seeded(730)),
        ),
        reference=reference_shuffle_each,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="FlatMLP",
        method="forward",
        description="flattens each set and gives one answer per set",
        run=check_flat_mlp,
    )
    checker.test(
        function="squared_error",
        description="gives the mean squared error without recording",
        run=check_squared_error,
    )
    checker.test_method(
        class_name="FlatMLP",
        method="forward",
        description="trained on sorted sets, fails on shuffled ones",
        run=check_sorted_mlp_fails,
    )


SETS2 = Exercise(
    name="sets2",
    chapter=SETS,
    title="Why the order must not matter",
    lesson=SETS2_LESSON,
    source=SETS2_SOURCE,
    solution=SETS2_SOLUTION,
    hints={
        "sort_each": "x.sort(dim=1) sorts along the elements and returns both the values and the indices. Return "
        "the values.",
        "shuffle_each": "Argsort the random numbers along dim=1 to get one order per set. Index x with "
        "torch.arange(len(x)).unsqueeze(1) and those orders.",
        "FlatMLP.forward": "x.flatten(1) gives one row of 8 numbers per set. Run self.net on it, then call "
        ".squeeze(1).",
        "squared_error": "Inside torch.no_grad(), take F.mse_loss of model(x).flatten() and y, then call .item().",
    },
    introduces=(),
    check=check_sets2,
    time_limit=30,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# sets3
# ---------------------------------------------------------------------------


class ReferenceEquivariantLayer(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
    ) -> None:
        super().__init__()
        self.own = nn.Linear(inputs, outputs)
        self.pooled = nn.Linear(inputs, outputs, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.own(x) + self.pooled(x.mean(dim=1, keepdim=True))


def reference_set_attention(
    x: Tensor,
    Wq: Tensor,
    Wk: Tensor,
    Wv: Tensor,
) -> Tensor:
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    weights = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]), dim=-1)

    return weights @ v


class ReferenceEquivariantNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.first = ReferenceEquivariantLayer(1, hidden)
        self.second = ReferenceEquivariantLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.second(torch.relu(self.first(x))))).squeeze(-1)


class PerElementNet(nn.Module):
    """An MLP on each element alone, which never sees the rest of its set, for the lesson's centre task."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class AttentionNet(nn.Module):
    """set_attention over a Linear of each element, then an MLP on each element and its mix, for the centre task."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Linear(1, hidden)
        self.Wq = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.Wk = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.Wv = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embed(x)
        mixed = reference_set_attention(embedded, self.Wq, self.Wk, self.Wv)

        return self.head(torch.cat([embedded, mixed], dim=-1)).squeeze(-1)


def centre_sets(
    count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sets of 8 numbers from -1 to 1, one feature each, and each element's value minus its set's mean."""
    x = torch.rand(count, ELEMENTS, 1, generator=generator) * 2 - 1

    return x, x.squeeze(-1) - x.squeeze(-1).mean(dim=1, keepdim=True)


class AttentionLayer(nn.Module):
    """set_attention with its own weights, and positions added first when told to, for the lesson's table."""

    def __init__(
        self,
        features: int,
        elements: int,
        positions: bool,
    ) -> None:
        super().__init__()
        self.Wq = nn.Parameter(torch.randn(features, features) / math.sqrt(features))
        self.Wk = nn.Parameter(torch.randn(features, features) / math.sqrt(features))
        self.Wv = nn.Parameter(torch.randn(features, features) / math.sqrt(features))
        self.positions = nn.Parameter(torch.randn(elements, features)) if positions else None

    def forward(self, x: Tensor) -> Tensor:
        if self.positions is not None:
            x = x + self.positions

        return reference_set_attention(x, self.Wq, self.Wk, self.Wv)


def equivariance_gap(
    layer: nn.Module,
    x: Tensor,
    order: Tensor,
) -> float:
    with torch.no_grad():
        return (layer(x[:, order]) - layer(x)[:, order]).abs().max().item()


SETS3_LESSON = prose(
    """
    Some tasks need one answer per element, like a label for every point.
    Then each answer should stay with its element. If you reorder the input,
    the output should come out reordered in the same way. A layer with this
    property is **equivariant**, and a model built from equivariant layers is
    equivariant too.

    Zaheer and colleagues found the most general equivariant Linear layer on
    a set. The output for each element is a Linear of that element plus a
    second Linear of the pooled set. The second term is the same for every
    element.

    ```text
    y_i = Lambda x_i + Gamma pool(x)
    ```

    In the file, Lambda is `own` and Gamma is `pooled`. The paper pooled with
    a sum or a max. This file uses the mean, so the pooled values do not grow
    with the number of elements. Each element sees the others only through
    the pool, and the pool ignores order. So no element's position affects
    the result.

    Attention without positions is also equivariant, and it is more
    flexible. Each element takes its own weighted average of all the
    elements' values. The weights depend only on the elements' content. Lee
    and colleagues built their Set Transformer from this kind of attention.

    To test a layer, compare its output for the reordered input with its
    output for the original input, reordered afterwards. The table gives the
    largest difference between the two on 4 random sets of 8 elements.

    | Layer | Largest change |
    |---|---|
    | The equivariant Linear | 0.0000001 |
    | Attention, no positions | 0.0000005 |
    | Attention, with learned positions added | 3.60 |

    Added positions break equivariance, because each element's output then
    depends on where it sits. A sequence needs positions, which is why
    position2 added them. A set must do without them.

    The pool is what lets each element know about the rest of its set. Here
    is a task that needs it. Each set has 8 numbers from -1 to 1, and the
    answer for each element is its value minus the mean of its set. Each
    model has 64 hidden features and was trained for 2,000 steps. The table
    gives its mean squared error over 3 seeds.

    | Model | Mean squared error |
    |---|---|
    | An MLP on each element alone | 0.037 to 0.038 |
    | `EquivariantNet` | below 0.00002 |
    | set_attention, then an MLP on each element | 0.0002 or less |

    Always guessing 0 gets an error of 0.29. The MLP sees each element alone
    and never the rest of its set, so it cannot compute the mean. Both set
    layers can.
    """,
)

SETS3_SOURCE = source_file(
    '''
    # sets3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.ones(2, 5, 3)
    pooled = x.mean(dim=1, keepdim=True)    # shape (2, 1, 3), so x + pooled adds it to every element
    numbers = torch.ones(2, 5, 1).squeeze(-1)    # shape (2, 5), with the last dimension of size 1 removed


    # TODO: Finish forward. x has shape (batch, elements, inputs). The output
    #   for each element is own applied to that element, plus pooled applied
    #   to the mean of all the elements in its set. The result has shape
    #   (batch, elements, outputs).
    class EquivariantLayer(nn.Module):
        def __init__(
            self,
            inputs: int,
            outputs: int,
        ) -> None:
            super().__init__()
            self.own = nn.Linear(inputs, outputs)
            self.pooled = nn.Linear(inputs, outputs, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            ...


    # TODO: Return attention among the elements of x, which has shape (batch,
    #   elements, features). Use no mask and no positions. The queries are
    #   x @ Wq, the keys x @ Wk and the values x @ Wv. Divide the scores by the
    #   square root of the number of features in each query.
    def set_attention(
        x: Tensor,
        Wq: Tensor,
        Wk: Tensor,
        Wv: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish forward. x has shape (batch, elements, 1). Run first, a
    #   ReLU, second, another ReLU and then head, in that order. Return one
    #   number per element, as a tensor of shape (batch, elements).
    class EquivariantNet(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.first = EquivariantLayer(1, hidden)
            self.second = EquivariantLayer(hidden, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> Tensor:
            ...
    ''',
)

SETS3_SOLUTION = source_file(
    '''
    # sets3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    x = torch.ones(2, 5, 3)
    pooled = x.mean(dim=1, keepdim=True)
    numbers = torch.ones(2, 5, 1).squeeze(-1)


    class EquivariantLayer(nn.Module):
        def __init__(
            self,
            inputs: int,
            outputs: int,
        ) -> None:
            super().__init__()
            self.own = nn.Linear(inputs, outputs)
            self.pooled = nn.Linear(inputs, outputs, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            return self.own(x) + self.pooled(x.mean(dim=1, keepdim=True))


    def set_attention(
        x: Tensor,
        Wq: Tensor,
        Wk: Tensor,
        Wv: Tensor,
    ) -> Tensor:
        q, k, v = x @ Wq, x @ Wk, x @ Wv
        weights = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]), dim=-1)

        return weights @ v


    class EquivariantNet(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.first = EquivariantLayer(1, hidden)
            self.second = EquivariantLayer(hidden, hidden)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Tensor) -> Tensor:
            hidden = torch.relu(self.second(torch.relu(self.first(x))))

            return self.head(hidden).squeeze(-1)
    ''',
)


def check_equivariant_layer(found: type) -> None:
    model = expect_module("EquivariantLayer(3, 4)", seeded_model(found, 3, 4))
    x = small_tensor(731, 2, 5, 3) / 3

    with torch.no_grad():
        expected = seeded_model(ReferenceEquivariantLayer, 3, 4)(x)

    expect_tensor(
        call="EquivariantLayer(3, 4) on 2 sets of 5",
        actual=model(x),
        expected=expected,
    )


def check_equivariant_net(found: type) -> None:
    model = expect_module("EquivariantNet(8)", seeded_model(found, 8))
    x = small_tensor(732, 3, 6, 1) / 3

    with torch.no_grad():
        expected = seeded_model(ReferenceEquivariantNet, 8)(x)

    expect_tensor(
        call="EquivariantNet(8) on 3 sets of 6",
        actual=model(x),
        expected=expected,
    )


def check_sets3(checker: Checker) -> None:
    checker.test_method(
        class_name="EquivariantLayer",
        method="forward",
        description="adds own of each element to pooled of the set's mean",
        run=check_equivariant_layer,
    )
    check_on_inputs(
        checker=checker,
        function="set_attention",
        description="computes attention over the elements with no positions",
        inputs=(
            (small_tensor(733, 2, 5, 4) / 3, small_tensor(734, 4, 4) / 3, small_tensor(735, 4, 4) / 3, small_tensor(736, 4, 3) / 3),
            (small_tensor(737, 1, 3, 2) / 3, small_tensor(738, 2, 6) / 3, small_tensor(739, 2, 6) / 3, small_tensor(740, 2, 2) / 3),
        ),
        reference=reference_set_attention,
        expect=expect_tensor,
    )
    checker.test_method(
        class_name="EquivariantNet",
        method="forward",
        description="stacks two equivariant layers and gives one number per element",
        run=check_equivariant_net,
    )


SETS3 = Exercise(
    name="sets3",
    chapter=SETS,
    title="Layers that reorder with their input",
    lesson=SETS3_LESSON,
    source=SETS3_SOURCE,
    solution=SETS3_SOLUTION,
    hints={
        "EquivariantLayer.forward": "x.mean(dim=1, keepdim=True) keeps the elements dimension with size 1. So "
        "when you add self.pooled of that mean to self.own(x), it is added at every element.",
        "set_attention": "The scores are q @ k.transpose(1, 2), divided by math.sqrt(q.shape[-1]). Take their "
        "softmax over dim=-1, then multiply the weights by v with @.",
        "EquivariantNet.forward": "Each layer keeps the shape (batch, elements, features), so you can chain them "
        "with torch.relu between them. self.head leaves a last dimension of size 1, and .squeeze(-1) removes it.",
    },
    introduces=(),
    check=check_sets3,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons' tables
# ---------------------------------------------------------------------------


def measure_order_gaps() -> dict[str, float]:
    """sets1's table, untrained models on 4 random sets of 8 elements with 3 features, reordered once."""
    x = torch.randn(4, ELEMENTS, 3, generator=torch.Generator().manual_seed(0))
    order = torch.randperm(ELEMENTS, generator=torch.Generator().manual_seed(1))

    return {
        "Linear on the flattened list": reference_order_gap(seeded_model(FlatLinear, ELEMENTS, 3, 2), x, order),
        "Deep Set": reference_order_gap(seeded_model(ReferenceDeepSet, 3, 16, 2), x, order),
    }


def measure_sets2(seeds: range = range(5)) -> dict[int, dict[str, float]]:
    """sets2's table by seed, each model trained by train_regressor for 2,000 steps on 1,000 sets and tested on
    2,000 new ones, sorted and shuffled, with the error of always guessing the average answer."""
    results = {}

    with pinned_threads():
        for seed in seeds:
            x, y = square_sets(1000, torch.Generator().manual_seed(10 * seed))
            test_x, test_y = square_sets(2000, torch.Generator().manual_seed(10 * seed + 1))
            sorted_tests = reference_sort_each(test_x)
            shuffled_tests = reference_shuffle_each(test_x, torch.Generator().manual_seed(10 * seed + 2))
            shuffled_x = reference_shuffle_each(x, torch.Generator().manual_seed(10 * seed + 3))
            runs = {
                "MLP on sorted sets": (ReferenceFlatMLP, (), reference_sort_each(x)),
                "MLP on shuffled sets": (ReferenceFlatMLP, (), shuffled_x),
                "Deep Set on shuffled sets": (ReferenceDeepSet, (1, 32, 1), shuffled_x),
            }
            row = {"guess": ((test_y - y.mean()) ** 2).mean().item()}

            for name, (make, arguments, inputs) in runs.items():
                torch.manual_seed(seed)
                model = train_regressor(make(*arguments), inputs, y, 2000)
                row[f"{name}, sorted tests"] = reference_squared_error(model, sorted_tests, test_y)
                row[f"{name}, shuffled tests"] = reference_squared_error(model, shuffled_tests, test_y)

            results[seed] = row

    return results


def measure_set_layers() -> dict[str, float]:
    """sets3's first table, untrained layers with 4 features on 4 random sets of 8 elements, reordered once."""
    x = torch.randn(4, ELEMENTS, 4, generator=torch.Generator().manual_seed(0))
    order = torch.randperm(ELEMENTS, generator=torch.Generator().manual_seed(1))

    return {
        "equivariant Linear": equivariance_gap(seeded_model(ReferenceEquivariantLayer, 4, 4), x, order),
        "attention, no positions": equivariance_gap(seeded_model(AttentionLayer, 4, ELEMENTS, False), x, order),
        "attention, positions added": equivariance_gap(seeded_model(AttentionLayer, 4, ELEMENTS, True), x, order),
    }


def measure_centre_task(seeds: range = range(3)) -> dict[int, dict[str, float]]:
    """sets3's second table by seed, each model trained with Adam at 0.003 for 2,000 steps of 64 new sets, and
    tested on 2,000 more, with the error of always guessing 0."""
    makers = {
        "MLP on each element": PerElementNet,
        "EquivariantNet": ReferenceEquivariantNet,
        "attention": AttentionNet,
    }
    results = {}

    with pinned_threads():
        for seed in seeds:
            test_x, test_y = centre_sets(2000, torch.Generator().manual_seed(100 + seed))
            row = {"guess 0": (test_y**2).mean().item()}

            for name, make in makers.items():
                torch.manual_seed(seed)
                model = make(64)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
                generator = torch.Generator().manual_seed(seed)

                for _ in range(2000):
                    x, y = centre_sets(64, generator)
                    loss = F.mse_loss(model(x), y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    row[name] = F.mse_loss(model(test_x), test_y).item()

            results[seed] = row

    return results
