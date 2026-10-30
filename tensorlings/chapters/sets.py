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
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .linear_layers import expect_module

SETS = Chapter(
    folder="72_sets",
    title="Sets",
    readme=source_file(
        """
        # Sets

        Section 10 is about building a problem's symmetries into the model.
        The first is the simplest. A set has no order, so a model of a set
        should give the same answer however its elements are listed. This
        chapter builds Deep Sets, shows a flat MLP failing on shuffled sets,
        and builds layers that reorder their output along with their input,
        attention among them.

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
    or the players of a team. They arrive as a list, but the order means
    nothing. A model of a set should give the same answer for every order of
    the same elements.

    A Linear on the flattened list can't. Each position has weights of its
    own, so moving an element moves it onto other weights.

    Zaheer and colleagues proved that, for elements from a countable set, every
    function that ignores order can be written in one shape, called Deep
    Sets.

    - phi runs on every element alone. A Linear on a tensor of shape `(batch, elements, features)` does that already, since it works on the last dimension.
    - A sum over the elements pools them into one vector. Adding doesn't care about order.
    - rho runs on the pooled vector.

    A mean or a max pools without order too. The mean loses how many elements
    there were, and the max keeps only the largest.

    ```python
    torch.tensor([[2.0], [2.0]]).mean(dim=0), torch.tensor([[2.0]]).mean(dim=0)    # tensor([2.]) both times
    torch.tensor([[2.0], [2.0]]).sum(dim=0), torch.tensor([[2.0]]).sum(dim=0)      # tensor([4.]) and tensor([2.])
    ```

    `x[:, order]` lists every set's elements in a new order. Here is how far
    each untrained model's answer moves when 4 random sets of 8 elements, 3
    features each, are reordered.

    | Model | Largest change in the answer |
    |---|---|
    | A Linear on the flattened list | 1.53 |
    | A Deep Set | 0.00000006 |

    The Deep Set moves only by rounding, since a sum in another order rounds
    a little differently.
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


    # TODO: Return each set's elements pooled into one row, for features of
    #   shape (batch, elements, features). how is "sum", "mean" or "max". The
    #   result has shape (batch, features).
    def pool(
        features: Tensor,
        how: str,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Run phi on every element of x, of shape (batch,
    #   elements, inputs), sum over the elements, and run rho on the sum. The
    #   result has shape (batch, outputs).
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


    # TODO: Return the largest change, as a plain number, between model's
    #   answers for x and for x with every set's elements listed in order.
    #   order is a tensor of positions, like tensor([2, 0, 1]). Record nothing
    #   for autograd.
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
        description="measures how far a reordering moves the answer",
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
        "pool": "One if for \"max\" with .amax(dim=1), one for \"mean\" with .mean(dim=1), and .sum(dim=1) for the rest.",
        "DeepSet.forward": "self.phi(x) keeps the elements apart. Sum it over dim=1, then pass the result to self.rho.",
        "order_gap": "Inside torch.no_grad(), take model(x) minus model(x[:, order]), then .abs().max().item().",
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
    Here is a task on sets. Each set holds 8 numbers between -1 and 1, and
    the answer is the sum of their squares, which no order changes. Three
    models trained for 2,000 steps on the same 1,000 sets, tested on 2,000 new
    ones, give this mean squared error.

    | Model | Trained on | Error, sorted tests | Error, shuffled tests |
    |---|---|---|---|
    | An MLP on the flat list | sorted sets | 0.0014 | 1.59 |
    | An MLP on the flat list | shuffled sets | 0.0086 | 0.025 |
    | A Deep Set | shuffled sets | 0.0002 | 0.0002 |

    Always guessing the average answer scores 0.70.

    The MLP trained on sorted sets learned that position 0 holds the smallest
    number. Shuffled, it does worse than a guess. Sorting every input first
    would rescue it here, but that needs an order to sort by. Points with
    many features, or the atoms of a molecule, have none that fits.

    The MLP trained on shuffled sets has to learn from examples that order
    doesn't matter, and 8 numbers have 40,320 orders. The Deep Set has it
    built in, so the same data teaches it the task alone. On shuffled tests
    its error is over 100 times smaller.

    `.argsort()` gives the positions that would sort a tensor. Sorting random
    numbers that way gives a random order. Two index tensors pick one element
    per pair of positions, as adder1's sums did.

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
    def sort_each(x: Tensor) -> Tensor:
        ...


    # TODO: Return x with each set's elements in its own random order. Draw
    #   torch.rand(batch, elements, generator=generator) once and argsort each
    #   row of it along the elements.
    def shuffle_each(
        x: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        ...


    # TODO: Finish forward. Flatten each set of x, of shape (batch, elements, 1),
    #   into one row, run net, and return one answer per set, shape (batch,).
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


    # TODO: Return the mean squared error of model's answers on x against y, as a
    #   plain number. Flatten the answers to one per set, since a model may
    #   give them as a column. Record nothing for autograd.
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
            f"Trained on sorted sets, FlatMLP should do well on sorted tests and badly on shuffled ones. Its errors "
            f"are {on_sorted:.3f} sorted and {on_shuffled:.3f} shuffled, and they should be below 0.05 and above 0.5.",
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
        "sort_each": "x.sort(dim=1) sorts along the elements and gives values and indices. Return the values.",
        "shuffle_each": "Argsort the random numbers along dim=1 to get an order per set. Index x with "
        "torch.arange(len(x)).unsqueeze(1) and that order.",
        "FlatMLP.forward": "x.flatten(1) gives one row of 8 per set. Run self.net, then .squeeze(1).",
        "squared_error": "Inside torch.no_grad(), F.mse_loss of model(x).flatten() and y, then .item().",
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


def reference_is_equivariant(
    layer: object,
    x: Tensor,
    order: Tensor,
    tolerance: float,
) -> bool:
    return torch.allclose(layer(x[:, order]), layer(x)[:, order], atol=tolerance)


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
    Some tasks want one answer per element, like a label for every point.
    Then the answers should follow the elements. Reorder the input, and the
    output comes out reordered the same way. A layer that does that is
    **equivariant**, and a model built from such layers is too.

    Zaheer and colleagues found the general equivariant Linear on a set. Each
    element gets its own transform plus one shared transform of the whole
    set's pool.

    ```text
    y_i = Lambda x_i + Gamma pool(x)
    ```

    They used a sum or a max for the pool. The mean here keeps the size the
    same for big and small sets. Every element sees the others only through
    the pool, so no element's position matters.

    Attention with no positions does the same job with more say. Each element
    picks its own weighted mix of the others, by content alone. Lee and
    colleagues built their Set Transformer from it.

    Take a layer's output for reordered input, and its output reordered after.
    Here is the largest gap between the two, on 4 random sets of 8 elements.

    | Layer | Largest change |
    |---|---|
    | The equivariant Linear | 0.0000001 |
    | Attention, no positions | 0.0000002 |
    | Attention, with learned positions added | 3.61 |

    Positions break it, as position1 showed. A sequence needs them, and a set
    must do without them.

    `torch.allclose(a, b, atol=1e-5)` says whether two tensors match within
    1e-5, as a plain True or False.
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
    pooled = x.mean(dim=1, keepdim=True)    # shape (2, 1, 3), adds to x's every element


    # TODO: Finish forward. Each element of x, of shape (batch, elements,
    #   inputs), gets own of itself plus pooled of the mean of its set's
    #   elements. The result has shape (batch, elements, outputs).
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


    # TODO: Return attention among the elements of x, of shape (batch,
    #   elements, features), with no mask and no positions. The queries are
    #   x @ Wq, the keys x @ Wk and the values x @ Wv. Divide the scores by the
    #   square root of the queries' size.
    def set_attention(
        x: Tensor,
        Wq: Tensor,
        Wk: Tensor,
        Wv: Tensor,
    ) -> Tensor:
        ...


    # TODO: Return whether layer is equivariant on x, as a plain True or False:
    #   whether layer of x reordered by order matches layer of x reordered by
    #   order afterwards, within tolerance. order reorders the elements.
    def is_equivariant(
        layer: nn.Module,
        x: Tensor,
        order: Tensor,
        tolerance: float,
    ) -> bool:
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


    def is_equivariant(
        layer: nn.Module,
        x: Tensor,
        order: Tensor,
        tolerance: float,
    ) -> bool:
        return torch.allclose(layer(x[:, order]), layer(x)[:, order], atol=tolerance)
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


def check_is_equivariant(found: object) -> None:
    x = small_tensor(732, 3, 6, 4) / 3
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    cases = (
        ("an EquivariantLayer(4, 4)", seeded_model(ReferenceEquivariantLayer, 4, 4), True),
        ("attention with no positions", seeded_model(AttentionLayer, 4, 6, False), True),
        ("attention with positions added", seeded_model(AttentionLayer, 4, 6, True), False),
    )

    for name, layer, wanted in cases:
        with torch.no_grad():
            actual = found(layer, x, order, 1e-5)

        expect_value(
            call=f"is_equivariant({name}, 3 sets of 6, an order, 1e-5)",
            actual=actual,
            expected=wanted,
        )


def check_sets3(checker: Checker) -> None:
    checker.test_method(
        class_name="EquivariantLayer",
        method="forward",
        description="adds each element's own transform to the set's pooled one",
        run=check_equivariant_layer,
    )
    check_on_inputs(
        checker=checker,
        function="set_attention",
        description="mixes the elements by content alone",
        inputs=(
            (small_tensor(733, 2, 5, 4) / 3, small_tensor(734, 4, 4) / 3, small_tensor(735, 4, 4) / 3, small_tensor(736, 4, 3) / 3),
            (small_tensor(737, 1, 3, 2) / 3, small_tensor(738, 2, 6) / 3, small_tensor(739, 2, 6) / 3, small_tensor(740, 2, 2) / 3),
        ),
        reference=reference_set_attention,
        expect=expect_tensor,
    )
    checker.test(
        function="is_equivariant",
        description="tells an equivariant layer from one that isn't",
        run=check_is_equivariant,
    )


SETS3 = Exercise(
    name="sets3",
    chapter=SETS,
    title="Layers that reorder with their input",
    lesson=SETS3_LESSON,
    source=SETS3_SOURCE,
    solution=SETS3_SOLUTION,
    hints={
        "EquivariantLayer.forward": "x.mean(dim=1, keepdim=True) keeps a dimension of 1 for the elements, so "
        "self.pooled of it adds to self.own(x) at every element.",
        "set_attention": "q @ k.transpose(1, 2), divided by math.sqrt(q.shape[-1]), softmax over dim=-1, then @ v.",
        "is_equivariant": "Compare layer(x[:, order]) with layer(x)[:, order] using torch.allclose and atol=tolerance.",
    },
    introduces=(),
    check=check_sets3,
)
