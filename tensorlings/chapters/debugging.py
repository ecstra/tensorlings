"""28_debugging, Karpathy's recipe for a training run that fails without a sound, one check at a time."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..course import OPTIMIZER_IMPORTS, QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    Mismatch,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .dropout import labelled_points
from .linear_layers import expect_module

DEBUGGING = Chapter(
    folder="28_debugging",
    title="Debugging training",
    readme=source_file(
        """
        # Debugging training

        A broken training run rarely crashes. It trains a little worse, and
        nothing says why. This chapter follows Karpathy's recipe for training
        neural networks, one check at a time. Check the loss at the start,
        overfit one batch, read each layer's gradient, look at the data, and
        watch for the usual slips.

        ## Further reading

        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        - [CS231n, babysitting the learning process](https://cs231n.github.io/neural-networks-3/#baby)
        - [Google, deep learning tuning playbook](https://github.com/google-research/tuning_playbook)
        """,
    ),
)


class ReferenceClassifier(nn.Module):
    def __init__(
        self,
        inputs: int,
        hidden: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.hidden(x)))


def seeded_classifier(
    seed: int,
    make: type = ReferenceClassifier,
) -> nn.Module:
    torch.manual_seed(seed)

    return make(6, 16, 3)


def reference_overfit(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    steps: int,
    lr: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# debug1
# ---------------------------------------------------------------------------

DEBUG1_LESSON = prose(
    """
    A broken training run rarely crashes. It runs, the loss falls a little,
    and the model comes out worse than it should be. Karpathy's recipe for
    training neural networks answers that with small checks, each run before
    trusting the next step. This chapter goes through them.

    The first check is the loss before any training. For a model that knows
    nothing it is `log(classes)`, as init3 said. A start far above that means
    the model starts confidently wrong, and something in it is off.

    The second is a baseline that ignores the inputs. The best such model
    always predicts each class's share of the labels. Its loss is the
    entropy of those shares, which is minus the sum of `share * log(share)`.
    Entropy measures how evenly the labels spread over the classes. It is 0
    when one class has every label, and `log(classes)` when all have the same
    share. A trained model that doesn't beat it isn't using its inputs at all.

    ```python
    shares = torch.tensor([0.5, 0.25, 0.25])
    -(shares * shares.log()).sum()    # tensor(1.0397), below log(3) = 1.0986
    ```

    A class with no labels has a share of 0, and `0 * log(0)` comes out as
    `nan` in PyTorch, so leave those classes out first.
    """,
)

DEBUG1_SOURCE = source_file(
    '''
    # debug1.py

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    shares = torch.tensor([0.5, 0.25, 0.25])
    entropy = -(shares * shares.log()).sum()         # tensor(1.0397)


    # TODO: Return the loss of the best model that ignores its inputs, the
    #   entropy of the class shares of the labels y, as a plain number. The
    #   classes argument only lets torch.bincount count a class that never
    #   shows up, and a share of 0 is left out anyway.
    def baseline_loss(
        y: Tensor,
        classes: int,
    ) -> float:
        ...


    # TODO: Return whether the model's loss on x and y, before any training,
    #   is within tolerance of log(classes), with classes the width of its output.
    def starts_well(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        tolerance: float,
    ) -> bool:
        ...


    # TODO: This classifier starts far above log(classes). Find the slip in it
    #   and fix it.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(torch.relu(self.hidden(x))) * 20
    ''',
)

DEBUG1_SOLUTION = source_file(
    '''
    # debug1.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    shares = torch.tensor([0.5, 0.25, 0.25])
    entropy = -(shares * shares.log()).sum()


    def baseline_loss(
        y: Tensor,
        classes: int,
    ) -> float:
        counts = torch.bincount(y, minlength=classes).float()
        shares = counts[counts > 0] / len(y)

        return -(shares * shares.log()).sum().item()


    def starts_well(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        tolerance: float,
    ) -> bool:
        with torch.no_grad():
            logits = model(x)

        return F.cross_entropy(logits, y).item() - math.log(logits.shape[1]) <= tolerance


    # The * 20 blew every score up, so the model started sure of a random class.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(torch.relu(self.hidden(x)))
    ''',
)


def reference_baseline(
    y: Tensor,
    classes: int,
) -> float:
    counts = torch.bincount(y, minlength=classes).float()
    shares = counts[counts > 0] / len(y)

    return -(shares * shares.log()).sum().item()


class ScaledClassifier(ReferenceClassifier):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * 20


def check_fixed_classifier(found: type) -> None:
    mine = expect_module("Classifier(6, 16, 3)", seeded_classifier(590, found))
    theirs = seeded_classifier(590)
    points, labels = labelled_points(592, 60, 6, 3)
    x = small_tensor(591, 5, 6)

    with torch.no_grad():
        scores = mine(points)

        if isinstance(scores, Tensor) and scores.shape == (60, 3):
            start = F.cross_entropy(scores, labels).item()

            if start > math.log(3) + 0.3:
                raise Mismatch(f"Classifier(6, 16, 3) starts at a loss of {start:.2f} on 60 points, far above log(3) = 1.10, so it starts confidently wrong.")

        expect_tensor(
            call="Classifier(6, 16, 3) on x of shape (5, 6)",
            actual=mine(x),
            expected=theirs(x),
        )


def check_debug1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="baseline_loss",
        description="gives the entropy of the class shares",
        inputs=(
            (torch.tensor([0, 0, 1, 2]), 3),
            (torch.tensor([1, 1, 1, 3, 3, 0]), 5),
        ),
        reference=reference_baseline,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    x, y = labelled_points(592, 60, 6, 3)
    check_on_inputs(
        checker=checker,
        function="starts_well",
        description="says whether the start is near log(classes)",
        inputs=(
            (Described(seeded_classifier(593), "a fresh classifier with 3 classes"), Described(x, "60 points"), Described(y, "their labels"), 0.3),
            (Described(seeded_classifier(593, ScaledClassifier), "a classifier whose scores are 20 times too large"), Described(x, "60 points"), Described(y, "their labels"), 0.3),
        ),
        reference=lambda model, x, y, tolerance: F.cross_entropy(model(x), y).item() - math.log(3) <= tolerance,
        expect=expect_value,
    )
    checker.test_method(
        class_name="Classifier",
        method="forward",
        description="gives the plain scores, so the start is near log(classes)",
        run=check_fixed_classifier,
    )


DEBUG1 = Exercise(
    name="debug1",
    chapter=DEBUGGING,
    title="The loss at the start",
    lesson=DEBUG1_LESSON,
    source=DEBUG1_SOURCE,
    solution=DEBUG1_SOLUTION,
    hints={
        "baseline_loss": "Count the labels with torch.bincount and minlength=classes, keep the counts above 0, and "
        "divide by len(y) for the shares. Then take minus the sum of shares times their logs.",
        "starts_well": "Take the cross entropy of model(x) against y without autograd, and subtract "
        "math.log of the output's width. Compare that gap with tolerance.",
        "Classifier.forward": "The scores are multiplied by something they shouldn't be.",
    },
    introduces=(),
    check=check_debug1,
)


# ---------------------------------------------------------------------------
# debug2
# ---------------------------------------------------------------------------

DEBUG2_LESSON = prose(
    """
    The next check is the strongest. Take one small batch and train on it
    alone. Any model worth training can learn a handful of examples by heart,
    so the loss should fall close to 0. If it can't, the bug is most likely
    in the model, the loss or the loop, and it is cheap to find.

    One classic slip is a model that ends with softmax while the loss is
    `F.cross_entropy`. The loss takes a softmax of its own, so the model's
    probabilities, all between 0 and 1, get squashed a second time. The scores
    can then never spread far enough apart. With 3 classes the loss can't go
    below about 0.55, however long it trains.

    ```python
    probabilities = torch.tensor([[1.0, 0.0, 0.0]])
    F.cross_entropy(probabilities, torch.tensor([0]))    # tensor(0.5514), the floor
    ```

    Train on one batch with Adam at a rate of 0.01. A sound model gets near 0
    within a few hundred steps.
    """,
)

DEBUG2_SOURCE = source_file(
    '''
    # debug2.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    probabilities = torch.tensor([[1.0, 0.0, 0.0]])
    floor = F.cross_entropy(probabilities, torch.tensor([0]))    # tensor(0.5514)


    # TODO: Return the last loss after steps steps of Adam at lr on the cross
    #   entropy of model on the one batch x and y.
    def overfit_one_batch(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        lr: float,
    ) -> float:
        ...


    # TODO: This classifier can't overfit one batch. Find the slip and fix it.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            return torch.softmax(self.head(torch.relu(self.hidden(x))), dim=1)
    ''',
)

DEBUG2_SOLUTION = source_file(
    '''
    # debug2.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    probabilities = torch.tensor([[1.0, 0.0, 0.0]])
    floor = F.cross_entropy(probabilities, torch.tensor([0]))


    def overfit_one_batch(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        lr: float,
    ) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(steps):
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    # F.cross_entropy takes a softmax itself, so the model hands it the raw scores.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            return self.head(torch.relu(self.hidden(x)))
    ''',
)


def check_overfit(found: object) -> None:
    x, y = labelled_points(600, 16, 6, 3)

    for steps in (20, 200):
        expect_close(
            call=f"overfit_one_batch on a classifier and 16 points, for {steps} steps at 0.01",
            actual=found(seeded_classifier(601), x, y, steps, 0.01),
            expected=reference_overfit(seeded_classifier(601), x, y, steps, 0.01),
            tolerance=1e-4,
        )


def check_overfittable(found: type) -> None:
    x, y = labelled_points(602, 16, 6, 3)
    model = expect_module("Classifier(6, 16, 3)", seeded_classifier(603, found))
    final = reference_overfit(model, x, y, 300, 0.01)

    if final > 0.05:
        raise Mismatch(f"Classifier(6, 16, 3) trained on one batch of 16 for 300 steps ends at a loss of {final:.3f}, but it should get below 0.05.")


def check_debug2(checker: Checker) -> None:
    checker.test(
        function="overfit_one_batch",
        description="trains with Adam on the one batch",
        run=check_overfit,
    )
    checker.test_method(
        class_name="Classifier",
        method="forward",
        description="can learn one batch by heart",
        run=check_overfittable,
    )


DEBUG2 = Exercise(
    name="debug2",
    chapter=DEBUGGING,
    title="Overfitting one batch",
    lesson=DEBUG2_LESSON,
    source=DEBUG2_SOURCE,
    solution=DEBUG2_SOLUTION,
    hints={
        "overfit_one_batch": "Make a torch.optim.Adam over model.parameters() at lr, then run the loop of sgd1 "
        "with F.cross_entropy, and return the last loss as a plain number.",
        "Classifier.forward": "F.cross_entropy wants raw scores, and this forward hands it something else.",
    },
    introduces=(),
    check=check_debug2,
    time_limit=20,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# debug3
# ---------------------------------------------------------------------------

DEBUG3_LESSON = prose(
    """
    After one backward pass, every parameter should have a gradient, and none
    should be much larger or smaller than the rest. A layer whose gradient is
    `None` took no part in the loss. Its forward was never called, or its
    output was thrown away, so it never trains, and nothing warns you.

    `model.named_parameters()` gives each parameter with its name, like
    `hidden.weight`. A gradient's size is its norm, from clipping1.

    ```python
    [name for name, parameter in model.named_parameters() if parameter.grad is None]    # the ones left out
    ```

    Tiny norms at the first layers next to large ones at the last mean the
    gradient vanishes on the way down, as in residual2. Huge ones anywhere
    mean it explodes. Logging every layer's norm during training catches both
    early.
    """,
)

DEBUG3_SOURCE = source_file(
    '''
    # debug3.py

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(2, 2)
    names = [name for name, _ in layer.named_parameters()]    # ['weight', 'bias']


    # TODO: Return a dict from each parameter's name in model to the norm of its
    #   gradient, as a plain number, and 0.0 for a parameter with no gradient.
    def gradient_norms(model: nn.Module) -> dict[str, float]:
        ...


    # TODO: Return the names in norms whose norm is 0.0, in order.
    def silent_layers(norms: dict[str, float]) -> list[str]:
        ...


    # TODO: One of this model's layers never trains. Find out which, with
    #   gradient_norms, and fix forward so every layer is used.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.middle = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            h = torch.relu(self.hidden(x))
            m = torch.relu(self.middle(h))

            return self.head(h)
    ''',
)

DEBUG3_SOLUTION = source_file(
    '''
    # debug3.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    from torch import Tensor

    layer = nn.Linear(2, 2)
    names = [name for name, _ in layer.named_parameters()]


    def gradient_norms(model: nn.Module) -> dict[str, float]:
        norms = {}

        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                norms[name] = 0.0
                continue

            norms[name] = math.sqrt((parameter.grad ** 2).sum().item())

        return norms


    def silent_layers(norms: dict[str, float]) -> list[str]:
        return [name for name, norm in norms.items() if norm == 0.0]


    # The head read h, so middle's output was thrown away and middle never got a gradient.
    class Classifier(nn.Module):
        def __init__(
            self,
            inputs: int,
            hidden: int,
            classes: int,
        ) -> None:
            super().__init__()
            self.hidden = nn.Linear(inputs, hidden)
            self.middle = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, classes)

        def forward(self, x: Tensor) -> Tensor:
            h = torch.relu(self.hidden(x))
            m = torch.relu(self.middle(h))

            return self.head(m)
    ''',
)


class ReferenceThreeLayers(nn.Module):
    def __init__(
        self,
        inputs: int,
        hidden: int,
        classes: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.middle = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.middle(torch.relu(self.hidden(x)))))


class SkippingThreeLayers(ReferenceThreeLayers):
    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.relu(self.hidden(x)))


def reference_gradient_norms(model: nn.Module) -> dict[str, float]:
    return {
        name: 0.0 if parameter.grad is None else math.sqrt((parameter.grad ** 2).sum().item())
        for name, parameter in model.named_parameters()
    }


def backward_once(
    make: type,
    seed: int,
) -> nn.Module:
    model = seeded_classifier(seed, make)
    x, y = labelled_points(seed, 20, 6, 3)
    F.cross_entropy(model(x), y).backward()

    return model


def expect_norms(
    call: str,
    actual: object,
    expected: dict[str, float],
) -> None:
    if not isinstance(actual, dict):
        raise Mismatch(f"{call} should give a dict from names to norms, but it gives {actual!r}.")

    if list(actual) != list(expected):
        raise Mismatch(f"{call} should have the names {list(expected)}, but it has {list(actual)}.")

    for name, norm in expected.items():
        expect_close(
            call=f"the norm of {name} from {call}",
            actual=actual[name],
            expected=norm,
            tolerance=1e-4 * max(norm, 1e-3),
        )


def check_gradient_norms(found: object) -> None:
    for make, name in ((ReferenceThreeLayers, "a three layer classifier"), (SkippingThreeLayers, "a three layer classifier that skips its middle layer")):
        expect_norms(
            call=f"gradient_norms of {name} after one backward pass",
            actual=found(backward_once(make, 610)),
            expected=reference_gradient_norms(backward_once(make, 610)),
        )


def check_uses_every_layer(found: type) -> None:
    mine = expect_module("Classifier(6, 16, 3)", seeded_classifier(611, found))
    theirs = seeded_classifier(611, ReferenceThreeLayers)
    points, labels = labelled_points(613, 20, 6, 3)
    x = small_tensor(612, 5, 6)
    scores = mine(points)

    if isinstance(scores, Tensor) and scores.shape == (20, 3):
        F.cross_entropy(scores, labels).backward()
        silent = [name for name, parameter in mine.named_parameters() if parameter.grad is None]

        if silent:
            listed = silent[0] if len(silent) == 1 else ", ".join(silent[:-1]) + " and " + silent[-1]
            raise Mismatch(f"After one backward pass of Classifier(6, 16, 3), {listed} got no gradient, so that layer never trains.")

    with torch.no_grad():
        expect_tensor(
            call="Classifier(6, 16, 3) on x of shape (5, 6)",
            actual=mine(x),
            expected=theirs(x),
        )


def check_debug3(checker: Checker) -> None:
    checker.test(
        function="gradient_norms",
        description="gives every parameter's gradient norm, 0.0 when there is none",
        run=check_gradient_norms,
    )
    check_on_inputs(
        checker=checker,
        function="silent_layers",
        description="names the parameters with no gradient",
        inputs=(
            ({"hidden.weight": 1.2, "middle.weight": 0.0, "middle.bias": 0.0, "head.weight": 0.4},),
            ({"a": 0.1, "b": 0.2},),
        ),
        reference=lambda norms: [name for name, norm in norms.items() if norm == 0.0],
        expect=expect_value,
    )
    checker.test_method(
        class_name="Classifier",
        method="forward",
        description="passes x through every layer",
        run=check_uses_every_layer,
    )


DEBUG3 = Exercise(
    name="debug3",
    chapter=DEBUGGING,
    title="Each layer's gradient",
    lesson=DEBUG3_LESSON,
    source=DEBUG3_SOURCE,
    solution=DEBUG3_SOLUTION,
    hints={
        "gradient_norms": "Loop over model.named_parameters(). Store 0.0 when .grad is None, and otherwise the square "
        "root of the sum of the gradient's squares.",
        "silent_layers": "Keep the names whose norm equals 0.0, in a list comprehension over norms.items().",
        "Classifier.forward": "Look at what self.head reads. middle's output is worked out and then never used.",
    },
    introduces=(),
    check=check_debug3,
)


# ---------------------------------------------------------------------------
# debug4
# ---------------------------------------------------------------------------

DEBUG4_LESSON = prose(
    """
    Karpathy's first step comes before any model. Look at the data. Read
    examples with their labels, count the labels, and look for repeats. Bugs
    in the data look like a model that learns badly.

    A common one is shuffling the inputs and the labels apart. Each needs the
    same order, or every example ends up with someone else's label. The model
    then can't beat the baseline of debug1 on new data, whatever it does.

    ```python
    perm = torch.randperm(5, generator=generator)
    inputs[perm], labels[perm]    # one order for both, so pairs stay together
    ```

    `torch.unique(x, dim=0)` gives the different rows of x, so the number of
    rows minus the number of different rows counts the repeats. A validation
    example repeated in the training set makes the validation loss look better
    than it is.
    """,
)

DEBUG4_SOURCE = source_file(
    '''
    # debug4.py

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    different = torch.unique(x, dim=0)               # tensor([[1., 2.], [3., 4.]])


    # TODO: Return x and y shuffled with one order from torch.randperm with
    #   generator, so each example keeps its label.
    def shuffle_together(
        x: Tensor,
        y: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        ...


    # TODO: This splits the data into minibatches, but the labels no longer
    #   belong to their inputs. Fix it.
    def make_batches(
        x: Tensor,
        y: Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> list[tuple[Tensor, Tensor]]:
        x_order = torch.randperm(len(x), generator=generator)
        y_order = torch.randperm(len(y), generator=generator)
        x, y = x[x_order], y[y_order]

        return [(x[start:start + batch_size], y[start:start + batch_size]) for start in range(0, len(x), batch_size)]


    # TODO: Return how many rows of x repeat a row that came before.
    def repeated_rows(x: Tensor) -> int:
        ...
    ''',
)

DEBUG4_SOLUTION = source_file(
    '''
    # debug4.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    different = torch.unique(x, dim=0)


    def shuffle_together(
        x: Tensor,
        y: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        order = torch.randperm(len(x), generator=generator)

        return x[order], y[order]


    # Two orders pair each input with another example's label, so one order serves both.
    def make_batches(
        x: Tensor,
        y: Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> list[tuple[Tensor, Tensor]]:
        order = torch.randperm(len(x), generator=generator)
        x, y = x[order], y[order]

        return [(x[start:start + batch_size], y[start:start + batch_size]) for start in range(0, len(x), batch_size)]


    def repeated_rows(x: Tensor) -> int:
        return len(x) - len(torch.unique(x, dim=0))
    ''',
)


def check_make_batches(found: object) -> None:
    x = torch.arange(20).float().reshape(10, 2)
    y = torch.arange(10)
    batches = found(x, y, 4, torch.Generator().manual_seed(620))

    if not isinstance(batches, list) or not batches:
        raise Mismatch("make_batches should give a list of (inputs, labels) pairs.")

    seen = 0

    for inputs, labels in batches:
        seen += len(labels)

        if not torch.equal(inputs[:, 0] / 2, labels.float()):
            raise Mismatch(
                f"make_batches gives a batch whose labels {labels.tolist()} don't belong to its inputs "
                f"{inputs[:, 0].tolist()}. Here each label is its input's first number divided by 2.",
            )

    if seen != 10:
        raise Mismatch(f"make_batches gives {seen} examples in all, but the data has 10.")


def check_debug4(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="shuffle_together",
        description="shuffles both with one order",
        inputs=(
            (small_tensor(621, 5, 2), torch.tensor([0, 1, 2, 3, 4]), seeded(622)),
            (small_tensor(623, 4, 3), torch.tensor([2, 2, 0, 1]), seeded(624)),
        ),
        reference=lambda x, y, generator: (lambda order: (x[order], y[order]))(torch.randperm(len(x), generator=generator)),
        expect=expect_parts("the inputs", "the labels"),
    )
    checker.test(
        function="make_batches",
        description="keeps every label with its input",
        run=check_make_batches,
    )
    check_on_inputs(
        checker=checker,
        function="repeated_rows",
        description="counts the rows that repeat an earlier one",
        inputs=(
            (torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [1.0, 2.0]]),),
            (torch.tensor([[0.0], [1.0], [2.0]]),),
        ),
        reference=lambda x: len(x) - len(torch.unique(x, dim=0)),
        expect=expect_value,
    )


DEBUG4 = Exercise(
    name="debug4",
    chapter=DEBUGGING,
    title="Looking at the data",
    lesson=DEBUG4_LESSON,
    source=DEBUG4_SOURCE,
    solution=DEBUG4_SOLUTION,
    hints={
        "shuffle_together": "Draw one order with torch.randperm(len(x), generator=generator) and index both x and y "
        "with it.",
        "make_batches": "Use one order for both, as shuffle_together does.",
        "repeated_rows": "Subtract the number of different rows, len(torch.unique(x, dim=0)), from len(x).",
    },
    introduces=(),
    check=check_debug4,
)


# ---------------------------------------------------------------------------
# debug5
# ---------------------------------------------------------------------------

DEBUG5_LESSON = prose(
    """
    A few slips come up again and again. Each one still trains, and a short
    run can even look better at first. Over a whole run it comes out worse.

    - no `zero_grad`: each step's gradient is added to every earlier one, so the steps grow and point the wrong way.
    - the wrong mode: dropout and BatchNorm act differently in training and in use, so `model.eval()` and `model.train()` must be set.
    - a summed loss: `reduction="sum"` makes the loss grow with the batch, so with SGD the right rate changes with the batch size.

    `F.cross_entropy` takes the mean over the batch by default. With
    `reduction="sum"` it adds up instead, so a batch of 64 gives a gradient
    64 times larger. With SGD the steps grow 64 times with it. Adam's steps
    stay about its rate, as adamw2 showed.

    Karpathy's recipe has more checks, like fixing the seed so runs repeat,
    and plotting predictions next to their labels. These are the ones that
    hide best. Each function below has one of them.
    """,
)

DEBUG5_SOURCE = source_file(
    '''
    # debug5.py

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: This trains with SGD, but each step's gradient piles onto the ones
    #   before. Fix it.
    def train_steps(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        lr: float,
    ) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(steps):
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        return loss.item()


    # TODO: This measures a model with dropout, but its answer changes from run
    #   to run. Fix it.
    def accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            return (model(x).argmax(dim=1) == y).float().mean().item()


    # TODO: This loss grows with the size of the batch. Fix it.
    def batch_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        return F.cross_entropy(model(x), y, reduction="sum")
    ''',
)

DEBUG5_SOLUTION = source_file(
    '''
    # debug5.py, a solution for comparison

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def train_steps(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        steps: int,
        lr: float,
    ) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(steps):
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def accuracy(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        model.eval()

        with torch.no_grad():
            return (model(x).argmax(dim=1) == y).float().mean().item()


    def batch_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        return F.cross_entropy(model(x), y)
    ''',
)


def dropout_classifier(seed: int) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 3))


def last_sgd_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    clears_gradients: bool,
) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(10):
        loss = F.cross_entropy(model(x), y)

        if clears_gradients:
            optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    return loss.item()


def check_train_steps(found: object) -> None:
    x, y = labelled_points(630, 40, 6, 3)
    call = "train_steps for 10 steps at 0.1 on a classifier and 40 points"
    actual = found(seeded_classifier(631), x, y, 10, 0.1)
    expected = last_sgd_loss(
        model=seeded_classifier(631),
        x=x,
        y=y,
        clears_gradients=True,
    )
    piled_up = last_sgd_loss(
        model=seeded_classifier(631),
        x=x,
        y=y,
        clears_gradients=False,
    )

    if isinstance(actual, float) and abs(actual - piled_up) <= 1e-5:
        raise Mismatch(
            f"{call} ends at a loss of {actual:.4f}, but a loop that clears the gradients each step ends at {expected:.4f}. "
            "The gradients were never cleared, so each step's gradient piled onto the ones before.",
        )

    expect_close(
        call=call,
        actual=actual,
        expected=expected,
        tolerance=1e-5,
    )


def check_accuracy(found: object) -> None:
    x, y = labelled_points(632, 200, 6, 3)
    model = dropout_classifier(633)
    actual = found(model, x, y)
    reference = dropout_classifier(633)
    reference.eval()

    with torch.no_grad():
        expected = (reference(x).argmax(dim=1) == y).float().mean().item()

    if model.training:
        raise Mismatch("accuracy left dropout on, so the answer changes from run to run. Call model.eval() first.")

    expect_close(
        call="accuracy of a classifier with dropout on 200 points",
        actual=actual,
        expected=expected,
    )


def check_batch_loss(found: object) -> None:
    model = seeded_classifier(634)

    for size in (4, 64):
        x, y = labelled_points(635 + size, size, 6, 3)

        with torch.no_grad():
            expected = F.cross_entropy(model(x), y)
            actual = found(model, x, y)

        expect_tensor(
            call=f"batch_loss on a batch of {size}",
            actual=actual,
            expected=expected,
        )


def check_debug5(checker: Checker) -> None:
    checker.test(
        function="train_steps",
        description="clears the gradients every step",
        run=check_train_steps,
    )
    checker.test(
        function="accuracy",
        description="measures with dropout off",
        run=check_accuracy,
    )
    checker.test(
        function="batch_loss",
        description="gives the mean loss over the batch",
        run=check_batch_loss,
    )


DEBUG5 = Exercise(
    name="debug5",
    chapter=DEBUGGING,
    title="The usual slips",
    lesson=DEBUG5_LESSON,
    source=DEBUG5_SOURCE,
    solution=DEBUG5_SOLUTION,
    hints={
        "train_steps": "Add optimizer.zero_grad() before loss.backward().",
        "accuracy": "Call model.eval() before measuring.",
        "batch_loss": "Leave out reduction=\"sum\", so F.cross_entropy takes the mean.",
    },
    introduces=(),
    check=check_debug5,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# quiz6
# ---------------------------------------------------------------------------

QUIZ6_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 4. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `adamw_for` | AdamW, with weight decay on the parameters of two or more dimensions only | adamw1, adamw2 |
    | `clipped_step` | clear, backward, clip the global norm, step | sgd1, clipping1 |
    | `set_rate` | warmup then cosine decay to a tenth of the base, set on every group | warmup1, schedule1 |
    | `start_and_baseline` | the loss before training and the loss of guessing the class shares | debug1 |
    """,
)

QUIZ6_SOURCE = source_file(
    '''
    # quiz6.py
    #
    # No examples and no new tools. Each function combines moves from Section 4.

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return AdamW for model at lr, with weight_decay on the parameters of
    #   two or more dimensions, in the first group, and no decay on the rest,
    #   in the second.
    def adamw_for(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        ...


    # TODO: Clear the gradients, run backward on loss, clip the global norm at
    #   max_norm, and step optimizer. Return the norm from before clipping, as a
    #   plain number.
    def clipped_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: Tensor,
        max_norm: float,
    ) -> float:
        ...


    # TODO: Set every group of optimizer to the rate at step and return it. The
    #   rate warms up over warmup steps, as warmup1's warmup_rate does. Then it
    #   falls along a cosine from base at step warmup to a tenth of base at
    #   step total, as warmup1's warmup_cosine does.
    def set_rate(
        optimizer: torch.optim.Optimizer,
        step: int,
        base: float,
        warmup: int,
        total: int,
    ) -> float:
        ...


    # TODO: Return the model's loss on x and y before any training, and the loss
    #   of always guessing each class's share of y, both as plain numbers.
    def start_and_baseline(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> tuple[float, float]:
        ...
    ''',
)

QUIZ6_SOLUTION = source_file(
    '''
    # quiz6.py, a solution for comparison

    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor


    def adamw_for(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
        kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]
        groups = [{"params": decayed, "weight_decay": weight_decay}, {"params": kept, "weight_decay": 0.0}]

        return torch.optim.AdamW(groups, lr=lr)


    def clipped_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: Tensor,
        max_norm: float,
    ) -> float:
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        return norm.item()


    def set_rate(
        optimizer: torch.optim.Optimizer,
        step: int,
        base: float,
        warmup: int,
        total: int,
    ) -> float:
        if step < warmup:
            rate = base * (step + 1) / warmup
        else:
            progress = (step - warmup) / (total - warmup)
            rate = base / 10 + (base - base / 10) * (1 + math.cos(math.pi * progress)) / 2

        for group in optimizer.param_groups:
            group["lr"] = rate

        return rate


    def start_and_baseline(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> tuple[float, float]:
        with torch.no_grad():
            logits = model(x)

        counts = torch.bincount(y, minlength=logits.shape[1]).float()
        shares = counts[counts > 0] / len(y)

        return F.cross_entropy(logits, y).item(), -(shares * shares.log()).sum().item()
    ''',
)


def check_adamw_for(found: object) -> None:
    model = seeded_classifier(650, ReferenceThreeLayers)
    call = "adamw_for a three layer classifier at 0.001 with weight decay 0.1"
    optimizer = found(model, 0.001, 0.1)

    if type(optimizer) is not torch.optim.AdamW:
        raise Mismatch(f"{call} is {type(optimizer).__name__}, but it should be torch.optim.AdamW.")

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    wanted = (
        ([name for name, parameter in model.named_parameters() if parameter.dim() >= 2], 0.1),
        ([name for name, parameter in model.named_parameters() if parameter.dim() < 2], 0.0),
    )

    if len(optimizer.param_groups) != 2:
        raise Mismatch(f"{call} should have two parameter groups, but it has {len(optimizer.param_groups)}.")

    for position, (group, (expected, decay)) in enumerate(zip(optimizer.param_groups, wanted)):
        got = [names.get(id(parameter), "something else") for parameter in group["params"]]

        if got != expected:
            raise Mismatch(f"Group {position} of {call} should hold {', '.join(expected)}, but it holds {', '.join(got)}.")

        if group["weight_decay"] != decay or group["lr"] != 0.001:
            raise Mismatch(f"Group {position} of {call} should have a weight decay of {decay} and a rate of 0.001.")


def check_clipped_step(found: object) -> None:
    x, y = labelled_points(651, 30, 6, 3)
    mine, theirs = seeded_classifier(652), seeded_classifier(652)
    mine_optimizer = torch.optim.SGD(mine.parameters(), lr=0.1)
    their_optimizer = torch.optim.SGD(theirs.parameters(), lr=0.1)

    for step in range(3):
        actual = found(mine, mine_optimizer, 50 * F.cross_entropy(mine(x), y), 1.0)
        their_optimizer.zero_grad()
        (50 * F.cross_entropy(theirs(x), y)).backward()
        expected = torch.nn.utils.clip_grad_norm_(theirs.parameters(), 1.0).item()
        their_optimizer.step()
        expect_close(
            call=f"the norm clipped_step returns on step {step + 1}, with the loss 50 times too large",
            actual=actual,
            expected=expected,
            tolerance=1e-4 * expected,
        )

    for (name, parameter), reference in zip(mine.named_parameters(), theirs.parameters()):
        expect_tensor(
            call=f"the {name} after 3 clipped steps",
            actual=parameter.detach(),
            expected=reference.detach(),
        )


def reference_set_rate(
    step: int,
    base: float,
    warmup: int,
    total: int,
) -> float:
    if step < warmup:
        return base * (step + 1) / warmup

    progress = (step - warmup) / (total - warmup)

    return base / 10 + (base - base / 10) * (1 + math.cos(math.pi * progress)) / 2


def check_set_rate(found: object) -> None:
    for step in (0, 9, 10, 55, 100):
        optimizer = torch.optim.SGD([{"params": [torch.zeros(1, requires_grad=True)]}, {"params": [torch.zeros(1, requires_grad=True)]}], lr=1.0)
        call = f"set_rate at step {step}, with base 0.01, 10 warmup steps and 100 in all"
        expected = reference_set_rate(step, 0.01, 10, 100)
        returned = found(optimizer, step, 0.01, 10, 100)
        expect_close(
            call=f"the rate {call} returns",
            actual=returned,
            expected=expected,
            tolerance=1e-12,
        )

        for position, group in enumerate(optimizer.param_groups):
            expect_close(
                call=f"the rate of group {position} after {call}",
                actual=group["lr"],
                expected=expected,
                tolerance=1e-12,
            )


def check_quiz6(checker: Checker) -> None:
    checker.test(
        function="adamw_for",
        description="decays the weight matrices only",
        run=check_adamw_for,
    )
    checker.test(
        function="clipped_step",
        description="clears, backs up, clips and steps",
        run=check_clipped_step,
    )
    checker.test(
        function="set_rate",
        description="warms up, then decays to a tenth along a cosine",
        run=check_set_rate,
    )
    x, y = labelled_points(653, 40, 6, 3)
    check_on_inputs(
        checker=checker,
        function="start_and_baseline",
        description="gives the starting loss and the baseline",
        inputs=(
            (Described(seeded_classifier(654), "a fresh classifier"), Described(x, "40 points"), Described(y, "their labels")),
        ),
        reference=lambda model, x, y: (F.cross_entropy(model(x), y).item(), reference_baseline(y, 3)),
        expect=expect_parts("the starting loss", "the baseline"),
    )


QUIZ6 = Exercise(
    name="quiz6",
    chapter=QUIZZES,
    title="Training, start to finish",
    lesson=QUIZ6_LESSON,
    source=QUIZ6_SOURCE,
    solution=QUIZ6_SOLUTION,
    hints={
        "adamw_for": "Make the two groups as adamw2's decay_groups does, then pass them to torch.optim.AdamW with "
        "lr=lr.",
        "clipped_step": "optimizer.zero_grad(), loss.backward(), torch.nn.utils.clip_grad_norm_ on "
        "model.parameters(), then optimizer.step(). Return the clip's result with .item().",
        "set_rate": "Work out the rate as warmup1's warmup_cosine does, with base / 10 as the minimum, then set "
        "group[\"lr\"] for every group.",
        "start_and_baseline": "The first is F.cross_entropy of model(x) and y without autograd. The second is "
        "debug1's baseline_loss, with the number of classes from the output's width.",
    },
    introduces=(),
    check=check_quiz6,
    imports_first=OPTIMIZER_IMPORTS,
)
