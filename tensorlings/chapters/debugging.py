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
from .adaptive import pinned_threads
from .dropout import labelled_points
from .linear_layers import expect_module
from .names import context_sets, names_model

DEBUGGING = Chapter(
    folder="28_debugging",
    title="Debugging training",
    readme=source_file(
        """
        # Debugging training

        A broken training run rarely crashes. It trains a little worse, and
        nothing says why. This chapter goes through the checks of Karpathy's
        recipe for training neural networks and Stanford's CS231n notes, one at
        a time. Check the loss at the start, overfit one batch, read each
        layer's gradient, look at the data, and watch for the usual slips.

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
    trusting the next step. This chapter goes through them, with one more from
    Stanford's CS231n notes.

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
    #   entropy of the class shares of the labels y, as a plain number. A
    #   class out of classes that no label belongs to has a share of 0, so
    #   leave it out.
    def baseline_loss(
        y: Tensor,
        classes: int,
    ) -> float:
        ...


    # TODO: Return whether the model's loss on x and y, before any training,
    #   is at most tolerance above log(classes), with classes the width of its
    #   output.
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
    Karpathy's next check takes one small batch and trains on it alone, and
    CS231n's notes do the same. Any model worth training can learn a handful
    of examples by heart,
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


    # TODO: Train model for steps steps of Adam at lr on the cross entropy of
    #   the one batch x and y. Return the loss of the last step, as a plain
    #   number, as sgd1's train does.
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
        call = f"overfit_one_batch on a classifier and 16 points, for {steps} steps at 0.01"
        actual = found(seeded_classifier(601), x, y, steps, 0.01)
        expected = reference_overfit(seeded_classifier(601), x, y, steps, 0.01)
        trained = seeded_classifier(601)
        reference_overfit(trained, x, y, steps, 0.01)

        with torch.no_grad():
            after_training = F.cross_entropy(trained(x), y).item()

        if isinstance(actual, float) and abs(actual - expected) > 1e-4 and abs(actual - after_training) <= 1e-4:
            raise Mismatch(
                f"{call} gives {actual:.4f}, the loss measured again after training. It should give {expected:.4f}, the "
                "loss of the last step, worked out before that step's update, as sgd1's train does.",
            )

        expect_close(
            call=call,
            actual=actual,
            expected=expected,
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
    After one backward pass, every parameter should have a gradient. A layer
    whose gradient is `None` took no part in the loss. Its forward was never
    called, or its output was thrown away, so it never trains, and nothing
    warns you. CS231n's notes watch each layer's gradient too.

    `model.named_parameters()` gives each parameter with its name, like
    `hidden.weight`. A gradient's size is its norm, from clipping1.

    ```python
    [name for name, parameter in model.named_parameters() if parameter.grad is None]    # the ones left out
    ```

    Sound layers can still differ a lot. Take a fresh names model on three
    seeds. Its norms ran from about 0.003 for the norm layer's weight to 1.1
    to 1.3 for the head's, nearly 500 times apart. A vanishing gradient is
    far smaller, like residual2's average gradient of `6e-24` at the first of
    51 plain layers.
    Huge norms anywhere mean it explodes. Logging every layer's norm during
    training catches both early.
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
        actual = mine(x)
        expected = theirs(x)

        if isinstance(actual, Tensor) and actual.shape == expected.shape and not torch.allclose(actual, expected, atol=1e-6):
            raise Mismatch(
                "Classifier(6, 16, 3) uses every layer but gives other scores than the fix. Change only what "
                "self.head reads, so it runs hidden, relu, middle, relu and head in that order.",
            )

        expect_tensor(
            call="Classifier(6, 16, 3) on x of shape (5, 6)",
            actual=actual,
            expected=expected,
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

    order = torch.cat([labels for _, labels in batches])
    expected = torch.randperm(10, generator=torch.Generator().manual_seed(620))

    if torch.equal(order, torch.arange(10)):
        raise Mismatch("make_batches keeps the examples in their first order, so it no longer shuffles. Keep one torch.randperm order for both.")

    if not torch.equal(order, expected):
        raise Mismatch(
            f"make_batches gives the examples in the order {order.tolist()}, but one torch.randperm(len(x), "
            f"generator=generator) gives {expected.tolist()}.",
        )


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
    A few slips come up again and again. Each one still runs, and a short run
    can even look better at first. Over a whole run it comes out worse.

    - no `zero_grad`: each step's gradient is added to every earlier one, so the steps grow and point the wrong way.
    - the wrong mode: dropout and BatchNorm act differently in training and in use. Measuring calls `model.eval()` first and `model.train()` after, as nanoGPT does.
    - a summed loss: `reduction="sum"` makes the loss grow with the batch, so with SGD the right rate changes with the batch size.

    `F.cross_entropy` takes the mean over the batch by default. With
    `reduction="sum"` it adds up instead, so a batch of 64 gives a gradient
    64 times larger. With SGD the steps grow 64 times with it. Adam's steps
    stay about its rate, as adamw2 showed.

    Here the classifier of these checks trained on 2000 points, with
    minibatches of 32 and SGD at 0.1, on three seeds. The table gives the
    loss on all 2000 points.

    | Loop | 10 steps | 200 steps | 1000 steps |
    |---|---|---|---|
    | sound | 0.97 to 1.05 | 0.43 to 0.44 | 0.42 |
    | no `zero_grad` | 0.59 to 0.83 | 1.33 to 1.60 | 1.56 to 115 |
    | summed loss | 0.66 to 0.91 | 0.66 to 0.71 | 0.61 to 0.71 |

    Both slips were ahead after 10 steps on every seed, and far behind by the
    end. Karpathy lists the first two, with debug2's softmax, among the most
    common neural net mistakes. These are the ones that hide best. Each
    function below has one of them.
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
            correct = (model(x).argmax(dim=1) == y).float().mean().item()

        # Back to training mode, as nanoGPT does after measuring, so the next training step has dropout on again.
        model.train()

        return correct


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
    reference = dropout_classifier(633)
    reference.eval()

    with torch.no_grad():
        expected = (reference(x).argmax(dim=1) == y).float().mean().item()

    # The model may be left in either mode after the call, since switching back to train() is right too. Dropout
    # still on shows as readings that change, and the fixed seed keeps that the same every check.
    torch.manual_seed(634)
    readings = [found(model, x, y) for _ in range(3)]

    if all(isinstance(reading, float) for reading in readings) and len(set(readings)) > 1:
        raise Mismatch(
            f"accuracy gives {', '.join(f'{reading:.3f}' for reading in readings)} on the same model and points, so "
            "dropout is still on. Call model.eval() before measuring.",
        )

    expect_close(
        call="accuracy of a classifier with dropout on 200 points",
        actual=readings[0],
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
        "train_steps": "Clear the old gradients with the optimizer's zero_grad before loss.backward().",
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
    | `decay_weights_only_` | decoupled weight decay, in place under no_grad, on the parameters of two or more dimensions only | adamw1, adamw2, modules1 |
    | `clipped_step` | clear, backward, clip the global norm, step | sgd1, clipping2 |
    | `warmup_to_picked_rate` | the range test's losses smoothed, a tenth of the best rate picked, then a warmup to it | schedule2, warmup1 |
    | `start_excess` | the loss before training, less the loss of guessing the class shares | init3, debug1 |
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


    # TODO: Shrink, in place, every parameter of model that has two or more
    #   dimensions, by the fraction lr * weight_decay of itself. So lr 0.1 and
    #   weight_decay 0.5 take 5% off each such number. Leave the parameters of
    #   one dimension, like biases, as they are. Change them without autograd
    #   recording it, and return nothing.
    def decay_weights_only_(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> None:
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


    # TODO: Return the learning rate at step of a warmup toward a rate picked
    #   by a range test, where losses[i] is the loss the test got at rates[i].
    #   Smooth losses with beta, correcting for the average starting at 0.
    #   The base rate is a tenth of the rate where the smoothed loss is lowest.
    #   The rate climbs to base over warmup steps, step 0 giving base / warmup,
    #   and stays at base after that.
    def warmup_to_picked_rate(
        rates: list[float],
        losses: list[float],
        beta: float,
        step: int,
        warmup: int,
    ) -> float:
        ...


    # TODO: Return how far the model's loss on x and y starts above the loss of
    #   always guessing each class's share of y, as a plain number. A class's
    #   share is the fraction of the labels in y that are that class. Work out
    #   the model's cross entropy without autograd, since nothing trains here.
    def start_excess(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
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


    def decay_weights_only_(
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.dim() >= 2:
                    parameter *= 1 - lr * weight_decay


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


    def warmup_to_picked_rate(
        rates: list[float],
        losses: list[float],
        beta: float,
        step: int,
        warmup: int,
    ) -> float:
        average = 0.0
        smoothed = []

        for k, loss in enumerate(losses, start=1):
            average = beta * average + (1 - beta) * loss
            smoothed.append(average / (1 - beta ** k))

        base = rates[smoothed.index(min(smoothed))] / 10

        return base * min(1.0, (step + 1) / warmup)


    def start_excess(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
    ) -> float:
        with torch.no_grad():
            start = F.cross_entropy(model(x), y).item()

        counts = torch.bincount(y).float()
        shares = counts[counts > 0] / len(y)
        baseline = -(shares * shares.log()).sum().item()

        return start - baseline
    ''',
)


def normed_classifier(seed: int) -> nn.Sequential:
    """A LayerNorm's weight has one dimension but is named weight, so a check by name decays it by mistake."""
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.ReLU(), nn.Linear(6, 3))


def check_decay_weights_only(found: object) -> None:
    for lr, weight_decay in ((0.1, 0.5), (0.01, 3.0)):
        model, expected = normed_classifier(650), normed_classifier(650)
        call = (
            "decay_weights_only_ on nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.ReLU(), nn.Linear(6, 3)), "
            f"with lr {lr} and weight_decay {weight_decay}"
        )
        found(model, lr, weight_decay)

        with torch.no_grad():
            for parameter in expected.parameters():
                if parameter.ndim > 1:
                    parameter.mul_(1 - lr * weight_decay)

        for (name, parameter), reference in zip(model.named_parameters(), expected.parameters()):
            expect_tensor(
                call=f"parameter {name} after {call}",
                actual=parameter.detach(),
                expected=reference.detach(),
            )


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


def reference_warmup_to_picked_rate(
    rates: list[float],
    losses: list[float],
    beta: float,
    step: int,
    warmup: int,
) -> float:
    corrected = []
    running = 0.0

    for count, loss in enumerate(losses, start=1):
        running = beta * running + (1 - beta) * loss
        corrected.append(running / (1 - beta**count))

    lowest = min(range(len(corrected)), key=corrected.__getitem__)
    base = rates[lowest] / 10

    if step >= warmup:
        return base

    return base * (step + 1) / warmup


# Each test's lowest loss is a lone dip that smoothing averages away, so the picked rate differs without smoothing.
RANGE_TESTS = (
    (
        [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512, 0.1024, 0.2048, 0.4096],
        [2.30, 2.29, 2.27, 2.20, 2.05, 1.40, 1.85, 1.70, 1.62, 1.58, 2.40, 3.90, 7.50],
        0.6,
        5,
    ),
    (
        [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512],
        [1.10, 1.08, 1.02, 0.95, 0.70, 0.93, 0.80, 0.85, 1.40, 2.60],
        0.8,
        4,
    ),
)


def lopsided_labels(*counts: int) -> Described:
    labels = torch.cat([torch.full((count,), label) for label, count in enumerate(counts)])
    shuffled = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(len(labels)))]
    listed = ", ".join(f"{count} of class {label}" for label, count in enumerate(counts))

    return Described(shuffled, f"{len(labels)} labels, {listed}")


def fresh_classifier(
    seed: int,
    classes: int,
) -> Described:
    torch.manual_seed(seed)

    return Described(ReferenceClassifier(6, 16, classes), f"a fresh classifier with {classes} classes")


def reference_start_excess(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> float:
    with torch.no_grad():
        logits = model(x)

    return F.cross_entropy(logits, y).item() - reference_baseline(y, logits.shape[1])


def check_quiz6(checker: Checker) -> None:
    checker.test(
        function="decay_weights_only_",
        description="shrinks the parameters of two or more dimensions in place",
        run=check_decay_weights_only,
    )
    checker.test(
        function="clipped_step",
        description="clears, backs up, clips and steps",
        run=check_clipped_step,
    )
    check_on_inputs(
        checker=checker,
        function="warmup_to_picked_rate",
        description="picks a tenth of the best smoothed rate and warms up to it",
        inputs=tuple(
            (rates, losses, beta, step, warmup)
            for rates, losses, beta, warmup in RANGE_TESTS
            for step in (0, warmup - 2, warmup + 3)
        ),
        reference=reference_warmup_to_picked_rate,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-6 * expected,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="start_excess",
        description="gives how far the starting loss sits above guessing the class shares",
        inputs=(
            (fresh_classifier(653, 3), Described(small_tensor(654, 20, 6) / 3, "20 points"), lopsided_labels(14, 4, 2)),
            (fresh_classifier(655, 4), Described(small_tensor(656, 12, 6) / 3, "12 points"), lopsided_labels(3, 0, 9)),
        ),
        reference=reference_start_excess,
        expect=expect_close,
    )


QUIZ6 = Exercise(
    name="quiz6",
    chapter=QUIZZES,
    title="Training, start to finish",
    lesson=QUIZ6_LESSON,
    source=QUIZ6_SOURCE,
    solution=QUIZ6_SOLUTION,
    hints={
        "decay_weights_only_": "Inside torch.no_grad, go through model.parameters(). Pick out the ones whose .dim() "
        "is at least 2, as adamw2's groups do, and scale each in place with *=, as adamw1's MyAdamW does.",
        "clipped_step": "Clear the gradients with the optimizer's zero_grad before backward. Clip with "
        "torch.nn.utils.clip_grad_norm_ after backward and before the step, and return what it gives with .item().",
        "warmup_to_picked_rate": "Smooth as schedule2's smooth does, dividing the kth average by 1 - beta ** k. "
        "Find the position of the smallest smoothed loss and take a tenth of that rate as base. Then warm up as "
        "warmup1's warmup_rate does.",
        "start_excess": "The start is F.cross_entropy of model(x) and y inside torch.no_grad, with .item(). The "
        "guessing loss is debug1's baseline_loss. Count each class with torch.bincount, keep the counts above "
        "zero and divide by len(y). Then take minus the sum of each share times its log.",
    },
    introduces=(),
    check=check_quiz6,
    imports_first=OPTIMIZER_IMPORTS,
)


# ---------------------------------------------------------------------------
# The runs behind the lessons
# ---------------------------------------------------------------------------


def measure_gradient_spread(seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, float]]:
    """debug3's lesson: each parameter's gradient norm in a fresh names model after one backward pass on the first 64
    training contexts, a healthy model whose norms still differ a lot. One dict per seed."""
    X, Y = context_sets()["train"]
    results: list[dict[str, float]] = []

    with pinned_threads():
        for seed in seeds:
            model = names_model(seed)
            F.cross_entropy(model(X[:64]), Y[:64]).backward()
            results.append(reference_gradient_norms(model))

    return results


def measure_slips(
    seeds: tuple[int, ...] = (0, 1, 2),
    marks: tuple[int, ...] = (10, 200, 1000),
) -> dict[str, list[dict[int, float]]]:
    """debug5's table: the classifier of these checks trained on 2000 points with minibatches of 32 and SGD at 0.1, by a
    sound loop, a loop that never clears the gradients, and one with a summed loss. Per seed, the mean loss on all 2000
    points at each mark."""
    x, y = labelled_points(700, 2000, 6, 3)
    loops = {
        "sound": (True, "mean"),
        "no zero_grad": (False, "mean"),
        "summed loss": (True, "sum"),
    }
    results: dict[str, list[dict[int, float]]] = {}

    with pinned_threads():
        for label, (clears_gradients, reduction) in loops.items():
            for seed in seeds:
                model = seeded_classifier(seed)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                generator = torch.Generator().manual_seed(seed)
                losses: dict[int, float] = {}

                for step in range(max(marks)):
                    batch = torch.randint(0, len(x), (32,), generator=generator)
                    loss = F.cross_entropy(model(x[batch]), y[batch], reduction=reduction)

                    if clears_gradients:
                        optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    if step + 1 in marks:
                        with torch.no_grad():
                            losses[step + 1] = F.cross_entropy(model(x), y).item()

                results.setdefault(label, []).append(losses)

    return results
