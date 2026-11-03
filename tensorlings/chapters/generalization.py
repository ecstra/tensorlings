"""13_generalization, splitting the data, overfitting, early stopping and weight decay, and quiz3, which closes Section 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..course import QUIZZES, Chapter, Exercise
from ..engine import (
    Checker,
    Described,
    check_on_inputs,
    expect_close,
    expect_parts,
    expect_tensor,
    expect_value,
    seeded,
    small_tensor,
)
from ..text import prose, source_file
from .gradient_descent import linear_data, named
from .tensors import pinned_threads

GENERALIZATION = Chapter(
    folder="13_generalization",
    title="Generalization",
    readme=source_file(
        """
        # Generalization

        A model is only useful on examples it hasn't seen. This chapter splits
        the data so that can be measured, shows a model fitting the noise of
        its training data, and the two common cures, stopping early and weight
        decay, chosen on a validation set.

        ## Further reading

        - [Dive into Deep Learning, generalization](https://d2l.ai/chapter_linear-regression/generalization.html)
        - [Dive into Deep Learning, weight decay](https://d2l.ai/chapter_linear-regression/weight-decay.html)
        - [Karpathy, a recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
        """,
    ),
)


def curve(
    seed: int,
    count: int,
) -> tuple[Tensor, Tensor]:
    """Noisy points of sin(3x) for x between -1 and 1, in doubles, so high powers of x stay accurate."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(count, generator=generator, dtype=torch.float64) * 2 - 1
    y = torch.sin(3 * x) + 0.2 * torch.randn(count, generator=generator, dtype=torch.float64)

    return x, y


def powers(
    x: Tensor,
    degree: int,
) -> Tensor:
    return torch.stack([x ** power for power in range(1, degree + 1)], dim=1)


def least_squares(
    features: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor]:
    with_ones = torch.cat([features, torch.ones(len(features), 1, dtype=features.dtype)], dim=1)
    solution = torch.linalg.lstsq(with_ones, y).solution

    return solution[:-1], solution[-1]


def mse(
    predictions: Tensor,
    targets: Tensor,
) -> float:
    return ((predictions - targets) ** 2).mean().item()


def named_curve(
    seed: int,
    count: int,
    name: str,
) -> tuple[Described, Described]:
    x, y = curve(seed, count)

    return Described(x, f"{name}_x with {count} points"), Described(y, f"{name}_y")


# ---------------------------------------------------------------------------
# splits1
# ---------------------------------------------------------------------------

SPLITS1_LESSON = prose(
    """
    A model can fit its training examples and still fail on new ones. So the
    data gets split three ways before anything is trained.

    - The **training set** is what the parameters are fitted on.
    - The **validation** set judges choices, like the learning rate or the size of the model, by how each does on examples it never trained on.
    - The **test set** is measured once, at the very end, to report how well the final model does.

    Making choices with the test set turns it into a second validation set,
    and the number reported at the end comes out too good.

    Shuffle before you cut. Data often arrives sorted, by date, by class or
    by where it came from. A cut without a shuffle puts different kinds of
    examples in different sets.

    ```python
    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    train, validation, test = order[:8], order[8:9], order[9:]
    ```

    A common split is 80, 10 and 10 percent. `int(0.8 * n)` rounds 80 percent
    of `n` down to a whole number, so the cuts fall at `int(0.8 * n)` and
    `int(0.9 * n)`.

    `torch.bincount` counts how many times each whole number appears, and
    `minlength` makes sure numbers that never appear still get a count of 0.
    On the labels of a set, it shows whether the set got some of every class.

    ```python
    torch.bincount(torch.tensor([0, 2, 2]), minlength=4)    # tensor([1, 0, 2, 0])
    ```
    """,
)

SPLITS1_SOURCE = source_file(
    '''
    # splits1.py

    import torch
    from torch import Tensor

    # Shuffle the positions, then cut them into three sets.
    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    train, validation, test = order[:8], order[8:9], order[9:]


    # TODO: Finish the last line, so this returns the training, validation and
    #   test positions for n examples, the shuffled order cut at the two cuts.
    def split_positions(
        n: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        order = torch.randperm(n, generator=generator)
        first_cut = int(0.8 * n)
        second_cut = int(0.9 * n)

        return ...


    # TODO: This cuts x and y without a shuffle, so with sorted data each set
    #   gets a different kind of example. Fix it so it cuts at the positions
    #   split_positions gives.
    def split_rows(
        x: Tensor,
        y: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        first_cut = int(0.8 * len(x))
        second_cut = int(0.9 * len(x))

        return (
            x[:first_cut],
            y[:first_cut],
            x[first_cut:second_cut],
            y[first_cut:second_cut],
            x[second_cut:],
            y[second_cut:],
        )


    # TODO: Return how many times each class from 0 to classes - 1 appears in
    #   labels.
    def class_counts(
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        ...
    ''',
)

SPLITS1_SOLUTION = source_file(
    '''
    # splits1.py, a solution for comparison

    import torch
    from torch import Tensor

    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    train, validation, test = order[:8], order[8:9], order[9:]


    def split_positions(
        n: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        order = torch.randperm(n, generator=generator)
        first_cut = int(0.8 * n)
        second_cut = int(0.9 * n)

        return order[:first_cut], order[first_cut:second_cut], order[second_cut:]


    def split_rows(
        x: Tensor,
        y: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        train, validation, test = split_positions(len(x), generator)

        return x[train], y[train], x[validation], y[validation], x[test], y[test]


    def class_counts(
        labels: Tensor,
        classes: int,
    ) -> Tensor:
        return torch.bincount(labels, minlength=classes)
    ''',
)


def reference_split(
    n: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    order = torch.randperm(n, generator=generator)

    return order[:int(0.8 * n)], order[int(0.8 * n):int(0.9 * n)], order[int(0.9 * n):]


def reference_split_rows(
    x: Tensor,
    y: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, ...]:
    train, validation, test = reference_split(len(x), generator)

    return x[train], y[train], x[validation], y[validation], x[test], y[test]


SORTED_LABELS = torch.arange(20) // 5


def check_splits1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="split_positions",
        description="shuffles, then cuts at 80 and 90 percent",
        inputs=(
            (10, seeded(0)),
            (25, seeded(3)),
        ),
        reference=reference_split,
        expect=expect_parts("the training positions", "the validation positions", "the test positions"),
    )
    # Sorted labels, where a cut without a shuffle leaves the last class alone in the test set.
    check_on_inputs(
        checker=checker,
        function="split_rows",
        description="cuts x and y at shuffled positions",
        inputs=(
            (Described(torch.arange(20.0).unsqueeze(1), "x with 20 rows"), Described(SORTED_LABELS, "sorted labels"), seeded(1)),
        ),
        reference=reference_split_rows,
        expect=expect_parts("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test"),
    )
    check_on_inputs(
        checker=checker,
        function="class_counts",
        description="counts every class, including ones that never appear",
        inputs=(
            (torch.tensor([0, 2, 2, 1, 2]), 3),
            (torch.tensor([1, 1, 3]), 5),
        ),
        reference=lambda labels, classes: torch.bincount(labels, minlength=classes),
        expect=expect_tensor,
    )


SPLITS1 = Exercise(
    name="splits1",
    chapter=GENERALIZATION,
    title="Train, validation and test",
    lesson=SPLITS1_LESSON,
    source=SPLITS1_SOURCE,
    solution=SPLITS1_SOLUTION,
    hints={
        "split_positions": "Slice order at the two cuts, as the line that sets train, validation and test does.",
        "split_rows": "Get the three sets of positions from split_positions(len(x), generator), then pick the rows "
        "of x and y at each.",
        "class_counts": "torch.bincount with minlength=classes, so a class that never appears still gets a 0.",
    },
    introduces=("validation",),
    check=check_splits1,
)


# ---------------------------------------------------------------------------
# overfitting1
# ---------------------------------------------------------------------------

OVERFITTING1_LESSON = prose(
    """
    A model with more parameters can bend to fit more. Give a linear model the
    powers of one input as its features, `x`, `x ** 2` and on up to
    `x ** degree`. Degree 1 fits a line, and degree 9 a curve that can wiggle
    up and down many times.

    `torch.stack(tensors, dim=1)` sets flat tensors side by side as the
    columns of a matrix.

    ```python
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.stack([x, x ** 2], dim=1)    # tensor([[1., 1.], [2., 4.], [3., 9.]])
    ```

    Here 10 noisy points of a curve were fitted by least squares, linreg3's
    tool. The fit was then measured on those 10 points and on 50 new ones from
    the same curve.

    | Degree | Training error | Validation error |
    |---|---|---|
    | 1 | 0.181 | 0.173 |
    | 3 | 0.011 | 0.057 |
    | 9 | 0.000 | 1757 |

    Training error only falls as the degree grows, since more freedom always
    fits the training points better. Validation error falls, then shoots up.
    Degree 9 passes through all 10 training points, noise and all, and swings
    wildly between them. Fitting the noise of the training set instead of the
    pattern under it is **overfitting**. A model too simple to fit even the
    training set is **underfitting**, like degree 1 here. The best degree is
    the one with the lowest validation error.
    """,
)

OVERFITTING1_SOURCE = source_file(
    '''
    # overfitting1.py

    import torch
    from torch import Tensor

    x = torch.tensor([1.0, 2.0, 3.0])

    # torch.stack sets flat tensors side by side as columns.
    first_two_powers = torch.stack([x, x ** 2], dim=1)        # tensor([[1., 1.], [2., 4.], [3., 9.]])


    # The least squares fit from linreg3, giving the weights and the bias.
    def least_squares(
        features: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with_ones = torch.cat([features, torch.ones(len(features), 1, dtype=features.dtype)], dim=1)
        solution = torch.linalg.lstsq(with_ones, y).solution

        return solution[:-1], solution[-1]


    # TODO: Return the powers of x from x ** 1 up to x ** degree, one column per
    #   power.
    def polynomial_features(
        x: Tensor,
        degree: int,
    ) -> Tensor:
        ...


    # TODO: Finish the three lines, so this fits the training points with
    #   features up to the given degree, and returns the mean squared error on
    #   the training points and on the validation points.
    def fit_errors(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        degree: int,
    ) -> tuple[float, float]:
        w, b = least_squares(polynomial_features(train_x, degree), train_y)
        train_predictions = ...
        validation_predictions = ...

        return ...


    # TODO: Finish the last line, so this returns the degree from degrees with the
    #   lowest validation error. errors holds each degree's validation error.
    def best_degree(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        degrees: list[int],
    ) -> int:
        errors = [fit_errors(train_x, train_y, validation_x, validation_y, degree)[1] for degree in degrees]

        return ...
    ''',
)

OVERFITTING1_SOLUTION = source_file(
    '''
    # overfitting1.py, a solution for comparison

    import torch
    from torch import Tensor

    x = torch.tensor([1.0, 2.0, 3.0])

    first_two_powers = torch.stack([x, x ** 2], dim=1)


    def least_squares(
        features: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with_ones = torch.cat([features, torch.ones(len(features), 1, dtype=features.dtype)], dim=1)
        solution = torch.linalg.lstsq(with_ones, y).solution

        return solution[:-1], solution[-1]


    def polynomial_features(
        x: Tensor,
        degree: int,
    ) -> Tensor:
        return torch.stack([x ** power for power in range(1, degree + 1)], dim=1)


    def fit_errors(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        degree: int,
    ) -> tuple[float, float]:
        w, b = least_squares(polynomial_features(train_x, degree), train_y)
        train_predictions = polynomial_features(train_x, degree) @ w + b
        validation_predictions = polynomial_features(validation_x, degree) @ w + b

        return (
            ((train_predictions - train_y) ** 2).mean().item(),
            ((validation_predictions - validation_y) ** 2).mean().item(),
        )


    def best_degree(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        degrees: list[int],
    ) -> int:
        errors = [fit_errors(train_x, train_y, validation_x, validation_y, degree)[1] for degree in degrees]

        return degrees[errors.index(min(errors))]
    ''',
)


def reference_fit_errors(
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    degree: int,
) -> tuple[float, float]:
    w, b = least_squares(powers(train_x, degree), train_y)

    return mse(powers(train_x, degree) @ w + b, train_y), mse(powers(validation_x, degree) @ w + b, validation_y)


def check_overfitting1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="polynomial_features",
        description="gives one column per power of x",
        inputs=(
            (torch.tensor([1.0, 2.0, 3.0]), 3),
            (torch.tensor([-1.0, 0.5]), 4),
        ),
        reference=powers,
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fit_errors",
        description="gives the training and validation errors of the fit",
        inputs=(
            (*named_curve(0, 10, "train"), *named_curve(1, 50, "validation"), 3),
            (*named_curve(2, 12, "train"), *named_curve(3, 40, "validation"), 6),
        ),
        reference=reference_fit_errors,
        expect=expect_parts("the training error", "the validation error"),
    )
    check_on_inputs(
        checker=checker,
        function="best_degree",
        description="gives the degree with the lowest validation error",
        inputs=(
            (*named_curve(0, 10, "train"), *named_curve(1, 50, "validation"), [1, 2, 3, 5, 7, 9]),
            (*named_curve(4, 15, "train"), *named_curve(5, 40, "validation"), [9, 1, 4, 2]),
        ),
        reference=lambda tx, ty, vx, vy, degrees: min(degrees, key=lambda degree: reference_fit_errors(tx, ty, vx, vy, degree)[1]),
        expect=expect_value,
    )


OVERFITTING1 = Exercise(
    name="overfitting1",
    chapter=GENERALIZATION,
    title="Overfitting",
    lesson=OVERFITTING1_LESSON,
    source=OVERFITTING1_SOURCE,
    solution=OVERFITTING1_SOLUTION,
    hints={
        "polynomial_features": "Make a list of x ** power for every power from 1 to degree with range, and stack it "
        "with dim=1.",
        "fit_errors": "Predict with the features of each set, @ w + b. The errors are the mean squared errors "
        "against train_y and validation_y, as plain numbers from .item().",
        "best_degree": "errors.index(min(errors)) gives the position of the lowest error. The degree at that "
        "position in degrees is the answer.",
    },
    introduces=("overfitting",),
    check=check_overfitting1,
)


# ---------------------------------------------------------------------------
# overfitting2
# ---------------------------------------------------------------------------

OVERFITTING2_LESSON = prose(
    """
    Trained by gradient descent, a model that can overfit gets there slowly.
    Here degree 9 powers of the same 10 points were trained for 3000 steps,
    with the validation error measured after every step. It was lowest, 0.069,
    at step 929, and had risen to 0.081 by the end, while the training error
    kept falling.

    So keep the parameters from the step with the lowest validation error.
    This is **early stopping**, and it costs nothing but a copy of the
    parameters.

    Keeping parameters means keeping a copy. `w.detach()` gives the same
    numbers without the autograd record, and `.clone()` copies them, so the
    in place updates of later steps can't change the copy.

    ```python
    best_w = w.detach().clone()
    ```

    Validation error is noisy and can bounce for a while before it truly turns
    up. So training usually waits a set number of steps, the **patience**,
    after the best one. If none of them beats it, training stops.

    `with torch.no_grad():` around the validation loss keeps it out of the
    autograd record, since it never takes part in a gradient.
    """,
)

OVERFITTING2_SOURCE = source_file(
    '''
    # overfitting2.py

    import torch
    from torch import Tensor


    # TODO: Finish the three lines in the if block, so this returns the w and b
    #   from the step with the lowest validation loss, and that step's number.
    def train_with_early_stopping(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor, int]:
        w = torch.zeros(train_x.shape[1], dtype=train_x.dtype, requires_grad=True)
        b = torch.zeros((), dtype=train_x.dtype, requires_grad=True)
        best_loss = float("inf")
        best_w, best_b, best_step = None, None, None

        for step in range(steps):
            loss = ((train_x @ w + b - train_y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
                validation_loss = ((validation_x @ w + b - validation_y) ** 2).mean().item()

            w.grad = None
            b.grad = None

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_w = ...
                best_b = ...
                best_step = ...

        return best_w, best_b, best_step


    history = [5.0, 4.0, 3.0, 3.5, 3.6]

    # The position of the lowest loss, and how many steps came after it.
    lowest_at = history.index(min(history))             # 2
    steps_since = len(history) - 1 - lowest_at          # 2


    # TODO: Return True when training should stop, when the lowest loss in history
    #   came more than patience steps before the last one.
    def should_stop(
        history: list[float],
        patience: int,
    ) -> bool:
        ...
    ''',
)

OVERFITTING2_SOLUTION = source_file(
    '''
    # overfitting2.py, a solution for comparison

    import torch
    from torch import Tensor


    def train_with_early_stopping(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor, int]:
        w = torch.zeros(train_x.shape[1], dtype=train_x.dtype, requires_grad=True)
        b = torch.zeros((), dtype=train_x.dtype, requires_grad=True)
        best_loss = float("inf")
        best_w, best_b, best_step = None, None, None

        for step in range(steps):
            loss = ((train_x @ w + b - train_y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
                validation_loss = ((validation_x @ w + b - validation_y) ** 2).mean().item()

            w.grad = None
            b.grad = None

            # Without clone, best_w would be w itself and follow every later update.
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_w = w.detach().clone()
                best_b = b.detach().clone()
                best_step = step

        return best_w, best_b, best_step


    history = [5.0, 4.0, 3.0, 3.5, 3.6]

    lowest_at = history.index(min(history))
    steps_since = len(history) - 1 - lowest_at


    def should_stop(
        history: list[float],
        patience: int,
    ) -> bool:
        best_step = history.index(min(history))

        return len(history) - 1 - best_step > patience
    ''',
)


def reference_early_stopping(
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, Tensor, float]:
    w = torch.zeros(train_x.shape[1], dtype=train_x.dtype, requires_grad=True)
    b = torch.zeros((), dtype=train_x.dtype, requires_grad=True)
    best = (float("inf"), None, None, None)

    for step in range(steps):
        ((train_x @ w + b - train_y) ** 2).mean().backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            validation_loss = ((validation_x @ w + b - validation_y) ** 2).mean().item()

        w.grad = None
        b.grad = None

        if validation_loss < best[0]:
            best = (validation_loss, w.detach().clone(), b.detach().clone(), float(step))

    return best[1], best[2], best[3]


def named_powers(
    seed: int,
    count: int,
    name: str,
) -> tuple[Described, Described]:
    x, y = curve(seed, count)

    return Described(powers(x, 9), f"{name}_x, powers up to 9 of {count} points"), Described(y, f"{name}_y")


def check_overfitting2(checker: Checker) -> None:
    # 1500 steps reach the best step and go well past it, so a copy that follows w ends up at the last step.
    check_on_inputs(
        checker=checker,
        function="train_with_early_stopping",
        description="keeps the parameters from the step with the lowest validation loss",
        inputs=(
            (*named_powers(0, 10, "train"), *named_powers(1, 50, "validation"), 1500, 0.5),
        ),
        reference=reference_early_stopping,
        expect=expect_parts("best_w", "best_b", "best_step"),
    )
    check_on_inputs(
        checker=checker,
        function="should_stop",
        description="says whether the best loss is more than patience steps old",
        inputs=(
            ([5.0, 4.0, 3.0, 3.5, 3.6], 2),
            ([5.0, 4.0, 3.0, 3.5, 3.6, 3.7], 2),
            ([5.0, 4.0, 3.0, 2.0], 1),
            ([1.0, 2.0, 3.0], 0),
        ),
        reference=lambda history, patience: len(history) - 1 - history.index(min(history)) > patience,
        expect=expect_value,
    )


OVERFITTING2 = Exercise(
    name="overfitting2",
    chapter=GENERALIZATION,
    title="Early stopping",
    lesson=OVERFITTING2_LESSON,
    source=OVERFITTING2_SOURCE,
    solution=OVERFITTING2_SOLUTION,
    hints={
        "train_with_early_stopping": "Keep copies of w and b with .detach().clone(), as the lesson shows, and keep "
        "step as the step number.",
        "should_stop": "Work out the steps since the lowest loss, as the line that sets steps_since does, and "
        "compare it with patience.",
    },
    introduces=(),
    check=check_overfitting2,
    time_limit=30,
)


# ---------------------------------------------------------------------------
# decay1
# ---------------------------------------------------------------------------

DECAY1_LESSON = prose(
    """
    An overfitted model tends to have huge weights, big positive and negative
    ones that cancel out on the training points and swing between them. The
    degree 9 fit of overfitting1 has weights as big as 1149 and -766.

    **Weight decay** adds a cost for large weights to the loss, the sum of the
    squared weights times a strength.

    ```text
    loss = mean squared error + strength * (w ** 2).sum()
    ```

    The model can still use large weights, but only where the data pays for
    them. The bias is usually left out, since it only shifts every prediction
    and can't make the fit swing.

    The extra cost has gradient `2 * strength * w`, which gets added to the
    gradient of the squared error. Gathered up, one step does this to `w`.

    ```text
    w = (1 - 2 * learning_rate * strength) * w - learning_rate * grad_w
    ```

    So every step shrinks each weight by a small fraction, then takes the
    usual step. The weights decay toward 0 unless the data pushes back, hence
    the name. Adding the squared weights to the loss is also called L2
    regularization. With plain gradient descent it does exactly what decay
    does, and Section 4 meets a way of training where the two part.
    """,
)

DECAY1_SOURCE = source_file(
    '''
    # decay1.py

    import torch
    from torch import Tensor

    w = torch.tensor([3.0, -1.0, 2.0])

    # The sum of the squared weights, the size weight decay puts a cost on.
    squared_size = (w ** 2).sum()                     # tensor(14.)

    # One step shrinks each weight by a fraction, here with learning rate 0.1,
    # strength 0.5 and a gradient of zero, so only the shrinking shows.
    shrunk = (1 - 2 * 0.1 * 0.5) * w - 0.1 * torch.zeros(3)    # tensor([2.7, -0.9, 1.8])


    # TODO: Return the mean squared error of x @ w + b against y, plus strength
    #   times the sum of the squared weights. The bias gets no cost.
    def loss_with_decay(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        strength: float,
    ) -> Tensor:
        ...


    # TODO: Return w after one step of gradient descent with weight decay, given
    #   the gradient of the mean squared error alone, grad_w.
    def decayed_step(
        w: Tensor,
        grad_w: Tensor,
        learning_rate: float,
        strength: float,
    ) -> Tensor:
        ...


    # TODO: Finish the line that sets loss, so this trains with weight decay and
    #   returns the final w and b.
    def fit_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
        b = torch.zeros((), dtype=x.dtype, requires_grad=True)

        for _ in range(steps):
            loss = ...
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return w.detach(), b.detach()
    ''',
)

DECAY1_SOLUTION = source_file(
    '''
    # decay1.py, a solution for comparison

    import torch
    from torch import Tensor

    w = torch.tensor([3.0, -1.0, 2.0])

    squared_size = (w ** 2).sum()

    shrunk = (1 - 2 * 0.1 * 0.5) * w - 0.1 * torch.zeros(3)


    def loss_with_decay(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        strength: float,
    ) -> Tensor:
        return ((x @ w + b - y) ** 2).mean() + strength * (w ** 2).sum()


    def decayed_step(
        w: Tensor,
        grad_w: Tensor,
        learning_rate: float,
        strength: float,
    ) -> Tensor:
        return w - learning_rate * (grad_w + 2 * strength * w)


    def fit_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
        steps: int,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
        b = torch.zeros((), dtype=x.dtype, requires_grad=True)

        for _ in range(steps):
            loss = loss_with_decay(x, y, w, b, strength)
            loss.backward()

            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad

            w.grad = None
            b.grad = None

        return w.detach(), b.detach()
    ''',
)


def reference_fit_with_decay(
    x: Tensor,
    y: Tensor,
    strength: float,
    steps: int,
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, requires_grad=True)

    for _ in range(steps):
        (((x @ w + b - y) ** 2).mean() + strength * (w ** 2).sum()).backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad = None
        b.grad = None

    return w.detach(), b.detach()


def check_decay1(checker: Checker) -> None:
    check_on_inputs(
        checker=checker,
        function="loss_with_decay",
        description="adds strength times the squared weights to the squared error",
        inputs=(
            (*named(*linear_data(135, 6, 2)), small_tensor(136, 2), torch.tensor(0.5), 0.1),
            (*named(*linear_data(137, 8, 3)), small_tensor(138, 3), torch.tensor(-1.0), 0.01),
        ),
        reference=lambda x, y, w, b, strength: ((x @ w + b - y) ** 2).mean() + strength * (w ** 2).sum(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-4,
        ),
    )
    check_on_inputs(
        checker=checker,
        function="decayed_step",
        description="takes the decay and the gradient off the weights",
        inputs=(
            (small_tensor(139, 3), small_tensor(140, 3), 0.1, 0.5),
            (small_tensor(141, 4), small_tensor(142, 4), 0.05, 0.01),
        ),
        reference=lambda w, grad_w, learning_rate, strength: w - learning_rate * (grad_w + 2 * strength * w),
        expect=expect_tensor,
    )
    check_on_inputs(
        checker=checker,
        function="fit_with_decay",
        description="trains on the loss with weight decay",
        inputs=(
            (*named(*linear_data(143, 10, 2)), 0.1, 100, 0.05),
            (*named_powers(0, 10, "train"), 0.001, 300, 0.5),
        ),
        reference=reference_fit_with_decay,
        expect=expect_parts("w", "b"),
    )


DECAY1 = Exercise(
    name="decay1",
    chapter=GENERALIZATION,
    title="Weight decay",
    lesson=DECAY1_LESSON,
    source=DECAY1_SOURCE,
    solution=DECAY1_SOLUTION,
    hints={
        "loss_with_decay": "The mean squared error as in descent2, plus strength times (w ** 2).sum().",
        "decayed_step": "Add 2 * strength * w to grad_w, then take the learning rate times that away from w.",
        "fit_with_decay": "Call loss_with_decay with x, y, w, b and strength.",
    },
    introduces=("weight decay",),
    check=check_decay1,
    time_limit=20,
)


# ---------------------------------------------------------------------------
# decay2
# ---------------------------------------------------------------------------

DECAY2_LESSON = prose(
    """
    The decay strength is chosen the way overfitting1 chose the degree. Try a
    few strengths ten times apart, fit with each on the training set, and keep
    the one with the lowest validation error. For the degree 9 fit of
    overfitting1 that goes like this.

    | Strength | Training error | Validation error |
    |---|---|---|
    | 0 | 0.000 | 1757 |
    | 0.0001 | 0.008 | 0.100 |
    | 0.001 | 0.010 | 0.071 |
    | 0.01 | 0.028 | 0.123 |
    | 0.1 | 0.157 | 0.143 |

    Too little decay overfits and too much underfits. Even a tiny strength
    tames the degree 9 fit.

    A polynomial also has its degree to turn down, as overfitting1 did. A net
    has no such dial. Here is one of the kind Section 3 builds, with 301
    parameters, fitted to the same 10 points from 5 random starting weights.

    | Strength | Training error | Validation error |
    |---|---|---|
    | 0 | 0.000 | 0.193 |
    | 0.0001 | 0.000 | 0.189 |
    | 0.001 | 0.000 | 0.128 |
    | 0.01 | 0.011 | 0.060 |
    | 0.1 | 0.202 | 0.149 |

    Strength 0.01 cuts the validation error from 0.193 to 0.060, lower on all
    5 starts. Decay is the dial for a model with no size to pick.

    Numbers chosen by trying them and comparing on the validation set are
    called **hyperparameters**. Gradient descent learns the parameters, and
    you choose the hyperparameters. The learning rate, the minibatch size, the
    degree and the decay strength are all hyperparameters.

    Once the choice is made, measure the chosen model on the test set, once,
    and report that number. It is the only honest estimate of how the model
    does on new data, since nothing was chosen by looking at it.

    Choosing on the test set makes its number too good. On 30 new points, the
    degree 9 fit with the strength picked on validation, 0.001, scores 0.154.
    Picking the strength on those 30 points instead chooses 0.01 and reports
    0.133.
    """,
)

DECAY2_SOURCE = source_file(
    '''
    # decay2.py

    import torch
    from torch import Tensor


    # Finds the weights gradient descent with weight decay settles on, directly,
    # the way least squares does for the plain squared error.
    def fit_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        x_mean = x.mean(dim=0)
        centered = x - x_mean
        penalty = torch.eye(x.shape[1], dtype=x.dtype) * strength * len(x)
        w = torch.linalg.solve(centered.T @ centered + penalty, centered.T @ (y - y.mean()))

        return w, y.mean() - x_mean @ w


    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([2.0, 4.0, 6.5], dtype=torch.float64)

    # Fit with a strength, then measure, here on the same points for a small example.
    w, b = fit_with_decay(x, y, 0.1)
    error = ((x @ w + b - y) ** 2).mean().item()


    # TODO: Finish the last line, so this returns the mean squared error on the
    #   validation set of the model fitted on the training set with this
    #   strength.
    def validation_error(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strength: float,
    ) -> float:
        w, b = fit_with_decay(train_x, train_y, strength)

        return ...


    # TODO: Finish the last line, so this returns the strength from strengths
    #   with the lowest validation error.
    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        errors = [validation_error(train_x, train_y, validation_x, validation_y, strength) for strength in strengths]

        return ...


    # TODO: Finish the two lines, so this picks the strength on the validation set,
    #   then returns the mean squared error on the test set of the model fitted
    #   with that strength.
    def test_error(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> float:
        strength = ...
        w, b = fit_with_decay(train_x, train_y, strength)

        return ...
    ''',
)

DECAY2_SOLUTION = source_file(
    '''
    # decay2.py, a solution for comparison

    import torch
    from torch import Tensor


    def fit_with_decay(
        x: Tensor,
        y: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        x_mean = x.mean(dim=0)
        centered = x - x_mean
        penalty = torch.eye(x.shape[1], dtype=x.dtype) * strength * len(x)
        w = torch.linalg.solve(centered.T @ centered + penalty, centered.T @ (y - y.mean()))

        return w, y.mean() - x_mean @ w


    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([2.0, 4.0, 6.5], dtype=torch.float64)

    w, b = fit_with_decay(x, y, 0.1)
    error = ((x @ w + b - y) ** 2).mean().item()


    def validation_error(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strength: float,
    ) -> float:
        w, b = fit_with_decay(train_x, train_y, strength)

        return ((validation_x @ w + b - validation_y) ** 2).mean().item()


    def pick_strength(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        strengths: list[float],
    ) -> float:
        errors = [validation_error(train_x, train_y, validation_x, validation_y, strength) for strength in strengths]

        return strengths[errors.index(min(errors))]


    def test_error(
        train_x: Tensor,
        train_y: Tensor,
        validation_x: Tensor,
        validation_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        strengths: list[float],
    ) -> float:
        strength = pick_strength(train_x, train_y, validation_x, validation_y, strengths)
        w, b = fit_with_decay(train_x, train_y, strength)

        return ((test_x @ w + b - test_y) ** 2).mean().item()
    ''',
)


def ridge(
    x: Tensor,
    y: Tensor,
    strength: float,
) -> tuple[Tensor, Tensor]:
    x_mean = x.mean(dim=0)
    centered = x - x_mean
    w = torch.linalg.solve(
        centered.T @ centered + torch.eye(x.shape[1], dtype=x.dtype) * strength * len(x),
        centered.T @ (y - y.mean()),
    )

    return w, y.mean() - x_mean @ w


def reference_validation_error(
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: Tensor,
    strength: float,
) -> float:
    w, b = ridge(train_x, train_y, strength)

    return mse(validation_x @ w + b, validation_y)


STRENGTHS = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]


def check_decay2(checker: Checker) -> None:
    sets = (
        (*named_powers(0, 10, "train"), *named_powers(1, 50, "validation")),
        (*named_powers(6, 14, "train"), *named_powers(7, 40, "validation")),
    )
    check_on_inputs(
        checker=checker,
        function="validation_error",
        description="fits on the training set and measures on the validation set",
        inputs=(
            (*sets[0], 0.001),
            (*sets[1], 0.1),
        ),
        reference=reference_validation_error,
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-6 * max(1.0, abs(expected)),
        ),
    )
    check_on_inputs(
        checker=checker,
        function="pick_strength",
        description="gives the strength with the lowest validation error",
        inputs=(
            (*sets[0], STRENGTHS),
            (*sets[1], [1.0, 0.01, 0.0001]),
        ),
        reference=lambda tx, ty, vx, vy, strengths: min(strengths, key=lambda strength: reference_validation_error(tx, ty, vx, vy, strength)),
        expect=expect_close,
    )
    # On the second test set, picking on the test set chooses 0.01 instead of 0.001, so that leak fails.
    check_on_inputs(
        checker=checker,
        function="test_error",
        description="picks on the validation set and measures once on the test set",
        inputs=(
            (*sets[0], *named_powers(8, 30, "test"), STRENGTHS),
            (*sets[0], *named_powers(20, 30, "test"), STRENGTHS),
        ),
        reference=lambda tx, ty, vx, vy, test_x, test_y, strengths: mse(
            test_x @ ridge(tx, ty, min(strengths, key=lambda s: reference_validation_error(tx, ty, vx, vy, s)))[0]
            + ridge(tx, ty, min(strengths, key=lambda s: reference_validation_error(tx, ty, vx, vy, s)))[1],
            test_y,
        ),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-6,
        ),
    )


DECAY2 = Exercise(
    name="decay2",
    chapter=GENERALIZATION,
    title="Choosing the strength",
    lesson=DECAY2_LESSON,
    source=DECAY2_SOURCE,
    solution=DECAY2_SOLUTION,
    hints={
        "validation_error": "Measure the way the line that sets error does, on validation_x and validation_y.",
        "pick_strength": "errors.index(min(errors)) is the position of the lowest error, as in best_degree of "
        "overfitting1.",
        "test_error": "The strength comes from pick_strength. The error is the mean squared error of test_x @ w + b "
        "against test_y.",
    },
    introduces=(),
    check=check_decay2,
)


# ---------------------------------------------------------------------------
# quiz3
# ---------------------------------------------------------------------------

QUIZ3_LESSON = prose(
    """
    This quiz has no examples and no new tools. Each function combines moves
    from Section 2. If a step is unclear, the exercise in the table shows it.

    | Function | Moves | Shown in |
    |---|---|---|
    | `classifier_loss` | class scores, cross entropy from logits | softmax3, likelihood2 |
    | `standardize_both` | the training set's mean and spread, used on both sets | descent4, splits1 |
    | `minibatch_step` | the loss on a minibatch's rows, one step of gradient descent | minibatch1, descent2 |
    | `decayed_classifier_loss` | cross entropy, plus weight decay on the weights | likelihood2, decay1 |
    """,
)

QUIZ3_SOURCE = source_file(
    '''
    # quiz3.py
    #
    # No examples and no new tools. Each function combines moves from Section 2.

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    # TODO: Return the cross entropy of softmax regression with weights W and
    #   bias b on the examples x, against labels.
    def classifier_loss(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
    ) -> Tensor:
        ...


    # TODO: Finish both lines, so this returns train_x and test_x, both
    #   standardized with the mean and the spread of train_x's columns.
    def standardize_both(
        train_x: Tensor,
        test_x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean, spread = ...

        return ...


    # TODO: Finish the line that sets loss and the last line, so this returns w
    #   and b after one step of gradient descent on the mean squared error of
    #   the rows of x and y at positions.
    def minibatch_step(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        positions: Tensor,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = w.clone().requires_grad_()
        b = b.clone().requires_grad_()
        loss = ...
        loss.backward()

        return ...


    # TODO: Return the cross entropy of softmax regression on x against labels,
    #   plus strength times the sum of the squared weights in W.
    def decayed_classifier_loss(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
        strength: float,
    ) -> Tensor:
        ...
    ''',
)

QUIZ3_SOLUTION = source_file(
    '''
    # quiz3.py, a solution for comparison

    import torch
    import torch.nn.functional as F
    from torch import Tensor


    def classifier_loss(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
    ) -> Tensor:
        return F.cross_entropy(x @ W + b, labels)


    # The test set gets the training set's numbers, never its own.
    def standardize_both(
        train_x: Tensor,
        test_x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean, spread = train_x.mean(dim=0), train_x.std(dim=0)

        return (train_x - mean) / spread, (test_x - mean) / spread


    def minibatch_step(
        x: Tensor,
        y: Tensor,
        w: Tensor,
        b: Tensor,
        positions: Tensor,
        learning_rate: float,
    ) -> tuple[Tensor, Tensor]:
        w = w.clone().requires_grad_()
        b = b.clone().requires_grad_()
        loss = ((x[positions] @ w + b - y[positions]) ** 2).mean()
        loss.backward()

        return (w - learning_rate * w.grad).detach(), (b - learning_rate * b.grad).detach()


    def decayed_classifier_loss(
        x: Tensor,
        W: Tensor,
        b: Tensor,
        labels: Tensor,
        strength: float,
    ) -> Tensor:
        return F.cross_entropy(x @ W + b, labels) + strength * (W ** 2).sum()
    ''',
)


def reference_minibatch_step(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    b: Tensor,
    positions: Tensor,
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    w = w.clone().requires_grad_()
    b = b.clone().requires_grad_()
    ((x[positions] @ w + b - y[positions]) ** 2).mean().backward()

    return (w - learning_rate * w.grad).detach(), (b - learning_rate * b.grad).detach()


def check_quiz3(checker: Checker) -> None:
    classifiers = (
        (small_tensor(144, 5, 2), small_tensor(145, 2, 3) / 2, small_tensor(146, 3) / 2, torch.tensor([0, 2, 1, 1, 0])),
        (small_tensor(147, 4, 3), small_tensor(148, 3, 3) / 2, small_tensor(149, 3) / 2, torch.tensor([2, 2, 0, 1])),
    )
    check_on_inputs(
        checker=checker,
        function="classifier_loss",
        description="gives the cross entropy of softmax regression",
        inputs=classifiers,
        reference=lambda x, W, b, labels: F.cross_entropy(x @ W + b, labels),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )
    # The test set is shifted far from the training set, so standardizing it with its own numbers gives other values.
    check_on_inputs(
        checker=checker,
        function="standardize_both",
        description="standardizes both sets with the training set's numbers",
        inputs=(
            (small_tensor(150, 6, 2), small_tensor(151, 3, 2) + 10),
        ),
        reference=lambda train_x, test_x: (
            (train_x - train_x.mean(dim=0)) / train_x.std(dim=0),
            (test_x - train_x.mean(dim=0)) / train_x.std(dim=0),
        ),
        expect=expect_parts("the training set", "the test set"),
    )
    x, y = linear_data(152, 8, 2)
    check_on_inputs(
        checker=checker,
        function="minibatch_step",
        description="takes one step on the loss of the chosen rows",
        inputs=(
            (*named(x, y), small_tensor(153, 2), torch.tensor(0.5), torch.tensor([6, 1, 3]), 0.1),
        ),
        reference=reference_minibatch_step,
        expect=expect_parts("w", "b"),
    )
    check_on_inputs(
        checker=checker,
        function="decayed_classifier_loss",
        description="adds weight decay on W to the cross entropy",
        inputs=tuple((*classifier, 0.1) for classifier in classifiers),
        reference=lambda x, W, b, labels, strength: F.cross_entropy(x @ W + b, labels) + strength * (W ** 2).sum(),
        expect=lambda call, actual, expected: expect_close(
            call=call,
            actual=actual,
            expected=expected,
            tolerance=1e-5,
        ),
    )


QUIZ3 = Exercise(
    name="quiz3",
    chapter=QUIZZES,
    title="Learning, start to finish",
    lesson=QUIZ3_LESSON,
    source=QUIZ3_SOURCE,
    solution=QUIZ3_SOLUTION,
    hints={
        "classifier_loss": "The scores are x @ W + b, as in softmax3. F.cross_entropy takes the scores and the "
        "labels.",
        "standardize_both": "Take the mean and the spread of train_x's columns, as column_stats does in descent4. "
        "Use those same two on both sets.",
        "minibatch_step": "The loss is the mean squared error on x[positions] and y[positions]. Then step each of w "
        "and b against its gradient, and .detach() the results.",
        "decayed_classifier_loss": "The cross entropy from classifier_loss, plus strength times (W ** 2).sum().",
    },
    introduces=(),
    check=check_quiz3,
)


# ---------------------------------------------------------------------------
# The runs behind decay2's lesson
# ---------------------------------------------------------------------------

LESSON_STRENGTHS = [0.0, 0.0001, 0.001, 0.01, 0.1]


def wide_net_errors(
    seed: int,
    strength: float,
    hidden: int = 100,
    steps: int = 20000,
    learning_rate: float = 0.002,
) -> tuple[float, float]:
    """Linear, tanh, Linear with 100 hidden neurons, 301 parameters, fitted to overfitting1's 10 points by gradient
    descent with weight decay on both weight matrices. The first layer starts with spread 10 and the second with
    spread 4 over the square root of the width, so the net starts steep and can fit every point with no degree to
    turn down. Gives the training and validation errors."""
    train_x, train_y = curve(0, 10)
    validation_x, validation_y = curve(1, 50)
    train_x, validation_x = train_x.unsqueeze(1), validation_x.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(hidden, 1, generator=generator, dtype=torch.float64) * 10).requires_grad_()
    b1 = (torch.randn(hidden, generator=generator, dtype=torch.float64) * 10).requires_grad_()
    W2 = (torch.randn(1, hidden, generator=generator, dtype=torch.float64) * 4 / hidden**0.5).requires_grad_()
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def net(x: Tensor) -> Tensor:
        return (torch.tanh(x @ W1.T + b1) @ W2.T + b2).squeeze(1)

    for _ in range(steps):
        loss = ((net(train_x) - train_y) ** 2).mean() + strength * ((W1**2).sum() + (W2**2).sum())
        loss.backward()

        with torch.no_grad():
            for parameter in (W1, b1, W2, b2):
                parameter -= learning_rate * parameter.grad
                parameter.grad = None

    with torch.no_grad():
        return mse(net(train_x), train_y), mse(net(validation_x), validation_y)


def measure_decay2(seeds: range = range(5)) -> dict[str, object]:
    """Every number in decay2's lesson. The wide net takes about six minutes."""
    with pinned_threads():
        train_x, train_y = curve(0, 10)
        validation_x, validation_y = curve(1, 50)
        x, validation_powers = powers(train_x, 9), powers(validation_x, 9)
        results: dict[str, object] = {}

        polynomial = {}

        for strength in LESSON_STRENGTHS:
            w, b = ridge(x, train_y, strength)
            polynomial[strength] = (mse(x @ w + b, train_y), mse(validation_powers @ w + b, validation_y))

        results["degree 9 table, training and validation error"] = polynomial

        net = {}
        net_by_seed = {}

        for strength in LESSON_STRENGTHS:
            runs = [wide_net_errors(seed, strength) for seed in seeds]
            net[strength] = (sum(run[0] for run in runs) / len(runs), sum(run[1] for run in runs) / len(runs))
            net_by_seed[strength] = [run[1] for run in runs]

        results["wide net table, training and validation error over the seeds"] = net
        results["wide net validation error by seed"] = net_by_seed

        # The check's second test set, where picking on the test set picks another strength.
        test_x, test_y = curve(20, 30)
        test_powers = powers(test_x, 9)
        honest = min(STRENGTHS, key=lambda strength: reference_validation_error(x, train_y, validation_powers, validation_y, strength))
        leaky = min(STRENGTHS, key=lambda strength: reference_validation_error(x, train_y, test_powers, test_y, strength))
        results["honest pick and its test error"] = (honest, reference_validation_error(x, train_y, test_powers, test_y, honest))
        results["pick on the test set and its test error"] = (leaky, reference_validation_error(x, train_y, test_powers, test_y, leaky))

    return results
